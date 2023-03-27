from typing import Sequence

from tqdm import tqdm

from unidistill.engine.callbacks import Callback
from unidistill.engine.callbacks.clearml_callback import ClearMLCallback
from unidistill.exps.base_exp import BaseExp
from unidistill.utils import torch_dist
import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from ..base_executor import BaseExecutor

__all__ = ["Det3DEvaluator"]


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2**ndim, ndim])
    return corners


def rotation_2d_reverse(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape \
            (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5.

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d_reverse(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    corners = torch.from_numpy(corners)
    return corners


class Det3DEvaluator(BaseExecutor):
    def __init__(
        self,
        exp: BaseExp,
        callbacks: Sequence[Callback],
        logger=None,
        eval_interval: int = -1,
    ) -> None:
        super().__init__(exp, callbacks, logger)
        self.eval_interval = eval_interval

    def eval(self):
        self.epoch += 1
        if self.eval_interval != -1 and self.epoch % self.eval_interval != 0:
            return
        exp = self.exp
        local_rank = torch_dist.get_rank()

        self.val_iter = iter(self.val_dataloader)
        self._invoke_callback("before_eval")
        self.model.cuda()
        self.model.eval()

        preds = []
        for step in tqdm(range(len(self.val_dataloader)), disable=(local_rank > 0)):
            data = next(self.val_iter)
            pred_item = exp.test_step(data)
            if type(pred_item) == dict and "pred_dicts" in pred_item:
                pred_dicts = pred_item["pred_dicts"]
                preds += self.val_dataloader.dataset.generate_prediction_dicts(
                    data, pred_dicts, exp.model.module.class_names
                )
            # mmdetection e.g. FCOS3D
            elif type(pred_item) == list:
                preds += pred_item
            else:
                raise NotImplementedError

            self._invoke_callback("after_step", step, {})

        torch_dist.synchronize()
        preds = sum(map(list, zip(*torch_dist.all_gather_object(preds))), [])[
            : len(self.val_dataloader.dataset)
        ]
        if local_rank == 0:
            self.logger.info("eval ...")
            clearml = [cb for cb in self.callbacks if isinstance(cb, ClearMLCallback)]
            kwargs = {}
            if clearml:
                clearml = clearml[0]
                kwargs = {"clearml_task": clearml.task, "iteration": self.global_step}
            result_str, result_dict = self.val_dataloader.dataset.evaluation(
                preds, exp.model.module.class_names, output_dir=exp.output_dir, **kwargs
            )
            self.logger.info(result_str)
        self._invoke_callback("after_eval", det_annos=preds)
