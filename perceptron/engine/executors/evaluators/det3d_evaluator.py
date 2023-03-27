from typing import Sequence

from tqdm import tqdm

from perceptron.engine.callbacks import Callback
from perceptron.engine.callbacks.clearml_callback import ClearMLCallback
from perceptron.exps.base_exp import BaseExp
from perceptron.utils import torch_dist
import matplotlib
matplotlib.use('AGG')
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
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1)
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
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
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
    def __init__(self, exp: BaseExp, callbacks: Sequence[Callback], logger=None, eval_interval: int = -1) -> None:
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
            """if step==0:
                _,ax = plt.subplots(1,1,figsize=(9,9))
                ax.patch.set_facecolor('k')
                gt_boxes_tmp = data['gt_boxes'][0]
                gt_boxes_tmp_bev = center_to_corner_box2d(gt_boxes_tmp[:,:2].cpu().detach().numpy(),gt_boxes_tmp[:,3:5].cpu().detach().numpy(),gt_boxes_tmp[:,6].cpu().detach().numpy(), origin=(0.5,0.5))
                for box in gt_boxes_tmp_bev:
                    for i in range(3):
                        ax.plot([box[i][0],box[i+1][0]],[box[i][1],box[i+1][1]],color='g',linewidth=2,alpha=0.9)
                    ax.plot([box[3][0],box[0][0]],[box[3][1],box[0][1]],color='g',linewidth=2,alpha=0.9)
                plt.xlim(-50,50)
                plt.ylim(-50,50)
                x = copy.deepcopy(data['points'][0,:,0]).cpu().detach().numpy()
                y = copy.deepcopy(data['points'][0,:,1]).cpu().detach().numpy()
                ax.scatter(x,y,s=0.75,marker='o',c='w')
                ax.set_aspect(1)
                plt.rcParams["savefig.dpi"] = 200
                plt.rcParams["figure.dpi"] = 200
                plt.savefig('/data/megvii_code/pcd.png')
                plt.close()"""
            pred_item = exp.test_step(data)
            if type(pred_item) == dict and "pred_dicts" in pred_item:
                pred_dicts = pred_item["pred_dicts"]
                preds += self.val_dataloader.dataset.generate_prediction_dicts(
                    data, pred_dicts, exp.model.module.class_names
                )
                """if step==7:
                    gt_boxes_tmp_bev = center_to_corner_box2d(pred_dicts[0]['pred_boxes'][:,:2].cpu().detach().numpy(),pred_dicts[0]['pred_boxes'][:,3:5].cpu().detach().numpy(),pred_dicts[0]['pred_boxes'][:,6].cpu().detach().numpy(), origin=(0.5,0.5))
                    j=0
                    for box in gt_boxes_tmp_bev:
                       j+=1
                      #if j in [1,2,3,6, 12,38,69,95,98,107,108,109,123,127,128, 145, 150,154, 159, 183,195,201,204]:
                       if j in [1,4,9,13, 12,13,15,22,23,27,36,47,56,60,61,62,65,67,69,71,73,77,86,90,95,96,98,104,105,108,109,115,122,123,124,125,126,127,128, 130,131,138,139,146,155, 156,157,154, 159, 183,195,201,204]:
                        for i in range(3):
                            ax.plot([box[i][0],box[i+1][0]],[box[i][1],box[i+1][1]],color='r',linewidth=2,alpha=0.9,label=step)
                        ax.plot([box[3][0],box[0][0]],[box[3][1],box[0][1]],color='r',linewidth=2,alpha=0.9,label=step)
                        #ax.text(box[0][0],box[0][1],'{}'.format(j),fontsize=3, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right')
                        #if step>=15:
                        #   break"""
            # mmdetection e.g. FCOS3D
            elif type(pred_item) == list:
                preds += pred_item
            else:
                raise NotImplementedError

            self._invoke_callback("after_step", step, {})

        torch_dist.synchronize()
        preds = sum(map(list, zip(*torch_dist.all_gather_object(preds))), [])[: len(self.val_dataloader.dataset)]
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
