import glob
import os
import pickle
import shutil
import subprocess
import time
import numpy as np

import refile
import torch

from loguru import logger
from threading import Thread
from refile import smart_path_join

from perceptron.engine.executors import Trainer
from .base_callback import Callback, MasterOnlyCallback

try:
    from cvpack2.checkpoint.c2_model_loading import align_and_update_state_dicts
except ImportError:
    # cvpack2 is an optional dependency at the moment
    pass


__all__ = ["CheckPointSaver", "CheckPointLoader", "CheckPointC2Loader"]


class CheckPointSaver(MasterOnlyCallback):
    def __init__(
        self,
        local_path,
        filename=r"checkpoint_epoch_{epoch}.pth",
        remote_path=None,
        save_interval: int = 5,
        num_keep_latest=None,
    ):
        self.local_path = local_path
        self.filename = filename
        self.remote_path = remote_path
        self.save_interval = save_interval
        self.num_keep_latest = num_keep_latest
        os.makedirs(local_path, exist_ok=True)

    def _copy_to_oss_background(self, source, dest, update_best_ckpt=False):
        """
        sync local checkpoint to oss in background
        """

        def worker(source, dest, update_best_ckpt=False, cur_best_ckpt_pth=""):
            for retry in range(8):
                try:
                    refile.s3.s3_upload(source, dest)
                    # make a text file contains the oss path
                    with open(source + ".tmp", "w") as f:
                        f.write(dest)
                    shutil.move(source + ".tmp", source)
                    logger.info("finish sync {} -> {}".format(source, dest))
                    if update_best_ckpt:
                        refile.s3.s3_copy(dest, cur_best_ckpt_pth)
                        logger.info("finish sync {} -> {}".format(dest, cur_best_ckpt_pth))
                    return
                except Exception as e:
                    logger.error(e)
                    time.sleep(int(2 ** retry))
                    continue
            logger.error("error copy {} -> {}, after {} retries".format(source, dest, retry))

        cur_best_ckpt_pth = ""
        if update_best_ckpt:
            if refile.smart_exists(self.remote_path):
                all_remote_ckpt_path = refile.s3.s3_listdir(self.remote_path)
                for item in all_remote_ckpt_path:
                    if item.startswith("best"):
                        refile.s3.s3_remove(self.remote_path + item)
                        break
            cur_best_ckpt_pth = smart_path_join(self.remote_path, "best_checkpoint.pth")

        thread = Thread(target=worker, args=(source, dest, update_best_ckpt, cur_best_ckpt_pth), daemon=False)
        thread.start()

    def _make_checkpoint(self, trainer: Trainer):
        model_state = None
        if hasattr(trainer, "ema_model"):
            model = trainer.ema_model.ema
        else:
            model = trainer.model
        if model:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_state = model.module.state_dict()
                model_state_cpu = type(model_state)()
                for key, val in model_state.items():
                    model_state_cpu[key] = val.cpu()
                model_state = model_state_cpu
            else:
                model_state = model.state_dict()

        optim_state = trainer.optimizer.state_dict() if trainer.optimizer else None

        callback_states = {}
        for cb in trainer.callbacks:
            if hasattr(cb, "state_dict"):
                cls_name = cb.__class__.__name__
                callback_states[cls_name] = cb.state_dict()

        ckpt = {
            "epoch": trainer.epoch,
            "it": trainer.global_step,
            "global_step": trainer.global_step,
            "model_state": model_state,
            "optimizer_state": optim_state,
            "lr_scheduelr": trainer.lr_scheduler.state_dict(),
            "callback": callback_states,
        }

        # save grad_scaler
        if hasattr(trainer, "grad_scaler"):
            ckpt["grad_scaler_state"] = trainer.grad_scaler.state_dict()

        return ckpt

    def after_epoch(self, trainer: Trainer, epoch: int, update_best_ckpt: bool = False):
        if (epoch + 1) % self.save_interval != 0:
            return
        filename = self.filename.format(epoch=epoch)
        save_path = smart_path_join(self.local_path, filename)
        torch.save(self._make_checkpoint(trainer), save_path)
        if self.remote_path:
            remote_path = smart_path_join(self.remote_path, filename)
            self._copy_to_oss_background(save_path, remote_path, update_best_ckpt)
        self._remove_out_of_date_ckpt()

    def _remove_out_of_date_ckpt(self):
        if not self.num_keep_latest:
            return

        ckpt_list = glob.glob(smart_path_join(self.local_path, self.filename.format(epoch="*")))
        ckpt_list.sort(key=os.path.getmtime)
        if len(ckpt_list) > self.num_keep_latest:
            for cur_file_idx in range(0, len(ckpt_list) - self.num_keep_latest):
                os.remove(ckpt_list[cur_file_idx])


class CheckPointLoader(Callback):
    def __init__(
        self,
        path,
        weight_only=False,
    ):
        self.path = self._resolve_path(path)
        self.weight_only = weight_only

    @staticmethod
    def _resolve_path(path):
        if refile.is_s3(path):
            return path

        path = os.path.realpath(path)
        if refile.smart_exists(path):
            output = subprocess.check_output(["file", "--mime-type", path, "-F", "@"])
            mime_type = output.decode().strip().split("@")[-1].strip()

            # text file contains an oss path
            if mime_type.startswith("text"):
                with open(path) as f:
                    path = f.read().strip()
        else:
            raise ValueError(f"{path} is not exists!!")
        return path

    def load_checkpoint(self, trainer: Trainer):
        logger.info(f"Loading parameters from checkpoint {self.path}")
        with refile.smart_open(self.path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device("cpu"))

        # TODO bulid model finetune callback
        model_state_dict = trainer.model.state_dict()
        checkpoint_state_dict = checkpoint["model_state"]
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    logger.info(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(k, shape_checkpoint, shape_model)
                    )
                    checkpoint_state_dict.pop(k)
        trainer.model.load_state_dict(checkpoint_state_dict, strict=False)

        if self.weight_only:
            return

        trainer.epoch = checkpoint.get("epoch", -1) + 1
        trainer.global_step = checkpoint.get("global_step", -1) + 1
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        trainer.lr_scheduler.load_state_dict(checkpoint.get("lr_scheduler", {}))
        trainer.lr_scheduler.step(trainer.global_step - 1)
        # resume callback
        for cb in trainer.callbacks:
            if hasattr(cb, "state_dict"):
                cls_name = cb.__class__.__name__
                if cls_name in checkpoint["callback"]:
                    cb.load_state_dict(checkpoint["callback"][cls_name])
        # resume grad_scaler
        if hasattr(trainer, "grad_scaler") and "grad_scaler_state" in checkpoint:
            trainer.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class CheckPointC2Loader(CheckPointLoader):
    def load_checkpoint(self, trainer: Trainer):
        if not self.path.endswith(".pkl"):
            return super().load_checkpoint(trainer)

        logger.info(f"Loading parameters from checkpoint {self.path}")
        checkpoint = self._reverse_c2_model(trainer)

        trainer.model.load_state_dict(checkpoint["model_state"])
        if self.weight_only:
            return

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
        """
        for k in list(state_dict.keys()):
            if "weight_order" in k:
                continue
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

    # in order to convert caffe model to C2
    # just use in pretrained model
    def _reverse_c2_model(self, trainer: Trainer):
        with refile.smart_open(self.path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self._convert_ndarray_to_tensor(data)
        model_state_dict = trainer.model.state_dict()
        align_and_update_state_dicts(
            model_state_dict,
            data,
            c2_conversion=True,
        )
        checkpoint = {}
        checkpoint["model_state"] = model_state_dict
        return checkpoint
