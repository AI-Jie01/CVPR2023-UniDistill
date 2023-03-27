import os

import tqdm
from tensorboardX import SummaryWriter

from .base_callback import MasterOnlyCallback
from perceptron.engine.executors import Trainer
from perceptron.utils.log_utils import AvgMeter

__all__ = ["ProgressBar", "LearningRateMonitor", "TextMonitor", "TensorBoardMonitor"]


class ProgressBar(MasterOnlyCallback):
    def __init__(self, logger=None) -> None:
        self.epoch_bar = None
        self.step_bar = None
        self.logger = logger

    def setup(self, trainer: Trainer):
        self.epoch_bar = tqdm.tqdm(initial=0, total=trainer.exp.max_epoch, desc="[Epoch]", dynamic_ncols=True)
        self.step_bar = tqdm.tqdm(initial=0, desc="[Step]", dynamic_ncols=True, leave=False)
        if self.logger:
            self.logger.remove(0)
            self.logger.add(lambda msg: self.step_bar.write(msg, end=""))

    def before_epoch(self, trainer: Trainer, epoch: int):
        self.epoch_bar.update(epoch - self.epoch_bar.n)
        if hasattr(trainer.exp, "total_step"):
            # Multi task
            self.step_bar.reset(trainer.exp.total_step)
        else:
            self.step_bar.reset(len(trainer.train_dataloader))

    def after_step(self, trainer: Trainer, step, data_dict, *args, **kwargs):
        self.step_bar.update()

    def after_train(self, trainer: Trainer):
        if self.step_bar:
            self.step_bar.close()
        if self.epoch_bar:
            self.epoch_bar.close()


class LearningRateMonitor:
    def _get_learning_rate(self, optimizer):
        if hasattr(optimizer, "lr"):
            lr = float(optimizer.lr)
        else:
            lr = optimizer.param_groups[0]["lr"]
        return lr


class TextMonitor(MasterOnlyCallback, LearningRateMonitor):
    def __init__(self, interval=10):
        self.interval = interval
        self.avg_loss = AvgMeter()
        self.ext_dict = {}

    def after_step(self, trainer: Trainer, step, data_dict, *args, **kwargs):
        self.avg_loss.update(kwargs["loss"])
        if kwargs["extra"] is not None:
            if hasattr(trainer.exp, "multi_task") and trainer.exp.multi_task:
                if len(self.ext_dict) == 0:
                    for dataset_name, ext_dict in kwargs["extra"].items():
                        if ext_dict is not None:
                            for key, val in ext_dict.items():
                                sub_k = f"{key}_{dataset_name}"
                                self.ext_dict[sub_k] = AvgMeter()
                for dataset_name, ext_dict in kwargs["extra"].items():
                    if ext_dict is not None:
                        for key, val in ext_dict.items():
                            sub_k = f"{key}_{dataset_name}"
                            self.ext_dict[sub_k].update(val)
            else:
                if len(self.ext_dict) == 0:
                    self.ext_dict = {k: AvgMeter() for k in kwargs["extra"]}
                for key, val in kwargs["extra"].items():
                    self.ext_dict[key].update(val)

        if step % self.interval != 0:
            return
        lr = self._get_learning_rate(trainer.optimizer)
        ext_info = "".join([f" {k}={v.window_avg :.4f}" for k, v in self.ext_dict.items()])

        trainer.logger.info(
            f"e:{trainer.epoch}[{step}/{self.total_step}] lr={lr :.4e} loss={self.avg_loss.window_avg :.4f}{ext_info}"
        )

    def before_epoch(self, trainer: Trainer, epoch: int):
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.logger.info(f"e:{epoch} learning rate = {lr :.4e}")
        if hasattr(trainer.exp, "total_step"):
            self.total_step = trainer.exp.total_step
        else:
            self.total_step = len(trainer.train_dataloader)


class TensorBoardMonitor(MasterOnlyCallback, LearningRateMonitor):
    def __init__(self, log_dir, interval=10):
        os.makedirs(log_dir, exist_ok=True)
        self.tb_log = SummaryWriter(log_dir=log_dir)
        self.interval = interval

    def after_step(self, trainer: Trainer, step, data_dict, *args, **kwargs):
        accumulated_iter = trainer.global_step
        if accumulated_iter % self.interval != 0:
            return
        lr = self._get_learning_rate(trainer.optimizer)
        self.tb_log.add_scalar("epoch", trainer.epoch, accumulated_iter)
        self.tb_log.add_scalar("train/loss", kwargs["loss"], accumulated_iter)
        self.tb_log.add_scalar("meta_data/learning_rate", lr, accumulated_iter)
        if hasattr(trainer.exp, "multi_task") and trainer.exp.multi_task:
            for dataset_name, ext_dict in kwargs["extra"].items():
                if ext_dict is not None:
                    for key, val in ext_dict.items():
                        self.tb_log.add_scalar(f"train/{key}_{dataset_name}", val, accumulated_iter)
        else:
            if kwargs["extra"] is not None:
                for key, val in kwargs["extra"].items():
                    self.tb_log.add_scalar(f"train/{key}", val, accumulated_iter)
