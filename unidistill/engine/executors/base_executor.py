from typing import Sequence

from unidistill.exps.base_exp import BaseExp
from unidistill.utils import torch_dist
from unidistill.engine.callbacks import Callback
import torch

__all__ = ["BaseExecutor"]


class BaseExecutor:
    def __init__(
        self, exp: BaseExp, callbacks: Sequence["Callback"], logger=None
    ) -> None:
        self.exp = exp
        self.callbacks = callbacks
        self.logger = logger
        self._invoke_callback("setup")

        self.epoch = 0
        self.global_step = 0
        self._invoke_callback("load_checkpoint")
        self._invoke_callback("after_init")
        self.exp.model.cuda()
        self.exp.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.exp.model)
        self.exp.model = torch.nn.parallel.DistributedDataParallel(
            self.exp.model,
            device_ids=[torch.distributed.get_rank()],
            find_unused_parameters=False,
        )

    @property
    def train_dataloader(self):
        return self.exp.train_dataloader

    @property
    def val_dataloader(self):
        return self.exp.val_dataloader

    @property
    def model(self):
        return self.exp.model

    @model.setter
    def model(self, value):
        self.exp.model = value

    @property
    def optimizer(self):
        return self.exp.optimizer

    @property
    def lr_scheduler(self):
        return self.exp.lr_scheduler

    def _invoke_callback(self, callback_name, *args, **kwargs):
        for cb in self.callbacks:
            if cb.enabled_rank is None or self.global_rank in cb.enabled_rank:
                func = getattr(cb, callback_name, None)
                if func:
                    func(self, *args, **kwargs)

    @property
    def global_rank(self):
        return torch_dist.get_rank()
