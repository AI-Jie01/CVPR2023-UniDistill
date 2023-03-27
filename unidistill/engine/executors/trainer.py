from typing import Sequence
import torch
import time
from unidistill.exps.base_exp import BaseExp
from unidistill.engine.callbacks import Callback

from .base_executor import BaseExecutor

__all__ = ["Trainer"]


class Trainer(BaseExecutor):
    def __init__(
        self,
        exp: BaseExp,
        callbacks: Sequence["Callback"],
        logger=None,
        use_amp=False,
        evaluator=None,
    ) -> None:
        super(Trainer, self).__init__(exp, callbacks, logger)
        self.use_amp = use_amp
        self.evaluator = evaluator
        if self.use_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def train(self):
        self._invoke_callback("before_train")
        self.model.cuda()
        self.model.train()
        self.optimizer_to(self.optimizer, next(self.model.parameters()).device)
        start_epoch = self.epoch

        if hasattr(self.exp, "multi_task") and self.exp.multi_task:
            self.logger.info("==> Multi task training:")
            self.train_iter = {}
            for dataset_name, (dataloader, _) in self.train_dataloader.items():
                self.train_iter[dataset_name] = iter(dataloader)
            for epoch in range(start_epoch, self.exp.max_epoch):
                self.epoch = epoch
                self.model.train()
                self.train_epoch_multitask(epoch)
        else:
            self.train_iter = iter(self.train_dataloader)
            for epoch in range(start_epoch, self.exp.max_epoch):
                self.epoch = epoch
                self.model.train()
                self.train_epoch(epoch)

        self._invoke_callback("after_train")

    def train_epoch(self, epoch):
        self._invoke_callback("before_epoch", epoch)
        sampler = self.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = self.train_dataloader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
            del self.train_iter
            time.sleep(5)
            self.train_iter = iter(self.train_dataloader)
        for step in range(len(self.train_dataloader)):
            try:
                data = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                data = next(self.train_iter)
            self.train_step(data, step)
        update_best_ckpt = False
        if self.evaluator is not None:
            self._invoke_callback("before_eval")
            update_best_ckpt = self.evaluator.eval()
            self._invoke_callback("after_eval")

        self._invoke_callback("after_epoch", epoch, update_best_ckpt=update_best_ckpt)

    def train_step(self, data, step):
        self._invoke_callback("before_step", step, data)
        self.model.train()
        self.optimizer.zero_grad()
        if not self.use_amp:
            ret = self.exp.training_step(data)
        else:
            with torch.cuda.amp.autocast():
                ret = self.exp.training_step(data)
        if isinstance(ret, torch.Tensor):
            loss = ret
            ext_dict = None
        elif isinstance(ret, tuple):
            loss, ext_dict = ret
            ext_dict = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in ext_dict.items()
            }
        else:
            raise TypeError
        self._invoke_callback("before_backward")
        if not self.use_amp:
            loss.backward()
            self._invoke_callback("before_optimize")
            self.optimizer.step()
        else:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(
                self.optimizer
            )  # NOTE: grads are unscaled before "before_optimize" callbacks
            self._invoke_callback("before_optimize")
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        self._invoke_callback(
            "after_step", step, data, loss=loss.detach(), extra=ext_dict
        )
        self.global_step += 1
        self.lr_scheduler.step(self.global_step)

    def train_epoch_multitask(self, epoch):
        self._invoke_callback("before_epoch", epoch)
        if (
            hasattr(self.exp, "train_backbone_epoch")
            and epoch >= self.exp.train_backbone_epoch
        ):
            self.exp.freeze_backbone(self.model)

        for step in range(self.exp.total_step):
            ext_dicts = {}
            total_loss = 0
            for dataset_name, (_, tasks) in self.train_dataloader.items():
                train_times = self.exp.dataset_training_times[dataset_name]
                avg_ext_dict = {}
                for _ in range(train_times):
                    try:
                        data = next(self.train_iter[dataset_name])
                    except StopIteration:
                        self.train_iter[dataset_name] = iter(
                            self.train_dataloader["dataset_name"][0]
                        )
                        data = next(self.train_iter[dataset_name])
                    loss, ext_dict = self.train_step_multitask(
                        {"dataset_name": dataset_name, "data": data, "tasks": tasks},
                        step,
                    )
                    total_loss += loss / train_times

                    # avg ext_dict
                    for k, v in ext_dict.items():
                        if isinstance(v, torch.Tensor):
                            if k not in avg_ext_dict:
                                avg_ext_dict[k] = v / train_times
                            else:
                                avg_ext_dict[k] += v / train_times
                ext_dicts[dataset_name] = avg_ext_dict

            self._invoke_callback(
                "after_step", step, data, loss=total_loss, extra=ext_dicts
            )
            self.global_step += 1
            self.lr_scheduler.step(self.global_step)

        update_best_ckpt = False
        if self.evaluator is not None:
            update_best_ckpt = self.evaluator.eval()
        self._invoke_callback("after_epoch", epoch, update_best_ckpt=update_best_ckpt)

    def train_step_multitask(self, data, step):
        self._invoke_callback("before_step", step, data)
        self.model.train()
        self.optimizer.zero_grad()
        if not self.use_amp:
            ret = self.exp.training_step(data)
        else:
            with torch.cuda.amp.autocast():
                ret = self.exp.training_step(data)
        if isinstance(ret, torch.Tensor):
            loss = ret
            ext_dict = None
        elif isinstance(ret, tuple):
            loss, ext_dict = ret
            ext_dict = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in ext_dict.items()
            }
        else:
            raise TypeError
        self._invoke_callback("before_backward")
        if not self.use_amp:
            loss.backward()
            self._invoke_callback("before_optimize")
            self.optimizer.step()
        else:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(
                self.optimizer
            )  # NOTE: grads are unscaled before "before_optimize" callbacks
            self._invoke_callback("before_optimize")
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        return loss.detach(), ext_dict

    # refer to: https://github.com/pytorch/pytorch/issues/8741
    @staticmethod
    def optimizer_to(optim, device):
        for param in optim.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
