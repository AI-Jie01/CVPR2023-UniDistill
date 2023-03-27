import math
import numpy as np
from functools import partial


__all__ = [
    "OnecycleLRScheduler",
    "CosineLRScheduler",
    "WarmCosineLRScheduler",
    "YoloxWarmCosineLRScheduler",
    "StepLRScheduler",
    "GroupLRScheduler",
    "OnecycleLRScheduler",
]


class _LRScheduler:
    def __init__(self, optimizer, lr, iters_per_epoch, total_epochs):
        self._optimizer = optimizer
        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs
        self._get_lr = self._get_lr_func()
        self.step(0)

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

    def step(self, iters):
        return self.update_lr(iters)

    def update_lr(self, iters):
        lr = self._get_lr(iters)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr_func(self):
        raise NotImplementedError


class CosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, lr, iters_per_epoch, total_epochs, end_lr=0.0):
        self.end_lr = end_lr
        super(CosineLRScheduler, self).__init__(
            optimizer, lr, iters_per_epoch, total_epochs
        )

    def _get_lr_func(self):
        end_lr = self.end_lr
        if end_lr > 0:
            lr_func = partial(cos_w_end_lr, self.lr, self.total_iters, end_lr)
        else:
            lr_func = partial(cos_lr, self.lr, self.total_iters)
        return lr_func


class WarmCosineLRScheduler(CosineLRScheduler):
    def __init__(
        self,
        optimizer,
        lr,
        iters_per_epoch,
        total_epochs,
        warmup_epochs=0.0,
        warmup_lr_start=1e-6,
        end_lr=0.0,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        super(WarmCosineLRScheduler, self).__init__(
            optimizer, lr, iters_per_epoch, total_epochs, end_lr
        )

    def _get_lr_func(self):
        warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
        if warmup_total_iters == 0:
            return super(WarmCosineLRScheduler, self)._get_lr_func()
        warmup_lr_start = self.warmup_lr_start
        end_lr = self.end_lr
        if end_lr > 0:
            lr_func = partial(
                warm_cos_w_end_lr,
                self.lr,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
                end_lr,
            )
        else:
            lr_func = partial(
                warm_cos_lr,
                self.lr,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
            )
        return lr_func


class YoloxWarmCosineLRScheduler(WarmCosineLRScheduler):
    def __init__(
        self,
        optimizer,
        lr,
        iters_per_epoch,
        total_epochs,
        warmup_epochs=0.0,
        warmup_lr_start=1e-6,
        no_aug_epochs=0,
        min_lr_ratio=0.2,
    ):
        self.no_aug_epochs = no_aug_epochs
        self.min_lr_ratio = min_lr_ratio
        super(YoloxWarmCosineLRScheduler, self).__init__(
            optimizer,
            lr,
            iters_per_epoch,
            total_epochs,
            warmup_epochs,
            warmup_lr_start,
            end_lr=0.0,
        )

    def _get_lr_func(self):
        warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
        if warmup_total_iters == 0:
            return super(WarmCosineLRScheduler, self)._get_lr_func()
        warmup_lr_start = self.warmup_lr_start
        no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
        lr_func = partial(
            yolox_warm_cos_lr,
            self.lr,
            self.min_lr_ratio,
            self.total_iters,
            warmup_total_iters,
            warmup_lr_start,
            no_aug_iters,
        )
        return lr_func


class StepLRScheduler(_LRScheduler):
    def __init__(
        self, optimizer, lr, iters_per_epoch, total_epochs, milestones, gamma=0.1
    ):
        self.milestones = milestones
        self.gamma = gamma
        super(StepLRScheduler, self).__init__(
            optimizer, lr, iters_per_epoch, total_epochs
        )

    def _get_lr_func(self):
        milestones = [
            int(self.total_iters * milestone / self.total_epochs)
            for milestone in self.milestones
        ]
        gamma = self.gamma
        lr_func = partial(multistep_lr, self.lr, milestones, gamma)
        return lr_func


class WarmupStepLRScheduler(StepLRScheduler):
    def __init__(
        self,
        optimizer,
        lr,
        iters_per_epoch,
        total_epochs,
        milestones,
        warmup_total_iter=1000,
        warmup_factor=1.0 / 1000,
        gamma=0.1,
    ):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_total_iter = warmup_total_iter
        self.warmup_factor = warmup_factor
        super(WarmupStepLRScheduler, self).__init__(
            optimizer, lr, iters_per_epoch, total_epochs, milestones, gamma
        )

    def _get_lr_func(self):
        warmup_total_iters = min(self.iters_per_epoch, self.warmup_total_iter)
        if warmup_total_iters == 0:
            return super()._get_lr_func()

        milestones = [
            int(self.total_iters * milestone / self.total_epochs)
            for milestone in self.milestones
        ]
        gamma = self.gamma
        warmup_factor = self.warmup_factor
        lr_func = partial(
            warm_linear_lr,
            self.lr,
            milestones,
            gamma,
            warmup_total_iters,
            warmup_factor,
        )
        return lr_func


class ConstantLRScheduler(_LRScheduler):
    def __init__(self, optimizer, lr, iters_per_epoch, total_epochs):
        super(ConstantLRScheduler, self).__init__(
            optimizer, lr, iters_per_epoch, total_epochs
        )

    def _get_lr_func(self):
        return lambda _: self.lr


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters < warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr


def yolox_warm_cos_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    no_aug_iter,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters < warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
        )
    return lr


def multistep_lr(lr, milestones, gamma, iters):
    """MultiStep learning rate"""
    for milestone in milestones:
        lr *= gamma if iters >= milestone else 1.0
    return lr


def warm_linear_lr(lr, milestones, gamma, warmup_total_iters, warmup_factor, iters):
    if iters >= warmup_total_iters:
        lr = multistep_lr(lr, milestones, gamma, iters)
    else:
        alpha = float(iters) / warmup_total_iters
        lr *= warmup_factor * (1 - alpha) + alpha
    return lr


def warm_cos_w_end_lr(
    lr, total_iters, warmup_total_iters, warmup_lr_start, end_lr, iters
):
    """Cosine learning rate with warm up."""
    if iters < warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start
    else:
        q = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
        lr = lr * q + end_lr * (1 - q)
    return lr


def cos_w_end_lr(lr, total_iters, end_lr, iters):
    """Cosine learning rate"""
    q = 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    lr = lr * q + end_lr * (1 - q)
    return lr


class GroupLRScheduler(_LRScheduler):
    """
    attach params_groups in optimizer with different lr_scheduler
    scheduler_dict: {"group1": scheduler1;, "group2": ...}
    """

    def __init__(
        self,
        scheduler_dict,
        optimizer,
        lr_dict,
        iters_per_epoch,
        total_epochs,
        **kwargs
    ):
        assert isinstance(scheduler_dict, dict)
        self.scheduler_dict = scheduler_dict
        self.lr_dict = lr_dict
        self._current_lr_dict = lr_dict.copy()
        super(GroupLRScheduler, self).__init__(
            optimizer, lr_dict, iters_per_epoch, total_epochs
        )

    def update_lr(self, iters):
        for param_group in self._optimizer.param_groups:
            name = param_group["name"]
            lr = self._get_lr[name](iters)
            param_group["lr"] = lr
            self._current_lr_dict[name] = lr
        return self._current_lr_dict

    def _get_lr_func(self):
        lr_func_dict = {}
        for name in self.scheduler_dict:
            lr_scheduler = self.scheduler_dict[name]
            assert isinstance(lr_scheduler, _LRScheduler)
            lr_func_dict[name] = lr_scheduler._get_lr
        return lr_func_dict


class OnecycleLRScheduler(_LRScheduler):
    def __init__(
        self, optimizer, lr, iters_per_epoch, total_epochs, moms, div_factor, pct_start
    ):
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.total_iters = iters_per_epoch * total_epochs
        optimizer.lr, optimizer.mom = lr, self.moms[0]

        self.lr_phases = (
            (
                0,
                int(self.total_iters * self.pct_start),
                partial(annealing_cos, lr / self.div_factor, lr),
            ),
            (
                int(self.total_iters * self.pct_start),
                self.total_iters,
                partial(annealing_cos, lr, lr / self.div_factor / 1e4),
            ),
        )
        self.mom_phases = (
            (
                0,
                int(self.total_iters * self.pct_start),
                partial(annealing_cos, *self.moms),
            ),
            (
                int(self.total_iters * self.pct_start),
                self.total_iters,
                partial(annealing_cos, *self.moms[::-1]),
            ),
        )
        super(OnecycleLRScheduler, self).__init__(
            optimizer, lr, iters_per_epoch, total_epochs
        )

    def _get_lr_func(self):
        pass

    def update_lr(self, iters):
        for start, end, func in self.lr_phases:
            if iters >= start:
                lr = func((iters - start) / (end - start))
                self._optimizer.lr = lr
        for start, end, func in self.mom_phases:
            if iters >= start:
                mom = func((iters - start) / (end - start))
                self._optimizer.mom = mom
        return lr, mom


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class TorchLRSchedulerWraper(_LRScheduler):
    def __init__(self, torch_lr_scheduler, iters_per_epoch, total_epochs):
        self.torch_lr_scheduler = torch_lr_scheduler
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs

    def step(self, iters):
        if iters % self.iters_per_epoch == 0:
            self.torch_lr_scheduler.step(iters // self.iters_per_epoch)
        lr = self.torch_lr_scheduler.get_last_lr()
        return lr

    def state_dict(self):
        return self.torch_lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        return self.torch_lr_scheduler.load_state_dict(state_dict)

    def update_lr(self, iters):
        pass

    def _get_lr_func(self):
        pass
