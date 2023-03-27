import os
import torch
from unidistill.utils import torch_dist
from .base_callback import Callback

__all__ = ["Profiler"]


class Profiler(Callback):
    def __init__(self, output_dir, profile_cfg=None):
        self.output_dir = output_dir
        self.profile_cfg = dict(
            wait=50,
            warmup=3,
            active=3,
            repeat=1,
            dir_name="profile",
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        if profile_cfg is not None:
            self.profile_cfg.update(profile_cfg)

    def _get_prof(self):
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.profile_cfg["wait"],
                warmup=self.profile_cfg["warmup"],
                active=self.profile_cfg["active"],
                repeat=self.profile_cfg["repeat"],
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(self.output_dir, self.profile_cfg["dir_name"]),
                worker_name=f"worker{torch_dist.get_rank()}",
            ),
            record_shapes=self.profile_cfg["record_shapes"],
            profile_memory=self.profile_cfg["profile_memory"],
            with_stack=self.profile_cfg["with_stack"],
        )
        return prof

    def before_epoch(self, executor, epoch):
        if epoch == 0:
            self.prof = self._get_prof()
            self.prof.__enter__()

    def after_step(self, executor, step, data_dict, *args, **kwargs):
        if hasattr(self, "prof"):
            self.prof.step()

    def after_epoch(self, executor, epoch, update_best_ckpt=False):
        if epoch == 0:
            self.prof.__exit__(None, None, None)
            del self.prof

    def before_eval(self, *args, **kwargs):
        self.prof = self._get_prof()
        self.prof.__enter__()

    def after_eval(self, *args, **kwargs):
        self.prof.__exit__(None, None, None)
        del self.prof
