from torch.nn.utils import clip_grad_norm_

from .base_callback import Callback

__all__ = ["ClipGrad"]


class ClipGrad(Callback):
    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def before_optimize(self, trainer):
        clip_grad_norm_(trainer.model.parameters(), self.max_norm)
