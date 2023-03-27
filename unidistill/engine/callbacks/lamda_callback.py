from typing import Callable, Optional

__all__ = ["LamdaCallback"]


class LamdaCallback:
    def __init__(
        self,
        setup: Optional[Callable] = None,
        load_checkpoint: Optional[Callable] = None,
        after_init: Optional[Callable] = None,
        before_train: Optional[Callable] = None,
        before_epoch: Optional[Callable] = None,
        before_step: Optional[Callable] = None,
        before_backward: Optional[Callable] = None,
        before_optimize: Optional[Callable] = None,
        after_step: Optional[Callable] = None,
        after_epoch: Optional[Callable] = None,
        after_train: Optional[Callable] = None,
    ) -> None:
        for k, v in locals().items():
            if k == "self":
                continue
            if v is not None:
                setattr(self, k, v)
