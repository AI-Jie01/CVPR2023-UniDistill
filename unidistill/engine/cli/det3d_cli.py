import sys
from unidistill.engine.callbacks import (
    CheckPointLoader,
    EvalResultsSaver,
    ClearMLCallback,
)
from unidistill.engine.executors import Det3DEvaluator
from unidistill.utils.log_utils import setup_logger
from unidistill.utils.misc import PyDecorator

from .base_cli import BaseCli

__all__ = ["Det3DCli"]


class Det3DCli(BaseCli):
    @PyDecorator.overrides(BaseCli)
    def get_evaluator(self, callbacks=None):
        exp = self.exp
        if self.args.ckpt is None and self.args.eval:
            sys.exit("No checkpoint is specified for evaluation")
        output_dir = self._get_exp_output_dir()
        exp.output_dir = output_dir
        logger = setup_logger(
            output_dir, distributed_rank=self.args.local_rank, filename="eval.log"
        )
        self._set_basic_log_message(logger)
        if callbacks is None:
            callbacks = [
                CheckPointLoader(self.args.ckpt, weight_only=True),
                EvalResultsSaver(exp.output_dir),
            ]
        if self.args.clearml:
            callbacks.append(ClearMLCallback())

        if not hasattr(exp, "eval_executor_class"):
            exp.eval_executor_class = Det3DEvaluator

        if self.args.eval or self.args.train_and_eval:
            evaluator = exp.eval_executor_class(
                exp=exp, callbacks=callbacks, logger=logger
            )
        else:
            raise NotImplementedError("Train Mode has no evaluator!")
        return evaluator
