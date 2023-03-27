import multiprocessing
import os
import subprocess
import sys
import argparse
import datetime
import warnings

from unidistill.exps import global_cfg
from unidistill.engine.callbacks import (
    CheckPointLoader,
    CheckPointSaver,
    ClearMLCallback,
    ProgressBar,
    TensorBoardMonitor,
    TextMonitor,
    ClipGrad,
    Profiler,
)
from unidistill.utils import torch_dist
from unidistill.utils.misc import sanitize_filename
from unidistill.utils.log_utils import setup_logger
from unidistill.utils.env import collect_env_info
from unidistill.engine.executors import Trainer
import torch

__all__ = ["BaseCli"]


class BaseCli:
    """Command line tools for any exp."""

    def __init__(self, Exp):
        """Make sure the order of initialization is: build_args --> build_env --> build_exp,
        since experiments depend on the environment and the environment depends on args.

        Args:
            Exp : experiment description class
        """
        self.ExpCls = Exp
        self.args = self._get_parser(Exp).parse_args()
        self.world_size = torch_dist.get_world_size()

    @property
    def exp(self):
        if not hasattr(self, "_exp"):
            exp = self.ExpCls(
                **{
                    x if y is not None else "none": y
                    for (x, y) in vars(self.args).items()
                },
                total_devices=self.world_size,
            )
            self.exp_updated_cfg_msg = exp.update_attr(self.args.exp_options)
            self._exp = exp
        return self._exp

    def _get_parser(self, Exp):
        parser = argparse.ArgumentParser()
        parser = Exp.add_argparse_args(parser)
        parser = self.add_argparse_args(parser)
        return parser

    def add_argparse_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--eval", dest="eval", action="store_true", help="conduct evaluation only"
        )
        parser.add_argument(
            "-te",
            "--train_and_eval",
            dest="train_and_eval",
            action="store_true",
            help="simultaneously train and eval",
        )
        parser.add_argument(
            "--export", dest="export", action="store_true", help="conduct exports only"
        )
        parser.add_argument(
            "--infer", dest="infer", action="store_true", help="conduct inference only"
        )
        parser.add_argument(
            "--profile",
            dest="profile",
            action="store_true",
            help="profile while train or eval",
        )
        parser.add_argument(
            "--find_unused_parameters",
            dest="find_unused_parameters",
            action="store_true",
            help="Whether to enable find_unused_parameters in DDP",
        )
        parser.add_argument(
            "-d", "--devices", default="0-7", type=str, help="device for training"
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default=None,
            help="checkpoint to start from or be evaluated",
        )
        parser.add_argument(
            "--pretrained_model",
            type=str,
            default=None,
            help="pretrained_model used by training",
        )
        parser.add_argument(
            "--sync_bn",
            type=int,
            default=0,
            help="sync bn nr_device_per_group (specially 0-> disable sync_bn, 1-> whole world)",
        )
        parser.add_argument(
            "--amp", dest="use_amp", action="store_true", help="use pytorch native amp"
        )
        parser.add_argument("--local_rank", type=int)
        clearml_parser = parser.add_mutually_exclusive_group(required=False)
        clearml_parser.add_argument(
            "--clearml",
            dest="clearml",
            action="store_true",
            help="enabel clearml for training",
        )
        clearml_parser.add_argument(
            "--no-clearml", dest="clearml", action="store_false", help="disable clearml"
        )
        parser.set_defaults(clearml=True)
        return parser

    def _get_exp_output_dir(self):
        exp_dir = os.path.join(
            global_cfg.output_root_dir, sanitize_filename(self.exp.exp_name)
        )
        os.makedirs(exp_dir, exist_ok=True)
        output_dir = None
        if self.args.ckpt:
            output_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(self.args.ckpt))
            )
        elif self.args.local_rank == 0:
            output_dir = os.path.join(
                exp_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            )
            os.makedirs(output_dir, exist_ok=True)
            # make a symlink "latest"
            symlink, symlink_tmp = os.path.join(exp_dir, "latest"), os.path.join(
                exp_dir, "latest_tmp"
            )
            if os.path.exists(symlink_tmp):
                os.remove(symlink_tmp)
            os.symlink(os.path.relpath(output_dir, exp_dir), symlink_tmp)
            os.rename(symlink_tmp, symlink)
        output_dir = torch_dist.all_gather_object(output_dir)[0]
        return output_dir

    def get_evaluator(self, callbacks=None):
        exp = self.exp
        if self.args.ckpt is None:
            warnings.warn("No checkpoint is specified for evaluation")
        if exp.eval_executor_class is None:
            sys.exit("No evaluator is specified for evaluation")

        output_dir = self._get_exp_output_dir()
        logger = setup_logger(
            output_dir, distributed_rank=self.args.local_rank, filename="eval.log"
        )
        self._set_basic_log_message(logger)
        if callbacks is None:
            callbacks = [
                CheckPointLoader(self.args.ckpt, weight_only=True),
            ]
        if self.args.profile:
            callbacks.append(
                Profiler(output_dir, getattr(self.exp, "profile_cfg", None))
            )
        evaluator = exp.eval_executor_class(exp=exp, callbacks=callbacks, logger=logger)
        return evaluator

    def get_exports(self, callbacks=None):
        exp = self.exp
        if self.args.ckpt is None:
            sys.exit("No checkpoint is specified for export")
        if exp.export_executor_class is None:
            sys.exit("No exports is specified for export")

        output_dir = self._get_exp_output_dir()
        logger = setup_logger(
            output_dir, distributed_rank=self.args.local_rank, filename="export.log"
        )
        self._set_basic_log_message(logger)
        if callbacks is None:
            callbacks = [
                CheckPointLoader(self.args.ckpt, weight_only=True),
            ]

        exports = exp.export_executor_class(
            exp=exp, callbacks=callbacks, logger=logger, output_dir=output_dir
        )
        return exports

    def _set_basic_log_message(self, logger):
        logger.opt(ansi=True).info(
            "<yellow>Cli arguments:</yellow>\n<blue>{}</blue>".format(self.args)
        )
        logger.info(f"exp_name: {self.exp.exp_name}")
        logger.opt(ansi=True).info(
            "<yellow>Used experiment configs</yellow>:\n<blue>{}</blue>".format(
                self.exp.get_cfg_as_str()
            )
        )
        if self.exp_updated_cfg_msg:
            logger.opt(ansi=True).info(
                "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(
                    self.exp_updated_cfg_msg
                )
            )
        logger.opt(ansi=True).info(
            "<yellow>Environment info:</yellow>\n<blue>{}</blue>".format(
                collect_env_info()
            )
        )

    def get_trainer(self, callbacks=None, evaluator=None):
        args = self.args
        exp = self.exp
        output_dir = self._get_exp_output_dir()

        logger = setup_logger(
            output_dir, distributed_rank=self.args.local_rank, filename="train.log"
        )
        self._set_basic_log_message(logger)

        if callbacks is None:
            callbacks = [
                ProgressBar(logger=logger),
                TextMonitor(interval=exp.print_interval),
                TensorBoardMonitor(
                    os.path.join(output_dir, "tensorboard"), interval=exp.print_interval
                ),
                CheckPointSaver(
                    local_path=os.path.join(output_dir, "dump_model"),
                    remote_path=exp.ckpt_oss_save_dir,
                    save_interval=exp.dump_interval,
                    num_keep_latest=exp.num_keep_latest_ckpt,
                ),
            ]
        if "grad_clip_value" in exp.__dict__:
            callbacks.append(ClipGrad(exp.grad_clip_value))
        if args.clearml:
            callbacks.append(ClearMLCallback())
        if args.ckpt:
            callbacks.append(CheckPointLoader(args.ckpt))
        if args.pretrained_model:
            callbacks.append(CheckPointLoader(args.pretrained_model, weight_only=True))
        if args.profile:
            callbacks.append(
                Profiler(output_dir, getattr(self.exp, "profile_cfg", None))
            )
        callbacks.extend(exp.callbacks)

        trainer = Trainer(
            exp=exp,
            callbacks=callbacks,
            logger=logger,
            use_amp=args.use_amp,
            evaluator=evaluator,
        )
        return trainer

    def get_infer(self, callbacks=None):
        exp = self.exp
        if self.args.ckpt is None:
            warnings.warn("No checkpoint is specified for inference")
        if exp.infer_executor_class is None:
            sys.exit("No inferer is specified for inference")

        output_dir = self._get_exp_output_dir()
        exp.output_dir = output_dir
        logger = setup_logger(
            output_dir, distributed_rank=self.args.local_rank, filename="infer.log"
        )
        self._set_basic_log_message(logger)
        if callbacks is None:
            callbacks = [
                CheckPointLoader(self.args.ckpt, weight_only=True),
            ]
        infer = exp.infer_executor_class(exp=exp, callbacks=callbacks, logger=logger)
        return infer

    def get_trainer_and_evaluator(
        self,
        callbacks=None,
    ):
        args = self.args
        exp = self.exp
        if exp.eval_executor_class is None:
            sys.exit("No evaluator is specified for evaluation")

        output_dir = self._get_exp_output_dir()
        exp.output_dir = output_dir
        logger = setup_logger(
            output_dir, distributed_rank=self.args.local_rank, filename="train_eval.log"
        )
        self._set_basic_log_message(logger)
        evaluator = exp.eval_executor_class(exp=exp, callbacks=[], logger=logger)
        if callbacks is None:
            callbacks = [
                ProgressBar(logger=logger),
                TextMonitor(interval=exp.print_interval),
                TensorBoardMonitor(
                    os.path.join(output_dir, "tensorboard"), interval=exp.print_interval
                ),
                CheckPointSaver(
                    local_path=os.path.join(output_dir, "dump_model"),
                    remote_path=exp.ckpt_oss_save_dir,
                    save_interval=exp.dump_interval,
                    num_keep_latest=exp.num_keep_latest_ckpt,
                ),
            ]
        if "grad_clip_value" in exp.__dict__:
            callbacks.append(ClipGrad(exp.grad_clip_value))
        if args.clearml:
            callbacks.append(ClearMLCallback())
        if args.ckpt:
            callbacks.append(CheckPointLoader(args.ckpt))
        if args.pretrained_model:
            callbacks.append(CheckPointLoader(args.pretrained_model, weight_only=True))
        callbacks.extend(exp.callbacks)

        trainer = Trainer(
            exp=exp,
            callbacks=callbacks,
            logger=logger,
            use_amp=args.use_amp,
            evaluator=evaluator,
        )
        return trainer

    def executor(self):
        if self.args.eval:
            self.get_evaluator().eval()
        elif self.args.train_and_eval:
            self.get_trainer_and_evaluator().train()
        elif self.args.export:
            self.get_exports().export()
        elif self.args.infer:
            self.get_infer().infer()
        else:
            self.get_trainer().train()

    def run(self):
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(self.args.local_rank)
        torch.distributed.barrier()
        self.executor()
