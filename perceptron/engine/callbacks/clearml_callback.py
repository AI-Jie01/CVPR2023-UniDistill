from clearml import Task

from loguru import logger
from .base_callback import MasterOnlyCallback
from perceptron.engine.executors import BaseExecutor

__all__ = ["ClearMLCallback"]


class ClearMLCallback(MasterOnlyCallback):
    def __init__(self):
        super().__init__()
        self.task_id = None

    def after_init(self, executor: BaseExecutor):
        if self.task_id is None:
            self.task = Task.init(
                project_name=executor.exp.project_name,
                task_name=executor.exp.exp_name,
                auto_connect_frameworks={"pytorch": False},
                reuse_last_task_id=False,
                continue_last_task=False,
            )
        else:
            self.task = Task.get_task(task_id=self.task_id)
            self.task.add_tags(["resume"])
            logger.info(f"continue from clearml task {self.task_id}")
        self.task.connect(executor.exp)
        if hasattr(executor.exp, "get_pcdet_cfg"):
            self.task.connect(executor.exp.get_pcdet_cfg(), "pcdet_config")

    def state_dict(self):
        return {"task_id": self.task.task_id}

    def load_state_dict(self, state_dict):
        self.task_id = state_dict["task_id"]
