from typing import Sequence

from tqdm import tqdm

from perceptron.engine.callbacks import Callback
from perceptron.exps.base_exp import BaseExp
from perceptron.utils import torch_dist

from ..base_executor import BaseExecutor

__all__ = ["Det3DInfer"]


class Det3DInfer(BaseExecutor):
    def __init__(self, exp: BaseExp, callbacks: Sequence[Callback], logger=None) -> None:
        super().__init__(exp, callbacks, logger)

    @property
    def test_dataloader(self):
        return self.exp.test_dataloader

    def infer(self):
        exp = self.exp
        local_rank = torch_dist.get_rank()

        self.infer_iter = iter(self.test_dataloader)
        self.model.cuda()
        self.model.eval()

        preds = []
        for step in tqdm(range(len(self.infer_iter)), disable=(local_rank > 0)):
            data = next(self.infer_iter)
            pred_item = exp.test_step(data)
            if type(pred_item) == dict and "pred_dicts" in pred_item:
                pred_dicts = pred_item["pred_dicts"]
                preds += self.test_dataloader.dataset.generate_prediction_dicts(
                    data, pred_dicts, exp.model.module.class_names
                )
            # mmdetection e.g. FCOS3D
            elif type(pred_item) == list:
                preds += pred_item
            else:
                raise NotImplementedError

        torch_dist.synchronize()
        preds = sum(map(list, zip(*torch_dist.all_gather_object(preds))), [])[: len(self.test_dataloader.dataset)]
        if local_rank == 0:
            self.logger.info("dump inference results ...")
            self.test_dataloader.dataset.dump_inference_results(preds, output_dir=exp.output_dir)
