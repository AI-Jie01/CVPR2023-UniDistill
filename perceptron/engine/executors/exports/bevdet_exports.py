import os

from typing import Sequence

from perceptron.exps.base_exp import BaseExp
from perceptron.engine.callbacks import Callback

from ..base_executor import BaseExecutor


__all__ = ["BevDetExports"]


class BevDetExports(BaseExecutor):
    def __init__(self, exp: BaseExp, callbacks: Sequence["Callback"], logger=None, output_dir=None) -> None:
        super(BevDetExports, self).__init__(exp, callbacks, logger)
        self.output_dir = output_dir
        # TODO args privide filename
        self.output_onnx_path = os.path.join(self.output_dir, "test_bev.onnx")
        self.output_trt_path = os.path.join(self.output_dir, "test_bev.trt")

    def export_onnx(self):
        exp = self.exp
        self.model.eval()
        exp.export_onnx(self.output_onnx_path)
        self.logger.info("export onnx model into {}".format(self.output_onnx_path))

    def export_trt(self):

        from perceptron.engine.executors.exports.onnx2trt import get_trt_from_onnx

        get_trt_from_onnx(self.output_onnx_path, self.output_trt_path)
        self.logger.info("export trt model into {}".format(self.output_trt_path))

    def export(self):
        self.export_onnx()
        self.export_trt()
