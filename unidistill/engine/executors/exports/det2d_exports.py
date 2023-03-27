import os
import cv2

from typing import Sequence

from unidistill.exps.base_exp import BaseExp
from unidistill.engine.callbacks import Callback

from ..base_executor import BaseExecutor

from .onnx import ONNXModel
from .utils import demo_inference, visualize, nms

__all__ = ["Det2DExports"]


class Det2DExports(BaseExecutor):
    def __init__(
        self,
        exp: BaseExp,
        callbacks: Sequence["Callback"],
        logger=None,
        output_dir=None,
    ) -> None:
        super(Det2DExports, self).__init__(exp, callbacks, logger)
        self.output_dir = output_dir
        # TODO args privide filename
        self.output_jit_path = os.path.join(self.output_dir, "test.jit")
        self.output_onnx_path = os.path.join(self.output_dir, "test.onnx")
        self.output_trt_path = os.path.join(self.output_dir, "test.trt")
        self.infer_image_path = os.path.join(
            self.output_dir.split("/outputs")[0], "demo/demo.jpg"
        )

    def export_torchscript(self):
        exp = self.exp
        self.model.eval()
        exp.export_torchscript(self.output_jit_path)
        self.logger.info("export onnx model into {}".format(self.output_jit_path))

    def export_onnx(self):
        exp = self.exp
        self.model.eval()
        exp.export_onnx(self.output_onnx_path)
        self.logger.info("export onnx model into {}".format(self.output_onnx_path))
        self.demo(model_class=ONNXModel, model_path=self.output_onnx_path)

    def export_trt(self):

        from unidistill.engine.executors.exports.onnx2trt import (
            TRTEngine,
            get_trt_from_onnx,
        )

        get_trt_from_onnx(self.output_onnx_path, self.output_trt_path)
        self.logger.info("export trt model into {}".format(self.output_trt_path))
        self.demo(model_class=TRTEngine, model_path=self.output_trt_path)

    def export(self):
        self.export_torchscript()
        self.export_onnx()
        self.export_trt()

    def demo(self, model_class, model_path=None):
        image = cv2.imread(self.infer_image_path)
        num_classes = self.exp.num_classes
        model = model_class(model_path, num_classes=num_classes)

        # check your model input image type
        # yolox/cvpack2 input image: 0-255 (image)
        # torchvision input image: 0-1 (image / 255)
        result = demo_inference(
            model, image, score_thrs=0.2, num_classes=num_classes, nms_func=nms
        )
        vis_img = visualize(image, result["boxes"])
        model_type = model_path.split(".")[-1]

        vis_path = os.path.join(self.output_dir, "{}_demo.jpg".format(model_type))
        cv2.imwrite(vis_path, vis_img)
        self.logger.info("model inference done")
