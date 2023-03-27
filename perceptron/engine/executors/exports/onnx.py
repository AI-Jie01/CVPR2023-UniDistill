import numpy as np

from perceptron.engine.executors.exports.utils import yolox_postprocess

__all__ = ["ONNXModel"]


class ONNXModel:
    """Forward the input images by an onnx model."""

    def __init__(self, onnx_path, num_classes=80):
        import onnxruntime

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1

        self.onnx_session = onnxruntime.InferenceSession(onnx_path, sess_options=so)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        self.num_classes = num_classes

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def __call__(
        self,
        image=None,
        transform=None,
        topk_candidates=1000,
        raw_result=False,
    ):
        image, scale = transform(image)
        predictions = self.onnx_session.run(self.output_name, input_feed={"images": image})
        if "yolox_output" in self.output_name:
            predictions = yolox_postprocess(predictions[0], img_size=(800, 800))[0]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xywh = predictions[:, 0:4]
            boxes = np.ones_like(boxes_xywh)
            boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
            boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
            boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
            boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        else:
            scores, boxes = predictions
        keep = scores.max(axis=1) > 0.1
        scores = scores[keep]
        boxes = boxes[keep]

        if raw_result:
            return boxes, scores

        scores = scores.flatten()
        # Keep top k top scoring indices only.
        num_topk = min(topk_candidates, len(boxes))
        # torch.sort is actually faster than .topk (at least on GPUs)
        topk_idxs = np.argsort(scores)

        scores = scores[topk_idxs][-num_topk:]
        topk_idxs = topk_idxs[-num_topk:]

        # filter out the proposals with low confidence score
        shift_idxs = topk_idxs // self.num_classes
        classes = topk_idxs % self.num_classes
        boxes = boxes[shift_idxs]

        # out: [boxes, scores]
        boxes = boxes * (*scale, *scale)
        return boxes, scores, classes
