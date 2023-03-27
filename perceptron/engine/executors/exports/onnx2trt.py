import os
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit  # noqa

import tensorrt as trt
import torch

from perceptron.engine.executors.exports.utils import yolox_postprocess

trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger()


__all__ = ["get_trt_from_onnx", "TRTEngine"]


def get_trt_from_onnx(onnx_file_path, engine_file_path):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser:
        builder.max_workspace_size = 1 << 22
        builder.max_batch_size = 1
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print("ONNX file {} not found, please run export_onnx first to generate it.".format(onnx_file_path))
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))

        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))

        # network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())


class TRTEngine:
    def __init__(self, engine_file_path, num_classes=80) -> None:
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.num_classes = num_classes

    def __call__(
        self,
        image=None,
        transform=None,
        topk_candidates=1000,
        raw_result=False,
    ):
        image, scale = transform(image)

        inputs, outputs, bindings, stream = allocate_buffers(self.engine)
        inputs[0].host = image.reshape(-1)

        predictions = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # TODO it's ugly;
        if len(predictions) == 1:
            predictions = predictions[0].reshape(1, -1, self.num_classes + 5)
            predictions = yolox_postprocess(predictions, img_size=(800, 800))[0]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xywh = predictions[:, 0:4]
            boxes = np.ones_like(boxes_xywh)
            boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
            boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
            boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
            boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        else:
            scores, boxes = predictions
        scores = scores.reshape(-1, self.num_classes)
        boxes = boxes.reshape(-1, 4)

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


class BevDepthTRTEngine:
    def __init__(self, engine_file_path, num_classes=80) -> None:
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.num_classes = num_classes

    def __call__(
        self,
        sweep_imgs=None,
        mats=None,
    ):

        mats_keys = ["sensor2ego_mats", "intrin_mats", "ida_mats", "bda_mat"]
        inverse_keys = ["intrin_mats", "ida_mats"]
        inputs, outputs, bindings, stream = allocate_buffers(self.engine)

        inputs[0].host = sweep_imgs.reshape(-1)
        for i, key in enumerate(mats_keys):
            if key in inverse_keys:
                mats[key] = np.linalg.inv(mats[key])
            inputs[i + 1].host = mats[key].reshape(-1)

        predictions = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        decode_keys = ["reg", "height", "dim", "rot", "vel", "heatmap"]
        preds = []
        for rank in range(0, 36, 6):
            rank_dict = {}
            for key in range(6):
                pred = predictions[rank + key]
                rank_dict[decode_keys[key]] = torch.from_numpy(pred).reshape(1, -1, 128, 128)
            preds.append([rank_dict])

        return preds


# from tensorrt.common
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host: \n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# from tensorrt.common
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


# from tensorrt.common
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]
