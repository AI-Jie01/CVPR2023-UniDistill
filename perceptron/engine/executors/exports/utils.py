from functools import partial
import numpy as np
import cv2


def demo_inference(
    model,
    image,
    score_thrs,
    num_classes=80,
    nms_thr=0.6,
    nms_func=None,
    target_shape=(800, 800),
):

    transform = partial(default_transform, target_shape=target_shape)
    boxes, scores, cls_idxs = model(np.ascontiguousarray(image, dtype=np.float32), transform=transform)
    assert len(boxes) == len(scores) and len(boxes) == len(cls_idxs)

    if isinstance(score_thrs, float):
        keep = scores > max(score_thrs, 0.2)
    else:
        score_thrs = np.asarray(score_thrs)
        keep = scores > np.maximum(score_thrs[cls_idxs], 0.2)

    pred_boxes = np.concatenate([boxes, scores[:, np.newaxis], cls_idxs[:, np.newaxis]], axis=1)
    pred_boxes = pred_boxes[keep]

    all_boxes = []
    for cls_idx in range(num_classes):
        keep_per_cls = pred_boxes[:, -1] == cls_idx
        if keep_per_cls.sum() > 0:
            pred_boxes_per_cls = pred_boxes[keep_per_cls].astype(np.float32)
            keep_idx = nms_func(pred_boxes_per_cls[:, :4], pred_boxes_per_cls[:, -1], nms_thr=nms_thr)
            for idx in keep_idx:
                all_boxes.append(
                    {
                        "class_name": cls_idx,
                        "x": float(pred_boxes_per_cls[idx][0]),
                        "y": float(pred_boxes_per_cls[idx][1]),
                        "w": float(pred_boxes_per_cls[idx][2] - pred_boxes_per_cls[idx][0]),
                        "h": float(pred_boxes_per_cls[idx][3] - pred_boxes_per_cls[idx][1]),
                        "score": float(pred_boxes_per_cls[idx][4]),
                    }
                )
    return {"boxes": all_boxes}


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def default_transform(img, target_shape=(800, 800)):

    h, w = img.shape[:2]
    ret = cv2.resize(img, (target_shape[0], target_shape[1]))
    ret = ret.astype(np.float32)
    ret = np.expand_dims(ret.transpose(2, 0, 1), 0)
    return ret, (w / target_shape[0], h / target_shape[1])


def visualize(img, result, conf=0.5):

    for boxes in result:
        class_name = boxes["class_name"]
        score = boxes["score"]
        if score < conf:
            continue
        x = boxes["x"]
        y = boxes["y"]
        w = boxes["w"]
        h = boxes["h"]
        x0 = int(x)
        y0 = int(y)
        x1 = int(x + w)
        y1 = int(y + h)

        _COLOR = np.array([0.000, 0.447, 0.741])
        img = img.copy()
        color = (_COLOR * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_name, score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLOR) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLOR * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def yolox_postprocess(outputs, img_size=(800, 800), p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs
