import numpy as np


# helper: map boxes from letterboxed image space back to original image space
def undo_letterbox_xywh(boxes_xywh, image_shape, ratio, dw, dh):
    """
    boxes_xywh: np.ndarray, shape [N, 4], format (cx, cy, w, h) in letterboxed 640x640 space
    image_shape: original image shape, e.g. (H, W, C) or (H, W)
    ratio, dw, dh: values returned by preprocess_yolov10/letterbox
    returns: np.ndarray, shape [N, 4], format (cx, cy, w, h) in original image space (clipped)
    """
    image_height, image_width = image_shape[:2]

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]

    # xywh -> xyxy (letterbox space)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    # remove padding, then unscale
    x1 = (x1 - dw) / ratio
    x2 = (x2 - dw) / ratio
    y1 = (y1 - dh) / ratio
    y2 = (y2 - dh) / ratio

    # clip to valid image bounds
    x1 = np.clip(x1, 0, image_width - 1)
    x2 = np.clip(x2, 0, image_width - 1)
    y1 = np.clip(y1, 0, image_height - 1)
    y2 = np.clip(y2, 0, image_height - 1)

    # back to xywh in original image space
    out_cx = (x1 + x2) / 2.0
    out_cy = (y1 + y2) / 2.0
    out_w = np.maximum(0.0, x2 - x1)
    out_h = np.maximum(0.0, y2 - y1)

    return np.stack([out_cx, out_cy, out_w, out_h], axis=1)