import cv2
import numpy as np


def letterbox(img, new_shape, padding_colour):
    """maintains aspect ratio by scaling down + adding padding"""
    
    shape = img.shape[:2]

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into two sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_colour)

    return img, ratio, (dw, dh)

def preprocess_yolov10(img):
    """
    Standard YOLOv10 preprocessing.
    Reference: ultralytics/engine/predictor.py
    """
    input_size = (640, 640)
    padding_colour = (114, 114, 114)

    img, ratio, (dw, dh) = letterbox(img, input_size,  # 1. Letterbox transformation
                                    padding_colour)   
    img = img[:, :, ::-1]                              # 2. BGR to RGB 
    img = img.transpose(2, 0, 1)                       # 3. HWC to CHW
    img = img.astype(np.float32) / 255.0               # 4. Normalize to [0, 1]
    
    img = np.expand_dims(img, axis=0)                  # add batch dim → [1,3,H,W]
    
    return img, ratio, dw, dh
