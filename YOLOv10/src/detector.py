from dataclasses import dataclass

import numpy as np
import torch
from preprocessing import preprocess_yolov10
from ultralytics import YOLOv10
from postprocessing import undo_letterbox_xywh

@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: tuple  # (cx, cy, w, h)

class YOLOv10Detector():
    def __init__(self, weights, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        
        self.model = YOLOv10(weights)
        
    # interface for the user, images ---> detections
    def get_detections(self, img):
        preprocessed_image, ratio, dw, dh = self._preprocess(img)
        raw_predictions = self._predict(preprocessed_image)
        detections = self._postprocess(raw_predictions, img.shape, ratio, dw, dh)
        return detections

    # private methods
    def _predict(self, processed_img):
        """
        Returns model raw output dict in eval/export=False mode:
        {
            "one2many": (y_many, x_many),
            "one2one": (y_one, x_one)
        }

        Where y is a tensor with shape (y_many : more than 1 anchors for the object)
        [BATCH_SIZE, 4 + NC (cx, cy, w, h + categories), ANCHOR_POINTS]
        """
        tensor = torch.from_numpy(processed_img)
        self.model.model.eval()
        with torch.no_grad():
            predictions = self.model.model(tensor)
        return predictions
        
    def _preprocess(self, img):
        return preprocess_yolov10(img)
    
    # convert the detector's output into standard Data Structure for detections
    def _postprocess(self, predictions, img_shape, ratio, dw, dh):
        """
        predictions: dict from v10Detect forward in eval mode
        img_shape: original image shape (H, W, C)
        ratio, dw, dh: letterbox params from preprocess
        """
        detections = []

        # predictions["one2one"] is (y_one, raw_feats)
        y = predictions["one2one"][0]  # tensor [B, 4+nc, N]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        pred = y[0].transpose(1, 0)  # [N, 4+nc]
        boxes_xywh = pred[:, :4]     # in letterboxed image coordinates (640-space)
        cls_scores = pred[:, 4:]     # [N, nc]

        class_ids = np.argmax(cls_scores, axis=1)
        confidences = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

        keep = confidences >= self.confidence_threshold
        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # Convert xywh(letterboxed) -> xywh in original image coordinates
        boxes_xywh_orig = undo_letterbox_xywh(
            boxes_xywh=boxes_xywh,
            image_shape=img_shape,
            ratio=ratio,
            dw=dw,
            dh=dh,
        )

        for box, cls_id, conf in zip(boxes_xywh_orig, class_ids, confidences):
            cx, cy, w, h = box.tolist()
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    confidence=float(conf),
                    bbox=(float(cx), float(cy), float(w), float(h)),
                )
            )

        return detections
