from detector_onnx import Detection
from ultralytics import YOLOv10


class YOLOv10Detector:
    """
    Inference using Official YOLOv10:
    - preprocessing (letterbox, normalization, etc.) is handled by Ultralytics
    - NMS/postprocessing is handled by Ultralytics
    - this class converts Results -> Detection dataclass
    """

    def __init__(self, weights, confidence_threshold=0.5, iou_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLOv10(weights)

    def get_detections(self, img):
        """
        img: OpenCV BGR ndarray (H, W, C)
        returns: list[Detection]
        """
        results = self.model.predict(
            source=img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        boxes_xywh_orig = boxes.xywh.detach().cpu()
        confidences = boxes.conf.detach().cpu()
        class_ids = boxes.cls.detach().cpu()

        detections = []
        for box, cls_id, conf in zip(boxes_xywh_orig, class_ids, confidences):
            cx, cy, w, h = box.tolist()
            detections.append(
                Detection(
                    class_id=int(cls_id.item()),
                    confidence=float(conf.item()),
                    bbox=(float(cx), float(cy), float(w), float(h)),
                )
            )

        return detections
