from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
from preprocessing import apply_letterbox_transform
from postprocessing import undo_letterbox_xyxy
from typing import List, Tuple


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h)


class YOLOv10DetectorONNX:
    def __init__(self, onnx_model: str, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

        available = ort.get_available_providers()
        
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_model, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_detections(self, img: np.ndarray) -> List[Detection]:
        processed_img, ratio, dw, dh = self._preprocess(img)
        raw_outputs = self._predict(processed_img)
        return self._postprocess(raw_outputs, img.shape, ratio, dw, dh)

    def _predict(self, processed_img: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: processed_img})

    def _preprocess(self, img: np.ndarray):
        """Returns ndarray [1,3,640,640], ratio, dw, dh"""
        input_size = (640, 640)
        padding_colour = (114, 114, 114)

        img, ratio, (dw, dh) = apply_letterbox_transform(img, input_size,  # 1. Letterbox transformation
                                        padding_colour)
        img = img[:, :, ::-1]                              # 2. BGR to RGB
        img = img.transpose(2, 0, 1)                       # 3. HWC to CHW
        img = img.astype(np.float32) / 255.0               # 4. Normalize to [0, 1]

        img = np.expand_dims(img, axis=0)                  # add batch dim → [1,3,H,W]

        return img, ratio, dw, dh

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        img_shape: Tuple[int, ...],
        ratio: float,
        dw: float,
        dh: float,
    ) -> List[Detection]:
        """
        Hugging Face onnx-community/yolov10m output format:
        output0 -> [B, N, 6] where each row is
        [xmin, ymin, xmax, ymax, score, class_id]
        """
        if not outputs:
            return []

        output0 = outputs[0]
        assert output0.ndim >= 2, f"Unexpected output rank: {output0.ndim}, shape={output0.shape}"
        preds = output0[0]  # [N, 6]

        if preds.shape[1] < 6:
            raise ValueError(f"Expected >=6 columns, got shape {preds.shape}")

        scores = preds[:, 4]
        confidence_mask = scores >= self.confidence_threshold
        preds = preds[confidence_mask]

        if preds.size == 0:
            return []

        boxes_xyxy = preds[:, :4].astype(np.float32)
        class_ids = preds[:, 5].astype(np.int32)
        confidences = preds[:, 4].astype(np.float32)

        boxes_xyxy_orig = undo_letterbox_xyxy(
            boxes_xyxy=boxes_xyxy,
            image_shape=img_shape,
            ratio=ratio,
            dw=dw,
            dh=dh,
        )

        detections = []

        for box, cls_id, conf in zip(boxes_xyxy_orig, class_ids, confidences):
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            detections.append(
                Detection(
                    class_id=int(cls_id),
                    confidence=float(conf),
                    bbox=(float(cx), float(cy), float(w), float(h)),
                )
            )

        return detections
    