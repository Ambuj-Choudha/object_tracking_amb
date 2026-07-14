from object_tracker.detectors.base import DetectorBase
from object_tracker.io_transforms.preprocessing import apply_letterbox_transform
from object_tracker.io_transforms.postprocessing import undo_letterbox_xyxy
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from object_tracker.types import Detection


class YOLOv10DetectorONNX(DetectorBase):
    def __init__(self, onnx_model: str, confidence_threshold: float | None = None) -> None:
        self.confidence_threshold = confidence_threshold

        available = ort.get_available_providers()
        
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_model, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_detections(self, img: np.ndarray) -> list[Detection]:
        processed_img, ratio, dw, dh = self._preprocess(img)
        raw_outputs = self._predict(processed_img)
        return self._postprocess(raw_outputs, img.shape, ratio, dw, dh)

    def _predict(self, processed_img: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: processed_img})  # type: ignore[no-any-return]

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
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
        img_shape: tuple[int, ...],
        ratio: float,
        dw: float,
        dh: float,
    ) -> list[Detection]:
        """
        Hugging Face onnx-community/yolov10m output format:
        output0 -> [B, N, 6] where each row is
        [xmin, ymin, xmax, ymax, score, class_id]
        ref: https://github.com/Abdurrahheem/yolov10/commit/aad320dd80b56694e590c950b25060a134966496
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
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        return detections
    