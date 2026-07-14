import numpy as np
import pytest

from object_tracker.detectors.yolov10 import YOLOv10DetectorONNX
from object_tracker.types import Detection


@pytest.fixture
def detector():
    """Bare YOLOv10DetectorONNX with no ONNX session — exercises _postprocess only."""
    det = object.__new__(YOLOv10DetectorONNX)
    det.confidence_threshold = 0.5
    return det


def _raw_output(*rows: list) -> list[np.ndarray]:
    """Wrap row lists into the [B, N, 6] array the model returns."""
    return [np.array([rows], dtype=np.float32)]


class TestPostprocess:
    def test_empty_outputs_returns_empty(self, detector):
        assert detector._postprocess([], (480, 640, 3), 1.0, 0.0, 0.0) == []

    def test_all_below_threshold_returns_empty(self, detector):
        outputs = _raw_output(
            [100, 100, 200, 200, 0.3, 0],
            [50, 50, 150, 150, 0.49, 1],
        )
        result = detector._postprocess(outputs, (480, 640, 3), 1.0, 0.0, 0.0)
        assert result == []

    def test_single_detection_coordinate_transform(self, detector):
        # 640x480 original, letterboxed with ratio=1.0, dh=80.
        # Box [100,180,300,380] in letterbox -> [100,100,300,300] original (xyxy)
        outputs = _raw_output([100, 180, 300, 380, 0.9, 2])
        result = detector._postprocess(outputs, (480, 640, 3), 1.0, 0.0, 80.0)

        assert len(result) == 1
        d = result[0]
        assert d.class_id == 2
        assert d.confidence == pytest.approx(0.9)
        x1, y1, x2, y2 = d.bbox
        assert x1 == pytest.approx(100.0)
        assert y1 == pytest.approx(100.0)
        assert x2 == pytest.approx(300.0)
        assert y2 == pytest.approx(300.0)

    def test_confidence_threshold_filters_rows(self, detector):
        outputs = _raw_output(
            [10, 10, 50, 50, 0.8, 0],   # passes
            [60, 60, 100, 100, 0.4, 1],  # filtered
            [120, 120, 200, 200, 0.6, 2], # passes
        )
        result = detector._postprocess(outputs, (480, 640, 3), 1.0, 0.0, 0.0)
        assert len(result) == 2
        assert {d.class_id for d in result} == {0, 2}

    def test_result_items_are_detection_dataclasses(self, detector):
        outputs = _raw_output([50, 50, 150, 150, 0.7, 3])
        result = detector._postprocess(outputs, (480, 640, 3), 1.0, 0.0, 0.0)

        assert len(result) == 1
        d = result[0]
        assert isinstance(d, Detection)
        assert isinstance(d.class_id, int)
        assert isinstance(d.confidence, float)
        assert len(d.bbox) == 4

    def test_degenerate_zero_area_box_preserved(self, detector):
        # A point box (x1==x2, y1==y2) must round-trip with zero area
        outputs = _raw_output([100, 100, 100, 100, 0.9, 0])
        result = detector._postprocess(outputs, (480, 640, 3), 1.0, 0.0, 0.0)

        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x2 - x1 == pytest.approx(0.0)
        assert y2 - y1 == pytest.approx(0.0)

    def test_wrong_column_count_raises_value_error(self, detector):
        bad = [np.zeros((1, 5, 5), dtype=np.float32)]
        with pytest.raises(ValueError, match="Expected >=6 columns"):
            detector._postprocess(bad, (480, 640, 3), 1.0, 0.0, 0.0)
