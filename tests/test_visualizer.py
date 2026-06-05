import numpy as np
import pytest

from object_tracker.types import Detection
from object_tracker.visualization.draw_detections import Visualizer


@pytest.fixture
def blank_image() -> np.ndarray:
    """200 x 300 black BGR image."""
    return np.zeros((200, 300, 3), dtype=np.uint8)


@pytest.fixture
def viz() -> Visualizer:
    return Visualizer(class_names=["cat", "dog", "bird"])


class TestVisualizerSmoke:
    def test_draw_single_detection_modifies_image(self, viz, blank_image):
        det = Detection(class_id=0, confidence=0.85, bbox=(150.0, 100.0, 80.0, 60.0))
        before = blank_image.copy()
        viz.draw_detections(blank_image, [det])
        assert not np.array_equal(blank_image, before), "image should be modified after drawing"

    def test_draw_no_detections_leaves_image_unchanged(self, viz, blank_image):
        before = blank_image.copy()
        viz.draw_detections(blank_image, [])
        assert np.array_equal(blank_image, before)

    def test_label_near_top_edge_does_not_raise(self, viz, blank_image):
        # y coordinate near 0 forces label to be drawn below the box
        det = Detection(class_id=1, confidence=0.5, bbox=(150.0, 5.0, 40.0, 8.0))
        viz.draw_detections(blank_image, [det])  # must not raise

    def test_draw_multiple_detections_does_not_raise(self, viz, blank_image):
        dets = [
            Detection(class_id=0, confidence=0.9, bbox=(50.0, 50.0, 40.0, 30.0)),
            Detection(class_id=2, confidence=0.6, bbox=(220.0, 150.0, 60.0, 50.0)),
        ]
        viz.draw_detections(blank_image, dets)
        assert blank_image.any()

    def test_custom_colour_map_is_accepted(self, blank_image):
        colours = [(255, 0, 0), (0, 255, 0)]
        viz = Visualizer(class_names=["a", "b"], colour_map=colours)
        det = Detection(class_id=1, confidence=0.7, bbox=(100.0, 100.0, 50.0, 50.0))
        viz.draw_detections(blank_image, [det])  # must not raise
