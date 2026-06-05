import numpy as np
import pytest

from object_tracker.io_transforms.postprocessing import undo_letterbox_xyxy


def _boxes(arr: list) -> np.ndarray:
    return np.array(arr, dtype=np.float32)


class TestUndoLetterboxXyxy:
    def test_identity_no_scale_no_padding(self):
        inp = _boxes([[10, 20, 100, 200]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=1.0, dw=0.0, dh=0.0)
        np.testing.assert_allclose(result, [[10, 20, 100, 200]])

    def test_scale_only_doubles_coordinates(self):
        # ratio=0.5 means the original image was shrunk by half; inverse doubles coords
        inp = _boxes([[100, 50, 200, 150]])
        result = undo_letterbox_xyxy(inp, (720, 1280), ratio=0.5, dw=0.0, dh=0.0)
        np.testing.assert_allclose(result, [[200, 100, 400, 300]])

    def test_vertical_padding_shifts_y(self):
        # dh=80 means 80 px were added top/bottom; subtracting it moves boxes up
        inp = _boxes([[50, 130, 200, 430]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=1.0, dw=0.0, dh=80.0)
        np.testing.assert_allclose(result, [[50, 50, 200, 350]])

    def test_combined_ratio_and_horizontal_padding(self):
        # ratio=0.75, dw=40: x1=(160-40)/0.75=160, x2=(280-40)/0.75=320
        inp = _boxes([[160, 0, 280, 120]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=0.75, dw=40.0, dh=0.0)
        np.testing.assert_allclose(result, [[160, 0, 320, 160]], rtol=1e-5)

    def test_clips_coords_to_image_upper_bound(self):
        # Box extends beyond 640x480; x2,y2 must be clipped to 639, 479
        inp = _boxes([[0, 0, 700, 700]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=1.0, dw=0.0, dh=0.0)
        np.testing.assert_allclose(result, [[0, 0, 639, 479]])

    def test_clips_negative_coords_to_zero(self):
        # Large dw/dh push coordinates negative; must be clipped to 0
        inp = _boxes([[0, 0, 100, 100]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=1.0, dw=200.0, dh=200.0)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.0)

    def test_multiple_boxes_transformed_independently(self):
        inp = _boxes([[10, 20, 100, 200], [50, 60, 150, 260]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=1.0, dw=0.0, dh=0.0)
        np.testing.assert_allclose(result, [[10, 20, 100, 200], [50, 60, 150, 260]])

    def test_output_shape_is_n_by_4(self):
        inp = _boxes([[10, 20, 100, 200], [50, 60, 150, 260], [0, 0, 50, 50]])
        result = undo_letterbox_xyxy(inp, (480, 640), ratio=1.0, dw=0.0, dh=0.0)
        assert result.shape == (3, 4)
