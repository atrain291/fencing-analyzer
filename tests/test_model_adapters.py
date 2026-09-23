from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tools.pose_eval.adapters import ModelAdapter, normalize_predictions


def test_normalization_keeps_joint_confidence_and_scales_covariance_in_both_axes():
    points = np.tile([100.0, 50.0], (1, 17, 1))
    confidence = np.full((1, 17), 0.8)
    points[0, 1] = [np.nan, 50]
    points[0, 2] = [250, 50]
    confidence[0, 3] = 0
    covariance = np.tile([[16.0, 6.0], [6.0, 9.0]], (1, 17, 1, 1))
    result = normalize_predictions([[20, 10, 180, 90]], [0.9], points, confidence, 200, 100, covariance)
    assert result[0]["box"] == [0.1, 0.1, 0.9, 0.9]
    nose = result[0]["keypoints"]["nose"]
    assert (nose["x"], nose["y"], nose["confidence"]) == (0.5, 0.5, 0.8)
    np.testing.assert_allclose(nose["covariance"], [[0.0004, 0.0003], [0.0003, 0.0009]])
    assert not {"left_eye", "right_eye", "left_ear"} & result[0]["keypoints"].keys()
    assert result[0]["track_id"] is None
    with pytest.raises(ValueError, match="counts"):
        normalize_predictions([], [0.9], points, confidence, 200, 100)


def test_rfdetr_boundary_receives_rgb_and_uses_separate_person_and_joint_confidence():
    image = np.full((100, 200, 3), [10, 20, 30], dtype=np.uint8)

    def predict(rgb, *, threshold):
        assert rgb[0, 0].tolist() == [30, 20, 10]
        assert rgb.flags.c_contiguous
        return SimpleNamespace(xy=np.tile([100.0, 25.0], (1, 17, 1)),
                               keypoint_confidence=np.full((1, 17), 0.6),
                               detection_confidence=np.array([0.95]),
                               data={"xyxy": np.array([[20, 10, 180, 90]])})

    adapter = ModelAdapter.__new__(ModelAdapter)
    adapter.model_id, adapter.threshold = "rfdetr-keypoint", 0.25
    adapter.torch = SimpleNamespace(inference_mode=nullcontext)
    adapter.model = SimpleNamespace(predict=predict)
    detection = adapter.predict(image)[0]
    assert detection["confidence"] == 0.95
    assert detection["keypoints"]["left_wrist"] == {"x": 0.5, "y": 0.25, "confidence": 0.6}


def test_real_bytetrack_preserves_ids_after_detection_reordering_and_missing_person():
    sv = pytest.importorskip("supervision", reason="optional model environment")
    adapter = ModelAdapter.__new__(ModelAdapter)
    adapter.sv, adapter.threshold = sv, 0.25
    adapter.reset()

    def person(x):
        return {"box": [x, 0.1, x + 0.2, 0.9], "confidence": 0.9,
                "track_id": None, "keypoints": {"nose": {"x": x + 0.1, "y": 0.2, "confidence": 0.9}}}

    first = adapter.track([person(0.1), person(0.7)], 640, 360)
    left, right = (p["track_id"] for p in first)
    assert left is not None and right is not None and left != right
    reordered = adapter.track([person(0.7), person(0.1)], 640, 360)
    assert [p["track_id"] for p in reordered] == [right, left]
    assert reordered[0]["keypoints"]["nose"]["x"] == pytest.approx(0.8)
    assert adapter.track([person(0.7)], 640, 360)[0]["track_id"] == right
    assert adapter.track([], 640, 360) == []
