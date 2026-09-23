import sys
from types import SimpleNamespace

import numpy as np

from app.pipeline import pose


class Tensor:
    def __init__(self, value):
        self.value = np.array(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Keypoints:
    def __init__(self, values):
        self.xyn = Tensor([[[value, value]] * 17 for value in values])
        self.conf = Tensor([[0.9] * 17 for _ in values])

    def __len__(self):
        return len(self.xyn.value)


def test_worker_persists_real_timestamps_and_stable_participants(monkeypatch):
    tracker = SimpleNamespace(reset_count=0)

    def reset():
        tracker.reset_count += 1

    tracker.reset = reset
    outputs = iter([
        SimpleNamespace(keypoints=Keypoints([0.2, 0.8]), boxes=SimpleNamespace(
            id=Tensor([10, 20]), xyxy=Tensor([[0, 0, 20, 40], [80, 0, 100, 40]]))),
        SimpleNamespace(keypoints=Keypoints([0.7]), boxes=SimpleNamespace(
            id=Tensor([20]), xyxy=Tensor([[70, 0, 90, 40]]))),
    ])
    model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]),
                            track=lambda *args, **kwargs: [next(outputs)])
    monkeypatch.setattr(pose, "_get_model", lambda: model)
    monkeypatch.setattr(pose, "iter_video_frames", lambda path: (frame for frame in [
        SimpleNamespace(index=0, timestamp_ms=0.0, image=np.zeros((2, 2, 3))),
        SimpleNamespace(index=1, timestamp_ms=83.0, image=np.zeros((2, 2, 3))),
    ]), raising=False)
    monkeypatch.setitem(sys.modules, "app.models.analysis", SimpleNamespace(Frame=SimpleNamespace))
    rows = []
    db = SimpleNamespace(add=rows.append, commit=lambda: None)
    summaries = pose.run_pose_estimation("fixture.mkv", {"fps": 30, "total_frames": 2}, 1, db)
    assert [row.timestamp_ms for row in rows] == [0, 83]
    assert rows[1].fencer_pose == {}
    assert rows[1].opponent_pose["left_wrist"]["x"] == 0.7
    assert summaries[1]["timestamp_ms"] == 83
    assert tracker.reset_count == 1
