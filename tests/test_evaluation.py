import json
from types import SimpleNamespace

import pytest

from tools.pose_eval.manifest import load_manifest
from tools.pose_eval.metrics import evaluate
from tools.pose_eval.report import create_report
from tools.pose_eval.runner import run_clip
from test_video import vfr_video


def test_manifest_resolves_video_relative_to_itself_and_rejects_duplicate_ids(tmp_path):
    video = tmp_path / "video.mkv"
    video.touch()
    manifest = tmp_path / "clips.json"
    clip = {"id": "lunge", "video": "video.mkv", "start_s": 1, "end_s": 2}
    manifest.write_text(json.dumps({"clips": [clip]}))
    assert load_manifest(manifest)[0].video == video
    manifest.write_text(json.dumps({"clips": [clip, clip]}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(manifest)
    clip["end_s"] = 0.5
    manifest.write_text(json.dumps({"clips": [clip]}))
    with pytest.raises(ValueError, match="end_s"):
        load_manifest(manifest)


def test_metrics_count_missing_visible_joints_and_only_score_manual_associations():
    records = [{"frame_index": 0, "width": 100, "height": 100, "detections": [
        {"track_id": 7, "keypoints": {"left_wrist": {"x": 0.13, "y": 0.24, "confidence": 0.9}}}
    ]}]
    annotations = [{"frame_index": 0, "people": [{"person_id": "a", "track_ids": {"yolov8n": 7},
        "keypoints": {"left_wrist": {"x": 0.1, "y": 0.2, "visible": True},
                      "right_wrist": {"x": 0.8, "y": 0.5, "visible": True},
                      "left_elbow": {"x": 0, "y": 0, "visible": False}}}]}]
    metrics = evaluate(records, annotations, "yolov8n")
    assert metrics["mean_visible_joint_error_px"] == pytest.approx(5)
    assert metrics["visible_joint_recall"] == 0.5
    assert metrics["annotated_visible_joints"] == 2
    assert metrics["matched_visible_joints"] == 1
    assert evaluate(records, None, "yolov8n")["mean_visible_joint_error_px"] is None
    assert evaluate(records, None, "yolov8n")["labeled_id_switches"] is None


def test_annotation_absent_frame_is_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="frame"):
        evaluate([], [{"frame_index": 3, "people": []}], "yolov8n")


class DeterministicAdapter:
    """External model boundary for testing real decode/export, not model accuracy."""
    metadata = {"model_id": "yolov8n", "checkpoint": "synthetic-test", "synthetic": True}

    def reset(self):
        pass

    def predict(self, image):
        return [{"box": [0.1, 0.1, 0.8, 0.9], "confidence": 0.9, "track_id": 1,
                 "keypoints": {"left_wrist": {"x": 0.2, "y": 0.3, "confidence": 0.9}}}]


def test_real_decoding_export_report_and_overwrite_protection(vfr_video, tmp_path):
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"clips": [{"id": "vfr", "video": str(vfr_video), "start_s": 0, "end_s": 0.2}]}))
    clip = load_manifest(manifest)[0]
    output = tmp_path / "results"
    result_path = run_clip(clip, DeterministicAdapter(), output, warmup=1)
    result = json.loads(result_path.read_text())
    assert [f["timestamp_ms"] for f in result["frames"]] == [0, 40, 120, 160]
    assert result["summary"]["frame_count"] == 4
    assert result["provenance"]["synthetic"] is True
    assert result["source"]["sha256"]
    with pytest.raises(FileExistsError):
        run_clip(clip, DeterministicAdapter(), output)
    report = create_report(result_path.parent)
    assert report.exists()
    assert json.loads(result_path.read_text())["metrics"]["mean_visible_joint_error_px"] is None


def test_report_escapes_untrusted_source_names_and_rejects_incompatible_runs(vfr_video, tmp_path):
    clip = SimpleNamespace(id="vfr", video=vfr_video, start_s=0, end_s=0.2, annotations=None)
    path = run_clip(clip, DeterministicAdapter(), tmp_path / "results", warmup=0)
    data = json.loads(path.read_text())
    data["source"]["name"] = '</script><script>alert("unsafe")</script>'
    path.write_text(json.dumps(data))
    report = create_report(path.parent)
    assert '</script><script>alert("unsafe")' not in report.read_text(encoding="utf-8")
    second = json.loads(path.read_text())
    second["provenance"]["model_id"] = "yolo26m"
    second["source"]["sha256"] = "different"
    (path.parent / "yolo26m.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="source"):
        create_report(path.parent)


@pytest.mark.parametrize("first_hash,second_hash", [(None, "labeled"), ("labels-a", "labels-b")])
def test_report_rejects_metrics_measured_with_different_annotations(vfr_video, tmp_path, first_hash, second_hash):
    clip = SimpleNamespace(id="vfr", video=vfr_video, start_s=0, end_s=0.2, annotations=None)
    path = run_clip(clip, DeterministicAdapter(), tmp_path / "results", warmup=0)
    data = json.loads(path.read_text())
    data["annotation_sha256"] = first_hash
    path.write_text(json.dumps(data))
    data["provenance"]["model_id"] = "yolo26m"
    data["annotation_sha256"] = second_hash
    (path.parent / "yolo26m.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="annotation"):
        create_report(path.parent)
