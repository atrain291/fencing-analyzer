"""Stage 2 — Per-frame pose estimation using YOLOv8-Pose (CUDA)."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n-pose.pt")  # downloads on first run; swap for larger model
        logger.info("YOLOv8-Pose model loaded")
    return _model


def run_pose_estimation(
    video_path: str,
    video_info: dict,
    bout_id: int,
    db,
) -> list[dict]:
    """
    Run YOLOv8-Pose on every frame of the video.
    Persists Frame records to the database and returns a summary list.
    """
    model = _get_model()
    from app.models.analysis import Frame

    results_summary = []
    frame_idx = 0
    fps = video_info.get("fps", 30)

    for result in model.track(video_path, stream=True, device="cuda"):
        timestamp_ms = int((frame_idx / fps) * 1000)

        fencer_pose = {}
        opponent_pose = {}

        if result.keypoints is not None and len(result.keypoints) > 0:
            kps = result.keypoints.xyn.cpu().numpy()  # normalized [0,1]
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            if len(kps) >= 1:
                fencer_pose = _keypoints_to_dict(kps[0], conf[0] if conf is not None else None)
            if len(kps) >= 2:
                opponent_pose = _keypoints_to_dict(kps[1], conf[1] if conf is not None else None)

        frame = Frame(
            bout_id=bout_id,
            timestamp_ms=timestamp_ms,
            fencer_pose=fencer_pose,
            opponent_pose=opponent_pose if opponent_pose else None,
        )
        db.add(frame)

        results_summary.append({"frame": frame_idx, "timestamp_ms": timestamp_ms})
        frame_idx += 1

        # Commit in batches to avoid huge transactions
        if frame_idx % 300 == 0:
            db.commit()
            logger.debug("Committed %d frames", frame_idx)

    db.commit()
    logger.info("Pose estimation complete: %d frames persisted", frame_idx)
    return results_summary


# COCO keypoint indices (YOLOv8-Pose uses COCO 17-point skeleton)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _keypoints_to_dict(kps: Any, conf: Any) -> dict:
    result = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(kps):
            x, y = float(kps[i][0]), float(kps[i][1])
            c = float(conf[i]) if conf is not None and i < len(conf) else 0.0
            result[name] = {"x": x, "y": y, "z": 0.0, "confidence": c}
    return result
