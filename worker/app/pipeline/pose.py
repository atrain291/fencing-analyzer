"""Stage 2 — Per-frame pose estimation using YOLOv8-Pose (CUDA)."""
import logging
from contextlib import closing
from typing import Any

from .participants import ParticipantMap
from .video import iter_video_frames

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n-pose.pt")  # downloads on first run; swap for larger model
        logger.info("YOLOv8-Pose model loaded")
    return _model


def run_pose_estimation(video_path, video_info, bout_id, db, progress_callback=None):
    """
    Run YOLOv8-Pose on every frame of the video.
    Persists Frame records to the database and returns a summary list.
    """
    model = _get_model()
    from app.models.analysis import Frame

    results_summary = []
    total_frames_hint = video_info.get("total_frames", 0)
    participants = ParticipantMap()
    # Cached weights are reusable; tracker history belongs to exactly one bout.
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()

    with closing(iter_video_frames(video_path)) as decoded:
        for frame in decoded:
            results = model.track(frame.image, persist=True, device="cuda", verbose=False)
            result = results[0]
            timestamp_ms = round(frame.timestamp_ms)

            fencer_pose = {}
            opponent_pose = {}

            if result.keypoints is not None and len(result.keypoints) > 0:
                kps = result.keypoints.xyn.cpu().numpy()  # normalized [0,1]
                conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    fencer_index, opponent_index = participants.select(
                        boxes.id.cpu().numpy(), boxes.xyxy.cpu().numpy())
                    if fencer_index is not None:
                        fencer_pose = _keypoints_to_dict(kps[fencer_index], conf[fencer_index] if conf is not None else None)
                    if opponent_index is not None:
                        opponent_pose = _keypoints_to_dict(kps[opponent_index], conf[opponent_index] if conf is not None else None)

            db_frame = Frame(
                bout_id=bout_id,
                timestamp_ms=timestamp_ms,
                fencer_pose=fencer_pose,
                opponent_pose=opponent_pose if opponent_pose else None,
            )
            db.add(db_frame)

            results_summary.append({"frame": frame.index, "timestamp_ms": timestamp_ms})
            frame_count = len(results_summary)

            if progress_callback and frame_count % 100 == 0:
                progress_callback(frame_count, total_frames_hint)

            # Commit in batches to avoid huge transactions
            if frame_count % 300 == 0:
                db.commit()
                logger.debug("Committed %d frames", frame_count)

    db.commit()
    logger.info("Pose estimation complete: %d frames persisted", len(results_summary))
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
