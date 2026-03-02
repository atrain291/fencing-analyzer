"""Stage 2 — Per-frame pose estimation using YOLOv8-Pose (CUDA)."""
import logging
import subprocess
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n-pose.pt")  # downloads on first run; swap for larger model
        logger.info("YOLOv8-Pose model loaded")
    return _model


def _bbox_iou_center(det_box, roi_box):
    """Check if detection center falls within ROI bounding box.
    det_box: [x1, y1, x2, y2] in normalized coords (from YOLO)
    roi_box: dict {x1, y1, x2, y2} in normalized coords (from user)
    Returns True if detection center is inside ROI.
    """
    cx = (det_box[0] + det_box[2]) / 2
    cy = (det_box[1] + det_box[3]) / 2
    return (roi_box['x1'] <= cx <= roi_box['x2'] and
            roi_box['y1'] <= cy <= roi_box['y2'])


def _assign_detections(result, fencer_bbox, opponent_bbox):
    """Assign YOLO detections to fencer/opponent roles using ROI boxes.
    Falls back to kps[0]=fencer, kps[1]=opponent if no ROI specified.
    Returns (fencer_kps, fencer_conf, opponent_kps, opponent_conf) or Nones.
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None, None, None, None

    kps_all = result.keypoints.xyn.cpu().numpy()
    conf_all = (result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None else None)

    # If no ROI specified, use index order (existing behavior)
    if fencer_bbox is None and opponent_bbox is None:
        fencer_kps = kps_all[0] if len(kps_all) >= 1 else None
        fencer_conf = conf_all[0] if conf_all is not None and len(conf_all) >= 1 else None
        opp_kps = kps_all[1] if len(kps_all) >= 2 else None
        opp_conf = conf_all[1] if conf_all is not None and len(conf_all) >= 2 else None
        return fencer_kps, fencer_conf, opp_kps, opp_conf

    # Get normalized bounding boxes for each detection
    fencer_kps = fencer_conf = opp_kps = opp_conf = None

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_norm = result.boxes.xyxyn.cpu().numpy()  # normalized [x1,y1,x2,y2]

        for i, det_box in enumerate(boxes_norm):
            if i >= len(kps_all):
                break
            kps = kps_all[i]
            conf = conf_all[i] if conf_all is not None else None

            if fencer_bbox is not None and fencer_kps is None:
                if _bbox_iou_center(det_box, fencer_bbox):
                    fencer_kps, fencer_conf = kps, conf

            if opponent_bbox is not None and opp_kps is None:
                if _bbox_iou_center(det_box, opponent_bbox):
                    opp_kps, opp_conf = kps, conf

    # Fallback: if ROI specified but no match found, return nothing for that role
    return fencer_kps, fencer_conf, opp_kps, opp_conf


def run_pose_estimation(video_path, video_info, bout_id, db, progress_callback=None,
                        fencer_bbox=None, opponent_bbox=None):
    """
    Run YOLOv8-Pose on every frame of the video.
    Persists Frame records to the database and returns a summary list.
    """
    model = _get_model()
    from app.models.analysis import Frame

    results_summary = []
    frame_idx = 0
    fps = video_info.get("fps", 30)
    duration = video_info.get("duration", 0)
    total_frames_hint = int(fps * duration) if duration else 0

    width  = video_info.get("width",  1920)
    height = video_info.get("height", 1080)
    codec  = video_info.get("codec",  "")

    # Map codec name to NVDEC decoder
    _NVDEC = {
        "hevc": "hevc_cuvid",
        "h264": "h264_cuvid",
        "vp9":  "vp9_cuvid",
        "av1":  "av1_cuvid",
    }
    nvdec_decoder = _NVDEC.get(codec)

    ffmpeg_cmd = ["ffmpeg", "-v", "error"]
    if nvdec_decoder:
        ffmpeg_cmd += ["-hwaccel", "cuda", "-c:v", nvdec_decoder]
    ffmpeg_cmd += [
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    frame_bytes = width * height * 3
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            results = model.track(frame, persist=True, device="cuda", verbose=False)
            result = results[0]

            timestamp_ms = int((frame_idx / fps) * 1000)

            fencer_pose = {}
            opponent_pose = {}

            fencer_kps, fencer_conf, opp_kps, opp_conf = _assign_detections(
                result, fencer_bbox, opponent_bbox
            )
            if fencer_kps is not None:
                fencer_pose = _keypoints_to_dict(fencer_kps, fencer_conf)
            if opp_kps is not None:
                opponent_pose = _keypoints_to_dict(opp_kps, opp_conf)

            db_frame = Frame(
                bout_id=bout_id,
                timestamp_ms=timestamp_ms,
                fencer_pose=fencer_pose,
                opponent_pose=opponent_pose if opponent_pose else None,
            )
            db.add(db_frame)

            results_summary.append({"frame": frame_idx, "timestamp_ms": timestamp_ms})
            frame_idx += 1

            if progress_callback and frame_idx % 100 == 0:
                progress_callback(frame_idx, total_frames_hint)

            # Commit in batches to avoid huge transactions
            if frame_idx % 300 == 0:
                db.commit()
                logger.debug("Committed %d frames", frame_idx)
    finally:
        proc.stdout.close()
        proc.wait()

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
