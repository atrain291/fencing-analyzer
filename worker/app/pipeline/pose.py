"""Stage 2 — Per-frame pose estimation using YOLOv8-Pose (CUDA)."""
import logging
import os
import subprocess
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model = None

# Custom tracker config with increased track_buffer for occlusion resilience
_TRACKER_CFG = os.path.join(os.path.dirname(__file__), "botsort_fencing.yaml")


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolo11x-pose.pt")  # YOLO11 extra-large: 69.4M params, best accuracy
        logger.info("YOLO11x-Pose model loaded")
    return _model


def _center_in_bbox(cx: float, cy: float, bbox: dict) -> bool:
    """Check if point (cx, cy) falls inside a normalized bbox {x1,y1,x2,y2}."""
    return bbox['x1'] <= cx <= bbox['x2'] and bbox['y1'] <= cy <= bbox['y2']


def _try_lock_id(result, bbox: dict, exclude_id: int | None) -> int | None:
    """
    Try to lock a tracker ID from the current frame using an ROI bbox.
    Returns the tracker ID of the first detection whose center falls in bbox,
    excluding exclude_id (so fencer and opponent don't get the same ID).
    Returns None if no match found or tracking IDs unavailable.
    """
    if result.boxes is None or result.boxes.id is None:
        return None
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
    for i, tid in enumerate(track_ids):
        if exclude_id is not None and tid == exclude_id:
            continue
        if i >= len(boxes_xyxyn):
            break
        box = boxes_xyxyn[i]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if _center_in_bbox(cx, cy, bbox):
            return int(tid)
    return None


def _best_other_id(result, exclude_id: int) -> int | None:
    """
    Return the tracker ID of the highest-confidence detection that is NOT exclude_id.
    Used to auto-assign the opponent when no opponent bbox was provided.
    Returns None if no other detection exists or tracking IDs unavailable.
    """
    if result.boxes is None or result.boxes.id is None:
        return None
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    best_tid = None
    best_conf = -1.0
    for i, tid in enumerate(track_ids):
        if tid == exclude_id:
            continue
        c = float(confs[i]) if i < len(confs) else 0.0
        if c > best_conf:
            best_conf = c
            best_tid = int(tid)
    return best_tid


def _get_kps_by_id(result, track_id: int):
    """
    Return (kps_array, conf_array) for the detection with the given tracker ID.
    Returns (None, None) if the ID is not present in this frame.
    """
    if result.boxes is None or result.boxes.id is None:
        return None, None
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    kps_all = result.keypoints.xyn.cpu().numpy()
    conf_all = (result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None else None)
    for i, tid in enumerate(track_ids):
        if tid == track_id:
            kps = kps_all[i] if i < len(kps_all) else None
            conf = conf_all[i] if conf_all is not None and i < len(conf_all) else None
            return kps, conf
    return None, None


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
    total_frames_hint = video_info.get("total_frames", 0)
    if not total_frames_hint:
        duration = video_info.get("duration_s", 0) or video_info.get("duration", 0)
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

    # Tracker ID locking state (persist across frames)
    fencer_track_id: int | None = None
    opponent_track_id: int | None = None
    roi_mode = fencer_bbox is not None or opponent_bbox is not None

    # Confirmation-based re-lock: when the locked ID goes missing, we look for
    # a new ID in the ROI.  Only switch if the same new ID appears in the ROI
    # for RELOCK_CONFIRM consecutive frames (filters transient occluders).
    RELOCK_MIN_WAIT = 15       # don't even start looking for a new ID until this many missing frames
    RELOCK_CONFIRM = 10        # new candidate must be in ROI for this many consecutive frames
    fencer_missing_count = 0
    fencer_candidate_id: int | None = None
    fencer_candidate_count = 0
    opponent_missing_count = 0
    opponent_candidate_id: int | None = None
    opponent_candidate_count = 0

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            results = model.track(frame, persist=True, device="cuda", verbose=False,
                                  tracker=_TRACKER_CFG)
            result = results[0]

            timestamp_ms = int((frame_idx / fps) * 1000)

            fencer_pose = {}
            opponent_pose = {}

            if result.keypoints is not None and len(result.keypoints) > 0:
                if roi_mode:
                    # --- Fencer tracking ---
                    if fencer_track_id is not None:
                        kps, conf = _get_kps_by_id(result, fencer_track_id)
                        if kps is not None:
                            # Locked ID still visible — normal path
                            fencer_pose = _keypoints_to_dict(kps, conf)
                            fencer_missing_count = 0
                            fencer_candidate_id = None
                            fencer_candidate_count = 0
                        else:
                            # Locked ID missing (occluded or tracker dropped it)
                            fencer_missing_count += 1
                            # After a grace period, look for a new ID in the ROI
                            if fencer_missing_count >= RELOCK_MIN_WAIT and fencer_bbox is not None:
                                roi_id = _try_lock_id(result, fencer_bbox, opponent_track_id)
                                if roi_id is not None:
                                    # Require the same new ID for RELOCK_CONFIRM frames
                                    if roi_id == fencer_candidate_id:
                                        fencer_candidate_count += 1
                                    else:
                                        fencer_candidate_id = roi_id
                                        fencer_candidate_count = 1
                                    if fencer_candidate_count >= RELOCK_CONFIRM:
                                        logger.info(
                                            "Frame %d: re-locked fencer ID %d -> %d "
                                            "(confirmed %d frames in ROI after %d missing)",
                                            frame_idx, fencer_track_id, roi_id,
                                            RELOCK_CONFIRM, fencer_missing_count)
                                        fencer_track_id = roi_id
                                        fencer_missing_count = 0
                                        fencer_candidate_id = None
                                        fencer_candidate_count = 0
                                        kps, conf = _get_kps_by_id(result, fencer_track_id)
                                        if kps is not None:
                                            fencer_pose = _keypoints_to_dict(kps, conf)
                                else:
                                    # Nobody in ROI — occluder still blocking, keep waiting
                                    fencer_candidate_id = None
                                    fencer_candidate_count = 0
                    elif fencer_bbox is not None:
                        # Initial lock (first frames of video)
                        fencer_track_id = _try_lock_id(result, fencer_bbox, opponent_track_id)
                        if fencer_track_id is not None:
                            logger.info("Frame %d: locked fencer tracker ID %d",
                                        frame_idx, fencer_track_id)
                            kps, conf = _get_kps_by_id(result, fencer_track_id)
                            if kps is not None:
                                fencer_pose = _keypoints_to_dict(kps, conf)

                    # --- Opponent tracking ---
                    if opponent_track_id is not None:
                        kps, conf = _get_kps_by_id(result, opponent_track_id)
                        if kps is not None:
                            opponent_pose = _keypoints_to_dict(kps, conf)
                            opponent_missing_count = 0
                            opponent_candidate_id = None
                            opponent_candidate_count = 0
                        else:
                            opponent_missing_count += 1
                            if opponent_missing_count >= RELOCK_MIN_WAIT and opponent_bbox is not None:
                                roi_id = _try_lock_id(result, opponent_bbox, fencer_track_id)
                                if roi_id is not None:
                                    if roi_id == opponent_candidate_id:
                                        opponent_candidate_count += 1
                                    else:
                                        opponent_candidate_id = roi_id
                                        opponent_candidate_count = 1
                                    if opponent_candidate_count >= RELOCK_CONFIRM:
                                        logger.info(
                                            "Frame %d: re-locked opponent ID %d -> %d "
                                            "(confirmed %d frames in ROI after %d missing)",
                                            frame_idx, opponent_track_id, roi_id,
                                            RELOCK_CONFIRM, opponent_missing_count)
                                        opponent_track_id = roi_id
                                        opponent_missing_count = 0
                                        opponent_candidate_id = None
                                        opponent_candidate_count = 0
                                        kps, conf = _get_kps_by_id(result, opponent_track_id)
                                        if kps is not None:
                                            opponent_pose = _keypoints_to_dict(kps, conf)
                                else:
                                    opponent_candidate_id = None
                                    opponent_candidate_count = 0
                    elif opponent_bbox is not None:
                        opponent_track_id = _try_lock_id(result, opponent_bbox, fencer_track_id)
                        if opponent_track_id is not None:
                            logger.info("Frame %d: locked opponent tracker ID %d",
                                        frame_idx, opponent_track_id)
                            kps, conf = _get_kps_by_id(result, opponent_track_id)
                            if kps is not None:
                                opponent_pose = _keypoints_to_dict(kps, conf)
                    elif fencer_track_id is not None:
                        # Auto-assign opponent (no bbox drawn): highest-confidence other
                        opponent_track_id = _best_other_id(result, fencer_track_id)
                        if opponent_track_id is not None:
                            logger.info("Frame %d: auto-assigned opponent tracker ID %d",
                                        frame_idx, opponent_track_id)
                            kps, conf = _get_kps_by_id(result, opponent_track_id)
                            if kps is not None:
                                opponent_pose = _keypoints_to_dict(kps, conf)

                else:
                    # No ROI: fall back to index order (kps[0]=fencer, kps[1]=opponent)
                    kps_all = result.keypoints.xyn.cpu().numpy()
                    conf_all = (result.keypoints.conf.cpu().numpy()
                                if result.keypoints.conf is not None else None)
                    if len(kps_all) >= 1:
                        fencer_pose = _keypoints_to_dict(
                            kps_all[0], conf_all[0] if conf_all is not None else None)
                    if len(kps_all) >= 2:
                        opponent_pose = _keypoints_to_dict(
                            kps_all[1], conf_all[1] if conf_all is not None else None)

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
