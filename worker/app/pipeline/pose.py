"""Stage 2 — Per-frame pose estimation using RTMPose WholeBody (rtmlib).

Uses a 3-thread pipeline to overlap FFmpeg decoding, pose inference, and
DB writes for better GPU utilization:

  Decode thread -> [decode_queue] -> Main thread (RTMPose + matching) -> [write_queue] -> Write thread (DB)

Person matching is proximity-based: each frame, detections are matched to
the fencer/opponent by closest center distance to last known position.
"""
import logging
import os
import subprocess
import threading
from queue import Queue
from typing import Any

from math import exp as _math_exp

import numpy as np

from app.pipeline.strip import ankles_on_strip

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Load RTMPose WholeBody model via rtmlib.

    Uses 'performance' mode (YOLOX-m detector + rtmw-dw-x-l pose) with
    ONNX Runtime GPU backend for best accuracy.
    """
    global _model
    if _model is None:
        from rtmlib import Wholebody
        _model = Wholebody(
            mode='performance',
            backend='onnxruntime',
            device='cuda',
        )
        logger.info("RTMPose WholeBody loaded (mode=performance, backend=onnxruntime, device=cuda)")
    return _model


def _bbox_from_keypoints(kps: np.ndarray, scores: np.ndarray,
                         width: int, height: int,
                         conf_thr: float = 0.3) -> dict | None:
    """Compute a normalized bounding box from keypoint positions.

    Uses only keypoints with normalized confidence >= conf_thr. Falls back to
    conf_thr=0.1 if no keypoints pass the primary threshold.
    Returns dict {x1, y1, x2, y2} in 0-1 range, or None if unusable.
    """
    norm_scores = np.array([_normalize_score(float(s)) for s in scores])
    valid = norm_scores >= conf_thr
    if not valid.any():
        valid = norm_scores >= 0.1
    if not valid.any():
        return None
    pts = kps[valid]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    # Normalize to 0-1
    return {
        "x1": float(x1 / width),
        "y1": float(y1 / height),
        "x2": float(x2 / width),
        "y2": float(y2 / height),
    }


def _center_of_bbox(bbox: dict) -> tuple[float, float]:
    """Return the normalized center (cx, cy) of a bbox dict."""
    return ((bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2)


def _center_in_bbox(cx: float, cy: float, bbox: dict) -> bool:
    """Check if point (cx, cy) falls inside a normalized bbox {x1,y1,x2,y2}."""
    return bbox['x1'] <= cx <= bbox['x2'] and bbox['y1'] <= cy <= bbox['y2']


def _detection_on_strip(kps: np.ndarray, scores: np.ndarray,
                        width: int, height: int,
                        strip: dict | None) -> bool:
    """Check if a detection's ankles are on the strip."""
    if not strip or not strip.get("polygon"):
        return True
    kps_dict = keypoints_to_dict(kps, scores, width, height)
    return ankles_on_strip(kps_dict, strip)


def _dist_sq(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """Squared Euclidean distance between two points."""
    return (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2


# Maximum normalized distance (squared) to accept a detection as the same
# person across consecutive frames.  0.15^2 = 0.0225 (~15% of frame diagonal).
_MAX_TRACK_DIST_SQ = 0.0225


def _find_in_roi(detections: list[dict], roi_bbox: dict,
                 exclude_indices: set[int],
                 strip: dict | None, all_kps: np.ndarray,
                 all_scores: np.ndarray, width: int, height: int) -> int | None:
    """Find the detection index whose center is inside the ROI bbox.

    Filters by strip and exclusion set. Returns index or None.
    """
    for i, det in enumerate(detections):
        if i in exclude_indices:
            continue
        cx, cy = det["cx"], det["cy"]
        if _center_in_bbox(cx, cy, roi_bbox):
            if _detection_on_strip(all_kps[i], all_scores[i], width, height, strip):
                return i
    return None


def _find_closest(detections: list[dict], target_cx: float, target_cy: float,
                  exclude_indices: set[int],
                  strip: dict | None, all_kps: np.ndarray,
                  all_scores: np.ndarray, width: int, height: int,
                  max_dist_sq: float | None = None) -> int | None:
    """Find the detection index closest to (target_cx, target_cy).

    Filters by strip, exclusion set, and optional max distance.
    Returns index or None.
    """
    best_idx = None
    best_dist = float('inf')
    for i, det in enumerate(detections):
        if i in exclude_indices:
            continue
        if not _detection_on_strip(all_kps[i], all_scores[i], width, height, strip):
            continue
        d = _dist_sq(det["cx"], det["cy"], target_cx, target_cy)
        if max_dist_sq is not None and d > max_dist_sq:
            continue
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _find_best_confidence(detections: list[dict], exclude_indices: set[int],
                          strip: dict | None, all_kps: np.ndarray,
                          all_scores: np.ndarray,
                          width: int, height: int) -> int | None:
    """Find the detection index with highest overall keypoint confidence.

    Filters by strip and exclusion set. Returns index or None.
    """
    best_idx = None
    best_conf = -1.0
    for i, det in enumerate(detections):
        if i in exclude_indices:
            continue
        if not _detection_on_strip(all_kps[i], all_scores[i], width, height, strip):
            continue
        # Mean normalized confidence of body keypoints (first 17)
        body_scores = np.array([_normalize_score(float(s)) for s in all_scores[i][:17]])
        mean_conf = float(body_scores[body_scores > 0.1].mean()) if (body_scores > 0.1).any() else 0.0
        if mean_conf > best_conf:
            best_conf = mean_conf
            best_idx = i
    return best_idx


def _frame_decoder(proc, width: int, height: int, frame_bytes: int,
                    decode_queue: Queue) -> None:
    """Decode thread: reads raw BGR frames from FFmpeg stdout and enqueues them.

    Runs in a background thread so decoding overlaps with pose inference.
    Puts (frame_idx, numpy_array) tuples into *decode_queue*.
    Sends a None sentinel when the pipe is exhausted or an error occurs.
    """
    frame_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            decode_queue.put((frame_idx, frame))
            frame_idx += 1
    except Exception:
        logger.exception("Decode thread error")
    finally:
        decode_queue.put(None)  # sentinel


def _db_writer(write_queue: Queue, bout_id: int, error_holder: list) -> None:
    """Write thread: creates Frame ORM objects from dicts and commits in batches.

    Runs in a background thread with its own SQLAlchemy session so DB I/O
    overlaps with pose inference on the main thread.  Receives plain dicts
    (not ORM objects) to avoid cross-thread session contamination.
    Stops when it receives a None sentinel.
    """
    from app.db import get_db_session
    from app.models.analysis import Frame

    with get_db_session() as db:
        count = 0
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                db_frame = Frame(**item)
                db.add(db_frame)
                count += 1
                if count % 300 == 0:
                    db.commit()
                    logger.debug("Write thread committed %d frames", count)
            db.commit()
            logger.debug("Write thread final commit: %d frames total", count)
        except Exception as exc:
            logger.exception("Write thread error at frame %d", count)
            error_holder.append(exc)
            db.rollback()
            while True:
                try:
                    remaining = write_queue.get_nowait()
                    if remaining is None:
                        break
                except Exception:
                    break


def run_pose_estimation(video_path, video_info, bout_id, db, progress_callback=None,
                        fencer_bbox=None, opponent_bbox=None, strip=None):
    """
    Run RTMPose WholeBody on every frame of the video.
    Persists Frame records to the database and returns a summary list.

    Uses a 3-thread pipeline for better GPU utilization:
      - Decode thread: FFmpeg stdout -> numpy frames (bounded queue, maxsize=8)
      - Main thread:   RTMPose inference + proximity-based matching
      - Write thread:  Frame dicts -> DB INSERT + batch commit (own session)
    """
    model = _get_model()

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
        "-s", f"{width}x{height}",
        "pipe:1",
    ]

    frame_bytes = width * height * 3
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    # --- Pipeline queues ---
    decode_queue: Queue = Queue(maxsize=8)
    write_queue: Queue = Queue(maxsize=64)
    write_errors: list[Exception] = []

    # --- Start background threads ---
    decode_thread = threading.Thread(
        target=_frame_decoder,
        args=(proc, width, height, frame_bytes, decode_queue),
        name="pose-decode",
        daemon=True,
    )
    write_thread = threading.Thread(
        target=_db_writer,
        args=(write_queue, bout_id, write_errors),
        name="pose-db-writer",
        daemon=True,
    )
    decode_thread.start()
    write_thread.start()

    # Proximity-based person matching state
    roi_mode = fencer_bbox is not None or opponent_bbox is not None
    fencer_locked = False
    opponent_locked = False
    fencer_last_cx: float | None = None
    fencer_last_cy: float | None = None
    opponent_last_cx: float | None = None
    opponent_last_cy: float | None = None

    # Re-lock after occlusion
    RELOCK_MIN_WAIT = 15
    RELOCK_CONFIRM = 3
    fencer_missing_count = 0
    fencer_candidate_cx: float | None = None
    fencer_candidate_cy: float | None = None
    fencer_candidate_count = 0
    opponent_missing_count = 0
    opponent_candidate_cx: float | None = None
    opponent_candidate_cy: float | None = None
    opponent_candidate_count = 0

    try:
        while True:
            item = decode_queue.get()
            if item is None:
                break
            frame_idx_decoded, frame = item

            # Run RTMPose WholeBody inference
            keypoints, scores = model(frame)
            # keypoints: (N, 133, 2) pixel coords, scores: (N, 133)

            timestamp_ms = int((frame_idx_decoded / fps) * 1000)

            fencer_pose = {}
            opponent_pose = {}

            n_detections = len(keypoints) if keypoints is not None and len(keypoints) > 0 else 0

            if n_detections > 0:
                # Build detection list with normalized centers
                detections = []
                for i in range(n_detections):
                    bbox = _bbox_from_keypoints(keypoints[i], scores[i], width, height)
                    if bbox is None:
                        detections.append({"cx": 0.0, "cy": 0.0, "bbox": None})
                        continue
                    cx, cy = _center_of_bbox(bbox)
                    detections.append({"cx": cx, "cy": cy, "bbox": bbox})

                if roi_mode:
                    fencer_idx = None
                    opponent_idx = None
                    exclude = set()

                    # --- Fencer matching ---
                    if fencer_locked and fencer_last_cx is not None:
                        # Try to find detection close to last known position
                        fencer_idx = _find_closest(
                            detections, fencer_last_cx, fencer_last_cy,
                            exclude, strip, keypoints, scores, width, height,
                            max_dist_sq=_MAX_TRACK_DIST_SQ)

                        if fencer_idx is not None:
                            fencer_missing_count = 0
                            fencer_candidate_cx = None
                            fencer_candidate_count = 0
                            fencer_last_cx = detections[fencer_idx]["cx"]
                            fencer_last_cy = detections[fencer_idx]["cy"]
                            fencer_pose = keypoints_to_dict(
                                keypoints[fencer_idx], scores[fencer_idx], width, height)
                            exclude.add(fencer_idx)
                        else:
                            # Fencer not found near last position
                            fencer_missing_count += 1
                            if fencer_missing_count >= RELOCK_MIN_WAIT:
                                # Expand search: find closest on-strip detection
                                candidate_idx = _find_closest(
                                    detections, fencer_last_cx, fencer_last_cy,
                                    exclude, strip, keypoints, scores, width, height)
                                if candidate_idx is not None:
                                    ccx = detections[candidate_idx]["cx"]
                                    ccy = detections[candidate_idx]["cy"]
                                    # Check if candidate is near previous candidate
                                    if (fencer_candidate_cx is not None
                                            and _dist_sq(ccx, ccy,
                                                         fencer_candidate_cx,
                                                         fencer_candidate_cy) < 0.01):
                                        fencer_candidate_count += 1
                                    else:
                                        fencer_candidate_cx = ccx
                                        fencer_candidate_cy = ccy
                                        fencer_candidate_count = 1
                                    if fencer_candidate_count >= RELOCK_CONFIRM:
                                        logger.info(
                                            "Frame %d: re-locked fencer (after %d missing frames)",
                                            frame_idx_decoded, fencer_missing_count)
                                        fencer_idx = candidate_idx
                                        fencer_last_cx = ccx
                                        fencer_last_cy = ccy
                                        fencer_missing_count = 0
                                        fencer_candidate_cx = None
                                        fencer_candidate_count = 0
                                        fencer_pose = keypoints_to_dict(
                                            keypoints[fencer_idx], scores[fencer_idx],
                                            width, height)
                                        exclude.add(fencer_idx)
                                else:
                                    fencer_candidate_cx = None
                                    fencer_candidate_count = 0

                    elif fencer_bbox is not None:
                        # Initial lock: find detection in user-selected ROI
                        fencer_idx = _find_in_roi(
                            detections, fencer_bbox, exclude, strip,
                            keypoints, scores, width, height)
                        if fencer_idx is not None:
                            logger.info("Frame %d: locked fencer", frame_idx_decoded)
                            fencer_locked = True
                            fencer_last_cx = detections[fencer_idx]["cx"]
                            fencer_last_cy = detections[fencer_idx]["cy"]
                            fencer_pose = keypoints_to_dict(
                                keypoints[fencer_idx], scores[fencer_idx], width, height)
                            exclude.add(fencer_idx)

                    # --- Opponent matching ---
                    if opponent_locked and opponent_last_cx is not None:
                        opponent_idx = _find_closest(
                            detections, opponent_last_cx, opponent_last_cy,
                            exclude, strip, keypoints, scores, width, height,
                            max_dist_sq=_MAX_TRACK_DIST_SQ)

                        if opponent_idx is not None:
                            opponent_missing_count = 0
                            opponent_candidate_cx = None
                            opponent_candidate_count = 0
                            opponent_last_cx = detections[opponent_idx]["cx"]
                            opponent_last_cy = detections[opponent_idx]["cy"]
                            opponent_pose = keypoints_to_dict(
                                keypoints[opponent_idx], scores[opponent_idx],
                                width, height)
                        else:
                            opponent_missing_count += 1
                            if opponent_missing_count >= RELOCK_MIN_WAIT:
                                candidate_idx = _find_closest(
                                    detections, opponent_last_cx, opponent_last_cy,
                                    exclude, strip, keypoints, scores, width, height)
                                if candidate_idx is not None:
                                    ccx = detections[candidate_idx]["cx"]
                                    ccy = detections[candidate_idx]["cy"]
                                    if (opponent_candidate_cx is not None
                                            and _dist_sq(ccx, ccy,
                                                         opponent_candidate_cx,
                                                         opponent_candidate_cy) < 0.01):
                                        opponent_candidate_count += 1
                                    else:
                                        opponent_candidate_cx = ccx
                                        opponent_candidate_cy = ccy
                                        opponent_candidate_count = 1
                                    if opponent_candidate_count >= RELOCK_CONFIRM:
                                        logger.info(
                                            "Frame %d: re-locked opponent (after %d missing frames)",
                                            frame_idx_decoded, opponent_missing_count)
                                        opponent_idx = candidate_idx
                                        opponent_last_cx = ccx
                                        opponent_last_cy = ccy
                                        opponent_missing_count = 0
                                        opponent_candidate_cx = None
                                        opponent_candidate_count = 0
                                        opponent_pose = keypoints_to_dict(
                                            keypoints[opponent_idx], scores[opponent_idx],
                                            width, height)
                                else:
                                    opponent_candidate_cx = None
                                    opponent_candidate_count = 0

                    elif opponent_bbox is not None:
                        opponent_idx = _find_in_roi(
                            detections, opponent_bbox, exclude, strip,
                            keypoints, scores, width, height)
                        if opponent_idx is not None:
                            logger.info("Frame %d: locked opponent", frame_idx_decoded)
                            opponent_locked = True
                            opponent_last_cx = detections[opponent_idx]["cx"]
                            opponent_last_cy = detections[opponent_idx]["cy"]
                            opponent_pose = keypoints_to_dict(
                                keypoints[opponent_idx], scores[opponent_idx],
                                width, height)

                    elif fencer_locked:
                        # Auto-assign opponent: highest confidence on-strip detection
                        opponent_idx = _find_best_confidence(
                            detections, exclude, strip, keypoints, scores, width, height)
                        if opponent_idx is not None:
                            logger.info("Frame %d: auto-assigned opponent", frame_idx_decoded)
                            opponent_locked = True
                            opponent_last_cx = detections[opponent_idx]["cx"]
                            opponent_last_cy = detections[opponent_idx]["cy"]
                            opponent_pose = keypoints_to_dict(
                                keypoints[opponent_idx], scores[opponent_idx],
                                width, height)

                else:
                    # No ROI: fall back to index order (0=fencer, 1=opponent)
                    if n_detections >= 1:
                        fencer_pose = keypoints_to_dict(
                            keypoints[0], scores[0], width, height)
                    if n_detections >= 2:
                        opponent_pose = keypoints_to_dict(
                            keypoints[1], scores[1], width, height)

            # Enqueue frame data for the write thread
            frame_data = {
                "bout_id": bout_id,
                "timestamp_ms": timestamp_ms,
                "fencer_pose": fencer_pose,
                "opponent_pose": opponent_pose if opponent_pose else None,
            }
            write_queue.put(frame_data)

            results_summary.append({"frame": frame_idx_decoded, "timestamp_ms": timestamp_ms})
            frame_idx = frame_idx_decoded + 1

            if progress_callback and frame_idx % 100 == 0:
                progress_callback(frame_idx, total_frames_hint)

    finally:
        write_queue.put(None)
        write_thread.join(timeout=120)
        if write_thread.is_alive():
            logger.error("Write thread did not finish within 120s timeout")
        decode_thread.join(timeout=10)
        proc.stdout.close()
        proc.wait()

    if write_errors:
        raise RuntimeError(
            f"DB write thread failed: {write_errors[0]}"
        ) from write_errors[0]

    logger.info("Pose estimation complete: %d frames persisted", frame_idx)
    return results_summary


# RTMPose WholeBody keypoint indices:
#   0-16:  COCO 17-point body skeleton
#   17-22: 6 foot keypoints (toe/heel detail for footwork analysis)
#   23-90: 68 face keypoints (not stored)
#   91-132: 42 hand keypoints (21 per hand — for blade direction)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    # Foot keypoints (RTMPose indices 17-22)
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

# Hand joint names (21 per hand) — COCO-WholeBody convention
_HAND_JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
# Left hand: RTMPose indices 91-111, Right hand: indices 112-132
_HAND_BASE_INDICES = {"lh": 91, "rh": 112}


def _normalize_score(raw: float) -> float:
    """Normalize SimCC logit scores to 0-1 probability range.

    RTMPose 'performance' mode returns SimCC heatmap logits (~0-8 range)
    rather than 0-1 probabilities. This sigmoid maps:
      0.5 -> 0.27, 1.0 -> 0.50, 2.0 -> 0.88, 3.0+ -> ~1.0
    """
    return 1.0 / (1.0 + _math_exp(-(raw - 1.0) * 2.0))


def keypoints_to_dict(kps: Any, scores: Any,
                      width: int = 1, height: int = 1) -> dict:
    """Convert keypoint arrays to a dict of named joints.

    kps: (133, 2) pixel coordinates from rtmlib
    scores: (133,) SimCC confidence values (normalized to 0-1 via sigmoid)
    width/height: frame dimensions for normalization (pixel -> 0-1)

    Stores body (17) + feet (6) + both hands (42) = 65 keypoints.
    """
    result = {}
    # Body + feet keypoints (indices 0-22)
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(kps):
            x = float(kps[i][0] / width) if width > 1 else float(kps[i][0])
            y = float(kps[i][1] / height) if height > 1 else float(kps[i][1])
            raw = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            c = _normalize_score(raw)
            if c < 0.05 and x == 0.0 and y == 0.0:
                continue
            result[name] = {"x": x, "y": y, "z": 0.0, "confidence": c}

    # Hand keypoints (indices 91-132) — for blade direction detection
    for prefix, base_idx in _HAND_BASE_INDICES.items():
        for j, joint_name in enumerate(_HAND_JOINT_NAMES):
            idx = base_idx + j
            if idx >= len(kps):
                break
            x = float(kps[idx][0] / width) if width > 1 else float(kps[idx][0])
            y = float(kps[idx][1] / height) if height > 1 else float(kps[idx][1])
            raw = float(scores[idx]) if scores is not None and idx < len(scores) else 0.0
            c = _normalize_score(raw)
            if c < 0.05 and x == 0.0 and y == 0.0:
                continue
            result[f"{prefix}_{joint_name}"] = {"x": x, "y": y, "z": 0.0, "confidence": c}

    return result
