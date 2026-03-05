"""Stage 2 — Per-frame pose estimation using YOLOv8-Pose (CUDA).

Uses a 3-thread pipeline to overlap FFmpeg decoding, YOLO inference, and
DB writes for better GPU utilization:

  Decode thread ─→ [decode_queue] ─→ Main thread (YOLO + tracking) ─→ [write_queue] ─→ Write thread (DB)

BoT-SORT tracking is sequential by design (frame N depends on frame N-1),
so inference itself cannot be parallelized.  But we can keep the GPU busy
by having the next decoded frame ready in the queue while the previous
frame's DB write happens concurrently.
"""
import logging
import os
import subprocess
import threading
from queue import Queue
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model = None

# Custom tracker config with increased track_buffer for occlusion resilience
_TRACKER_CFG = os.path.join(os.path.dirname(__file__), "botsort_fencing.yaml")


def _get_model():
    """Load YOLO11x-Pose model, preferring TensorRT engine over PyTorch weights.

    TensorRT engine files are GPU-architecture-specific and must be exported
    once per machine via `python export_tensorrt.py` inside the worker container.
    """
    global _model
    if _model is None:
        from ultralytics import YOLO
        engine_path = os.path.join(os.path.dirname(__file__), "..", "..", "yolo11x-pose.engine")
        if os.path.exists(engine_path):
            _model = YOLO(engine_path, task="pose")
            logger.info("YOLO11x-Pose TensorRT engine loaded from %s", engine_path)
        else:
            _model = YOLO("yolo11x-pose.pt")
            logger.info(
                "YOLO11x-Pose PyTorch model loaded (no TensorRT engine found at %s)",
                engine_path,
            )
    return _model


def _center_in_bbox(cx: float, cy: float, bbox: dict) -> bool:
    """Check if point (cx, cy) falls inside a normalized bbox {x1,y1,x2,y2}."""
    return bbox['x1'] <= cx <= bbox['x2'] and bbox['y1'] <= cy <= bbox['y2']


def _try_lock_id(result, bbox: dict, exclude_ids: set[int]) -> int | None:
    """
    Try to lock a tracker ID from the current frame using an ROI bbox.
    Returns the tracker ID of the first detection whose center falls in bbox,
    excluding any ID in exclude_ids (partner ID + known bystanders).
    Returns None if no match found or tracking IDs unavailable.
    """
    if result.boxes is None or result.boxes.id is None:
        return None
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
    for i, tid in enumerate(track_ids):
        if int(tid) in exclude_ids:
            continue
        if i >= len(boxes_xyxyn):
            break
        box = boxes_xyxyn[i]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if _center_in_bbox(cx, cy, bbox):
            return int(tid)
    return None


def _all_track_ids(result) -> set[int]:
    """Return the set of all tracker IDs visible in this frame."""
    if result.boxes is None or result.boxes.id is None:
        return set()
    return set(result.boxes.id.cpu().numpy().astype(int).tolist())


def _get_center_by_id(result, track_id: int) -> tuple[float, float] | None:
    """Return normalized center (cx, cy) for the given tracker ID, or None."""
    if result.boxes is None or result.boxes.id is None:
        return None
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
    for i, tid in enumerate(track_ids):
        if tid == track_id and i < len(boxes_xyxyn):
            box = boxes_xyxyn[i]
            return (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
    return None


def _closest_id_excluding(result, cx: float, cy: float, exclude_ids: set[int]) -> int | None:
    """Find the tracker ID whose detection center is closest to (cx, cy),
    excluding any ID in exclude_ids. Returns None if no candidates."""
    if result.boxes is None or result.boxes.id is None:
        return None
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
    best_tid = None
    best_dist = float('inf')
    for i, tid in enumerate(track_ids):
        if int(tid) in exclude_ids:
            continue
        if i >= len(boxes_xyxyn):
            break
        box = boxes_xyxyn[i]
        dcx = float((box[0] + box[2]) / 2)
        dcy = float((box[1] + box[3]) / 2)
        dist = (dcx - cx) ** 2 + (dcy - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_tid = int(tid)
    return best_tid


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


def _frame_decoder(proc, width: int, height: int, frame_bytes: int,
                    decode_queue: Queue) -> None:
    """Decode thread: reads raw BGR frames from FFmpeg stdout and enqueues them.

    Runs in a background thread so decoding overlaps with YOLO inference.
    Puts ``(frame_idx, numpy_array)`` tuples into *decode_queue*.
    Sends a ``None`` sentinel when the pipe is exhausted or an error occurs.
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
        decode_queue.put(None)  # sentinel — main thread will stop reading


def _db_writer(write_queue: Queue, bout_id: int, error_holder: list) -> None:
    """Write thread: creates Frame ORM objects from dicts and commits in batches.

    Runs in a background thread with its own SQLAlchemy session so DB I/O
    overlaps with YOLO inference on the main thread.  Receives plain dicts
    (not ORM objects) to avoid cross-thread session contamination.
    Stops when it receives a ``None`` sentinel.

    If an error occurs, it is stored in *error_holder* (a shared list) so the
    main thread can detect and re-raise it after joining.
    """
    from app.db import get_db_session
    from app.models.analysis import Frame

    with get_db_session() as db:
        count = 0
        try:
            while True:
                item = write_queue.get()
                if item is None:  # sentinel — main thread is done
                    break
                db_frame = Frame(**item)
                db.add(db_frame)
                count += 1
                if count % 300 == 0:
                    db.commit()
                    logger.debug("Write thread committed %d frames", count)
            db.commit()  # final batch
            logger.debug("Write thread final commit: %d frames total", count)
        except Exception as exc:
            logger.exception("Write thread error at frame %d", count)
            error_holder.append(exc)
            db.rollback()
            # Drain remaining items so the main thread's put() won't block
            while True:
                try:
                    remaining = write_queue.get_nowait()
                    if remaining is None:
                        break
                except Exception:
                    break


def run_pose_estimation(video_path, video_info, bout_id, db, progress_callback=None,
                        fencer_bbox=None, opponent_bbox=None):
    """
    Run YOLOv8-Pose on every frame of the video.
    Persists Frame records to the database and returns a summary list.

    Uses a 3-thread pipeline for better GPU utilization:
      - Decode thread: FFmpeg stdout -> numpy frames (bounded queue, maxsize=8)
      - Main thread:   YOLO inference + BoT-SORT tracking (sequential, as required)
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
    # Bounded to limit memory: 8 decoded frames ~= 8 * 1920*1080*3 ~= 47 MB
    decode_queue: Queue = Queue(maxsize=8)
    write_queue: Queue = Queue(maxsize=64)

    # Shared list for the write thread to report errors back to the main thread
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

    # Tracker ID locking state (persist across frames)
    fencer_track_id: int | None = None
    opponent_track_id: int | None = None
    roi_mode = fencer_bbox is not None or opponent_bbox is not None

    # Every tracker ID that is NOT the fencer or opponent.  When re-locking
    # after occlusion, these IDs are excluded so we don't grab a bystander
    # who walked through the ROI.
    known_other_ids: set[int] = set()

    # Confirmation-based re-lock: when the locked ID goes missing, we look for
    # a new ID in the ROI.  Only switch if the same new ID appears in the ROI
    # for RELOCK_CONFIRM consecutive frames (filters transient occluders).
    RELOCK_MIN_WAIT = 15       # don't even start looking for a new ID until this many missing frames
    RELOCK_CONFIRM = 3         # new candidate must be stable for this many consecutive frames
    fencer_missing_count = 0
    fencer_candidate_id: int | None = None
    fencer_candidate_count = 0
    fencer_last_cx: float | None = None   # last known normalized center
    fencer_last_cy: float | None = None
    opponent_missing_count = 0
    opponent_candidate_id: int | None = None
    opponent_candidate_count = 0
    opponent_last_cx: float | None = None
    opponent_last_cy: float | None = None

    try:
        while True:
            item = decode_queue.get()
            if item is None:  # sentinel from decode thread
                break
            frame_idx_decoded, frame = item

            persist = frame_idx_decoded > 0  # False on first frame resets stale tracker/GMC state
            results = model.track(frame, persist=persist, device="cuda", verbose=False,
                                  imgsz=1280, tracker=_TRACKER_CFG)
            result = results[0]

            timestamp_ms = int((frame_idx_decoded / fps) * 1000)

            fencer_pose = {}
            opponent_pose = {}
            fencer_found = False
            opponent_found = False

            if result.keypoints is not None and len(result.keypoints) > 0:
                if roi_mode:
                    # Collect all visible IDs so we can tag bystanders
                    frame_ids = _all_track_ids(result)

                    # Build exclusion sets for each role
                    fencer_exclude = known_other_ids.copy()
                    if opponent_track_id is not None:
                        fencer_exclude.add(opponent_track_id)
                    opponent_exclude = known_other_ids.copy()
                    if fencer_track_id is not None:
                        opponent_exclude.add(fencer_track_id)

                    # --- Fencer tracking ---
                    if fencer_track_id is not None:
                        kps, conf = _get_kps_by_id(result, fencer_track_id)
                        if kps is not None:
                            fencer_pose = keypoints_to_dict(kps, conf)
                            fencer_found = True
                            fencer_missing_count = 0
                            fencer_candidate_id = None
                            fencer_candidate_count = 0
                            # Track last known position for proximity re-lock
                            center = _get_center_by_id(result, fencer_track_id)
                            if center:
                                fencer_last_cx, fencer_last_cy = center
                        else:
                            fencer_missing_count += 1
                            if fencer_missing_count >= RELOCK_MIN_WAIT:
                                # Proximity to last known position
                                new_id = None
                                if fencer_last_cx is not None:
                                    new_id = _closest_id_excluding(
                                        result, fencer_last_cx, fencer_last_cy, fencer_exclude)
                                if new_id is not None:
                                    if new_id == fencer_candidate_id:
                                        fencer_candidate_count += 1
                                    else:
                                        fencer_candidate_id = new_id
                                        fencer_candidate_count = 1
                                    if fencer_candidate_count >= RELOCK_CONFIRM:
                                        logger.info(
                                            "Frame %d: re-locked fencer ID %d -> %d "
                                            "(after %d missing frames)",
                                            frame_idx_decoded, fencer_track_id, new_id,
                                            fencer_missing_count)
                                        fencer_track_id = new_id
                                        fencer_missing_count = 0
                                        fencer_candidate_id = None
                                        fencer_candidate_count = 0
                                        kps, conf = _get_kps_by_id(result, fencer_track_id)
                                        if kps is not None:
                                            fencer_pose = keypoints_to_dict(kps, conf)
                                            fencer_found = True
                                            center = _get_center_by_id(result, fencer_track_id)
                                            if center:
                                                fencer_last_cx, fencer_last_cy = center
                                else:
                                    fencer_candidate_id = None
                                    fencer_candidate_count = 0
                    elif fencer_bbox is not None:
                        fencer_track_id = _try_lock_id(result, fencer_bbox, fencer_exclude)
                        if fencer_track_id is not None:
                            logger.info("Frame %d: locked fencer tracker ID %d",
                                        frame_idx_decoded, fencer_track_id)
                            fencer_found = True
                            kps, conf = _get_kps_by_id(result, fencer_track_id)
                            if kps is not None:
                                fencer_pose = keypoints_to_dict(kps, conf)
                            center = _get_center_by_id(result, fencer_track_id)
                            if center:
                                fencer_last_cx, fencer_last_cy = center

                    # --- Opponent tracking ---
                    if opponent_track_id is not None:
                        kps, conf = _get_kps_by_id(result, opponent_track_id)
                        if kps is not None:
                            opponent_pose = keypoints_to_dict(kps, conf)
                            opponent_found = True
                            opponent_missing_count = 0
                            opponent_candidate_id = None
                            opponent_candidate_count = 0
                            center = _get_center_by_id(result, opponent_track_id)
                            if center:
                                opponent_last_cx, opponent_last_cy = center
                        else:
                            opponent_missing_count += 1
                            if opponent_missing_count >= RELOCK_MIN_WAIT:
                                # Proximity to last known position
                                new_id = None
                                if opponent_last_cx is not None:
                                    new_id = _closest_id_excluding(
                                        result, opponent_last_cx, opponent_last_cy, opponent_exclude)
                                if new_id is not None:
                                    if new_id == opponent_candidate_id:
                                        opponent_candidate_count += 1
                                    else:
                                        opponent_candidate_id = new_id
                                        opponent_candidate_count = 1
                                    if opponent_candidate_count >= RELOCK_CONFIRM:
                                        logger.info(
                                            "Frame %d: re-locked opponent ID %d -> %d "
                                            "(after %d missing frames)",
                                            frame_idx_decoded, opponent_track_id, new_id,
                                            opponent_missing_count)
                                        opponent_track_id = new_id
                                        opponent_missing_count = 0
                                        opponent_candidate_id = None
                                        opponent_candidate_count = 0
                                        kps, conf = _get_kps_by_id(result, opponent_track_id)
                                        if kps is not None:
                                            opponent_pose = keypoints_to_dict(kps, conf)
                                            opponent_found = True
                                            center = _get_center_by_id(result, opponent_track_id)
                                            if center:
                                                opponent_last_cx, opponent_last_cy = center
                                else:
                                    opponent_candidate_id = None
                                    opponent_candidate_count = 0
                    elif opponent_bbox is not None:
                        opponent_track_id = _try_lock_id(result, opponent_bbox, opponent_exclude)
                        if opponent_track_id is not None:
                            logger.info("Frame %d: locked opponent tracker ID %d",
                                        frame_idx_decoded, opponent_track_id)
                            opponent_found = True
                            kps, conf = _get_kps_by_id(result, opponent_track_id)
                            if kps is not None:
                                opponent_pose = keypoints_to_dict(kps, conf)
                            center = _get_center_by_id(result, opponent_track_id)
                            if center:
                                opponent_last_cx, opponent_last_cy = center
                    elif fencer_track_id is not None:
                        opponent_track_id = _best_other_id(result, fencer_track_id)
                        if opponent_track_id is not None:
                            logger.info("Frame %d: auto-assigned opponent tracker ID %d",
                                        frame_idx_decoded, opponent_track_id)
                            opponent_found = True
                            kps, conf = _get_kps_by_id(result, opponent_track_id)
                            if kps is not None:
                                opponent_pose = keypoints_to_dict(kps, conf)
                            center = _get_center_by_id(result, opponent_track_id)
                            if center:
                                opponent_last_cx, opponent_last_cy = center

                    # Tag bystanders ONLY when both fencer and opponent are
                    # positively identified in this frame.  If either is
                    # missing, a new tracker ID might belong to the missing
                    # person (tracker ID re-assignment after occlusion).
                    # Also never tag current re-lock candidates as bystanders.
                    if fencer_found and opponent_found:
                        protected = {fencer_track_id, opponent_track_id,
                                     fencer_candidate_id, opponent_candidate_id}
                        for tid in frame_ids:
                            if tid not in protected:
                                known_other_ids.add(tid)

                else:
                    # No ROI: fall back to index order (kps[0]=fencer, kps[1]=opponent)
                    kps_all = result.keypoints.xyn.cpu().numpy()
                    conf_all = (result.keypoints.conf.cpu().numpy()
                                if result.keypoints.conf is not None else None)
                    if len(kps_all) >= 1:
                        fencer_pose = keypoints_to_dict(
                            kps_all[0], conf_all[0] if conf_all is not None else None)
                    if len(kps_all) >= 2:
                        opponent_pose = keypoints_to_dict(
                            kps_all[1], conf_all[1] if conf_all is not None else None)

            # Enqueue frame data as a plain dict for the write thread.
            # The write thread creates Frame ORM objects with its own DB session
            # to avoid cross-thread SQLAlchemy session usage.
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
        # Signal write thread to finish
        write_queue.put(None)

        # Wait for the write thread to drain and commit
        write_thread.join(timeout=120)
        if write_thread.is_alive():
            logger.error("Write thread did not finish within 120s timeout")

        # Wait for decode thread (should already be done since we consumed sentinel)
        decode_thread.join(timeout=10)

        # Clean up FFmpeg process
        proc.stdout.close()
        proc.wait()

    # Re-raise any error from the write thread so the pipeline fails properly
    if write_errors:
        raise RuntimeError(
            f"DB write thread failed: {write_errors[0]}"
        ) from write_errors[0]

    logger.info("Pose estimation complete: %d frames persisted", frame_idx)
    return results_summary


# COCO keypoint indices (YOLOv8-Pose uses COCO 17-point skeleton)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def keypoints_to_dict(kps: Any, conf: Any) -> dict:
    result = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(kps):
            x, y = float(kps[i][0]), float(kps[i][1])
            c = float(conf[i]) if conf is not None and i < len(conf) else 0.0
            # Skip undetected keypoints (YOLO11x returns 0,0 for missing joints)
            if c < 0.01 and x == 0.0 and y == 0.0:
                continue
            result[name] = {"x": x, "y": y, "z": 0.0, "confidence": c}
    return result
