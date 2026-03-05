"""Stage 2 — Heuristic blade tip projection from wrist/elbow keypoints.

Refinements (Priority 1):
  1a. Persistent weapon arm — determined once from fencer orientation
  1b. EMA temporal smoothing — reduces tip jitter before velocity calc
  1c. Fixed blade length — calibrated from max arm extension, constant for bout

Refinements (Priority 2):
  2a. Occlusion bridging — linear interpolation across short gaps (<= 5 frames)
      instead of resetting all state when confidence drops
  2b. Composite confidence score (0.0-1.0) per frame: keypoint + temporal + extension
"""
import math
import logging

logger = logging.getLogger(__name__)

_CONF_THRESHOLD = 0.3
_BATCH_SIZE = 300
_EMA_ALPHA = 0.4  # smoothing factor: lower = smoother, higher = more responsive
_BLADE_ARM_RATIO = 1.5  # 90cm blade / ~60cm arm
_MAX_GAP_FRAMES = 5  # max gap to interpolate across (~150ms at 30fps)

# Confidence weights
_KP_WEIGHT = 0.4
_TEMPORAL_WEIGHT = 0.4
_EXTENSION_WEIGHT = 0.2
_MAX_EXPECTED_JUMP = 0.15  # 15% of frame diagonal


def _determine_weapon_arm(frames: list, orientation: str) -> str:
    """
    Determine weapon arm from fencer orientation.

    Checks which COCO wrist keypoint is more consistently extended toward
    the opponent across the first 100 frames.
    """
    right_forward = 0
    left_forward = 0
    for frame in frames[:100]:
        pose = frame.fencer_pose
        if not pose:
            continue
        rw = pose.get("right_wrist", {})
        lw = pose.get("left_wrist", {})
        if rw.get("confidence", 0) < _CONF_THRESHOLD or lw.get("confidence", 0) < _CONF_THRESHOLD:
            continue
        if orientation == "right":
            # Facing screen-right: weapon arm extends rightward (higher x)
            if rw["x"] > lw["x"]:
                right_forward += 1
            else:
                left_forward += 1
        else:
            # Facing screen-left: weapon arm extends leftward (lower x)
            if rw["x"] < lw["x"]:
                right_forward += 1
            else:
                left_forward += 1

    weapon = "right" if right_forward >= left_forward else "left"
    logger.info("Weapon arm determined: %s (right_fwd=%d, left_fwd=%d, orientation=%s)",
                weapon, right_forward, left_forward, orientation)
    return weapon


def _calibrate_blade_length(frames: list, weapon: str, ar: float) -> float:
    """
    Measure blade length from the maximum observed arm extension.

    Scans frames for the longest shoulder->wrist distance and multiplies
    by _BLADE_ARM_RATIO. This gives a physically consistent blade length
    (real epee is always 90cm regardless of arm position).
    """
    max_arm_len = 0.0
    samples = 0

    for frame in frames:
        pose = frame.fencer_pose
        if not pose:
            continue

        wrist = pose.get(f"{weapon}_wrist", {})
        shoulder = pose.get(f"{weapon}_shoulder", {})

        if (wrist.get("confidence", 0) < _CONF_THRESHOLD
                or shoulder.get("confidence", 0) < _CONF_THRESHOLD):
            continue

        arm_dx = (wrist["x"] - shoulder["x"]) * ar
        arm_dy = wrist["y"] - shoulder["y"]
        arm_len = math.sqrt(arm_dx * arm_dx + arm_dy * arm_dy)
        if arm_len > max_arm_len:
            max_arm_len = arm_len
        samples += 1

    if max_arm_len < 0.01:
        blade_length = 0.3 * _BLADE_ARM_RATIO  # fallback
        logger.warning("Blade calibration: no valid arm measurements, using fallback %.3f", blade_length)
    else:
        blade_length = max_arm_len * _BLADE_ARM_RATIO
        logger.info("Blade length calibrated: %.3f (max arm=%.3f from %d samples)",
                    blade_length, max_arm_len, samples)

    return blade_length


def _compute_raw_tip(pose: dict, weapon: str, ar: float,
                     blade_scale: float) -> tuple[float, float, float, float] | None:
    """
    Compute raw (unsmoothed) blade tip from wrist/elbow keypoints.

    Returns (tip_x, tip_y, wrist_conf, elbow_conf) in normalised coordinates,
    or None if keypoints are below confidence threshold or the direction vector
    is degenerate.
    """
    wrist = pose.get(f"{weapon}_wrist", {})
    elbow = pose.get(f"{weapon}_elbow", {})

    wrist_conf = wrist.get("confidence", 0.0)
    elbow_conf = elbow.get("confidence", 0.0)

    if wrist_conf < _CONF_THRESHOLD or elbow_conf < _CONF_THRESHOLD:
        return None

    wx, wy = wrist["x"], wrist["y"]
    ex, ey = elbow["x"], elbow["y"]

    # Blade direction in aspect-ratio-correct space
    dx = (wx - ex) * ar
    dy = wy - ey
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return None
    dx /= mag
    dy /= mag

    # Project tip from wrist along blade direction (fixed length)
    tip_x = wx + (dx / ar) * blade_scale
    tip_y = wy + dy * blade_scale
    return (tip_x, tip_y, wrist_conf, elbow_conf)


def _interpolate_gap(last_tip: tuple[float, float], last_ts: int,
                     next_tip: tuple[float, float], next_ts: int,
                     gap_frames: list, db) -> None:
    """
    Linearly interpolate blade tip across buffered gap frames and write
    BladeState records. Velocity is computed from interpolated positions.

    gap_frames is a list of Frame ORM objects buffered during the gap.
    """
    from app.models.analysis import BladeState

    total_steps = len(gap_frames) + 1  # +1 for the next valid frame
    prev_x, prev_y = last_tip
    prev_ts = last_ts

    for i, gap_frame in enumerate(gap_frames):
        t = (i + 1) / total_steps  # interpolation parameter (0, 1)
        interp_x = last_tip[0] + t * (next_tip[0] - last_tip[0])
        interp_y = last_tip[1] + t * (next_tip[1] - last_tip[1])

        ts = gap_frame.timestamp_ms
        dt = (ts - prev_ts) / 1000.0
        if dt > 0:
            vel_x = (interp_x - prev_x) / dt
            vel_y = (interp_y - prev_y) / dt
            speed = math.sqrt(vel_x * vel_x + vel_y * vel_y)
        else:
            vel_x, vel_y, speed = 0.0, 0.0, 0.0

        blade = BladeState(
            frame_id=gap_frame.id,
            tip_xyz={"x": interp_x, "y": interp_y, "z": 0.0},
            nominal_xyz={"x": interp_x, "y": interp_y, "z": 0.0},
            velocity_xyz={"x": vel_x, "y": vel_y, "z": 0.0},
            speed=speed,
        )
        db.add(blade)

        prev_x, prev_y = interp_x, interp_y
        prev_ts = ts


def run_blade_tracking(frames: list, video_info: dict, db,
                       orientation: str = "right") -> None:
    """
    Compute nominal blade tip position for each frame using wrist-elbow geometry.

    Blade direction = elbow->wrist vector (normalized).
    Blade length = fixed, calibrated from max arm extension x 1.5.
    Tip position is EMA-smoothed before velocity calculation.

    Priority 2a: short gaps (< _MAX_GAP_FRAMES) where keypoint confidence
    drops are bridged via linear interpolation instead of resetting state.
    """
    from app.models.analysis import BladeState

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    ar = width / height

    # 1a. Persistent weapon arm from orientation
    weapon = _determine_weapon_arm(frames, orientation)

    # 1c. Fixed blade length from calibration
    blade_scale = _calibrate_blade_length(frames, weapon, ar)

    # EMA state
    smooth_x: float | None = None
    smooth_y: float | None = None
    prev_smooth_x: float | None = None
    prev_smooth_y: float | None = None
    prev_ts: int | None = None

    # Occlusion bridging state
    last_valid_tip: tuple[float, float] | None = None  # last smoothed tip
    last_valid_ts: int | None = None
    gap_frames: list = []  # buffered frames during a gap

    # Confidence state
    prev_raw_tip_x: float | None = None
    prev_raw_tip_y: float | None = None

    processed = 0
    interpolated = 0

    for frame in frames:
        pose = frame.fencer_pose
        raw_tip = None
        if pose:
            raw_tip = _compute_raw_tip(pose, weapon, ar, blade_scale)

        if raw_tip is None:
            # --- No valid tip for this frame ---
            if last_valid_tip is not None:
                # We have prior state — buffer this frame for potential interpolation
                gap_frames.append(frame)
                if len(gap_frames) > _MAX_GAP_FRAMES:
                    # Gap too long — abandon buffered frames, reset state
                    logger.debug("Blade gap exceeded %d frames at ts=%d, resetting",
                                 _MAX_GAP_FRAMES, frame.timestamp_ms)
                    gap_frames.clear()
                    smooth_x = None
                    smooth_y = None
                    prev_smooth_x = None
                    prev_smooth_y = None
                    prev_ts = None
                    last_valid_tip = None
                    last_valid_ts = None
                    prev_raw_tip_x = None
                    prev_raw_tip_y = None
            # else: no prior state, nothing to buffer
            continue

        # --- Valid tip for this frame ---
        raw_tip_x, raw_tip_y, wrist_conf, elbow_conf = raw_tip

        # Check if we need to backfill a gap
        if gap_frames:
            # We have buffered gap frames — interpolate them
            gap_count = len(gap_frames)
            last_gap_ts = gap_frames[-1].timestamp_ms

            _interpolate_gap(
                last_valid_tip, last_valid_ts,
                raw_tip, frame.timestamp_ms,
                gap_frames, db,
            )
            interpolated += gap_count
            processed += gap_count
            logger.debug("Blade gap bridged: %d frames interpolated ending at ts=%d",
                         gap_count, frame.timestamp_ms)
            gap_frames.clear()

            # Resume EMA from the last interpolated position to avoid
            # a discontinuity. The last interpolated point sits at
            # t = gap_count / (gap_count + 1) along the interpolation line.
            t = gap_count / (gap_count + 1)
            smooth_x = last_valid_tip[0] + t * (raw_tip_x - last_valid_tip[0])
            smooth_y = last_valid_tip[1] + t * (raw_tip_y - last_valid_tip[1])
            prev_smooth_x = smooth_x
            prev_smooth_y = smooth_y
            prev_ts = last_gap_ts

        # 1b. EMA smoothing
        if smooth_x is None:
            smooth_x = raw_tip_x
            smooth_y = raw_tip_y
        else:
            smooth_x = _EMA_ALPHA * raw_tip_x + (1 - _EMA_ALPHA) * smooth_x
            smooth_y = _EMA_ALPHA * raw_tip_y + (1 - _EMA_ALPHA) * smooth_y

        tip_xyz = {"x": smooth_x, "y": smooth_y, "z": 0.0}

        # 2b. Composite confidence score
        kp_conf = min(wrist_conf, elbow_conf)

        if prev_raw_tip_x is not None:
            jump_dx = raw_tip_x - prev_raw_tip_x
            jump_dy = raw_tip_y - prev_raw_tip_y
            jump_dist = math.sqrt(jump_dx * jump_dx + jump_dy * jump_dy)
            temporal_conf = 1.0 - min(jump_dist / _MAX_EXPECTED_JUMP, 1.0)
        else:
            temporal_conf = 1.0

        wrist_kp = pose.get(f"{weapon}_wrist", {})
        shoulder = pose.get(f"{weapon}_shoulder", {})
        if shoulder.get("confidence", 0) >= _CONF_THRESHOLD:
            arm_dx_ext = (wrist_kp["x"] - shoulder["x"]) * ar
            arm_dy_ext = wrist_kp["y"] - shoulder["y"]
            current_arm_len = math.sqrt(arm_dx_ext * arm_dx_ext + arm_dy_ext * arm_dy_ext)
            calibrated_max_arm = blade_scale / _BLADE_ARM_RATIO
            extension_conf = min(current_arm_len / calibrated_max_arm, 1.0) if calibrated_max_arm > 0 else 0.5
        else:
            extension_conf = 0.5

        blade_confidence = (
            _KP_WEIGHT * kp_conf
            + _TEMPORAL_WEIGHT * temporal_conf
            + _EXTENSION_WEIGHT * extension_conf
        )

        prev_raw_tip_x = raw_tip_x
        prev_raw_tip_y = raw_tip_y

        # Velocity from smoothed positions
        ts = frame.timestamp_ms
        if prev_smooth_x is not None and prev_ts is not None:
            dt = (ts - prev_ts) / 1000.0
            if dt > 0:
                vel_x = (smooth_x - prev_smooth_x) / dt
                vel_y = (smooth_y - prev_smooth_y) / dt
                speed = math.sqrt(vel_x * vel_x + vel_y * vel_y)
                velocity_xyz = {"x": vel_x, "y": vel_y, "z": 0.0}
            else:
                velocity_xyz = {"x": 0.0, "y": 0.0, "z": 0.0}
                speed = 0.0
        else:
            velocity_xyz = {"x": 0.0, "y": 0.0, "z": 0.0}
            speed = 0.0

        blade = BladeState(
            frame_id=frame.id,
            tip_xyz=tip_xyz,
            nominal_xyz={"x": raw_tip_x, "y": raw_tip_y, "z": 0.0},
            velocity_xyz=velocity_xyz,
            speed=speed,
            confidence=blade_confidence,
        )
        db.add(blade)

        prev_smooth_x = smooth_x
        prev_smooth_y = smooth_y
        prev_ts = ts
        last_valid_tip = (smooth_x, smooth_y)
        last_valid_ts = ts
        processed += 1

        if processed % _BATCH_SIZE == 0:
            db.commit()
            logger.debug("Blade tracking: committed %d frames", processed)

    db.commit()
    logger.info("Blade tracking complete: %d frames with blade state (%d interpolated)",
                processed, interpolated)
