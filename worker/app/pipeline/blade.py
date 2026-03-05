"""Stage 2 — Heuristic blade tip projection from wrist/elbow keypoints.

Refinements (Priority 1):
  1a. Persistent weapon arm — determined once from fencer orientation
  1b. EMA temporal smoothing — reduces tip jitter before velocity calc
  1c. Fixed blade length — calibrated from max arm extension, constant for bout
"""
import math
import logging

logger = logging.getLogger(__name__)

_CONF_THRESHOLD = 0.3
_BATCH_SIZE = 300
_EMA_ALPHA = 0.4  # smoothing factor: lower = smoother, higher = more responsive
_BLADE_ARM_RATIO = 1.5  # 90cm blade / ~60cm arm


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

    Scans frames for the longest shoulder→wrist distance and multiplies
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


def run_blade_tracking(frames: list, video_info: dict, db,
                       orientation: str = "right") -> None:
    """
    Compute nominal blade tip position for each frame using wrist-elbow geometry.

    Blade direction = elbow→wrist vector (normalized).
    Blade length = fixed, calibrated from max arm extension × 1.5.
    Tip position is EMA-smoothed before velocity calculation.
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
    processed = 0

    for frame in frames:
        pose = frame.fencer_pose
        if not pose:
            smooth_x = None
            smooth_y = None
            prev_smooth_x = None
            prev_smooth_y = None
            prev_ts = None
            continue

        wrist = pose.get(f"{weapon}_wrist", {})
        elbow = pose.get(f"{weapon}_elbow", {})

        wrist_conf = wrist.get("confidence", 0.0)
        elbow_conf = elbow.get("confidence", 0.0)

        if wrist_conf < _CONF_THRESHOLD or elbow_conf < _CONF_THRESHOLD:
            smooth_x = None
            smooth_y = None
            prev_smooth_x = None
            prev_smooth_y = None
            prev_ts = None
            continue

        wx, wy = wrist["x"], wrist["y"]
        ex, ey = elbow["x"], elbow["y"]

        # Blade direction in aspect-ratio-correct space
        dx = (wx - ex) * ar
        dy = wy - ey
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            smooth_x = None
            smooth_y = None
            prev_smooth_x = None
            prev_smooth_y = None
            prev_ts = None
            continue
        dx /= mag
        dy /= mag

        # Project tip from wrist along blade direction (fixed length)
        raw_tip_x = wx + (dx / ar) * blade_scale
        raw_tip_y = wy + dy * blade_scale

        # 1b. EMA smoothing
        if smooth_x is None:
            smooth_x = raw_tip_x
            smooth_y = raw_tip_y
        else:
            smooth_x = _EMA_ALPHA * raw_tip_x + (1 - _EMA_ALPHA) * smooth_x
            smooth_y = _EMA_ALPHA * raw_tip_y + (1 - _EMA_ALPHA) * smooth_y

        tip_xyz = {"x": smooth_x, "y": smooth_y, "z": 0.0}

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
        )
        db.add(blade)

        prev_smooth_x = smooth_x
        prev_smooth_y = smooth_y
        prev_ts = ts
        processed += 1

        if processed % _BATCH_SIZE == 0:
            db.commit()
            logger.debug("Blade tracking: committed %d frames", processed)

    db.commit()
    logger.info("Blade tracking complete: %d frames with blade state", processed)
