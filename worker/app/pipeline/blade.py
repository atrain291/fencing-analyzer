"""Stage 2 — Heuristic blade tip projection from wrist/elbow keypoints."""
import math
import logging

logger = logging.getLogger(__name__)

_CONF_THRESHOLD = 0.3
_WEAPON_CONF_THRESHOLD = 0.4
_BATCH_SIZE = 300


def run_blade_tracking(frames: list, video_info: dict, db) -> None:
    """
    Compute nominal blade tip position for each frame using wrist-elbow geometry.

    Blade direction = elbow→wrist vector (normalized).
    Blade length ≈ 1.5× the shoulder-to-wrist distance (90cm blade / ~60cm arm).
    Tip is projected from wrist along blade direction.
    """
    from app.models.analysis import BladeState

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    ar = width / height  # aspect ratio for correct vector math

    prev_tip: dict | None = None
    prev_ts: int | None = None
    processed = 0

    for frame in frames:
        pose = frame.fencer_pose
        if not pose:
            prev_tip = None
            prev_ts = None
            continue

        # Determine weapon arm: prefer right, fallback to left
        r_wrist = pose.get("right_wrist", {})
        l_wrist = pose.get("left_wrist", {})
        r_conf = r_wrist.get("confidence", 0.0)
        l_conf = l_wrist.get("confidence", 0.0)

        if r_conf >= _WEAPON_CONF_THRESHOLD:
            weapon = "right"
        elif l_conf >= _WEAPON_CONF_THRESHOLD:
            weapon = "left"
        else:
            prev_tip = None
            prev_ts = None
            continue

        wrist = pose.get(f"{weapon}_wrist", {})
        elbow = pose.get(f"{weapon}_elbow", {})
        shoulder = pose.get(f"{weapon}_shoulder", {})

        wrist_conf = wrist.get("confidence", 0.0)
        elbow_conf = elbow.get("confidence", 0.0)

        if wrist_conf < _CONF_THRESHOLD or elbow_conf < _CONF_THRESHOLD:
            prev_tip = None
            prev_ts = None
            continue

        wx, wy = wrist["x"], wrist["y"]
        ex, ey = elbow["x"], elbow["y"]

        # Blade direction in aspect-ratio-correct space
        dx = (wx - ex) * ar
        dy = wy - ey
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            prev_tip = None
            prev_ts = None
            continue
        dx /= mag
        dy /= mag

        # Blade scale: use arm length (shoulder→wrist) × 1.5, fallback 0.3
        sx, sy = shoulder.get("x", 0.0), shoulder.get("y", 0.0)
        shoulder_conf = shoulder.get("confidence", 0.0)
        if shoulder_conf >= _CONF_THRESHOLD:
            arm_dx = (wx - sx) * ar
            arm_dy = wy - sy
            arm_len = math.sqrt(arm_dx * arm_dx + arm_dy * arm_dy)
        else:
            arm_len = 0.3  # ~30% of frame height as default

        blade_scale = arm_len * 1.5

        # Project tip back to normalized coords
        tip_x = wx + (dx / ar) * blade_scale
        tip_y = wy + dy * blade_scale
        tip_xyz = {"x": tip_x, "y": tip_y, "z": 0.0}

        # Velocity from previous frame
        ts = frame.timestamp_ms
        if prev_tip is not None and prev_ts is not None:
            dt = (ts - prev_ts) / 1000.0
            if dt > 0:
                vel_x = (tip_x - prev_tip["x"]) / dt
                vel_y = (tip_y - prev_tip["y"]) / dt
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
            nominal_xyz=tip_xyz,  # nominal = tip for Stage 2 (no flex correction yet)
            velocity_xyz=velocity_xyz,
            speed=speed,
        )
        db.add(blade)

        prev_tip = tip_xyz
        prev_ts = ts
        processed += 1

        if processed % _BATCH_SIZE == 0:
            db.commit()
            logger.debug("Blade tracking: committed %d frames", processed)

    db.commit()
    logger.info("Blade tracking complete: %d frames with blade state", processed)
