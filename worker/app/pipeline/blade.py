"""Stage 2/3 — Heuristic blade tip projection from wrist/elbow keypoints.

Refinements (Priority 1):
  1a. Persistent weapon arm — determined once from fencer orientation
  1b. Temporal smoothing — Kalman filter (replaced EMA in Priority 3b)
  1c. Fixed blade length — calibrated from max arm extension, constant for bout

Refinements (Priority 2):
  2a. Occlusion bridging — Kalman predict-only coasting during gaps
      (replaced linear interpolation in Priority 3b)
  2b. Composite confidence score (0.0-1.0) per frame: keypoint + temporal + extension

Refinements (Priority 3):
  3a. Wrist angulation correction — deflects blade toward opponent based on arm extension
  3b. Kalman filter — 4-state [x, y, vx, vy] replaces EMA + manual velocity + gap interp
"""
import math
import logging

import numpy as np

logger = logging.getLogger(__name__)

_CONF_THRESHOLD = 0.3
_BATCH_SIZE = 300
_BLADE_ARM_RATIO = 1.5  # 90cm blade / ~60cm arm
_MAX_GAP_FRAMES = 5  # max gap to coast through before resetting Kalman state

# Confidence weights
_KP_WEIGHT = 0.4
_TEMPORAL_WEIGHT = 0.4
_EXTENSION_WEIGHT = 0.2
_MAX_EXPECTED_JUMP = 0.15  # 15% of frame diagonal

# Kalman filter parameters
_KF_PROCESS_NOISE = 50.0   # high: blade tips accelerate fast
_KF_MEASUREMENT_NOISE = 0.001  # ~3% frame diagonal std dev

# Wrist angulation parameters (Priority 3a)
_MIN_DEFLECTION_DEG = 5.0    # at rest / en garde
_MAX_DEFLECTION_DEG = 25.0   # at full extension
_EXT_RATIO_LOW = 0.55        # below this, use min deflection
_EXT_RATIO_HIGH = 0.95       # above this, use max deflection
_DEFLECTION_CURVE = 1.4      # exponential ramp


class BladeKalmanFilter:
    """2D Kalman filter for blade tip tracking.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, q: float = _KF_PROCESS_NOISE, r: float = _KF_MEASUREMENT_NOISE):
        self.q = q
        self.r_base = r
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.initialized = False

    def init_state(self, x: float, y: float):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([0.001, 0.001, 1.0, 1.0])
        self.initialized = True

    def predict(self, dt: float):
        if not self.initialized:
            return
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = np.array([
            [dt3 / 3, 0, dt2 / 2, 0],
            [0, dt3 / 3, 0, dt2 / 2],
            [dt2 / 2, 0, dt, 0],
            [0, dt2 / 2, 0, dt],
        ]) * self.q

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_x: float, z_y: float, confidence: float = 1.0):
        if not self.initialized:
            self.init_state(z_x, z_y)
            return

        z = np.array([z_x, z_y])
        scale = self.r_base / max(confidence, 0.1)
        R = np.eye(2) * scale

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(math.sqrt(self.x[2] ** 2 + self.x[3] ** 2))

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.initialized = False


def _determine_weapon_arm(frames: list, orientation: str) -> str:
    """Determine weapon arm from fencer orientation."""
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
            if rw["x"] > lw["x"]:
                right_forward += 1
            else:
                left_forward += 1
        else:
            if rw["x"] < lw["x"]:
                right_forward += 1
            else:
                left_forward += 1

    weapon = "right" if right_forward >= left_forward else "left"
    logger.info("Weapon arm determined: %s (right_fwd=%d, left_fwd=%d, orientation=%s)",
                weapon, right_forward, left_forward, orientation)
    return weapon


def _calibrate_blade_length(frames: list, weapon: str, ar: float) -> float:
    """Measure blade length from the maximum observed arm extension."""
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
        blade_length = 0.3 * _BLADE_ARM_RATIO
        logger.warning("Blade calibration: no valid arm measurements, using fallback %.3f", blade_length)
    else:
        blade_length = max_arm_len * _BLADE_ARM_RATIO
        logger.info("Blade length calibrated: %.3f (max arm=%.3f from %d samples)",
                    blade_length, max_arm_len, samples)

    return blade_length


def _compute_wrist_angulation(pose: dict, weapon: str, ar: float,
                              orientation: str) -> float:
    """
    Compute wrist angulation angle (radians) based on arm extension ratio.

    Returns a signed angle to rotate the blade direction vector. The blade
    deflects toward the opponent (inward) proportional to arm extension.
    Higher extension → more angulation (epee fencers aim with the wrist).
    """
    shoulder = pose.get(f"{weapon}_shoulder", {})
    elbow = pose.get(f"{weapon}_elbow", {})
    wrist = pose.get(f"{weapon}_wrist", {})

    if (shoulder.get("confidence", 0) < _CONF_THRESHOLD
            or elbow.get("confidence", 0) < _CONF_THRESHOLD
            or wrist.get("confidence", 0) < _CONF_THRESHOLD):
        return 0.0

    sx, sy = shoulder["x"], shoulder["y"]
    ex, ey = elbow["x"], elbow["y"]
    wx, wy = wrist["x"], wrist["y"]

    # Compute arm extension ratio in AR-corrected space
    d_se = math.sqrt(((ex - sx) * ar) ** 2 + (ey - sy) ** 2)
    d_ew = math.sqrt(((wx - ex) * ar) ** 2 + (wy - ey) ** 2)
    d_sw = math.sqrt(((wx - sx) * ar) ** 2 + (wy - sy) ** 2)

    segment_sum = d_se + d_ew
    if segment_sum < 1e-6:
        return 0.0

    extension_ratio = d_sw / segment_sum

    # Map extension to deflection angle with exponential ramp
    t = (extension_ratio - _EXT_RATIO_LOW) / (_EXT_RATIO_HIGH - _EXT_RATIO_LOW)
    t = max(0.0, min(1.0, t))
    t_curved = t ** _DEFLECTION_CURVE

    deflection_deg = _MIN_DEFLECTION_DEG + t_curved * (_MAX_DEFLECTION_DEG - _MIN_DEFLECTION_DEG)

    # Determine deflection direction (sign): toward opponent
    # Forearm direction vector (elbow→wrist)
    forearm_dx = (wx - ex) * ar
    forearm_dy = wy - ey

    # Opponent is to the left if facing right, to the right if facing left.
    # We want to deflect the blade DOWNWARD toward the opponent's body center.
    # In epee, the default angulation is slightly downward from the forearm
    # axis (toward opponent's wrist/torso which is typically at or below arm height).
    #
    # Use opponent skeleton if available, otherwise use a fixed inward+downward bias.
    # For now: deflect toward the opponent side (inward).
    # Cross product of forearm_dir with "down" vector gives the sign.
    # Facing right: opponent is to the right (higher x in screen), deflect clockwise (negative angle)
    # Facing left: opponent is to the left (lower x in screen), deflect counter-clockwise (positive angle)
    if orientation == "right":
        sign = -1.0  # clockwise rotation = blade tip moves downward/inward
    else:
        sign = 1.0

    return sign * math.radians(deflection_deg)


def _compute_raw_tip(pose: dict, weapon: str, ar: float,
                     blade_scale: float, orientation: str) -> tuple[float, float, float, float] | None:
    """
    Compute raw (unsmoothed) blade tip from wrist/elbow keypoints.

    Includes wrist angulation correction (Priority 3a): the blade direction
    is rotated toward the opponent based on arm extension ratio.

    Returns (tip_x, tip_y, wrist_conf, elbow_conf) in normalised coordinates,
    or None if keypoints are below confidence threshold.
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

    # 3a. Wrist angulation correction
    angulation = _compute_wrist_angulation(pose, weapon, ar, orientation)
    if angulation != 0.0:
        cos_a = math.cos(angulation)
        sin_a = math.sin(angulation)
        dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a

    # Project tip from wrist along blade direction (fixed length)
    tip_x = wx + (dx / ar) * blade_scale
    tip_y = wy + dy * blade_scale
    return (tip_x, tip_y, wrist_conf, elbow_conf)


def run_blade_tracking(frames: list, video_info: dict, db,
                       orientation: str = "right") -> None:
    """
    Compute nominal blade tip position for each frame using wrist-elbow geometry.

    Priority 3a: wrist angulation correction rotates blade toward opponent.
    Priority 3b: Kalman filter [x, y, vx, vy] replaces EMA smoothing,
    handles occlusion by coasting on velocity (predict-only), and provides
    velocity estimates directly (no finite differencing).
    """
    from app.models.analysis import BladeState

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    ar = width / height

    # 1a. Persistent weapon arm from orientation
    weapon = _determine_weapon_arm(frames, orientation)

    # 1c. Fixed blade length from calibration
    blade_scale = _calibrate_blade_length(frames, weapon, ar)
    calibrated_max_arm = blade_scale / _BLADE_ARM_RATIO

    # 3b. Kalman filter state
    kf = BladeKalmanFilter()
    prev_ts: int | None = None
    gap_count = 0

    # Confidence state
    prev_raw_tip_x: float | None = None
    prev_raw_tip_y: float | None = None

    processed = 0
    coasted = 0

    for frame in frames:
        pose = frame.fencer_pose
        ts = frame.timestamp_ms

        # Compute dt
        if prev_ts is not None:
            dt = (ts - prev_ts) / 1000.0
            if dt <= 0:
                dt = 1.0 / 30.0
        else:
            dt = 1.0 / 30.0

        raw_tip = None
        if pose:
            raw_tip = _compute_raw_tip(pose, weapon, ar, blade_scale, orientation)

        if raw_tip is None:
            # --- No valid measurement ---
            if not kf.initialized:
                prev_ts = ts
                continue

            gap_count += 1
            if gap_count > _MAX_GAP_FRAMES:
                # Gap too long — reset Kalman state
                logger.debug("Blade gap exceeded %d frames at ts=%d, resetting Kalman",
                             _MAX_GAP_FRAMES, ts)
                kf.reset()
                gap_count = 0
                prev_raw_tip_x = None
                prev_raw_tip_y = None
                prev_ts = ts
                continue

            # Coast: predict-only step (no measurement update)
            kf.predict(dt)
            smooth_x, smooth_y = kf.position
            vel_x, vel_y = kf.velocity

            blade = BladeState(
                frame_id=frame.id,
                tip_xyz={"x": smooth_x, "y": smooth_y, "z": 0.0},
                nominal_xyz={"x": smooth_x, "y": smooth_y, "z": 0.0},
                velocity_xyz={"x": vel_x, "y": vel_y, "z": 0.0},
                speed=kf.speed,
                # No confidence for coasted frames (same as old interpolation behavior)
            )
            db.add(blade)
            coasted += 1
            processed += 1
            prev_ts = ts

            if processed % _BATCH_SIZE == 0:
                db.commit()
            continue

        # --- Valid measurement ---
        raw_tip_x, raw_tip_y, wrist_conf, elbow_conf = raw_tip
        gap_count = 0

        # 2b. Composite confidence score
        kp_conf = min(wrist_conf, elbow_conf)

        if prev_raw_tip_x is not None:
            jump_dx = raw_tip_x - prev_raw_tip_x
            jump_dy = raw_tip_y - prev_raw_tip_y
            jump_dist = math.sqrt(jump_dx * jump_dx + jump_dy * jump_dy)
            temporal_conf = 1.0 - min(jump_dist / _MAX_EXPECTED_JUMP, 1.0)
        else:
            temporal_conf = 1.0

        if pose:
            wrist_kp = pose.get(f"{weapon}_wrist", {})
            shoulder = pose.get(f"{weapon}_shoulder", {})
            if shoulder.get("confidence", 0) >= _CONF_THRESHOLD:
                arm_dx_ext = (wrist_kp["x"] - shoulder["x"]) * ar
                arm_dy_ext = wrist_kp["y"] - shoulder["y"]
                current_arm_len = math.sqrt(arm_dx_ext * arm_dx_ext + arm_dy_ext * arm_dy_ext)
                extension_conf = min(current_arm_len / calibrated_max_arm, 1.0) if calibrated_max_arm > 0 else 0.5
            else:
                extension_conf = 0.5
        else:
            extension_conf = 0.5

        blade_confidence = (
            _KP_WEIGHT * kp_conf
            + _TEMPORAL_WEIGHT * temporal_conf
            + _EXTENSION_WEIGHT * extension_conf
        )

        prev_raw_tip_x = raw_tip_x
        prev_raw_tip_y = raw_tip_y

        # 3b. Kalman predict + update
        kf.predict(dt)
        kf.update(raw_tip_x, raw_tip_y, blade_confidence)

        smooth_x, smooth_y = kf.position
        vel_x, vel_y = kf.velocity

        blade = BladeState(
            frame_id=frame.id,
            tip_xyz={"x": smooth_x, "y": smooth_y, "z": 0.0},
            nominal_xyz={"x": raw_tip_x, "y": raw_tip_y, "z": 0.0},
            velocity_xyz={"x": vel_x, "y": vel_y, "z": 0.0},
            speed=kf.speed,
            confidence=blade_confidence,
        )
        db.add(blade)

        prev_ts = ts
        processed += 1

        if processed % _BATCH_SIZE == 0:
            db.commit()
            logger.debug("Blade tracking: committed %d frames", processed)

    db.commit()
    logger.info("Blade tracking complete: %d frames with blade state (%d coasted)",
                processed, coasted)
