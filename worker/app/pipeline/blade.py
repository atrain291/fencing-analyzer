"""Stage 2/3 — Blade tip projection from hand keypoints + wrist/elbow fallback.

Blade direction cascade (best to worst):
  1. Hand keypoints: index finger MCP→PIP direction (actual grip orientation)
  2. Hand keypoints: MCP line perpendicular (knuckle line, 90° rotated)
  3. Hand keypoints: wrist→MCP midpoint (coarsest hand signal)
  4. Forearm: elbow→wrist + wrist angulation correction (original heuristic)

Refinements (Priority 1-3b) — all implemented:
  1a. Persistent weapon arm — determined once from fencer orientation
  1c. Fixed blade length — calibrated from max arm extension
  2a. Occlusion bridging — Kalman predict-only coasting during gaps
  2b. Composite confidence score (0.0-1.0): keypoint + temporal + extension
  3a. Wrist angulation correction (fallback path only)
  3b. Kalman filter — 4-state [x, y, vx, vy]
  3c. Hand keypoint blade direction (this file)
"""
import math
import logging

import numpy as np

logger = logging.getLogger(__name__)

_CONF_THRESHOLD = 0.3
_HAND_CONF_THRESHOLD = 0.2  # lower threshold for hand keypoints
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


def _compute_hand_blade_direction(
    hand_kps: dict, ar: float,
) -> tuple[float, float, float] | None:
    """Derive blade direction from weapon hand keypoints.

    Tries three methods in order of accuracy:
    1. Index finger MCP→PIP direction (best proxy for grip/blade axis)
    2. MCP line perpendicular (knuckle line rotated 90°)
    3. Hand wrist→MCP midpoint (coarsest signal)

    Args:
        hand_kps: dict of hand keypoints {joint_name: {x, y, confidence}}
                  (already filtered to weapon hand, no prefix)
        ar: aspect ratio (width/height) for distance correction

    Returns: (dx, dy, method_confidence) in AR-corrected space, or None.
    """
    thr = _HAND_CONF_THRESHOLD

    def _kp(name):
        kp = hand_kps.get(name, {})
        if kp.get("confidence", 0) >= thr and (kp.get("x", 0) != 0 or kp.get("y", 0) != 0):
            return kp
        return None

    # Method 1: Index finger MCP→PIP direction
    idx_mcp = _kp("index_mcp")
    idx_pip = _kp("index_pip")
    if idx_mcp and idx_pip:
        dx = (idx_pip["x"] - idx_mcp["x"]) * ar
        dy = idx_pip["y"] - idx_mcp["y"]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 1e-6:
            method_conf = min(idx_mcp["confidence"], idx_pip["confidence"])
            return (dx / mag, dy / mag, method_conf)

    # Method 2: MCP line perpendicular
    mid_mcp = _kp("middle_mcp")
    if idx_mcp and mid_mcp:
        # MCP line runs across knuckles; blade is perpendicular
        line_dx = (mid_mcp["x"] - idx_mcp["x"]) * ar
        line_dy = mid_mcp["y"] - idx_mcp["y"]
        mag = math.sqrt(line_dx * line_dx + line_dy * line_dy)
        if mag > 1e-6:
            # Perpendicular: rotate 90° (choice of direction resolved by checking
            # that result points away from wrist, i.e. toward fingertips)
            perp_dx = -line_dy / mag
            perp_dy = line_dx / mag

            # Check direction: should point from wrist toward fingers
            hw = _kp("wrist")
            if hw:
                mcp_mid_x = (idx_mcp["x"] + mid_mcp["x"]) / 2
                mcp_mid_y = (idx_mcp["y"] + mid_mcp["y"]) / 2
                to_mcp_dx = (mcp_mid_x - hw["x"]) * ar
                to_mcp_dy = mcp_mid_y - hw["y"]
                if perp_dx * to_mcp_dx + perp_dy * to_mcp_dy < 0:
                    perp_dx, perp_dy = -perp_dx, -perp_dy

            method_conf = min(idx_mcp["confidence"], mid_mcp["confidence"]) * 0.8
            return (perp_dx, perp_dy, method_conf)

    # Method 3: Hand wrist → MCP midpoint
    hw = _kp("wrist")
    any_mcp = idx_mcp or mid_mcp or _kp("ring_mcp")
    if hw and any_mcp:
        target = any_mcp  # best available MCP
        dx = (target["x"] - hw["x"]) * ar
        dy = target["y"] - hw["y"]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 1e-6:
            method_conf = min(hw["confidence"], target["confidence"]) * 0.6
            return (dx / mag, dy / mag, method_conf)

    return None


def _compute_raw_tip(pose: dict, weapon: str, ar: float,
                     blade_scale: float, orientation: str,
                     hand_kps: dict | None = None) -> tuple[float, float, float, float] | None:
    """
    Compute raw (unsmoothed) blade tip position.

    Uses hand keypoints for blade direction when available (Priority 3c),
    falling back to elbow→wrist + wrist angulation (Priority 3a).

    Returns (tip_x, tip_y, wrist_conf, direction_conf) in normalised coordinates,
    or None if keypoints are below confidence threshold.
    """
    wrist = pose.get(f"{weapon}_wrist", {})
    elbow = pose.get(f"{weapon}_elbow", {})

    wrist_conf = wrist.get("confidence", 0.0)

    if wrist_conf < _CONF_THRESHOLD:
        return None

    wx, wy = wrist["x"], wrist["y"]

    # Try hand keypoints first for blade direction
    hand_result = None
    if hand_kps:
        hand_result = _compute_hand_blade_direction(hand_kps, ar)

    if hand_result is not None:
        dx, dy, hand_conf = hand_result
        # Blend with forearm direction when hand confidence is moderate
        elbow_conf = elbow.get("confidence", 0.0)
        if elbow_conf >= _CONF_THRESHOLD:
            ex, ey = elbow["x"], elbow["y"]
            forearm_dx = (wx - ex) * ar
            forearm_dy = wy - ey
            forearm_mag = math.sqrt(forearm_dx * forearm_dx + forearm_dy * forearm_dy)
            if forearm_mag > 1e-6:
                forearm_dx /= forearm_mag
                forearm_dy /= forearm_mag
                # Blend: hand_conf < 0.3 → mostly forearm; > 0.5 → mostly hand
                blend = min(1.0, max(0.0, (hand_conf - 0.2) / 0.3))
                dx = blend * dx + (1.0 - blend) * forearm_dx
                dy = blend * dy + (1.0 - blend) * forearm_dy
                mag = math.sqrt(dx * dx + dy * dy)
                if mag > 1e-6:
                    dx /= mag
                    dy /= mag
        direction_conf = hand_conf
    else:
        # Fallback: elbow→wrist direction with angulation correction
        elbow_conf = elbow.get("confidence", 0.0)
        if elbow_conf < _CONF_THRESHOLD:
            return None
        ex, ey = elbow["x"], elbow["y"]
        dx = (wx - ex) * ar
        dy = wy - ey
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            return None
        dx /= mag
        dy /= mag

        # 3a. Wrist angulation correction (only for forearm-based direction)
        angulation = _compute_wrist_angulation(pose, weapon, ar, orientation)
        if angulation != 0.0:
            cos_a = math.cos(angulation)
            sin_a = math.sin(angulation)
            dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
        direction_conf = elbow_conf

    # Project tip from wrist along blade direction (fixed length)
    tip_x = wx + (dx / ar) * blade_scale
    tip_y = wy + dy * blade_scale
    return (tip_x, tip_y, wrist_conf, direction_conf)


def run_blade_tracking(frames: list, video_info: dict, db,
                       orientation: str = "right") -> None:
    """
    Compute nominal blade tip position for each frame.

    Uses hand keypoints for blade direction when available (Priority 3c),
    falling back to elbow→wrist + wrist angulation (Priority 3a).
    Kalman filter (Priority 3b) smooths and coasts through occlusions.
    """
    from app.models.analysis import BladeState

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    ar = width / height

    # 1a. Persistent weapon arm from orientation
    weapon = _determine_weapon_arm(frames, orientation)
    hand_prefix = "rh" if weapon == "right" else "lh"

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
            # Extract weapon hand keypoints (strip prefix for blade direction fn)
            hand_kps = {
                k[len(hand_prefix) + 1:]: v
                for k, v in pose.items()
                if k.startswith(hand_prefix + "_")
            } or None
            raw_tip = _compute_raw_tip(pose, weapon, ar, blade_scale, orientation,
                                       hand_kps=hand_kps)

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
