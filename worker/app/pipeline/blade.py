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
_HAND_MIN_SPAN_NORM = 0.008  # minimum normalised distance for hand direction methods
                             # (~15px at 1920w). Below this, hand keypoints are too
                             # close together to derive a reliable direction.
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
# In fencing, the wrist deflects the blade significantly from the forearm axis.
# En garde: blade nearly horizontal while forearm droops ~45° → large deflection.
# Full extension: blade aligns more with arm → moderate deflection.
_MIN_DEFLECTION_DEG = 15.0   # minimum (arm extended, blade aligns with forearm)
_MAX_DEFLECTION_DEG = 45.0   # maximum (arm bent, en garde — blade is much more horizontal)
_EXT_RATIO_LOW = 0.55        # below this, use max deflection (arm bent = blade horizontal)
_EXT_RATIO_HIGH = 0.95       # above this, use min deflection (arm extended = blade aligns)
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
    # INVERTED: bent arm (low extension) → max deflection, extended → min deflection
    # In en garde (ext ~0.6), blade is nearly horizontal while forearm droops → large correction
    # At full extension (ext ~0.95), blade aligns with arm → small correction
    t = (extension_ratio - _EXT_RATIO_LOW) / (_EXT_RATIO_HIGH - _EXT_RATIO_LOW)
    t = max(0.0, min(1.0, t))
    t_curved = t ** _DEFLECTION_CURVE

    deflection_deg = _MAX_DEFLECTION_DEG - t_curved * (_MAX_DEFLECTION_DEG - _MIN_DEFLECTION_DEG)

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

    At typical fencing video scale (full body in 1080p), the hand is only
    ~20-30px across. Individual joint-to-joint distances (MCP→PIP) are just
    3-5px — too small for reliable direction. We gate all methods on a minimum
    spatial spread and prefer the wrist→fingertip-centroid vector (~20px)
    over individual joint directions.

    Methods (in order of robustness at fencing scale):
    1. Wrist → fingertip centroid (largest span, most stable)
    2. Wrist → any MCP (fallback if fingertips unavailable)

    Args:
        hand_kps: dict of hand keypoints {joint_name: {x, y, confidence}}
                  (already filtered to weapon hand, no prefix)
        ar: aspect ratio (width/height) for distance correction

    Returns: (dx, dy, method_confidence) in AR-corrected space, or None.
    """
    thr = _HAND_CONF_THRESHOLD
    min_span = _HAND_MIN_SPAN_NORM

    def _kp(name):
        kp = hand_kps.get(name, {})
        if kp.get("confidence", 0) >= thr and (kp.get("x", 0) != 0 or kp.get("y", 0) != 0):
            return kp
        return None

    hw = _kp("wrist")
    if not hw:
        return None

    # Method 1: Wrist → fingertip centroid
    # Average all visible fingertip positions — much more spatially robust
    # than individual joint-to-joint vectors at low resolution
    tip_names = ["index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    tips = [_kp(n) for n in tip_names]
    valid_tips = [t for t in tips if t is not None]

    if len(valid_tips) >= 2:
        cx = sum(t["x"] for t in valid_tips) / len(valid_tips)
        cy = sum(t["y"] for t in valid_tips) / len(valid_tips)
        dx = (cx - hw["x"]) * ar
        dy = cy - hw["y"]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag >= min_span:
            min_conf = min(t["confidence"] for t in valid_tips)
            method_conf = min(hw["confidence"], min_conf) * 0.7
            return (dx / mag, dy / mag, method_conf)

    # Method 2: Wrist → any MCP (coarser but still directional)
    for mcp_name in ["index_mcp", "middle_mcp", "ring_mcp"]:
        mcp = _kp(mcp_name)
        if mcp:
            dx = (mcp["x"] - hw["x"]) * ar
            dy = mcp["y"] - hw["y"]
            mag = math.sqrt(dx * dx + dy * dy)
            if mag >= min_span:
                method_conf = min(hw["confidence"], mcp["confidence"]) * 0.5
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
        # Always blend hand direction with angulated forearm direction.
        # At fencing video scale, hand keypoints are small (~20px span) and
        # noisy. The forearm+angulation provides a stable base direction.
        elbow_conf = elbow.get("confidence", 0.0)
        if elbow_conf >= _CONF_THRESHOLD:
            ex, ey = elbow["x"], elbow["y"]
            forearm_dx = (wx - ex) * ar
            forearm_dy = wy - ey
            forearm_mag = math.sqrt(forearm_dx * forearm_dx + forearm_dy * forearm_dy)
            if forearm_mag > 1e-6:
                forearm_dx /= forearm_mag
                forearm_dy /= forearm_mag
                # Apply wrist angulation to the forearm direction
                angulation = _compute_wrist_angulation(pose, weapon, ar, orientation)
                if angulation != 0.0:
                    cos_a = math.cos(angulation)
                    sin_a = math.sin(angulation)
                    forearm_dx, forearm_dy = (
                        forearm_dx * cos_a - forearm_dy * sin_a,
                        forearm_dx * sin_a + forearm_dy * cos_a,
                    )
                # Blend: hand gets at most 50% weight to prevent noise domination
                blend = min(0.5, max(0.0, (hand_conf - 0.2) / 0.6))
                dx = blend * dx + (1.0 - blend) * forearm_dx
                dy = blend * dy + (1.0 - blend) * forearm_dy
                mag = math.sqrt(dx * dx + dy * dy)
                if mag > 1e-6:
                    dx /= mag
                    dy /= mag
        direction_conf = max(hand_conf, elbow_conf)
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

    # Forward-hemisphere check: blade must have a component toward the opponent.
    # Facing right → opponent is to the right → dx must be positive (in AR space)
    # Facing left → opponent is to the left → dx must be negative (in AR space)
    forward_dx = 1.0 if orientation == "right" else -1.0
    if dx * forward_dx < 0:
        # Blade direction points backwards — flip to point forward.
        # This happens when the arm is bent and elbow→wrist vector is misleading.
        dx = forward_dx
        dy = 0.0  # horizontal toward opponent as fallback

    # Project tip from wrist along blade direction (fixed length)
    tip_x = wx + (dx / ar) * blade_scale
    tip_y = wy + dy * blade_scale
    return (tip_x, tip_y, wrist_conf, direction_conf)


def _run_blade_for_subject(
    frames: list, video_info: dict, db,
    subject: str, pose_key: str,
    forward_sign: float,
) -> int:
    """Run blade tracking for a single subject (fencer or opponent).

    Args:
        frames: list of Frame ORM objects
        video_info: dict with width, height, fps
        db: database session
        subject: "fencer" or "opponent"
        pose_key: attribute name on Frame ("fencer_pose" or "opponent_pose")
        forward_sign: +1.0 if opponent is to the right, -1.0 if to the left

    Returns: number of frames processed
    """
    from app.models.analysis import BladeState

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    ar = width / height

    # Derive orientation from forward_sign for angulation and other heuristics
    # forward_sign > 0 means opponent is to the right, so this subject faces right
    subject_orientation = "right" if forward_sign > 0 else "left"

    # Determine weapon arm for this subject
    weapon = _determine_weapon_arm_from_poses(frames, pose_key, forward_sign)
    hand_prefix = "rh" if weapon == "right" else "lh"

    # Fixed blade length from calibration
    blade_scale = _calibrate_blade_length_from_poses(frames, weapon, ar, pose_key)
    calibrated_max_arm = blade_scale / _BLADE_ARM_RATIO

    kf = BladeKalmanFilter()
    prev_ts: int | None = None
    gap_count = 0
    prev_raw_tip_x: float | None = None
    prev_raw_tip_y: float | None = None

    processed = 0
    coasted = 0

    for frame in frames:
        pose = getattr(frame, pose_key)
        ts = frame.timestamp_ms

        if prev_ts is not None:
            dt = (ts - prev_ts) / 1000.0
            if dt <= 0:
                dt = 1.0 / 30.0
        else:
            dt = 1.0 / 30.0

        raw_tip = None
        if pose:
            hand_kps = {
                k[len(hand_prefix) + 1:]: v
                for k, v in pose.items()
                if k.startswith(hand_prefix + "_")
            } or None
            raw_tip = _compute_raw_tip(pose, weapon, ar, blade_scale,
                                       subject_orientation, hand_kps=hand_kps)

        if raw_tip is None:
            if not kf.initialized:
                prev_ts = ts
                continue

            gap_count += 1
            if gap_count > _MAX_GAP_FRAMES:
                kf.reset()
                gap_count = 0
                prev_raw_tip_x = None
                prev_raw_tip_y = None
                prev_ts = ts
                continue

            kf.predict(dt)
            smooth_x, smooth_y = kf.position
            vel_x, vel_y = kf.velocity

            blade = BladeState(
                frame_id=frame.id,
                subject=subject,
                tip_xyz={"x": smooth_x, "y": smooth_y, "z": 0.0},
                nominal_xyz={"x": smooth_x, "y": smooth_y, "z": 0.0},
                velocity_xyz={"x": vel_x, "y": vel_y, "z": 0.0},
                speed=kf.speed,
            )
            db.add(blade)
            coasted += 1
            processed += 1
            prev_ts = ts

            if processed % _BATCH_SIZE == 0:
                db.commit()
            continue

        raw_tip_x, raw_tip_y, wrist_conf, dir_conf = raw_tip
        gap_count = 0

        kp_conf = min(wrist_conf, dir_conf)

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

        kf.predict(dt)
        kf.update(raw_tip_x, raw_tip_y, blade_confidence)

        smooth_x, smooth_y = kf.position
        vel_x, vel_y = kf.velocity

        blade = BladeState(
            frame_id=frame.id,
            subject=subject,
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

    db.commit()
    logger.info("Blade tracking [%s] complete: %d frames (%d coasted)", subject, processed, coasted)
    return processed


def _compute_forward_sign(frames: list, pose_key: str, opponent_pose_key: str) -> float:
    """Compute forward direction (toward opponent) from actual pose positions.

    Returns +1.0 if opponent is to the right, -1.0 if to the left.
    Falls back to +1.0 if no opponent data.
    """
    sum_dx = 0.0
    count = 0
    for frame in frames[:200]:
        pose = getattr(frame, pose_key)
        opp = getattr(frame, opponent_pose_key)
        if not pose or not opp:
            continue
        # Use hip midpoint as body center (more stable than wrist)
        s_lh = pose.get("left_hip", {})
        s_rh = pose.get("right_hip", {})
        o_lh = opp.get("left_hip", {})
        o_rh = opp.get("right_hip", {})
        if (s_lh.get("confidence", 0) < _CONF_THRESHOLD
                and s_rh.get("confidence", 0) < _CONF_THRESHOLD):
            continue
        if (o_lh.get("confidence", 0) < _CONF_THRESHOLD
                and o_rh.get("confidence", 0) < _CONF_THRESHOLD):
            continue
        sx = 0.0
        sn = 0
        if s_lh.get("confidence", 0) >= _CONF_THRESHOLD:
            sx += s_lh["x"]; sn += 1
        if s_rh.get("confidence", 0) >= _CONF_THRESHOLD:
            sx += s_rh["x"]; sn += 1
        ox = 0.0
        on = 0
        if o_lh.get("confidence", 0) >= _CONF_THRESHOLD:
            ox += o_lh["x"]; on += 1
        if o_rh.get("confidence", 0) >= _CONF_THRESHOLD:
            ox += o_rh["x"]; on += 1
        if sn > 0 and on > 0:
            sum_dx += (ox / on) - (sx / sn)
            count += 1

    if count == 0:
        logger.warning("Cannot determine forward direction for %s — no opponent data, defaulting to +1", pose_key)
        return 1.0

    avg_dx = sum_dx / count
    sign = 1.0 if avg_dx > 0 else -1.0
    logger.info("Forward direction [%s]: %+.1f (avg opponent dx=%.3f, %d samples)",
                pose_key, sign, avg_dx, count)
    return sign


def _determine_weapon_arm_from_poses(frames: list, pose_key: str, forward_sign: float) -> str:
    """Determine weapon arm from poses — the arm that extends toward the opponent."""
    right_forward = 0
    left_forward = 0
    for frame in frames[:100]:
        pose = getattr(frame, pose_key)
        if not pose:
            continue
        rw = pose.get("right_wrist", {})
        lw = pose.get("left_wrist", {})
        if rw.get("confidence", 0) < _CONF_THRESHOLD or lw.get("confidence", 0) < _CONF_THRESHOLD:
            continue
        # The weapon arm is the one extending more toward the opponent
        if forward_sign > 0:
            # Opponent is to the right → weapon arm has higher x
            if rw["x"] > lw["x"]:
                right_forward += 1
            else:
                left_forward += 1
        else:
            # Opponent is to the left → weapon arm has lower x
            if rw["x"] < lw["x"]:
                right_forward += 1
            else:
                left_forward += 1

    weapon = "right" if right_forward >= left_forward else "left"
    logger.info("Weapon arm [%s]: %s (right_fwd=%d, left_fwd=%d)",
                pose_key, weapon, right_forward, left_forward)
    return weapon


def _calibrate_blade_length_from_poses(frames: list, weapon: str, ar: float,
                                        pose_key: str) -> float:
    """Measure blade length from max arm extension (generalized for any subject)."""
    max_arm_len = 0.0
    samples = 0
    for frame in frames:
        pose = getattr(frame, pose_key)
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
    else:
        blade_length = max_arm_len * _BLADE_ARM_RATIO
    logger.info("Blade length [%s]: %.3f (max arm=%.3f from %d samples)",
                pose_key, blade_length, max_arm_len, samples)
    return blade_length


def run_blade_tracking(frames: list, video_info: dict, db,
                       orientation: str = "right") -> None:
    """Run blade tracking for both fencer and opponent."""
    # Compute forward direction from actual positions (not orientation heuristic)
    fencer_fwd = _compute_forward_sign(frames, "fencer_pose", "opponent_pose")
    opponent_fwd = _compute_forward_sign(frames, "opponent_pose", "fencer_pose")

    _run_blade_for_subject(frames, video_info, db,
                           subject="fencer", pose_key="fencer_pose",
                           forward_sign=fencer_fwd)
    _run_blade_for_subject(frames, video_info, db,
                           subject="opponent", pose_key="opponent_pose",
                           forward_sign=opponent_fwd)
