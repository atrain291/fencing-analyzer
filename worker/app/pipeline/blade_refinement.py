"""Post-WHAM blade refinement — uses 3D mesh data to improve blade tip tracking.

Two-pass system:
  Pass 1 (blade.py): runs immediately after pose estimation, uses 2D keypoints only
  Pass 2 (this file): runs after WHAM completes, reads MeshState rows, refines BladeState

What WHAM 3D provides:
  - True wrist rotation from SMPL body_pose (replaces heuristic 5-25° deflection)
  - 3D wrist→hand direction vector (camera-angle-independent blade axis)
  - Depth-aware arm length for blade calibration
  - Smooth trajectory to fill gaps where 2D keypoints dropped
"""
import math
import logging

import numpy as np

from app.pipeline.blade import BladeKalmanFilter, _BLADE_ARM_RATIO

logger = logging.getLogger(__name__)

# SMPL joint indices (matching SMPL_JOINT_NAMES in wham/app/inference.py)
# body_pose is 69 floats: 23 joints × 3 axis-angle values
# Joints 0-22: pelvis, l_hip, r_hip, spine1, l_knee, r_knee, spine2,
#   l_ankle, r_ankle, spine3, l_foot, r_foot, neck,
#   l_collar, r_collar, head, l_shoulder, r_shoulder,
#   l_elbow, r_elbow, l_wrist, r_wrist, l_hand, r_hand
# NOTE: body_pose excludes pelvis (that's global_orient), so indices shift by 1:
#   body_pose[0:3] = left_hip, ..., body_pose[57:60] = left_wrist, body_pose[60:63] = right_wrist
_SMPL_WRIST_IDX = {"left": 19, "right": 20}  # index into 23-joint body_pose (0-based, no pelvis)

_BATCH_SIZE = 300


def _axis_angle_to_rotation_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert 3-element axis-angle to 3x3 rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _extract_wrist_rotation(body_pose: list, weapon: str) -> np.ndarray:
    """Extract wrist rotation matrix from SMPL body_pose.

    body_pose: flat list of 69 floats (23 joints × 3 axis-angle, pelvis excluded)
    Returns: 3x3 rotation matrix for the wrist joint.
    """
    idx = _SMPL_WRIST_IDX[weapon]
    aa = np.array(body_pose[idx * 3: idx * 3 + 3], dtype=np.float64)
    return _axis_angle_to_rotation_matrix(aa)


def _compute_3d_blade_direction(joints_3d: dict, weapon: str) -> tuple[float, float, float] | None:
    """Compute blade direction from WHAM 3D wrist→hand vector.

    Returns normalised (dx, dy, dz) in WHAM camera space, or None.
    """
    wrist = joints_3d.get(f"{weapon}_wrist")
    hand = joints_3d.get(f"{weapon}_hand")
    if not wrist or not hand:
        return None

    dx = hand["x"] - wrist["x"]
    dy = hand["y"] - wrist["y"]
    dz = hand["z"] - wrist["z"]
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if mag < 1e-8:
        return None
    return (dx / mag, dy / mag, dz / mag)


def _project_direction_to_2d(
    dir_3d: tuple[float, float, float],
    wrist_3d: dict,
    width: int, height: int,
) -> tuple[float, float] | None:
    """Project a 3D direction vector to 2D screen-space direction.

    Uses perspective projection approximation. Returns normalised 2D (dx, dy)
    in screen-space where x=right, y=down, suitable for blade tip projection.
    """
    # Approximate focal length from typical video FOV (~55-65 degrees)
    focal = width / (2 * math.tan(math.radians(30)))

    wz = wrist_3d.get("z", 1.0)
    if wz < 0.1:
        return None

    # Project wrist and wrist+direction to screen, take the difference
    # WHAM camera space: x=right, y=up, z=forward (away from camera)
    wx = wrist_3d["x"]
    wy = wrist_3d["y"]

    # Small epsilon step along direction
    eps = 0.01
    dx3, dy3, dz3 = dir_3d
    end_x = wx + dx3 * eps
    end_y = wy + dy3 * eps
    end_z = wz + dz3 * eps
    if end_z < 0.1:
        return None

    # Screen projection: x_screen = focal * X/Z, y_screen = -focal * Y/Z (Y flipped)
    sx_w = focal * wx / wz
    sy_w = -focal * wy / wz
    sx_e = focal * end_x / end_z
    sy_e = -focal * end_y / end_z

    sdx = sx_e - sx_w
    sdy = sy_e - sy_w
    smag = math.sqrt(sdx * sdx + sdy * sdy)
    if smag < 1e-8:
        return None

    # Normalise to unit vector in screen-pixel space, then convert to
    # normalised coordinates (divide by frame dimensions)
    return (sdx / smag, sdy / smag)


def _get_median_3d_arm_length(mesh_states: list, weapon: str) -> float | None:
    """Get median 3D arm length (shoulder→elbow + elbow→wrist) from WHAM data.

    The 3D arm length is nearly constant across frames (skeleton doesn't change),
    so the median is a robust estimate of the true arm length in SMPL units.
    Returns None if insufficient data.
    """
    arm_lengths = []
    for ms in mesh_states:
        j3d = ms.joints_3d
        if not j3d:
            continue
        shoulder = j3d.get(f"{weapon}_shoulder")
        elbow = j3d.get(f"{weapon}_elbow")
        wrist = j3d.get(f"{weapon}_wrist")
        if not shoulder or not elbow or not wrist:
            continue
        se = math.sqrt(sum((elbow[k] - shoulder[k]) ** 2 for k in ("x", "y", "z")))
        ew = math.sqrt(sum((wrist[k] - elbow[k]) ** 2 for k in ("x", "y", "z")))
        arm_lengths.append(se + ew)

    if len(arm_lengths) < 5:
        return None

    arm_lengths.sort()
    median = arm_lengths[len(arm_lengths) // 2]
    logger.info("3D median arm length: %.4f (%d samples)", median, len(arm_lengths))
    return median


def _compute_per_frame_blade_scale(
    frame, pose_key: str, ms, weapon: str,
    arm_3d_median: float, ar: float,
) -> float | None:
    """Compute per-frame blade scale using 3D foreshortening correction.

    Compares the 2D projected arm length to the stable 3D arm length to
    determine how much the arm (and blade) are foreshortened in this frame.
    Returns blade_scale in normalised AR-corrected coordinates, or None.
    """
    pose = getattr(frame, pose_key) if isinstance(pose_key, str) else pose_key
    if not pose:
        return None

    shoulder = pose.get(f"{weapon}_shoulder", {})
    elbow = pose.get(f"{weapon}_elbow", {})
    wrist = pose.get(f"{weapon}_wrist", {})

    if (shoulder.get("confidence", 0) < 0.2
            or elbow.get("confidence", 0) < 0.2
            or wrist.get("confidence", 0) < 0.2):
        return None

    # 2D segmented arm: shoulder→elbow + elbow→wrist in AR-normalised space
    se_dx = (elbow["x"] - shoulder["x"]) * ar
    se_dy = elbow["y"] - shoulder["y"]
    ew_dx = (wrist["x"] - elbow["x"]) * ar
    ew_dy = wrist["y"] - elbow["y"]
    arm_2d = (math.sqrt(se_dx**2 + se_dy**2)
              + math.sqrt(ew_dx**2 + ew_dy**2))

    if arm_2d < 0.01 or arm_3d_median < 1e-4:
        return None

    # Foreshortening ratio: how much of the 3D arm projects to 2D in this frame
    # Ranges from ~0.5 (arm pointing at camera) to ~1.0 (arm parallel to image plane)
    foreshortening = arm_2d / arm_3d_median

    # Blade scale = true arm proportion * blade ratio * foreshortening
    # But since arm_2d already encodes foreshortening:
    #   blade_scale = arm_3d_median * BLADE_ARM_RATIO * foreshortening
    #               = arm_3d_median * BLADE_ARM_RATIO * (arm_2d / arm_3d_median)
    #               = arm_2d * BLADE_ARM_RATIO
    # This is mathematically equivalent, but the key difference from Pass 1 is
    # that this is PER-FRAME (arm_2d varies) instead of using a fixed P90.
    blade_scale = arm_2d * _BLADE_ARM_RATIO

    return blade_scale


def refine_blade_from_mesh(bout_id: int, db, video_info: dict,
                           orientation: str | None = None) -> int:
    """Refine blade states using WHAM 3D mesh data.

    Reads MeshState and BladeState rows from DB. For each frame with both:
    1. Compute 3D blade direction from wrist→hand vector
    2. Apply SMPL wrist rotation for true angulation
    3. Compute per-frame blade scale using 3D arm foreshortening
    4. Project to 2D and recompute blade tip
    5. Re-run Kalman filter over entire sequence with refined measurements

    Returns: number of refined frames.
    """
    from app.models.analysis import Frame, BladeState, MeshState

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    ar = width / height

    # Load frames with their blade states and mesh states
    frames = (
        db.query(Frame)
        .filter(Frame.bout_id == bout_id)
        .order_by(Frame.timestamp_ms)
        .all()
    )
    if not frames:
        logger.warning("Refinement: no frames for bout %d", bout_id)
        return 0

    frame_ids = [f.id for f in frames]

    # Load existing blade states indexed by frame_id (both subjects)
    blade_rows = (
        db.query(BladeState)
        .filter(BladeState.frame_id.in_(frame_ids), BladeState.subject == "fencer")
        .all()
    )
    blade_by_frame = {bs.frame_id: bs for bs in blade_rows}

    # Load fencer mesh states indexed by frame_id
    mesh_rows = (
        db.query(MeshState)
        .filter(MeshState.frame_id.in_(frame_ids), MeshState.subject == "fencer")
        .all()
    )
    mesh_by_frame = {ms.frame_id: ms for ms in mesh_rows}

    if not mesh_rows:
        logger.info("Refinement: no fencer mesh states for bout %d", bout_id)
        return 0

    # Determine weapon arm using position-based forward direction
    from app.pipeline.blade import _compute_forward_sign, _determine_weapon_arm_from_poses
    fwd_sign = _compute_forward_sign(frames, "fencer_pose", "opponent_pose")
    weapon = _determine_weapon_arm_from_poses(frames, "fencer_pose", fwd_sign)

    # Get median 3D arm length (stable — skeleton doesn't change shape)
    arm_3d_median = _get_median_3d_arm_length(mesh_rows, weapon)
    if arm_3d_median is None:
        logger.info("Refinement: insufficient 3D arm data for bout %d", bout_id)
        return 0

    # Also get the Pass 1 blade scale for fallback
    from app.pipeline.blade import _calibrate_blade_length_from_poses
    pass1_blade_scale = _calibrate_blade_length_from_poses(frames, weapon, ar, "fencer_pose")

    # First pass: compute refined measurements for all frames that have mesh data
    refined_measurements: dict[int, tuple[float, float, float]] = {}  # frame_id → (tip_x, tip_y, confidence)

    for frame in frames:
        ms = mesh_by_frame.get(frame.id)
        if not ms or not ms.joints_3d:
            continue

        pose = frame.fencer_pose
        if not pose:
            continue

        wrist_2d = pose.get(f"{weapon}_wrist", {})
        if wrist_2d.get("confidence", 0) < 0.1:
            continue

        wx, wy = wrist_2d["x"], wrist_2d["y"]

        # Get 3D blade direction from WHAM wrist→hand vector
        dir_3d = _compute_3d_blade_direction(ms.joints_3d, weapon)
        if dir_3d is None:
            continue

        # Apply SMPL wrist rotation to refine direction
        if ms.body_pose and len(ms.body_pose) >= 63:
            wrist_rot = _extract_wrist_rotation(ms.body_pose, weapon)
            dir_vec = np.array(dir_3d)
            rotated = wrist_rot @ dir_vec
            dir_3d = (float(rotated[0]), float(rotated[1]), float(rotated[2]))

        # Project 3D direction to 2D screen space
        wrist_3d = ms.joints_3d.get(f"{weapon}_wrist")
        if not wrist_3d:
            continue

        dir_2d = _project_direction_to_2d(dir_3d, wrist_3d, width, height)
        if dir_2d is None:
            continue

        dx_screen, dy_screen = dir_2d

        # Per-frame blade scale from 3D-corrected arm measurement
        per_frame_scale = _compute_per_frame_blade_scale(
            frame, "fencer_pose", ms, weapon, arm_3d_median, ar)
        blade_scale = per_frame_scale if per_frame_scale else pass1_blade_scale

        # Project tip from 2D wrist position along projected direction
        tip_x = wx + (dx_screen / ar) * blade_scale
        tip_y = wy + dy_screen * blade_scale

        # Confidence: base it on 2D wrist confidence (we know the 3D data is good)
        wrist_conf = wrist_2d.get("confidence", 0.5)
        mesh_conf = 0.8  # WHAM produces smooth consistent output
        refined_conf = max(wrist_conf, mesh_conf)

        refined_measurements[frame.id] = (tip_x, tip_y, refined_conf)

    if not refined_measurements:
        logger.info("Refinement: no frames could be refined for bout %d", bout_id)
        return 0

    logger.info("Refinement: %d/%d frames have 3D measurements, re-running Kalman",
                len(refined_measurements), len(frames))

    # Second pass: re-run Kalman filter over all frames, blending Pass 1 and 3D measurements
    kf = BladeKalmanFilter()
    prev_ts = None
    gap_count = 0
    refined_count = 0

    for frame in frames:
        ts = frame.timestamp_ms
        if prev_ts is not None:
            dt = (ts - prev_ts) / 1000.0
            if dt <= 0:
                dt = 1.0 / 30.0
        else:
            dt = 1.0 / 30.0
        prev_ts = ts

        bs = blade_by_frame.get(frame.id)

        # Determine measurement source: prefer 3D refined, fall back to Pass 1
        refined = refined_measurements.get(frame.id)
        if refined is not None:
            meas_x, meas_y, meas_conf = refined
            source = "wham_3d"
        elif bs is not None and bs.confidence is not None:
            # Use Pass 1 measurement (from nominal_xyz which is the raw measurement)
            nom = bs.nominal_xyz
            if nom and nom.get("x") is not None:
                meas_x = nom["x"]
                meas_y = nom["y"]
                meas_conf = bs.confidence
                source = "pass1"
            else:
                meas_x = meas_y = meas_conf = None
                source = None
        else:
            meas_x = meas_y = meas_conf = None
            source = None

        if meas_x is None:
            # No measurement — coast or skip
            if not kf.initialized:
                continue
            gap_count += 1
            if gap_count > 5:
                kf.reset()
                gap_count = 0
                continue
            kf.predict(dt)
            if bs:
                sx, sy = kf.position
                vx, vy = kf.velocity
                bs.tip_xyz = {"x": sx, "y": sy, "z": 0.0}
                bs.velocity_xyz = {"x": vx, "y": vy, "z": 0.0}
                bs.speed = kf.speed
                refined_count += 1
            continue

        gap_count = 0
        kf.predict(dt)
        kf.update(meas_x, meas_y, meas_conf)

        sx, sy = kf.position
        vx, vy = kf.velocity

        if bs:
            bs.tip_xyz = {"x": sx, "y": sy, "z": 0.0}
            bs.velocity_xyz = {"x": vx, "y": vy, "z": 0.0}
            bs.speed = kf.speed
            if source == "wham_3d":
                bs.nominal_xyz = {"x": meas_x, "y": meas_y, "z": 0.0}
                bs.confidence = meas_conf
            refined_count += 1

        if refined_count % _BATCH_SIZE == 0:
            db.commit()

    db.commit()
    logger.info("Blade refinement complete for bout %d: %d frames updated (%d with 3D data)",
                bout_id, refined_count, len(refined_measurements))
    return refined_count
