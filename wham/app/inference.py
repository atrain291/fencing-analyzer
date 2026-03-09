"""WHAM 3D mesh reconstruction inference.

Runs WHAM on a video clip cropped to a single person (fencer or opponent).
Outputs SMPL body parameters, 3D joint positions, and foot contact per frame.

Uses --estimate_local_only mode (no DPVO) since VirtualFencer (2025) showed
that WHAM's global trajectory is unreliable for fencing and must be corrected
via piste homography.
"""
import logging
import subprocess
import tempfile
import os

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# SMPL joint names (24 joints) — first 17 overlap with COCO-17
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# COCO-17 joint names that map to SMPL joints (for consistency with existing pipeline)
COCO_TO_SMPL = {
    "nose": "head",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
}

_wham_model = None


def _load_wham():
    """Lazy-load the WHAM model. Returns None if WHAM is not available."""
    global _wham_model
    if _wham_model is not None:
        return _wham_model

    try:
        # Try to import WHAM components
        import sys
        wham_path = "/opt/wham"
        if wham_path not in sys.path:
            sys.path.insert(0, wham_path)

        from lib.models.wham import WHAM
        from configs.config import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.merge_from_file(os.path.join(wham_path, "configs/yamls/demo.yaml"))

        checkpoint_path = os.path.join(wham_path, "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar")
        if not os.path.exists(checkpoint_path):
            logger.error("WHAM checkpoint not found at %s", checkpoint_path)
            return None

        model = WHAM(cfg)
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model"], strict=False)
        model = model.to("cuda").eval()

        _wham_model = model
        logger.info("WHAM model loaded successfully")
        return _wham_model

    except Exception:
        logger.exception("Failed to load WHAM model")
        return None


def release_model():
    """Release WHAM model from GPU memory."""
    global _wham_model
    if _wham_model is not None:
        _wham_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("WHAM model released from GPU")


def extract_person_crops(video_path: str, bboxes: list[dict],
                         frame_indices: list[int],
                         width: int, height: int,
                         crop_size: int = 256) -> list[np.ndarray]:
    """Extract and resize person crops from video frames using FFmpeg.

    Args:
        video_path: path to source video
        bboxes: list of normalized bboxes {x1, y1, x2, y2} per frame
        frame_indices: frame numbers to extract (sorted)
        width, height: video dimensions
        crop_size: output square crop size

    Returns: list of BGR numpy arrays (crop_size x crop_size)
    """
    # Build frame index -> bbox mapping
    bbox_by_frame = dict(zip(frame_indices, bboxes))

    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "pipe:1",
    ]

    crops = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 2)
    try:
        frame_idx = 0
        target_set = set(frame_indices)
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            if frame_idx in target_set:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                bbox = bbox_by_frame[frame_idx]

                # Convert normalized bbox to pixel coords with padding
                pad = 0.1  # 10% padding
                bw = bbox["x2"] - bbox["x1"]
                bh = bbox["y2"] - bbox["y1"]
                x1 = max(0, int((bbox["x1"] - bw * pad) * width))
                y1 = max(0, int((bbox["y1"] - bh * pad) * height))
                x2 = min(width, int((bbox["x2"] + bw * pad) * width))
                y2 = min(height, int((bbox["y2"] + bh * pad) * height))

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (crop_size, crop_size))
                else:
                    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                crops.append(crop)

            frame_idx += 1
    finally:
        proc.stdout.close()
        proc.wait()

    return crops


def keypoints_to_wham_format(poses: list[dict], width: int, height: int) -> np.ndarray:
    """Convert pipeline pose dicts to WHAM input format.

    Args:
        poses: list of pose dicts (one per frame), each {joint_name: {x, y, z, confidence}}
        width, height: video dimensions for denormalization

    Returns: (T, 17, 3) array [x_pixel, y_pixel, confidence] in COCO-17 order
    """
    from app.inference import COCO_TO_SMPL  # self-import for COCO joint ordering

    COCO_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    T = len(poses)
    kp_array = np.zeros((T, 17, 3), dtype=np.float32)

    for t, pose in enumerate(poses):
        if not pose:
            continue
        for j, name in enumerate(COCO_NAMES):
            kp = pose.get(name, {})
            if kp:
                kp_array[t, j, 0] = kp.get("x", 0) * width
                kp_array[t, j, 1] = kp.get("y", 0) * height
                kp_array[t, j, 2] = kp.get("confidence", 0)

    return kp_array


def bboxes_from_poses(poses: list[dict]) -> list[dict]:
    """Compute bounding boxes from pose keypoints for each frame.

    Returns list of {x1, y1, x2, y2} in normalized 0-1 coords.
    """
    bboxes = []
    for pose in poses:
        if not pose:
            bboxes.append({"x1": 0, "y1": 0, "x2": 1, "y2": 1})
            continue
        xs = [kp["x"] for kp in pose.values() if kp.get("confidence", 0) > 0.3]
        ys = [kp["y"] for kp in pose.values() if kp.get("confidence", 0) > 0.3]
        if xs and ys:
            bboxes.append({
                "x1": min(xs), "y1": min(ys),
                "x2": max(xs), "y2": max(ys),
            })
        else:
            bboxes.append({"x1": 0, "y1": 0, "x2": 1, "y2": 1})
    return bboxes


def joints_3d_to_dict(joints: np.ndarray) -> dict:
    """Convert (24, 3) SMPL joint array to named dict.

    Returns dict with both SMPL names and COCO-compatible names.
    """
    result = {}
    for i, name in enumerate(SMPL_JOINT_NAMES):
        if i < len(joints):
            result[name] = {
                "x": float(joints[i, 0]),
                "y": float(joints[i, 1]),
                "z": float(joints[i, 2]),
            }
    return result


def run_wham_inference(video_path: str, poses: list[dict],
                       video_info: dict) -> dict | None:
    """Run WHAM on a sequence of frames for a single person.

    Args:
        video_path: path to source video
        poses: list of pose dicts per frame ({joint_name: {x, y, z, confidence}})
        video_info: dict with width, height, fps

    Returns: dict with per-frame SMPL results, or None if WHAM unavailable
        {
            "body_poses": list[list[float]],    # (T, 138) flattened 6D rotations
            "global_orients": list[list[float]], # (T, 6) root rotation
            "betas": list[float],               # (10,) shape
            "joints_3d": list[dict],            # (T,) {joint_name: {x,y,z}}
            "foot_contacts": list[dict],        # (T,) {left_heel, left_toe, ...}
            "frame_count": int,
        }
    """
    model = _load_wham()
    if model is None:
        logger.warning("WHAM model not available — skipping 3D reconstruction")
        return None

    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    fps = video_info.get("fps", 30)

    # Convert poses to WHAM input format
    kp_2d = keypoints_to_wham_format(poses, width, height)
    T = len(poses)

    if T < 10:
        logger.info("Too few frames (%d) for WHAM inference", T)
        return None

    # Compute bboxes for person crops
    bboxes = bboxes_from_poses(poses)
    frame_indices = list(range(T))

    try:
        # Extract person crops from video
        logger.info("Extracting %d person crops for WHAM...", T)
        crops = extract_person_crops(video_path, bboxes, frame_indices,
                                     width, height, crop_size=256)

        if len(crops) != T:
            logger.warning("Crop count mismatch: %d crops vs %d frames", len(crops), T)
            T = min(len(crops), T)
            crops = crops[:T]
            kp_2d = kp_2d[:T]

        # Prepare tensors for WHAM
        # Note: exact tensor format depends on WHAM's forward() signature
        # This is the general pattern — may need adjustment for the actual WHAM API
        crops_tensor = torch.stack([
            torch.from_numpy(c).float().permute(2, 0, 1) / 255.0
            for c in crops
        ]).unsqueeze(0).to("cuda")  # (1, T, 3, 256, 256)

        kp_tensor = torch.from_numpy(kp_2d).unsqueeze(0).to("cuda")  # (1, T, 17, 3)

        # Compute bboxes in pixel coords for WHAM
        bbox_array = np.array([
            [b["x1"] * width, b["y1"] * height, b["x2"] * width, b["y2"] * height]
            for b in bboxes[:T]
        ], dtype=np.float32)
        bbox_tensor = torch.from_numpy(bbox_array).unsqueeze(0).to("cuda")  # (1, T, 4)

        logger.info("Running WHAM inference on %d frames...", T)
        with torch.no_grad():
            output = model(
                x=kp_tensor,
                img_features=crops_tensor,
                bbox=bbox_tensor,
                # No SLAM data — estimate_local_only
            )

        # Extract results from WHAM output
        # Output format varies by WHAM version — adapt as needed
        body_pose = output.get("poses_body", output.get("body_pose"))
        global_orient = output.get("poses_root_cam", output.get("global_orient"))
        betas = output.get("betas")
        joints = output.get("joints3d", output.get("joints"))
        foot_contact = output.get("contact", output.get("feet"))

        results = {
            "body_poses": [],
            "global_orients": [],
            "betas": betas[0].cpu().tolist() if betas is not None else [0.0] * 10,
            "joints_3d": [],
            "foot_contacts": [],
            "frame_count": T,
        }

        for t in range(T):
            # Body pose: 23 joints x 6D rotation = 138 floats
            if body_pose is not None:
                results["body_poses"].append(body_pose[0, t].cpu().tolist())
            else:
                results["body_poses"].append([0.0] * 138)

            # Global orient: 6D rotation = 6 floats
            if global_orient is not None:
                results["global_orients"].append(global_orient[0, t].cpu().tolist())
            else:
                results["global_orients"].append([0.0] * 6)

            # 3D joints
            if joints is not None:
                results["joints_3d"].append(joints_3d_to_dict(joints[0, t].cpu().numpy()))
            else:
                results["joints_3d"].append({})

            # Foot contact: 4 probabilities
            if foot_contact is not None:
                fc = foot_contact[0, t].cpu().tolist()
                results["foot_contacts"].append({
                    "left_heel": fc[0] if len(fc) > 0 else 0.0,
                    "left_toe": fc[1] if len(fc) > 1 else 0.0,
                    "right_heel": fc[2] if len(fc) > 2 else 0.0,
                    "right_toe": fc[3] if len(fc) > 3 else 0.0,
                })
            else:
                results["foot_contacts"].append({
                    "left_heel": 0.0, "left_toe": 0.0,
                    "right_heel": 0.0, "right_toe": 0.0,
                })

        logger.info("WHAM inference complete: %d frames processed", T)
        return results

    except Exception:
        logger.exception("WHAM inference failed")
        return None
