"""Fencing strip (piste) auto-detection from preview frames.

Detects the strip region by sampling floor color near fencers' ankles,
then segmenting the strip surface via LAB color similarity. Falls back
to a geometric estimate from ankle positions if color detection fails.

The detected strip polygon is used downstream to constrain skeleton
tracking — only detections with feet on the strip are considered for
fencer/opponent ID locking.
"""
import math
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Color sampling
_SAMPLE_PATCH_W = 30   # pixels wide
_SAMPLE_PATCH_H = 15   # pixels tall
_SAMPLE_BELOW_PX = 10  # sample below ankles (floor, not leg)

# Segmentation
_L_WEIGHT = 0.3   # luminance weight (low — strip identity is color, not brightness)
_A_WEIGHT = 1.0
_B_WEIGHT = 1.0
_THRESHOLD_MIN = 15.0
_THRESHOLD_MAX = 40.0
_THRESHOLD_MAD_SCALE = 2.5

# Contour scoring
_MIN_AREA_FRACTION = 0.02   # contour must be >= 2% of frame area
_MIN_ASPECT_RATIO = 2.0     # strip must be wider than tall

# Fallback geometric margin (fraction of frame dimension)
_FALLBACK_X_MARGIN = 0.08
_FALLBACK_Y_ABOVE = 0.03    # above ankles
_FALLBACK_Y_BELOW = 0.07    # below ankles (ankles are ON the strip)


def _extract_ankle_positions(detections: list, width: int, height: int) -> list[tuple[int, int]]:
    """Get ankle pixel positions from detection keypoints."""
    ankles = []
    for det in detections:
        kps = det.get("keypoints", {})
        la = kps.get("left_ankle", {})
        ra = kps.get("right_ankle", {})
        for ankle in [la, ra]:
            if ankle.get("confidence", 0) >= 0.3:
                x = ankle.get("x", 0)
                y = ankle.get("y", 0)
                if x > 0 and y > 0:
                    ankles.append((int(x * width), int(y * height)))
    return ankles


def _sample_strip_color(frames_bgr: list[np.ndarray],
                        all_ankles: list[list[tuple[int, int]]]) -> tuple[np.ndarray, float] | None:
    """Sample floor color near ankle positions in LAB space.

    Returns (median_lab, weighted_mad) or None if insufficient samples.
    """
    all_pixels = []

    for frame, ankles in zip(frames_bgr, all_ankles):
        if not ankles:
            continue
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        for ax, ay in ankles:
            # Sample patch below the ankle (floor surface)
            y_start = min(ay + _SAMPLE_BELOW_PX, h - 1)
            y_end = min(y_start + _SAMPLE_PATCH_H, h)
            x_start = max(ax - _SAMPLE_PATCH_W // 2, 0)
            x_end = min(ax + _SAMPLE_PATCH_W // 2, w)

            if y_end <= y_start or x_end <= x_start:
                continue

            patch = lab[y_start:y_end, x_start:x_end]
            all_pixels.append(patch.reshape(-1, 3))

    if not all_pixels:
        return None

    pixels = np.concatenate(all_pixels, axis=0).astype(np.float32)
    if len(pixels) < 20:
        return None

    median_lab = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels - median_lab), axis=0)
    weighted_mad = math.sqrt(
        _L_WEIGHT * mad[0] ** 2 + _A_WEIGHT * mad[1] ** 2 + _B_WEIGHT * mad[2] ** 2
    )

    return median_lab, weighted_mad


def _segment_strip(frame_bgr: np.ndarray, median_lab: np.ndarray,
                   threshold: float, ankle_y_range: tuple[int, int]) -> np.ndarray | None:
    """Segment strip pixels by LAB color distance. Returns binary mask."""
    h, w = frame_bgr.shape[:2]
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    diff = lab - median_lab
    weights = np.array([_L_WEIGHT, _A_WEIGHT, _B_WEIGHT])
    dist = np.sqrt(np.sum(weights * diff ** 2, axis=2))

    mask = (dist < threshold).astype(np.uint8) * 255

    # Morphological cleanup — horizontal bias (strip is wide and short)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Constrain to horizontal band around ankles
    y_min = max(0, ankle_y_range[0] - int(h * 0.10))
    y_max = min(h, ankle_y_range[1] + int(h * 0.15))
    mask[:y_min, :] = 0
    mask[y_max:, :] = 0

    return mask


def _find_best_contour(mask: np.ndarray, ankle_x_range: tuple[int, int],
                       frame_w: int, frame_h: int) -> tuple[np.ndarray, float] | None:
    """Find the contour most likely to be the strip. Returns (contour, score) or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame_w * frame_h

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * _MIN_AREA_FRACTION:
            continue

        rect = cv2.minAreaRect(contour)
        rect_dims = sorted(rect[1])
        if rect_dims[0] < 1:
            continue
        aspect = rect_dims[1] / rect_dims[0]
        if aspect < _MIN_ASPECT_RATIO:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        width_fraction = bw / frame_w

        # Check if contour spans both fencers' ankle x-positions
        contains_fencers = (bx <= ankle_x_range[0] and bx + bw >= ankle_x_range[1])

        score = (
            (area / frame_area) * 0.3
            + min(aspect / 10.0, 1.0) * 0.2
            + width_fraction * 0.2
            + (1.0 if contains_fencers else 0.0) * 0.3
        )

        if score > best_score:
            best_score = score
            best = contour

    if best is None:
        return None
    return best, best_score


def _contour_to_polygon(contour: np.ndarray, width: int, height: int) -> list[list[float]]:
    """Convert contour to a normalized 4-point polygon via minAreaRect."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # Normalize to 0-1 coordinates
    polygon = [[float(p[0] / width), float(p[1] / height)] for p in box]
    return polygon


def _geometric_fallback(all_ankles_flat: list[tuple[int, int]],
                        width: int, height: int) -> dict:
    """Estimate strip bounds purely from ankle positions."""
    if not all_ankles_flat:
        # Absolute fallback: full frame width, middle vertical band
        return {
            "polygon": [[0, 0.35], [1, 0.35], [1, 0.65], [0, 0.65]],
            "confidence": 0.1,
            "method": "geometric_fallback",
        }

    xs = [a[0] for a in all_ankles_flat]
    ys = [a[1] for a in all_ankles_flat]

    x_min = max(0, min(xs) - int(width * _FALLBACK_X_MARGIN))
    x_max = min(width, max(xs) + int(width * _FALLBACK_X_MARGIN))
    y_center = int(np.median(ys))
    y_min = max(0, y_center - int(height * _FALLBACK_Y_ABOVE))
    y_max = min(height, y_center + int(height * _FALLBACK_Y_BELOW))

    polygon = [
        [x_min / width, y_min / height],
        [x_max / width, y_min / height],
        [x_max / width, y_max / height],
        [x_min / width, y_max / height],
    ]

    return {
        "polygon": polygon,
        "confidence": 0.3,
        "method": "geometric_fallback",
    }


def detect_strip(frames_bgr: list[np.ndarray],
                 frame_detections: list[list[dict]],
                 width: int, height: int) -> dict:
    """
    Detect the fencing strip from preview frames.

    Args:
        frames_bgr: list of BGR numpy arrays (the raw preview frames)
        frame_detections: list of detection lists (one per frame), each detection
                          has 'keypoints' dict with ankle positions
        width: frame width in pixels
        height: frame height in pixels

    Returns:
        dict with keys: polygon, confidence, method, and optionally color_lab
    """
    # Collect ankle positions per frame
    all_ankles = [_extract_ankle_positions(dets, width, height) for dets in frame_detections]
    all_ankles_flat = [a for frame_ankles in all_ankles for a in frame_ankles]

    if not all_ankles_flat:
        logger.warning("Strip detection: no valid ankle positions found")
        return _geometric_fallback(all_ankles_flat, width, height)

    # Phase 1: Sample strip color
    color_result = _sample_strip_color(frames_bgr, all_ankles)
    if color_result is None:
        logger.info("Strip detection: insufficient color samples, using geometric fallback")
        return _geometric_fallback(all_ankles_flat, width, height)

    median_lab, weighted_mad = color_result
    threshold = max(_THRESHOLD_MIN, min(_THRESHOLD_MAD_SCALE * weighted_mad, _THRESHOLD_MAX))

    # Compute ankle bounding range for contour scoring
    ankle_xs = [a[0] for a in all_ankles_flat]
    ankle_ys = [a[1] for a in all_ankles_flat]
    ankle_x_range = (min(ankle_xs), max(ankle_xs))
    ankle_y_range = (min(ankle_ys), max(ankle_ys))

    # Phase 2-3: Segment and find contours per frame
    frame_results = []
    for frame_bgr, ankles in zip(frames_bgr, all_ankles):
        if not ankles:
            continue

        mask = _segment_strip(frame_bgr, median_lab, threshold, ankle_y_range)
        if mask is None:
            continue

        result = _find_best_contour(mask, ankle_x_range, width, height)
        if result is not None:
            contour, score = result
            polygon = _contour_to_polygon(contour, width, height)
            frame_results.append((polygon, score))

    # Phase 4: Multi-frame consensus
    if len(frame_results) >= 1:
        # Use the highest-scoring frame's polygon
        best_polygon, best_score = max(frame_results, key=lambda x: x[1])
        confidence = min(1.0, best_score * 1.2)

        # If we have 3+ frames, average the polygons for stability
        if len(frame_results) >= 3:
            polygons = [r[0] for r in frame_results]
            avg_polygon = np.median(polygons, axis=0).tolist()
            best_polygon = avg_polygon
            confidence = min(1.0, confidence * 1.1)

        logger.info("Strip detection: color segmentation succeeded (confidence=%.2f, %d/%d frames)",
                    confidence, len(frame_results), len(frames_bgr))
        return {
            "polygon": best_polygon,
            "confidence": float(confidence),
            "color_lab": [float(v) for v in median_lab],
            "method": "color_segmentation",
        }

    logger.info("Strip detection: color segmentation failed, using geometric fallback")
    return _geometric_fallback(all_ankles_flat, width, height)


def point_in_strip(x: float, y: float, strip: dict, margin: float = 0.02) -> bool:
    """
    Check if a normalized point (x, y) falls within the strip polygon.

    Uses the bounding rect of the polygon with optional margin for tolerance.
    """
    polygon = strip.get("polygon", [])
    if not polygon:
        return True  # no strip data — don't filter

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin

    return x_min <= x <= x_max and y_min <= y <= y_max


def ankles_on_strip(keypoints: dict, strip: dict, margin: float = 0.02) -> bool:
    """
    Check if a detection's ankles are on the strip.

    Returns True if at least one ankle with sufficient confidence is on the strip,
    or if no valid ankles exist (don't filter what we can't verify).
    """
    if not strip or not strip.get("polygon"):
        return True

    la = keypoints.get("left_ankle", {})
    ra = keypoints.get("right_ankle", {})

    checked = 0
    on_strip = 0

    for ankle in [la, ra]:
        if ankle.get("confidence", 0) >= 0.3:
            x, y = ankle.get("x", 0), ankle.get("y", 0)
            if x > 0 and y > 0:
                checked += 1
                if point_in_strip(x, y, strip, margin):
                    on_strip += 1

    if checked == 0:
        return True  # can't verify — don't filter
    return on_strip > 0
