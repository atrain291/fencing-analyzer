"""Fencer orientation detection — shared utility for blade and action stages."""
import logging

logger = logging.getLogger(__name__)

_CONF_THRESHOLD = 0.3


def detect_orientation(frames: list) -> str:
    """
    Determine fencer facing direction from shoulder/hip x positions.

    Compares average left vs right shoulder+hip x across all valid frames.
    If left body side has higher x on average, fencer faces right (weapon arm
    toward lower x).  Otherwise fencer faces left.

    Returns 'right' or 'left'.
    """
    sum_left_x = 0.0
    sum_right_x = 0.0
    count = 0

    for frame in frames:
        pose = frame.fencer_pose
        if not pose:
            continue

        ls = pose.get("left_shoulder", {})
        rs = pose.get("right_shoulder", {})
        lh = pose.get("left_hip", {})
        rh = pose.get("right_hip", {})

        has_left = (ls.get("confidence", 0) >= _CONF_THRESHOLD
                    or lh.get("confidence", 0) >= _CONF_THRESHOLD)
        has_right = (rs.get("confidence", 0) >= _CONF_THRESHOLD
                     or rh.get("confidence", 0) >= _CONF_THRESHOLD)
        if not (has_left and has_right):
            continue

        left_x = 0.0
        left_n = 0
        if ls.get("confidence", 0) >= _CONF_THRESHOLD:
            left_x += ls["x"]
            left_n += 1
        if lh.get("confidence", 0) >= _CONF_THRESHOLD:
            left_x += lh["x"]
            left_n += 1

        right_x = 0.0
        right_n = 0
        if rs.get("confidence", 0) >= _CONF_THRESHOLD:
            right_x += rs["x"]
            right_n += 1
        if rh.get("confidence", 0) >= _CONF_THRESHOLD:
            right_x += rh["x"]
            right_n += 1

        sum_left_x += left_x / left_n
        sum_right_x += right_x / right_n
        count += 1

    if count == 0:
        return "right"

    avg_left = sum_left_x / count
    avg_right = sum_right_x / count

    orientation = "right" if avg_left > avg_right else "left"
    logger.info("Fencer orientation detected: facing %s (left_x=%.3f, right_x=%.3f, %d samples)",
                orientation, avg_left, avg_right, count)
    return orientation
