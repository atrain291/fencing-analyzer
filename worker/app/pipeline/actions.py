"""Stage 2 — Rule-based action classification (advance, retreat, lunge, en_garde)."""
import logging

logger = logging.getLogger(__name__)

_WINDOW_MS = 400        # sliding window for velocity estimation
_MIN_ACTION_MS = 100    # minimum duration to record an action
_CONF_THRESHOLD = 0.3

# Velocity thresholds (normalized units / second)
_ADV_RET_THRESHOLD = 0.30   # foot-centroid speed for advance/retreat
_LUNGE_FRONT_THRESHOLD = 0.45  # front foot extension speed for lunge
_LUNGE_BACK_THRESHOLD = 0.15   # rear foot must stay mostly still


def run_action_classification(bout_id: int, frames: list, db) -> list[dict]:
    """
    Classify fencing actions from per-frame ankle positions.
    Returns a list of action dicts and writes Action records to the DB.
    """
    from app.models.analysis import Action

    # Build position timeline from frames with valid ankle keypoints
    timeline = []
    for frame in frames:
        pose = frame.fencer_pose
        if not pose:
            continue

        la = pose.get("left_ankle", {})
        ra = pose.get("right_ankle", {})
        if la.get("confidence", 0) < _CONF_THRESHOLD or ra.get("confidence", 0) < _CONF_THRESHOLD:
            continue

        lax, lay = la["x"], la["y"]
        rax, ray = ra["x"], ra["y"]

        # Front foot = the one with lower x (closer to opponent, assuming right-facing)
        if lax < rax:
            front_x, back_x = lax, rax
        else:
            front_x, back_x = rax, lax

        foot_center_x = (lax + rax) / 2

        timeline.append({
            "ts": frame.timestamp_ms,
            "foot_center_x": foot_center_x,
            "front_x": front_x,
            "back_x": back_x,
        })

    if len(timeline) < 5:
        logger.info("Action classification: too few frames (%d), skipping", len(timeline))
        return []

    # Sliding window classification
    raw_actions: list[dict] = []
    current_type: str | None = None
    current_start: int | None = None
    current_conf: float = 0.0

    for i in range(len(timeline) - 1):
        t0 = timeline[i]

        # Find window end (~400ms ahead)
        j = i
        while j < len(timeline) - 1 and timeline[j + 1]["ts"] - t0["ts"] < _WINDOW_MS:
            j += 1
        t1 = timeline[j]

        dt = (t1["ts"] - t0["ts"]) / 1000.0
        if dt < 0.05:
            continue

        dx_center = (t1["foot_center_x"] - t0["foot_center_x"]) / dt
        dx_front = (t1["front_x"] - t0["front_x"]) / dt
        dx_back = (t1["back_x"] - t0["back_x"]) / dt

        # Classify
        if dx_front > _LUNGE_FRONT_THRESHOLD and abs(dx_back) < _LUNGE_BACK_THRESHOLD:
            action_type = "lunge"
            confidence = min(1.0, dx_front / 0.70)
        elif dx_center > _ADV_RET_THRESHOLD:
            action_type = "advance"
            confidence = min(1.0, dx_center / 0.55)
        elif dx_center < -_ADV_RET_THRESHOLD:
            action_type = "retreat"
            confidence = min(1.0, abs(dx_center) / 0.55)
        else:
            action_type = "en_garde"
            confidence = 0.8

        if action_type == current_type:
            current_conf = max(current_conf, confidence)
        else:
            # Emit previous action if long enough and not en_garde
            if (current_type and current_type != "en_garde"
                    and current_start is not None
                    and t0["ts"] - current_start >= _MIN_ACTION_MS):
                raw_actions.append({
                    "type": current_type,
                    "start_ms": current_start,
                    "end_ms": t0["ts"],
                    "confidence": current_conf,
                })
            current_type = action_type
            current_start = t0["ts"]
            current_conf = confidence

    # Emit final action
    if (current_type and current_type != "en_garde"
            and current_start is not None
            and timeline[-1]["ts"] - current_start >= _MIN_ACTION_MS):
        raw_actions.append({
            "type": current_type,
            "start_ms": current_start,
            "end_ms": timeline[-1]["ts"],
            "confidence": current_conf,
        })

    # Write Action records to DB
    for a in raw_actions:
        action = Action(
            bout_id=bout_id,
            type=a["type"],
            start_ms=a["start_ms"],
            end_ms=a["end_ms"],
            outcome=None,
            confidence=a["confidence"],
        )
        db.add(action)

    db.commit()
    logger.info("Action classification complete: %d actions detected", len(raw_actions))
    return raw_actions
