"""Stage 2 — Rule-based action classification.

Detected actions:
  advance, retreat, lunge, en_garde, fleche, step_lunge, check_step, recovery
"""
import logging

logger = logging.getLogger(__name__)

_WINDOW_MS = 400        # sliding window for velocity estimation
_MIN_ACTION_MS = 100    # minimum duration to record an action
_CONF_THRESHOLD = 0.3

# Velocity thresholds (normalized units / second)
_ADV_RET_THRESHOLD = 0.30   # foot-centroid speed for advance/retreat
_LUNGE_FRONT_THRESHOLD = 0.45  # front foot extension speed for lunge
_LUNGE_BACK_THRESHOLD = 0.15   # rear foot must stay mostly still

# New action thresholds
_FLECHE_BOTH_THRESHOLD = 0.50  # both ankles moving forward fast
_CHECK_STEP_MAX_MS = 200       # max duration for a check-step
_CHECK_STEP_DIST_RATIO = 0.50  # max distance ratio vs normal advance
_STEP_LUNGE_WINDOW_MS = 300    # time window: advance -> front foot hits lunge speed
_RECOVERY_BACKWARD_THRESHOLD = 0.15  # center of mass moving backward
_RECOVERY_STANCE_TOLERANCE = 0.15    # stance width must normalize (ratio vs initial)


def _detect_orientation(frames: list) -> str:
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

        # Need at least one shoulder and one hip on each side
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
        return "right"  # default assumption

    avg_left = sum_left_x / count
    avg_right = sum_right_x / count

    # If the anatomical left side has higher x, the fencer faces screen-right
    # (their left shoulder is further right on screen).
    orientation = "right" if avg_left > avg_right else "left"
    logger.info("Fencer orientation detected: facing %s (left_x=%.3f, right_x=%.3f, %d samples)",
                orientation, avg_left, avg_right, count)
    return orientation


def run_action_classification(bout_id: int, frames: list, db) -> list[dict]:
    """
    Classify fencing actions from per-frame ankle positions.
    Returns a list of action dicts and writes Action records to the DB.
    """
    from app.models.analysis import Action

    # Detect which direction the fencer faces
    orientation = _detect_orientation(frames)

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

        # Determine front/back foot based on orientation.
        # Facing right: front foot has LOWER x (closer to left edge, toward opponent).
        # Facing left:  front foot has HIGHER x (closer to right edge, toward opponent).
        if orientation == "right":
            if lax < rax:
                front_x, back_x = lax, rax
                front_label, back_label = "left", "right"
            else:
                front_x, back_x = rax, lax
                front_label, back_label = "right", "left"
        else:
            # Facing left: front foot has higher x
            if lax > rax:
                front_x, back_x = lax, rax
                front_label, back_label = "left", "right"
            else:
                front_x, back_x = rax, lax
                front_label, back_label = "right", "left"

        foot_center_x = (lax + rax) / 2
        stance_width = abs(front_x - back_x)

        # Gather hip data for center-of-mass estimate
        lh = pose.get("left_hip", {})
        rh = pose.get("right_hip", {})
        com_x = foot_center_x  # fallback
        hip_count = 0
        hip_sum = 0.0
        if lh.get("confidence", 0) >= _CONF_THRESHOLD:
            hip_sum += lh["x"]
            hip_count += 1
        if rh.get("confidence", 0) >= _CONF_THRESHOLD:
            hip_sum += rh["x"]
            hip_count += 1
        if hip_count > 0:
            com_x = hip_sum / hip_count

        timeline.append({
            "ts": frame.timestamp_ms,
            "foot_center_x": foot_center_x,
            "front_x": front_x,
            "back_x": back_x,
            "lax": lax,
            "rax": rax,
            "stance_width": stance_width,
            "com_x": com_x,
        })

    if len(timeline) < 5:
        logger.info("Action classification: too few frames (%d), skipping", len(timeline))
        return []

    # Compute baseline stance width from first 10 valid frames
    baseline_stance = sum(t["stance_width"] for t in timeline[:10]) / min(10, len(timeline))
    if baseline_stance < 0.01:
        baseline_stance = 0.05  # safety floor

    # Sign convention: "forward" depends on orientation.
    # Facing right: forward = negative dx (decreasing x toward opponent on the left).
    # Facing left:  forward = positive dx (increasing x toward opponent on the right).
    fwd_sign = -1.0 if orientation == "right" else 1.0

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

        # Raw velocities (screen units/sec, sign-adjusted so positive = forward)
        dx_center = fwd_sign * (t1["foot_center_x"] - t0["foot_center_x"]) / dt
        dx_front = fwd_sign * (t1["front_x"] - t0["front_x"]) / dt
        dx_back = fwd_sign * (t1["back_x"] - t0["back_x"]) / dt
        dx_com = fwd_sign * (t1["com_x"] - t0["com_x"]) / dt

        # Both-ankle forward speeds (for fleche detection)
        dx_left = fwd_sign * (t1["lax"] - t0["lax"]) / dt
        dx_right = fwd_sign * (t1["rax"] - t0["rax"]) / dt

        # Back ankle crosses front ankle? (fleche indicator)
        back_crossed_front = (t0["back_x"] != t1["back_x"] and
                              ((orientation == "right" and t1["back_x"] < t1["front_x"]) or
                               (orientation == "left" and t1["back_x"] > t1["front_x"])))

        # ---- Classification priority (most specific first) ----

        # 1. Fleche: both ankles move forward fast, back ankle crosses front
        if (dx_left > _FLECHE_BOTH_THRESHOLD and dx_right > _FLECHE_BOTH_THRESHOLD
                and back_crossed_front):
            action_type = "fleche"
            confidence = min(1.0, min(dx_left, dx_right) / 0.70)

        # 2. Lunge: front foot extends fast, back foot stays still
        elif dx_front > _LUNGE_FRONT_THRESHOLD and abs(dx_back) < _LUNGE_BACK_THRESHOLD:
            action_type = "lunge"
            confidence = min(1.0, dx_front / 0.70)

        # 3. Advance: both feet move forward together
        elif dx_center > _ADV_RET_THRESHOLD:
            action_type = "advance"
            confidence = min(1.0, dx_center / 0.55)

        # 4. Retreat: both feet move backward together
        elif dx_center < -_ADV_RET_THRESHOLD:
            action_type = "retreat"
            confidence = min(1.0, abs(dx_center) / 0.55)

        # 5. En garde: relatively stationary
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

    # ---- Second pass: refine composite actions ----
    raw_actions = _refine_actions(raw_actions, timeline, baseline_stance, fwd_sign)

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


def _refine_actions(actions: list[dict], timeline: list[dict],
                    baseline_stance: float, fwd_sign: float) -> list[dict]:
    """
    Second pass over detected actions to identify composite patterns:
      - step_lunge: advance immediately followed by lunge within 300ms
      - check_step: very short advance (< 200ms, < 50% normal distance)
      - recovery: backward center-of-mass movement + stance normalization after attack
    """
    if not actions:
        return actions

    # Build a quick timestamp->index lookup for the timeline
    ts_to_idx = {}
    for idx, t in enumerate(timeline):
        ts_to_idx[t["ts"]] = idx

    def _find_nearest_idx(ts: int) -> int | None:
        if ts in ts_to_idx:
            return ts_to_idx[ts]
        # Linear scan for nearest
        best_idx = None
        best_dist = float("inf")
        for idx, t in enumerate(timeline):
            d = abs(t["ts"] - ts)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    # Compute median advance distance for check_step comparison
    advance_distances = []
    for a in actions:
        if a["type"] == "advance":
            si = _find_nearest_idx(a["start_ms"])
            ei = _find_nearest_idx(a["end_ms"])
            if si is not None and ei is not None:
                dist = abs(timeline[ei]["foot_center_x"] - timeline[si]["foot_center_x"])
                advance_distances.append(dist)

    median_advance_dist = (sorted(advance_distances)[len(advance_distances) // 2]
                           if advance_distances else 0.1)

    refined: list[dict] = []
    skip_next = False

    for i, a in enumerate(actions):
        if skip_next:
            skip_next = False
            continue

        # --- Step-lunge: advance immediately followed by lunge ---
        if a["type"] == "advance" and i + 1 < len(actions):
            next_a = actions[i + 1]
            gap = next_a["start_ms"] - a["end_ms"]
            if next_a["type"] == "lunge" and gap <= _STEP_LUNGE_WINDOW_MS:
                refined.append({
                    "type": "step_lunge",
                    "start_ms": a["start_ms"],
                    "end_ms": next_a["end_ms"],
                    "confidence": max(a["confidence"], next_a["confidence"]),
                })
                skip_next = True
                continue

        # --- Check-step: very short advance ---
        if a["type"] == "advance":
            duration = a["end_ms"] - a["start_ms"]
            si = _find_nearest_idx(a["start_ms"])
            ei = _find_nearest_idx(a["end_ms"])
            if si is not None and ei is not None:
                dist = abs(timeline[ei]["foot_center_x"] - timeline[si]["foot_center_x"])
                if (duration < _CHECK_STEP_MAX_MS
                        and dist < _CHECK_STEP_DIST_RATIO * median_advance_dist):
                    refined.append({
                        "type": "check_step",
                        "start_ms": a["start_ms"],
                        "end_ms": a["end_ms"],
                        "confidence": a["confidence"] * 0.9,
                    })
                    continue

        # --- Recovery: backward movement after attack, stance normalizes ---
        if a["type"] == "retreat" and i > 0:
            prev_a = refined[-1] if refined else actions[i - 1]
            if prev_a["type"] in ("lunge", "fleche", "step_lunge"):
                # Check that stance width returns toward baseline
                si = _find_nearest_idx(a["start_ms"])
                ei = _find_nearest_idx(a["end_ms"])
                if si is not None and ei is not None:
                    start_stance = timeline[si]["stance_width"]
                    end_stance = timeline[ei]["stance_width"]
                    # Stance is normalizing if end stance is closer to baseline than start
                    normalizing = (abs(end_stance - baseline_stance)
                                   < abs(start_stance - baseline_stance) + _RECOVERY_STANCE_TOLERANCE)
                    # Also check center-of-mass is moving backward
                    dt = (timeline[ei]["ts"] - timeline[si]["ts"]) / 1000.0
                    if dt > 0:
                        com_vel = fwd_sign * (timeline[ei]["com_x"] - timeline[si]["com_x"]) / dt
                        if com_vel < -_RECOVERY_BACKWARD_THRESHOLD and normalizing:
                            refined.append({
                                "type": "recovery",
                                "start_ms": a["start_ms"],
                                "end_ms": a["end_ms"],
                                "confidence": a["confidence"] * 0.85,
                            })
                            continue

        # Default: keep original action
        refined.append(a)

    return refined
