"""Stage 2 — Rule-based action classification.

Detected actions:
  advance, retreat, lunge, en_garde, fleche, step_lunge, check_step,
  recovery, preparation
"""
import logging

from app.pipeline.orientation import detect_orientation

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

# Blade speed thresholds (normalized units / second)
_BLADE_ATTACK_THRESHOLD = 0.40   # blade speed indicating genuine attack intent
_BLADE_PREP_THRESHOLD = 0.15     # blade speed below this during foot movement = preparation


def run_action_classification(bout_id: int, frames: list, db,
                              blade_speeds: dict[int, dict] | None = None,
                              opponent_blade_speeds: dict[int, dict] | None = None) -> list[dict]:
    """
    Classify fencing actions from per-frame ankle positions and blade speed.
    Runs classification for both fencer and opponent poses.

    Returns a list of action dicts and writes Action records to the DB.
    """
    from app.models.analysis import Action

    orientation = detect_orientation(frames)

    # Build timestamp->blade_speed lookups
    def _build_blade_lookup(speeds: dict[int, dict] | None) -> dict[int, float]:
        lookup: dict[int, float] = {}
        if speeds:
            for frame in frames:
                bs = speeds.get(frame.id)
                if bs and bs.get("speed") is not None:
                    lookup[frame.timestamp_ms] = bs["speed"]
        return lookup

    fencer_blade_by_ts = _build_blade_lookup(blade_speeds)
    opponent_blade_by_ts = _build_blade_lookup(opponent_blade_speeds)

    all_actions = []

    # Classify fencer actions
    fencer_actions = _classify_subject(frames, orientation, fencer_blade_by_ts, pose_key="fencer_pose")
    for a in fencer_actions:
        a["subject"] = "fencer"
    all_actions.extend(fencer_actions)

    # Classify opponent actions (now with blade data)
    opponent_actions = _classify_subject(frames, orientation, opponent_blade_by_ts, pose_key="opponent_pose")
    for a in opponent_actions:
        a["subject"] = "opponent"
    all_actions.extend(opponent_actions)

    # Write Action records to DB
    for a in all_actions:
        action = Action(
            bout_id=bout_id,
            subject=a["subject"],
            type=a["type"],
            start_ms=a["start_ms"],
            end_ms=a["end_ms"],
            outcome=None,
            confidence=a["confidence"],
            blade_speed_avg=a.get("blade_speed_avg"),
            blade_speed_peak=a.get("blade_speed_peak"),
        )
        db.add(action)

    db.commit()
    logger.info("Action classification complete: %d fencer + %d opponent actions",
                len(fencer_actions), len(opponent_actions))
    return all_actions


def _classify_subject(frames: list, orientation: str,
                      blade_by_ts: dict[int, float],
                      pose_key: str) -> list[dict]:
    """Classify actions for a single subject (fencer or opponent) from their pose data."""
    has_blade = len(blade_by_ts) > 0

    # Determine forward sign — opponent faces opposite direction
    if pose_key == "opponent_pose":
        fwd_sign = 1.0 if orientation == "right" else -1.0
    else:
        fwd_sign = -1.0 if orientation == "right" else 1.0

    # Build position timeline from frames with valid ankle keypoints
    timeline = []
    for frame in frames:
        pose = getattr(frame, pose_key, None) if hasattr(frame, pose_key) else frame.get(pose_key) if isinstance(frame, dict) else None
        if pose is None:
            pose = getattr(frame, pose_key, None)
        if not pose:
            continue

        la = pose.get("left_ankle", {})
        ra = pose.get("right_ankle", {})
        if la.get("confidence", 0) < _CONF_THRESHOLD or ra.get("confidence", 0) < _CONF_THRESHOLD:
            continue

        lax, lay = la["x"], la["y"]
        rax, ray = ra["x"], ra["y"]

        if orientation == "right":
            if lax < rax:
                front_x, back_x = lax, rax
            else:
                front_x, back_x = rax, lax
        else:
            if lax > rax:
                front_x, back_x = lax, rax
            else:
                front_x, back_x = rax, lax

        foot_center_x = (lax + rax) / 2
        stance_width = abs(front_x - back_x)

        lh = pose.get("left_hip", {})
        rh = pose.get("right_hip", {})
        com_x = foot_center_x
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
        return []

    baseline_stance = sum(t["stance_width"] for t in timeline[:10]) / min(10, len(timeline))
    if baseline_stance < 0.01:
        baseline_stance = 0.05

    # Sliding window classification
    raw_actions: list[dict] = []
    current_type: str | None = None
    current_start: int | None = None
    current_conf: float = 0.0

    for i in range(len(timeline) - 1):
        t0 = timeline[i]

        j = i
        while j < len(timeline) - 1 and timeline[j + 1]["ts"] - t0["ts"] < _WINDOW_MS:
            j += 1
        t1 = timeline[j]

        dt = (t1["ts"] - t0["ts"]) / 1000.0
        if dt < 0.05:
            continue

        dx_center = fwd_sign * (t1["foot_center_x"] - t0["foot_center_x"]) / dt
        dx_front = fwd_sign * (t1["front_x"] - t0["front_x"]) / dt
        dx_back = fwd_sign * (t1["back_x"] - t0["back_x"]) / dt
        dx_com = fwd_sign * (t1["com_x"] - t0["com_x"]) / dt

        dx_left = fwd_sign * (t1["lax"] - t0["lax"]) / dt
        dx_right = fwd_sign * (t1["rax"] - t0["rax"]) / dt

        back_crossed_front = (t0["back_x"] != t1["back_x"] and
                              ((orientation == "right" and t1["back_x"] < t1["front_x"]) or
                               (orientation == "left" and t1["back_x"] > t1["front_x"])))

        # Get average blade speed in this window
        window_blade_speed = _avg_blade_speed(blade_by_ts, t0["ts"], t1["ts"])

        # ---- Classification priority (most specific first) ----

        # 1. Fleche: both ankles move forward fast, back ankle crosses front
        if (dx_left > _FLECHE_BOTH_THRESHOLD and dx_right > _FLECHE_BOTH_THRESHOLD
                and back_crossed_front):
            action_type = "fleche"
            confidence = min(1.0, min(dx_left, dx_right) / 0.70)
            if has_blade and window_blade_speed is not None:
                confidence = _blend_blade_confidence(confidence, window_blade_speed)

        # 2. Lunge: front foot extends fast, back foot stays still
        elif dx_front > _LUNGE_FRONT_THRESHOLD and abs(dx_back) < _LUNGE_BACK_THRESHOLD:
            if has_blade and window_blade_speed is not None and window_blade_speed < _BLADE_PREP_THRESHOLD:
                action_type = "preparation"
                confidence = min(1.0, dx_front / 0.70) * 0.85
            else:
                action_type = "lunge"
                confidence = min(1.0, dx_front / 0.70)
                if has_blade and window_blade_speed is not None:
                    confidence = _blend_blade_confidence(confidence, window_blade_speed)

        # 3. Advance: both feet move forward together
        elif dx_center > _ADV_RET_THRESHOLD:
            if has_blade and window_blade_speed is not None and window_blade_speed < _BLADE_PREP_THRESHOLD:
                action_type = "preparation"
                confidence = min(1.0, dx_center / 0.55) * 0.85
            else:
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

    # ---- Third pass: enrich actions with blade metrics ----
    if has_blade:
        raw_actions = _enrich_with_blade(raw_actions, blade_by_ts)

    return raw_actions


def _avg_blade_speed(blade_by_ts: dict[int, float], start_ts: int, end_ts: int) -> float | None:
    """Average blade speed across timestamps in the window."""
    if not blade_by_ts:
        return None
    speeds = [s for ts, s in blade_by_ts.items() if start_ts <= ts <= end_ts]
    return sum(speeds) / len(speeds) if speeds else None


def _blend_blade_confidence(foot_confidence: float, blade_speed: float) -> float:
    """Boost action confidence when blade speed confirms attack intent."""
    blade_factor = min(1.0, blade_speed / _BLADE_ATTACK_THRESHOLD)
    # 70% foot-based, 30% blade-based
    return foot_confidence * 0.7 + blade_factor * 0.3


def _enrich_with_blade(actions: list[dict], blade_by_ts: dict[int, float]) -> list[dict]:
    """Add blade_speed_avg and blade_speed_peak to each action."""
    for a in actions:
        speeds = [s for ts, s in blade_by_ts.items()
                  if a["start_ms"] <= ts <= a["end_ms"]]
        if speeds:
            a["blade_speed_avg"] = sum(speeds) / len(speeds)
            a["blade_speed_peak"] = max(speeds)
    return actions


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

    ts_to_idx = {}
    for idx, t in enumerate(timeline):
        ts_to_idx[t["ts"]] = idx

    def _find_nearest_idx(ts: int) -> int | None:
        if ts in ts_to_idx:
            return ts_to_idx[ts]
        best_idx = None
        best_dist = float("inf")
        for idx, t in enumerate(timeline):
            d = abs(t["ts"] - ts)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

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
                si = _find_nearest_idx(a["start_ms"])
                ei = _find_nearest_idx(a["end_ms"])
                if si is not None and ei is not None:
                    start_stance = timeline[si]["stance_width"]
                    end_stance = timeline[ei]["stance_width"]
                    normalizing = (abs(end_stance - baseline_stance)
                                   < abs(start_stance - baseline_stance) + _RECOVERY_STANCE_TOLERANCE)
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
