"""Stage 2 — LLM coaching synthesis via Claude API."""
import os
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def _build_per_action_breakdown(actions: list[dict]) -> str:
    """Build per-action detail and grouped summary statistics."""
    if not actions:
        return "No actions detected."

    # Sort by start time
    sorted_actions = sorted(actions, key=lambda a: a["start_ms"])

    # Per-action detail lines
    detail_lines = []
    for i, a in enumerate(sorted_actions, 1):
        duration_ms = a["end_ms"] - a["start_ms"]
        start_s = a["start_ms"] / 1000.0
        end_s = a["end_ms"] / 1000.0
        conf = a.get("confidence", 0.0)
        detail_lines.append(
            f"  {i}. {a['type']:10s}  {start_s:5.1f}s - {end_s:5.1f}s  "
            f"({duration_ms}ms)  conf={conf:.2f}"
        )

    # Group by type for summary stats
    by_type: dict[str, list[int]] = defaultdict(list)
    for a in sorted_actions:
        by_type[a["type"]].append(a["end_ms"] - a["start_ms"])

    summary_lines = []
    for atype, durations in sorted(by_type.items()):
        count = len(durations)
        avg_ms = sum(durations) / count
        min_ms = min(durations)
        max_ms = max(durations)
        summary_lines.append(
            f"  {atype}: count={count}, avg={avg_ms:.0f}ms, "
            f"min={min_ms}ms, max={max_ms}ms"
        )

    return (
        "Individual actions (chronological):\n"
        + "\n".join(detail_lines)
        + "\n\nSummary by type:\n"
        + "\n".join(summary_lines)
    )


def _build_blade_metrics(blade_states: list[dict]) -> str:
    """Compute blade tip speed statistics from blade state records."""
    if not blade_states:
        return "No blade tracking data available."

    speeds = [bs["speed"] for bs in blade_states if bs.get("speed") is not None]
    if not speeds:
        return "No blade speed measurements available."

    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    min_speed = min(speeds)

    # Speed distribution buckets (normalized units/second)
    slow = sum(1 for s in speeds if s < 1.0)
    medium = sum(1 for s in speeds if 1.0 <= s < 3.0)
    fast = sum(1 for s in speeds if s >= 3.0)
    total = len(speeds)

    slow_pct = slow / total * 100
    medium_pct = medium / total * 100
    fast_pct = fast / total * 100

    # Tip stability: standard deviation of speed
    mean = avg_speed
    variance = sum((s - mean) ** 2 for s in speeds) / total
    std_dev = math.sqrt(variance)

    return (
        f"Tip speed (normalized units/sec):\n"
        f"  Average: {avg_speed:.2f}\n"
        f"  Max: {max_speed:.2f}\n"
        f"  Min: {min_speed:.2f}\n"
        f"  Std deviation: {std_dev:.2f}\n"
        f"Speed distribution ({total} samples):\n"
        f"  Slow (<1.0):     {slow_pct:5.1f}%  ({slow} frames)\n"
        f"  Medium (1.0-3.0): {medium_pct:5.1f}%  ({medium} frames)\n"
        f"  Fast (>3.0):     {fast_pct:5.1f}%  ({fast} frames)"
    )


def _build_footwork_sequence(actions: list[dict]) -> str:
    """Build chronological footwork movement pattern string."""
    if not actions:
        return "No footwork sequence detected."

    sorted_actions = sorted(actions, key=lambda a: a["start_ms"])
    sequence = [a["type"] for a in sorted_actions]

    # Build readable sequence with timestamps
    seq_parts = []
    for a in sorted_actions:
        t = a["start_ms"] / 1000.0
        seq_parts.append(f"{a['type']}@{t:.1f}s")

    # Detect repeated patterns (e.g., advance-advance-lunge)
    pattern_str = " -> ".join(sequence)

    # Count transitions
    transitions: dict[str, int] = defaultdict(int)
    for i in range(len(sequence) - 1):
        key = f"{sequence[i]} -> {sequence[i+1]}"
        transitions[key] = transitions.get(key, 0) + 1

    transition_lines = []
    for trans, count in sorted(transitions.items(), key=lambda x: -x[1]):
        transition_lines.append(f"  {trans}: {count}x")

    return (
        f"Movement sequence: {pattern_str}\n"
        f"Detailed: {', '.join(seq_parts)}\n"
        f"Transition patterns:\n"
        + ("\n".join(transition_lines) if transition_lines else "  (none)")
    )


def _build_timing_analysis(actions: list[dict], bout_duration_s: float) -> str:
    """Analyze timing: gaps between actions, en_garde vs moving time."""
    if not actions:
        return "No timing data available."

    sorted_actions = sorted(actions, key=lambda a: a["start_ms"])

    # Time spent in classified actions vs total bout duration
    total_action_ms = sum(a["end_ms"] - a["start_ms"] for a in sorted_actions)
    total_action_s = total_action_ms / 1000.0
    en_garde_s = max(0.0, bout_duration_s - total_action_s)

    # Gaps between consecutive actions (recovery / preparation time)
    gaps_ms = []
    for i in range(len(sorted_actions) - 1):
        gap = sorted_actions[i + 1]["start_ms"] - sorted_actions[i]["end_ms"]
        if gap > 0:
            gaps_ms.append(gap)

    gap_section = ""
    if gaps_ms:
        avg_gap = sum(gaps_ms) / len(gaps_ms)
        min_gap = min(gaps_ms)
        max_gap = max(gaps_ms)
        gap_section = (
            f"Gaps between actions ({len(gaps_ms)} intervals):\n"
            f"  Average: {avg_gap:.0f}ms\n"
            f"  Min: {min_gap}ms\n"
            f"  Max: {max_gap}ms\n"
        )
    else:
        gap_section = "Gaps between actions: insufficient data\n"

    # Time from bout start to first action, last action to bout end
    first_action_delay = sorted_actions[0]["start_ms"] / 1000.0
    last_action_end = sorted_actions[-1]["end_ms"] / 1000.0
    trailing_time = max(0.0, bout_duration_s - last_action_end)

    # Tempo: actions per minute
    if bout_duration_s > 0:
        tempo = len(sorted_actions) / (bout_duration_s / 60.0)
    else:
        tempo = 0.0

    return (
        f"Bout duration: {bout_duration_s:.1f}s\n"
        f"Time in actions: {total_action_s:.1f}s ({total_action_s/max(bout_duration_s,0.1)*100:.0f}%)\n"
        f"Time in en_garde: {en_garde_s:.1f}s ({en_garde_s/max(bout_duration_s,0.1)*100:.0f}%)\n"
        f"Tempo: {tempo:.1f} actions/minute\n"
        f"Time to first action: {first_action_delay:.1f}s\n"
        f"Time after last action: {trailing_time:.1f}s\n"
        f"{gap_section}"
    )


def synthesize_coaching_feedback(
    bout_id: int,
    pose_results: list[dict],
    action_results: list[dict],
    blade_states: list[dict],
    db,
) -> str:
    """
    Generate natural language coaching feedback using Claude.
    Stage 2: includes per-action details, blade tracking metrics,
    footwork patterns, and timing analysis.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM synthesis")
        return "Analysis complete. Set ANTHROPIC_API_KEY to enable AI coaching feedback."

    total_frames = len(pose_results)
    fps = 30
    duration_s = round(total_frames / fps, 1) if total_frames > 0 else 0.0

    # Build enriched data sections
    action_breakdown = _build_per_action_breakdown(action_results)
    blade_metrics = _build_blade_metrics(blade_states)
    footwork_sequence = _build_footwork_sequence(action_results)
    timing_analysis = _build_timing_analysis(action_results, duration_s)

    prompt = f"""You are an expert epee fencing coach reviewing a bout analysis. Below is detailed data from computer vision analysis. Use it to provide specific, actionable coaching.

=== BOUT OVERVIEW ===
Total frames: {total_frames}
Duration: {duration_s}s
Frames with blade tracking: {len(blade_states)}

=== ACTION BREAKDOWN ===
{action_breakdown}

=== BLADE TRACKING METRICS ===
{blade_metrics}

=== FOOTWORK PATTERNS ===
{footwork_sequence}

=== TIMING ANALYSIS ===
{timing_analysis}

=== ANALYSIS INSTRUCTIONS ===
Based on the data above, provide coaching feedback on these specific areas:

1. **Footwork tempo and rhythm**: Comment on the action tempo ({_safe_tempo(action_results, duration_s)} actions/min), gaps between actions, and whether the fencer shows good rhythmic variation or is too predictable.

2. **Blade control and tip discipline**: Analyze the tip speed distribution. High % of fast movements may indicate lack of control. Comment on speed consistency (std deviation).

3. **Attack preparation quality**: Look at what happens before lunges — are there preparatory advances? Is there a pattern of advance-advance-lunge (good) vs isolated lunges (risky)?

4. **Recovery patterns after attacks**: Look at what follows lunges — is there a retreat (good recovery) or another advance (over-committing)?

5. **Specific drills**: Recommend 2-3 targeted drills to address the weaknesses identified above. Be specific (e.g., "practice advance-lunge-recover drill at 60bpm metronome").

Keep the response under 350 words, well-structured with the 5 numbered sections above, and coach-like in tone. Reference specific numbers from the data."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    coaching_text = message.content[0].text
    logger.info("LLM synthesis complete for bout %d", bout_id)
    return coaching_text


def _safe_tempo(actions: list[dict], duration_s: float) -> str:
    """Calculate tempo string safely for prompt interpolation."""
    if not actions or duration_s <= 0:
        return "N/A"
    tempo = len(actions) / (duration_s / 60.0)
    return f"{tempo:.1f}"
