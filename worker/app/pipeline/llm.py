"""Stage 2 — LLM coaching synthesis via Claude API."""
import os
import logging

logger = logging.getLogger(__name__)


def synthesize_coaching_feedback(
    bout_id: int,
    pose_results: list[dict],
    action_results: list[dict],
    db,
) -> str:
    """
    Generate natural language coaching feedback using Claude.
    Stage 2: includes blade tracking confirmation and action classification summary.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM synthesis")
        return "Analysis complete. Set ANTHROPIC_API_KEY to enable AI coaching feedback."

    total_frames = len(pose_results)
    duration_s = round(total_frames / 30, 1)

    # Summarise detected actions
    action_counts: dict[str, int] = {}
    for a in action_results:
        action_counts[a["type"]] = action_counts.get(a["type"], 0) + 1

    action_summary = ", ".join(
        f"{count} {atype}{'s' if count != 1 else ''}"
        for atype, count in sorted(action_counts.items())
    ) or "none detected"

    prompt = f"""You are an expert epee fencing coach analyzing a training bout.

Video summary:
- Total frames analyzed: {total_frames}
- Approximate duration: {duration_s} seconds
- Detected actions: {action_summary}

Stage 2 of the analysis pipeline is now complete:
- Pose skeleton overlay on every frame ✓
- Blade tip trajectory projected from wrist geometry ✓
- Rule-based action classification ✓

Upcoming pipeline additions:
- Depth estimation and 3D blade tracking (Stage 3)
- Kinetic chain readiness scores (Stage 3)
- Attack quality metrics and effective distance penalty (Stage 4)
- Comparison against reference technique skeletons (Stage 5)

Based on the detected actions, provide:
1. A brief acknowledgment that Stage 2 analysis is complete
2. A short observation about the action patterns detected (advances, retreats, lunges)
3. One actionable piece of epee coaching advice specific to the action mix observed
4. An encouraging closing note

Keep the response concise (under 220 words) and coach-like in tone."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    coaching_text = message.content[0].text
    logger.info("LLM synthesis complete for bout %d", bout_id)
    return coaching_text
