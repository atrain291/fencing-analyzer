"""Stage 1 — LLM coaching synthesis via Claude API."""
import os
import logging

logger = logging.getLogger(__name__)


def synthesize_coaching_feedback(bout_id: int, pose_results: list[dict], db) -> str:
    """
    Generate natural language coaching feedback using Claude.
    In Stage 1 this is based on pose summary statistics only.
    Later stages will ground this in blade tracking, threat metrics, etc.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM synthesis")
        return "Analysis complete. Set ANTHROPIC_API_KEY to enable AI coaching feedback."

    total_frames = len(pose_results)
    duration_s = round(total_frames / 30, 1)  # approximate

    prompt = f"""You are an expert epee fencing coach analyzing a training bout.

Video summary:
- Total frames analyzed: {total_frames}
- Approximate duration: {duration_s} seconds

This is Stage 1 of the analysis pipeline. Full biomechanical data (blade tracking, kinetic chain,
threat metrics) will be available in later pipeline stages.

Based on the fact that pose estimation ran successfully on this video:
1. Briefly acknowledge the analysis is complete
2. Explain what the system detected (pose skeleton overlay on every frame)
3. List what additional analysis will be added as the pipeline matures:
   - Blade tracking and tip trajectory
   - Kinetic chain readiness scores
   - Attack quality metrics and effective distance penalty
   - Comparison against reference technique skeletons
4. Provide one piece of general epee coaching advice relevant to beginners

Keep the response concise (under 200 words) and encouraging."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    coaching_text = message.content[0].text
    logger.info("LLM synthesis complete for bout %d", bout_id)
    return coaching_text
