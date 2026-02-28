"""Stage 1 — Video ingest and metadata extraction via FFmpeg."""
import subprocess
import json
import logging

logger = logging.getLogger(__name__)


def ingest_video(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    stream = data["streams"][0]
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])

    info = {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration_s": float(stream.get("duration", 0)),
        "codec": stream.get("codec_name"),
    }
    logger.info("Video info: %s", info)
    return info
