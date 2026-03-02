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
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-select_streams", "v:0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    stream = data["streams"][0]
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])

    nb_packets = stream.get("nb_read_packets")
    total_frames = int(nb_packets) if nb_packets else int(fps * float(stream.get("duration", 0)))

    info = {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration_s": float(stream.get("duration", 0)),
        "codec": stream.get("codec_name"),
        "total_frames": total_frames,
    }
    logger.info("Video info: %s", info)
    return info
