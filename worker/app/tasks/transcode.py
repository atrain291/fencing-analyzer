"""
Transcode video to H.264/MP4 for browser playback.

iPhones and modern cameras typically encode as HEVC (H.265), which Firefox
does not support.  This task produces a browser-friendly H.264 copy while
keeping the original file intact for the ML pipeline.
"""
import logging
import os
import subprocess
import uuid

from app.celery_app import celery_app
from app.db import get_db_session

logger = logging.getLogger(__name__)


def _probe_video(path: str) -> dict:
    """Return codec_name and container format via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-show_entries", "format=format_name",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json
    info = json.loads(result.stdout)
    codec = info.get("streams", [{}])[0].get("codec_name", "unknown")
    fmt = info.get("format", {}).get("format_name", "unknown")
    return {"codec": codec, "format": fmt}


def _needs_faststart(path: str) -> bool:
    """Check if an MP4 file is missing the faststart (moov before mdat) flag."""
    cmd = [
        "ffprobe", "-v", "trace", "-i", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # ffprobe -v trace prints atom positions; if moov comes after mdat, we need faststart
    stderr = result.stderr
    moov_pos = stderr.find("type:'moov'")
    mdat_pos = stderr.find("type:'mdat'")
    if moov_pos == -1 or mdat_pos == -1:
        return False
    return moov_pos > mdat_pos


@celery_app.task(name="worker.tasks.transcode.transcode_for_web", bind=True)
def transcode_for_web(self, bout_id: int, video_path: str):
    """
    Produce a browser-compatible H.264/MP4 from the uploaded video.

    Updates bout.video_url to point to the new file.
    """
    logger.info("Transcode task started for bout %d, input: %s", bout_id, video_path)

    out_name = f"{uuid.uuid4()}_web.mp4"
    out_path = os.path.join("/app/uploads", out_name)

    try:
        probe = _probe_video(video_path)
        codec = probe["codec"]
        fmt = probe["format"]
        logger.info("Bout %d: detected codec=%s, format=%s", bout_id, codec, fmt)

        if codec == "h264" and "mp4" in fmt:
            # Already H.264 in MP4 — just add faststart if missing
            if _needs_faststart(video_path):
                logger.info("Bout %d: H.264/MP4 but needs faststart, remuxing", bout_id)
                cmd = [
                    "ffmpeg", "-v", "error",
                    "-i", video_path,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    out_path, "-y",
                ]
                subprocess.run(cmd, check=True)
            else:
                logger.info("Bout %d: already H.264/MP4 with faststart, skipping transcode", bout_id)
                # No new file needed — keep original video_url
                return {"bout_id": bout_id, "action": "skip"}
        else:
            # Transcode — try GPU (h264_nvenc) first, fall back to libx264
            logger.info("Bout %d: transcoding %s to H.264", bout_id, codec)
            if not _transcode_nvenc(video_path, out_path):
                _transcode_libx264(video_path, out_path)

        # Update DB
        web_url = f"/uploads/{out_name}"
        with get_db_session() as db:
            from app.models import Bout
            bout = db.get(Bout, bout_id)
            if bout:
                bout.video_url = web_url
                db.commit()
                logger.info("Bout %d: video_url updated to %s", bout_id, web_url)

        return {"bout_id": bout_id, "action": "transcoded", "output": out_path}

    except Exception as exc:
        logger.exception("Transcode failed for bout %d", bout_id)
        # Non-fatal: the bout still works, just won't play in Firefox
        return {"bout_id": bout_id, "action": "failed", "error": str(exc)}


def _transcode_nvenc(input_path: str, output_path: str) -> bool:
    """Try GPU-accelerated H.264 encode. Returns True on success."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-hwaccel", "cuda",
        "-i", input_path,
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path, "-y",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("NVENC transcode succeeded")
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("NVENC failed (%s), falling back to libx264", exc.stderr[:200] if exc.stderr else "")
        # Clean up partial output
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def _transcode_libx264(input_path: str, output_path: str):
    """CPU-based H.264 encode as fallback."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path, "-y",
    ]
    subprocess.run(cmd, check=True)
    logger.info("libx264 transcode succeeded")
