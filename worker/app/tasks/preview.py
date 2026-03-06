"""
Preview skeleton extraction task.

Extracts a handful of frames from the first few seconds of a video,
runs RTMPose WholeBody pose detection, and saves annotated preview data
so the frontend can show detected skeletons for the user to assign
fencer / opponent roles before the full pipeline runs.
"""
import logging
import os
import subprocess

import cv2
import numpy as np

from app.celery_app import celery_app
from app.db import get_db_session
from app.pipeline.ingest import ingest_video
from app.pipeline.pose import _get_model, keypoints_to_dict, _bbox_from_keypoints, _normalize_score
from app.pipeline.strip import detect_strip

logger = logging.getLogger(__name__)

PREVIEW_DIR = "/app/uploads/previews"
PREVIEW_TIMESTAMPS_MS = [0, 500, 1000, 1500, 2000]
MIN_CONFIDENCE = 0.3


def _extract_frame(video_path: str, timestamp_s: float, width: int, height: int) -> np.ndarray:
    """Extract a single raw BGR frame at the given timestamp using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-ss", str(timestamp_s),
        "-i", video_path,
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    frame_bytes = width * height * 3
    if len(result.stdout) < frame_bytes:
        raise ValueError(
            f"FFmpeg returned {len(result.stdout)} bytes, expected {frame_bytes} "
            f"at t={timestamp_s:.3f}s"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))


@celery_app.task(name="worker.tasks.preview.preview_skeletons", bind=True)
def preview_skeletons(self, bout_id: int, video_path: str):
    """
    Extract preview frames and run RTMPose WholeBody pose detection.

    Saves JPEG images and a structured preview_data dict on the Bout record
    so the frontend can render skeleton overlays for fencer/opponent selection.
    """
    logger.info("Starting preview extraction for bout %d, video: %s", bout_id, video_path)

    with get_db_session() as db:
        from app.models import Bout

        bout = db.query(Bout).get(bout_id)
        if not bout:
            raise ValueError(f"Bout {bout_id} not found")

        try:
            # Stage 1: Get video metadata
            video_info = ingest_video(video_path)
            fps = video_info.get("fps", 30)
            width = video_info.get("width", 1920)
            height = video_info.get("height", 1080)
            duration_s = video_info.get("duration_s", 0)
            duration_ms = duration_s * 1000

            # Filter timestamps to those within the video duration
            timestamps_ms = [ts for ts in PREVIEW_TIMESTAMPS_MS if ts < duration_ms or ts == 0]
            if not timestamps_ms:
                timestamps_ms = [0]

            # Stage 2: Extract frames via FFmpeg
            frames_bgr = []
            for ts_ms in timestamps_ms:
                ts_s = ts_ms / 1000.0
                try:
                    frame = _extract_frame(video_path, ts_s, width, height)
                    frames_bgr.append((ts_ms, frame))
                except Exception as exc:
                    logger.warning(
                        "Failed to extract frame at %dms for bout %d: %s",
                        ts_ms, bout_id, exc,
                    )
                    continue

            if not frames_bgr:
                raise RuntimeError("Could not extract any preview frames from video")

            # Stage 3: Run RTMPose WholeBody on each frame
            model = _get_model()
            os.makedirs(PREVIEW_DIR, exist_ok=True)

            preview_frames = []
            for i, (ts_ms, frame) in enumerate(frames_bgr):
                # Save raw frame as JPEG
                image_key = f"previews/bout_{bout_id}_frame_{i}.jpg"
                image_path = os.path.join("/app/uploads", image_key)
                cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Run pose detection
                all_kps, all_scores = model(frame)
                # all_kps: (N, 133, 2) pixel coords, all_scores: (N, 133)

                detections = []
                n_dets = len(all_kps) if all_kps is not None and len(all_kps) > 0 else 0

                for j in range(n_dets):
                    kps = all_kps[j]
                    sc = all_scores[j]

                    # Filter by mean body keypoint confidence (normalized)
                    body_scores = sc[:17]
                    norm_body = np.array([_normalize_score(float(s)) for s in body_scores])
                    mean_conf = float(norm_body[norm_body > 0.1].mean()) \
                        if (norm_body > 0.1).any() else 0.0
                    if mean_conf < MIN_CONFIDENCE:
                        continue

                    bbox = _bbox_from_keypoints(kps, sc, width, height)
                    if bbox is None:
                        continue

                    keypoints = keypoints_to_dict(kps, sc, width, height)

                    detections.append({
                        "index": j,
                        "bbox": bbox,
                        "confidence": mean_conf,
                        "keypoints": keypoints,
                    })

                preview_frames.append({
                    "frame_index": i,
                    "timestamp_ms": ts_ms,
                    "image_key": image_key,
                    "detections": detections,
                })

            # Stage 4: Detect fencing strip from preview frames
            raw_frames = [frame for _, frame in frames_bgr]
            frame_dets = [pf["detections"] for pf in preview_frames]
            piste = detect_strip(raw_frames, frame_dets, width, height)
            logger.info("Strip detection: method=%s, confidence=%.2f",
                        piste.get("method"), piste.get("confidence", 0))

            # Stage 5: Persist preview data
            preview_data = {"frames": preview_frames, "piste": piste}
            bout.preview_data = preview_data
            bout.status = "preview_ready"
            db.commit()

            logger.info(
                "Preview extraction complete for bout %d: %d frames, %d total detections",
                bout_id,
                len(preview_frames),
                sum(len(f["detections"]) for f in preview_frames),
            )
            return {"bout_id": bout_id, "status": "preview_ready"}

        except Exception as exc:
            logger.exception("Preview extraction failed for bout %d", bout_id)
            bout.status = "failed"
            bout.error = str(exc)
            db.commit()
            raise
