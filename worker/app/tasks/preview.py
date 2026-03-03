"""
Preview skeleton extraction task.

Extracts a handful of frames from the first few seconds of a video,
runs YOLO pose detection (predict, not track), and saves annotated
preview data so the frontend can show detected skeletons for the user
to assign fencer / opponent roles before the full pipeline runs.
"""
import logging
import os
import subprocess

import cv2
import numpy as np

from app.celery_app import celery_app
from app.db import get_db_session
from app.pipeline.ingest import ingest_video
from app.pipeline.pose import _get_model, keypoints_to_dict

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
    Extract preview frames and run YOLO pose detection.

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

            # Stage 3: Run YOLO predict on each frame
            model = _get_model()
            os.makedirs(PREVIEW_DIR, exist_ok=True)

            preview_frames = []
            for i, (ts_ms, frame) in enumerate(frames_bgr):
                # Save raw frame as JPEG
                image_key = f"previews/bout_{bout_id}_frame_{i}.jpg"
                image_path = os.path.join("/app/uploads", image_key)
                cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Run pose prediction (not tracking)
                results = model.predict(frame, device="cuda", verbose=False)
                result = results[0]

                detections = []
                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                    and result.keypoints is not None
                ):
                    boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
                    boxes_conf = result.boxes.conf.cpu().numpy()
                    boxes_cls = result.boxes.cls.cpu().numpy()
                    kps_all = result.keypoints.xyn.cpu().numpy()
                    conf_all = (
                        result.keypoints.conf.cpu().numpy()
                        if result.keypoints.conf is not None
                        else None
                    )

                    for j in range(len(result.boxes)):
                        # Filter: only person class (0) with sufficient confidence
                        cls_id = int(boxes_cls[j])
                        confidence = float(boxes_conf[j])
                        if cls_id != 0 or confidence < MIN_CONFIDENCE:
                            continue

                        box = boxes_xyxyn[j]
                        bbox = {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }

                        kps = kps_all[j] if j < len(kps_all) else None
                        conf = conf_all[j] if conf_all is not None and j < len(conf_all) else None
                        keypoints = keypoints_to_dict(kps, conf) if kps is not None else {}

                        detections.append({
                            "index": j,
                            "bbox": bbox,
                            "confidence": confidence,
                            "keypoints": keypoints,
                        })

                preview_frames.append({
                    "frame_index": i,
                    "timestamp_ms": ts_ms,
                    "image_key": image_key,
                    "detections": detections,
                })

            # Stage 4: Persist preview data
            preview_data = {"frames": preview_frames}
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
