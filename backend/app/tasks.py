"""Thin shim: sends tasks to the Celery worker queue."""
from celery import Celery
from app.config import settings

celery_app = Celery("fencing", broker=settings.redis_url, backend=settings.redis_url)


def dispatch_pipeline(bout_id: int, video_path: str):
    """Dispatch the video processing pipeline to the worker queue."""
    return celery_app.send_task(
        "worker.tasks.video_pipeline.run_pipeline",
        args=[bout_id, video_path],
        queue="video_pipeline",
    )


def dispatch_preview(bout_id: int, video_path: str):
    """Dispatch the skeleton preview task to the worker queue."""
    return celery_app.send_task(
        "worker.tasks.preview.preview_skeletons",
        args=[bout_id, video_path],
        queue="video_pipeline",
    )


def dispatch_transcode(bout_id: int, video_path: str):
    """Dispatch H.264 transcode for browser-compatible playback."""
    return celery_app.send_task(
        "worker.tasks.transcode.transcode_for_web",
        args=[bout_id, video_path],
        queue="video_pipeline",
    )
