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
