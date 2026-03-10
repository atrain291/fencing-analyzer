import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "fencing_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.video_pipeline", "app.tasks.preview", "app.tasks.transcode",
             "app.tasks.blade_refinement"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one video at a time per worker
)
