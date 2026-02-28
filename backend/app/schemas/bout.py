from datetime import datetime
from pydantic import BaseModel


class BoutRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    session_id: int
    status: str
    result: str | None
    video_url: str | None
    duration_ms: int | None
    pipeline_progress: dict
    created_at: datetime


class BoutUploadResponse(BaseModel):
    bout_id: int
    task_id: str
    status: str
