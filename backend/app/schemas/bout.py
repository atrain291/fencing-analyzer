from datetime import datetime
from pydantic import BaseModel


class FrameRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    timestamp_ms: int
    fencer_pose: dict
    opponent_pose: dict | None


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
    frames: list[FrameRead] = []


class BoutUploadResponse(BaseModel):
    bout_id: int
    task_id: str
    status: str
