from datetime import datetime
from pydantic import BaseModel


class BladeStateRead(BaseModel):
    model_config = {"from_attributes": True}

    tip_xyz: dict
    nominal_xyz: dict | None
    velocity_xyz: dict
    speed: float | None


class FrameRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    timestamp_ms: int
    fencer_pose: dict
    opponent_pose: dict | None
    blade_state: BladeStateRead | None = None


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
    task_id: str | None
    status: str
