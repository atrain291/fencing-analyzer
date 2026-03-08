from datetime import datetime
from pydantic import AliasChoices, BaseModel, Field


class BladeStateRead(BaseModel):
    model_config = {"from_attributes": True}

    tip_xyz: dict
    nominal_xyz: dict | None
    velocity_xyz: dict
    speed: float | None
    confidence: float | None = None


class FrameRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    timestamp_ms: int
    fencer_pose: dict
    opponent_pose: dict | None
    blade_state: BladeStateRead | None = None


class ActionRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    bout_id: int
    subject: str = "fencer"
    type: str
    start_ms: int
    end_ms: int
    outcome: str | None
    confidence: float | None
    blade_speed_avg: float | None = None
    blade_speed_peak: float | None = None


class AnalysisRead(BaseModel):
    model_config = {"from_attributes": True, "populate_by_name": True}

    id: int
    bout_id: int
    coaching_text: str | None = Field(
        validation_alias=AliasChoices("coaching_text", "llm_summary"),
    )
    patterns: dict | None
    practice_plan: dict | None


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
    analysis: AnalysisRead | None = None


class BoutSummary(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    status: str
    created_at: datetime
    video_url: str | None
    duration_ms: int | None


class BoutUploadResponse(BaseModel):
    bout_id: int
    task_id: str | None
    status: str
