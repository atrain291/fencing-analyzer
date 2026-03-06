from datetime import datetime, timezone

from sqlalchemy import String, Integer, Float, Boolean, JSON, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Action(Base):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(primary_key=True)
    bout_id: Mapped[int] = mapped_column(ForeignKey("bouts.id"))
    type: Mapped[str] = mapped_column(String(50))
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float | None] = mapped_column(Float)
    blade_speed_avg: Mapped[float | None] = mapped_column(Float)
    blade_speed_peak: Mapped[float | None] = mapped_column(Float)


class Frame(Base):
    __tablename__ = "frames"
    id: Mapped[int] = mapped_column(primary_key=True)
    bout_id: Mapped[int] = mapped_column(ForeignKey("bouts.id"))
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    fencer_pose: Mapped[dict] = mapped_column(JSON)
    opponent_pose: Mapped[dict | None] = mapped_column(JSON)
    action_id: Mapped[int | None] = mapped_column(ForeignKey("actions.id"), nullable=True)


class BladeState(Base):
    __tablename__ = "blade_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)
    tip_xyz: Mapped[dict] = mapped_column(JSON)
    velocity_xyz: Mapped[dict] = mapped_column(JSON)
    speed: Mapped[float | None] = mapped_column(Float)
    flex_offset_xyz: Mapped[dict | None] = mapped_column(JSON)
    nominal_xyz: Mapped[dict | None] = mapped_column(JSON)
    correction_cost: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class ThreatMetrics(Base):
    __tablename__ = "threat_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)
    distance_to_nearest_target: Mapped[float | None] = mapped_column(Float)
    closing_velocity: Mapped[float | None] = mapped_column(Float)
    correction_cost: Mapped[float | None] = mapped_column(Float)
    attack_viability: Mapped[bool | None] = mapped_column(Boolean)


class KineticState(Base):
    __tablename__ = "kinetic_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)
    launch_readiness: Mapped[float | None] = mapped_column(Float)
    weight_distribution: Mapped[float | None] = mapped_column(Float)
    knee_angles: Mapped[dict | None] = mapped_column(JSON)
    off_arm_orientation: Mapped[dict | None] = mapped_column(JSON)
    recovery_complete: Mapped[bool | None] = mapped_column(Boolean)


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    bout_id: Mapped[int] = mapped_column(ForeignKey("bouts.id"), unique=True)
    technique_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    patterns: Mapped[dict] = mapped_column(JSON, default=dict)
    fatigue_markers: Mapped[dict] = mapped_column(JSON, default=dict)
    llm_summary: Mapped[str | None] = mapped_column(String(8000))
    practice_plan: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
