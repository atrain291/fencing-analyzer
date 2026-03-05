from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey, Integer, Float, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    bout_id: Mapped[int] = mapped_column(ForeignKey("bouts.id"))
    type: Mapped[str] = mapped_column(String(50))  # lunge, advance, parry, etc.
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str | None] = mapped_column(String(50))  # touch, miss, parried
    confidence: Mapped[float | None] = mapped_column(Float)

    bout: Mapped["Bout"] = relationship(back_populates="actions")  # noqa: F821


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    bout_id: Mapped[int] = mapped_column(ForeignKey("bouts.id"))
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    fencer_pose: Mapped[dict] = mapped_column(JSON)   # 33 keypoints {x,y,z,confidence}
    opponent_pose: Mapped[dict | None] = mapped_column(JSON)
    action_id: Mapped[int | None] = mapped_column(ForeignKey("actions.id"), nullable=True)

    bout: Mapped["Bout"] = relationship(back_populates="frames")  # noqa: F821
    blade_state: Mapped["BladeState | None"] = relationship(back_populates="frame", cascade="all, delete-orphan")
    threat_metrics: Mapped["ThreatMetrics | None"] = relationship(back_populates="frame", cascade="all, delete-orphan")
    kinetic_state: Mapped["KineticState | None"] = relationship(back_populates="frame", cascade="all, delete-orphan")


class BladeState(Base):
    __tablename__ = "blade_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)
    tip_xyz: Mapped[dict] = mapped_column(JSON)          # {x, y, z}
    velocity_xyz: Mapped[dict] = mapped_column(JSON)     # {x, y, z}
    speed: Mapped[float | None] = mapped_column(Float)   # m/s
    flex_offset_xyz: Mapped[dict | None] = mapped_column(JSON)
    nominal_xyz: Mapped[dict | None] = mapped_column(JSON)
    correction_cost: Mapped[float | None] = mapped_column(Float)  # 0.0–1.0
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    frame: Mapped["Frame"] = relationship(back_populates="blade_state")


class ThreatMetrics(Base):
    __tablename__ = "threat_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)
    distance_to_nearest_target: Mapped[float | None] = mapped_column(Float)
    distance_to_optimal_target: Mapped[float | None] = mapped_column(Float)
    closing_velocity: Mapped[float | None] = mapped_column(Float)
    correction_angle: Mapped[float | None] = mapped_column(Float)
    correction_cost: Mapped[float | None] = mapped_column(Float)
    projected_arrival_xyz: Mapped[dict | None] = mapped_column(JSON)
    attack_viability: Mapped[bool | None] = mapped_column(Boolean)

    frame: Mapped["Frame"] = relationship(back_populates="threat_metrics")


class KineticState(Base):
    __tablename__ = "kinetic_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"), unique=True)
    launch_readiness: Mapped[float | None] = mapped_column(Float)   # 0–100
    weight_distribution: Mapped[float | None] = mapped_column(Float)  # -1.0 back to 1.0 forward
    knee_angles: Mapped[dict | None] = mapped_column(JSON)           # {left, right}
    off_arm_orientation: Mapped[dict | None] = mapped_column(JSON)
    spine_angle: Mapped[float | None] = mapped_column(Float)
    recovery_complete: Mapped[bool | None] = mapped_column(Boolean)

    frame: Mapped["Frame"] = relationship(back_populates="kinetic_state")


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
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    bout: Mapped["Bout"] = relationship(back_populates="analysis")  # noqa: F821
