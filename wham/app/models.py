"""Minimal ORM models for WHAM worker — only what it reads/writes."""
from sqlalchemy import String, Integer, Float, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Frame(Base):
    __tablename__ = "frames"
    id: Mapped[int] = mapped_column(primary_key=True)
    bout_id: Mapped[int] = mapped_column(ForeignKey("bouts.id"))
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    fencer_pose: Mapped[dict] = mapped_column(JSON)
    opponent_pose: Mapped[dict | None] = mapped_column(JSON)


class MeshState(Base):
    __tablename__ = "mesh_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    subject: Mapped[str] = mapped_column(String(20))
    body_pose: Mapped[dict] = mapped_column(JSON)
    global_orient: Mapped[dict] = mapped_column(JSON)
    betas: Mapped[dict] = mapped_column(JSON)
    joints_3d: Mapped[dict] = mapped_column(JSON)
    global_translation: Mapped[dict | None] = mapped_column(JSON)
    foot_contact: Mapped[dict | None] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float)
