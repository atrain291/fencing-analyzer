from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    fencer_id: Mapped[int] = mapped_column(ForeignKey("fencers.id"))
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    location: Mapped[str | None] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(String(2000))

    fencer: Mapped["Fencer"] = relationship(back_populates="sessions")  # noqa: F821
    bouts: Mapped[list["Bout"]] = relationship(back_populates="session")


class Bout(Base):
    __tablename__ = "bouts"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    opponent_id: Mapped[int | None] = mapped_column(ForeignKey("fencers.id"), nullable=True)
    result: Mapped[str | None] = mapped_column(String(10))  # win / loss / draw
    video_url: Mapped[str | None] = mapped_column(String(1024))
    video_key: Mapped[str | None] = mapped_column(String(1024))  # S3/R2 object key
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    # Processing state
    status: Mapped[str] = mapped_column(String(50), default="pending")
    # pending | queued | processing | complete | failed
    task_id: Mapped[str | None] = mapped_column(String(255))  # Celery task ID
    error: Mapped[str | None] = mapped_column(String(2000))
    pipeline_progress: Mapped[dict] = mapped_column(JSON, default=dict)
    fencer_bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    opponent_bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["Session"] = relationship(back_populates="bouts")
    actions: Mapped[list["Action"]] = relationship(back_populates="bout")  # noqa: F821
    frames: Mapped[list["Frame"]] = relationship(back_populates="bout")  # noqa: F821
    analysis: Mapped["Analysis | None"] = relationship(back_populates="bout")  # noqa: F821
