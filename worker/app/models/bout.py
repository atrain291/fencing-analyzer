from sqlalchemy import String, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Bout(Base):
    __tablename__ = "bouts"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    task_id: Mapped[str | None] = mapped_column(String(255))
    error: Mapped[str | None] = mapped_column(String(2000))
    pipeline_progress: Mapped[dict] = mapped_column(JSON, default=dict)
    video_key: Mapped[str | None] = mapped_column(String(1024))
    fencer_bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    opponent_bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
