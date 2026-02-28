from datetime import datetime, timezone
from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Fencer(Base):
    __tablename__ = "fencers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)

    sessions: Mapped[list["Session"]] = relationship(back_populates="fencer")  # noqa: F821
