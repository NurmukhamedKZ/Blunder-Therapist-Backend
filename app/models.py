import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    plan: Mapped[str] = mapped_column(String, default="free", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    games: Mapped[list["Game"]] = relationship(back_populates="profile", cascade="all, delete-orphan")


class Game(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("profiles.user_id"), nullable=False)
    pgn: Mapped[str] = mapped_column(String, nullable=False)
    eval_per_ply: Mapped[list] = mapped_column(JSON, nullable=False)
    time_per_ply: Mapped[list] = mapped_column(JSON, nullable=False)
    player_color: Mapped[str] = mapped_column(String, nullable=False)
    result: Mapped[str] = mapped_column(String, nullable=False)
    played_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    profile: Mapped["Profile"] = relationship(back_populates="games")
    tilt_report: Mapped["TiltReport | None"] = relationship(
        back_populates="game", uselist=False, cascade="all, delete-orphan"
    )


class TiltReport(Base):
    __tablename__ = "tilt_reports"
    __table_args__ = (UniqueConstraint("game_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.id"), nullable=False)
    headline: Mapped[str] = mapped_column(String, nullable=False)
    diagnosis: Mapped[str] = mapped_column(String, nullable=False)
    pattern_label: Mapped[str] = mapped_column(String, nullable=False)
    evidence_plies: Mapped[list] = mapped_column(JSON, nullable=False)
    suggestion: Mapped[str] = mapped_column(String, nullable=False)

    game: Mapped["Game"] = relationship(back_populates="tilt_report")