import uuid
from datetime import datetime, timezone

from sqlalchemy import Integer, Text, String, DateTime, JSON, ForeignKey, UniqueConstraint
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
    platform_game_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
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




class GameSummary(Base):
    __tablename__ = "game_summaries"
    __table_args__ = (UniqueConstraint("game_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("profiles.user_id"), nullable=False, index=True)
    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.id"), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_facts: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    game_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ImportJob(Base):
    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("profiles.user_id"), nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String, nullable=False)
    username: Mapped[str] = mapped_column(String, nullable=False)
    period_days: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    total_games: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_games: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DecisionDNA(Base):
    __tablename__ = "decision_dna"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("profiles.user_id"), nullable=False, index=True)
    dna: Mapped[dict] = mapped_column(JSON, nullable=False)
    games_count: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)