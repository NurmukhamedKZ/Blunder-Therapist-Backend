"""Tests for import_job orchestrator — mocks platform fetch and Stockfish."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
import pytest

from app.models import ImportJob, Game, TiltReport, Profile
from app.services.chess_import import RawGame
from app.services.import_job import run_import


MINIMAL_PGN = """[Event "Test"]
[White "alice"]
[Black "bob"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

FAKE_EVALS = [20, -30, 50, -200, 100, 300, 100, 100]
FAKE_TIMES = [5.0, 4.0, 3.0, 2.0, 6.0, 1.0, 4.0, 3.0]

FAKE_RAW_GAME = RawGame(
    pgn=MINIMAL_PGN,
    platform="chess.com",
    platform_game_id="chesscom_999",
    player_color="white",
    result="win",
    time_per_ply=FAKE_TIMES,
    eval_per_ply=FAKE_EVALS,
)

FAKE_TILT = {
    "headline": "Tilt detected",
    "diagnosis": "You rushed",
    "pattern_label": "speed_tilt",
    "evidence_plies": [3],
    "suggestion": "Slow down",
}


@pytest.fixture
async def import_job(db):
    profile = Profile(user_id="test-user-id", plan="free")
    db.add(profile)
    job = ImportJob(
        user_id="test-user-id",
        platform="chess.com",
        username="alice",
        period_days=30,
        status="pending",
        total_games=0,
        processed_games=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


def _session_ctx(db):
    """Wrap the test db session as an async context manager."""
    class _Ctx:
        async def __aenter__(self):
            return db
        async def __aexit__(self, *args):
            pass
    return _Ctx()


@pytest.mark.asyncio
async def test_run_import_saves_game_and_report(db, import_job):
    with (
        patch("app.services.import_job.fetch_chesscom_games", new=AsyncMock(return_value=[FAKE_RAW_GAME])),
        patch("app.services.import_job.run_tilt_detector", new=AsyncMock(return_value=FAKE_TILT)),
        patch("app.services.import_job.AsyncSessionLocal", return_value=_session_ctx(db)),
    ):
        await run_import(import_job.id)

    await db.refresh(import_job)
    assert import_job.status == "done"
    assert import_job.total_games == 1
    assert import_job.processed_games == 1

    from sqlalchemy import select
    games = (await db.execute(select(Game).where(Game.platform_game_id == "chesscom_999"))).scalars().all()
    assert len(games) == 1
    reports = (await db.execute(select(TiltReport).where(TiltReport.game_id == games[0].id))).scalars().all()
    assert len(reports) == 1
    assert reports[0].headline == "Tilt detected"


@pytest.mark.asyncio
async def test_run_import_skips_duplicate(db, import_job):
    with (
        patch("app.services.import_job.fetch_chesscom_games", new=AsyncMock(return_value=[FAKE_RAW_GAME, FAKE_RAW_GAME])),
        patch("app.services.import_job.run_tilt_detector", new=AsyncMock(return_value=FAKE_TILT)),
        patch("app.services.import_job.AsyncSessionLocal", return_value=_session_ctx(db)),
    ):
        await run_import(import_job.id)

    from sqlalchemy import select
    games = (await db.execute(select(Game).where(Game.platform_game_id == "chesscom_999"))).scalars().all()
    assert len(games) == 1  # second import skipped


@pytest.mark.asyncio
async def test_run_import_marks_failed_on_fetch_error(db, import_job):
    with (
        patch("app.services.import_job.fetch_chesscom_games", new=AsyncMock(side_effect=ValueError("Username not found on chess.com: alice"))),
        patch("app.services.import_job.AsyncSessionLocal", return_value=_session_ctx(db)),
    ):
        await run_import(import_job.id)

    await db.refresh(import_job)
    assert import_job.status == "failed"
    assert "Username not found" in import_job.error
