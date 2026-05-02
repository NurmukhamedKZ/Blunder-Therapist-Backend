"""Background job: fetch games from Chess.com/Lichess and run full tilt analysis."""
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Game, ImportJob, TiltReport
from app.services.chess_import import RawGame, fetch_chesscom_games, fetch_lichess_games
from app.services.features import extract_features
from app.services.stockfish_service import analyze_pgn
from app.services.tilt_detector import run_tilt_detector

log = structlog.get_logger()


async def run_import(job_id: str) -> None:
    """Entry point for the background import task."""
    try:
        async with AsyncSessionLocal() as db:
            await _run_import(db, job_id)
    except Exception as e:
        log.error("import_job_failed", job_id=job_id, exc_info=True)
        # Open a fresh session — the main session may be in a dirty state
        await _fail_job(job_id, str(e))


async def _run_import(db: AsyncSession, job_id: str) -> None:
    result = await db.execute(select(ImportJob).where(ImportJob.id == job_id))
    job = result.scalar_one()
    job.status = "running"
    await db.commit()

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=job.period_days)

    if job.platform == "chess.com":
        games = await fetch_chesscom_games(job.username, since, until)
    else:
        games = await fetch_lichess_games(job.username, since, until)

    job.total_games = len(games)
    await db.commit()

    if not games:
        job.status = "done"
        job.finished_at = datetime.now(timezone.utc)
        await db.commit()
        return

    for raw in games:
        try:
            await _process_game(db, job, raw)
        except Exception as e:
            log.warning("import_game_skipped", platform_game_id=raw.platform_game_id, error=str(e))
            await db.rollback()  # clear any unflushed state before next game

    job.status = "done"
    job.finished_at = datetime.now(timezone.utc)
    await db.commit()


async def _process_game(db: AsyncSession, job: ImportJob, raw: RawGame) -> None:
    existing = await db.execute(
        select(Game).where(Game.platform_game_id == raw.platform_game_id)
    )
    if existing.scalar_one_or_none():
        job.processed_games += 1
        await db.commit()
        return

    eval_per_ply = raw.eval_per_ply if raw.eval_per_ply is not None else await analyze_pgn(raw.pgn)

    features = extract_features(
        pgn=raw.pgn,
        eval_per_ply=eval_per_ply,
        time_per_ply=raw.time_per_ply,
        player_color=raw.player_color,
        result=raw.result,
    )
    tilt_result = await run_tilt_detector(features)

    game = Game(
        user_id=job.user_id,
        pgn=raw.pgn,
        eval_per_ply=eval_per_ply,
        time_per_ply=raw.time_per_ply,
        player_color=raw.player_color,
        result=raw.result,
        platform_game_id=raw.platform_game_id,
    )
    db.add(game)
    await db.flush()
    db.add(TiltReport(game_id=game.id, **tilt_result))

    job.processed_games += 1
    await db.commit()
    log.info("import_game_done", platform_game_id=raw.platform_game_id, job_id=job.id)


async def _fail_job(job_id: str, error: str) -> None:
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ImportJob).where(ImportJob.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error = error
                job.finished_at = datetime.now(timezone.utc)
                await db.commit()
    except Exception:
        log.error("fail_job_update_failed", job_id=job_id, exc_info=True)
