"""Background DNA computation.

Triggered from /api/agent/close-session when total_games is a multiple of 5.
Pulls the user's last 5 games, runs the existing run_decision_dna LLM call,
upserts a DecisionDNA row.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import time
import structlog

from app.models import Game, DecisionDNA
from app.services.features import extract_features
from app.services.dna_decision import run_decision_dna

DNA_TRIGGER_INTERVAL = 5

log = structlog.get_logger()


def should_recompute_dna(total_games: int) -> bool:
    return total_games > 0 and total_games % DNA_TRIGGER_INTERVAL == 0


async def run_dna_for_user(db: AsyncSession, user_id: str) -> None:
    rows = await db.execute(
        select(Game)
        .where(Game.user_id == user_id)
        .order_by(Game.played_at.desc())
        .limit(DNA_TRIGGER_INTERVAL)
    )
    games = rows.scalars().all()
    if len(games) < 3:
        return  # silently skip — defensive, shouldn't happen on the 5-boundary

    t0 = time.monotonic()
    log.info("dna_job_triggered", user_id=user_id, total_games=len(games))

    features = []
    for g in games:
        try:
            features.append(
                extract_features(
                    pgn=g.pgn, eval_per_ply=g.eval_per_ply,
                    time_per_ply=g.time_per_ply, player_color=g.player_color,
                    result=g.result,
                )
            )
        except ValueError:
            continue
    if len(features) < 3:
        return

    try:
        dna = await run_decision_dna(features)
        db.add(DecisionDNA(user_id=user_id, dna=dna, games_count=len(games)))
        await db.commit()
        
        duration_ms = round((time.monotonic() - t0) * 1000)
        log.info("dna_job_done", duration_ms=duration_ms)
    except Exception:
        log.error("dna_job_error", exc_info=True)
        raise
