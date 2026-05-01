"""LangChain tools the agent can call.

Tools split into two layers:
  - `_*_impl(db, user_id, ...)` — pure async DB function, used by tests.
  - `@tool` wrappers — get db + user_id from ToolRuntime, call the impl.

This split keeps tools testable without spinning up a runtime.
"""
from typing import Any
from langchain.tools import tool, ToolRuntime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal
from app.models import Game, TiltReport, GameSummary

_FILTER_KEYWORDS = {
    "losses": ("result", "loss"),
    "wins": ("result", "win"),
    "draws": ("result", "draw"),
}


async def _list_past_games_impl(
    db: AsyncSession, user_id: str, filter: str | None
) -> list[dict[str, Any]]:
    stmt = (
        select(Game)
        .options(selectinload(Game.tilt_report))
        .where(Game.user_id == user_id)
        .order_by(Game.played_at.desc())
        .limit(20)
    )
    if filter:
        f = filter.lower().strip()
        if f in _FILTER_KEYWORDS:
            col, val = _FILTER_KEYWORDS[f]
            stmt = stmt.where(getattr(Game, col) == val)
    rows = await db.execute(stmt)
    games = rows.scalars().all()

    summaries: dict[str, GameSummary] = {}
    if games:
        sum_rows = await db.execute(
            select(GameSummary).where(GameSummary.game_id.in_([g.id for g in games]))
        )
        summaries = {s.game_id: s for s in sum_rows.scalars().all()}

    return [
        {
            "game_id": g.id,
            "played_at": g.played_at.isoformat() if g.played_at else None,
            "result": g.result,
            "player_color": g.player_color,
            "tilt_pattern": g.tilt_report.pattern_label if g.tilt_report else None,
            "summary_excerpt": (summaries[g.id].summary[:200] if g.id in summaries else None),
        }
        for g in games
    ]


async def _get_game_details_impl(
    db: AsyncSession, user_id: str, game_id: str
) -> dict[str, Any]:
    row = await db.execute(
        select(Game)
        .options(selectinload(Game.tilt_report))
        .where(Game.id == game_id, Game.user_id == user_id)
    )
    game = row.scalar_one_or_none()
    if game is None:
        return {"error": "not found"}

    summary_row = await db.execute(
        select(GameSummary).where(GameSummary.game_id == game_id)
    )
    chat_summary = summary_row.scalar_one_or_none()

    return {
        "game_id": game.id,
        "pgn": game.pgn,
        "eval_per_ply": game.eval_per_ply,
        "time_per_ply": game.time_per_ply,
        "player_color": game.player_color,
        "result": game.result,
        "played_at": game.played_at.isoformat() if game.played_at else None,
        "tilt_report": (
            {
                "headline": game.tilt_report.headline,
                "diagnosis": game.tilt_report.diagnosis,
                "pattern_label": game.tilt_report.pattern_label,
                "evidence_plies": game.tilt_report.evidence_plies,
                "suggestion": game.tilt_report.suggestion,
            }
            if game.tilt_report
            else None
        ),
        "chat_summary": (
            {"summary": chat_summary.summary, "key_facts": chat_summary.key_facts}
            if chat_summary
            else None
        ),
    }


@tool
async def list_past_games(filter: str | None, runtime: ToolRuntime) -> list[dict]:
    """List the user's past games, most recent first (max 20).

    Args:
        filter: optional hint. Supported: "losses", "wins", "draws".
                Unknown values are ignored.

    Returns a list of {game_id, played_at, result, player_color, tilt_pattern,
    summary_excerpt}.
    """
    user_id = runtime.context["user_id"]
    async with AsyncSessionLocal() as db:
        return await _list_past_games_impl(db, user_id, filter)


@tool
async def get_game_details(game_id: str, runtime: ToolRuntime) -> dict:
    """Fetch the full record for one game, including PGN, evals, times,
    tilt report, and chat summary.

    If the game doesn't exist or doesn't belong to the user, returns
    {"error": "not found"}.
    """
    user_id = runtime.context["user_id"]
    async with AsyncSessionLocal() as db:
        return await _get_game_details_impl(db, user_id, game_id)
