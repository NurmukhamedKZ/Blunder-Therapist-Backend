"""Game history endpoints."""
import re
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import CurrentUser, get_current_user
from app.models import Game, TiltReport
from app.schemas.api import (
    GameDetailResponse,
    GameListResponse,
    GameSummary,
    TiltDetectorResponse,
    TiltReportOut,
)
from app.services.features import extract_features
from app.services.tilt_detector import run_tilt_detector

router = APIRouter(prefix="/api/games", tags=["games"])

log = structlog.get_logger()


def _extract_opponent(pgn: str, player_color: str) -> str:
    if not pgn:
        return "Blunder.Therapist"
    opponent_color = "Black" if player_color == "white" else "White"
    match = re.search(fr'\[{opponent_color}\s+"(.*?)"\]', pgn, re.IGNORECASE)
    if match:
        name = match.group(1)
        if name != "?" and name.strip():
            return name
    return "Blunder.Therapist"


def _get_platform(platform_game_id: str | None) -> str:
    if not platform_game_id:
        return "bot"
    if platform_game_id.startswith("chesscom"):
        return "chess.com"
    if platform_game_id.startswith("lichess"):
        return "lichess"
    return "bot"


def _report_out(report: TiltReport | None) -> TiltReportOut | None:
    if report is None:
        return None
    return TiltReportOut(
        headline=report.headline,
        diagnosis=report.diagnosis,
        pattern_label=report.pattern_label,
        evidence_plies=report.evidence_plies,
        suggestion=report.suggestion,
    )


@router.get("", response_model=GameListResponse)
async def list_games(
    page: int = 1,
    page_size: int = 20,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    total_result = await db.execute(
        select(func.count()).where(Game.user_id == user.user_id)
    )
    total = total_result.scalar_one()

    rows = await db.execute(
        select(Game)
        .options(selectinload(Game.tilt_report))
        .where(Game.user_id == user.user_id)
        .order_by(Game.played_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    games = rows.scalars().all()
    log.info("game_list", page=page, count=len(games))

    return GameListResponse(
        games=[
            GameSummary(
                id=g.id,
                player_color=g.player_color,
                result=g.result,
                played_at=g.played_at,
                tilt_report=_report_out(g.tilt_report),
                opponent_name=_extract_opponent(g.pgn, g.player_color),
                platform=_get_platform(g.platform_game_id),
            )
            for g in games
        ],
        total=total,
    )


@router.get("/{game_id}", response_model=GameDetailResponse)
async def get_game(
    game_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = await db.execute(
        select(Game)
        .options(selectinload(Game.tilt_report))
        .where(Game.id == game_id, Game.user_id == user.user_id)
    )
    game = row.scalar_one_or_none()
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    log.info("game_detail", game_id=game_id)
    return GameDetailResponse(
        id=game.id,
        pgn=game.pgn,
        eval_per_ply=game.eval_per_ply,
        time_per_ply=game.time_per_ply,
        player_color=game.player_color,
        result=game.result,
        played_at=game.played_at,
        tilt_report=_report_out(game.tilt_report),
        opponent_name=_extract_opponent(game.pgn, game.player_color),
        platform=_get_platform(game.platform_game_id),
    )


@router.post("/{game_id}/report", response_model=TiltDetectorResponse)
async def rerun_report(
    game_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-run tilt detector on a stored game (e.g., if LLM failed originally)."""
    log.info("report_requested", game_id=game_id)
    row = await db.execute(
        select(Game)
        .options(selectinload(Game.tilt_report))
        .where(Game.id == game_id, Game.user_id == user.user_id)
    )
    game = row.scalar_one_or_none()
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    try:
        features = extract_features(
            pgn=game.pgn,
            eval_per_ply=game.eval_per_ply,
            time_per_ply=game.time_per_ply,
            player_color=game.player_color,
            result=game.result,
        )
        result = await run_tilt_detector(features)
    except (ValueError, Exception) as e:
        raise HTTPException(status_code=502, detail=str(e))

    if game.tilt_report is not None:
        for key, val in result.items():
            setattr(game.tilt_report, key, val)
    else:
        db.add(TiltReport(game_id=game.id, **result))
    await db.commit()

    return TiltDetectorResponse(**result)