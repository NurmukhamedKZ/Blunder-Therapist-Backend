"""Blunder Therapist FastAPI app."""
import os
from contextlib import asynccontextmanager
from uuid import uuid4 as _uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import CurrentUser, get_current_user, require_pro
from app.models import Game, TiltReport
from app.schemas.api import (
    AnalyzeGameRequest,
    TiltDetectorResponse,
    DecisionDNARequest,
    DecisionDNAResponse,
    CoachChatRequest,
    CoachChatResponse,
)
from app.services.features import extract_features, features_to_llm_summary
from app.services.llm import run_tilt_detector, run_decision_dna, run_coach_chat
from app.routers import games as games_router
from app.routers import agent as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("TESTING"):
        from app import models  # noqa: F401
        from app.database import engine, Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        from app.services.agent import init_agent, shutdown_agent
        await init_agent(settings.database_url, in_memory=False)
        try:
            yield
        finally:
            await shutdown_agent()
    else:
        yield


app = FastAPI(title="Blunder Therapist API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(games_router.router)
app.include_router(agent_router.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "blunder-therapist"}


@app.post("/api/analyze-game", response_model=TiltDetectorResponse)
async def analyze_game(
    req: AnalyzeGameRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run Tilt Detector on a single game; persist game + report."""
    try:
        features = extract_features(
            pgn=req.pgn,
            eval_per_ply=req.eval_per_ply,
            time_per_ply=req.time_per_ply,
            player_color=req.player_color,
            result=req.result,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist the game row first
    game = Game(
        id=req.client_game_id or str(_uuid4()),
        user_id=user.user_id,
        pgn=req.pgn,
        eval_per_ply=req.eval_per_ply,
        time_per_ply=req.time_per_ply,
        player_color=req.player_color,
        result=req.result,
    )
    db.add(game)
    await db.commit()
    await db.refresh(game)

    # Run LLM analysis; if it fails the game row stays (can be re-run)
    try:
        result = await run_tilt_detector(features)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    report = TiltReport(game_id=game.id, **result)
    db.add(report)
    await db.commit()

    return TiltDetectorResponse(**result)


@app.post("/api/decision-dna", response_model=DecisionDNAResponse)
async def decision_dna(
    req: DecisionDNARequest,
    user: CurrentUser = Depends(require_pro),
    db: AsyncSession = Depends(get_db),
):
    """Build Decision DNA from the user's last N stored games."""
    rows = await db.execute(
        select(Game)
        .where(Game.user_id == user.user_id)
        .order_by(Game.played_at.desc())
        .limit(req.n)
    )
    games = rows.scalars().all()
    if len(games) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 games on record")

    try:
        all_features = [
            extract_features(
                pgn=g.pgn,
                eval_per_ply=g.eval_per_ply,
                time_per_ply=g.time_per_ply,
                player_color=g.player_color,
                result=g.result,
            )
            for g in games
        ]
        result = await run_decision_dna(all_features)
        return DecisionDNAResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/coach", response_model=CoachChatResponse)
async def coach_chat(
    req: CoachChatRequest,
    user: CurrentUser = Depends(require_pro),
    db: AsyncSession = Depends(get_db),
):
    """Chat with the memory-aware AI coach using stored game history."""
    rows = await db.execute(
        select(Game)
        .where(Game.user_id == user.user_id)
        .order_by(Game.played_at.desc())
        .limit(10)
    )
    recent_games = rows.scalars().all()

    memory_parts: list[str] = []
    for i, g in enumerate(recent_games):
        try:
            f = extract_features(
                pgn=g.pgn,
                eval_per_ply=g.eval_per_ply,
                time_per_ply=g.time_per_ply,
                player_color=g.player_color,
                result=g.result,
            )
            memory_parts.append(f"--- Game {i + 1} ({f.result}) ---\n{features_to_llm_summary(f)}")
        except ValueError:
            continue

    game_memory = "\n\n".join(memory_parts) if memory_parts else "No games on file yet."
    history = [{"role": t.role, "content": t.content} for t in req.history]
    reply = await run_coach_chat(req.message, history, game_memory)
    return CoachChatResponse(reply=reply)