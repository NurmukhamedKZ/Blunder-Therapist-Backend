"""Blunder Therapist FastAPI app."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas.api import (
    AnalyzeGameRequest,
    TiltDetectorResponse,
    DecisionDNARequest,
    DecisionDNAResponse,
    CoachChatRequest,
    CoachChatResponse,
)
from app.services.features import extract_features, features_to_llm_summary
from app.services.llm import (
    run_tilt_detector,
    run_decision_dna,
    run_coach_chat,
)


app = FastAPI(title="Blunder Therapist API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "blunder-therapist"}


@app.post("/api/analyze-game", response_model=TiltDetectorResponse)
async def analyze_game(req: AnalyzeGameRequest):
    """Run Tilt Detector on a single game."""
    try:
        features = extract_features(
            pgn=req.pgn,
            eval_per_ply=req.eval_per_ply,
            time_per_ply=req.time_per_ply,
            player_color=req.player_color,
            result=req.result,
        )
        result = await run_tilt_detector(features)
        return TiltDetectorResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/decision-dna", response_model=DecisionDNAResponse)
async def decision_dna(req: DecisionDNARequest):
    """Build a Decision DNA profile from N games."""
    try:
        all_features = [
            extract_features(
                pgn=g.pgn,
                eval_per_ply=g.eval_per_ply,
                time_per_ply=g.time_per_ply,
                player_color=g.player_color,
                result=g.result,
            )
            for g in req.games
        ]
        result = await run_decision_dna(all_features)
        return DecisionDNAResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/coach", response_model=CoachChatResponse)
async def coach_chat(req: CoachChatRequest):
    """Chat with the memory-aware AI coach."""
    # Build a compact memory of recent games for the LLM context
    memory_parts: list[str] = []
    for i, g in enumerate(req.recent_games[-10:]):  # cap at 10 most recent
        try:
            f = extract_features(
                pgn=g.pgn,
                eval_per_ply=g.eval_per_ply,
                time_per_ply=g.time_per_ply,
                player_color=g.player_color,
                result=g.result,
            )
            memory_parts.append(
                f"--- Game {i+1} ({f.result}) ---\n{features_to_llm_summary(f)}"
            )
        except ValueError:
            continue
    game_memory = "\n\n".join(memory_parts) if memory_parts else "No games on file yet."

    history = [{"role": t.role, "content": t.content} for t in req.history]
    reply = await run_coach_chat(req.message, history, game_memory)
    return CoachChatResponse(reply=reply)
