"""API request/response schemas."""
from typing import Literal
from pydantic import BaseModel, Field


# ---------- Tilt Detector ----------

class AnalyzeGameRequest(BaseModel):
    pgn: str = Field(..., description="Full PGN of the game")
    eval_per_ply: list[int] = Field(
        ...,
        description="Stockfish eval after each ply, in centipawns from White's POV",
    )
    time_per_ply: list[float] = Field(..., description="Seconds spent on each ply")
    player_color: Literal["white", "black"]
    result: Literal["win", "loss", "draw"]


class TiltDetectorResponse(BaseModel):
    headline: str
    diagnosis: str
    pattern_label: str
    evidence_plies: list[int]
    suggestion: str


# ---------- Decision DNA ----------

class DecisionDNARequest(BaseModel):
    games: list[AnalyzeGameRequest] = Field(..., min_length=3)


class GMComparison(BaseModel):
    name: str
    similarity_pct: int
    why: str


class DecisionDNAResponse(BaseModel):
    type_name: str
    tagline: str
    summary: str
    core_strength: str
    core_weakness: str
    gm_comparison: GMComparison


# ---------- Coach Chat ----------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CoachChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = Field(default_factory=list)
    recent_games: list[AnalyzeGameRequest] = Field(default_factory=list)


class CoachChatResponse(BaseModel):
    reply: str
