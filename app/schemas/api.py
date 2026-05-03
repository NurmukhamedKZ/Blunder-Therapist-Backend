"""API request/response schemas."""
from datetime import datetime
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
    client_game_id: str | None = None


class TiltDetectorResponse(BaseModel):
    headline: str
    diagnosis: str
    pattern_label: str
    evidence_plies: list[int]
    suggestion: str


# ---------- Decision DNA ----------

class DecisionDNARequest(BaseModel):
    n: int = Field(default=5, ge=3, le=20, description="Number of recent games to analyze")


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


class CoachChatResponse(BaseModel):
    reply: str


# ---------- Game History ----------

class TiltReportOut(BaseModel):
    headline: str
    diagnosis: str
    pattern_label: str
    evidence_plies: list[int]
    suggestion: str


class GameSummary(BaseModel):
    id: str
    player_color: str
    result: str
    played_at: datetime
    tilt_report: TiltReportOut | None
    opponent_name: str | None = None
    platform: str | None = None


class GameDetailResponse(BaseModel):
    id: str
    pgn: str
    eval_per_ply: list[int]
    time_per_ply: list[float]
    player_color: str
    result: str
    played_at: datetime
    tilt_report: TiltReportOut | None
    opponent_name: str | None = None
    platform: str | None = None


class GameListResponse(BaseModel):
    games: list[GameSummary]
    total: int


# ---------- Game Import ----------

class ImportRequest(BaseModel):
    platform: Literal["chess.com", "lichess"]
    username: str
    period_days: Literal[30, 90]


class ImportJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    total_games: int
    processed_games: int
    error: str | None
    finished_at: datetime | None


class ImportJobListResponse(BaseModel):
    jobs: list[ImportJobStatus]