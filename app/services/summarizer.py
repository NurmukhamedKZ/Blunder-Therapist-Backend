"""Post-game chat summarizer.

Reads the full agent/user thread (including [OBSERVATION] blunder events and
the injected tilt report) and extracts a deep game analysis plus durable facts
about the player that should inform future sessions.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import time
import structlog

from app.config import settings


SUMMARIZER_SYSTEM = """You are a chess coach analyst summarizing ONE game session.

The transcript contains:
- [OBSERVATION] messages injected by the system (blunder events with ply/eval/time data,
  and a final game_end JSON with the full tilt report).
- USER messages — the player's own chat messages.
- ASSISTANT messages — the coach's responses.

Your job is to extract TWO things:

──────────────────────────────────────────────
1. summary (3-5 sentences)
   Focus on the PLAYER's experience: what they said, how they felt, what they
   acknowledged or committed to. Mention the game result and the main behavioral
   theme (e.g. "time-pressure blunders in the endgame"). Do NOT just restate the
   tilt-report — humanise it.

2. key_facts (3-8 short strings)
   Durable player facts for the next session. Examples:
   "prefers short answers", "wants to improve endgames",
   "called this a tilt loss", "plays at night when tired",
   "gets impatient in drawn positions", "opened with 1.e4 as White".

──────────────────────────────────────────────
3. game_analysis — structured deep analysis of the GAME itself.

   opening: opening name/variation if you can detect it from move data (e.g.
     "Sicilian Defense, Najdorf Variation"), otherwise "".

   game_result: "win" | "loss" | "draw" | "unknown"
     (read from tilt report or user comments).

   blunders: list of every blunder that appears in [OBSERVATION] messages.
     For each blunder extract:
       ply        — integer ply number
       san        — move in SAN notation
       eval_before — centipawn eval before (integer)
       eval_after  — centipawn eval after (integer)
       time_taken  — seconds the player spent (float)
       blunder_type — classify as one of:
           "tactical oversight"   (missed a tactic / hanging piece)
           "time pressure"        (time_taken < 3s AND significant eval drop)
           "calculation error"    (long think followed by a bad move)
           "positional mistake"   (slow eval decay, not a one-move tactic)
           "panic blunder"        (eval was already bad, player rushed)
           "pattern blindness"    (recurring pattern in this game)
       note — 1-sentence plain-English description of what went wrong.

   main_mistakes: 2-5 plain-text strings describing the most important errors
     in the game (can reference specific plies).

   loss_reasons: list of reasons why the player lost (empty if they won/drew).
     Examples: ["lost on time", "blundered a piece on ply 32 when ahead",
                "tilt after early mistake cascaded into three blunders"].

   behavioral_patterns: list of behavioral signals observed in this game.
     Examples: ["plays fast under 5 minutes", "long think before blunder on ply 17",
                "recovered well after early blunder", "tilt pattern: speeds up after mistakes"].

   improvement_areas: 2-4 concrete things this player should focus on next,
     based purely on THIS game's evidence.

──────────────────────────────────────────────
Rules:
- If the conversation is empty/trivial return summary="" key_facts=[] and empty game_analysis.
- Use data from [OBSERVATION] messages for blunder details; do NOT invent eval numbers.
- For blunder_type, prefer "time pressure" only when time_taken < 3 seconds.
- Do NOT repeat the tilt report verbatim in summary.
"""

log = structlog.get_logger()


class BlunderNote(BaseModel):
    ply: int
    san: str
    eval_before: int
    eval_after: int
    time_taken: float
    blunder_type: str
    note: str


class GameAnalysis(BaseModel):
    opening: str = Field(default="", description="opening name/variation or empty string")
    game_result: str = Field(default="unknown", description="win | loss | draw | unknown")
    blunders: list[BlunderNote] = Field(default_factory=list)
    main_mistakes: list[str] = Field(default_factory=list)
    loss_reasons: list[str] = Field(default_factory=list)
    behavioral_patterns: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)


class ChatSummaryOutput(BaseModel):
    summary: str = Field(description="3-5 sentences on player experience and game result")
    key_facts: list[str] = Field(default_factory=list, description="durable facts to remember")
    game_analysis: GameAnalysis = Field(default_factory=GameAnalysis)


_summarizer = ChatOpenAI(
    model=settings.model_fast, api_key=settings.openai_api_key, temperature=0.2
).with_structured_output(ChatSummaryOutput)


async def summarize_chat(messages: list[dict]) -> ChatSummaryOutput:
    """Summarize a list of {role, content} messages (including [OBSERVATION] entries)."""
    if not messages:
        return ChatSummaryOutput(summary="", key_facts=[], game_analysis=GameAnalysis())

    t0 = time.monotonic()
    log.info("summarize_start", message_count=len(messages))

    transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    result = await _summarizer.ainvoke([
        SystemMessage(SUMMARIZER_SYSTEM),
        HumanMessage(f"Here is the full game session transcript:\n\n{transcript}"),
    ])

    duration_ms = round((time.monotonic() - t0) * 1000)
    log.info("summarize_done", duration_ms=duration_ms,
             blunder_count=len(result.game_analysis.blunders))
    return result
