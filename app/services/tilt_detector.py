"""
LLM service - the AI brain of Blunder Therapist.

We use OpenAI structured outputs (json_object mode) so the frontend can
trust the response shape. Prompts are the actual product moat - iterate
on them aggressively.
"""
import json
import time
import structlog
from typing import Literal
from openai import AsyncOpenAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.services.features import GameFeatures, features_to_llm_summary


client = AsyncOpenAI(api_key=settings.openai_api_key)

model = ChatOpenAI(model="gpt-5.4-nano", api_key=settings.openai_api_key)

log = structlog.get_logger()

class TiltLLMResponse(BaseModel):
    headline: str = Field(description="one short sentence, max 8 words, the core finding")
    diagnosis: str = Field(description="80-150 word explanation anchored in specific data")
    pattern_label: Literal[
        "tilt", "panic", "overconfidence", "analysis_paralysis",
        "rushing", "frustration", "steady", "focused"
    ]
    evidence_plies: list[int] = Field(description="ply numbers cited in the diagnosis")
    suggestion: str = Field(description="one concrete behavioral suggestion, 1-2 sentences, not a chess tip")

model_structured = model.with_structured_output(TiltLLMResponse)


# ---------- TILT DETECTOR ----------

TILT_DETECTOR_SYSTEM = """You are a chess psychologist (NOT a chess engine).
Your job is to look at a single game's behavioral data and find the ONE most
human-meaningful pattern. You are warm, specific, and grounded in evidence
from the data given to you.

Rules:
- Never give move-by-move chess advice. We have engines for that.
- Always anchor your insight in a SPECIFIC moment from the data (cite ply
  numbers, time spent, eval changes).
- Length: 80-150 words for the diagnosis. Be concise.
- The "pattern" should be a recognizable real-world emotion or habit:
  tilt, panic, overconfidence, indecision, frustration, rushing, freezing.
- Avoid clinical/therapy jargon. Talk like a smart friend.
- If the data shows the player played WELL, say so. Don't invent problems.

Return JSON with exactly these fields:
{
  "headline": "<one short sentence, max 8 words, the core finding>",
  "diagnosis": "<80-150 word explanation, anchored in specific data>",
  "pattern_label": "<one of: tilt, panic, overconfidence, analysis_paralysis,
                    rushing, frustration, steady, focused>",
  "evidence_plies": [<list of ply numbers cited in the diagnosis>],
  "suggestion": "<ONE concrete behavioral suggestion, 1-2 sentences. Not a chess tip.>"
}
"""


async def run_tilt_detector(features: GameFeatures) -> TiltLLMResponse:
    """Run the Tilt Detector on a single game's features."""
    summary = features_to_llm_summary(features)
    t0 = time.monotonic()
    log.info("tilt_start")
    try:
        response = await model_structured.ainvoke([
            SystemMessage(TILT_DETECTOR_SYSTEM),
            HumanMessage(f"Analyze this game.\n\n{summary}")
        ])
        duration_ms = round((time.monotonic() - t0) * 1000)
        log.info("tilt_done", duration_ms=duration_ms)
        return response
    except Exception:
        log.error("tilt_error", exc_info=True)
        raise
