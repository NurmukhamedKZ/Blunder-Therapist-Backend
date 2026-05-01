"""
LLM service - the AI brain of Blunder Therapist.

We use OpenAI structured outputs (json_object mode) so the frontend can
trust the response shape. Prompts are the actual product moat - iterate
on them aggressively.
"""
import json
from typing import Literal
from openai import AsyncOpenAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.services.features import GameFeatures, features_to_llm_summary


client = AsyncOpenAI(api_key=settings.openai_api_key)

model = ChatOpenAI(model="gpt-5.4-nano", api_key=settings.openai_api_key)

# ---------- DECISION DNA ----------

DECISION_DNA_SYSTEM = """You are profiling a chess player's DECISION-MAKING
STYLE based on aggregate stats from 5+ of their games. This profile should
feel uncannily personal - the kind of thing someone would screenshot and post.

Style guide:
- Be specific. Generic horoscope-language is forbidden.
- Use vivid metaphors. Players are "Aggressive Tacticians", "Patient Architects",
  "Reckless Gamblers", "Defensive Counter-punchers" - invent the type that fits.
- Reference real grandmasters when there's a clear style match (Tal=attacking
  chaos, Karpov=positional squeeze, Carlsen=technique, Fischer=clarity, Petrosian
  =prophylaxis, Kasparov=initiative). Give a similarity percentage.
- One core strength, one core weakness. Both must be falsifiable from data.

Return JSON with exactly:
{
  "type_name": "<2-3 word player archetype, e.g. 'Aggressive Tactician'>",
  "tagline": "<one punchy sentence, max 12 words>",
  "summary": "<80-120 words describing this player's approach>",
  "core_strength": "<one specific strength, 1 sentence>",
  "core_weakness": "<one specific weakness, 1 sentence>",
  "gm_comparison": {
    "name": "<grandmaster name>",
    "similarity_pct": <integer 5-40 - keep humble, never claim 50%+>,
    "why": "<one sentence on the resemblance>"
  }
}
"""


async def run_decision_dna(games: list[GameFeatures]) -> dict:
    """Build a Decision DNA profile from N games."""
    if len(games) < 3:
        raise ValueError("Need at least 3 games for a meaningful DNA profile")

    # Aggregate across games
    total_blunders = sum(g.blunder_count for g in games)
    total_moves = sum(g.total_moves for g in games)
    wins = sum(1 for g in games if g.result == "win")
    losses = sum(1 for g in games if g.result == "loss")
    draws = sum(1 for g in games if g.result == "draw")

    avg_cp_loss = sum(g.avg_centipawn_loss for g in games) / len(games)
    avg_time = sum(g.avg_time_per_move for g in games) / len(games)
    cascade_rate = sum(1 for g in games if g.has_blunder_cascade) / len(games)

    # Phase concentration
    opening_b = sum(g.opening_blunders for g in games)
    middle_b = sum(g.middlegame_blunders for g in games)
    end_b = sum(g.endgame_blunders for g in games)

    # Blunder-speed ratios across games (the psychology signal)
    avg_blunder_speed = sum(
        g.blunder_speed_ratio for g in games if g.blunder_speed_ratio > 0
    ) / max(1, sum(1 for g in games if g.blunder_speed_ratio > 0))

    aggregate = (
        f"AGGREGATE STATS ({len(games)} games)\n"
        f"  Record: {wins}W-{losses}L-{draws}D\n"
        f"  Total blunders: {total_blunders} across {total_moves} moves "
        f"({100 * total_blunders / max(1, total_moves):.1f}% blunder rate)\n"
        f"  Avg centipawn loss per move: {avg_cp_loss:.1f}\n"
        f"  Avg thinking time per move: {avg_time:.1f}s\n"
        f"  Avg blunder-speed ratio: {avg_blunder_speed:.2f}\n"
        f"  Cascade rate (% games with blunder cascades): {100*cascade_rate:.0f}%\n"
        f"  Blunder distribution: opening={opening_b}, middle={middle_b}, "
        f"end={end_b}\n"
    )
    response = await client.chat.completions.create(
        model=settings.model_fast,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DECISION_DNA_SYSTEM},
            {"role": "user", "content": aggregate},
        ],
        temperature=0.85,  # higher creativity for vivid archetype
    )
    return json.loads(response.choices[0].message.content)


