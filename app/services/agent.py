"""LangChain agent setup, system prompt builder, streaming wrapper."""
from typing import AsyncIterator
import json

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import GameSummary, DecisionDNA
from app.services.agent_tools import list_past_games, get_game_details

BASE_SYSTEM = """You are the user's per-game chess coach inside their game-analysis sidebar.

Behavioral, not tactical:
- Comment on PROCESS (timing, hesitation, panic, recovery), NOT on engine moves.
- If the user asks "what should I play?" or "is this move good?" — politely
  redirect to behavioral framing. We are not an engine.

Style:
- Short, warm, specific. Talk like a smart friend, not a therapist or a textbook.
- When commenting on an in-game observation (e.g. "[OBSERVATION] Ply 17:
  blunder ..."), respond with AT MOST one short paragraph, and ONLY if the
  last thing you said is at least 5 plies ago. Otherwise output nothing.
- When the user explicitly asks something, always reply.

Tools:
- list_past_games: search the user's history. Use when they reference past games.
- get_game_details: pull full data for one specific past game.
"""

NO_MEMORY_NOTE = "No past games yet — this is a fresh user. Don't pretend to remember."


def _format_summaries(summaries: list[GameSummary]) -> str:
    if not summaries:
        return NO_MEMORY_NOTE
    bullets = []
    for s in summaries:
        bullets.append(f"- ({s.created_at.date()}) {s.summary}")
        for fact in s.key_facts:
            bullets.append(f"    • {fact}")
    return "Recent coaching history (oldest → newest):\n" + "\n".join(bullets)


def _format_dna(dna: DecisionDNA | None) -> str:
    if dna is None:
        return "No DNA profile yet (need 5+ games)."
    d = dna.dna
    return (
        f"Player style: {d.get('type_name')} — {d.get('tagline')}.\n"
        f"Strength: {d.get('core_strength')}\n"
        f"Weakness: {d.get('core_weakness')}\n"
        f"Reminds of: {d.get('gm_comparison', {}).get('name')} "
        f"(~{d.get('gm_comparison', {}).get('similarity_pct')}%)."
    )


async def build_system_prompt(db: AsyncSession, user_id: str) -> str:
    sum_rows = await db.execute(
        select(GameSummary)
        .where(GameSummary.user_id == user_id)
        .order_by(GameSummary.created_at.desc())
        .limit(5)
    )
    summaries = list(reversed(sum_rows.scalars().all()))  # oldest -> newest

    dna_rows = await db.execute(
        select(DecisionDNA)
        .where(DecisionDNA.user_id == user_id)
        .order_by(DecisionDNA.computed_at.desc())
        .limit(1)
    )
    dna = dna_rows.scalar_one_or_none()

    return (
        BASE_SYSTEM
        + "\n\n=== MEMORY ===\n"
        + _format_summaries(summaries)
        + "\n\n=== DNA ===\n"
        + _format_dna(dna)
    )


# --- Agent singleton ---

_checkpointer: AsyncPostgresSaver | MemorySaver | None = None
_agent = None
_pg_cm = None  # async context manager handle for AsyncPostgresSaver.from_conn_string


async def init_agent(database_url: str, *, in_memory: bool = False) -> None:
    global _checkpointer, _agent, _pg_cm
    if in_memory:
        _checkpointer = MemorySaver()
    else:
        # AsyncPostgresSaver expects a sync libpq URL (psycopg). Strip async driver.
        pg_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        _pg_cm = AsyncPostgresSaver.from_conn_string(pg_url)
        _checkpointer = await _pg_cm.__aenter__()
        await _checkpointer.setup()

    _agent = create_agent(
        model=ChatOpenAI(
            model=settings.model_smart,
            api_key=settings.openai_api_key,
            temperature=0.6,
        ),
        tools=[list_past_games, get_game_details],
        checkpointer=_checkpointer,
    )


async def shutdown_agent() -> None:
    global _pg_cm
    if _pg_cm is not None:
        await _pg_cm.__aexit__(None, None, None)
        _pg_cm = None


def get_agent():
    if _agent is None:
        raise RuntimeError("Agent not initialized — call init_agent in lifespan")
    return _agent


def get_checkpointer():
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized")
    return _checkpointer


async def stream_agent_response(
    thread_id: str,
    user_id: str,
    jwt: str,
    new_messages: list[BaseMessage],
    seed_system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings (event/data lines) for the agent's response."""
    config = {"configurable": {"thread_id": thread_id}}
    context = {"user_id": user_id, "jwt": jwt}

    if seed_system_prompt is not None:
        new_messages = [SystemMessage(seed_system_prompt), *new_messages]

    async for chunk, _meta in get_agent().astream(
        {"messages": new_messages},
        config=config,
        context=context,
        stream_mode="messages",
    ):
        text = getattr(chunk, "content", None)
        if isinstance(text, str) and text:
            yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
    yield "event: done\ndata: {}\n\n"


async def thread_has_state(thread_id: str) -> bool:
    config = {"configurable": {"thread_id": thread_id}}
    state = await get_checkpointer().aget(config)
    return state is not None and bool(state.get("channel_values", {}).get("messages"))
