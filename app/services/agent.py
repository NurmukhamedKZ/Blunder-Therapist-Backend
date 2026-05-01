"""AgentService — singleton LangGraph agent with typed context and prompt cache."""
from __future__ import annotations

import json
from typing import AsyncIterator

from langchain.agents import create_agent
from langchain.agents.middleware.types import dynamic_prompt, ModelRequest
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.models import GameSummary, DecisionDNA
from app.services.agent_tools import AgentContext, list_past_games, get_game_details

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
    """Build the full personalised system prompt for a user. Used by tests and middleware."""
    sum_rows = await db.execute(
        select(GameSummary)
        .where(GameSummary.user_id == user_id)
        .order_by(GameSummary.created_at.desc())
        .limit(5)
    )
    summaries = list(reversed(sum_rows.scalars().all()))

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


async def _build_prompt_for_user(user_id: str) -> str:
    """Open a fresh DB session and build the prompt. Used by the middleware closure."""
    async with AsyncSessionLocal() as db:
        return await build_system_prompt(db, user_id)


class AgentService:
    def __init__(self) -> None:
        self._agent = None
        self._checkpointer: AsyncPostgresSaver | MemorySaver | None = None
        self._pg_cm = None
        self._prompt_cache: dict[str, str] = {}

    async def init(self, database_url: str, *, in_memory: bool = False) -> None:
        if in_memory:
            self._checkpointer = MemorySaver()
        else:
            pg_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
            self._pg_cm = AsyncPostgresSaver.from_conn_string(pg_url)
            self._checkpointer = await self._pg_cm.__aenter__()
            await self._checkpointer.setup()

        @dynamic_prompt
        async def _personalized_prompt(request: ModelRequest[AgentContext]) -> str:
            thread_id = (
                request.runtime.execution_info.thread_id
                if request.runtime and request.runtime.execution_info
                else None
            )
            if thread_id and thread_id in self._prompt_cache:
                return self._prompt_cache[thread_id]
            user_id = request.runtime.context.user_id
            prompt = await _build_prompt_for_user(user_id)
            if thread_id:
                self._prompt_cache[thread_id] = prompt
            return prompt

        self._agent = create_agent(
            model=ChatOpenAI(
                model=settings.model_smart,
                api_key=settings.openai_api_key,
                temperature=0.6,
            ),
            tools=[list_past_games, get_game_details],
            checkpointer=self._checkpointer,
            context_schema=AgentContext,
            middleware=[_personalized_prompt],
        )

    async def shutdown(self) -> None:
        if self._pg_cm is not None:
            await self._pg_cm.__aexit__(None, None, None)
            self._pg_cm = None

    async def stream(
        self,
        thread_id: str,
        context: AgentContext,
        messages: list[BaseMessage],
    ) -> AsyncIterator[str]:
        """Yield SSE-formatted strings (event/data lines) for the agent's response."""
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk, _meta in self._agent.astream(
            {"messages": messages},
            config=config,
            context=context,
            stream_mode="messages",
        ):
            text = getattr(chunk, "content", None)
            if isinstance(text, str) and text:
                yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
        yield "event: done\ndata: {}\n\n"

    async def has_state(self, thread_id: str) -> bool:
        """Returns True if the checkpointer has messages for this thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = await self._checkpointer.aget(config)
        return state is not None and bool(state.get("channel_values", {}).get("messages"))

    async def get_messages(self, thread_id: str) -> list[BaseMessage]:
        """Returns all stored messages for the thread (for close-session and history)."""
        config = {"configurable": {"thread_id": thread_id}}
        state = await self._checkpointer.aget(config)
        if state is None:
            return []
        if isinstance(state, dict):
            return state.get("channel_values", {}).get("messages", [])
        return state.get("messages", []) or []


agent_service = AgentService()