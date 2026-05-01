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

# ---------- COACH CHAT ----------

COACH_SYSTEM = """You are the user's personal chess coach. You have access
to a memory of their last games (provided in context). When they ask you
questions, ground your answers in their ACTUAL play, not generic chess advice.

Rules:
- Refer to specific games by ply number or critical moment when relevant.
- Be warm but honest. If they're avoiding a weakness, name it.
- If a question is unrelated to their chess, gently redirect.
- Keep responses conversational, 2-4 short paragraphs max.
- If you don't have enough data to answer, say so.
"""


async def run_coach_chat(
    user_message: str,
    history: list[dict],  # [{role, content}, ...]
    game_memory: str,  # pre-formatted summary of recent games
) -> str:
    """Chat with the AI coach. Stateless on the LLM side - we pass full
    history each call.
    """
    messages = [
        {"role": "system", "content": COACH_SYSTEM},
        {
            "role": "system",
            "content": f"USER'S RECENT GAMES (for context):\n\n{game_memory}",
        },
        *history,
        {"role": "user", "content": user_message},
    ]
    response = await client.chat.completions.create(
        model=settings.model_smart,
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )
    return response.choices[0].message.content
