"""Post-game chat summarizer.

Reads the user/agent dialogue from a closed game thread and extracts a short
summary plus a list of durable facts (preferences, commitments, recurring
patterns) that should inform future games.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings


SUMMARIZER_SYSTEM = """You are summarizing a coaching conversation about ONE chess game.
Your output is read by a future AI coach when this user starts their NEXT game.

Extract:
- summary: 2-4 sentences, focused on what the USER said, felt, or agreed to.
  Do NOT re-summarize the game itself — there's a separate tilt report for that.
- key_facts: 1-5 short bullet strings of durable facts that future-you should
  remember. Examples: "user prefers short answers", "wants to work on endgames",
  "called this a tilt loss", "plays at night when tired".

If the conversation is empty or trivial, return summary="" and key_facts=[].
"""


class ChatSummaryOutput(BaseModel):
    summary: str = Field(description="2-4 sentences focused on user's words and intent")
    key_facts: list[str] = Field(default_factory=list, description="durable facts to remember")


_summarizer = ChatOpenAI(
    model=settings.model_fast, api_key=settings.openai_api_key, temperature=0.3
).with_structured_output(ChatSummaryOutput)


async def summarize_chat(messages: list[dict]) -> ChatSummaryOutput:
    """Summarize a list of {role, content} messages."""
    if not messages:
        return ChatSummaryOutput(summary="", key_facts=[])
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    return await _summarizer.ainvoke([
        SystemMessage(SUMMARIZER_SYSTEM),
        HumanMessage(transcript)
    ])
