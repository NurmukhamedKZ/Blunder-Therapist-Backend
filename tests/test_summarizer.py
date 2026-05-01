import pytest
from unittest.mock import AsyncMock, patch

from app.services.summarizer import summarize_chat, ChatSummaryOutput


@pytest.mark.asyncio
async def test_summarize_chat_returns_structure():
    fake = ChatSummaryOutput(
        summary="User reflected on rushing in middlegame; agreed to try slower openings.",
        key_facts=["rushes in middlegame", "wants to slow down"],
    )
    mock = AsyncMock(return_value=fake)
    with patch("app.services.summarizer._summarizer", new=AsyncMock(ainvoke=mock)):
        result = await summarize_chat(
            messages=[
                {"role": "user", "content": "I keep blundering on move 20"},
                {"role": "assistant", "content": "You moved fast. Try doubling time."},
            ],
        )
    assert result.summary.startswith("User reflected")
    assert "rushes in middlegame" in result.key_facts


@pytest.mark.asyncio
async def test_summarize_chat_handles_empty_messages():
    result = await summarize_chat(messages=[])
    assert result.summary == ""
    assert result.key_facts == []
