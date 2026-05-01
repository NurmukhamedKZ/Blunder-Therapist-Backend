import pytest
import json
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import select
from app.models import Profile, Game, TiltReport, GameSummary, DecisionDNA
from app.services.agent import agent_service


async def _drain_sse(response) -> list[dict]:
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_observe_game_start_returns_done(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    await db.commit()

    async with authed_client.stream(
        "POST", "/api/agent/observe",
        json={"thread_id": "g-1", "event": "game_start", "payload": {}},
    ) as r:
        assert r.status_code == 200
        body = await r.aread()
        assert b"event: done" in body


@pytest.mark.asyncio
async def test_observe_blunder_streams_token(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    await db.commit()

    async def fake_stream(thread_id, ctx, messages):
        yield 'event: token\ndata: {"text": "Hmm"}\n\n'
        yield 'event: done\ndata: {}\n\n'

    with patch.object(agent_service, "stream", new=fake_stream):
        async with authed_client.stream(
            "POST", "/api/agent/observe",
            json={"thread_id": "g-1", "event": "blunder",
                  "payload": {"ply": 17, "san": "Qxh7", "eval_before": 50,
                              "eval_after": -250, "time_taken": 4.0}},
        ) as r:
            body = await r.aread()
            assert b'"text": "Hmm"' in body


@pytest.mark.asyncio
async def test_observe_game_end_loads_tilt_report(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    g = Game(user_id="test-user-id", pgn="x", eval_per_ply=[0],
             time_per_ply=[1.0], player_color="white", result="loss")
    db.add(g)
    await db.flush()
    db.add(TiltReport(game_id=g.id, headline="Rushed in middlegame",
                      diagnosis="d", pattern_label="rushing",
                      evidence_plies=[10], suggestion="s"))
    await db.commit()

    captured = {}

    async def fake_stream(thread_id, ctx, messages):
        captured["msgs"] = messages
        yield 'event: done\ndata: {}\n\n'

    with patch.object(agent_service, "stream", new=fake_stream):
        async with authed_client.stream(
            "POST", "/api/agent/observe",
            json={"thread_id": g.id, "event": "game_end",
                  "payload": {"game_id": g.id}},
        ) as r:
            await r.aread()

    assert "Rushed in middlegame" in captured["msgs"][0].content


@pytest.mark.asyncio
async def test_message_streams(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    await db.commit()

    async def fake_stream(thread_id, ctx, messages):
        yield 'event: token\ndata: {"text": "hi"}\n\n'
        yield 'event: done\ndata: {}\n\n'

    with patch.object(agent_service, "stream", new=fake_stream):
        async with authed_client.stream(
            "POST", "/api/agent/message",
            json={"thread_id": "g-1", "text": "what do you think?"},
        ) as r:
            body = await r.aread()
            assert b'"text": "hi"' in body


@pytest.mark.asyncio
async def test_close_session_writes_summary(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    g = Game(user_id="test-user-id", pgn="x", eval_per_ply=[0],
             time_per_ply=[1.0], player_color="white", result="win")
    db.add(g)
    await db.commit()

    fake_msgs = [HumanMessage("hey")]

    from app.services.summarizer import ChatSummaryOutput
    summary_obj = ChatSummaryOutput(summary="we chatted", key_facts=["a"])

    with patch.object(agent_service, "get_messages", new=AsyncMock(return_value=fake_msgs)), \
         patch("app.routers.agent.summarize_chat", new=AsyncMock(return_value=summary_obj)):
        r = await authed_client.post(
            "/api/agent/close-session",
            json={"thread_id": g.id},
        )
        assert r.status_code == 204

    rows = await db.execute(select(GameSummary).where(GameSummary.game_id == g.id))
    saved = rows.scalar_one()
    assert saved.summary == "we chatted"
    assert saved.key_facts == ["a"]


@pytest.mark.asyncio
async def test_close_session_idempotent(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    g = Game(user_id="test-user-id", pgn="x", eval_per_ply=[0],
             time_per_ply=[1.0], player_color="white", result="win")
    db.add(g)
    await db.flush()
    db.add(GameSummary(user_id="test-user-id", game_id=g.id,
                       summary="existing", key_facts=[]))
    await db.commit()

    r = await authed_client.post("/api/agent/close-session",
                                 json={"thread_id": g.id})
    assert r.status_code == 204
    rows = await db.execute(select(GameSummary).where(GameSummary.game_id == g.id))
    assert len(rows.scalars().all()) == 1


@pytest.mark.asyncio
async def test_close_session_triggers_dna_at_5(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    games = []
    for _ in range(5):
        g = Game(user_id="test-user-id", pgn="1.e4 e5", eval_per_ply=[0, 30],
                 time_per_ply=[1.0, 1.0], player_color="white", result="win")
        db.add(g)
        games.append(g)
    await db.commit()
    await db.refresh(games[-1])

    with patch.object(agent_service, "get_messages", new=AsyncMock(return_value=[])), \
         patch("app.routers.agent.run_dna_for_user", new=AsyncMock()) as dna_mock:
        r = await authed_client.post("/api/agent/close-session",
                                     json={"thread_id": games[-1].id})
        assert r.status_code == 204
        dna_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_history_returns_messages(authed_client, db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    await db.commit()

    fake_msgs = [HumanMessage("hi"), AIMessage("hello")]

    with patch.object(agent_service, "get_messages", new=AsyncMock(return_value=fake_msgs)):
        r = await authed_client.get("/api/agent/history/g-1")
        assert r.status_code == 200
        data = r.json()
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_full_game_flow(authed_client, db):
    """observe game_end → close-session → game_summary persisted."""
    db.add(Profile(user_id="test-user-id", plan="free"))
    g = Game(user_id="test-user-id", pgn="1.e4 e5 2.Nf3 Nc6",
             eval_per_ply=[0, 30, 25, 20], time_per_ply=[1.0, 1.0, 1.0, 1.0],
             player_color="white", result="win")
    db.add(g)
    await db.flush()
    db.add(TiltReport(game_id=g.id, headline="Steady play",
                      diagnosis="d", pattern_label="focused",
                      evidence_plies=[], suggestion="s"))
    await db.commit()

    async def fake_stream(thread_id, ctx, messages):
        yield 'event: token\ndata: {"text": "ok"}\n\n'
        yield 'event: done\ndata: {}\n\n'

    fake_msgs = [HumanMessage("hey"), AIMessage("hi")]

    from app.services.summarizer import ChatSummaryOutput
    summary_obj = ChatSummaryOutput(summary="full flow", key_facts=[])

    with patch.object(agent_service, "stream", new=fake_stream), \
         patch.object(agent_service, "get_messages", new=AsyncMock(return_value=fake_msgs)), \
         patch("app.routers.agent.summarize_chat", new=AsyncMock(return_value=summary_obj)):

        async with authed_client.stream("POST", "/api/agent/observe",
                json={"thread_id": g.id, "event": "game_end",
                      "payload": {"game_id": g.id}}) as r:
            await r.aread()

        r2 = await authed_client.post("/api/agent/close-session",
                                       json={"thread_id": g.id})
        assert r2.status_code == 204

    rows = await db.execute(select(GameSummary).where(GameSummary.game_id == g.id))
    assert rows.scalar_one().summary == "full flow"