import pytest
from datetime import datetime, timezone
from app.models import Profile, Game, TiltReport, GameSummary, DecisionDNA
from app.services.agent_tools import _list_past_games_impl, _get_game_details_impl
from app.services.agent import build_system_prompt


@pytest.mark.asyncio
async def test_list_past_games_returns_user_games_only(db):
    db.add(Profile(user_id="alice", plan="free"))
    db.add(Profile(user_id="bob", plan="free"))
    g1 = Game(user_id="alice", pgn="x", eval_per_ply=[0], time_per_ply=[1.0],
              player_color="white", result="win",
              played_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    g2 = Game(user_id="bob", pgn="y", eval_per_ply=[0], time_per_ply=[1.0],
              player_color="white", result="loss",
              played_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    db.add_all([g1, g2])
    await db.flush()
    db.add(TiltReport(game_id=g1.id, headline="h", diagnosis="d",
                      pattern_label="tilt", evidence_plies=[1], suggestion="s"))
    await db.commit()

    out = await _list_past_games_impl(db, user_id="alice", filter=None)
    assert len(out) == 1
    assert out[0]["game_id"] == g1.id
    assert out[0]["tilt_pattern"] == "tilt"


@pytest.mark.asyncio
async def test_list_past_games_filter_losses(db):
    db.add(Profile(user_id="alice2", plan="free"))
    for r in ["win", "loss", "loss", "draw"]:
        db.add(Game(user_id="alice2", pgn="x", eval_per_ply=[0], time_per_ply=[1.0],
                    player_color="white", result=r))
    await db.commit()

    out = await _list_past_games_impl(db, user_id="alice2", filter="losses")
    assert len(out) == 2
    assert all(g["result"] == "loss" for g in out)


@pytest.mark.asyncio
async def test_get_game_details_rejects_other_user(db):
    db.add(Profile(user_id="alice3", plan="free"))
    db.add(Profile(user_id="bob3", plan="free"))
    g = Game(user_id="bob3", pgn="x", eval_per_ply=[0], time_per_ply=[1.0],
             player_color="white", result="win")
    db.add(g)
    await db.commit()

    out = await _get_game_details_impl(db, user_id="alice3", game_id=g.id)
    assert out == {"error": "not found"}


@pytest.mark.asyncio
async def test_get_game_details_returns_full_record(db):
    db.add(Profile(user_id="alice4", plan="free"))
    g = Game(user_id="alice4", pgn="1.e4", eval_per_ply=[0, 30], time_per_ply=[2.0, 1.5],
             player_color="white", result="win")
    db.add(g)
    await db.flush()
    db.add(TiltReport(game_id=g.id, headline="h", diagnosis="d",
                      pattern_label="focused", evidence_plies=[], suggestion="s"))
    db.add(GameSummary(user_id="alice4", game_id=g.id, summary="all good", key_facts=["x"]))
    await db.commit()

    out = await _get_game_details_impl(db, user_id="alice4", game_id=g.id)
    assert out["pgn"] == "1.e4"
    assert out["eval_per_ply"] == [0, 30]
    assert out["tilt_report"]["pattern_label"] == "focused"
    assert out["chat_summary"]["key_facts"] == ["x"]


@pytest.mark.asyncio
async def test_build_system_prompt_includes_memory(db):
    db.add(Profile(user_id="alice5", plan="free"))
    g = Game(user_id="alice5", pgn="x", eval_per_ply=[0], time_per_ply=[1.0],
             player_color="white", result="loss")
    db.add(g)
    await db.flush()
    db.add(GameSummary(user_id="alice5", game_id=g.id,
                       summary="rushed in middlegame", key_facts=["rushes"]))
    db.add(DecisionDNA(user_id="alice5",
                       dna={"type_name": "Aggressive Tactician", "tagline": "x",
                            "summary": "y", "core_strength": "z",
                            "core_weakness": "w", "gm_comparison": {
                                "name": "Tal", "similarity_pct": 20, "why": "."}},
                       games_count=5))
    await db.commit()

    prompt = await build_system_prompt(db, user_id="alice5")
    assert "Aggressive Tactician" in prompt
    assert "rushes" in prompt or "rushed in middlegame" in prompt
    assert "behavioral" in prompt.lower()


@pytest.mark.asyncio
async def test_build_system_prompt_handles_no_memory(db):
    db.add(Profile(user_id="newbie", plan="free"))
    await db.commit()
    prompt = await build_system_prompt(db, user_id="newbie")
    assert "no past games" in prompt.lower() or "no memory" in prompt.lower()
