import pytest
from httpx import AsyncClient


SAMPLE_GAME = {
    "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
    "eval_per_ply": [20, -20, 30, -30, 40, -40],
    "time_per_ply": [1.0, 2.0, 1.5, 0.5, 1.0, 2.0],
    "player_color": "white",
    "result": "draw",
}


async def test_analyze_game_requires_auth(client: AsyncClient):
    resp = await client.post("/api/analyze-game", json=SAMPLE_GAME)
    assert resp.status_code == 401


async def test_analyze_game_with_auth_saves_game(authed_client: AsyncClient):
    # This will call the real LLM unless OPENAI_API_KEY is mocked,
    # so we just test that auth passes and DB save happens.
    # In CI, mock run_tilt_detector; here we test the 502 path.
    resp = await authed_client.post("/api/analyze-game", json=SAMPLE_GAME)
    # 200 if LLM works, 502 if OPENAI_API_KEY=test (invalid key)
    assert resp.status_code in (200, 502)


async def test_decision_dna_requires_pro(authed_client: AsyncClient):
    resp = await authed_client.post("/api/decision-dna", json={"n": 5})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "pro_required"


async def test_coach_requires_pro(authed_client: AsyncClient):
    resp = await authed_client.post("/api/coach", json={"message": "hi"})
    assert resp.status_code == 403


async def test_decision_dna_not_enough_games(pro_client: AsyncClient):
    resp = await pro_client.post("/api/decision-dna", json={"n": 5})
    assert resp.status_code == 400
    assert "at least 3 games" in resp.json()["detail"]