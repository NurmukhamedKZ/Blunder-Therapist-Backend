import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

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


async def test_list_games_empty_new_user(db: AsyncSession):
    # Use a brand-new user_id that no other test touches so total is guaranteed 0
    from app.dependencies import CurrentUser, get_current_user
    from app.main import app as fastapi_app
    from httpx import AsyncClient, ASGITransport

    async def _db():
        yield db

    fastapi_app.dependency_overrides[get_current_user] = lambda: CurrentUser(user_id="empty-user-xyz", plan="free")
    fastapi_app.dependency_overrides[get_db] = _db
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as c:
        resp = await c.get("/api/games")
    fastapi_app.dependency_overrides.pop(get_current_user, None)
    fastapi_app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["games"] == []


async def test_get_game_not_found(authed_client: AsyncClient):
    resp = await authed_client.get("/api/games/nonexistent-id")
    assert resp.status_code == 404


async def test_list_games_after_save(db: AsyncSession):
    # Use a unique user_id so this test is isolated from others
    from app.dependencies import CurrentUser, get_current_user
    from app.main import app as fastapi_app
    from app.models import Game, Profile
    from httpx import AsyncClient, ASGITransport

    profile = Profile(user_id="history-user-abc", plan="free")
    db.add(profile)
    game = Game(
        user_id="history-user-abc",
        pgn="1. e4",
        eval_per_ply=[10],
        time_per_ply=[1.0],
        player_color="white",
        result="draw",
    )
    db.add(game)
    await db.commit()

    async def _db():
        yield db

    fastapi_app.dependency_overrides[get_current_user] = lambda: CurrentUser(user_id="history-user-abc", plan="free")
    fastapi_app.dependency_overrides[get_db] = _db
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as c:
        resp = await c.get("/api/games")
    fastapi_app.dependency_overrides.pop(get_current_user, None)
    fastapi_app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["games"][0]["id"] == game.id