"""Tests for /api/import endpoints."""
from unittest.mock import patch
import pytest

from app.models import Profile


@pytest.fixture(autouse=True)
async def profile(db):
    db.add(Profile(user_id="test-user-id", plan="free"))
    await db.commit()


@pytest.mark.asyncio
async def test_create_import_job_returns_202(authed_client):
    with patch("app.routers.imports.asyncio.create_task"):
        resp = await authed_client.post("/api/import", json={
            "platform": "chess.com",
            "username": "alice",
            "period_days": 30,
        })
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "pending"
    assert data["total_games"] == 0
    assert data["processed_games"] == 0
    assert "job_id" in data


@pytest.mark.asyncio
async def test_get_import_job_status(authed_client):
    with patch("app.routers.imports.asyncio.create_task"):
        create_resp = await authed_client.post("/api/import", json={
            "platform": "lichess",
            "username": "bob",
            "period_days": 90,
        })
    job_id = create_resp.json()["job_id"]

    resp = await authed_client.get(f"/api/import/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_get_import_job_not_found(authed_client):
    resp = await authed_client.get("/api/import/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_import_jobs(authed_client):
    with patch("app.routers.imports.asyncio.create_task"):
        await authed_client.post("/api/import", json={"platform": "chess.com", "username": "alice", "period_days": 30})
        await authed_client.post("/api/import", json={"platform": "lichess", "username": "alice", "period_days": 90})

    resp = await authed_client.get("/api/import")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["jobs"]) >= 2
