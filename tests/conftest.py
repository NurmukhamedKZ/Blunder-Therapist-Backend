import os

# Must be set before any app imports so pydantic-settings picks them up
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_JWT_SECRET", "super-secret-test-key-32-chars-xxxxx!")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport

from app.database import Base, get_db
from app.dependencies import CurrentUser, get_current_user
from app.main import app

_ENGINE = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_Session = async_sessionmaker(_ENGINE, expire_on_commit=False)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    async with _ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with _Session() as session:
        yield session


@pytest_asyncio.fixture
async def client(db: AsyncSession) -> AsyncClient:
    async def _get_db():
        yield db

    app.dependency_overrides[get_db] = _get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def authed_client(db: AsyncSession) -> AsyncClient:
    async def _get_db():
        yield db

    def _get_user():
        return CurrentUser(user_id="test-user-id", plan="free")

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user] = _get_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_current_user, None)


@pytest_asyncio.fixture
async def pro_client(db: AsyncSession) -> AsyncClient:
    async def _get_db():
        yield db

    def _get_user():
        return CurrentUser(user_id="test-user-id", plan="pro")

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user] = _get_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_current_user, None)


@pytest_asyncio.fixture(autouse=True)
async def _agent_in_memory():
    """Force agent to use MemorySaver in tests (no Postgres)."""
    from app.services import agent as agent_module
    await agent_module.init_agent(database_url="", in_memory=True)
    yield
    agent_module._agent = None
    agent_module._checkpointer = None


@pytest_asyncio.fixture(autouse=True)
async def _clean_test_user(db):
    """Delete known test user rows after each test to prevent UNIQUE conflicts."""
    yield
    from sqlalchemy import text
    for uid in ("test-user-id", "u1", "u2"):
        await db.execute(text(f"DELETE FROM game_summaries WHERE user_id = '{uid}'"))
        await db.execute(text(
            f"DELETE FROM tilt_reports WHERE game_id IN "
            f"(SELECT id FROM games WHERE user_id = '{uid}')"
        ))
        await db.execute(text(f"DELETE FROM games WHERE user_id = '{uid}'"))
        await db.execute(text(f"DELETE FROM decision_dna WHERE user_id = '{uid}'"))
        await db.execute(text(f"DELETE FROM profiles WHERE user_id = '{uid}'"))
    await db.commit()
