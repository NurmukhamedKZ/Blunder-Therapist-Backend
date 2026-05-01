import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base
from app.models import Profile, Game, TiltReport


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def test_create_profile(session):
    profile = Profile(user_id="user-1", plan="free")
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    assert profile.user_id == "user-1"
    assert profile.plan == "free"
    assert profile.created_at is not None


async def test_create_game_and_tilt_report(session):
    profile = Profile(user_id="user-2", plan="free")
    session.add(profile)
    await session.flush()

    game = Game(
        user_id="user-2",
        pgn="1. e4 e5",
        eval_per_ply=[10, -10],
        time_per_ply=[1.0, 2.0],
        player_color="white",
        result="win",
    )
    session.add(game)
    await session.flush()

    report = TiltReport(
        game_id=game.id,
        headline="Test",
        diagnosis="Diag",
        pattern_label="tilt",
        evidence_plies=[3, 5],
        suggestion="Breathe",
    )
    session.add(report)
    await session.commit()

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Game).options(selectinload(Game.tilt_report)).where(Game.id == game.id)
    )
    loaded = result.scalar_one()
    assert loaded.tilt_report is not None
    assert loaded.tilt_report.headline == "Test"