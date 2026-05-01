import pytest
from unittest.mock import AsyncMock, patch

from app.models import Profile, Game, DecisionDNA
from app.services.dna_job import run_dna_for_user, should_recompute_dna
from sqlalchemy import select


def _add_games(db, user_id: str, n: int) -> None:
    for _ in range(n):
        db.add(Game(user_id=user_id, pgn="1.e4 e5", eval_per_ply=[0, 30, 0],
                    time_per_ply=[1.0, 1.0, 1.0], player_color="white", result="win"))


def test_should_recompute_dna_strict_multiples_of_5():
    assert should_recompute_dna(5) is True
    assert should_recompute_dna(10) is True
    assert should_recompute_dna(0) is False
    assert should_recompute_dna(4) is False
    assert should_recompute_dna(7) is False


@pytest.mark.asyncio
async def test_run_dna_for_user_upserts(db):
    db.add(Profile(user_id="u1", plan="free"))
    _add_games(db, "u1", 5)
    await db.commit()

    fake_dna = {
        "type_name": "Aggressive Tactician", "tagline": "fights early",
        "summary": "...", "core_strength": "...", "core_weakness": "...",
        "gm_comparison": {"name": "Tal", "similarity_pct": 22, "why": "..."},
    }
    with patch("app.services.dna_job.run_decision_dna",
               new=AsyncMock(return_value=fake_dna)):
        await run_dna_for_user(db, user_id="u1")

    rows = await db.execute(select(DecisionDNA).where(DecisionDNA.user_id == "u1"))
    saved = rows.scalars().all()
    assert len(saved) == 1
    assert saved[0].dna["type_name"] == "Aggressive Tactician"
    assert saved[0].games_count == 5
