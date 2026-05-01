# backend/tests/test_event_detector.py
import pytest
from app.services.event_detector import classify_ply

@pytest.mark.parametrize("eval_before,eval_after,player_color,expected", [
    # White blunders: eval drops by >=200cp from White's POV
    (50, -250, "white", "blunder"),
    (50, -150, "white", None),
    # Black blunders: eval rises by >=200cp from White's POV
    (-50, 250, "black", "blunder"),
    (-50, 150, "black", None),
    # Wrong color player isn't penalized for opponent's eval swing
    (50, -250, "black", None),
    (-50, 250, "white", None),
    # Mate scoring (we store ±9999) — clamp, only flag big swings
    (100, -9999, "white", "blunder"),
    # Identical evals: no event
    (0, 0, "white", None),
])
def test_classify_ply(eval_before, eval_after, player_color, expected):
    assert classify_ply(eval_before, eval_after, player_color, time_taken=2.0) == expected