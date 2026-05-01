"""Detect behaviorally meaningful events from a single ply.

Pure function — no LLM, no IO. Called server-side on every observe POST and
client-side as a quick pre-filter to avoid hitting the API for nothing.
"""
from typing import Literal

BLUNDER_THRESHOLD_CP = 200

EventType = Literal["blunder"]


def classify_ply(
    eval_before: float,
    eval_after: float,
    player_color: str,
    time_taken: float,
) -> EventType | None:
    """Return an event tag if the ply is behaviorally interesting, else None.

    Evals are stored from White's POV (centipawns). A blunder for the human
    player is a swing of >=200cp AGAINST them.
    """
    delta = eval_after - eval_before  # +ve = good for white
    if player_color == "white" and delta < -BLUNDER_THRESHOLD_CP:
        return "blunder"
    if player_color == "black" and delta > BLUNDER_THRESHOLD_CP:
        return "blunder"
    return None