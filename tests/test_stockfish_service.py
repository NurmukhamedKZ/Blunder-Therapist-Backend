"""Tests for stockfish_service — mocks the binary."""
from unittest.mock import patch, MagicMock
import pytest
from app.services.stockfish_service import _analyze_pgn_sync, MATE_CP

SIMPLE_PGN = """[Event "Test"]
[White "a"]
[Black "b"]

1. e4 e5 2. Nf3 Nc6 *
"""


def _make_stockfish_mock(eval_sequence: list[dict]) -> MagicMock:
    sf = MagicMock()
    sf.get_evaluation.side_effect = eval_sequence
    return sf


def test_analyze_pgn_sync_centipawn_evals():
    evals_from_sf = [
        {"type": "cp", "value": 20},
        {"type": "cp", "value": 15},
        {"type": "cp", "value": 45},
        {"type": "cp", "value": 30},
    ]
    mock_sf = _make_stockfish_mock(evals_from_sf)

    with patch("app.services.stockfish_service.Stockfish", return_value=mock_sf):
        result = _analyze_pgn_sync(SIMPLE_PGN, depth=10)

    assert result == [20, 15, 45, 30]
    assert mock_sf.set_fen_position.call_count == 4


def test_analyze_pgn_sync_mate_becomes_max_cp():
    evals_from_sf = [
        {"type": "cp", "value": 200},
        {"type": "mate", "value": 3},    # white forces mate
        {"type": "mate", "value": -2},   # black forces mate
        {"type": "cp", "value": -100},
    ]
    mock_sf = _make_stockfish_mock(evals_from_sf)

    with patch("app.services.stockfish_service.Stockfish", return_value=mock_sf):
        result = _analyze_pgn_sync(SIMPLE_PGN, depth=10)

    assert result[1] == MATE_CP
    assert result[2] == -MATE_CP
