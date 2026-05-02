"""Tests for chess platform import helpers."""
import pytest
from app.services.chess_import import parse_clock_times, parse_eval_annotations, map_chesscom_result


PGN_WITH_CLOCKS = """[Event "Live Chess"]
[White "alice"]
[Black "bob"]
[TimeControl "600"]

1. e4 {[%clk 0:09:55]} e5 {[%clk 0:09:58]} 2. Nf3 {[%clk 0:09:44]} Nc6 {[%clk 0:09:50]} *
"""

PGN_WITH_EVALS = """[Event "Rated Blitz game"]
[White "alice"]
[Black "bob"]

1. e4 {[%eval 0.20]} e5 {[%eval 0.15]} 2. Nf3 {[%eval 0.45]} Nc6 {[%eval 0.30]} *
"""

PGN_WITH_MATE_EVAL = """[Event "Rated Blitz game"]
[White "alice"]
[Black "bob"]

1. e4 {[%eval #3]} e5 {[%eval #-2]} *
"""

PGN_NO_ANNOTATIONS = """[Event "Live Chess"]
[White "alice"]
[Black "bob"]

1. e4 e5 2. Nf3 Nc6 *
"""


def test_parse_clock_times_returns_per_ply_seconds():
    times = parse_clock_times(PGN_WITH_CLOCKS)
    # ply 0 (white e4): no previous same-color clock → 0.0
    # ply 1 (black e5): no previous same-color clock → 0.0
    # ply 2 (white Nf3): 9:55 - 9:44 = 11 seconds
    # ply 3 (black Nc6): 9:58 - 9:50 = 8 seconds
    assert len(times) == 4
    assert times[0] == 0.0
    assert times[1] == 0.0
    assert times[2] == pytest.approx(11.0)
    assert times[3] == pytest.approx(8.0)


def test_parse_clock_times_no_annotations_returns_empty():
    times = parse_clock_times(PGN_NO_ANNOTATIONS)
    assert times == []


def test_parse_eval_annotations_centipawns():
    evals = parse_eval_annotations(PGN_WITH_EVALS)
    assert evals is not None
    assert len(evals) == 4
    assert evals[0] == 20   # 0.20 → 20 cp
    assert evals[1] == 15
    assert evals[2] == 45
    assert evals[3] == 30


def test_parse_eval_annotations_mate():
    evals = parse_eval_annotations(PGN_WITH_MATE_EVAL)
    assert evals is not None
    assert evals[0] == 10_000   # #3 = white forced mate → +10000
    assert evals[1] == -10_000  # #-2 = black forced mate → -10000


def test_parse_eval_annotations_no_evals_returns_none():
    evals = parse_eval_annotations(PGN_NO_ANNOTATIONS)
    assert evals is None


def test_map_chesscom_result_win():
    assert map_chesscom_result("win", "white") == ("white", "win")


def test_map_chesscom_result_resigned():
    # player resigned means the player (black) lost
    assert map_chesscom_result("resigned", "black") == ("black", "loss")


def test_map_chesscom_result_checkmated():
    # white was checkmated → white loses
    assert map_chesscom_result("checkmated", "white") == ("white", "loss")


def test_map_chesscom_result_draw():
    assert map_chesscom_result("agreed", "white") == ("white", "draw")
    assert map_chesscom_result("stalemate", "black") == ("black", "draw")
    assert map_chesscom_result("insufficient", "white") == ("white", "draw")
    assert map_chesscom_result("50move", "black") == ("black", "draw")
    assert map_chesscom_result("repetition", "white") == ("white", "draw")
