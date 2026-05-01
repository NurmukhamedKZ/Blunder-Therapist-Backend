"""
Feature extraction from a chess game.

This is the heart of Blunder Therapist. We take raw move data + Stockfish
evaluations + per-move thinking time, and produce structured behavioral
features that an LLM can interpret psychologically.

Think of it as: PGN -> behavioral fingerprint -> LLM prompt input.
"""
from dataclasses import dataclass, asdict
from typing import Literal

import chess
import chess.pgn
import io


# A "blunder" is conventionally a move that loses 200+ centipawns of advantage.
# A "mistake" is 100-199cp. An "inaccuracy" is 50-99cp.
BLUNDER_CP = 200
MISTAKE_CP = 100
INACCURACY_CP = 50


@dataclass
class MoveFeature:
    """Per-move features."""
    ply: int  # 0-indexed half-move number
    san: str  # e.g. "Nf3"
    side: Literal["white", "black"]
    eval_before: int  # centipawns from White's perspective
    eval_after: int
    eval_delta: int  # positive delta = good for the side that just moved
    time_spent_sec: float  # how long the player thought
    is_blunder: bool
    is_mistake: bool
    is_inaccuracy: bool
    move_quality: Literal["brilliant", "good", "ok", "inaccuracy", "mistake", "blunder"]


@dataclass
class GameFeatures:
    """Aggregated features for a whole game from one player's perspective."""
    # Identity
    player_color: Literal["white", "black"]
    result: Literal["win", "loss", "draw"]
    total_moves: int

    # Move quality
    blunder_count: int
    mistake_count: int
    inaccuracy_count: int
    avg_centipawn_loss: float  # average cp lost per move (lower = better)

    # Timing patterns - this is where psychological signal lives
    avg_time_per_move: float
    median_time_per_move: float
    fastest_move_sec: float
    slowest_move_sec: float

    # The killer feature: time-pressure detection
    # We compute thinking time in the moves AFTER opponent did something
    # significant (capture, check, threat). If the player's time drops sharply
    # after an opponent's capture, that's a tilt signal.
    avg_time_before_blunders: float  # average time spent on blundered moves
    avg_time_overall: float
    blunder_speed_ratio: float  # avg_time_before_blunders / avg_time_overall
    # ratio < 0.5 = "blundered while rushing" = classic tilt
    # ratio > 1.5 = "blundered while overthinking" = analysis paralysis

    # Cascading mistakes (= tilt)
    # A "cascade" = 2+ blunders/mistakes within 3 moves of each other
    has_blunder_cascade: bool
    cascade_start_ply: int | None  # ply where the worst cascade began

    # Phase-of-game patterns
    opening_blunders: int  # plies 0-19
    middlegame_blunders: int  # plies 20-49
    endgame_blunders: int  # plies 50+

    # Per-move detail (for the LLM to reason about specific moments)
    moves: list[MoveFeature]


def _classify_move(eval_delta_cp: int) -> str:
    """Classify a single move quality from the player's perspective.

    eval_delta_cp is signed FROM THE MOVING PLAYER'S PERSPECTIVE:
        positive = move improved their position
        negative = move worsened their position
    """
    if eval_delta_cp >= 0:
        return "good" if eval_delta_cp > 50 else "ok"
    loss = -eval_delta_cp
    if loss >= BLUNDER_CP:
        return "blunder"
    if loss >= MISTAKE_CP:
        return "mistake"
    if loss >= INACCURACY_CP:
        return "inaccuracy"
    return "ok"


def extract_features(
    pgn: str,
    eval_per_ply: list[int],  # Stockfish eval AFTER each ply, in centipawns from White's POV
    time_per_ply: list[float],  # seconds spent on each ply
    player_color: Literal["white", "black"],
    result: Literal["win", "loss", "draw"],
) -> GameFeatures:
    """Extract behavioral features from a game.

    eval_per_ply[i] is the position evaluation AFTER ply i was played.
    eval_per_ply[-1] before any move = 0 (starting position is balanced).

    NOTE: We expect the caller (the frontend, using stockfish.js) to pre-compute
    evaluations. We don't run Stockfish on the backend - WASM in the browser
    is faster and free.
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("Could not parse PGN")

    # Walk the game and build per-move features
    moves: list[MoveFeature] = []
    board = game.board()
    prev_eval = 0  # starting position

    for ply, node in enumerate(game.mainline()):
        san = board.san(node.move)
        side: Literal["white", "black"] = "white" if board.turn == chess.WHITE else "black"
        board.push(node.move)

        eval_after = eval_per_ply[ply] if ply < len(eval_per_ply) else prev_eval
        # Convert to "from this player's perspective"
        # If white just moved: their delta = eval_after - prev_eval
        # If black just moved: their delta = -(eval_after - prev_eval)
        raw_delta = eval_after - prev_eval
        player_delta = raw_delta if side == "white" else -raw_delta

        quality = _classify_move(player_delta)
        time_spent = time_per_ply[ply] if ply < len(time_per_ply) else 0.0

        moves.append(MoveFeature(
            ply=ply,
            san=san,
            side=side,
            eval_before=prev_eval,
            eval_after=eval_after,
            eval_delta=player_delta,
            time_spent_sec=time_spent,
            is_blunder=quality == "blunder",
            is_mistake=quality == "mistake",
            is_inaccuracy=quality == "inaccuracy",
            move_quality=quality,  # type: ignore[arg-type]
        ))
        prev_eval = eval_after

    # Now aggregate, but ONLY counting moves by the target player
    player_moves = [m for m in moves if m.side == player_color]

    if not player_moves:
        raise ValueError("No moves found for the given player color")

    blunders = [m for m in player_moves if m.is_blunder]
    mistakes = [m for m in player_moves if m.is_mistake]
    inaccs = [m for m in player_moves if m.is_inaccuracy]

    # Average centipawn loss = how bad each move was on average
    cp_losses = [max(0, -m.eval_delta) for m in player_moves]
    avg_cp_loss = sum(cp_losses) / len(cp_losses)

    # Timing
    times = [m.time_spent_sec for m in player_moves if m.time_spent_sec > 0]
    times_sorted = sorted(times) if times else [0.0]
    avg_time = sum(times) / len(times) if times else 0.0
    median_time = times_sorted[len(times_sorted) // 2]

    # The behavioral signal: speed during blunders vs overall
    blunder_times = [m.time_spent_sec for m in blunders if m.time_spent_sec > 0]
    avg_blunder_time = sum(blunder_times) / len(blunder_times) if blunder_times else 0.0
    blunder_speed_ratio = (avg_blunder_time / avg_time) if avg_time > 0 else 1.0

    # Cascade detection: 2+ blunders/mistakes within 3 plies of each other
    # (within the player's own moves, so within 6 actual plies total)
    bad_indices = [
        i for i, m in enumerate(player_moves)
        if m.is_blunder or m.is_mistake
    ]
    cascade_start_ply: int | None = None
    has_cascade = False
    for i in range(len(bad_indices) - 1):
        if bad_indices[i + 1] - bad_indices[i] <= 2:  # within 2 of player's moves = within 4 plies
            has_cascade = True
            cascade_start_ply = player_moves[bad_indices[i]].ply
            break

    # Phase of game
    def phase_count(items, lo, hi):
        return sum(1 for m in items if lo <= m.ply < hi)

    return GameFeatures(
        player_color=player_color,
        result=result,
        total_moves=len(player_moves),
        blunder_count=len(blunders),
        mistake_count=len(mistakes),
        inaccuracy_count=len(inaccs),
        avg_centipawn_loss=round(avg_cp_loss, 1),
        avg_time_per_move=round(avg_time, 2),
        median_time_per_move=round(median_time, 2),
        fastest_move_sec=round(min(times), 2) if times else 0.0,
        slowest_move_sec=round(max(times), 2) if times else 0.0,
        avg_time_before_blunders=round(avg_blunder_time, 2),
        avg_time_overall=round(avg_time, 2),
        blunder_speed_ratio=round(blunder_speed_ratio, 2),
        has_blunder_cascade=has_cascade,
        cascade_start_ply=cascade_start_ply,
        opening_blunders=phase_count(blunders, 0, 20),
        middlegame_blunders=phase_count(blunders, 20, 50),
        endgame_blunders=phase_count(blunders, 50, 1000),
        moves=moves,
    )


def features_to_llm_summary(f: GameFeatures) -> str:
    """Convert features to a compact, LLM-readable summary.

    We deliberately DON'T dump every move - the LLM doesn't need them. We
    give it the aggregate signals + the 3-5 most important moments.
    """
    # Pick the moments that matter: blunders + the move just before each blunder
    important_plies: set[int] = set()
    for m in f.moves:
        if m.side == f.player_color and (m.is_blunder or m.is_mistake):
            important_plies.add(m.ply)
            if m.ply >= 2:
                important_plies.add(m.ply - 2)  # opponent's move before mine
                important_plies.add(m.ply - 1)  # the position right before

    critical_moments = [m for m in f.moves if m.ply in important_plies]

    lines = [
        f"GAME SUMMARY (player = {f.player_color}, result = {f.result})",
        f"  Total moves played: {f.total_moves}",
        f"  Blunders: {f.blunder_count} | Mistakes: {f.mistake_count} | Inaccuracies: {f.inaccuracy_count}",
        f"  Avg centipawn loss per move: {f.avg_centipawn_loss}",
        "",
        "TIMING SIGNAL (this is where psychology shows up):",
        f"  Avg time per move: {f.avg_time_per_move}s",
        f"  Avg time on blundered moves: {f.avg_time_before_blunders}s",
        f"  Blunder-speed ratio: {f.blunder_speed_ratio}",
        f"    (< 0.5 = rushed-while-blundering = TILT signal)",
        f"    (> 1.5 = overthought blunders = ANALYSIS PARALYSIS)",
        "",
        f"CASCADE: {'YES' if f.has_blunder_cascade else 'no'}"
        + (f" - started at ply {f.cascade_start_ply}" if f.has_blunder_cascade else ""),
        f"PHASE BLUNDERS: opening={f.opening_blunders}, middlegame={f.middlegame_blunders}, endgame={f.endgame_blunders}",
        "",
        "CRITICAL MOMENTS:",
    ]
    for m in critical_moments[:12]:  # cap to avoid bloating the prompt
        lines.append(
            f"  ply {m.ply} ({m.side}): {m.san}  "
            f"[eval {m.eval_before:+d} -> {m.eval_after:+d}, "
            f"delta {m.eval_delta:+d}, "
            f"thought {m.time_spent_sec:.1f}s, "
            f"quality={m.move_quality}]"
        )
    return "\n".join(lines)
