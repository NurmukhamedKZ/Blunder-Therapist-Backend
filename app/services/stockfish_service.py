"""Async Stockfish wrapper for backend game evaluation."""
import asyncio
import io

import chess
import chess.pgn
import structlog
from stockfish import Stockfish

log = structlog.get_logger()

MATE_CP = 10_000
_DEFAULT_DEPTH = 15


async def analyze_pgn(pgn: str, depth: int = _DEFAULT_DEPTH) -> list[int]:
    """Return per-ply Stockfish evals (centipawns, White's POV) for a game PGN.

    Runs in a thread pool executor to avoid blocking the event loop.
    Raises StockfishException if the binary is not installed.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _analyze_pgn_sync, pgn, depth)


def _analyze_pgn_sync(pgn: str, depth: int) -> list[int]:
    sf = Stockfish(depth=depth)
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    evals: list[int] = []

    for node in game.mainline():
        board.push(node.move)
        sf.set_fen_position(board.fen())
        ev = sf.get_evaluation()
        if ev["type"] == "cp":
            evals.append(ev["value"])
        else:
            evals.append(MATE_CP if ev["value"] > 0 else -MATE_CP)

    log.debug("stockfish_analysis_done", ply_count=len(evals))
    return evals
