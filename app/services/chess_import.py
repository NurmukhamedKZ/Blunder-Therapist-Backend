"""Fetch and parse games from Chess.com and Lichess public APIs."""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import httpx
import structlog

log = structlog.get_logger()

_CHESSCOM_WIN_RESULTS = {"win"}
_CHESSCOM_LOSS_RESULTS = {"resigned", "checkmated", "timeout", "abandoned", "bughousepartnerlose"}
_CHESSCOM_DRAW_RESULTS = {"agreed", "stalemate", "insufficient", "50move", "repetition", "timevsinsufficient"}

_CLK_RE = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+)\]')
_EVAL_RE = re.compile(r'\[%eval\s+(#-?\d+|[+-]?\d+\.?\d*)\]')

MATE_CP = 10_000


@dataclass
class RawGame:
    pgn: str
    platform: str
    platform_game_id: str
    player_color: Literal["white", "black"]
    result: Literal["win", "loss", "draw"]
    time_per_ply: list[float]
    eval_per_ply: list[int] | None


def parse_clock_times(pgn: str) -> list[float]:
    """Extract per-ply thinking time (seconds) from [%clk] annotations.

    Returns empty list if no annotations found.
    For the first move of each color there is no prior same-color clock,
    so those entries are 0.0.
    """
    clocks = []
    for m in _CLK_RE.finditer(pgn):
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        clocks.append(h * 3600 + mi * 60 + s)

    if not clocks:
        return []

    times: list[float] = []
    for i, clk in enumerate(clocks):
        if i < 2:
            times.append(0.0)
        else:
            times.append(float(clocks[i - 2] - clk))

    return times


def parse_eval_annotations(pgn: str) -> list[int] | None:
    """Extract per-ply engine evals (centipawns, White's POV) from [%eval] annotations.

    Returns None if no annotations found (Stockfish will be used instead).
    Forced-mate annotations (#N) map to ±MATE_CP.
    """
    evals: list[int] = []
    for m in _EVAL_RE.finditer(pgn):
        val = m.group(1)
        if val.startswith("#"):
            mate_in = int(val[1:])
            evals.append(MATE_CP if mate_in > 0 else -MATE_CP)
        else:
            evals.append(int(float(val) * 100))

    return evals if evals else None


def map_chesscom_result(
    result_str: str,
    player_color: Literal["white", "black"],
) -> tuple[Literal["white", "black"], Literal["win", "loss", "draw"]]:
    """Map Chess.com result string (from that player's perspective) to (color, outcome)."""
    if result_str in _CHESSCOM_WIN_RESULTS:
        return player_color, "win"
    if result_str in _CHESSCOM_LOSS_RESULTS:
        return player_color, "loss"
    return player_color, "draw"


async def fetch_chesscom_games(
    username: str,
    since: datetime,
    until: datetime,
) -> list[RawGame]:
    """Fetch games from Chess.com public API for the given date range.

    Skips 'daily' (correspondence) games — no real clock data.
    Raises ValueError if username not found.
    """
    months: list[tuple[int, int]] = []
    cur = since.replace(day=1)
    while cur <= until:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    games: list[RawGame] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for year, month in months:
            url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}"
            resp = await client.get(url)
            if resp.status_code == 404:
                raise ValueError(f"Username not found on chess.com: {username}")
            resp.raise_for_status()
            data = resp.json()

            for g in data.get("games", []):
                if g.get("time_class") == "daily":
                    continue
                end_ts = g.get("end_time", 0)
                end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                if not (since <= end_dt <= until):
                    continue

                white_name = g["white"]["username"].lower()
                if white_name == username.lower():
                    player_color: Literal["white", "black"] = "white"
                    result_str = g["white"]["result"]
                else:
                    player_color = "black"
                    result_str = g["black"]["result"]

                _, result = map_chesscom_result(result_str, player_color)
                pgn = g["pgn"]
                platform_game_id = "chesscom_" + g["url"].rstrip("/").split("/")[-1]

                games.append(RawGame(
                    pgn=pgn,
                    platform="chess.com",
                    platform_game_id=platform_game_id,
                    player_color=player_color,
                    result=result,
                    time_per_ply=parse_clock_times(pgn),
                    eval_per_ply=None,
                ))

    log.info("chesscom_fetch_done", username=username, count=len(games))
    return games


async def fetch_lichess_games(
    username: str,
    since: datetime,
    until: datetime,
) -> list[RawGame]:
    """Fetch games from Lichess public API for the given date range.

    Uses evals from Lichess analysis when available; otherwise eval_per_ply is None.
    Raises ValueError if username not found.
    """
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "since": since_ms,
        "until": until_ms,
        "evals": "true",
        "clocks": "true",
        "perfType": "bullet,blitz,rapid,classical",
    }
    headers = {"Accept": "application/x-ndjson"}

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code == 404:
            raise ValueError(f"Username not found on lichess: {username}")
        resp.raise_for_status()
        body = resp.text

    games: list[RawGame] = []
    for line in body.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        game_data = json.loads(line)
        raw = _parse_lichess_game(game_data, username)
        if raw:
            games.append(raw)

    log.info("lichess_fetch_done", username=username, count=len(games))
    return games


def _parse_lichess_game(data: dict, username: str) -> RawGame | None:
    players = data.get("players", {})
    white_name = (players.get("white", {}).get("user", {}).get("name") or "").lower()

    if white_name == username.lower():
        player_color: Literal["white", "black"] = "white"
    else:
        player_color = "black"

    winner = data.get("winner")
    if winner is None:
        result: Literal["win", "loss", "draw"] = "draw"
    elif winner == player_color:
        result = "win"
    else:
        result = "loss"

    pgn = data.get("pgn", "")
    if not pgn:
        return None

    return RawGame(
        pgn=pgn,
        platform="lichess",
        platform_game_id="lichess_" + data["id"],
        player_color=player_color,
        result=result,
        time_per_ply=parse_clock_times(pgn),
        eval_per_ply=parse_eval_annotations(pgn),
    )
