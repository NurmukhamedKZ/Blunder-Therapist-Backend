"""Agent chat endpoints."""
import json
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from langchain_core.messages import HumanMessage
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import CurrentUser, get_current_user
from app.models import Game, TiltReport, GameSummary
from app.schemas.agent import (
    ObserveRequest, MessageRequest, CloseSessionRequest,
    HistoryMessage, HistoryResponse,
)
from app.services.agent import agent_service
from app.services.agent_tools import AgentContext
from app.services.dna_job import should_recompute_dna, run_dna_for_user
from app.services.summarizer import summarize_chat

router = APIRouter(prefix="/api/agent", tags=["agent"])

log = structlog.get_logger()

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _extract_jwt(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.lower().startswith("bearer ") else ""


def _format_observation_message(event: str, payload: dict) -> str:
    if event == "game_start":
        return "[OBSERVATION] Game started — watch and stay quiet unless asked."

    if event == "blunder":
        moves: list[dict] = payload.get("moves_since_last_observe") or []
        lines = [
            f"[OBSERVATION] Blunder at ply {payload.get('ply')} "
            f"({payload.get('san')}): eval {payload.get('eval_before')}→"
            f"{payload.get('eval_after')}cp in {payload.get('time_taken')}s.",
        ]
        if moves:
            lines.append("Moves leading here (ply / san / eval_after / time):")
            for m in moves:
                marker = " ← BLUNDER" if m["ply"] == payload.get("ply") else ""
                lines.append(
                    f"  ply {m['ply']}: {m['san']:<8} "
                    f"eval={m['eval_after']:+d}cp  t={m['time_sec']:.1f}s{marker}"
                )
        lines.append("\nRespond in 1-2 sentences max. Be brief and direct.")
        return "\n".join(lines)

    return f"[OBSERVATION] {event}: {json.dumps(payload)}"


@router.post("/observe")
async def observe(
    req: ObserveRequest,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    jwt_token = _extract_jwt(request)
    ctx = AgentContext(user_id=user.user_id, jwt=jwt_token)
    run_log = log.bind(thread_id=req.thread_id)
    run_log.info("observe_event", game_event=req.event)

    if req.event == "game_start":
        async def _empty():
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    elif req.event == "game_end":
        game_id = req.payload.get("game_id") or req.thread_id
        row = await db.execute(
            select(Game)
            .options(selectinload(Game.tilt_report))
            .where(Game.id == game_id, Game.user_id == user.user_id)
        )
        game = row.scalar_one_or_none()
        if game is None or game.tilt_report is None:
            raise HTTPException(status_code=404, detail="game or tilt report not found")
        tr = game.tilt_report
        all_moves: list[dict] = req.payload.get("all_moves") or []
        tilt_text = "The game just ended. Here's the tilt analysis:\n" + json.dumps({
            "headline": tr.headline,
            "diagnosis": tr.diagnosis,
            "pattern_label": tr.pattern_label,
            "evidence_plies": tr.evidence_plies,
            "suggestion": tr.suggestion,
        })
        if all_moves:
            evidence = set(tr.evidence_plies or [])
            tilt_text += "\n\nFull game move list (ply / san / eval_after / time):\n"
            tilt_text += "\n".join(
                f"  ply {m['ply']}: {m['san']:<8} "
                f"eval={m['eval_after']:+d}cp  t={m['time_sec']:.1f}s"
                + (" <<<" if m["ply"] in evidence else "")
                for m in all_moves
            )
        tilt_text += "\n\nReact conversationally — don't repeat the report verbatim, riff on it."
        new_msgs = [HumanMessage(tilt_text)]
    elif req.event == "blunder":
        new_msgs = [HumanMessage(_format_observation_message(req.event, req.payload))]
    elif req.event == "arrival":
        # Get last few games to provide context for the greeting
        res = await db.execute(
            select(Game)
            .where(Game.user_id == user.user_id)
            .order_by(Game.played_at.desc())
            .limit(3)
        )
        recent_games = res.scalars().all()
        
        arrival_text = "[OBSERVATION] User just arrived at the dashboard."
        if recent_games:
            arrival_text += "\nRecent games summary:"
            for g in recent_games:
                arrival_text += f"\n- {g.played_at.date()}: {g.result} as {g.player_color}"
        
        arrival_text += "\n\nGreet the user warmly. Mention their recent performance if relevant. Keep it brief (2-3 sentences)."
        new_msgs = [HumanMessage(arrival_text)]
    else:
        new_msgs = [HumanMessage(_format_observation_message(req.event, req.payload))]
    
    StreamingResponse(
        agent_service.stream(req.thread_id, ctx, new_msgs),
        media_type="text/event-stream",
    )
    


@router.post("/message")
async def message(
    req: MessageRequest,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    jwt_token = _extract_jwt(request)
    ctx = AgentContext(user_id=user.user_id, jwt=jwt_token)
    run_log = log.bind(thread_id=req.thread_id)
    run_log.info("message_received", text_length=len(req.text))
    return StreamingResponse(
        agent_service.stream(req.thread_id, ctx, [HumanMessage(req.text)]),
        media_type="text/event-stream",
    )


@router.post("/close-session", status_code=204)
async def close_session(
    req: CloseSessionRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run_log = log.bind(thread_id=req.thread_id)
    run_log.info("close_session_start")

    existing = await db.execute(
        select(GameSummary).where(GameSummary.game_id == req.thread_id)
    )
    if existing.scalar_one_or_none() is not None:
        run_log.info("close_session_skipped", reason="already_exists")
        return Response(status_code=204)

    raw_msgs = await agent_service.get_messages(req.thread_id)
    msgs_dicts = []
    for m in raw_msgs:
        role = _ROLE_MAP.get(getattr(m, "type", ""), "system")
        if role == "system":
            continue
        msgs_dicts.append({"role": role, "content": getattr(m, "content", "")})

    summary = await summarize_chat(msgs_dicts)

    game_row = await db.execute(
        select(Game).where(Game.id == req.thread_id, Game.user_id == user.user_id)
    )
    game = game_row.scalar_one_or_none()
    if game is None:
        run_log.info("close_session_skipped", reason="game_not_found")
        return Response(status_code=204)

    db.add(GameSummary(
        user_id=user.user_id, game_id=game.id,
        summary=summary.summary,
        key_facts=summary.key_facts,
        game_analysis=summary.game_analysis.model_dump() if summary.game_analysis else None,
    ))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        run_log.info("close_session_skipped", reason="concurrent_insert")
        return Response(status_code=204)

    total = (await db.execute(
        select(func.count()).where(Game.user_id == user.user_id)
    )).scalar_one()
    if should_recompute_dna(total):
        try:
            await run_dna_for_user(db, user.user_id)
        except Exception:
            pass

    run_log.info("close_session_done", msg_count=len(msgs_dicts), dna_triggered=should_recompute_dna(total))
    return Response(status_code=204)


@router.get("/history/{thread_id}", response_model=HistoryResponse)
async def history(
    thread_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    raw = await agent_service.get_messages(thread_id)
    out = []
    for m in raw:
        role = _ROLE_MAP.get(getattr(m, "type", ""), "system")
        if role == "system":
            continue
        out.append(HistoryMessage(role=role, content=getattr(m, "content", "")))
        
    log.info("history_fetched", thread_id=thread_id, message_count=len(out))
    return HistoryResponse(messages=out)