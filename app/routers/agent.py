"""Agent chat endpoints."""
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from langchain_core.messages import HumanMessage
from sqlalchemy import select, func
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

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _extract_jwt(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.lower().startswith("bearer ") else ""


def _format_observation_message(event: str, payload: dict) -> str:
    if event == "blunder":
        return (
            f"[OBSERVATION] Ply {payload.get('ply')}: blunder, "
            f"san={payload.get('san')}, eval {payload.get('eval_before')}→"
            f"{payload.get('eval_after')}cp, {payload.get('time_taken')}s think."
        )
    if event == "game_start":
        return "[OBSERVATION] Game started — watch and stay quiet unless asked."
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

    if req.event == "game_start":
        async def _empty():
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    if req.event == "game_end":
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
        tilt_text = (
            "The game just ended. Here's the tilt analysis:\n"
            + json.dumps({
                "headline": tr.headline,
                "diagnosis": tr.diagnosis,
                "pattern_label": tr.pattern_label,
                "evidence_plies": tr.evidence_plies,
                "suggestion": tr.suggestion,
            })
            + "\n\nReact conversationally — don't repeat the report verbatim, riff on it."
        )
        new_msgs = [HumanMessage(tilt_text)]
    else:
        new_msgs = [HumanMessage(_format_observation_message(req.event, req.payload))]

    return StreamingResponse(
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
    existing = await db.execute(
        select(GameSummary).where(GameSummary.game_id == req.thread_id)
    )
    if existing.scalar_one_or_none() is not None:
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
        return Response(status_code=204)

    db.add(GameSummary(
        user_id=user.user_id, game_id=game.id,
        summary=summary.summary, key_facts=summary.key_facts,
    ))
    await db.commit()

    total = (await db.execute(
        select(func.count()).where(Game.user_id == user.user_id)
    )).scalar_one()
    if should_recompute_dna(total):
        try:
            await run_dna_for_user(db, user.user_id)
        except Exception:
            pass

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
    return HistoryResponse(messages=out)