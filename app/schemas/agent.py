from typing import Literal, Any
from pydantic import BaseModel


class ObserveRequest(BaseModel):
    thread_id: str
    event: Literal["game_start", "blunder", "game_end", "arrival"]
    payload: dict[str, Any] = {}


class MessageRequest(BaseModel):
    thread_id: str
    text: str


class CloseSessionRequest(BaseModel):
    thread_id: str


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]
