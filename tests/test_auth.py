import pytest
from fastapi import HTTPException
from jose import jwt

from app.config import settings
from app.dependencies import _decode_token


def _make_token(sub: str = "user-1", aud: str = "authenticated", **extra) -> str:
    payload = {"sub": sub, "aud": aud, **extra}
    return jwt.encode(payload, settings.supabase_jwt_secret, algorithm="HS256")


def test_valid_token_decodes():
    token = _make_token(sub="abc-123")
    payload = _decode_token(token)
    assert payload["sub"] == "abc-123"


def test_invalid_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        _decode_token("not.a.valid.token")
    assert exc_info.value.status_code == 401


def test_wrong_secret_raises_401():
    bad_token = jwt.encode(
        {"sub": "x", "aud": "authenticated"},
        "wrong-secret-that-is-long-enough-xxxxx!",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc_info:
        _decode_token(bad_token)
    assert exc_info.value.status_code == 401


def test_wrong_audience_raises_401():
    token = _make_token(sub="x", aud="anon")
    with pytest.raises(HTTPException) as exc_info:
        _decode_token(token)
    assert exc_info.value.status_code == 401