from dataclasses import dataclass

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Profile

_bearer = HTTPBearer()

# Cache JWKS public keys so we only fetch them once
_jwks_cache: dict[str, dict] = {}


def _get_jwks_keys() -> dict[str, dict]:
    """Fetch and cache JWKS public keys from Supabase."""
    if _jwks_cache:
        return _jwks_cache
    jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
    resp = httpx.get(jwks_url, timeout=10)
    resp.raise_for_status()
    for key_data in resp.json().get("keys", []):
        kid = key_data["kid"]
        _jwks_cache[kid] = key_data
    return _jwks_cache


def _decode_token(token: str) -> dict:
    # Peek at the header to get kid and alg
    try:
        header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc

    alg = header.get("alg", "HS256")

    if alg == "ES256":
        kid = header.get("kid")
        keys = _get_jwks_keys()
        if kid not in keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unknown signing key",
            )
        public_key = keys[kid]
        try:
            return jwt.decode(
                token,
                public_key,
                algorithms=["ES256"],
                audience="authenticated",
            )
        except JWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            ) from exc
    else:
        # Legacy HS256 path
        try:
            return jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except JWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            ) from exc


@dataclass
class CurrentUser:
    user_id: str
    plan: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    payload = _decode_token(credentials.credentials)
    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing sub claim")

    result = await db.execute(select(Profile).where(Profile.user_id == user_id))
    profile = result.scalar_one_or_none()
    if profile is None:
        profile = Profile(user_id=user_id, plan="free")
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return CurrentUser(user_id=user_id, plan=profile.plan)


async def require_pro(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.plan != "pro":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="pro_required")
    return user