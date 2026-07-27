"""API authentication: Bearer token validation."""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings

_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """
    FastAPI dependency: validates the Bearer token.

    Raises HTTP 401 if missing or invalid.
    Returns the validated API key on success.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unauthorized", "message": "Authorization header required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    provided = credentials.credentials.encode()
    expected = settings.MERIDIAN_API_KEY.encode()

    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unauthorized", "message": "Invalid API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
