"""Bearer API-key authentication."""

from __future__ import annotations

from hmac import compare_digest
from typing import Annotated, cast

from fastapi import Header, Request

from app.config import Settings
from app.errors import OpenAIHTTPError


def get_settings(request: Request) -> Settings:
    """Retrieve the application settings from application state."""
    return cast(Settings, request.app.state.settings)


async def require_api_key(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require a configured Bearer token using constant-time comparisons."""
    scheme, _, key = (authorization or "").partition(" ")
    settings = get_settings(request)
    if scheme.lower() != "bearer" or not any(compare_digest(key, item) for item in settings.api_keys):
        raise OpenAIHTTPError(
            401,
            "Invalid API key.",
            "authentication_error",
            None,
            "invalid_api_key",
            {"WWW-Authenticate": "Bearer"},
        )
