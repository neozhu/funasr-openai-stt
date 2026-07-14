"""OpenAI-compatible HTTP errors."""

from __future__ import annotations

from typing import cast

from fastapi import Request
from fastapi.responses import JSONResponse


class OpenAIHTTPError(Exception):
    """An HTTP error rendered using the OpenAI error envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        param: str | None,
        code: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        }
        self.headers = headers


async def openai_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Serialize service errors without exposing implementation details."""
    error = cast(OpenAIHTTPError, exc)
    return JSONResponse(error.detail, status_code=error.status_code, headers=error.headers)


async def validation_error_handler(_: Request, __: Exception) -> JSONResponse:
    """Return request validation failures in the public error envelope."""
    return JSONResponse(
        {
            "error": {
                "message": "Request validation failed.",
                "type": "invalid_request_error",
                "param": None,
                "code": "validation_error",
            }
        },
        status_code=422,
    )
