"""Health routes."""

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Report that the HTTP process is running."""
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request) -> Response:
    """Report whether the model has completed startup."""
    if not request.app.state.ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready", "model": "sensevoice", "device": "cpu"})
