"""OpenAI-compatible model routes."""

from fastapi import APIRouter, Depends, Request

from app.auth import require_api_key

router = APIRouter(prefix="/v1", tags=["openai"])


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models(request: Request) -> dict[str, object]:
    """List the compatible local model aliases."""
    ready = request.app.state.ready
    return {
        "object": "list",
        "data": [
            {
                "id": "sensevoice",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "ready": ready,
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "ready": ready,
            },
        ],
    }
