"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.transcriptions import router as transcriptions_router
from app.config import Settings
from app.errors import OpenAIHTTPError, openai_error_handler, validation_error_handler
from app.services.asr import AsrService
from app.services.audio import AudioService


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate or create a request identifier for each HTTP response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def create_app(
    settings: Settings,
    asr_service: AsrService | None = None,
    audio_service: AudioService | None = None,
) -> FastAPI:
    """Create the configured HTTP application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        settings.api_keys
        if audio_service is None:
            application.state.audio_service = AudioService(settings)
            application.state.audio_service.cleanup_abandoned()
        if asr_service is None:
            service = AsrService(settings)
            await service.load()
            await service.warmup()
            application.state.asr_service = service
        application.state.ready = True
        yield
        application.state.ready = False

    app = FastAPI(
        title=settings.app_name,
        docs_url="/docs" if settings.asr_enable_docs else None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.ready = False
    app.state.asr_service = asr_service
    app.state.audio_service = audio_service
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(OpenAIHTTPError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(transcriptions_router)
    return app


app = create_app(Settings())
