"""OpenAI-compatible transcription route."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from app.auth import require_api_key
from app.errors import OpenAIHTTPError
from app.services.asr import InferenceTimeoutError, ServiceBusyError, Transcription

router = APIRouter(prefix="/v1/audio", tags=["openai"])

MODEL_ALIASES = {
    "sensevoice",
    "SenseVoiceSmall",
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
}
LANGUAGES = {"auto", "zh", "en"}
FORMATS = {"json", "text", "verbose_json"}


@router.post("/transcriptions", dependencies=[Depends(require_api_key)])
async def create_transcription(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    prompt: Annotated[str | None, Form()] = None,
    temperature: Annotated[float | None, Form()] = None,
    timestamp_granularities: Annotated[list[str] | None, Form(alias="timestamp_granularities[]")] = None,
) -> Response:
    """Transcribe one uploaded audio file using SenseVoiceSmall."""
    del prompt, temperature
    _validate_options(model, language, response_format, timestamp_granularities)
    audio_service = request.app.state.audio_service
    prepared = await audio_service.prepare(file)
    try:
        try:
            transcription = await request.app.state.asr_service.transcribe(prepared.normalized_path, language or "auto")
        except ServiceBusyError as exc:
            raise OpenAIHTTPError(
                429,
                "The transcription service is currently busy.",
                "rate_limit_error",
                None,
                "service_busy",
            ) from exc
        except InferenceTimeoutError as exc:
            raise OpenAIHTTPError(
                504,
                "Transcription inference timed out.",
                "server_error",
                None,
                "inference_timeout",
            ) from exc
        return _render(transcription, language or "auto", response_format, prepared.duration_seconds)
    finally:
        await audio_service.cleanup(prepared)


def _validate_options(
    model: str,
    language: str | None,
    response_format: str,
    timestamp_granularities: list[str] | None,
) -> None:
    if model not in MODEL_ALIASES:
        raise OpenAIHTTPError(
            400,
            "The requested model does not exist.",
            "invalid_request_error",
            "model",
            "model_not_found",
        )
    if language is not None and language not in LANGUAGES:
        raise OpenAIHTTPError(400, "Unsupported language.", "invalid_request_error", "language", "invalid_request")
    if response_format not in FORMATS:
        raise OpenAIHTTPError(
            400,
            "Unsupported response format.",
            "invalid_request_error",
            "response_format",
            "unsupported_parameter",
        )
    if timestamp_granularities and any(value != "segment" for value in timestamp_granularities):
        raise OpenAIHTTPError(
            400,
            "Word-level timestamps are not supported.",
            "invalid_request_error",
            "timestamp_granularities",
            "unsupported_parameter",
        )


def _render(transcription: Transcription, language: str, response_format: str, duration: float) -> Response:
    if response_format == "text":
        return PlainTextResponse(transcription.text)
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": language,
                "duration": duration,
                "text": transcription.text,
                "segments": transcription.segments,
            }
        )
    return JSONResponse({"text": transcription.text})
