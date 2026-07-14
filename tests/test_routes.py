from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, create_app
from app.services.asr import ServiceBusyError, Transcription
from app.services.audio import PreparedAudio

AUTH = {"Authorization": "Bearer test-key"}
AUDIO = {"file": ("audio.wav", b"audio", "audio/wav")}


@dataclass
class FakeAudioService:
    cleaned: bool = False

    async def prepare(self, _: object) -> PreparedAudio:
        return PreparedAudio(Path("input.wav"), Path("normalized.wav"), 12.48, 5)

    async def cleanup(self, _: PreparedAudio) -> None:
        self.cleaned = True


class FakeAsrService:
    busy = False

    async def transcribe(self, _: Path, __: str) -> Transcription:
        if self.busy:
            raise ServiceBusyError
        return Transcription(text="今天 review the design.", segments=[])


def test_rejects_empty_api_key_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_API_KEYS", "")

    with pytest.raises(ValueError, match="ASR_API_KEYS"):
        Settings().api_keys


def test_asgi_entrypoint_is_available() -> None:
    assert app.title == "FunASR Speech API"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ASR_API_KEYS", "test-key,rotated-key")
    return TestClient(create_app(Settings(), asr_service=FakeAsrService(), audio_service=FakeAudioService()))


def test_live_health_is_public(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer bad"}])
def test_models_rejects_missing_or_invalid_key(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/v1/models", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_models_lists_compatible_aliases(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "sensevoice",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "ready": False,
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "ready": False,
            },
        ],
    }


def test_ready_health_reports_not_ready_until_model_is_loaded(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_ready_health_reports_cpu_model_when_loaded(client: TestClient) -> None:
    client.app.state.ready = True
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "sensevoice", "device": "cpu"}


def test_request_id_is_preserved(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "request-123"})

    assert response.headers["x-request-id"] == "request-123"


@pytest.mark.parametrize(
    ("response_format", "expected_body", "content_type"),
    [
        ("json", {"text": "今天 review the design."}, "application/json"),
        ("text", "今天 review the design.", "text/plain"),
        (
            "verbose_json",
            {
                "task": "transcribe",
                "language": "auto",
                "duration": 12.48,
                "text": "今天 review the design.",
                "segments": [],
            },
            "application/json",
        ),
    ],
)
def test_transcription_response_formats(
    client: TestClient,
    response_format: str,
    expected_body: dict[str, object] | str,
    content_type: str,
) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=AUTH,
        data={"model": "whisper-1", "response_format": response_format},
        files=AUDIO,
    )

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.json() == expected_body if response_format != "text" else response.text == expected_body


@pytest.mark.parametrize(
    ("form_data", "code"),
    [
        ({"model": "unknown"}, "model_not_found"),
        ({"model": "sensevoice", "language": "ja"}, "invalid_request"),
        ({"model": "sensevoice", "response_format": "srt"}, "unsupported_parameter"),
        (
            {"model": "sensevoice", "timestamp_granularities[]": "word"},
            "unsupported_parameter",
        ),
    ],
)
def test_transcription_rejects_unsupported_options(client: TestClient, form_data: dict[str, str], code: str) -> None:
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data=form_data, files=AUDIO)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


def test_transcription_cleans_temporary_files(client: TestClient) -> None:
    audio_service = client.app.state.audio_service
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "sensevoice"}, files=AUDIO)

    assert response.status_code == 200
    assert audio_service.cleaned is True


def test_transcription_returns_busy_error(client: TestClient) -> None:
    client.app.state.asr_service.busy = True
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "sensevoice"}, files=AUDIO)

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "service_busy"


def test_transcription_validation_error_uses_openai_envelope(client: TestClient) -> None:
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "sensevoice"})

    assert response.status_code == 422
    assert response.json()["error"] == {
        "message": "Request validation failed.",
        "type": "invalid_request_error",
        "param": None,
        "code": "validation_error",
    }
