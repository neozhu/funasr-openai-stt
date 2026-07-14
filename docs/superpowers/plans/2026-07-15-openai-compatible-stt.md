# OpenAI-Compatible STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Deliver a Python 3.12 CPU-only FastAPI service with OpenAI-compatible transcription, models, and health endpoints backed by a single loaded FunASR model.

**Architecture:** \`main\` owns the FastAPI lifespan, request IDs, and router registration. Routes use application-state services: \`AsrService\` owns the loaded FunASR model and concurrency gate; \`AudioService\` owns streamed temporary uploads, ffprobe validation, and ffmpeg normalization. All automated tests call FastAPI routes with fake services.

**Tech Stack:** Python 3.12, FastAPI, Pydantic Settings, AnyIO, FunASR, CPU PyTorch, pytest, httpx, ffmpeg, Docker Compose.

## Global Constraints

- Use Python 3.12; local setup is \`python -m venv .venv\` and \`pip\`, with pins in \`requirements.txt\`.
- Run inference on CPU only and with one Uvicorn worker.
- Do not retain uploads or transcripts, fake timestamps, log sensitive request data, add CUDA, or add broad fallback recovery.
- Tests are FastAPI route tests only; replace FunASR and ffmpeg at service boundaries.
- Keep OpenAI errors shaped as \`{"error":{"message","type","param","code"}}\`.

---

### Task 1: Bootstrap configuration and error responses

**Files:**
- Create: \`requirements.txt\`, \`pyproject.toml\`, \`.env.example\`, \`.gitignore\`
- Create: \`app/__init__.py\`, \`app/config.py\`, \`app/errors.py\`
- Test: \`tests/test_routes.py\`

**Interfaces:**
- Produces: \`Settings\` and \`openai_error(status, message, type, param, code)\`.

- [ ] **Step 1: Write the failing test**

\`\`\`python
def test_rejects_empty_api_key_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_API_KEYS", "")
    with pytest.raises(ValueError, match="ASR_API_KEYS"):
        Settings().api_keys
\`\`\`

- [ ] **Step 2: Verify RED**

Run: \`python -m pytest tests/test_routes.py::test_rejects_empty_api_key_configuration -q\`

Expected: FAIL because \`Settings\` is undefined.

- [ ] **Step 3: Write minimal code**

\`\`\`python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    asr_api_keys: str

    @property
    def api_keys(self) -> tuple[str, ...]:
        keys = tuple(key.strip() for key in self.asr_api_keys.split(",") if key.strip())
        if not keys:
            raise ValueError("ASR_API_KEYS must contain at least one key")
        return keys
\`\`\`

- [ ] **Step 4: Verify GREEN**

Run: \`python -m pytest tests/test_routes.py::test_rejects_empty_api_key_configuration -q\`

Expected: PASS.

### Task 2: Implement lifespan, health routes, API-key authentication, and request IDs

**Files:**
- Create: \`app/auth.py\`, \`app/api/__init__.py\`, \`app/api/health.py\`, \`app/main.py\`
- Modify: \`app/config.py\`, \`tests/test_routes.py\`

**Interfaces:**
- Produces: \`create_app(settings, asr_service, audio_service)\`, \`GET /health/live\`, \`GET /health/ready\`, and \`require_api_key\`.

- [ ] **Step 1: Write failing route tests**

\`\`\`python
def test_live_health_is_public(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_models_rejects_missing_or_invalid_key(client: TestClient) -> None:
    for headers in ({}, {"Authorization": "Bearer bad"}):
        response = client.get("/v1/models", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "invalid_api_key"
\`\`\`

- [ ] **Step 2: Verify RED**

Run: \`python -m pytest tests/test_routes.py -k "live_health or models_rejects" -q\`

Expected: FAIL because the app and routes do not exist.

- [ ] **Step 3: Write minimal code**

\`\`\`python
async def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    scheme, _, key = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not any(compare_digest(key, item) for item in settings.api_keys):
        raise OpenAIHTTPError(401, "Invalid API key.", "authentication_error", None, "invalid_api_key")

@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}
\`\`\`

- [ ] **Step 4: Verify GREEN**

Run: \`python -m pytest tests/test_routes.py -k "live_health or models_rejects" -q\`

Expected: PASS.

### Task 3: Implement temporary audio processing and bounded ASR inference

**Files:**
- Create: \`app/services/__init__.py\`, \`app/services/audio.py\`, \`app/services/asr.py\`
- Modify: \`app/config.py\`, \`app/main.py\`, \`tests/test_routes.py\`

**Interfaces:**
- Produces: \`AudioService.prepare(upload) -> PreparedAudio\`, \`AudioService.cleanup(prepared)\`, and \`AsrService.transcribe(path, language) -> Transcription\`.

- [ ] **Step 1: Write failing route tests through fakes**

\`\`\`python
def test_transcription_cleans_files_after_success(client: TestClient, fake_audio: FakeAudio) -> None:
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "whisper-1"}, files=AUDIO)
    assert response.status_code == 200
    assert fake_audio.cleaned is True

def test_transcription_returns_busy_error(client: TestClient) -> None:
    client.app.state.asr_service.busy = True
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "sensevoice"}, files=AUDIO)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "service_busy"
\`\`\`

- [ ] **Step 2: Verify RED**

Run: \`python -m pytest tests/test_routes.py -k "cleans_files or busy_error" -q\`

Expected: FAIL because the transcription boundary is missing.

- [ ] **Step 3: Write minimal code**

\`\`\`python
@dataclass
class PreparedAudio:
    input_path: Path
    normalized_path: Path
    duration_seconds: float
    size_bytes: int

async def transcribe(self, path: Path, language: str) -> Transcription:
    async with self._slot:
        result = await asyncio.wait_for(
            anyio.to_thread.run_sync(self._generate, path, language),
            timeout=self._settings.asr_inference_timeout_seconds,
        )
    return parse_transcription(result)
\`\`\`

\`AudioService\` streams chunks to server-generated names, enforces byte limit while writing, runs \`ffprobe\` and \`ffmpeg\` with argument lists and timeouts, and always removes created files.

- [ ] **Step 4: Verify GREEN**

Run: \`python -m pytest tests/test_routes.py -k "cleans_files or busy_error" -q\`

Expected: PASS.

### Task 4: Implement OpenAI-compatible models and transcription routes

**Files:**
- Create: \`app/api/models.py\`, \`app/api/transcriptions.py\`
- Modify: \`app/api/__init__.py\`, \`app/main.py\`, \`tests/test_routes.py\`

**Interfaces:**
- Produces: authenticated \`GET /v1/models\` and \`POST /v1/audio/transcriptions\`.

- [ ] **Step 1: Write failing route tests**

\`\`\`python
@pytest.mark.parametrize(("response_format", "content_type"), [
    ("json", "application/json"),
    ("text", "text/plain"),
    ("verbose_json", "application/json"),
])
def test_transcription_response_formats(client: TestClient, response_format: str, content_type: str) -> None:
    response = client.post(
        "/v1/audio/transcriptions", headers=AUTH,
        data={"model": "whisper-1", "response_format": response_format}, files=AUDIO,
    )
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]

def test_unknown_model_is_openai_error(client: TestClient) -> None:
    response = client.post("/v1/audio/transcriptions", headers=AUTH, data={"model": "unknown"}, files=AUDIO)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_not_found"
\`\`\`

- [ ] **Step 2: Verify RED**

Run: \`python -m pytest tests/test_routes.py -k "response_formats or unknown_model" -q\`

Expected: FAIL because these routes do not exist.

- [ ] **Step 3: Write minimal code**

\`\`\`python
MODEL_ALIASES = {"sensevoice", "SenseVoiceSmall", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}
LANGUAGES = {"auto", "zh", "en"}
FORMATS = {"json", "text", "verbose_json"}

@router.post("/v1/audio/transcriptions", dependencies=[Depends(require_api_key)])
async def create_transcription(...) -> Response:
    validate_options(model, language, response_format, timestamp_granularities)
    prepared = await audio_service.prepare(file)
    try:
        output = await asr_service.transcribe(prepared.normalized_path, language or "auto")
        return render_transcription(output, language or "auto", response_format, prepared.duration_seconds)
    finally:
        await audio_service.cleanup(prepared)
\`\`\`

- [ ] **Step 4: Verify GREEN and cover route errors**

Add route cases for all model aliases, \`zh\`/ \`en\`, unsupported language/format/word timestamps, empty file, size limit, 503 readiness, \`/v1/models\`, and readiness output.

Run: \`python -m pytest tests/test_routes.py -q\`

Expected: PASS.

### Task 5: Add Docker deployment, documentation, and quality verification

**Files:**
- Create: \`Dockerfile\`, \`compose.yaml\`, \`.dockerignore\`, \`scripts/smoke_test.py\`
- Modify: \`README.md\`, \`.env.example\`, \`tests/test_routes.py\`

**Interfaces:**
- Consumes: \`app.main:app\` and documented environment configuration.
- Produces: a non-root CPU image, persistent \`/models\` cache, and documented local/Docker workflows.

- [ ] **Step 1: Write the failing readiness contract test**

\`\`\`python
def test_ready_health_reports_cpu_model(client: TestClient) -> None:
    client.app.state.ready = True
    response = client.get("/health/ready")
    assert response.json() == {"status": "ready", "model": "sensevoice", "device": "cpu"}
\`\`\`

- [ ] **Step 2: Verify RED**

Run: \`python -m pytest tests/test_routes.py::test_ready_health_reports_cpu_model -q\`

Expected: FAIL until the exact readiness contract is implemented.

- [ ] **Step 3: Implement deployment and documentation**

Use \`python:3.12-slim-bookworm\`, install only \`ffmpeg libsndfile1 curl ca-certificates\`, create a non-root user, install pinned dependencies before copying \`app\`, use one Uvicorn worker, and health-check \`/health/live\`. Compose must use the named \`/models\` volume and a \`/tmp\` tmpfs. README documents virtualenv setup, model download, audio limits, authentication, curl/OpenAI use, Docker, tuning, security, and compatibility limitations.

- [ ] **Step 4: Verify GREEN and repository quality**

Run: \`python -m pytest tests/test_routes.py -q && python -m ruff check . && python -m ruff format --check . && python -m mypy app\`

Expected: every command exits 0.

## Plan Self-Review

- Every required area has a task: lifecycle/model load, API compatibility, auth, audio processing and cleanup, CPU concurrency, health, Docker/Compose, docs, and Python 3.12 local setup.
- The test plan only contains FastAPI route tests and uses fakes for FunASR/ffmpeg.
- Public paths, interfaces, expected test outcomes, and validation commands are specified without placeholders.

