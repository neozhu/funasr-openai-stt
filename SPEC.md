# FastAPI FunASR Speech-to-Text Service — Development Specification

## 1. Project Overview

Build a production-ready, CPU-based Speech-to-Text backend service using:

* FastAPI
* FunASR
* SenseVoiceSmall
* FSMN-VAD
* PyTorch CPU runtime
* Docker
* Docker Compose

The service must recognize:

* Mandarin Chinese
* English
* Mixed Chinese and English speech

The public API must be compatible with the OpenAI Speech-to-Text API structure so that existing OpenAI SDK clients can use the service by changing only `base_url` and `api_key`.

Recommended repository name:

```text
funasr-openai-stt-service
```

Recommended service name:

```text
FunASR Speech API
```

---

## 2. Primary Goals

The service must:

1. Run entirely on CPU.
2. Load the ASR model once during application startup.
3. Accept uploaded audio files.
4. Return transcribed text.
5. Support Chinese, English, and mixed-language audio.
6. Expose an OpenAI-compatible transcription endpoint.
7. Authenticate requests using Bearer API keys.
8. Support Docker and Docker Compose local deployment.
9. Persist downloaded model files between container restarts.
10. Include health checks, structured logging, tests, and safe temporary-file handling.

---

## 3. Non-Goals

The first version must not include:

* Real-time WebSocket transcription
* Microphone streaming
* Speaker diarization
* Audio translation
* GPU/CUDA support
* Batch job management
* Persistent storage of uploaded audio
* User account management
* Database dependencies
* Web frontend

These features should remain outside the initial implementation to keep the service maintainable and predictable.

---

## 4. Technology Stack

Use:

```text
Python 3.11
FastAPI
Uvicorn
FunASR
PyTorch CPU
python-multipart
Pydantic Settings
ffmpeg / ffprobe
structlog or standard JSON logging
pytest
httpx
ruff
mypy
```

Use a lock file and pin all tested dependency versions.

Recommended package management:

```text
uv
```

Required files:

```text
pyproject.toml
uv.lock
```

Do not install CUDA-related dependencies.

---

## 5. ASR Model Configuration

Use the following FunASR pipeline:

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={
        "max_single_segment_time": 30_000,
    },
    device="cpu",
    disable_update=True,
)
```

For inference:

```python
result = model.generate(
    input=audio_path,
    language="auto",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
```

Process final output using:

```python
from funasr.utils.postprocess_utils import rich_transcription_postprocess
```

The model must be:

* Loaded once using FastAPI lifespan management.
* Stored in application state or a dedicated singleton service.
* Warmed up during startup using a short bundled audio sample.
* Marked ready only after model loading and warmup succeed.

Do not load the model inside an endpoint handler.

---

## 6. Language Handling

Supported language values:

```text
auto
zh
en
```

Behavior:

* Missing `language`: use `auto`.
* `language=auto`: recommended for Chinese-English mixed audio.
* `language=zh`: optimize for primarily Chinese audio.
* `language=en`: optimize for primarily English audio.
* ISO values such as `zh` and `en` must be accepted.
* Unsupported language values must return HTTP 400.

Do not force `language=zh` for mixed-language recordings.

---

## 7. OpenAI-Compatible API

### 7.1 Transcription Endpoint

```http
POST /v1/audio/transcriptions
```

Content type:

```text
multipart/form-data
```

Authentication:

```http
Authorization: Bearer <api-key>
```

### Request Fields

| Field                       | Required | Type        | Description                       |
| --------------------------- | -------: | ----------- | --------------------------------- |
| `file`                      |      Yes | Binary file | Audio file                        |
| `model`                     |      Yes | String      | OpenAI or local model alias       |
| `language`                  |       No | String      | `auto`, `zh`, or `en`             |
| `response_format`           |       No | String      | `json`, `text`, or `verbose_json` |
| `prompt`                    |       No | String      | Accepted for client compatibility |
| `temperature`               |       No | Number      | Accepted but not applied          |
| `timestamp_granularities[]` |       No | Array       | Only `segment` may be supported   |

### Accepted Model Aliases

All of the following aliases should map to SenseVoiceSmall:

```text
sensevoice
SenseVoiceSmall
whisper-1
gpt-4o-transcribe
gpt-4o-mini-transcribe
```

The actual model implementation remains:

```text
iic/SenseVoiceSmall
```

This alias mapping allows applications already using OpenAI clients to migrate without changing their model configuration immediately.

Unknown models must return:

```http
HTTP 400
```

with OpenAI-style error code:

```text
model_not_found
```

---

## 8. Response Formats

### 8.1 JSON

Request:

```text
response_format=json
```

Response:

```json
{
  "text": "今天我们需要 review the Salesforce integration design."
}
```

Content type:

```text
application/json
```

---

### 8.2 Plain Text

Request:

```text
response_format=text
```

Response body:

```text
今天我们需要 review the Salesforce integration design.
```

Content type:

```text
text/plain; charset=utf-8
```

---

### 8.3 Verbose JSON

Request:

```text
response_format=verbose_json
```

Response:

```json
{
  "task": "transcribe",
  "language": "auto",
  "duration": 12.48,
  "text": "今天我们需要 review the Salesforce integration design.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 6.32,
      "text": "今天我们需要 review",
      "speaker": null
    },
    {
      "id": 1,
      "start": 6.32,
      "end": 12.48,
      "text": "the Salesforce integration design.",
      "speaker": null
    }
  ]
}
```

When reliable segment timestamps are unavailable, return:

```json
{
  "task": "transcribe",
  "language": "auto",
  "duration": 12.48,
  "text": "Transcribed text",
  "segments": []
}
```

Do not generate fake timestamps.

Word-level timestamps are not required in the first version.

When the client explicitly requests:

```text
timestamp_granularities[]=word
```

return HTTP 400 with:

```json
{
  "error": {
    "message": "Word-level timestamps are not supported.",
    "type": "invalid_request_error",
    "param": "timestamp_granularities",
    "code": "unsupported_parameter"
  }
}
```

---

## 9. API Key Authentication

API keys must be supplied through an environment variable:

```env
ASR_API_KEYS=key-one,key-two,key-three
```

Support multiple keys to allow key rotation.

Authentication rules:

1. Read the `Authorization` header.
2. Require the `Bearer` scheme.
3. Compare keys using `hmac.compare_digest`.
4. Never log API keys.
5. Strip whitespace around configured keys.
6. Reject empty API key configuration during startup.
7. Return the same response for missing and invalid keys.

Unauthorized response:

```http
HTTP 401
WWW-Authenticate: Bearer
```

```json
{
  "error": {
    "message": "Invalid API key.",
    "type": "authentication_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

The following endpoints require authentication:

```text
/v1/audio/transcriptions
/v1/models
```

The minimal liveness endpoint may remain unauthenticated:

```text
/health/live
```

---

## 10. Models Endpoint

Expose:

```http
GET /v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "sensevoice",
      "object": "model",
      "created": 0,
      "owned_by": "local",
      "ready": true
    },
    {
      "id": "whisper-1",
      "object": "model",
      "created": 0,
      "owned_by": "local",
      "ready": true
    }
  ]
}
```

---

## 11. Health Endpoints

### Liveness

```http
GET /health/live
```

Response:

```json
{
  "status": "ok"
}
```

This endpoint must not inspect or expose model paths, API keys, system details, or environment variables.

### Readiness

```http
GET /health/ready
```

Response when ready:

```json
{
  "status": "ready",
  "model": "sensevoice",
  "device": "cpu"
}
```

Response when the model is unavailable:

```http
HTTP 503
```

```json
{
  "status": "not_ready"
}
```

Docker health checks should use:

```text
/health/live
```

Deployment monitoring should use:

```text
/health/ready
```

---

## 12. Audio File Support

Support at least:

```text
wav
mp3
mp4
mpeg
mpga
m4a
webm
flac
ogg
```

Default maximum upload size:

```env
ASR_MAX_UPLOAD_MB=25
```

Default maximum audio duration:

```env
ASR_MAX_AUDIO_DURATION_SECONDS=3600
```

Reject files exceeding configured limits before inference.

Do not trust only:

* Filename extension
* Client-provided MIME type
* `Content-Length`

Validate the audio using `ffprobe`.

---

## 13. Audio Processing Pipeline

For every request:

1. Authenticate the request.
2. Validate the model alias and form parameters.
3. Stream the upload to a temporary file.
4. Do not load the entire upload into memory.
5. Enforce the upload-size limit while streaming.
6. Inspect audio metadata using `ffprobe`.
7. Reject invalid, unsupported, or oversized audio.
8. Normalize the audio using `ffmpeg`.
9. Run inference in a worker thread.
10. Format the response.
11. Delete all temporary files in a `finally` block.

Normalize input to:

```text
WAV
16 kHz
mono
PCM signed 16-bit little-endian
```

Example ffmpeg operation:

```bash
ffmpeg \
  -nostdin \
  -hide_banner \
  -loglevel error \
  -i input-file \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  output.wav
```

Use a subprocess timeout.

Do not use `shell=True`.

Never include user-provided filenames directly in shell commands.

---

## 14. Temporary File Handling

Use a dedicated configurable directory:

```env
ASR_TEMP_DIR=/tmp/funasr
```

Requirements:

* Generate server-side random filenames.
* Do not preserve the original filename.
* Clean files after successful requests.
* Clean files after failed requests.
* Clean abandoned files during startup.
* Do not persist audio or transcripts by default.
* Do not log transcript content unless explicitly enabled.

Optional cleanup configuration:

```env
ASR_TEMP_FILE_MAX_AGE_MINUTES=60
```

---

## 15. CPU Concurrency Model

FunASR inference is blocking and CPU-intensive.

Use:

```python
await anyio.to_thread.run_sync(...)
```

or:

```python
await asyncio.to_thread(...)
```

Do not run inference directly on the FastAPI event loop.

Use a global semaphore:

```env
ASR_MAX_CONCURRENT_REQUESTS=1
ASR_QUEUE_TIMEOUT_SECONDS=10
```

Recommended defaults:

```text
ASR_MAX_CONCURRENT_REQUESTS=1
UVICORN_WORKERS=1
```

When all inference slots are busy:

* Wait up to `ASR_QUEUE_TIMEOUT_SECONDS`.
* Return HTTP 429 after the timeout.

Response:

```json
{
  "error": {
    "message": "The transcription service is currently busy.",
    "type": "rate_limit_error",
    "param": null,
    "code": "service_busy"
  }
}
```

Do not increase Uvicorn workers to improve throughput because every worker would load its own model and consume additional memory.

Scale vertically by adjusting:

```env
ASR_CPU_THREADS
OMP_NUM_THREADS
MKL_NUM_THREADS
```

Scale horizontally by running multiple independent service containers behind a reverse proxy only after measuring memory usage and latency.

---

## 16. Application Lifecycle

Use FastAPI lifespan management.

Startup sequence:

1. Load settings.
2. Validate API keys.
3. Configure PyTorch CPU threads.
4. Create temporary directories.
5. Remove abandoned temporary files.
6. Load SenseVoiceSmall.
7. Run model warmup.
8. Set application readiness state.

Shutdown sequence:

1. Stop accepting new inference jobs.
2. Wait for active requests up to a configured timeout.
3. Clean temporary files.
4. Release model references.
5. Shut down cleanly.

Model load failures must stop the application instead of starting an unusable service.

---

## 17. Configuration

Provide `.env.example`:

```env
APP_NAME=FunASR Speech API
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

ASR_API_KEYS=replace-with-a-long-random-api-key

ASR_MODEL_ID=iic/SenseVoiceSmall
ASR_MODEL_ALIAS=sensevoice
ASR_DEVICE=cpu
ASR_LANGUAGE=auto
ASR_USE_ITN=true

ASR_MAX_UPLOAD_MB=25
ASR_MAX_AUDIO_DURATION_SECONDS=3600
ASR_MAX_CONCURRENT_REQUESTS=1
ASR_QUEUE_TIMEOUT_SECONDS=10
ASR_INFERENCE_TIMEOUT_SECONDS=600

ASR_CPU_THREADS=8
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8

ASR_TEMP_DIR=/tmp/funasr
ASR_TEMP_FILE_MAX_AGE_MINUTES=60

MODELSCOPE_CACHE=/models/modelscope
HF_HOME=/models/huggingface

ASR_LOG_TRANSCRIPT=false
ASR_ENABLE_DOCS=true
```

Use Pydantic Settings.

Secrets must not have default production values.

---

## 18. Suggested Project Structure

```text
funasr-openai-stt-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── lifespan.py
│   ├── config.py
│   ├── dependencies.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── models.py
│   │   └── transcriptions.py
│   ├── auth/
│   │   ├── __init__.py
│   │   └── api_key.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── models.py
│   │   └── transcriptions.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr_service.py
│   │   ├── audio_service.py
│   │   └── temp_file_service.py
│   └── observability/
│       ├── __init__.py
│       └── logging.py
├── scripts/
│   ├── download_models.py
│   └── smoke_test.py
├── tests/
│   ├── unit/
│   │   ├── test_auth.py
│   │   ├── test_config.py
│   │   ├── test_errors.py
│   │   └── test_response_formats.py
│   ├── integration/
│   │   ├── test_health.py
│   │   ├── test_models.py
│   │   └── test_transcriptions.py
│   └── assets/
│       ├── chinese.wav
│       ├── english.wav
│       └── mixed-zh-en.wav
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .env.example
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

---

## 19. Error Response Format

All API errors must follow an OpenAI-style structure:

```json
{
  "error": {
    "message": "Human-readable error message.",
    "type": "invalid_request_error",
    "param": "file",
    "code": "invalid_audio_file"
  }
}
```

Required mappings:

| HTTP status | Code                       |
| ----------: | -------------------------- |
|         400 | `invalid_request`          |
|         400 | `unsupported_audio_format` |
|         400 | `invalid_audio_file`       |
|         400 | `unsupported_parameter`    |
|         401 | `invalid_api_key`          |
|         413 | `file_too_large`           |
|         422 | `validation_error`         |
|         429 | `service_busy`             |
|         500 | `internal_error`           |
|         503 | `model_not_ready`          |
|         504 | `inference_timeout`        |

Do not return Python tracebacks to clients.

---

## 20. Request IDs and Logging

Generate a request ID for every request.

Accept an incoming:

```http
X-Request-ID
```

or generate a UUID.

Return:

```http
X-Request-ID: <request-id>
```

Use structured JSON logs containing:

```text
timestamp
level
request_id
method
path
status_code
model
language
response_format
audio_duration_seconds
audio_size_bytes
queue_wait_ms
inference_ms
total_ms
error_code
```

Do not log:

* API keys
* Authorization headers
* Raw audio
* Raw transcript text
* Temporary file paths containing user data
* Original filenames

---

## 21. Dockerfile Requirements

Use a CPU-only base image.

Recommended base:

```dockerfile
FROM python:3.11-slim-bookworm
```

Install only required system packages:

```text
ffmpeg
libsndfile1
curl
ca-certificates
```

Requirements:

* Use a non-root user.
* Use a fixed working directory.
* Install dependencies before copying application source for build caching.
* Use a lock file.
* Do not include `.env`.
* Do not include test audio in the production image unless needed for warmup.
* Use one Uvicorn worker.
* Add a Docker health check.
* Write temporary data only to `/tmp`.
* Store model cache under `/models`.

Entrypoint:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips="*"
```

Do not enable `--reload` in the production image.

---

## 22. Docker Compose Requirements

Provide a `compose.yaml` similar to:

```yaml
services:
  funasr-stt:
    build:
      context: .
      dockerfile: Dockerfile
    image: funasr-openai-stt-service:local
    container_name: funasr-stt
    restart: unless-stopped

    env_file:
      - .env

    ports:
      - "48733:8000"

    volumes:
      - funasr-models:/models

    tmpfs:
      - /tmp:size=1g,mode=1777

    environment:
      APP_HOST: 0.0.0.0
      APP_PORT: 8000
      ASR_DEVICE: cpu
      MODELSCOPE_CACHE: /models/modelscope
      HF_HOME: /models/huggingface

    healthcheck:
      test:
        [
          "CMD",
          "curl",
          "--fail",
          "--silent",
          "http://localhost:8000/health/live"
        ]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 180s

    security_opt:
      - no-new-privileges:true

    cap_drop:
      - ALL

volumes:
  funasr-models:
```

Model cache must survive container recreation.

The first startup may download model files. Subsequent startups must reuse the persistent volume.

---

## 23. OpenAI Python SDK Example

The service must work with:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ["ASR_API_KEY"],
)

with open("mixed-language.wav", "rb") as audio_file:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json",
    )

print(result.text)
```

Only the following client configuration should need changing when migrating from OpenAI:

```python
base_url="http://localhost:8000/v1"
api_key="<local-api-key>"
```

---

## 24. cURL Example

```bash
curl --request POST \
  --url http://localhost:8000/v1/audio/transcriptions \
  --header "Authorization: Bearer ${ASR_API_KEY}" \
  --form "file=@mixed-language.wav" \
  --form "model=whisper-1" \
  --form "response_format=json"
```

For mixed Chinese-English audio, omit the language parameter or use:

```bash
--form "language=auto"
```

Verbose response:

```bash
curl --request POST \
  --url http://localhost:8000/v1/audio/transcriptions \
  --header "Authorization: Bearer ${ASR_API_KEY}" \
  --form "file=@meeting.wav" \
  --form "model=sensevoice" \
  --form "language=auto" \
  --form "response_format=verbose_json"
```

---

## 25. Testing Requirements

### Unit Tests

Test:

* Valid API key
* Invalid API key
* Missing Authorization header
* Multiple configured API keys
* Constant-time key comparison path
* Model alias mapping
* Language validation
* Response format validation
* OpenAI-style error serialization
* Upload-size enforcement
* Temporary-file cleanup
* Semaphore timeout
* Transcript result parsing

Mock the FunASR model in unit tests.

### Integration Tests

Test using real audio files:

1. Chinese-only audio
2. English-only audio
3. Chinese-English mixed audio
4. Audio containing silence
5. Invalid audio file
6. Unsupported extension
7. Empty file
8. File above upload limit
9. Unauthorized request
10. Concurrent requests
11. `json` response
12. `text` response
13. `verbose_json` response
14. OpenAI Python SDK compatibility

### Docker Smoke Test

The following command must succeed:

```bash
docker compose up --build -d
```

Then:

```bash
curl --fail http://localhost:8000/health/live
```

And:

```bash
python scripts/smoke_test.py \
  --base-url http://localhost:8000/v1 \
  --api-key "$ASR_API_KEY" \
  --audio tests/assets/mixed-zh-en.wav
```

---

## 26. Code Quality Requirements

The implementation must:

* Use type annotations.
* Pass `ruff check`.
* Pass `ruff format --check`.
* Pass `mypy`.
* Pass all pytest tests.
* Avoid global mutable state except controlled application state.
* Keep API routing separate from inference implementation.
* Keep audio conversion separate from ASR inference.
* Use dependency injection for authentication and service access.
* Avoid broad `except Exception` unless converting errors at the application boundary.
* Preserve exception chaining using `raise ... from exc`.
* Ensure all subprocesses have timeouts.
* Ensure all temporary resources use context managers or `finally` cleanup.
* Include docstrings for public classes and methods.

---

## 27. README Requirements

The README must include:

1. Project purpose
2. Architecture overview
3. Hardware assumptions
4. Supported audio formats
5. Supported languages
6. Environment variables
7. Local Python setup
8. Docker build instructions
9. Docker Compose deployment
10. First-start model download behavior
11. API authentication
12. OpenAI SDK example
13. cURL example
14. Health checks
15. Troubleshooting
16. CPU performance tuning
17. Security considerations
18. Known compatibility limitations

---

## 28. OpenAI Compatibility Limitations

Document these limitations clearly:

* The API shape is compatible with OpenAI transcription clients, but the underlying model is SenseVoiceSmall.
* `prompt` is accepted for compatibility but is not applied in the first version.
* `temperature` is accepted but ignored because SenseVoiceSmall inference is not sampling-based.
* `stream=true` is not supported.
* Word-level timestamps are not supported.
* `srt`, `vtt`, and `diarized_json` are not supported initially.
* `/v1/audio/translations` is not implemented.
* Speaker diarization is not implemented.
* Model aliases do not change the underlying inference model.
* Recognition quality must be validated against representative production recordings.

Unsupported parameters should either:

1. Be safely accepted and documented as ignored, or
2. Return an explicit OpenAI-style `unsupported_parameter` error.

Do not silently claim that unsupported functionality was executed.

---

## 29. Security Requirements

Before exposing the service outside a trusted network:

* Place it behind HTTPS.
* Use a reverse proxy such as Traefik, NGINX, or Caddy.
* Configure request body limits at both proxy and application layers.
* Configure proxy read timeouts for long audio.
* Apply IP restrictions or rate limiting where appropriate.
* Store API keys in environment secrets or a secret manager.
* Do not expose `/docs` publicly unless needed.
* Do not log transcripts by default.
* Do not retain audio by default.
* Do not run the container as root.

The FastAPI API-key middleware remains required even when a reverse proxy also provides authentication.

---

## 30. Acceptance Criteria

The implementation is complete when:

1. `docker compose up --build` starts the service successfully.
2. The model runs without CUDA or GPU dependencies.
3. The model is loaded only once.
4. Chinese audio returns Chinese text.
5. English audio returns English text.
6. Mixed-language audio preserves both Chinese and English content.
7. OpenAI Python SDK calls succeed using a custom `base_url`.
8. Requests without a valid Bearer API key return HTTP 401.
9. Uploaded files are never retained after request completion.
10. Files exceeding the configured limit return HTTP 413.
11. Invalid audio returns an OpenAI-style error.
12. Concurrent overload returns HTTP 429 rather than exhausting memory.
13. `/health/live` and `/health/ready` work correctly.
14. Model cache survives container recreation.
15. Unit, integration, and Docker smoke tests pass.
16. The application runs with one Uvicorn worker.
17. The container runs as a non-root user.
18. Logs do not contain API keys, audio, or transcript contents.
