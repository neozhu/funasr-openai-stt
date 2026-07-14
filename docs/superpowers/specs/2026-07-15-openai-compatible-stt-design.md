# OpenAI-Compatible STT Service Design

## Goal

Build a CPU-only FastAPI service that exposes OpenAI-compatible transcription and model endpoints backed by FunASR SenseVoiceSmall.

## Architecture

The application is a single FastAPI process. `app.main` installs request-ID and JSON logging middleware, registers health and OpenAI-compatible routes, and owns a lifespan that validates settings, prepares temporary storage, loads and warms the model once, and exposes a ready state.

Routes are deliberately thin. Dependencies perform Bearer-key authentication and retrieve services from `app.state`. `AudioService` streams uploads to a server-generated temporary file, probes and normalizes them through `ffprobe` and `ffmpeg`, then removes every generated file. `AsrService` serializes CPU inference through an async semaphore and calls the loaded FunASR model in a worker thread. Formatting stays in the transcription route/schema layer so the ASR service returns only parsed transcription data.

## API Behavior

`POST /v1/audio/transcriptions` accepts the OpenAI multipart shape. It maps supported aliases to SenseVoiceSmall, accepts `auto`, `zh`, and `en`, ignores compatible `prompt` and `temperature` fields, and returns `json`, `text`, or `verbose_json`. Unsupported models, languages, response formats, word timestamps, invalid audio, file limits, busy capacity, and unavailable models use the required OpenAI-style error envelope.

`GET /v1/models` requires authentication and reports local aliases plus readiness. `/health/live` is unauthenticated, while `/health/ready` reflects model startup state.

## Data Flow

1. Request middleware establishes or propagates `X-Request-ID`.
2. The protected route checks the Bearer API key using constant-time comparisons.
3. The transcription route validates form options and streams the upload to temporary storage within the configured byte limit.
4. Audio service validates metadata with `ffprobe`, normalizes to 16 kHz mono PCM WAV with `ffmpeg`, and reports the duration.
5. ASR service acquires the queue slot, runs `model.generate` in a worker thread with the specified FunASR options, and parses the returned text and any real timestamps.
6. The route renders the requested OpenAI-compatible format. A `finally` block removes temporary audio files.

## Deployment and Configuration

Pydantic Settings is the sole configuration boundary. API keys have no production default and invalid/empty configuration prevents startup. The Docker image uses Python 3.12 slim, CPU-only PyTorch dependencies, ffmpeg, a non-root user, `/tmp` for transient data, and `/models` for persistent model caches. Compose mounts `/models` as a named volume and exposes the liveness health check. Local development uses `python -m venv .venv` with `pip` and a pinned `requirements.txt`, rather than an `uv` workflow.

## Testing

Tests are FastAPI route tests only, using dependency and service overrides to avoid downloading models or invoking ffmpeg. They cover health/readiness, auth, models, transcription response formats, invalid route parameters, upload limits, service-busy behavior, and cleanup through the route boundary. Docker smoke testing remains documented but is not automated in the test suite.

## Constraints

- Python 3.12, CPU-only runtime, one Uvicorn worker.
- Local setup uses `python` and a virtual environment; dependency versions are pinned in `requirements.txt`.
- No database, CUDA, WebSockets, translations, diarization, or persistent uploaded audio.
- No fake timestamps, no API-key/transcript/audio logging, and no broad recovery fallbacks.
- Keep service boundaries direct and small; do not add dependencies beyond the specified stack.
