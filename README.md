# FunASR OpenAI-Compatible STT

CPU-only FastAPI speech-to-text service powered by FunASR SenseVoiceSmall. It accepts Chinese, English, and mixed Chinese-English audio through an OpenAI-compatible transcription API.

## Quick start

Use Python 3.12 locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Set a real `ASR_API_KEYS` value in `.env`, then run:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The first startup downloads SenseVoiceSmall into the configured model cache. In Docker Compose that cache is persisted in the `funasr-models` volume.

## API

Supported formats: WAV, MP3, MP4, MPEG/MPGA, M4A, WebM, FLAC, and OGG. Uploads default to 25 MB and one hour. Supported languages are `auto`, `zh`, and `en`; use `auto` for mixed speech.

```bash
curl --request POST http://localhost:8000/v1/audio/transcriptions \
  --header "Authorization: Bearer $ASR_API_KEY" \
  --form "file=@mixed-language.wav" \
  --form "model=whisper-1" \
  --form "response_format=json"
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local-key")
with open("mixed-language.wav", "rb") as audio:
    result = client.audio.transcriptions.create(model="whisper-1", file=audio)
print(result.text)
```

`GET /health/live` is unauthenticated. `GET /health/ready`, `GET /v1/models`, and `POST /v1/audio/transcriptions` require a Bearer key.

## Docker

```bash
cp .env.example .env
# Edit ASR_API_KEYS in .env first.
docker compose up --build
```

The image runs as a non-root user, uses one Uvicorn worker, keeps transient audio under `/tmp`, and stores model downloads under `/models`.

## Compatibility and operations

`prompt` and `temperature` are accepted but ignored. Word timestamps, SRT/VTT, streaming, translations, and speaker diarization are not implemented. Model aliases (`whisper-1`, `gpt-4o-transcribe`, and `sensevoice`) all use SenseVoiceSmall.

Keep the service behind HTTPS and an upload-limiting reverse proxy outside trusted networks. Do not expose `/docs` publicly unless required. CPU throughput is controlled by `ASR_CPU_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `ASR_MAX_CONCURRENT_REQUESTS`; increase them only after measuring latency and memory.
