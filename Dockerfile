# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELSCOPE_CACHE=/models/modelscope \
    HF_HOME=/models/huggingface \
    ASR_TEMP_DIR=/tmp/funasr

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /models /tmp/funasr \
    && chown -R appuser:appuser /models /tmp/funasr

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple -r requirements.txt

COPY app ./app

RUN python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download

from app.config import Settings
from app.services.asr import BUNDLED_MODELS_DIR

model_id = Settings().asr_model_id
snapshot_download(model_id, local_dir=str(BUNDLED_MODELS_DIR / model_id))
snapshot_download(
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    local_dir=str(BUNDLED_MODELS_DIR / "fsmn-vad"),
)
PY

USER appuser

RUN --network=none python - <<'PY'
import asyncio

from app.config import Settings
from app.services.asr import AsrService

async def verify_models():
    service = AsrService(Settings())
    await service.load()
    await service.warmup()

asyncio.run(verify_models())
PY

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=180s \
    CMD curl --fail --silent http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
