"""Application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "FunASR Speech API"
    asr_api_keys: str = ""
    asr_model_id: str = "iic/SenseVoiceSmall"
    asr_model_alias: str = "sensevoice"
    asr_device: str = "cpu"
    asr_max_upload_mb: int = 25
    asr_max_audio_duration_seconds: int = 3600
    asr_max_concurrent_requests: int = 1
    asr_queue_timeout_seconds: float = 10
    asr_inference_timeout_seconds: float = 600
    asr_cpu_threads: int = 8
    asr_temp_dir: Path = Path("/tmp/funasr")
    asr_temp_file_max_age_minutes: int = 60
    asr_log_transcript: bool = False
    asr_enable_docs: bool = True

    @property
    def api_keys(self) -> tuple[str, ...]:
        """Return configured API keys without surrounding whitespace."""
        keys = tuple(key.strip() for key in self.asr_api_keys.split(",") if key.strip())
        if not keys:
            raise ValueError("ASR_API_KEYS must contain at least one key")
        return keys
