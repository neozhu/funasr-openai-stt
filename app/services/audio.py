"""Temporary audio file processing."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import anyio
from fastapi import UploadFile

from app.config import Settings
from app.errors import OpenAIHTTPError


@dataclass
class PreparedAudio:
    """A validated, normalized temporary audio file."""

    input_path: Path
    normalized_path: Path
    duration_seconds: float
    size_bytes: int


class AudioService:
    """Stream, validate, normalize, and remove uploaded audio."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._directory = settings.asr_temp_dir

    async def prepare(self, upload: UploadFile) -> PreparedAudio:
        """Create a normalized WAV file for ASR inference."""
        self._directory.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        input_path = self._directory / f"{token}.input"
        normalized_path = self._directory / f"{token}.wav"
        limit = self._settings.asr_max_upload_mb * 1024 * 1024
        size = 0
        try:
            with input_path.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise OpenAIHTTPError(
                            413,
                            "Audio file exceeds the configured size limit.",
                            "invalid_request_error",
                            "file",
                            "file_too_large",
                        )
                    output.write(chunk)
            if size == 0:
                raise OpenAIHTTPError(
                    400,
                    "Audio file is empty.",
                    "invalid_request_error",
                    "file",
                    "invalid_audio_file",
                )
            duration = await self._probe(input_path)
            if duration > self._settings.asr_max_audio_duration_seconds:
                raise OpenAIHTTPError(
                    400,
                    "Audio duration exceeds the configured limit.",
                    "invalid_request_error",
                    "file",
                    "invalid_audio_file",
                )
            await self._normalize(input_path, normalized_path)
            return PreparedAudio(input_path, normalized_path, duration, size)
        except BaseException:
            await self.cleanup(PreparedAudio(input_path, normalized_path, 0, size))
            raise
        finally:
            await upload.close()

    async def cleanup(self, audio: PreparedAudio) -> None:
        """Delete temporary files associated with one upload."""
        for path in (audio.input_path, audio.normalized_path):
            await anyio.to_thread.run_sync(path.unlink, True)

    def cleanup_abandoned(self) -> None:
        """Remove stale files left after an interrupted process."""
        self._directory.mkdir(parents=True, exist_ok=True)
        cutoff = self._settings.asr_temp_file_max_age_minutes * 60
        now = time.time()
        for path in self._directory.iterdir():
            if path.is_file() and now - path.stat().st_mtime > cutoff:
                path.unlink(missing_ok=True)

    async def _probe(self, path: Path) -> float:
        result = await self._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        try:
            return float(json.loads(result.stdout)["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OpenAIHTTPError(
                400,
                "Invalid or unsupported audio file.",
                "invalid_request_error",
                "file",
                "invalid_audio_file",
            ) from exc

    async def _normalize(self, input_path: Path, output_path: Path) -> None:
        await self._run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

    async def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = await anyio.to_thread.run_sync(
                lambda: subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OpenAIHTTPError(
                400,
                "Invalid or unsupported audio file.",
                "invalid_request_error",
                "file",
                "invalid_audio_file",
            ) from exc
        if result.returncode != 0:
            raise OpenAIHTTPError(
                400,
                "Invalid or unsupported audio file.",
                "invalid_request_error",
                "file",
                "invalid_audio_file",
            )
        return result
