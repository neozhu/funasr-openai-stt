"""FunASR model lifecycle and CPU inference."""

from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio

from app.config import Settings


class ServiceBusyError(Exception):
    """Raised when no inference slot becomes available in time."""


class InferenceTimeoutError(Exception):
    """Raised when inference exceeds its configured deadline."""


@dataclass
class Transcription:
    """Parsed text and optional model-provided segments."""

    text: str
    segments: list[dict[str, object]]


class AsrService:
    """Own one CPU FunASR model and serialize inference requests."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._slot = asyncio.Semaphore(settings.asr_max_concurrent_requests)

    async def load(self) -> None:
        """Load the configured model once during application startup."""
        await anyio.to_thread.run_sync(self._load)

    async def warmup(self) -> None:
        """Run a short silent sample after loading the model."""
        await anyio.to_thread.run_sync(self._warmup)

    async def transcribe(self, path: Path, language: str) -> Transcription:
        """Run blocking CPU inference outside the event loop."""
        try:
            await asyncio.wait_for(self._slot.acquire(), self._settings.asr_queue_timeout_seconds)
        except TimeoutError as exc:
            raise ServiceBusyError from exc
        try:
            try:
                result = await asyncio.wait_for(
                    anyio.to_thread.run_sync(self._generate, path, language),
                    self._settings.asr_inference_timeout_seconds,
                )
            except TimeoutError as exc:
                raise InferenceTimeoutError from exc
            return self._parse(result)
        finally:
            self._slot.release()

    def _load(self) -> None:
        import torch
        from funasr import AutoModel

        torch.set_num_threads(self._settings.asr_cpu_threads)
        self._model = AutoModel(
            model=self._settings.asr_model_id,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30_000},
            device="cpu",
            disable_update=True,
        )

    def _generate(self, path: Path, language: str) -> list[dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")
        return cast(
            list[dict[str, Any]],
            self._model.generate(
                input=str(path),
                language=language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            ),
        )

    def _warmup(self) -> None:
        descriptor, name = tempfile.mkstemp(suffix=".wav")
        os.close(descriptor)
        sample = Path(name)
        try:
            with wave.open(str(sample), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\x00\x00" * 1_600)
            self._generate(sample, "auto")
        finally:
            sample.unlink(missing_ok=True)

    @staticmethod
    def _parse(result: list[dict[str, Any]]) -> Transcription:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess("".join(item.get("text", "") for item in result))
        return Transcription(text=text, segments=[])
