"""Run a basic authenticated transcription request against a running service."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def main() -> None:
    """Send one audio file to the configured service."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    with args.audio.open("rb") as audio:
        response = httpx.post(
            f"{args.base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {args.api_key}"},
            data={"model": "whisper-1", "response_format": "json"},
            files={"file": (args.audio.name, audio, "audio/wav")},
            timeout=660,
        )
    response.raise_for_status()
    print(response.json()["text"])


if __name__ == "__main__":
    main()
