"""Convert uploaded audio bytes to PCM WAV for audio.cpp ASR."""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional, Tuple

from fastapi import HTTPException

MAX_AUDIO_UPLOAD_BYTES = 60 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 60


class AudioConvertError(Exception):
    """Raised when media cannot be converted to WAV."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def is_wav_content(content: bytes) -> bool:
    return (
        len(content) >= 12
        and content[:4] == b"RIFF"
        and content[8:12] == b"WAVE"
    )


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


def ensure_wav_bytes(
    content: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Return PCM WAV bytes and a .wav filename.

    Already-WAV payloads are returned unchanged. Other formats are converted
    with ffmpeg (pcm_s16le). Raises AudioConvertError on failure.
    """
    if not content:
        raise AudioConvertError("Empty audio upload")
    if len(content) > MAX_AUDIO_UPLOAD_BYTES:
        raise AudioConvertError(
            f"Audio upload exceeds {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    if is_wav_content(content):
        out_name = _wav_filename(filename)
        return content, out_name

    if not ffmpeg_available():
        raise AudioConvertError(
            "ffmpeg is not installed; cannot convert non-WAV audio uploads",
            status_code=503,
        )

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "pipe:0",
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ],
            input=content,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioConvertError("Audio conversion timed out") from exc
    except OSError as exc:
        raise AudioConvertError(
            f"Failed to run ffmpeg: {exc}",
            status_code=503,
        ) from exc

    if proc.returncode != 0 or not proc.stdout:
        detail = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        hint = detail or "unsupported or corrupt audio"
        name_hint = filename or content_type or "upload"
        raise AudioConvertError(f"Could not convert {name_hint} to WAV: {hint}")

    if not is_wav_content(proc.stdout):
        raise AudioConvertError("ffmpeg produced output that is not a valid WAV")

    return proc.stdout, _wav_filename(filename)


def ensure_wav_bytes_http(
    content: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Like ensure_wav_bytes but raises FastAPI HTTPException."""
    try:
        return ensure_wav_bytes(content, filename=filename, content_type=content_type)
    except AudioConvertError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _wav_filename(filename: Optional[str]) -> str:
    base = (filename or "audio").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    if not base:
        return "audio.wav"
    if "." in base:
        base = base.rsplit(".", 1)[0]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
    safe = safe.strip("._") or "audio"
    return f"{safe}.wav"
