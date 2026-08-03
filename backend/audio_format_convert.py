"""Convert uploaded audio bytes to PCM WAV for audio.cpp ASR."""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import wave
from typing import Optional, Tuple

from fastapi import HTTPException

MAX_AUDIO_UPLOAD_BYTES = 60 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 60

# ASR-friendly defaults when decoding compressed formats.
ASR_SAMPLE_RATE = 16000
ASR_CHANNELS = 1


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


def wav_data_chunk_readable(content: bytes) -> bool:
    """Return True when the WAV ``data`` chunk size matches available bytes.

    ffmpeg's ``pipe:1`` WAV muxer often writes ``0xFFFFFFFF`` size fields because
    it cannot seek the stream to patch the header. audio.cpp then fails with
    ``failed to read WAV data chunk``.
    """
    if not is_wav_content(content):
        return False
    offset = 12
    length = len(content)
    while offset + 8 <= length:
        chunk_id = content[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        data_start = offset + 8
        data_end = data_start + chunk_size
        if chunk_id == b"data":
            return data_end <= length
        # Chunks are word-aligned.
        offset = data_end + (chunk_size % 2)
    return False


def pcm16le_to_wav(pcm: bytes, *, channels: int, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def ensure_wav_bytes(
    content: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Return PCM WAV bytes and a .wav filename.

    Already-WAV payloads with a readable ``data`` chunk are returned unchanged.
    Other formats (and broken pipe-WAV) are converted with ffmpeg to mono 16 kHz
    PCM16, then wrapped with a correct RIFF header via the stdlib ``wave``
    module (avoids ffmpeg stdout size-field bugs). Raises AudioConvertError on
    failure.
    """
    if not content:
        raise AudioConvertError("Empty audio upload")
    if len(content) > MAX_AUDIO_UPLOAD_BYTES:
        raise AudioConvertError(
            f"Audio upload exceeds {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    if is_wav_content(content) and wav_data_chunk_readable(content):
        return content, _wav_filename(filename)

    if not ffmpeg_available():
        raise AudioConvertError(
            "ffmpeg is not installed; cannot convert non-WAV audio uploads",
            status_code=503,
        )

    # Decode to raw PCM on stdout (size fields are irrelevant for s16le),
    # then write a seek-correct WAV header ourselves.
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
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                str(ASR_CHANNELS),
                "-ar",
                str(ASR_SAMPLE_RATE),
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

    wav_bytes = pcm16le_to_wav(
        proc.stdout,
        channels=ASR_CHANNELS,
        sample_rate=ASR_SAMPLE_RATE,
    )
    if not wav_data_chunk_readable(wav_bytes):
        raise AudioConvertError("converted WAV failed validation")

    return wav_bytes, _wav_filename(filename)


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
