"""Convert uploaded audio bytes to PCM WAV for audio.cpp ASR."""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import tempfile
import wave
from typing import Any, Optional, Tuple

from fastapi import HTTPException

MAX_AUDIO_UPLOAD_BYTES = 60 * 1024 * 1024
MAX_CONVERTED_AUDIO_BYTES = 60 * 1024 * 1024
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


def _run_ffmpeg(content: bytes, output_args: list[str], *, operation: str) -> bytes:
    """Run one bounded conversion; callers in async routes must use a worker thread.

    Output goes to disk rather than an unbounded captured stdout buffer. ffmpeg's
    size guard can overshoot by one encoded packet, so reject the result instead
    of returning a silently truncated recording when the guard is reached.
    """
    if len(content) > MAX_AUDIO_UPLOAD_BYTES:
        raise AudioConvertError("Audio conversion input exceeds 60 MiB limit", status_code=413)
    if not ffmpeg_available():
        raise AudioConvertError("ffmpeg is not installed; cannot convert audio", status_code=503)
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-protocol_whitelist", "pipe", "-i", "pipe:0", "-map", "0:a:0",
                    "-vn", "-sn", "-dn", *output_args,
                    "-fs", str(MAX_CONVERTED_AUDIO_BYTES + 1), "pipe:1",
                ],
                input=content,
                stdout=output,
                stderr=errors,
                timeout=FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioConvertError(f"{operation} timed out", status_code=504) from exc
        except OSError as exc:
            raise AudioConvertError(f"Failed to run ffmpeg: {exc}", status_code=503) from exc
        if output.tell() > MAX_CONVERTED_AUDIO_BYTES:
            raise AudioConvertError("Converted audio exceeds 60 MiB limit", status_code=413)
        if proc.returncode != 0 or output.tell() == 0:
            errors.seek(0)
            detail = errors.read(4096).decode("utf-8", errors="replace").strip()
            raise AudioConvertError(f"{operation} failed: {detail or 'unsupported or corrupt audio'}")
        output.seek(0)
        return output.read(MAX_CONVERTED_AUDIO_BYTES + 1)


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
            status_code=413,
        )

    if is_wav_content(content) and wav_data_chunk_readable(content):
        return content, _wav_filename(filename)

    # Decode to raw PCM on stdout (size fields are irrelevant for s16le),
    # then write a seek-correct WAV header ourselves.
    pcm = _run_ffmpeg(
        content,
        ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", str(ASR_CHANNELS),
         "-ar", str(ASR_SAMPLE_RATE)],
        operation="Audio conversion",
    )

    wav_bytes = pcm16le_to_wav(
        pcm,
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


SPEECH_PASSTHROUGH_FORMATS = frozenset({"", "wav", "wave", "json", "b64_json"})
SPEECH_CONTENT_TYPES = {
    "wav": "audio/wav",
    "wave": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}
_SPEECH_FFMPEG_FORMATS = {
    "mp3": ("libmp3lame", "mp3"),
    "opus": ("libopus", "opus"),
    "aac": ("aac", "adts"),
    "flac": ("flac", "flac"),
}


def normalize_speech_response_format(value: Any) -> str:
    return str(value or "").strip().lower()


def wav_to_pcm16le(content: bytes) -> bytes:
    if not is_wav_content(content):
        raise AudioConvertError("PCM export requires a WAV payload")
    try:
        with wave.open(io.BytesIO(content), "rb") as wf:
            if wf.getsampwidth() == 2 and wav_data_chunk_readable(content):
                return wf.readframes(wf.getnframes())
    except (wave.Error, EOFError):
        # Python's wave reader does not decode IEEE float or every extensible
        # WAV. Never label 24/32-bit samples (or floats) as signed PCM16.
        pass
    return _run_ffmpeg(
        content, ["-c:a", "pcm_s16le", "-f", "s16le"], operation="PCM conversion"
    )


def encode_wav_speech_format(content: bytes, response_format: str) -> Tuple[bytes, str]:
    """Convert audio.cpp WAV bytes to an OpenAI speech ``response_format``."""
    fmt = normalize_speech_response_format(response_format)
    if fmt in SPEECH_PASSTHROUGH_FORMATS or fmt == "wav" or fmt == "wave":
        return content, SPEECH_CONTENT_TYPES["wav"]
    if fmt == "pcm":
        return wav_to_pcm16le(content), SPEECH_CONTENT_TYPES["pcm"]
    ffmpeg_fmt = _SPEECH_FFMPEG_FORMATS.get(fmt)
    if not ffmpeg_fmt:
        raise AudioConvertError(
            f"Unsupported speech response_format '{response_format}'. "
            "Use wav, pcm, mp3, opus, aac, or flac.",
            status_code=400,
        )
    codec, muxer = ffmpeg_fmt
    extra: list[str] = []
    if fmt == "opus":
        extra.extend(["-application", "voip", "-b:a", "64k"])
    encoded = _run_ffmpeg(
        content, ["-c:a", codec, *extra, "-f", muxer], operation="Speech format conversion"
    )
    return encoded, SPEECH_CONTENT_TYPES[fmt]
