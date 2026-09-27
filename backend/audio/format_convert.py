"""Convert uploaded audio bytes to PCM WAV for audio.cpp.

The core server still accepts only WAV. A readable WAV file is forwarded
unchanged. Other formats are decoded to PCM16 at the source sample rate and
channel count. 16 kHz mono is the fallback when the layout cannot be read.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from typing import Any, Optional, Tuple

from fastapi import HTTPException

# The server default body limit is 2 GiB. Studio buffers the upload and the
# converted WAV, so the proxy cap stays well under that while still allowing
# long recordings.
MAX_AUDIO_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_CONVERTED_AUDIO_BYTES = MAX_AUDIO_UPLOAD_BYTES
FFMPEG_TIMEOUT_SECONDS = 180
PROBE_TIMEOUT_SECONDS = 30

# Used only when the source layout cannot be probed.
ASR_SAMPLE_RATE = 16000
ASR_CHANNELS = 1
_MAX_SAMPLE_RATE = 384000
_MAX_CHANNELS = 8

_HZ_LAYOUT_RE = re.compile(
    r"(\d+)\s+Hz,\s+(mono|stereo|quad|[0-9]+\.[0-9]+|[0-9]+\s+channels)",
    re.IGNORECASE,
)
_CHANNEL_WORDS = {
    "mono": 1,
    "stereo": 2,
    "quad": 4,
    "2.1": 3,
    "3.0": 3,
    "4.0": 4,
    "5.0": 5,
    "5.1": 6,
    "6.1": 7,
    "7.1": 8,
}


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


def audio_mib_limit() -> int:
    return MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)


def audio_limit_detail(kind: str) -> str:
    return f"{kind} exceeds {audio_mib_limit()} MiB limit"


def _valid_layout(sample_rate: Any, channels: Any) -> Optional[Tuple[int, int]]:
    try:
        rate = int(sample_rate)
        count = int(channels)
    except (TypeError, ValueError):
        return None
    if rate <= 0 or rate > _MAX_SAMPLE_RATE or count <= 0 or count > _MAX_CHANNELS:
        return None
    return rate, count


def _channels_from_layout_token(token: str) -> Optional[int]:
    word = token.strip().lower()
    if word in _CHANNEL_WORDS:
        return _CHANNEL_WORDS[word]
    if word.endswith("channels"):
        head = word.split()[0]
        if head.isdigit():
            return int(head)
    return None


def _layout_from_ffmpeg_stderr(stderr: bytes) -> Optional[Tuple[int, int]]:
    text = stderr.decode("utf-8", errors="replace")
    match = _HZ_LAYOUT_RE.search(text)
    if not match:
        return None
    channels = _channels_from_layout_token(match.group(2))
    if channels is None:
        return None
    return _valid_layout(match.group(1), channels)


def _layout_from_ffprobe_json(stdout: bytes) -> Optional[Tuple[int, int]]:
    try:
        payload = json.loads(stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0] if isinstance(streams[0], dict) else {}
    return _valid_layout(stream.get("sample_rate"), stream.get("channels"))


def _capture_tool(argv: list[str], content: bytes, *, timeout: int) -> Tuple[int, bytes, bytes]:
    try:
        proc = subprocess.run(
            argv,
            input=content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, b"", b""
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def probe_audio_layout(content: bytes) -> Optional[Tuple[int, int]]:
    """Return ``(sample_rate, channels)`` for the first audio stream.

    ``None`` means the layout could not be read. Callers then use the 16 kHz
    mono fallback.
    """
    if not content:
        return None
    if shutil.which("ffprobe"):
        _code, stdout, _stderr = _capture_tool(
            [
                "ffprobe", "-hide_banner", "-loglevel", "error",
                "-protocol_whitelist", "pipe",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "json",
                "pipe:0",
            ],
            content,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        layout = _layout_from_ffprobe_json(stdout)
        if layout:
            return layout
    if not ffmpeg_available():
        return None
    _code, _stdout, stderr = _capture_tool(
        [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-protocol_whitelist", "pipe",
            "-i", "pipe:0",
            "-f", "null", "-",
        ],
        content,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    return _layout_from_ffmpeg_stderr(stderr)


def _run_ffmpeg(content: bytes, output_args: list[str], *, operation: str) -> bytes:
    """Run one bounded conversion; callers in async routes must use a worker thread.

    Output goes to disk rather than an unbounded captured stdout buffer. ffmpeg's
    size guard can overshoot by one encoded packet, so reject the result instead
    of returning a silently truncated recording when the guard is reached.
    """
    if len(content) > MAX_AUDIO_UPLOAD_BYTES:
        raise AudioConvertError(audio_limit_detail("Audio conversion input"), status_code=413)
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
            raise AudioConvertError(audio_limit_detail("Converted audio"), status_code=413)
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
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> Tuple[bytes, str]:
    """Return PCM WAV bytes and a .wav filename.

    Already-WAV payloads with a readable ``data`` chunk are returned unchanged,
    including their sample rate. Other formats (and broken pipe-WAV) are
    converted with ffmpeg to PCM16 at the source rate and channel count, then
    wrapped with a correct RIFF header via the stdlib ``wave`` module (avoids
    ffmpeg stdout size-field bugs). Pass ``sample_rate`` and ``channels``
    together to override that layout, for a model whose scan requires a fixed
    rate. When the layout cannot be read, the fallback is 16 kHz mono.
    Raises AudioConvertError on failure.
    """
    if not content:
        raise AudioConvertError("Empty audio upload")
    if len(content) > MAX_AUDIO_UPLOAD_BYTES:
        raise AudioConvertError(
            audio_limit_detail("Audio upload"),
            status_code=413,
        )

    if is_wav_content(content) and wav_data_chunk_readable(content):
        return content, _wav_filename(filename)

    forced = _valid_layout(sample_rate, channels) if (
        sample_rate is not None and channels is not None
    ) else None
    rate, count = forced or probe_audio_layout(content) or (ASR_SAMPLE_RATE, ASR_CHANNELS)

    # Decode to raw PCM on stdout (size fields are irrelevant for s16le),
    # then write a seek-correct WAV header ourselves.
    pcm = _run_ffmpeg(
        content,
        ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", str(count), "-ar", str(rate)],
        operation="Audio conversion",
    )

    wav_bytes = pcm16le_to_wav(
        pcm,
        channels=count,
        sample_rate=rate,
    )
    if not wav_data_chunk_readable(wav_bytes):
        raise AudioConvertError("converted WAV failed validation")

    return wav_bytes, _wav_filename(filename)


def ensure_wav_bytes_http(
    content: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> Tuple[bytes, str]:
    """Like ensure_wav_bytes but raises FastAPI HTTPException."""
    try:
        return ensure_wav_bytes(
            content,
            filename=filename,
            content_type=content_type,
            sample_rate=sample_rate,
            channels=channels,
        )
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
