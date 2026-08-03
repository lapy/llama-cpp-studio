"""Tests for WAV detection and ffmpeg conversion helpers."""

from __future__ import annotations

import io
import wave

import pytest

from backend import audio_format_convert as convert


def _minimal_wav(frames: int = 1600, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def test_is_wav_content_detects_riff_wave():
    assert convert.is_wav_content(_minimal_wav())
    assert not convert.is_wav_content(b"OggS\x00\x00")
    assert not convert.is_wav_content(b"")


def test_ensure_wav_bytes_passthrough_for_wav():
    wav = _minimal_wav()
    out, name = convert.ensure_wav_bytes(wav, filename="clip.ogg")
    assert out == wav
    assert name == "clip.wav"


def test_ensure_wav_bytes_rejects_empty():
    with pytest.raises(convert.AudioConvertError, match="Empty"):
        convert.ensure_wav_bytes(b"")


def test_ensure_wav_bytes_converts_with_ffmpeg(monkeypatch):
    wav = _minimal_wav()

    def fake_run(cmd, input=None, capture_output=None, timeout=None, check=None):
        class Result:
            returncode = 0
            stdout = wav
            stderr = b""

        assert "ffmpeg" in cmd[0]
        assert input == b"not-wav-bytes"
        return Result()

    monkeypatch.setattr(convert, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    out, name = convert.ensure_wav_bytes(b"not-wav-bytes", filename="memo.opus")
    assert out == wav
    assert name == "memo.wav"


def test_ensure_wav_bytes_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(convert, "ffmpeg_available", lambda: False)
    with pytest.raises(convert.AudioConvertError) as exc:
        convert.ensure_wav_bytes(b"OggSfake", filename="a.ogg")
    assert exc.value.status_code == 503
