"""Tests for WAV detection and ffmpeg conversion helpers."""

from __future__ import annotations

import io
import struct
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


def _pipe_style_wav_with_bad_data_size(pcm_bytes: int = 32) -> bytes:
    """Mimic ffmpeg stdout WAV: valid RIFF/WAVE/fmt, data size 0xFFFFFFFF."""
    fmt = struct.pack(
        "<HHIIHH",
        1,  # PCM
        1,  # mono
        16000,
        32000,
        2,
        16,
    )
    pcm = b"\x00" * pcm_bytes
    # RIFF size also often left as 0xFFFFFFFF for pipes.
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
        + pcm
    )


def test_is_wav_content_detects_riff_wave():
    assert convert.is_wav_content(_minimal_wav())
    assert not convert.is_wav_content(b"OggS\x00\x00")
    assert not convert.is_wav_content(b"")


def test_wav_data_chunk_readable_accepts_honest_wav():
    assert convert.wav_data_chunk_readable(_minimal_wav())


def test_wav_data_chunk_readable_rejects_pipe_size_bug():
    bad = _pipe_style_wav_with_bad_data_size()
    assert convert.is_wav_content(bad)
    assert not convert.wav_data_chunk_readable(bad)


def test_ensure_wav_bytes_passthrough_for_wav():
    wav = _minimal_wav()
    out, name = convert.ensure_wav_bytes(wav, filename="clip.ogg")
    assert out == wav
    assert name == "clip.wav"


def test_ensure_wav_bytes_rejects_empty():
    with pytest.raises(convert.AudioConvertError, match="Empty"):
        convert.ensure_wav_bytes(b"")


def test_ensure_wav_bytes_converts_with_ffmpeg(monkeypatch):
    pcm = b"\x00\x00" * 80

    def fake_run(cmd, input=None, capture_output=None, timeout=None, check=None):
        class Result:
            returncode = 0
            stdout = pcm
            stderr = b""

        assert "ffmpeg" in cmd[0]
        assert "-f" in cmd and "s16le" in cmd
        assert input == b"not-wav-bytes"
        return Result()

    monkeypatch.setattr(convert, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    out, name = convert.ensure_wav_bytes(b"not-wav-bytes", filename="memo.opus")
    assert name == "memo.wav"
    assert convert.is_wav_content(out)
    assert convert.wav_data_chunk_readable(out)
    with wave.open(io.BytesIO(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.readframes(wf.getnframes()) == pcm


def test_ensure_wav_bytes_rewrites_broken_pipe_wav(monkeypatch):
    """Broken ffmpeg-pipe WAV must be re-decoded, not passed through."""
    bad = _pipe_style_wav_with_bad_data_size()
    pcm = b"\x01\x00" * 40

    def fake_run(cmd, input=None, capture_output=None, timeout=None, check=None):
        class Result:
            returncode = 0
            stdout = pcm
            stderr = b""

        assert input == bad
        return Result()

    monkeypatch.setattr(convert, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    out, _name = convert.ensure_wav_bytes(bad, filename="pipe.wav")
    assert convert.wav_data_chunk_readable(out)
    assert out != bad


def test_ensure_wav_bytes_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(convert, "ffmpeg_available", lambda: False)
    with pytest.raises(convert.AudioConvertError) as exc:
        convert.ensure_wav_bytes(b"OggSfake", filename="a.ogg")
    assert exc.value.status_code == 503


def test_pcm16le_to_wav_roundtrip():
    pcm = b"\x00\x10\xff\x7f"
    wav = convert.pcm16le_to_wav(pcm, channels=1, sample_rate=16000)
    assert convert.wav_data_chunk_readable(wav)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.readframes(2) == pcm


def test_encode_wav_speech_format_pcm_and_passthrough():
    wav = _minimal_wav()
    pcm, media = convert.encode_wav_speech_format(wav, "pcm")
    assert media == "audio/pcm"
    assert pcm == convert.wav_to_pcm16le(wav)
    same, wav_type = convert.encode_wav_speech_format(wav, "wav")
    assert same == wav
    assert wav_type == "audio/wav"


def test_encode_wav_speech_format_rejects_unknown():
    with pytest.raises(convert.AudioConvertError) as exc:
        convert.encode_wav_speech_format(_minimal_wav(), "wma")
    assert exc.value.status_code == 400
