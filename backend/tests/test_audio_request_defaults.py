"""Request defaults normalization across task types."""

import pytest

from backend.audio_request_defaults import (
    normalize_request_defaults,
    normalize_task_defaults,
)
from backend.audio_transcription_defaults import normalize_transcription_defaults
from backend.audio_voice_presets import normalize_speech_defaults


def test_normalize_speech_defaults_coerces_numeric_fields():
    out = normalize_speech_defaults(
        {
            "voice": "assistant",
            "temperature": "0.8",
            "top_p": "0.95",
            "seed": "42",
            "max_tokens": "256",
            "instructions": "  warm narrator  ",
            "options": {"speed": "1.1", "no_ref": ""},
        }
    )
    assert out["voice"] == "assistant"
    assert out["temperature"] == 0.8
    assert out["top_p"] == 0.95
    assert out["seed"] == 42
    assert out["max_tokens"] == 256
    assert out["instructions"] == "warm narrator"
    assert out["options"] == {"speed": "1.1"}


def test_normalize_speech_defaults_ignores_invalid_numbers():
    out = normalize_speech_defaults({"seed": "not-a-number", "temperature": "bad"})
    assert "seed" not in out
    assert "temperature" not in out


def test_normalize_speech_defaults_rejects_non_dict():
    assert normalize_speech_defaults(None) == {}
    assert normalize_speech_defaults([]) == {}


def test_normalize_transcription_defaults_preserves_bools_in_options():
    out = normalize_transcription_defaults(
        {
            "language": "da-DK",
            "stream": False,
            "options": {
                "punctuation": True,
                "enable_thinking": False,
                "lookahead_tokens": "8",
            },
        }
    )
    assert out["language"] == "da-DK"
    assert out["stream"] is False
    assert out["options"]["punctuation"] is True
    assert out["options"]["enable_thinking"] is False
    assert out["options"]["lookahead_tokens"] == "8"


def test_normalize_task_defaults_for_ace_step_gen():
    out = normalize_task_defaults(
        {
            "text": "cinematic pop",
            "lyrics": "We rise",
            "duration_seconds": "60",
            "num_inference_steps": "8",
            "guidance_scale": "1.5",
            "repaint_start": "10.5",
            "stream": True,
            "options": {
                "task_route": "text2music",
                "track_name": "vocals",
                "infinite_mode": True,
            },
        }
    )
    assert out["text"] == "cinematic pop"
    assert out["duration_seconds"] == 60.0
    assert out["num_inference_steps"] == 8
    assert out["guidance_scale"] == 1.5
    assert out["repaint_start"] == 10.5
    assert out["options"]["task_route"] == "text2music"
    assert out["options"]["infinite_mode"] is True


def test_normalize_task_defaults_for_vad_chunking():
    out = normalize_task_defaults(
        {
            "audio": "/tmp/speech.wav",
            "vad_chunk_max_seconds": "45",
            "vad_chunk_merge_gap_seconds": "0.5",
            "options": {"threshold": "0.55"},
        }
    )
    assert out["audio"] == "/tmp/speech.wav"
    assert out["vad_chunk_max_seconds"] == 45.0
    assert out["options"]["threshold"] == "0.55"


def test_normalize_task_defaults_for_seed_vc_options():
    out = normalize_task_defaults(
        {
            "audio": "source.wav",
            "voice_ref": "target.wav",
            "num_inference_steps": 30,
            "options": {
                "length_adjust": "1.2",
                "f0_condition": True,
                "semi_tone_shift": "2",
            },
        }
    )
    assert out["num_inference_steps"] == 30
    assert out["options"]["length_adjust"] == "1.2"
    assert out["options"]["f0_condition"] is True
    assert out["options"]["semi_tone_shift"] == "2"


@pytest.mark.parametrize(
    ("key", "value", "expected_key"),
    [
        ("speech_defaults", {"voice": "M1"}, "voice"),
        ("transcription_defaults", {"language": "en"}, "language"),
        ("task_defaults", {"text": "prompt"}, "text"),
    ],
)
def test_normalize_request_defaults_dispatches_by_key(key, value, expected_key):
    out = normalize_request_defaults(key, value)
    assert out[expected_key] == value[expected_key]


def test_normalize_request_defaults_unknown_key_uses_task_normalizer():
    out = normalize_request_defaults("custom_defaults", {"text": "hello"})
    assert out["text"] == "hello"
