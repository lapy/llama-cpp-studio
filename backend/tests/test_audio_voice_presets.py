"""Voice preset validation edge cases."""

import pytest

from backend.audio_voice_presets import (
    normalize_default_voice_preset,
    normalize_voice_preset,
    normalize_voice_presets,
    validate_voice_presets,
)


def test_normalize_voice_preset_requires_at_least_one_field():
    assert normalize_voice_preset({}, model_root="/tmp") is None
    assert normalize_voice_preset({"language": "en"}, model_root="/tmp") is None


def test_normalize_voice_preset_accepts_voice_id_only():
    out = normalize_voice_preset({"voice_id": "M1"}, model_root="/tmp")
    assert out == {"voice_id": "M1"}


def test_normalize_voice_presets_skips_empty_names_and_invalid_entries(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    presets = normalize_voice_presets(
        {
            "": {"voice_id": "x"},
            "valid": {"voice_id": "M1"},
            "invalid": "not-a-dict",
        },
        model_root=str(model_root),
    )
    assert list(presets.keys()) == ["valid"]


def test_validate_voice_presets_missing_named_default(tmp_path):
    errors: list[str] = []
    validate_voice_presets(
        {
            "voice_presets": {"assistant": {"voice_id": "M1"}},
            "default_voice_preset": "missing",
        },
        model_root=str(tmp_path),
        errors=errors,
    )
    assert any("not defined in voice_presets" in err for err in errors)


def test_validate_voice_presets_missing_voice_ref_file(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    errors: list[str] = []
    validate_voice_presets(
        {
            "voice_presets": {
                "clone": {"voice_ref": "missing.wav"},
            }
        },
        model_root=str(model_root),
        errors=errors,
    )
    assert any("voice_ref does not exist" in err for err in errors)


def test_validate_voice_presets_inline_default_object_missing_fields(tmp_path):
    errors: list[str] = []
    validate_voice_presets(
        {"default_voice_preset": {"language": "en"}},
        model_root=str(tmp_path),
        errors=errors,
    )
    assert any("must include voice_id, voice_ref, or reference_text" in err for err in errors)


def test_normalize_default_voice_preset_inline_object(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF")
    out = normalize_default_voice_preset(
        {"voice_ref": str(wav)},
        model_root=str(tmp_path),
    )
    assert out["voice_ref"] == str(wav.resolve())
