"""Experimental audio.cpp kill-switch behavior."""

import pytest
from fastapi import HTTPException

from backend.feature_flags import audio_cpp_enabled
from backend.routes.audio_cpp_versions import _require_audio_cpp_enabled


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("off", False),
    ],
)
def test_audio_cpp_feature_flag(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("AUDIO_CPP_ENABLED", raising=False)
    else:
        monkeypatch.setenv("AUDIO_CPP_ENABLED", raw)
    assert audio_cpp_enabled() is expected


def test_disabled_audio_routes_fail_closed(monkeypatch):
    monkeypatch.setenv("AUDIO_CPP_ENABLED", "false")
    with pytest.raises(HTTPException) as caught:
        _require_audio_cpp_enabled()
    assert caught.value.status_code == 404
    assert "AUDIO_CPP_ENABLED" in caught.value.detail

