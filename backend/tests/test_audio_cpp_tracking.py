"""Unit tests for audio.cpp release-tag conventions and tracking helpers."""

from __future__ import annotations

from backend.audio_cpp_tracking import (
    is_audio_release_tag,
    resolve_latest_github_release,
    resolve_latest_release_tag,
)
from backend.routes.audio_cpp_versions import _ref_kind


def test_is_audio_release_tag_current_convention():
    assert is_audio_release_tag("release-0.5.1")
    assert is_audio_release_tag("release-0.5")
    assert is_audio_release_tag("release-0.3-qwen3-tts")
    assert is_audio_release_tag("RELEASE-0.4.2")


def test_is_audio_release_tag_legacy_and_semver():
    assert is_audio_release_tag("v0.2.0-windows-prebuilt")
    assert is_audio_release_tag("v1.2.3")
    assert is_audio_release_tag("1.2.3")


def test_is_audio_release_tag_rejects_branches_and_shas():
    assert not is_audio_release_tag("main")
    assert not is_audio_release_tag("dev")
    assert not is_audio_release_tag("feature/catalog")
    assert not is_audio_release_tag("release")  # no version
    assert not is_audio_release_tag("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


def test_ref_kind_classifies_release_tags():
    assert _ref_kind("release-0.5.1") == "release"
    assert _ref_kind("release-0.3-qwen3-tts") == "release"
    assert _ref_kind("v0.2.0-windows-prebuilt") == "release"
    assert _ref_kind("main") == "branch"
    assert _ref_kind("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef") == "commit"


def test_resolve_latest_github_release_parses_payload(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "tag_name": "release-0.5.1",
                "name": "audio.cpp 0.5.1",
                "html_url": "https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5.1",
                "published_at": "2026-01-15T00:00:00Z",
                "target_commitish": "main",
                "prerelease": False,
            }

    monkeypatch.setattr(
        "backend.audio_cpp_tracking.requests.get",
        lambda *a, **k: FakeResponse(),
    )
    release = resolve_latest_github_release()
    assert release["tag_name"] == "release-0.5.1"
    assert release["target_commitish"] == "main"
    assert resolve_latest_release_tag() == "release-0.5.1"


def test_resolve_latest_github_release_returns_none_on_error(monkeypatch):
    class FakeResponse:
        status_code = 404

        def json(self):
            return {}

    monkeypatch.setattr(
        "backend.audio_cpp_tracking.requests.get",
        lambda *a, **k: FakeResponse(),
    )
    assert resolve_latest_github_release() is None
    assert resolve_latest_release_tag() is None
