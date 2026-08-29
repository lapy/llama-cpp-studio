"""Unit tests for audio.cpp release-tag conventions and tracking helpers."""

from __future__ import annotations

from pathlib import Path

from backend.audio_cpp_manager import AUDIO_CPP_DEFAULT_REF
from backend.audio_cpp_tracking import (
    is_audio_release_tag,
    resolve_bootstrap_tracking_ref,
    resolve_latest_github_release,
    resolve_latest_release_tag,
)
from backend.routes.audio_cpp_versions import _ref_kind


def test_is_audio_release_tag_current_convention():
    assert is_audio_release_tag("v0.7.0")
    assert is_audio_release_tag("v0.6.2-release-test")
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
    assert _ref_kind("v0.7.0") == "release"
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
                "tag_name": "v0.7.0",
                "name": "audio.cpp 0.7.0",
                "html_url": "https://github.com/0xShug0/audio.cpp/releases/tag/v0.7.0",
                "published_at": "2026-08-26T00:00:00Z",
                "target_commitish": "main",
                "prerelease": False,
            }

    monkeypatch.setattr(
        "backend.audio_cpp_tracking.requests.get",
        lambda *a, **k: FakeResponse(),
    )
    release = resolve_latest_github_release()
    assert release["tag_name"] == "v0.7.0"
    assert release["target_commitish"] == "main"
    assert resolve_latest_release_tag() == "v0.7.0"


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


def test_resolve_bootstrap_tracking_ref_prefers_latest_release(monkeypatch):
    monkeypatch.setattr(
        "backend.audio_cpp_tracking.resolve_latest_release_tag",
        lambda: "v0.7.0",
    )
    assert resolve_bootstrap_tracking_ref() == "v0.7.0"


def test_audio_cpp_default_ref_is_main():
    assert AUDIO_CPP_DEFAULT_REF == "main"


def test_persisted_engine_pin_uses_v2_manager_and_commit():
    path = Path(__file__).resolve().parents[2] / "data" / "config" / "engines.yaml"
    if not path.is_file():
        return
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    audio = data.get("audio_cpp") or {}
    versions = audio.get("versions") or []
    if not versions:
        return
    active = audio.get("active_version")
    row = next(
        (item for item in versions if item.get("version") == active),
        versions[0],
    )
    assert str(row.get("model_manager_path") or "").endswith("model_manager_v2.py")
    assert len(str(row.get("source_commit") or "")) == 40
    settings = audio.get("build_settings") or {}
    if settings.get("tracking_ref"):
        assert settings["tracking_ref"] in {"main", AUDIO_CPP_DEFAULT_REF}
