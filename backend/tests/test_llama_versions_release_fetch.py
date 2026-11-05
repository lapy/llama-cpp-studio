"""Tests for llama_github_refs (llama.cpp releases; ik_llama.cpp main tip)."""

from unittest.mock import MagicMock, patch

import pytest


def _resp(status_code=200, json_data=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data if json_data is not None else []
    m.raise_for_status = MagicMock()
    return m


def test_stable_and_nightly_tag_helpers():
    from backend.engines.llama_cpp.github_refs import (
        is_nightly_release_tag,
        is_stable_release_tag,
    )

    assert is_stable_release_tag("v0.4.1")
    assert not is_stable_release_tag("b10566")
    assert not is_stable_release_tag("v0.4.1-rc1")
    assert not is_stable_release_tag("deadbeef")
    assert is_nightly_release_tag("b10566")
    assert is_nightly_release_tag("B12")
    assert not is_nightly_release_tag("v0.4.1")
    assert not is_nightly_release_tag("bb4caa754018")


def test_parse_latest_release_prefers_stable_over_nightly():
    from backend.engines.llama_cpp.github_refs import parse_latest_release

    raw = [
        {
            "tag_name": "b11000",
            "draft": False,
            "prerelease": True,
            "html_url": "https://example.test/b11000",
        },
        {
            "tag_name": "v0.4.1",
            "draft": False,
            "prerelease": False,
            "html_url": "https://example.test/v0.4.1",
        },
        {"tag_name": "v0.0.0-draft", "draft": True, "prerelease": False},
    ]
    assert parse_latest_release(raw)["tag_name"] == "v0.4.1"


def test_parse_latest_release_falls_back_to_nightly_tag():
    from backend.engines.llama_cpp.github_refs import parse_latest_release

    raw = [
        {"tag_name": "b1234", "draft": False, "prerelease": True},
        {"tag_name": "skip", "draft": True, "prerelease": False},
    ]
    assert parse_latest_release(raw)["tag_name"] == "b1234"


def test_fetch_latest_llama_cpp_release():
    from backend.engines.llama_cpp import github_refs as llama_github_refs

    rel = _resp(
        200,
        {
            "tag_name": "v0.4.1",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-09-14",
            "html_url": "https://x",
        },
    )

    with patch.object(llama_github_refs.requests, "get", return_value=rel) as g:
        out = llama_github_refs.fetch_latest_release_for_repository_source("llama.cpp")
    assert out["tag_name"] == "v0.4.1"
    assert g.call_count == 1
    assert g.call_args[0][0].endswith("/releases/latest")


def test_fetch_latest_llama_cpp_release_falls_back_to_listing():
    from backend.engines.llama_cpp import github_refs as llama_github_refs

    missing = _resp(404, {})
    listing = _resp(
        200,
        [
            {
                "tag_name": "b1234",
                "draft": False,
                "prerelease": True,
                "published_at": "2024-01-01",
                "html_url": "https://x",
            }
        ],
    )

    with patch.object(
        llama_github_refs.requests, "get", side_effect=[missing, listing]
    ) as g:
        out = llama_github_refs.fetch_latest_release_for_repository_source("llama.cpp")
    assert out["tag_name"] == "b1234"
    assert g.call_count == 2
    assert g.call_args_list[0][0][0].endswith("/releases/latest")
    assert "/releases?" in g.call_args_list[1][0][0]


def test_fetch_latest_release_ik_llama_is_none():
    from backend.engines.llama_cpp import github_refs as llama_github_refs

    with patch.object(llama_github_refs.requests, "get") as g:
        out = llama_github_refs.fetch_latest_release_for_repository_source(
            "ik_llama.cpp"
        )
    assert out is None
    g.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_build_ref_uses_stable_tag(monkeypatch):
    from backend.services import upstream_versions

    async def fake_get(url):
        if url.endswith("/releases/latest"):
            return {
                "tag_name": "v0.4.1",
                "draft": False,
                "prerelease": False,
            }
        raise AssertionError(url)

    monkeypatch.setattr(upstream_versions, "_get_json", fake_get)
    ref, kind = await upstream_versions.resolve_build_ref("llama_cpp")
    assert ref == "v0.4.1"
    assert kind == "release"


@pytest.mark.asyncio
async def test_resolve_build_ref_falls_back_to_nightly_tag(monkeypatch):
    from backend.services import upstream_versions

    async def fake_get(url):
        if url.endswith("/releases/latest"):
            return None
        if "/releases?" in url:
            return [
                {"tag_name": "b11000", "draft": False, "prerelease": True},
            ]
        raise AssertionError(url)

    monkeypatch.setattr(upstream_versions, "_get_json", fake_get)
    ref, kind = await upstream_versions.resolve_build_ref("llama_cpp")
    assert ref == "b11000"
    assert kind == "release"


def test_fetch_ik_llama_main_tip_commit():
    from backend.engines.llama_cpp import github_refs as llama_github_refs

    body = _resp(
        200,
        [
            {
                "sha": "abc123deadbeef",
                "html_url": "https://github.com/ikawrakow/ik_llama.cpp/commit/abc",
                "commit": {
                    "committer": {"date": "2025-01-02T00:00:00Z"},
                    "message": "fix thing\n\nbody",
                },
            }
        ],
    )

    with patch.object(llama_github_refs.requests, "get", return_value=body):
        out = llama_github_refs.fetch_ik_llama_main_tip_commit()
    assert out["sha"] == "abc123deadbeef"
    assert out["commit_date"] == "2025-01-02T00:00:00Z"
    assert out["message"] == "fix thing"
    assert "commit/abc" in out["html_url"]
