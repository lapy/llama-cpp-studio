"""Tests for llama_github_refs (llama.cpp releases; ik_llama.cpp main tip)."""

from unittest.mock import MagicMock, patch


def _resp(status_code=200, json_data=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data if json_data is not None else []
    m.raise_for_status = MagicMock()
    return m


def test_fetch_latest_llama_cpp_release():
    from backend import llama_github_refs

    rel = _resp(
        200,
        [{"tag_name": "b1234", "draft": False, "published_at": "2024-01-01", "html_url": "https://x"}],
    )

    with patch.object(llama_github_refs.requests, "get", return_value=rel) as g:
        out = llama_github_refs.fetch_latest_release_for_repository_source("llama.cpp")
    assert out["tag_name"] == "b1234"
    assert g.call_count == 1
    assert "llama.cpp/releases" in g.call_args[0][0]


def test_fetch_latest_release_ik_llama_is_none():
    from backend import llama_github_refs

    with patch.object(llama_github_refs.requests, "get") as g:
        out = llama_github_refs.fetch_latest_release_for_repository_source("ik_llama.cpp")
    assert out is None
    g.assert_not_called()


def test_fetch_ik_llama_main_tip_commit():
    from backend import llama_github_refs

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
