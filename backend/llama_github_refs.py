"""Resolve latest llama.cpp release from GitHub; ik_llama.cpp uses ``main`` tip commit only."""

from __future__ import annotations

from typing import Any, Optional

import requests

LLAMA_CPP_RELEASES_URL = (
    "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=10"
)
IK_LLAMA_MAIN_COMMITS_URL = (
    "https://api.github.com/repos/ikawrakow/ik_llama.cpp/commits?sha=main&per_page=1"
)
# Connect and read deadlines so a stalled GitHub call cannot block a worker forever.
GITHUB_TIMEOUT = (5, 20)


def parse_ik_llama_commit(raw: Any) -> Optional[dict[str, Any]]:
    commits = raw if isinstance(raw, list) else []
    commit = commits[0] if commits else None
    if not isinstance(commit, dict):
        return None
    sha = commit.get("sha")
    if not sha or not isinstance(sha, str):
        return None
    html_url = commit.get("html_url")
    if not html_url:
        html_url = f"https://github.com/ikawrakow/ik_llama.cpp/commit/{sha}"
    commit_body = (
        (commit.get("commit") or {}) if isinstance(commit.get("commit"), dict) else {}
    )
    committer = (
        (commit_body.get("committer") or {})
        if isinstance(commit_body.get("committer"), dict)
        else {}
    )
    message = commit_body.get("message")
    if isinstance(message, str):
        message = message.split("\n", 1)[0].strip()
    else:
        message = None
    return {
        "sha": sha,
        "html_url": html_url,
        "commit_date": committer.get("date"),
        "message": message,
    }


def parse_latest_release(raw: Any) -> Optional[dict]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, list):
        return None
    for release in raw:
        if isinstance(release, dict) and not release.get("draft"):
            return release
    return None


def fetch_ik_llama_main_tip_commit() -> Optional[dict[str, Any]]:
    """Latest commit on ``main`` (no tags/releases)."""
    response = requests.get(
        IK_LLAMA_MAIN_COMMITS_URL,
        allow_redirects=True,
        timeout=GITHUB_TIMEOUT,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return parse_ik_llama_commit(response.json())


def fetch_latest_release_for_repository_source(
    repository_source: str,
) -> Optional[dict]:
    """Non-draft GitHub releases for ``llama.cpp`` only. ik_llama.cpp does not use releases."""
    if repository_source != "llama.cpp":
        return None

    response = requests.get(
        LLAMA_CPP_RELEASES_URL,
        allow_redirects=True,
        timeout=GITHUB_TIMEOUT,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return parse_latest_release(response.json())
