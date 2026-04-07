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


def fetch_ik_llama_main_tip_commit() -> Optional[dict[str, Any]]:
    """Latest commit on ``main`` (no tags/releases)."""
    response = requests.get(IK_LLAMA_MAIN_COMMITS_URL, allow_redirects=True)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    raw = response.json()
    commits = raw if isinstance(raw, list) else []
    c = commits[0] if commits else None
    if not isinstance(c, dict):
        return None
    sha = c.get("sha")
    if not sha or not isinstance(sha, str):
        return None
    html_url = c.get("html_url")
    if not html_url:
        html_url = f"https://github.com/ikawrakow/ik_llama.cpp/commit/{sha}"
    commit_body = (c.get("commit") or {}) if isinstance(c.get("commit"), dict) else {}
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


def fetch_latest_release_for_repository_source(
    repository_source: str,
) -> Optional[dict]:
    """Non-draft GitHub releases for ``llama.cpp`` only. ik_llama.cpp does not use releases."""
    if repository_source != "llama.cpp":
        return None

    response = requests.get(LLAMA_CPP_RELEASES_URL, allow_redirects=True)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    releases = response.json()
    if isinstance(releases, dict):
        return releases
    if not isinstance(releases, list):
        return None

    for release in releases:
        if isinstance(release, dict) and not release.get("draft"):
            return release
    return None
