"""Resolve latest llama.cpp release from GitHub; ik_llama.cpp uses ``main`` tip commit only.

llama.cpp publishes two tracks (see ggml-org/ggml discussion 1579):

- ``vX.Y.Z`` — stable release, GitHub "Latest". This is what Studio installs
  from the release button.
- ``bNNNN`` — nightly, cut on almost every ``master`` commit. ``NNNN`` is the
  commit count on that branch, not a commit SHA. Still a valid source ref.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import requests

# GitHub "Latest" is the newest non-prerelease (a stable ``vX.Y.Z`` tag).
LLAMA_CPP_LATEST_RELEASE_URL = (
    "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
)
# Fallback listing. Nightlies are published far more often than stable tags,
# so the first page is usually only ``bNNNN`` prereleases.
LLAMA_CPP_RELEASES_URL = (
    "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=20"
)
_STABLE_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_NIGHTLY_TAG_RE = re.compile(r"^b\d+$", re.IGNORECASE)
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


def is_stable_release_tag(value: str) -> bool:
    """True for a llama.cpp stable tag such as ``v0.4.1``."""
    return bool(_STABLE_TAG_RE.fullmatch(str(value or "").strip()))


def is_nightly_release_tag(value: str) -> bool:
    """True for a llama.cpp nightly tag such as ``b10566``.

    The number is the ``master`` commit count, not a commit SHA.
    """
    return bool(_NIGHTLY_TAG_RE.fullmatch(str(value or "").strip()))


def _usable_release(release: Any) -> bool:
    if not isinstance(release, dict) or release.get("draft"):
        return False
    return bool(str(release.get("tag_name") or "").strip())


def parse_latest_release(raw: Any) -> Optional[dict]:
    """Pick the llama.cpp release Studio should treat as latest.

    Prefer a non-prerelease ``vX.Y.Z`` tag. If the payload has none (older
    listings, or a page of nightly ``bNNNN`` tags), use the newest non-draft
    release so a build can still be resolved.
    """
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [item for item in raw if isinstance(item, dict)]
    else:
        return None

    usable = [item for item in candidates if _usable_release(item)]
    for item in usable:
        tag = str(item.get("tag_name") or "")
        if not item.get("prerelease") and is_stable_release_tag(tag):
            return item
    for item in usable:
        if not item.get("prerelease"):
            return item
    return usable[0] if usable else None


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
    """Stable ``vX.Y.Z`` release for ``llama.cpp``.

    Uses GitHub Latest, then the releases list when that tag is missing.
    ik_llama.cpp does not use releases.
    """
    if repository_source != "llama.cpp":
        return None

    response = requests.get(
        LLAMA_CPP_LATEST_RELEASE_URL,
        allow_redirects=True,
        timeout=GITHUB_TIMEOUT,
    )
    if response.status_code != 404:
        response.raise_for_status()
        parsed = parse_latest_release(response.json())
        if parsed and is_stable_release_tag(str(parsed.get("tag_name") or "")):
            return parsed

    response = requests.get(
        LLAMA_CPP_RELEASES_URL,
        allow_redirects=True,
        timeout=GITHUB_TIMEOUT,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return parse_latest_release(response.json())
