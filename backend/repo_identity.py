"""Identify whether a source repo URL is a fork of an engine's canonical GitHub repo."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

# Canonical upstream remotes (owner is what defines “not a fork”).
CANONICAL_REPOSITORY_URLS: Dict[str, str] = {
    "llama_cpp": "https://github.com/ggerganov/llama.cpp.git",
    "ik_llama": "https://github.com/ikawrakow/ik_llama.cpp.git",
    "audio_cpp": "https://github.com/0xShug0/audio.cpp.git",
    "lmdeploy": "https://github.com/InternLM/lmdeploy.git",
    "1cat_vllm": "https://github.com/1CatAI/1Cat-vLLM.git",
}

_GITHUB_SSH = re.compile(
    r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
    re.IGNORECASE,
)


def normalize_git_url(url: str) -> str:
    value = str(url or "").strip().rstrip("/")
    if value.lower().endswith(".git"):
        value = value[:-4]
    return value


def github_owner_repo(url: str) -> Optional[Tuple[str, str]]:
    """Return (owner, repo) for a GitHub remote, or None if not parseable."""
    value = str(url or "").strip()
    if not value:
        return None

    ssh = _GITHUB_SSH.match(value)
    if ssh:
        return ssh.group("owner"), ssh.group("repo")

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if host not in {"github.com", "www.github.com"}:
        return None
    parts = [p for p in (parsed.path or "").split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return owner, repo


def is_github_fork(repository_url: str, canonical_url: str) -> bool:
    """True when the GitHub username/org differs from the canonical upstream owner.

    Non-GitHub remotes that are not the same URL as canonical are also treated as forks.
    """
    remote = str(repository_url or "").strip()
    canonical = str(canonical_url or "").strip()
    if not remote:
        return False
    if not canonical:
        return False

    remote_gh = github_owner_repo(remote)
    canon_gh = github_owner_repo(canonical)
    if canon_gh and remote_gh:
        return remote_gh[0].lower() != canon_gh[0].lower()

    # Different host / unparseable GitHub → custom remote, mark as fork.
    return normalize_git_url(remote).lower() != normalize_git_url(canonical).lower()


def is_github_fork_for_engine(engine_id: str, repository_url: str) -> bool:
    canonical = CANONICAL_REPOSITORY_URLS.get(str(engine_id or ""))
    if not canonical:
        return False
    return is_github_fork(repository_url, canonical)


def source_build_type_labels(
    repository_url: str,
    canonical_url: str,
    *,
    patches: bool = False,
) -> Dict[str, Any]:
    """Labels for a source-built version row.

    - ``type``: user-facing badge (fork / patched / source)
    - ``install_type``: behavioral kind for sync/activate (always source for git builds)
    - ``is_fork``: explicit flag
    """
    fork = is_github_fork(repository_url, canonical_url)
    if fork:
        display = "fork"
    elif patches:
        display = "patched"
    else:
        display = "source"
    return {
        "type": display,
        "install_type": "source",
        "is_fork": fork,
    }


def source_build_type_labels_for_engine(
    engine_id: str,
    repository_url: str,
    *,
    patches: bool = False,
) -> Dict[str, Any]:
    canonical = CANONICAL_REPOSITORY_URLS.get(str(engine_id or ""), "")
    return source_build_type_labels(repository_url, canonical, patches=patches)
