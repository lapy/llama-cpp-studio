"""Persisted tracking settings for audio.cpp updates (outside cmake build config).

audio.cpp publishes GitHub Releases whose tags follow ``vX.Y.Z``
(e.g. ``v0.7.0``) or the older ``release-X.Y(.Z)`` form, with
``target_commitish`` typically ``main``. Studio builds from the tag or
branch (source), not from Windows prebuilt zip assets.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional, Tuple

import requests

from backend.audio_cpp_manager import (
    AUDIO_CPP_REPOSITORY,
    AudioCppBuildConfig,
    get_audio_cpp_manager,
)
from backend.audio_build_options import coerce_build_settings
from backend.data_store import get_store
from backend.logging_config import get_logger

logger = get_logger(__name__)

_TRACKING_KEYS = frozenset({"tracking_ref", "repository_url"})
_GITHUB_REPO = "0xShug0/audio.cpp"

# Current: v0.7.0
# Older: release-0.5.1, release-0.3-qwen3-tts
# Legacy: v0.2.0-windows-prebuilt
_AUDIO_RELEASE_TAG_RE = re.compile(
    r"^(?:"
    r"release-\d+(?:\.\d+)*(?:[-+][\w.-]*)?"
    r"|v?\d+(?:\.\d+)+(?:[-+][\w.-]*)?"
    r")$",
    re.IGNORECASE,
)


def is_audio_release_tag(value: str) -> bool:
    """True if *value* looks like an audio.cpp GitHub release tag."""
    return bool(_AUDIO_RELEASE_TAG_RE.fullmatch(str(value or "").strip()))


def _cmake_dict(raw: Optional[dict]) -> Dict[str, Any]:
    return coerce_build_settings(raw)


def split_settings(raw: Optional[dict]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split stored settings into tracking fields and cmake build config."""
    raw = raw if isinstance(raw, dict) else {}
    tracking = {
        "tracking_ref": str(raw.get("tracking_ref") or "").strip(),
        "repository_url": str(raw.get("repository_url") or "").strip(),
    }
    return tracking, _cmake_dict(raw)


def merge_settings(
    *,
    tracking_ref: Optional[str] = None,
    repository_url: Optional[str] = None,
    build_config: Optional[dict] = None,
    existing: Optional[dict] = None,
) -> Dict[str, Any]:
    existing_tracking, existing_cmake = split_settings(existing)
    cmake = (
        coerce_build_settings(build_config)
        if build_config is not None
        else existing_cmake
    )
    return {
        **cmake,
        "tracking_ref": (
            str(tracking_ref).strip()
            if tracking_ref is not None
            else existing_tracking["tracking_ref"]
        ),
        "repository_url": (
            str(repository_url).strip()
            if repository_url is not None
            else existing_tracking["repository_url"]
            or AUDIO_CPP_REPOSITORY
        ),
    }


def resolve_latest_github_release() -> Optional[Dict[str, Any]]:
    """Return metadata for the latest GitHub release, or None.

    Prefer ``/releases/latest`` (non-prerelease). Tags are ``vX.Y.Z`` or
    the older ``release-*`` convention.
    """

    def _request() -> Optional[Dict[str, Any]]:
        headers = {"Accept": "application/vnd.github+json"}
        try:
            response = requests.get(
                f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest",
                headers=headers,
                timeout=20,
            )
            if response.status_code != 200:
                return None
            body = response.json() or {}
            tag = str(body.get("tag_name") or "").strip()
            if not tag:
                return None
            return {
                "tag_name": tag,
                "name": str(body.get("name") or "").strip() or tag,
                "html_url": str(body.get("html_url") or "").strip() or None,
                "published_at": body.get("published_at"),
                "target_commitish": str(body.get("target_commitish") or "").strip()
                or None,
                "prerelease": bool(body.get("prerelease")),
            }
        except requests.RequestException as exc:
            logger.debug("audio.cpp latest release lookup failed: %s", exc)
        return None

    return _request()


def resolve_latest_release_tag() -> Optional[str]:
    """Return the latest GitHub release tag for audio.cpp, or None."""
    release = resolve_latest_github_release()
    if not release:
        return None
    return str(release.get("tag_name") or "").strip() or None


def resolve_bootstrap_tracking_ref() -> str:
    """Resolve a default tracking ref from GitHub (latest release tag or default branch)."""
    tag = resolve_latest_release_tag()
    if tag:
        return tag

    def _request() -> str:
        headers = {"Accept": "application/vnd.github+json"}
        try:
            repo = requests.get(
                f"https://api.github.com/repos/{_GITHUB_REPO}",
                headers=headers,
                timeout=20,
            )
            repo.raise_for_status()
            branch = str((repo.json() or {}).get("default_branch") or "").strip()
            if branch:
                return branch
        except requests.RequestException as exc:
            logger.warning("audio.cpp default branch lookup failed: %s", exc)
        return "main"

    return _request()


async def ensure_tracking_settings(store=None) -> Dict[str, Any]:
    """Return settings with tracking_ref/repository_url populated and persisted if missing."""
    store = store or get_store()
    raw = store.get_engine_build_settings("audio_cpp") or {}
    tracking, cmake = split_settings(raw)
    changed = False
    if not tracking["repository_url"]:
        tracking["repository_url"] = AUDIO_CPP_REPOSITORY
        changed = True
    if not tracking["tracking_ref"]:
        tracking["tracking_ref"] = await asyncio.to_thread(resolve_bootstrap_tracking_ref)
        changed = True
    merged = {**cmake, **tracking}
    if changed or set(raw.keys()) & _TRACKING_KEYS != _TRACKING_KEYS:
        store.update_engine_build_settings("audio_cpp", merged)
    return merged


def get_tracking_and_build(store=None) -> Tuple[str, str, AudioCppBuildConfig]:
    """Synchronous read of tracking ref, repo URL, and build config (no bootstrap)."""
    store = store or get_store()
    tracking, cmake = split_settings(store.get_engine_build_settings("audio_cpp"))
    return (
        tracking["tracking_ref"] or "main",
        tracking["repository_url"] or AUDIO_CPP_REPOSITORY,
        get_audio_cpp_manager().build_config_from_dict(cmake),
    )


__all__ = [
    "ensure_tracking_settings",
    "get_tracking_and_build",
    "is_audio_release_tag",
    "merge_settings",
    "resolve_bootstrap_tracking_ref",
    "resolve_latest_github_release",
    "resolve_latest_release_tag",
    "split_settings",
]
