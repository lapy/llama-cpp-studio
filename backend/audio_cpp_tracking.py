"""Persisted tracking settings for audio.cpp updates (outside cmake build config)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

import requests

from backend.audio_cpp_manager import (
    AUDIO_CPP_REPOSITORY,
    AudioCppBuildConfig,
    get_audio_cpp_manager,
)
from backend.data_store import get_store
from backend.logging_config import get_logger

logger = get_logger(__name__)

_TRACKING_KEYS = frozenset({"tracking_ref", "repository_url"})
_GITHUB_REPO = "0xShug0/audio.cpp"


def _cmake_dict(raw: Optional[dict]) -> Dict[str, Any]:
    return get_audio_cpp_manager().build_config_from_dict(raw).__dict__


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
        get_audio_cpp_manager().build_config_from_dict(build_config).__dict__
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


def resolve_bootstrap_tracking_ref() -> str:
    """Resolve a default tracking ref from GitHub (latest release tag or default branch)."""

    def _request() -> str:
        headers = {"Accept": "application/vnd.github+json"}
        try:
            release = requests.get(
                f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest",
                headers=headers,
                timeout=20,
            )
            if release.status_code == 200:
                tag = str((release.json() or {}).get("tag_name") or "").strip()
                if tag:
                    return tag
        except requests.RequestException as exc:
            logger.debug("audio.cpp latest release lookup failed: %s", exc)
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
    "merge_settings",
    "resolve_bootstrap_tracking_ref",
    "split_settings",
]
