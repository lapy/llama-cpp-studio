"""Upstream version checks for source engines.

Register a new engine here instead of adding another branch to the HTTP route.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from backend.http_client import get_text
from backend.engines.llama_cpp.github_refs import (
    IK_LLAMA_MAIN_COMMITS_URL,
    LLAMA_CPP_LATEST_RELEASE_URL,
    LLAMA_CPP_RELEASES_URL,
    is_stable_release_tag,
    parse_ik_llama_commit,
    parse_latest_release,
)

UpstreamAdapter = Callable[[], Awaitable[dict]]

LLAMA_CPP_COMMITS_URL = (
    "https://api.github.com/repos/ggerganov/llama.cpp/commits?per_page=1"
)


class UpstreamRequestError(Exception):
    def __init__(self, status_code: Optional[int], detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def _get_json(url: str) -> Any:
    try:
        response = await get_text(url)
    except httpx.HTTPError as exc:
        raise UpstreamRequestError(None, f"Network error: {exc}") from exc
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise UpstreamRequestError(
            response.status_code, f"GitHub API error: HTTP {response.status_code}"
        )
    return response.json()


def _commit_summary(tip: Optional[dict]) -> Optional[dict]:
    if not isinstance(tip, dict) or not tip.get("sha"):
        return None
    commit = tip.get("commit") if isinstance(tip.get("commit"), dict) else {}
    committer = (
        commit.get("committer") if isinstance(commit.get("committer"), dict) else {}
    )
    return {
        "sha": tip.get("sha"),
        "commit_date": committer.get("date"),
        "message": commit.get("message"),
    }


async def _llama_cpp_release() -> Optional[dict]:
    """Stable ``vX.Y.Z`` from GitHub Latest, else the newest release listing."""
    latest_payload = await _get_json(LLAMA_CPP_LATEST_RELEASE_URL)
    parsed = parse_latest_release(latest_payload)
    if parsed and is_stable_release_tag(str(parsed.get("tag_name") or "")):
        return parsed
    listing = await _get_json(LLAMA_CPP_RELEASES_URL)
    return parse_latest_release(listing)


async def llama_cpp_update_status() -> dict:
    release = await _llama_cpp_release()
    commit_payload = await _get_json(LLAMA_CPP_COMMITS_URL)
    commits = commit_payload if isinstance(commit_payload, list) else []
    tip = commits[0] if commits else None
    return {
        "latest_release": (
            {
                "tag_name": release["tag_name"],
                "published_at": release.get("published_at"),
                "html_url": release.get("html_url"),
            }
            if release
            else None
        ),
        "latest_commit": _commit_summary(tip if isinstance(tip, dict) else None),
    }


async def ik_llama_update_status() -> dict:
    payload = await _get_json(IK_LLAMA_MAIN_COMMITS_URL)
    tip = parse_ik_llama_commit(payload)
    return {
        "latest_release": None,
        "latest_commit": (
            {
                "sha": tip["sha"],
                "commit_date": tip.get("commit_date"),
                "message": tip.get("message"),
            }
            if tip
            else None
        ),
    }


UPSTREAM_ADAPTERS: Dict[str, UpstreamAdapter] = {
    "llama_cpp": llama_cpp_update_status,
    "ik_llama": ik_llama_update_status,
}


def register_upstream_adapter(engine_id: str, adapter: UpstreamAdapter) -> None:
    UPSTREAM_ADAPTERS[engine_id] = adapter


async def check_engine_updates(source: Optional[str]) -> dict:
    engine_id = "ik_llama" if source == "ik_llama" else "llama_cpp"
    adapter = UPSTREAM_ADAPTERS.get(engine_id)
    if adapter is None:
        raise UpstreamRequestError(404, f"No upstream adapter registered for {engine_id}")
    return await adapter()


async def resolve_build_ref(engine: str) -> tuple[str, str]:
    """Return ``(source_ref, ref_type)`` for an engine update build."""
    if engine == "ik_llama":
        payload = await _get_json(IK_LLAMA_MAIN_COMMITS_URL)
        tip = parse_ik_llama_commit(payload)
        if not tip or not tip.get("sha"):
            raise UpstreamRequestError(
                404, "Could not resolve latest commit on ik_llama.cpp main"
            )
        return tip["sha"], "ref"
    release = await _llama_cpp_release()
    if not release or not release.get("tag_name"):
        raise UpstreamRequestError(404, "No release found for this engine")
    return release["tag_name"], "release"
