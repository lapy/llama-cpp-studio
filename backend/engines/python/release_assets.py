"""GitHub release wheel selection with declared asset rules.

A recipe can ask for release assets without engine-specific Python: it declares
ordered groups of filename prefixes and the prefixes that make a group
installable. The first installable group wins; its matched wheel URLs are
returned with the release tag.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Tuple

import httpx


class ReleaseAssetError(RuntimeError):
    """The release does not contain an installable asset group."""


async def fetch_github_release(repository: str, version: Any = None) -> Mapping[str, Any]:
    """Return the GitHub release JSON for a tag (or the latest release)."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        if version:
            tag = str(version)
            tag = tag if tag.startswith("v") else f"v{tag}"
            url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
        else:
            url = f"https://api.github.com/repos/{repository}/releases/latest"
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


def select_wheel_urls(
    release: Mapping[str, Any],
    groups: List[Mapping[str, Any]],
    *,
    error: str,
) -> Tuple[str, List[str]]:
    """Return ``(tag, wheel_urls)`` for the first matching declared group."""
    assets = release.get("assets") or []
    for group in groups:
        prefixes = [str(prefix).lower() for prefix in (group.get("prefixes") or ())]
        required = [
            str(prefix).lower() for prefix in (group.get("required_prefixes") or prefixes[:1])
        ]
        matched: Dict[str, str] = {}
        for asset in assets:
            name = str(asset.get("name") or "").lower()
            if not name.endswith(".whl"):
                continue
            for prefix in prefixes:
                if prefix and name.startswith(prefix):
                    url = asset.get("browser_download_url")
                    if url:
                        matched[prefix] = str(url)
                    break
        if not any(prefix in matched for prefix in required):
            continue
        urls = [matched[prefix] for prefix in prefixes if prefix in matched]
        if urls:
            return str(release.get("tag_name") or ""), urls
    raise ReleaseAssetError(error)
