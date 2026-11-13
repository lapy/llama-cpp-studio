"""Drift audit: every transform asset still applies to the fork's current files.

An asset whose anchor text disappeared fails the install with a
:class:`TransformError`; the user only finds out when the build refuses. This
module applies each engine's assets to the *live* upstream files, so the
mismatch is reported as a test failure naming the asset and the file to update.

It needs the network and the forks' default branches, so it is opt-in::

    STUDIO_UPSTREAM_ANCHORS=1 python -m pytest backend/tests/test_engine_upstream_anchors.py

The scheduled ``upstream-anchors`` workflow runs exactly that command.
"""

from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.transforms import (
    TransformError,
    apply_transform,
    load_transform,
)

_RAW_BASE = "https://raw.githubusercontent.com"


def _raw_url(repo_url: str, branch: str, path: str) -> str:
    """Map an engine repository URL to the raw file URL for ``path``."""
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"https", "http"} or parsed.netloc != "github.com":
        raise ValueError(f"not a GitHub HTTPS repository URL: {repo_url!r}")
    repo = parsed.path.strip("/")
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    if repo.count("/") != 1:
        raise ValueError(f"not an owner/repository path: {repo_url!r}")
    return f"{_RAW_BASE}/{repo}/{branch}/{path}"


def _fetch(url: str) -> str | None:
    """Return the file text, or None when the path does not exist upstream."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "llama-cpp-studio-anchor-audit"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _engines_with_transforms():
    """(engine_id, definition, transform names in recipe order) for real forks."""
    cases = []
    for engine_id, definition in load_default_engine_definitions().definitions.items():
        names: list[str] = []
        for flow in ("source", "sync"):
            for step in dict(definition.recipes).get(flow, ()):
                if step["uses"] != "vcs.apply_transform":
                    continue
                name = step.get("with", {}).get("transform")
                if name and name not in names:
                    names.append(name)
        if names:
            cases.append((engine_id, definition, tuple(names)))
    return cases


def test_raw_url_trims_the_git_suffix_and_uses_the_branch():
    assert (
        _raw_url("https://github.com/dg1kjd/sglang-sxm2.git", "main", "scripts/install_v100.sh")
        == "https://raw.githubusercontent.com/dg1kjd/sglang-sxm2/main/scripts/install_v100.sh"
    )


@pytest.mark.parametrize(
    "repo_url",
    [
        "git@github.com:dg1kjd/sglang-sxm2.git",
        "https://example.test/dg1kjd/sglang-sxm2.git",
        "https://github.com/dg1kjd",
    ],
)
def test_raw_url_rejects_non_github_https_repositories(repo_url):
    with pytest.raises(ValueError):
        _raw_url(repo_url, "main", "scripts/install_v100.sh")


network = pytest.mark.skipif(
    os.environ.get("STUDIO_UPSTREAM_ANCHORS") != "1",
    reason="set STUDIO_UPSTREAM_ANCHORS=1 to audit the forks' live sources",
)

_CASES = _engines_with_transforms()


class AnchorDrift(AssertionError):
    """An asset no longer applies to the fork's current file."""


def _check_assets(engine_id, repo_url, branch, transforms, fetch=_fetch):
    """Apply every transform to a checkout materialized from ``fetch``."""
    with tempfile.TemporaryDirectory(prefix=f"anchors-{engine_id}-") as checkout:
        for name in transforms:
            asset = load_transform(name)
            url = _raw_url(repo_url, branch, asset.file)
            text = fetch(url)
            if text is None:
                if asset.optional_file:
                    continue
                raise AnchorDrift(f"{engine_id}: upstream file is gone: {url}")
            path = Path(checkout, asset.file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        for name in transforms:
            try:
                apply_transform(name, checkout)
            except TransformError as exc:
                raise AnchorDrift(
                    f"{engine_id}: {exc}\n"
                    f"The fork changed; update backend/engines/definitions/assets/{name}.yaml "
                    f"(upstream: {repo_url}@{branch})."
                ) from exc


def test_anchor_drift_names_the_asset_and_the_upstream_file():
    """The offline half of the audit: a changed layout raises an actionable error."""
    name = "sglang_sxm2/install_v100_studio"
    with pytest.raises(AnchorDrift, match="update backend/engines/definitions/assets/"):
        _check_assets(
            "sglang_sxm2",
            "https://github.com/dg1kjd/sglang-sxm2.git",
            "main",
            (name,),
            fetch=lambda url: "echo the fork changed\n",
        )


@network
@pytest.mark.network
@pytest.mark.parametrize(
    "engine_id, definition, transforms", _CASES, ids=[case[0] for case in _CASES]
)
def test_assets_apply_to_the_forks_current_sources(engine_id, definition, transforms):
    lifecycle = dict(definition.lifecycle)
    _check_assets(
        engine_id,
        lifecycle["default_repo_url"],
        lifecycle.get("default_branch") or "main",
        transforms,
    )
