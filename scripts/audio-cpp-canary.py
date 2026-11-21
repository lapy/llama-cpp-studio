#!/usr/bin/env python3
"""Canary the newest audio.cpp release against Studio's integration contract.

Resolve the latest GitHub release (``--tag`` pins one), download its Linux x64
CPU prebuilt archive, verify the published ``sha256`` digest when available,
extract it, and run ``audiocpp_cli --list-loaders --json``. The loader payload
is composed with
:func:`backend.engines.audio_cpp.integration.composer.compose_version_contract`
and checked against the assumptions Studio's audio integration depends on:

* loader JSON ``schema_version`` is 1 and parses into at least one family;
* every advertised ``api_endpoints`` value maps to a registry operation or to
  the generic ``/v1/tasks/run`` route (unresolved names are listed exactly);
* the required core operations exist (speech, transcriptions,
  transcriptions_details, alignments, tasks_run);
* contract drift warnings are reported, and unreviewed ones fail the run;
* when the release ships ``tools/model_manager_v2.py``, its ``list --json``
  catalog returns package rows (a missing script is a warning: binary release
  archives may omit source-only tools).

A binary that cannot execute at all (loader/ABI error, for example a prebuilt
that requires a newer glibc than the host) is an explicit ``abi_unsupported``
failure. It fails the run; it is never skipped.

Failure taxonomy (the ``failure`` field on failed report checks):
``contract_drift``     the engine contradicts a Studio contract assumption;
``abi_unsupported``    the binary cannot run on this host (glibc/loader);
``download_integrity`` release, asset, digest, size, or archive problem;
``probe_error``        the CLI ran but its probe output was unusable.

Usage:
    python scripts/audio-cpp-canary.py --report /tmp/audio-cpp-canary.json
    python scripts/audio-cpp-canary.py --tag v0.9.3
    python scripts/audio-cpp-canary.py --keep

Exit status is 0 when every check passed (warnings allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from backend.engines.audio_cpp.integration.composer import compose_version_contract
from backend.engines.audio_cpp.integration.contract_registry import (
    CONTRACT_SCHEMA,
    TASKS_RUN_ALIAS,
    operation_for_native_path,
    operation_for_studio_path,
)
from backend.engines.audio_cpp.prebuilt import HostTarget, extract_archive, select_prebuilt
from backend.engines.scan.help_parsers import (
    parse_audio_cpp_loaders_json,
    try_parse_json_payload,
)

GITHUB_REPOSITORY = "0xShug0/audio.cpp"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases"
RELEASES_PAGE = f"https://github.com/{GITHUB_REPOSITORY}/releases"
CANARY_REPORT_SCHEMA = "audio-cpp-canary/v1"

FAILURE_KINDS = (
    "contract_drift",
    "abi_unsupported",
    "download_integrity",
    "probe_error",
)

#: The generic task route is the one advertised route allowed to have no
#: dedicated registry operation; Studio forwards it through the task proxy.
GENERIC_TASK_ROUTES = frozenset({TASKS_RUN_ALIAS, "/v1/tasks/run"})

#: Operations every Studio consumer expects the composed contract to expose.
REQUIRED_OPERATIONS = (
    "speech",
    "transcriptions",
    "transcriptions_details",
    "alignments",
    "tasks_run",
)

REQUIRED_LOADER_SCHEMA = 1

#: Reviewed drift warnings tolerated without ``--allow-drift``. Keep empty
#: unless a warning has been triaged AND the contract registry was updated;
#: unresolved ``api_endpoints`` still fail the endpoint check regardless.
KNOWN_DRIFT_WARNINGS: Tuple[str, ...] = ()

USER_AGENT = "llama-cpp-studio-audio-canary"
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_STDERR_TAIL = 4000

_DOWNLOAD_LINK_RE = re.compile(
    r'href="(?P<path>/[^"]*?/releases/download/[^"]*?/(?P<name>[^"/]+))"'
)
_SHA256_RE = re.compile(r"sha256:([0-9a-fA-F]{64})")
_GLIBC_TEXT_RE = re.compile(r"GLIBC_(\d+\.\d+)")
_GLIBC_BYTES_RE = re.compile(rb"GLIBC_(\d+\.\d+)")

#: stderr/shared-library text that means the binary itself cannot run here.
_ABI_MARKERS = (
    "glibc_",
    "error while loading shared libraries",
    "cannot open shared object file",
    "cannot execute",
    "exec format error",
    "elf interpreter",
    "symbol lookup error",
)


class CanaryFailure(Exception):
    """A classified failure; ``kind`` is one of :data:`FAILURE_KINDS`."""

    def __init__(self, kind: str, message: str, *, details: Optional[dict] = None):
        if kind not in FAILURE_KINDS:
            raise ValueError(f"unknown canary failure kind: {kind}")
        super().__init__(message)
        self.kind = kind
        self.message = str(message)
        self.details = dict(details or {})


class _ReleaseLookupError(RuntimeError):
    """The GitHub API or release page did not yield release metadata."""


@dataclass(frozen=True)
class CanaryOptions:
    tag: str = ""
    asset: str = ""
    report: str = ""
    allow_drift: bool = False
    keep: bool = False
    http_timeout: float = 20.0
    probe_timeout: float = 60.0


@dataclass
class ProbeResult:
    argv: List[str]
    cwd: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    exec_error: str = ""


@dataclass
class Check:
    id: str
    status: str
    message: str
    failure: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "failure": self.failure,
            "message": self.message,
            "details": self.details,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _version_tuple(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return (0,)


def host_glibc_version() -> str:
    """Host glibc version (e.g. ``2.39``), or ``""`` on a non-glibc host."""
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION") or ""
    except (AttributeError, OSError, ValueError):
        value = ""
    parts = value.split()
    if parts:
        return parts[-1]
    lib, version = platform.libc_ver()
    return version if lib.lower() == "glibc" else ""


def required_glibc_from_text(text: str) -> str:
    versions = _GLIBC_TEXT_RE.findall(str(text or ""))
    return max(versions, key=_version_tuple) if versions else ""


def required_glibc_from_binary(path: str) -> str:
    """Highest ``GLIBC_x.y`` string embedded in the binary, best effort.

    The dynamic loader prints required versions when it fails; scanning the
    file covers a missing-interpreter ``ENOENT`` where stderr is empty.
    """
    found = set()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                found.update(
                    match.group(1).decode("ascii") for match in _GLIBC_BYTES_RE.finditer(chunk)
                )
    except OSError:
        return ""
    return max(found, key=_version_tuple) if found else ""


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _release_headers() -> Dict[str, str]:
    return {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}


def _normalize_release(body: dict, *, source: str) -> dict:
    tag = str(body.get("tag_name") or body.get("tag") or "").strip()
    if not tag:
        raise _ReleaseLookupError("release metadata has no tag_name")
    assets: List[dict] = []
    for raw in body.get("assets") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        url = str(raw.get("browser_download_url") or raw.get("url") or "").strip()
        if not name or not url:
            continue
        assets.append(
            {
                "name": name,
                "browser_download_url": url,
                "size": raw.get("size"),
                "digest": str(raw.get("digest") or "").strip() or None,
            }
        )
    return {
        "tag": tag,
        "name": str(body.get("name") or "").strip() or tag,
        "html_url": str(body.get("html_url") or "").strip() or None,
        "published_at": body.get("published_at"),
        "prerelease": body.get("prerelease"),
        "assets": assets,
        "source": source,
    }


def _fetch_release_api(tag: str, timeout: float) -> dict:
    url = f"{RELEASES_API}/tags/{quote(tag, safe='')}" if tag else f"{RELEASES_API}/latest"
    try:
        response = requests.get(url, headers=_release_headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise _ReleaseLookupError(f"request failed: {exc}") from exc
    if response.status_code != 200:
        raise _ReleaseLookupError(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise _ReleaseLookupError("response was not JSON") from exc
    if not isinstance(body, dict):
        raise _ReleaseLookupError("release response was not a JSON object")
    return _normalize_release(body, source="github-api")


def _latest_tag_from_release_page(timeout: float) -> str:
    """Resolve the latest tag from the releases page redirect (no API quota)."""
    url = f"{RELEASES_PAGE}/latest"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            allow_redirects=False,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise _ReleaseLookupError(f"release page request failed: {exc}") from exc
    location = str(response.headers.get("Location") or response.url or "")
    marker = "/releases/tag/"
    if marker not in location:
        raise _ReleaseLookupError(
            f"release page did not redirect to a tag (HTTP {response.status_code})"
        )
    tag = location.split(marker, 1)[1].split("?", 1)[0].split("#", 1)[0].strip("/")
    if not tag:
        raise _ReleaseLookupError("release page redirect had an empty tag")
    return tag


def parse_release_assets_html(html: str) -> List[dict]:
    """Parse assets (and their published sibling sha256) from a release page.

    GitHub's ``releases/expanded_assets/<tag>`` fragment lists each asset as a
    download link followed by its ``sha256:`` digest. Parsing it keeps the
    canary useful when the unauthenticated API is rate limited (the norm on
    shared CI runner IPs) and still lets the run verify the digest.
    """
    text = str(html or "")
    links = list(_DOWNLOAD_LINK_RE.finditer(text))
    assets: List[dict] = []
    seen = set()
    for index, match in enumerate(links):
        name = unescape(match.group("name")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        end = links[index + 1].start() if index + 1 < len(links) else len(text)
        digest_match = _SHA256_RE.search(text, match.end(), end)
        assets.append(
            {
                "name": name,
                "browser_download_url": f"https://github.com{unescape(match.group('path'))}",
                "size": None,
                "digest": f"sha256:{digest_match.group(1)}" if digest_match else None,
            }
        )
    return assets


def _assets_from_release_page(tag: str, timeout: float) -> List[dict]:
    url = f"{RELEASES_PAGE}/expanded_assets/{quote(tag, safe='')}"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise _ReleaseLookupError(f"release assets request failed: {exc}") from exc
    if response.status_code != 200:
        raise _ReleaseLookupError(f"HTTP {response.status_code}")
    assets = parse_release_assets_html(response.text)
    if not assets:
        raise _ReleaseLookupError("release assets page listed no downloadable assets")
    return assets


def resolve_release(tag: str, timeout: float) -> dict:
    """Return normalized release metadata for ``tag`` (or the latest release).

    The GitHub API is authoritative and supplies the asset ``digest``. When the
    unauthenticated quota is exhausted, fall back to the public release page;
    it publishes the same ``sha256`` digests next to each asset.
    """
    wanted = str(tag or "").strip()
    api_error = ""
    try:
        return _fetch_release_api(wanted, timeout)
    except _ReleaseLookupError as exc:
        api_error = str(exc)

    try:
        resolved_tag = wanted or _latest_tag_from_release_page(timeout)
        assets = _assets_from_release_page(resolved_tag, timeout)
    except _ReleaseLookupError as exc:
        raise CanaryFailure(
            "download_integrity",
            f"could not resolve audio.cpp release {wanted or 'latest'}: "
            f"GitHub API: {api_error}; release page: {exc}",
        ) from exc
    return {
        "tag": resolved_tag,
        "name": resolved_tag,
        "html_url": f"{RELEASES_PAGE}/tag/{resolved_tag}",
        "published_at": None,
        "prerelease": None,
        "assets": assets,
        "source": "github-releases-page",
    }


def select_cpu_asset(release: dict, asset_name: str = "") -> Tuple[dict, str]:
    """Pick the Linux x64 CPU prebuilt (or an explicit ``--asset`` override)."""
    assets = [item for item in release.get("assets") or [] if isinstance(item, dict)]
    tag = str(release.get("tag") or "")
    wanted = str(asset_name or "").strip()
    if wanted:
        selected = next((item for item in assets if item.get("name") == wanted), None)
        if selected is None:
            raise CanaryFailure(
                "download_integrity",
                f"release {tag} has no asset named {wanted!r}",
                details={"available_assets": sorted(str(a.get("name")) for a in assets)},
            )
        reason = "explicit --asset override"
    else:
        # Always ask for the portable CPU archive: the canary only needs to run
        # the CLI on the CI host, never to use that host's AVX-512 or GPU.
        plan = select_prebuilt(assets, HostTarget(linux_x64=True))
        if plan is None:
            available = sorted(
                str(item.get("name")) for item in assets if "ubuntu-x64" in str(item.get("name"))
            )
            raise CanaryFailure(
                "download_integrity",
                f"release {tag} has no Linux x64 CPU prebuilt archive",
                details={"available_ubuntu_assets": available},
            )
        selected = next(
            (item for item in assets if item.get("name") == plan.asset_name),
            None,
        )
        if selected is None:
            raise CanaryFailure(
                "download_integrity",
                f"release {tag} selected {plan.asset_name!r} but it is absent from the asset list",
            )
        reason = plan.reason
    if not str(selected.get("browser_download_url") or "").strip():
        raise CanaryFailure(
            "download_integrity",
            f"asset {selected.get('name')!r} has no download URL",
        )
    return selected, reason


def download_asset(url: str, dest: str, *, timeout: float) -> int:
    """Stream one HTTP(S) release asset to ``dest``; return its byte size."""
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"}:
        raise CanaryFailure("download_integrity", f"refusing to download non-HTTP URL: {url}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    received = 0
    try:
        with requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            allow_redirects=True,
            timeout=(min(15.0, timeout), timeout),
        ) as response:
            response.raise_for_status()
            with open(dest, "wb") as handle:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > MAX_DOWNLOAD_BYTES:
                        raise CanaryFailure(
                            "download_integrity",
                            f"asset exceeded the {MAX_DOWNLOAD_BYTES} byte download cap",
                        )
                    handle.write(chunk)
    except requests.RequestException as exc:
        raise CanaryFailure("download_integrity", f"asset download failed: {exc}") from exc
    except OSError as exc:
        raise CanaryFailure("download_integrity", f"asset download failed: {exc}") from exc
    return received


def verify_asset_digest(path: str, asset: dict) -> Tuple[str, str]:
    """Verify the published digest when present; return ``(sha256, warning)``."""
    actual = _sha256_file(path)
    published = str(asset.get("digest") or "").strip()
    if not published:
        return actual, "asset has no published digest; integrity rests on the archive SHA256SUMS"
    algorithm, _, expected = published.partition(":")
    if algorithm.lower() != "sha256" or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise CanaryFailure(
            "download_integrity",
            f"asset digest {published!r} is not a supported sha256 digest",
        )
    if actual.lower() != expected.lower():
        raise CanaryFailure(
            "download_integrity",
            f"asset digest mismatch for {asset.get('name')}: "
            f"expected {expected.lower()}, got {actual}",
            details={"expected_sha256": expected.lower(), "actual_sha256": actual},
        )
    return actual, ""


def verify_asset_size(path: str, asset: dict) -> int:
    """Reject a truncated/extended download when the release declares a size."""
    actual = os.path.getsize(path)
    published = asset.get("size")
    if isinstance(published, int) and published > 0 and actual != published:
        raise CanaryFailure(
            "download_integrity",
            f"asset size mismatch for {asset.get('name')}: "
            f"expected {published} bytes, got {actual}",
            details={"expected_size": published, "actual_size": actual},
        )
    return actual


def extract_release_archive(archive_path: str, dest_dir: str) -> dict:
    try:
        binaries = extract_archive(archive_path, dest_dir)
    except Exception as exc:  # extract_archive signals shape/checksum problems
        raise CanaryFailure("download_integrity", f"archive extraction failed: {exc}") from exc
    cli = str(binaries.get("cli_binary_path") or "")
    if not cli or not os.path.isfile(cli):
        raise CanaryFailure("download_integrity", "release archive did not contain audiocpp_cli")
    return binaries


def _library_dirs(cli_path: str, root: str) -> List[str]:
    candidates = [
        os.path.dirname(os.path.abspath(cli_path)),
        os.path.abspath(root),
        os.path.join(root, "lib"),
        os.path.join(root, "lib64"),
        os.path.join(root, "build", "lib"),
        os.path.join(root, "build", "bin"),
    ]
    dirs: List[str] = []
    seen = set()
    for candidate in candidates:
        path = os.path.abspath(candidate)
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        dirs.append(path)
    return dirs


def probe_env(cli_path: str, root: str) -> Dict[str, str]:
    env = dict(os.environ)
    existing = str(env.get("LD_LIBRARY_PATH") or "").strip()
    library_dirs = _library_dirs(cli_path, root)
    if existing:
        library_dirs.append(existing)
    if library_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    return env


def run_cli_probe(cli_path: str, *, cwd: str, timeout: float) -> ProbeResult:
    argv = [cli_path, "--list-loaders", "--json"]
    result = ProbeResult(argv=argv, cwd=cwd)
    try:
        process = subprocess.run(
            argv,
            cwd=cwd,
            env=probe_env(cli_path, cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result.timed_out = True
        result.stdout = str(exc.stdout or "")
        result.stderr = str(exc.stderr or "")
        return result
    except OSError as exc:
        result.exec_error = f"{type(exc).__name__}: {exc}"
        return result
    result.returncode = process.returncode
    result.stdout = process.stdout or ""
    result.stderr = process.stderr or ""
    return result


def _looks_like_abi_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _ABI_MARKERS)


def _stderr_tail(text: str) -> str:
    return str(text or "").strip()[-MAX_STDERR_TAIL:]


def _abi_failure(probe: ProbeResult, cli_path: str, *, host_glibc: str) -> CanaryFailure:
    text = "\n".join(part for part in (probe.stderr, probe.stdout, probe.exec_error) if part)
    required = required_glibc_from_text(text) or required_glibc_from_binary(cli_path)
    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    if required:
        message = (
            f"audio.cpp CLI cannot execute on this host: requires GLIBC_{required}, "
            f"host glibc is {host_glibc or 'unknown'} (abi_unsupported)"
        )
    else:
        message = (
            f"audio.cpp CLI cannot execute on this host (host glibc {host_glibc or 'unknown'}) "
            "(abi_unsupported)"
        )
    if last_line:
        message = f"{message}: {last_line}"
    return CanaryFailure(
        "abi_unsupported",
        message,
        details={
            "required_glibc": required or None,
            "host_glibc": host_glibc or None,
            "returncode": probe.returncode,
            "exec_error": probe.exec_error or None,
            "stderr_tail": _stderr_tail(text),
        },
    )


def classify_probe_failure(
    probe: ProbeResult, *, cli_path: str, host_glibc: str = ""
) -> Optional[CanaryFailure]:
    """Map a probe result to ``abi_unsupported``/``probe_error`` (or None)."""
    if probe.timed_out:
        return CanaryFailure(
            "probe_error",
            f"audiocpp_cli --list-loaders --json timed out after the probe timeout: {probe.argv}",
        )
    if probe.exec_error:
        if _looks_like_abi_error(probe.exec_error) or "no such file or directory" in (
            probe.exec_error.lower()
        ):
            return _abi_failure(probe, cli_path, host_glibc=host_glibc)
        return CanaryFailure("probe_error", f"could not start audiocpp_cli: {probe.exec_error}")
    if probe.returncode != 0:
        if _looks_like_abi_error(probe.stderr) or _looks_like_abi_error(probe.stdout):
            return _abi_failure(probe, cli_path, host_glibc=host_glibc)
        return CanaryFailure(
            "probe_error",
            "audiocpp_cli --list-loaders --json exited "
            f"{probe.returncode}: {_stderr_tail(probe.stderr)}",
            details={"returncode": probe.returncode, "stderr_tail": _stderr_tail(probe.stderr)},
        )
    if not str(probe.stdout or "").strip():
        return CanaryFailure(
            "probe_error",
            "audiocpp_cli --list-loaders --json printed no output",
            details={"stderr_tail": _stderr_tail(probe.stderr)},
        )
    return None


def parse_loaders_payload(payload: Any) -> Tuple[dict, dict]:
    """Validate the loader JSON envelope and return ``(payload, capabilities)``."""
    if not isinstance(payload, dict):
        raise CanaryFailure("probe_error", "loader JSON root is not an object")
    schema = payload.get("schema_version")
    if schema != REQUIRED_LOADER_SCHEMA:
        raise CanaryFailure(
            "contract_drift",
            f"loader JSON schema_version is {schema!r}, expected {REQUIRED_LOADER_SCHEMA}",
            details={"schema_version": schema},
        )
    capabilities = parse_audio_cpp_loaders_json(payload)
    if not capabilities.get("families"):
        raise CanaryFailure(
            "contract_drift",
            "loader JSON advertised no families; Studio cannot derive a version contract",
        )
    return payload, capabilities


def _advertised_endpoint_rows(capabilities: dict) -> List[dict]:
    rows: List[dict] = []
    endpoints = capabilities.get("family_endpoints")
    if not isinstance(endpoints, dict):
        return rows
    for family in sorted(endpoints):
        values = endpoints.get(family) or []
        if isinstance(values, str):
            values = [values]
        for endpoint in values:
            name = str(endpoint or "").strip()
            if name:
                rows.append({"family": str(family), "endpoint": name})
    return rows


def audit_contract(capabilities: dict, *, allow_drift: bool = False) -> dict:
    """Compose the contract and return its audit checks plus routing facts."""
    descriptor = compose_version_contract(capabilities)
    checks: List[Check] = []

    schema = descriptor.get("schema")
    operations = descriptor.get("operations") or []
    if schema != CONTRACT_SCHEMA:
        checks.append(
            Check(
                "contract_composition",
                "failed",
                f"composed contract schema is {schema!r}, expected {CONTRACT_SCHEMA!r}",
                failure="contract_drift",
                details={"schema": schema},
            )
        )
    elif not operations:
        checks.append(
            Check(
                "contract_composition",
                "failed",
                "composed contract exposes no operations",
                failure="contract_drift",
            )
        )
    else:
        checks.append(
            Check(
                "contract_composition",
                "passed",
                f"composed {CONTRACT_SCHEMA} with {len(operations)} operations",
                details={"schema": schema, "operation_count": len(operations)},
            )
        )

    operation_ids = [str(op.get("id") or "") for op in operations]
    missing_operations = [op for op in REQUIRED_OPERATIONS if op not in operation_ids]
    if missing_operations:
        checks.append(
            Check(
                "required_operations",
                "failed",
                "composed contract is missing required operations: "
                + ", ".join(missing_operations),
                failure="contract_drift",
                details={"required": list(REQUIRED_OPERATIONS), "missing": missing_operations},
            )
        )
    else:
        checks.append(
            Check(
                "required_operations",
                "passed",
                "required core operations present: " + ", ".join(REQUIRED_OPERATIONS),
                details={"required": list(REQUIRED_OPERATIONS)},
            )
        )

    endpoint_rows = _advertised_endpoint_rows(capabilities)
    advertised: List[str] = []
    unresolved: List[dict] = []
    for row in endpoint_rows:
        endpoint = row["endpoint"]
        if endpoint not in advertised:
            advertised.append(endpoint)
        if endpoint in GENERIC_TASK_ROUTES:
            continue
        if operation_for_native_path(endpoint) or operation_for_studio_path(endpoint):
            continue
        unresolved.append(row)
    if unresolved:
        names = ", ".join(f"{row['endpoint']} ({row['family']})" for row in unresolved)
        checks.append(
            Check(
                "advertised_endpoints",
                "failed",
                f"{len(unresolved)} advertised api_endpoints value(s) do not resolve to a "
                f"Studio operation or the generic {TASKS_RUN_ALIAS} route: {names}",
                failure="contract_drift",
                details={
                    "advertised": advertised,
                    "unresolved": unresolved,
                    "generic_routes": sorted(GENERIC_TASK_ROUTES),
                },
            )
        )
    else:
        checks.append(
            Check(
                "advertised_endpoints",
                "passed",
                f"all {len(advertised)} advertised api_endpoints value(s) resolve",
                details={"advertised": advertised, "unresolved": []},
            )
        )

    drift_warnings = [str(item) for item in descriptor.get("drift_warnings") or []]
    unexpected = [item for item in drift_warnings if item not in KNOWN_DRIFT_WARNINGS]
    drift_details = {
        "warnings": drift_warnings,
        "unexpected": unexpected,
        "allow_drift": bool(allow_drift),
    }
    if not drift_warnings:
        checks.append(
            Check("drift_warnings", "passed", "no contract drift warnings", details=drift_details)
        )
    elif allow_drift:
        checks.append(
            Check(
                "drift_warnings",
                "warning",
                f"{len(drift_warnings)} contract drift warning(s) reported; --allow-drift set: "
                + " | ".join(drift_warnings),
                details=drift_details,
            )
        )
    elif unexpected:
        checks.append(
            Check(
                "drift_warnings",
                "failed",
                f"{len(unexpected)} unexpected contract drift warning(s): "
                + " | ".join(unexpected),
                failure="contract_drift",
                details=drift_details,
            )
        )
    else:
        checks.append(
            Check(
                "drift_warnings",
                "passed",
                f"{len(drift_warnings)} reviewed drift warning(s) reported",
                details=drift_details,
            )
        )

    counts = {
        "families": len(capabilities.get("families") or []),
        "tasks": len(capabilities.get("tasks") or []),
        "endpoints": len(advertised),
        "operations": len(operations),
    }
    return {
        "descriptor": descriptor,
        "counts": counts,
        "operations": {
            "required": list(REQUIRED_OPERATIONS),
            "missing": missing_operations,
            "available": operation_ids,
        },
        "endpoints": {"advertised": advertised, "unresolved": unresolved},
        "drift": {
            "warnings": drift_warnings,
            "unexpected": unexpected,
            "allow_drift": bool(allow_drift),
        },
        "checks": checks,
    }


def find_model_manager_script(root: str) -> str:
    direct = os.path.join(root, "tools", "model_manager_v2.py")
    if os.path.isfile(direct):
        return direct
    for dirpath, dirnames, filenames in os.walk(root):
        relative = os.path.relpath(dirpath, root)
        depth = 0 if relative == "." else relative.count(os.sep) + 1
        if depth >= 4:
            dirnames[:] = []
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "community_models", "model_specs", "models"}
        ]
        if "model_manager_v2.py" in filenames:
            return os.path.join(dirpath, "model_manager_v2.py")
    return ""


def probe_model_manager(root: str, *, timeout: float) -> Tuple[Check, dict]:
    """Run the shipped ``model_manager_v2.py list --json`` catalog contract."""
    script = find_model_manager_script(root)
    info = {"script": None, "packages": None, "identity_fields": None}
    if not script:
        return (
            Check(
                "model_manager",
                "warning",
                "tools/model_manager_v2.py is not shipped in this release asset "
                "(source-only layout); catalog contract not exercised",
            ),
            info,
        )
    info["script"] = os.path.relpath(script, root)
    argv = [sys.executable, script, "list", "--json"]
    try:
        process = subprocess.run(
            argv,
            cwd=os.path.dirname(script),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            Check(
                "model_manager",
                "failed",
                f"model_manager_v2.py list --json timed out after {timeout}s",
                failure="probe_error",
                details=dict(info),
            ),
            info,
        )
    except OSError as exc:
        return (
            Check(
                "model_manager",
                "failed",
                f"could not start model_manager_v2.py list --json: {exc}",
                failure="probe_error",
                details=dict(info),
            ),
            info,
        )
    if process.returncode != 0:
        return (
            Check(
                "model_manager",
                "failed",
                f"model_manager_v2.py list --json exited {process.returncode}: "
                f"{_stderr_tail(process.stderr)}",
                failure="probe_error",
                details={**info, "returncode": process.returncode},
            ),
            info,
        )
    try:
        payload = json.loads(process.stdout or "")
    except json.JSONDecodeError:
        return (
            Check(
                "model_manager",
                "failed",
                "model_manager_v2.py list --json returned non-JSON output",
                failure="probe_error",
                details={**info, "stdout_tail": str(process.stdout or "")[-MAX_STDERR_TAIL:]},
            ),
            info,
        )
    if not isinstance(payload, list):
        return (
            Check(
                "model_manager",
                "failed",
                "model_manager_v2.py list --json returned a JSON object, expected a package list",
                failure="contract_drift",
                details={**info, "payload_type": type(payload).__name__},
            ),
            info,
        )
    rows = [row for row in payload if isinstance(row, dict)]
    info["packages"] = len(rows)
    if not rows:
        return (
            Check(
                "model_manager",
                "failed",
                "model_manager_v2.py list --json returned no package rows",
                failure="contract_drift",
                details=dict(info),
            ),
            info,
        )
    sample = rows[0]
    identity = all(key in sample for key in ("family", "id", "target_directory", "repo"))
    info["identity_fields"] = identity
    if not identity:
        return (
            Check(
                "model_manager",
                "warning",
                f"model_manager_v2.py list --json returned {len(rows)} package row(s) but the "
                "first row lacks Studio identity fields (family/id/target_directory/repo)",
                details=dict(info),
            ),
            info,
        )
    return (
        Check(
            "model_manager",
            "passed",
            f"model_manager_v2.py list --json returned {len(rows)} package row(s)",
            details=dict(info),
        ),
        info,
    )


def _record(checks: List[Check], check: Check) -> Check:
    checks.append(check)
    label = {"passed": "PASS", "warning": "WARN", "failed": "FAIL"}.get(
        check.status, check.status.upper()
    )
    print(f"[{label}] {check.id}: {check.message}", flush=True)
    return check


def _failure_check(check_id: str, failure: CanaryFailure) -> Check:
    details = dict(failure.details)
    details["failure_kind"] = failure.kind
    return Check(check_id, "failed", failure.message, failure=failure.kind, details=details)


def _release_report(release: dict) -> dict:
    return {
        "tag": release.get("tag"),
        "name": release.get("name"),
        "source": release.get("source"),
        "html_url": release.get("html_url"),
        "published_at": release.get("published_at"),
        "prerelease": release.get("prerelease"),
        "asset_count": len(release.get("assets") or []),
    }


def _build_report(
    *,
    checks: List[Check],
    release_info: dict,
    asset_info: dict,
    counts: dict,
    audit: dict,
    model_manager: dict,
    host_glibc: str,
    started: float,
    tmp_dir: str,
    keep: bool,
) -> dict:
    failed = [check for check in checks if check.status == "failed"]
    seen_kinds = {str(check.failure) for check in failed if check.failure}
    failure_kinds = [kind for kind in FAILURE_KINDS if kind in seen_kinds]
    passed = not failed
    duration = round(time.monotonic() - started, 2)
    count_text = (
        f"families={counts.get('families', 0)} tasks={counts.get('tasks', 0)} "
        f"endpoints={counts.get('endpoints', 0)} operations={counts.get('operations', 0)}"
    )
    tag = release_info.get("tag") or "unknown"
    if passed:
        summary = f"{tag}: PASS ({count_text})"
    else:
        summary = f"{tag}: FAIL ({', '.join(failure_kinds)}) ({count_text})"
    return {
        "schema": CANARY_REPORT_SCHEMA,
        "generated_at": _utc_now(),
        "passed": passed,
        "failure_kinds": failure_kinds,
        "release": release_info,
        "asset": asset_info,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "glibc": host_glibc or None,
        },
        "counts": counts,
        "endpoints": audit.get("endpoints", {"advertised": [], "unresolved": []}),
        "required_operations": audit.get(
            "operations", {"required": list(REQUIRED_OPERATIONS), "missing": []}
        ),
        "drift": audit.get("drift", {"warnings": [], "unexpected": [], "allow_drift": False}),
        "model_manager": model_manager,
        "checks": [check.as_dict() for check in checks],
        "duration_seconds": duration,
        "tmp_dir": tmp_dir if keep else None,
        "summary": summary,
    }


def _write_report(path: str, report: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run_canary(options: CanaryOptions, *, release: Optional[dict] = None) -> dict:
    """Run the canary and return its report (also written when ``report`` is set).

    ``release`` is an injection point for tests; production callers leave it
    unset and the latest GitHub release is resolved over the network.
    """
    started = time.monotonic()
    host_glibc = host_glibc_version()
    checks: List[Check] = []
    counts: dict = {"families": 0, "tasks": 0, "endpoints": 0, "operations": 0}
    audit: dict = {}
    release_info: dict = {}
    asset_info: dict = {}
    model_manager_info: dict = {"script": None, "packages": None, "identity_fields": None}
    tmp_dir = tempfile.mkdtemp(prefix="audio-cpp-canary-")

    def finish() -> dict:
        report = _build_report(
            checks=checks,
            release_info=release_info,
            asset_info=asset_info,
            counts=counts,
            audit=audit,
            model_manager=model_manager_info,
            host_glibc=host_glibc,
            started=started,
            tmp_dir=tmp_dir,
            keep=options.keep,
        )
        if options.report:
            _write_report(options.report, report)
            print(f"report: {options.report}", flush=True)
        print(f"result: {report['summary']}", flush=True)
        return report

    try:
        try:
            if release is None:
                release = resolve_release(options.tag, options.http_timeout)
        except CanaryFailure as exc:
            _record(checks, _failure_check("release", exc))
            return finish()
        release_info = _release_report(release)
        _record(
            checks,
            Check(
                "release",
                "passed",
                f"release {release_info['tag']} resolved via {release_info['source']} "
                f"({release_info['asset_count']} assets)",
                details=release_info,
            ),
        )

        try:
            asset, selection = select_cpu_asset(release, options.asset)
        except CanaryFailure as exc:
            _record(checks, _failure_check("asset_selection", exc))
            return finish()
        asset_info = {
            "name": asset.get("name"),
            "url": asset.get("browser_download_url"),
            "selection": selection,
            "published_size": asset.get("size"),
            "published_digest": asset.get("digest"),
            "sha256": None,
            "size": None,
        }
        _record(
            checks,
            Check(
                "asset_selection",
                "passed",
                f"{asset_info['name']} ({selection})",
                details={"name": asset_info["name"], "selection": selection},
            ),
        )

        archive_path = os.path.join(tmp_dir, str(asset.get("name") or "audio.cpp.tar.gz"))
        try:
            download_asset(
                str(asset.get("browser_download_url") or ""),
                archive_path,
                timeout=options.http_timeout,
            )
            size = verify_asset_size(archive_path, asset)
            asset_info["size"] = size
            _record(
                checks,
                Check(
                    "download",
                    "passed",
                    f"downloaded {size} bytes to the temporary directory",
                    details={"size": size},
                ),
            )
        except CanaryFailure as exc:
            _record(checks, _failure_check("download", exc))
            return finish()

        try:
            actual_sha256, digest_warning = verify_asset_digest(archive_path, asset)
            asset_info["sha256"] = actual_sha256
            if digest_warning:
                _record(
                    checks,
                    Check(
                        "asset_digest",
                        "warning",
                        digest_warning,
                        details={"sha256": actual_sha256},
                    ),
                )
            else:
                _record(
                    checks,
                    Check(
                        "asset_digest",
                        "passed",
                        f"sha256 {actual_sha256} matches the published digest",
                        details={"sha256": actual_sha256},
                    ),
                )
        except CanaryFailure as exc:
            _record(checks, _failure_check("asset_digest", exc))
            return finish()

        extract_dir = os.path.join(tmp_dir, "extract")
        try:
            binaries = extract_release_archive(archive_path, extract_dir)
        except CanaryFailure as exc:
            _record(checks, _failure_check("archive_extraction", exc))
            return finish()
        cli_path = str(binaries.get("cli_binary_path") or "")
        _record(
            checks,
            Check(
                "archive_extraction",
                "passed",
                f"extracted archive; audiocpp_cli at {os.path.relpath(cli_path, extract_dir)}",
                details={
                    "cli": os.path.relpath(cli_path, extract_dir),
                    "server": os.path.relpath(
                        str(binaries.get("server_binary_path") or ""), extract_dir
                    ),
                },
            ),
        )

        model_manager_check, model_manager_info = probe_model_manager(
            extract_dir, timeout=options.probe_timeout
        )
        _record(checks, model_manager_check)

        probe = run_cli_probe(cli_path, cwd=extract_dir, timeout=options.probe_timeout)
        probe_failure = classify_probe_failure(probe, cli_path=cli_path, host_glibc=host_glibc)
        if probe_failure is not None:
            _record(checks, _failure_check("cli_probe", probe_failure))
            return finish()
        _record(
            checks,
            Check(
                "cli_probe",
                "passed",
                "audiocpp_cli --list-loaders --json exited 0",
                details={"returncode": probe.returncode},
            ),
        )

        payload = try_parse_json_payload(probe.stdout)
        if payload is None:
            _record(
                checks,
                _failure_check(
                    "loader_json",
                    CanaryFailure(
                        "probe_error",
                        "audiocpp_cli --list-loaders --json did not print parseable JSON",
                        details={"stdout_tail": str(probe.stdout or "")[-MAX_STDERR_TAIL:]},
                    ),
                ),
            )
            return finish()
        try:
            _, capabilities = parse_loaders_payload(payload)
        except CanaryFailure as exc:
            _record(checks, _failure_check("loader_json", exc))
            return finish()

        counts = {
            "families": len(capabilities.get("families") or []),
            "tasks": len(capabilities.get("tasks") or []),
            "endpoints": len({row["endpoint"] for row in _advertised_endpoint_rows(capabilities)}),
            "operations": 0,
        }
        _record(
            checks,
            Check(
                "loader_json",
                "passed",
                f"schema_version=1 families={counts['families']} tasks={counts['tasks']} "
                f"endpoints={counts['endpoints']}",
                details={
                    "schema_version": payload.get("schema_version"),
                    "families": counts["families"],
                    "tasks": counts["tasks"],
                    "endpoints": counts["endpoints"],
                },
            ),
        )

        audit = audit_contract(capabilities, allow_drift=options.allow_drift)
        counts = audit["counts"]
        for check in audit["checks"]:
            _record(checks, check)
        return finish()
    finally:
        if options.keep:
            print(f"kept temporary directory: {tmp_dir}", flush=True)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_args(argv: Optional[List[str]] = None) -> CanaryOptions:
    parser = argparse.ArgumentParser(
        description="Canary the newest audio.cpp release against Studio's integration contract.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default="",
        help="GitHub release tag to test (default: the latest release)",
    )
    parser.add_argument(
        "--asset",
        default="",
        help="exact release asset name to use instead of the Linux x64 CPU prebuilt",
    )
    parser.add_argument(
        "--report",
        default="",
        help="write the JSON report to this path",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="report contract drift warnings as warnings instead of failing (unresolved "
        "api_endpoints still fail)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary download/extraction directory for debugging",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for release/download requests (default: 20)",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=60.0,
        help="timeout in seconds for CLI and model-manager probes (default: 60)",
    )
    args = parser.parse_args(argv)
    return CanaryOptions(
        tag=str(args.tag or "").strip(),
        asset=str(args.asset or "").strip(),
        report=str(args.report or "").strip(),
        allow_drift=bool(args.allow_drift),
        keep=bool(args.keep),
        http_timeout=max(1.0, float(args.http_timeout)),
        probe_timeout=max(1.0, float(args.probe_timeout)),
    )


def main(argv: Optional[List[str]] = None) -> int:
    options = parse_args(argv)
    try:
        report = run_canary(options)
    except CanaryFailure as exc:
        print(f"FAIL {exc.kind}: {exc.message}", file=sys.stderr)
        return 1
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
