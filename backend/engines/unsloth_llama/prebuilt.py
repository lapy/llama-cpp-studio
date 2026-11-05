"""Select Unsloth llama.cpp mix prebuilts from the published release manifest.

The mix ships ``llama-prebuilt-manifest.json`` next to the archives. That file
is the GPU/runtime profile: runtime line, SM coverage, and portable vs targeted
bundles. This module reads that published JSON. It does not import Unsloth Studio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MANIFEST_ASSET_NAME = "llama-prebuilt-manifest.json"
CHECKSUM_ASSET_NAME = "llama-prebuilt-sha256.json"
MAX_RELEASE_WALKBACK = 8

_TAG_BUILD = re.compile(r"^b(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class HostTarget:
    cuda_major: Optional[int] = None
    sms: Tuple[int, ...] = ()
    has_nvidia: bool = False


@dataclass(frozen=True)
class PublishedArtifact:
    asset_name: str
    bundle_profile: str = ""
    runtime_line: str = ""
    coverage_class: str = ""
    supported_sms: Tuple[int, ...] = ()
    min_sm: Optional[int] = None
    max_sm: Optional[int] = None
    rank: int = 0
    install_kind: str = ""

    @property
    def is_cpu(self) -> bool:
        blob = " ".join(
            (
                self.install_kind,
                self.bundle_profile,
                self.runtime_line,
                self.asset_name,
            )
        ).lower()
        return "cpu" in blob and "cuda" not in self.runtime_line.lower()

    @property
    def is_linux_x64(self) -> bool:
        name = self.asset_name.lower()
        kind = self.install_kind.lower()
        if any(token in name for token in ("windows", "macos", "darwin", "arm64", "aarch64")):
            return False
        if "linux-x64" in name or "linux-amd64" in name:
            return True
        return kind.startswith("linux") and "arm" not in kind and "win" not in kind

    @property
    def is_cuda(self) -> bool:
        return self.runtime_line.startswith("cuda") or "cuda" in self.install_kind.lower()


@dataclass(frozen=True)
class ReleasePlan:
    tag: str
    asset_name: str
    flavor: str
    artifact: PublishedArtifact
    skipped_newer: Tuple[str, ...] = ()
    upstream_tag: str = ""
    reason: str = ""


def parse_sm(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        major = int(value)
        minor = int(round((value - major) * 10))
        return major * 10 + minor if major > 0 else None
    text = str(value).strip().lower().replace("sm_", "").replace("sm", "")
    if not text:
        return None
    if "." in text:
        major_s, _, minor_s = text.partition(".")
        if major_s.isdigit() and minor_s[:1].isdigit():
            return int(major_s) * 10 + int(minor_s[0])
        return None
    if text.isdigit():
        return int(text)
    return None


def parse_cuda_major(version: Optional[str]) -> Optional[int]:
    text = str(version or "").strip()
    if not text:
        return None
    first = text.split(".", 1)[0]
    if first.isdigit():
        return int(first)
    return None


def host_target_from_gpu_info(
    gpu_info: Optional[Dict[str, Any]],
    *,
    cuda_major: Optional[int] = None,
) -> HostTarget:
    info = gpu_info if isinstance(gpu_info, dict) else {}
    sms: List[int] = []
    for gpu in info.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        sm = parse_sm(gpu.get("compute_capability"))
        if sm:
            sms.append(sm)
    vendor = str(info.get("vendor") or "").lower()
    has_nvidia = vendor == "nvidia" and int(info.get("device_count") or 0) > 0
    if cuda_major is None:
        cuda_major = parse_cuda_major(info.get("cuda_version"))
    return HostTarget(
        cuda_major=cuda_major,
        sms=tuple(sorted(set(sms))),
        has_nvidia=has_nvidia or bool(sms),
    )


def parse_base_build(tag: Optional[str]) -> Optional[int]:
    text = str(tag or "").strip()
    match = _TAG_BUILD.match(text)
    return int(match.group(1)) if match else None


def is_behind(installed: Optional[str], latest: Optional[str]) -> bool:
    """True when ``latest`` is a real upgrade over ``installed``.

    Same tag is current. A higher ``bNNNN`` base is newer. At the same base a
    mix beats a bare ``bNNNN``, and a different mix is newer. A bare base never
    replaces a mix at that base.
    """
    left = str(installed or "").strip()
    right = str(latest or "").strip()
    if not left or not right or left == right:
        return False
    installed_base = parse_base_build(left)
    latest_base = parse_base_build(right)
    if installed_base is None or latest_base is None:
        return left != right
    if latest_base != installed_base:
        return latest_base > installed_base
    return right != f"b{latest_base}"


def is_published_release(release: Any) -> bool:
    if not isinstance(release, dict):
        return False
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        return False
    return not release.get("draft") and not release.get("prerelease")


def sort_releases_by_publish_time(releases: Iterable[Any]) -> List[Dict[str, Any]]:
    items = [item for item in releases if is_published_release(item)]
    items.sort(
        key=lambda item: str(item.get("published_at") or item.get("created_at") or ""),
        reverse=True,
    )
    return items


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.lstrip("-").isdigit() else None


def parse_artifact(raw: Any, *, fallback_name: str = "") -> Optional[PublishedArtifact]:
    if isinstance(raw, str) and fallback_name:
        return PublishedArtifact(asset_name=fallback_name)
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("asset_name") or raw.get("name") or fallback_name or "").strip()
    if not name:
        return None
    supported: List[int] = []
    for item in raw.get("supported_sms") or []:
        sm = parse_sm(item)
        if sm:
            supported.append(sm)
    return PublishedArtifact(
        asset_name=name,
        bundle_profile=str(raw.get("bundle_profile") or "").strip(),
        runtime_line=str(raw.get("runtime_line") or "").strip(),
        coverage_class=str(raw.get("coverage_class") or "").strip(),
        supported_sms=tuple(sorted(set(supported))),
        min_sm=_int_or_none(raw.get("min_sm")),
        max_sm=_int_or_none(raw.get("max_sm")),
        rank=_int_or_none(raw.get("rank")) or 0,
        install_kind=str(raw.get("install_kind") or raw.get("kind") or "").strip(),
    )


def parse_published_artifacts(manifest: Any) -> List[PublishedArtifact]:
    if not isinstance(manifest, dict):
        return []
    raw = manifest.get("artifacts")
    parsed: List[PublishedArtifact] = []
    if isinstance(raw, list):
        for item in raw:
            artifact = parse_artifact(item)
            if artifact:
                parsed.append(artifact)
    elif isinstance(raw, dict):
        for name, item in raw.items():
            artifact = parse_artifact(item, fallback_name=str(name))
            if artifact:
                parsed.append(artifact)
    return parsed


def manifest_upstream_tag(manifest: Any) -> str:
    if not isinstance(manifest, dict):
        return ""
    return str(manifest.get("upstream_tag") or "").strip()


def artifact_covers_sms(artifact: PublishedArtifact, sms: Sequence[int]) -> bool:
    if not sms:
        return artifact.coverage_class == "portable" or artifact.is_cpu
    if artifact.min_sm is None or artifact.max_sm is None:
        return False
    supported = set(artifact.supported_sms)
    for sm in sms:
        if sm < artifact.min_sm or sm > artifact.max_sm:
            return False
        if supported and sm not in supported:
            return False
    return True


def _runtime_lines(host: HostTarget) -> Tuple[str, ...]:
    if host.cuda_major == 13:
        return ("cuda13", "cuda12")
    if host.cuda_major == 12:
        return ("cuda12",)
    if host.has_nvidia:
        return ("cuda12", "cuda13")
    return ()


def _pick_for_runtime(
    artifacts: Sequence[PublishedArtifact],
    runtime_line: str,
    host: HostTarget,
) -> Optional[Tuple[PublishedArtifact, str]]:
    matching = [
        item
        for item in artifacts
        if item.is_linux_x64 and item.is_cuda and item.runtime_line == runtime_line
    ]
    if not matching:
        return None
    covering = [item for item in matching if artifact_covers_sms(item, host.sms)]
    targeted = [item for item in covering if item.coverage_class != "portable"]
    if targeted:
        chosen = min(
            targeted,
            key=lambda item: (
                (item.max_sm or 0) - (item.min_sm or 0),
                item.rank,
                item.max_sm or 0,
                item.asset_name,
            ),
        )
        return chosen, f"tightest {chosen.coverage_class} coverage for {runtime_line}"
    portable = [item for item in covering if item.coverage_class == "portable"]
    if portable:
        chosen = min(portable, key=lambda item: (item.rank, item.asset_name))
        reason = (
            f"portable {runtime_line} (compute capability unknown)"
            if not host.sms
            else f"portable fallback for {runtime_line}"
        )
        return chosen, reason
    return None


def _cpu_artifact(artifacts: Sequence[PublishedArtifact]) -> Optional[PublishedArtifact]:
    cpus = [item for item in artifacts if item.is_linux_x64 and item.is_cpu]
    if not cpus:
        return None
    return min(cpus, key=lambda item: (item.rank, item.asset_name))


def select_linux_x64_artifact(
    manifest: Any,
    host: HostTarget,
) -> Optional[Tuple[PublishedArtifact, str]]:
    artifacts = parse_published_artifacts(manifest)
    if host.has_nvidia or host.cuda_major in {12, 13}:
        for runtime in _runtime_lines(host):
            picked = _pick_for_runtime(artifacts, runtime, host)
            if picked:
                return picked
        return None
    cpu = _cpu_artifact(artifacts)
    if cpu:
        return cpu, "cpu bundle (no NVIDIA GPU)"
    return None


def select_asset_flavor(cuda_major: Optional[int]) -> str:
    if cuda_major == 13:
        return "cuda13-portable"
    if cuda_major == 12:
        return "cuda12-portable"
    return "cpu"


def linux_x64_asset_name(tag: str, flavor: str) -> str:
    return f"app-{str(tag).strip()}-linux-x64-{flavor}.tar.gz"


def find_named_asset(assets: Sequence[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    wanted = str(name or "").strip()
    for asset in assets:
        if str(asset.get("name") or "") == wanted:
            return asset
    return None


def select_release_asset(
    assets: List[Dict[str, Any]], tag: str, flavor: str
) -> Optional[Dict[str, Any]]:
    return find_named_asset(assets, linux_x64_asset_name(tag, flavor))


def plan_from_manifest(
    *,
    tag: str,
    assets: Sequence[Dict[str, Any]],
    manifest: Any,
    host: HostTarget,
    skipped_newer: Sequence[str] = (),
) -> Optional[ReleasePlan]:
    picked = select_linux_x64_artifact(manifest, host)
    if not picked:
        return None
    artifact, reason = picked
    if not find_named_asset(assets, artifact.asset_name):
        return None
    flavor = artifact.bundle_profile or artifact.runtime_line or "cpu"
    return ReleasePlan(
        tag=tag,
        asset_name=artifact.asset_name,
        flavor=flavor,
        artifact=artifact,
        skipped_newer=tuple(skipped_newer),
        upstream_tag=manifest_upstream_tag(manifest),
        reason=reason,
    )


def fallback_plan_from_filenames(
    *,
    tag: str,
    assets: Sequence[Dict[str, Any]],
    host: HostTarget,
    skipped_newer: Sequence[str] = (),
) -> Optional[ReleasePlan]:
    flavor = select_asset_flavor(host.cuda_major)
    asset = select_release_asset(list(assets), tag, flavor)
    if not asset:
        return None
    name = str(asset.get("name") or "")
    artifact = PublishedArtifact(
        asset_name=name,
        bundle_profile=flavor,
        runtime_line=flavor.split("-", 1)[0] if flavor.startswith("cuda") else "",
        coverage_class="portable" if "portable" in flavor else "",
        install_kind="linux-cpu" if flavor == "cpu" else "linux-cuda",
    )
    return ReleasePlan(
        tag=tag,
        asset_name=name,
        flavor=flavor,
        artifact=artifact,
        skipped_newer=tuple(skipped_newer),
        reason=f"filename fallback {flavor}",
    )


def resolve_release_plan(
    releases: Sequence[Dict[str, Any]],
    manifests: Dict[str, Any],
    host: HostTarget,
    *,
    pinned: bool = False,
    max_walkback: int = MAX_RELEASE_WALKBACK,
) -> ReleasePlan:
    skipped: List[str] = []
    last_error = "no published Unsloth llama.cpp releases were usable"
    limit = 1 if pinned else max(1, max_walkback)
    for release in sort_releases_by_publish_time(releases)[:limit]:
        tag = str(release.get("tag_name") or "").strip()
        assets = [item for item in (release.get("assets") or []) if isinstance(item, dict)]
        manifest = manifests.get(tag)
        plan = None
        if manifest is not None:
            plan = plan_from_manifest(
                tag=tag,
                assets=assets,
                manifest=manifest,
                host=host,
                skipped_newer=skipped,
            )
        if plan is None:
            plan = fallback_plan_from_filenames(
                tag=tag,
                assets=assets,
                host=host,
                skipped_newer=skipped,
            )
        if plan is not None:
            return plan
        last_error = (
            f"No Linux x64 bundle in {tag} covers this host "
            f"(CUDA {host.cuda_major or 'none'}, SM {','.join(str(sm) for sm in host.sms) or 'unknown'})"
        )
        if pinned:
            raise RuntimeError(last_error)
        skipped.append(tag)
    raise RuntimeError(last_error)
