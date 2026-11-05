"""Explicitly typed engine-surface response contracts (refactoring plan Phase 3 gate).

The plan requires response contracts to be "unified or explicitly typed per
surface" BEFORE any generic ``/api/engines/{engine_id}`` router delegates to
them (docs/config-driven-engine-refactoring-plan.md, Phase 3). This module is
the typing half of that gate: every divergence the survey found is modeled
here EXACTLY as the current producers emit it — no field added, renamed, or
made required that today's routes do not already guarantee. Nothing is wired
into the routes yet; delegation tranches import these as their response_model
contracts and pin that the live payloads still validate.

Companion survey: docs/engine-surface-contract-matrix.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _OpenModel(BaseModel):
    """Contract mirror: unknown keys pass through so a producer that grows
    a field never breaks a consumer typed against this module."""

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------
# check-updates — five distinct contracts, kept distinct (per-surface typing)
# --------------------------------------------------------------------------


class PypiCheckUpdates(_OpenModel):
    """Factory ``_check_updates`` pypi branch (lmdeploy/vllm/sglang/...)."""

    latest_version: Optional[str] = None
    current_version: Optional[str] = None
    update_available: bool = False
    releases: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class CommitCheckUpdates(_OpenModel):
    """Factory github_commit branch (sglang_v100-style fork tracking)."""

    latest_version: Optional[str] = None
    latest_commit: Optional[str] = None
    current_version: Optional[str] = None
    update_available: bool = False
    url: Optional[str] = None


class ReleasesCheckUpdates(_OpenModel):
    """Factory github_releases branch — note the ABSENCE of
    current_version/update_available: the factory's own contract already
    varies per update-source kind, so a single "factory releases
    dictionary" never existed to unify with."""

    latest_version: Optional[str] = None
    releases: List[str] = Field(default_factory=list)


class UnslothCheckUpdates(_OpenModel):
    """GET /unsloth-llama/check-updates — url and release_url are always the
    same value (pinned by test_engine_contracts)."""

    latest_version: Optional[str] = None
    current_version: Optional[str] = None
    update_available: bool = False
    url: Optional[str] = None
    release_url: Optional[str] = None


# llama.cpp's declared model lives with the route (schemas.tasks); re-exported
# here so the contract module is the single import site for delegators.
from backend.schemas.tasks import UpdateCheckResponse  # noqa: E402  re-export


class AudioCheckUpdates(_OpenModel):
    """GET /api/audio-cpp/check-updates — drift-aware, channel-aware."""

    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    update_available: bool = False
    latest_commit: Dict[str, Any] = Field(default_factory=dict)
    latest_release: Optional[Dict[str, Any]] = None
    update_channel: str = "branch"
    tracking_ref: Optional[str] = None
    repository_url: Optional[str] = None
    contract_fingerprint: Optional[str] = None


# --------------------------------------------------------------------------
# status — venv engines share an installer contract; native families diverge
# --------------------------------------------------------------------------


class VenvEngineStatus(_OpenModel):
    """Factory GET {prefix}/status: the installer's status dict; the route's
    error fallback emits exactly these eleven keys (plus any engine's
    status_fallback overlay), so every field here is Optional."""

    engine: Optional[str] = None
    installed: bool = False
    version: Optional[str] = None
    binary_path: Optional[str] = None
    venv_path: Optional[str] = None
    operation: Optional[str] = None
    operation_started_at: Optional[float] = None
    progress_task_id: Optional[str] = None
    last_error: Optional[str] = None
    install_type: Optional[str] = None
    source_repo: Optional[str] = None
    source_branch: Optional[str] = None


class CapabilityDelta(_OpenModel):
    """audio.cpp capability diff; producers always emit all six lists."""

    added_families: List[str] = Field(default_factory=list)
    removed_families: List[str] = Field(default_factory=list)
    added_tasks: List[str] = Field(default_factory=list)
    removed_tasks: List[str] = Field(default_factory=list)
    families_without_tasks: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AudioEngineStatus(_OpenModel):
    """GET /api/audio-cpp/status — the drift-aware status the plan names."""

    installed: bool = False
    active: Optional[Dict[str, Any]] = None
    runnable: bool = False
    models_root: Optional[str] = None
    model_manager_ready: bool = False
    tracking_ref: Optional[str] = None
    repository_url: Optional[str] = None
    contract_fingerprint: Optional[str] = None
    contract_changed: bool = False
    previous_contract_fingerprint: Optional[str] = None
    capability_delta: CapabilityDelta = Field(default_factory=CapabilityDelta)
    families: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    discovery_source: Optional[str] = None
    catalog_source: Optional[str] = None
    contract_schema: Optional[str] = None
    drift_warnings: List[str] = Field(default_factory=list)
    affected_models: List[Any] = Field(default_factory=list)
    supported_build_backends: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# audio.cpp typed extension surfaces (never folded into a generic router)
# --------------------------------------------------------------------------


class AudioContractResponse(_OpenModel):
    """GET /api/audio-cpp/contract — the composed integration descriptor
    (schema/operations/provenance/defaults_keys/task_routing/families/
    family_routing/drift_warnings/upstream_contract/spec_source/
    descriptor_fingerprint) plus the three envelope fields the route adds.
    Descriptor internals stay Dict-typed: they are engine data, and the plan
    keeps this surface an extension route."""

    engine_version: str = ""
    contract_fingerprint: Optional[str] = None
    contract_changed: bool = False
    scanned_at: Optional[Any] = None
    schema: Optional[str] = None
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    task_routing: Dict[str, str] = Field(default_factory=dict)
    generic_studio_path: Optional[str] = None
    families: List[str] = Field(default_factory=list)
    family_routing: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    drift_warnings: List[str] = Field(default_factory=list)


class CommunityVoiceItem(_OpenModel):
    id: str = ""
    filename: str = ""
    label: str = ""
    reference_text: str = ""
    size_bytes: int = 0
    installed: bool = False
    path: str = ""
    voice_id: str = ""
    storage: str = "community"
    modified_at: Optional[str] = None


class CommunityVoicesStatus(_OpenModel):
    """GET /api/audio-cpp/community-voices."""

    source: Dict[str, Any] = Field(default_factory=dict)
    installed: bool = False
    voice_dir: Optional[str] = None
    items: List[CommunityVoiceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# install scheduling — the one genuinely shared envelope across families
# --------------------------------------------------------------------------


class InstallAccepted(_OpenModel):
    """Every family's install/build/sync/retry POST that schedules a task
    returns this core (message, task_id, status="started") plus family
    extras (audio: version_name/source_ref/prebuilt...; venv installers:
    their own result dicts). The generic router may pin ONLY this core;
    extras ride as-is until their producers are surveyed individually."""

    message: str = ""
    task_id: str = ""
    status: str = "started"


class CancelAccepted(_OpenModel):
    """POST .../cancel family. audio.cpp answers with the BuildTaskManager
    envelope {ok, terminated?, message, task_id?}; the venv factory's cancel
    has its own installer dict. All-optional + extras is the honest pin of
    what every producer currently guarantees (nothing)."""

    ok: Optional[bool] = None
    terminated: Optional[bool] = None
    cancelled: Optional[bool] = None
    message: Optional[str] = None
    task_id: Optional[str] = None
