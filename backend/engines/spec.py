"""Immutable engine capability record shared by the registry and definitions.

Extracted from ``backend.engines.registry`` so definition validation can build
a spec without importing the registry (which now loads the authoritative
definitions itself). ``backend.engines.registry`` re-exports ``EngineSpec``;
external imports keep working unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    repository_source: str
    install_kind: str
    runtime_kind: str
    scanner_kind: str
    active_path_fields: Tuple[str, ...]
    artifact_formats: FrozenSet[str]
    package_kinds: FrozenSet[str]
    tasks: FrozenSet[str]
    input_modalities: FrozenSet[str]
    output_modalities: FrozenSet[str]
    supports_embeddings: bool = False
    experimental: bool = False
    # Per-surface maturity labels (e.g. speech_asr=stable). When set, ``experimental``
    # is only a coarse UI hint — prefer maturity_surfaces for honesty.
    maturity_surfaces: Tuple[Tuple[str, str], ...] = ()
    # Installer lifecycle facts (schema-v2 ``lifecycle``). Data only: it never
    # reaches the descriptor API (popped in ``to_api_dict``), so the frontend
    # payload is unchanged; the exporter writes it into the generated YAML so a
    # definition-driven installer can be built from the definition alone.
    # A plain dict keeps equality canonical (the loader returns a key-sorted
    # mapping with list-valued token fields).
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    # Phase 6 opt-in: a schema-v2 ``runtime_profiles.default`` payload that
    # lets an exported definition drive the engine's launch compiler. Also
    # excluded from the descriptor API.
    runtime_profile: Dict[str, Any] = field(default_factory=dict)
    # R5 launch policy (schema-v2 ``launch_policy``): CPU support, GPU-only
    # environment invariants and the runtime-context config key. Excluded from
    # the descriptor API; the launch compiler and capability context read it
    # from the spec instead of hardcoded engine-id sets.
    launch_policy: Dict[str, Any] = field(default_factory=dict)
    # R68 build-option catalog (schema-v2 ``build_options``): the static CMake
    # option table + UI categories a build-settings surface renders from.
    # Excluded from the descriptor API; the engine derives its option
    # dataclasses from here.
    build_options: Dict[str, Any] = field(default_factory=dict)
    # R69 protocol contract (schema-v2 ``protocol``): inspect probe flags and
    # the artifact path contract. Excluded from the descriptor API; the engine's
    # probe/artifact helpers read it from the spec.
    protocol: Dict[str, Any] = field(default_factory=dict)
    # R69 declared extension surfaces (schema-v2 ``extensions``): the typed
    # engine-owned endpoints the engine's router serves beyond the generic
    # lifecycle verbs. Excluded from the descriptor API.
    extensions: Tuple[Dict[str, Any], ...] = ()
    # Request-default vocabularies (schema-v2 ``request_defaults``): which keys a
    # task/transcription/speech defaults payload may carry and their types.
    # Excluded from the descriptor API; the normalizers read it from the spec.
    request_defaults: Dict[str, Any] = field(default_factory=dict)
    # Model-catalog surface (schema-v2 ``catalog``): which catalog providers
    # serve this engine. Excluded from the descriptor API; the catalog service
    # reads it instead of matching engine ids.
    catalog: Dict[str, Any] = field(default_factory=dict)
    # Upstream identity (schema-v2 ``upstream``): the canonical repository URL
    # fork detection compares against. Excluded from the descriptor API.
    upstream: Dict[str, Any] = field(default_factory=dict)
    # Install-settings defaults (schema-v2 ``settings_defaults``): the frozen
    # per-engine source_repo/source_branch/version defaults a venv engine's
    # settings surface starts from. Excluded from the descriptor API.
    settings_defaults: Dict[str, Any] = field(default_factory=dict)
    # Startup data directories (schema-v2 ``data_dirs``): engine-owned folders
    # created under the data root at startup. Excluded from the descriptor API.
    data_dirs: Tuple[str, ...] = ()
    # Model-compatibility declarations (schema-v2 ``model_compat``): which
    # artifact formats this engine is the DEFAULT for, and which formats it
    # stores as a prepared bundle. Excluded from the descriptor API.
    model_compat: Dict[str, Any] = field(default_factory=dict)
    # Per-engine build-settings defaults (schema-v2 ``build_defaults``):
    # layered on top of the catalog defaults (e.g. ik_llama needs
    # LLAMA_BUILD_EXAMPLES=ON). Excluded from the descriptor API.
    build_defaults: Dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Lifecycle is installer data, not a descriptor field: it is excluded
        # from the API payload (and therefore from YAML/spec equivalence).
        payload.pop("lifecycle", None)
        payload.pop("runtime_profile", None)
        payload.pop("launch_policy", None)
        payload.pop("build_options", None)
        payload.pop("protocol", None)
        payload.pop("extensions", None)
        payload.pop("request_defaults", None)
        payload.pop("catalog", None)
        payload.pop("upstream", None)
        payload.pop("settings_defaults", None)
        payload.pop("data_dirs", None)
        payload.pop("model_compat", None)
        payload.pop("build_defaults", None)
        for key in (
            "artifact_formats",
            "package_kinds",
            "tasks",
            "input_modalities",
            "output_modalities",
        ):
            payload[key] = sorted(payload[key])
        payload["active_path_fields"] = list(self.active_path_fields)
        payload["maturity_surfaces"] = {key: value for key, value in self.maturity_surfaces}
        # R77: the one launch-policy fact the frontend filters need is exposed
        # as an explicit descriptor field (the policy block itself stays out of
        # the payload). Default false: an engine that can run on CPU declares it.
        payload["cpu_supported"] = bool(dict(self.launch_policy).get("cpu_supported", False))
        # R78: GPU architectures the engine declares support for (empty = none
        # declared); exposed so the hardware filter needs no engine-id checks.
        payload["gpu_archs"] = sorted(
            str(item) for item in (dict(self.launch_policy).get("gpu_archs") or ())
        )
        return payload
