"""Typed engine definitions — schema versions 1 and 2.

Part of the configuration-driven engine refactoring
(docs/config-driven-engine-refactoring-plan.md, Phases 2-3). Definitions are
loaded, validated, and template-expanded; descriptor payloads stay
shadow-compared against the Python ``ENGINE_REGISTRY`` until each engine's
cutover, while ``recipes`` are executed only by engines already cut over
(LMDeploy release — the authoritative list lives in the cutover allow-list
test). Schema v2 added the forward-looking sections exercised by the
reference definitions — ``settings_defaults``, ``model_sources``,
``discovery``, named ``runtime_profiles``, and typed install ``recipes`` over
the ``INSTALL_PRIMITIVES`` vocabulary — because the plan requires proving the
worst-case engine (sglang_v100) is expressible BEFORE Phase 3 builds the
executor that runs them. This loader rejects unknown fields, so future
phases extend the vocabulary deliberately instead of accreting free-form
keys.

This module also owns two shared vocabularies later phases consume:
``validate_condition`` (the typed condition algebra for recipes) and
``validate_asset_selectors`` (unambiguous release-asset selection), so their
contracts are defined and tested before their first executor.

Validation is hand-rolled (no pydantic) because startup rejection must be
precise: duplicate YAML keys, duplicate engine IDs, unknown fields/primitives,
missing references, inheritance cycles, invalid condition types, and ambiguous
artifact selectors each get their own error, carrying the source path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, FrozenSet, List, Mapping, Sequence, Tuple

import yaml

from backend.engines.spec import EngineSpec

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
NATIVE_HOST_KINDS = frozenset({"cmake_source", "audio_cpp", "release_assets"})

# Schema v2 adds the recipe/profile/settings sections validated by the
# reference-definition spike (docs/config-driven-engine-refactoring-plan.md
# risk: "worst-case engine must be expressible BEFORE Phase 3"). Loader
# validates them; executors arrive with Phases 3/6, so nothing at runtime
# consumes them yet. v1 documents must not carry these sections.
V2_ONLY_KEYS = (
    "settings_defaults",
    "model_sources",
    "discovery",
    "runtime_profiles",
    "recipes",
    "lifecycle",
    "launch_policy",
    "build_options",
    "protocol",
    "extensions",
    "request_defaults",
    "catalog",
    "upstream",
    "data_dirs",
    "build_defaults",
    "model_compat",
    # Native source-build policy: repository/artifact naming and build choices.
    "source_build",
    # Prebuilt-archive release policy (repository, manifest/checksum asset
    # names, release walk-back budget) consumed by a native release host.
    "release_assets",
    # Declared native host mechanism. Selection is by host name, not engine id.
    "native_host",
    # Presentation family consumed by the frontend overlay. Not engine logic.
    "ui_kind",
)

UI_KINDS = frozenset({"audio", "sglang", "native", "venv"})

# YAML campaign S3: what a generic (data-driven) lifecycle registration and
# the GenericVenvInstaller need beyond identity/capabilities. Purely optional
# — the nine generated definitions never carry it, and the generic surface
# answers verbs only for engines that do.
LIFECYCLE_KEYS = frozenset(
    {
        "url_prefix",
        "update_source",
        "prefer_source_install",
        "status_fallback",
        "version_setting_key",
        "binary_name",
        "root_name",
        "log_name",
        "state_name",
        "distribution_names",
        "default_repo_url",
        "default_branch",
        # Phase-3 typed installer policy (read by GenericVenvInstaller):
        # interpreter range, build-environment composition and the
        # version-forget failure policy.
        "python_min",
        "python_max",
        "build_env",
        "forget_errors",
        # Reusable binary-resolution override: the named environment variable
        # may point at the engine binary. The generic installer owns the
        # mechanism (path or PATH-name, exactly one implementation); a
        # definition only selects whether it applies.
        "binary_override_env",
        # Status projection: which binary the status verb reports, which row
        # fields and constants it carries, and whether engine/removed_at keys
        # are present. Replaces per-engine `status()` overrides.
        "status",
        # Start-response wording with a `{label}` placeholder.
        "source_started_message",
        # Streaming install-log behavior: heartbeat and progress milestones.
        "log",
        # Exact interpreter requirement (e.g. "3.12") with the generic guard.
        "python_exact",
        # Engines with no pip release build from source on install_release.
        "release_from_source",
        # Extra install-record metadata merged over the build-env metadata.
        "runtime_metadata",
        # Whether build-env metadata is recorded at all (some forks record none).
        "record_runtime_metadata",
        # Extra builder environment defaults (setdefault, not override).
        "build_env_vars",
        # CUDA toolkit fallback when the managed installer has no path.
        "cuda_home_fallback",
        # Compose/ensure a Rust toolchain for source builds of Python wheels.
        "build_rust",
        # Directory (relative to the version root) where source wheels land.
        "source_dist_dir",
        # Legacy task-id/manager token used by routes and the UI.
        "manager_name",
        # Install-row type for GitHub-release wheel installs.
        "release_install_type",
    }
)
UPDATE_SOURCE_KINDS = frozenset({"pypi", "github_commit", "github_releases"})
BUILD_ENVS = frozenset({"", "cuda"})
FORGET_ERROR_POLICIES = frozenset({"swallow", "raise"})
STATUS_BINARIES = frozenset({"binary_name", "venv_python"})
# Where a declared help probe runs. ``bin_dir`` is the executable's directory
# (the historical default for definition scanners); ``venv_root`` is the
# virtualenv root, which venv engines' legacy scanners used so relative
# resources resolve as they do at launch.
DISCOVERY_CWDS = frozenset({"bin_dir", "venv_root"})
# How a definition discovers its parameter catalog:
# * ``probes`` — the declarative probe contract below (argv/parser/env);
# * ``adapter`` — the declared ``scanner_kind`` keeps its shared Python probe
#   because the mechanism (native invocation/LD paths, audio inspect) is domain
#   code, not per-engine data. Explicit, so a new YAML engine never silently
#   inherits an unrelated adapter.
DISCOVERY_MODES = frozenset({"probes", "adapter"})
_LIFECYCLE_STR_KEYS = (
    "url_prefix",
    "version_setting_key",
    "binary_name",
    "root_name",
    "log_name",
    "state_name",
    "default_repo_url",
    "default_branch",
    "python_min",
    "python_max",
    "binary_override_env",
)


def _validate_lifecycle(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    _require(isinstance(value, Mapping), "lifecycle must be a mapping", where)
    unknown = set(value) - LIFECYCLE_KEYS
    _require(not unknown, f"lifecycle has unknown keys {sorted(unknown)}", where)
    update_source = value.get("update_source")
    if update_source is not None:
        _require(
            isinstance(update_source, Mapping), "lifecycle.update_source must be a mapping", where
        )
        unknown = set(update_source) - {
            "kind",
            "package",
            "repo",
            "branch",
            "timeout",
            "error_status",
        }
        _require(
            not unknown,
            f"lifecycle.update_source has unknown keys {sorted(unknown)}",
            where,
        )
        if "timeout" in update_source:
            timeout = update_source["timeout"]
            _require(
                isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0,
                "lifecycle.update_source.timeout must be a positive number",
                where,
            )
        if "error_status" in update_source:
            status = update_source["error_status"]
            _require(
                isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599,
                "lifecycle.update_source.error_status must be an HTTP error status",
                where,
            )
        kind = _check_kind(
            "lifecycle.update_source.kind",
            update_source.get("kind"),
            UPDATE_SOURCE_KINDS,
            where,
        )
        if kind == "pypi":
            _require(
                bool(str(update_source.get("package") or "").strip()),
                "lifecycle.update_source kind pypi requires package",
                where,
            )
        else:
            _require(
                bool(str(update_source.get("repo") or "").strip()),
                f"lifecycle.update_source kind {kind} requires repo",
                where,
            )
    if "prefer_source_install" in value:
        _check_bool("lifecycle.prefer_source_install", value["prefer_source_install"], where)
    if "status_fallback" in value:
        _require(
            isinstance(value["status_fallback"], Mapping),
            "lifecycle.status_fallback must be a mapping",
            where,
        )
    if "distribution_names" in value:
        _check_token_list("lifecycle.distribution_names", value["distribution_names"], where)
    for key in ("python_min", "python_max"):
        if key in value:
            _check_str(f"lifecycle.{key}", value[key], where)
    if "binary_override_env" in value:
        name = value["binary_override_env"]
        _require(
            isinstance(name, str) and bool(_ENV_NAME_RE.match(name)),
            f"lifecycle.binary_override_env must be an environment variable name, got {name!r}",
            where,
        )
    if "build_env" in value:
        _check_kind("lifecycle.build_env", value["build_env"] or "", BUILD_ENVS, where)
    if "forget_errors" in value:
        _check_kind("lifecycle.forget_errors", value["forget_errors"], FORGET_ERROR_POLICIES, where)
    if "python_exact" in value:
        _check_str("lifecycle.python_exact", value["python_exact"], where)
    if "release_from_source" in value:
        _check_bool("lifecycle.release_from_source", value["release_from_source"], where)
    if "runtime_metadata" in value:
        metadata = value["runtime_metadata"]
        _require(
            isinstance(metadata, Mapping), "lifecycle.runtime_metadata must be a mapping", where
        )
        for name, item in metadata.items():
            _require(
                isinstance(name, str) and bool(_TOKEN_RE.match(name)),
                f"lifecycle.runtime_metadata has invalid key {name!r}",
                where,
            )
            _check_scalar(f"lifecycle.runtime_metadata.{name}", item, where)
    if "record_runtime_metadata" in value:
        _check_bool("lifecycle.record_runtime_metadata", value["record_runtime_metadata"], where)
    if "build_env_vars" in value:
        variables = value["build_env_vars"]
        _require(isinstance(variables, Mapping), "lifecycle.build_env_vars must map", where)
        for name, item in variables.items():
            _require(
                isinstance(name, str) and bool(_ENV_NAME_RE.match(name)),
                f"lifecycle.build_env_vars has invalid name {name!r}",
                where,
            )
            _check_scalar(f"lifecycle.build_env_vars.{name}", item, where)
    if "cuda_home_fallback" in value:
        _check_str("lifecycle.cuda_home_fallback", value["cuda_home_fallback"], where)
    if "build_rust" in value:
        _check_bool("lifecycle.build_rust", value["build_rust"], where)
    for key in ("source_dist_dir", "manager_name", "release_install_type"):
        if key in value:
            _check_str(f"lifecycle.{key}", value[key], where)
    if "source_started_message" in value:
        message = _check_str(
            "lifecycle.source_started_message", value["source_started_message"], where
        )
        _require(
            "{" not in message.replace("{label}", ""),
            "lifecycle.source_started_message supports only the {label} placeholder",
            where,
        )
    if "status" in value:
        status = value["status"]
        _require(isinstance(status, Mapping), "lifecycle.status must be a mapping", where)
        unknown = set(status) - {
            "binary",
            "include_engine",
            "include_removed_at",
            "row_fields",
            "constants",
        }
        _require(not unknown, f"lifecycle.status has unknown keys {sorted(unknown)}", where)
        if "binary" in status:
            _check_kind("lifecycle.status.binary", status["binary"], STATUS_BINARIES, where)
        for key in ("include_engine", "include_removed_at"):
            if key in status:
                _check_bool(f"lifecycle.status.{key}", status[key], where)
        if "row_fields" in status:
            _check_token_list("lifecycle.status.row_fields", status["row_fields"], where)
        if "constants" in status:
            constants = status["constants"]
            _require(isinstance(constants, Mapping), "lifecycle.status.constants must map", where)
            for name, constant in constants.items():
                _require(
                    isinstance(name, str) and bool(_TOKEN_RE.match(name)),
                    f"lifecycle.status.constants has invalid key {name!r}",
                    where,
                )
                _check_scalar(f"lifecycle.status.constants.{name}", constant, where)
    if "log" in value:
        log = value["log"]
        _require(isinstance(log, Mapping), "lifecycle.log must be a mapping", where)
        unknown = set(log) - {"heartbeat_seconds", "progress_milestones"}
        _require(not unknown, f"lifecycle.log has unknown keys {sorted(unknown)}", where)
        if "heartbeat_seconds" in log:
            heartbeat = log["heartbeat_seconds"]
            _require(
                isinstance(heartbeat, (int, float))
                and not isinstance(heartbeat, bool)
                and heartbeat > 0,
                "lifecycle.log.heartbeat_seconds must be a positive number",
                where,
            )
        if "progress_milestones" in log:
            milestones = log["progress_milestones"]
            _require(
                isinstance(milestones, Sequence)
                and not isinstance(milestones, (str, bytes))
                and bool(milestones),
                "lifecycle.log.progress_milestones must be a non-empty list",
                where,
            )
            for milestone in milestones:
                _require(
                    isinstance(milestone, Mapping),
                    "lifecycle.log.progress_milestones entries must map",
                    where,
                )
                unknown = set(milestone) - {"contains", "progress", "message"}
                _require(
                    not unknown,
                    f"lifecycle.log milestone has unknown keys {sorted(unknown)}",
                    where,
                )
                contains = milestone.get("contains")
                _require(
                    isinstance(contains, str) and bool(contains),
                    "lifecycle.log milestone requires a non-empty 'contains'",
                    where,
                )
                progress = milestone.get("progress")
                _require(
                    isinstance(progress, (int, float))
                    and not isinstance(progress, bool)
                    and 0 <= progress <= 100,
                    "lifecycle.log milestone progress must be 0..100",
                    where,
                )
                message = milestone.get("message")
                _require(
                    isinstance(message, str) and bool(message),
                    "lifecycle.log milestone requires a message",
                    where,
                )
                _require(
                    "{" not in message.replace("{label}", ""),
                    "lifecycle.log milestone message supports only the {label} placeholder",
                    where,
                )
    for key in _LIFECYCLE_STR_KEYS:
        if key in value:
            _check_str(f"lifecycle.{key}", value[key], where)
    return tuple((key, value[key]) for key in sorted(value))


def _validate_launch_policy(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Launch compiler policy: CPU support, GPU archs, env invariants, ctx key."""
    _require(isinstance(value, Mapping), "launch_policy must be a mapping", where)
    unknown = set(value) - {
        "cpu_supported",
        "gpu_archs",
        "gpu_only_invariants",
        "runtime_context_key",
        "capcompat",
    }
    _require(not unknown, f"launch_policy has unknown keys {sorted(unknown)}", where)
    if "capcompat" in value:
        _check_bool("launch_policy.capcompat", value["capcompat"], where)
    if "cpu_supported" in value:
        _check_bool("launch_policy.cpu_supported", value["cpu_supported"], where)
    if "gpu_archs" in value:
        archs = value["gpu_archs"]
        _require(
            isinstance(archs, Sequence) and not isinstance(archs, (str, bytes)) and bool(archs),
            "launch_policy.gpu_archs must be a non-empty list",
            where,
        )
        for arch in archs:
            _require(
                isinstance(arch, str) and bool(_TOKEN_RE.match(arch)),
                f"launch_policy.gpu_archs contains invalid arch {arch!r}",
                where,
            )
    if "gpu_only_invariants" in value:
        invariants = value["gpu_only_invariants"]
        _require(
            isinstance(invariants, Sequence) and not isinstance(invariants, (str, bytes)),
            "launch_policy.gpu_only_invariants must be a list of environment names",
            where,
        )
        for name in invariants:
            _require(
                isinstance(name, str) and bool(_ENV_NAME_RE.match(name)),
                f"launch_policy.gpu_only_invariants contains invalid name {name!r}",
                where,
            )
    if "runtime_context_key" in value:
        _check_str("launch_policy.runtime_context_key", value["runtime_context_key"], where)
    normalized = dict(value)
    if "gpu_archs" in normalized:
        normalized["gpu_archs"] = tuple(normalized["gpu_archs"])
    if "gpu_only_invariants" in normalized:
        normalized["gpu_only_invariants"] = tuple(normalized["gpu_only_invariants"])
    return tuple((key, normalized[key]) for key in sorted(normalized))


_BUILD_OPTION_KINDS = frozenset({"bool", "str", "enum", "int"})
_BUILD_OPTION_KEYS = frozenset(
    {
        "key",
        "field",
        "kind",
        "default",
        "label",
        "desc",
        "category",
        "cmake",
        "requires",
        "enum_values",
        "special",
        # R94: which engines of a shared family expose this entry in the UI.
        "engines",
    }
)
_BUILD_CATEGORY_KEYS = frozenset({"id", "label", "collapsed", "requires", "engines"})
_BUILD_REQUIRES_PSEUDO = frozenset({"model_set_custom", "cuda_or_hip"})


def _validate_build_options(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Static build-option catalog: types, defaults, categories, references."""
    _require(isinstance(value, Mapping), "build_options must be a mapping", where)
    unknown = set(value) - {
        "build_types",
        "build_type_default",
        "primary_keys",
        "categories",
        "options",
    }
    _require(not unknown, f"build_options has unknown keys {sorted(unknown)}", where)

    build_types = value.get("build_types")
    _require(
        isinstance(build_types, Sequence)
        and not isinstance(build_types, (str, bytes))
        and bool(build_types),
        "build_options.build_types must be a non-empty list",
        where,
    )
    for item in build_types:
        _check_str("build_options.build_types item", item, where)
    build_type_default = value.get("build_type_default")
    _require(
        isinstance(build_type_default, str) and build_type_default in build_types,
        "build_options.build_type_default must be one of build_types",
        where,
    )

    primary_keys = value.get("primary_keys") or []
    _require(
        isinstance(primary_keys, Sequence) and not isinstance(primary_keys, (str, bytes)),
        "build_options.primary_keys must be a list",
        where,
    )
    for item in primary_keys:
        _check_str("build_options.primary_keys item", item, where)

    categories = value.get("categories")
    _require(
        isinstance(categories, Sequence)
        and not isinstance(categories, (str, bytes))
        and bool(categories),
        "build_options.categories must be a non-empty list",
        where,
    )
    category_ids: List[str] = []
    for index, category in enumerate(categories):
        cwhere = f"{where}.build_options.categories[{index}]"
        _require(isinstance(category, Mapping), "category must be a mapping", cwhere)
        unknown = set(category) - _BUILD_CATEGORY_KEYS
        _require(not unknown, f"category has unknown keys {sorted(unknown)}", cwhere)
        category_id = _check_str("id", category.get("id"), cwhere)
        _require(bool(_TOKEN_RE.match(category_id)), f"invalid category id {category_id!r}", cwhere)
        _check_str("label", category.get("label"), cwhere)
        if "collapsed" in category:
            _check_bool("collapsed", category["collapsed"], cwhere)
        if "requires" in category:
            _check_str("requires", category["requires"], cwhere)
        if "engines" in category:
            _check_token_list("engines", category["engines"], cwhere)
        category_ids.append(category_id)
    _require(
        len(set(category_ids)) == len(category_ids),
        "duplicate build-option category id",
        where,
    )

    options = value.get("options")
    _require(
        isinstance(options, Sequence) and not isinstance(options, (str, bytes)) and bool(options),
        "build_options.options must be a non-empty list",
        where,
    )
    seen_keys: List[str] = []
    seen_fields: List[str] = []
    for index, option in enumerate(options):
        owhere = f"{where}.build_options.options[{index}]"
        _require(isinstance(option, Mapping), "option must be a mapping", owhere)
        unknown = set(option) - _BUILD_OPTION_KEYS
        _require(not unknown, f"option has unknown keys {sorted(unknown)}", owhere)
        key = _check_str("key", option.get("key"), owhere)
        field = _check_str("field", option.get("field"), owhere)
        for token in (key, field):
            _require(bool(_TOKEN_RE.match(token)), f"invalid build-option token {token!r}", owhere)
        _require(key not in seen_keys, f"duplicate build-option key {key!r}", owhere)
        _require(field not in seen_fields, f"duplicate build-option field {field!r}", owhere)
        seen_keys.append(key)
        seen_fields.append(field)
        kind = _check_kind("kind", option.get("kind"), _BUILD_OPTION_KINDS, owhere)
        _check_str("label", option.get("label"), owhere)
        _check_str("desc", option.get("desc"), owhere)
        category = _check_str("category", option.get("category"), owhere)
        _require(category in category_ids, f"unknown build-option category {category!r}", owhere)
        default = option.get("default")
        if kind == "bool":
            _check_bool("default", default, owhere)
        elif kind == "int":
            _require(
                isinstance(default, int) and not isinstance(default, bool),
                "default must be an integer",
                owhere,
            )
            _require(default >= 0, "default must be non-negative", owhere)
        else:
            # String defaults may be empty (e.g. a free-text CMake-args field).
            _require(isinstance(default, str), "default must be a string", owhere)
        for key_name in ("cmake", "requires", "special"):
            if key_name in option:
                _check_str(key_name, option[key_name], owhere)
        if "engines" in option:
            _check_token_list("engines", option["engines"], owhere)
        if kind == "enum":
            enum_values = option.get("enum_values")
            _require(
                isinstance(enum_values, Sequence)
                and not isinstance(enum_values, (str, bytes))
                and bool(enum_values),
                "enum option requires a non-empty enum_values list",
                owhere,
            )
            for item in enum_values:
                _check_str("enum_values item", item, owhere)
            _require(default in enum_values, "default must be one of enum_values", owhere)
        elif "enum_values" in option:
            _require(False, "enum_values is only allowed for kind 'enum'", owhere)

    for index, category in enumerate(categories):
        requires = category.get("requires")
        if requires:
            _require(
                requires in seen_keys,
                f"category requires unknown build-option {requires!r}",
                f"{where}.build_options.categories[{index}]",
            )
    for index, option in enumerate(options):
        requires = option.get("requires")
        if requires:
            _require(
                requires in seen_keys or requires in _BUILD_REQUIRES_PSEUDO,
                f"option requires unknown build-option {requires!r}",
                f"{where}.build_options.options[{index}]",
            )
    return tuple((key, value[key]) for key in sorted(value))


_PROTOCOL_PROBE_KEYS = frozenset(
    {
        "model_flag",
        "family_flag",
        "spec_override_flag",
        "load_option_flag",
        "inspect_flag",
        "json_flag",
    }
)
_PROTOCOL_ARTIFACT_KEYS = frozenset(
    {"builtin_family", "root_gguf_name", "gguf_suffix", "builtin_id_pattern"}
)


def _validate_protocol(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Engine protocol constants: inspect probe flags + artifact path contract."""
    _require(isinstance(value, Mapping), "protocol must be a mapping", where)
    unknown = set(value) - {"probe", "artifact", "launch"}
    _require(not unknown, f"protocol has unknown keys {sorted(unknown)}", where)
    probe = value.get("probe")
    _require(isinstance(probe, Mapping), "protocol.probe must be a mapping", where)
    unknown = set(probe) - _PROTOCOL_PROBE_KEYS
    _require(not unknown, f"protocol.probe has unknown keys {sorted(unknown)}", where)
    for key, flag in probe.items():
        flag_text = _check_str(f"protocol.probe.{key}", flag, where)
        _require(flag_text.startswith("--"), f"protocol.probe.{key} must be a flag", where)
    artifact = value.get("artifact")
    _require(isinstance(artifact, Mapping), "protocol.artifact must be a mapping", where)
    unknown = set(artifact) - _PROTOCOL_ARTIFACT_KEYS
    _require(not unknown, f"protocol.artifact has unknown keys {sorted(unknown)}", where)
    for key, raw in artifact.items():
        _check_str(f"protocol.artifact.{key}", raw, where)
    suffix = artifact.get("gguf_suffix")
    if suffix is not None:
        _require(
            str(suffix).startswith("."),
            "protocol.artifact.gguf_suffix must start with '.'",
            where,
        )
    pattern = artifact.get("builtin_id_pattern")
    if pattern is not None:
        try:
            re.compile(str(pattern))
        except re.error as exc:  # pragma: no cover - defensive
            raise DefinitionError(
                f"{where}: protocol.artifact.builtin_id_pattern is not a regex: {exc}"
            ) from None
    launch = value.get("launch")
    if launch is not None:
        _require(isinstance(launch, Mapping), "protocol.launch must be a mapping", where)
        unknown = set(launch) - {"config_flag", "config_macro"}
        _require(not unknown, f"protocol.launch has unknown keys {sorted(unknown)}", where)
        config_flag = _check_str("protocol.launch.config_flag", launch.get("config_flag"), where)
        _require(
            config_flag.startswith("--"),
            "protocol.launch.config_flag must be a flag",
            where,
        )
        config_macro = _check_str("protocol.launch.config_macro", launch.get("config_macro"), where)
        _require(
            bool(_TOKEN_RE.match(config_macro)),
            f"invalid launch macro name {config_macro!r}",
            where,
        )
    return tuple((key, value[key]) for key in sorted(value))


def _validate_extensions(value: Any, where: str) -> Tuple[Mapping[str, Any], ...]:
    """Typed extension surfaces the engine's own router declares and serves."""
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "extensions must be a list",
        where,
    )
    seen_ids: List[str] = []
    seen_paths: List[str] = []
    out: List[Mapping[str, Any]] = []
    for index, extension in enumerate(value):
        ewhere = f"{where}.extensions[{index}]"
        _require(isinstance(extension, Mapping), "extension must be a mapping", ewhere)
        unknown = set(extension) - {"id", "path", "methods", "description"}
        _require(not unknown, f"extension has unknown keys {sorted(unknown)}", ewhere)
        extension_id = _check_str("id", extension.get("id"), ewhere)
        _require(
            bool(_TOKEN_RE.match(extension_id)), f"invalid extension id {extension_id!r}", ewhere
        )
        path = _check_str("path", extension.get("path"), ewhere)
        _require(path.startswith("/"), "extension path must start with '/'", ewhere)
        methods = extension.get("methods") or ["GET"]
        _require(
            isinstance(methods, Sequence)
            and not isinstance(methods, (str, bytes))
            and bool(methods),
            "extension methods must be a non-empty list",
            ewhere,
        )
        for method in methods:
            _check_str("methods item", method, ewhere)
            _require(
                str(method).isupper(), f"extension method {method!r} must be upper-case", ewhere
            )
        if "description" in extension:
            _check_str("description", extension["description"], ewhere)
        _require(extension_id not in seen_ids, f"duplicate extension id {extension_id!r}", ewhere)
        _require(path not in seen_paths, f"duplicate extension path {path!r}", ewhere)
        seen_ids.append(extension_id)
        seen_paths.append(path)
        out.append(dict(extension))
    return tuple(out)


def _validate_request_defaults(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Per-surface request-default vocabularies (allowed keys + their types)."""
    _require(isinstance(value, Mapping), "request_defaults must be a mapping", where)
    unknown = set(value) - set(_REQUEST_DEFAULTS_SECTIONS)
    _require(not unknown, f"request_defaults has unknown sections {sorted(unknown)}", where)
    for section in _REQUEST_DEFAULTS_SECTIONS:
        if section not in value:
            continue
        section_value = value[section]
        swhere = f"{where}.request_defaults.{section}"
        _require(isinstance(section_value, Mapping), f"{section} must be a mapping", swhere)
        unknown = set(section_value) - _REQUEST_DEFAULTS_KEYS
        _require(not unknown, f"{section} has unknown keys {sorted(unknown)}", swhere)
        all_keys: List[str] = []
        typed_keys: List[str] = []
        for key_name in sorted(section_value):
            raw = section_value[key_name]
            _require(
                isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)),
                f"{key_name} must be a list",
                swhere,
            )
            for item in raw:
                _check_str(f"{key_name} item", item, swhere)
            _require(
                len(set(raw)) == len(raw),
                f"{key_name} contains duplicates",
                swhere,
            )
            if key_name == "keys":
                all_keys.extend(raw)
                continue
            for item in raw:
                _require(
                    item not in typed_keys,
                    f"request-default key {item!r} appears in more than one type list",
                    swhere,
                )
                typed_keys.append(item)
    return tuple((key, value[key]) for key in sorted(value))


# Kind vocabularies. Tests assert these stay in sync with their dispatch
# sites (SCANNER_KIND_DISPATCH in engines/scan/scanner.py, the installer
# adapter map, and proxy runtime kinds) without this module importing them.
INSTALL_KINDS: FrozenSet[str] = frozenset({"cmake", "prebuilt", "python_venv"})
RUNTIME_KINDS: FrozenSet[str] = frozenset(
    {"llama_server", "lmdeploy", "onecat_vllm", "vllm", "sglang", "audio_cpp"}
)
SCANNER_KINDS: FrozenSet[str] = frozenset(
    {
        "llama_help",
        "lmdeploy_help",
        "vllm_help",
        "sglang_help",
        "sglang_diffusion_help",
        "audio_cpp",
    }
)

MATURITY_STATUSES: FrozenSet[str] = frozenset({"stable", "experimental", "specialized"})

# --- schema v2 vocabularies -------------------------------------------------

_REQUEST_DEFAULTS_SECTIONS = ("task", "transcription", "speech")
_REQUEST_DEFAULTS_KEYS = frozenset({"keys", "bool_keys", "int_keys", "float_keys"})

MODEL_SOURCE_KINDS: FrozenSet[str] = frozenset(
    {"hf_repo", "gguf_file", "prepared_bundle", "builtin"}
)
PROFILE_EXECUTABLE_KINDS: FrozenSet[str] = frozenset(
    {"venv_console_script", "venv_python", "native_artifact"}
)
MODEL_BINDING_MODES: FrozenSet[str] = frozenset({"positional", "flag"})
MODEL_BINDING_SOURCES: FrozenSet[str] = frozenset({"huggingface_id", "local_path"})
READYNESS_KINDS: FrozenSet[str] = frozenset({"health_endpoint", "log_line", "process_exit"})
PROFILE_KEYS: FrozenSet[str] = frozenset(
    {
        "description",
        "default",
        "executable",
        "argv_prefix",
        "model_binding",
        "port_binding",
        "env",
        "scan",
        "runtime_switch",
        "launch_family",
        "launch_args",
    }
)
# A profile may NAME the launch mechanism that composes it (family profile)
# instead of declaring an argv skeleton. The mechanism lives in one shared
# place and owns everything that is ENGINE-FAMILY behaviour — executable
# resolution, model-source composition, companions, library path, alias — so
# the profile only declares the Studio-owned port flag (and an optional
# prefix). A family profile may not ALSO declare the keys the mechanism
# composes: a declared-but-ignored key is worse than an absent one.
LAUNCH_FAMILIES: FrozenSet[str] = frozenset({"gguf_llama_server"})
FAMILY_PROFILE_KEYS: FrozenSet[str] = frozenset(
    {
        "description",
        "default",
        "launch_family",
        "argv_prefix",
        "port_binding",
        "launch_args",
        # The family composes executable/cwd/argv, but a managed cache tree is
        # still Studio data anchored to the install row. A family profile may
        # declare ONLY `env.caches`; cwd/cuda/required belong to the mechanism.
        "env",
    }
)
LAUNCH_ARG_KEYS: FrozenSet[str] = frozenset(
    {"model", "hf_repo", "alias", "mmproj", "draft", "spec_type"}
)
RECIPE_STEP_KEYS: FrozenSet[str] = frozenset(
    {
        "id",
        "uses",
        "run",
        "with",
        "when",
        "produces",
        "requires",
        "timeout_s",
        "accepted_exit_codes",
        "cleanup",
    }
)

# Install-plan primitives the Phase 3 executor must implement (the plan's
# "install primitives must be shared and typed, not arbitrary shell").
# Recipes may only reference these names; raw argv is expressible solely via
# a step's ``run`` for genuinely host-level one-offs, never inside engines.
INSTALL_PRIMITIVES: FrozenSet[str] = frozenset(
    {
        "host.inspect",
        "prereqs.resolve_version",
        "prereqs.resolve_release_assets",
        "artifact.resolve_candidates",
        "archive.download_extract",
        "vcs.clone",
        "vcs.sync",
        "vcs.checkout",
        "vcs.apply_patch",
        "vcs.capture_identity",
        "vcs.workspace_checkout",
        "vcs.seal_checkout",
        "vcs.capture_commit",
        "vcs.prepare_build",
        # Reusable fork mechanisms: versioned, applicability-checked file
        # transforms and a declared builder environment.
        "vcs.apply_transform",
        "build.prepare_fork",
        "venv.create",
        "env.compose",
        "env.compose_ccache",
        "exec.run_logged",
        "toolchain.ensure",
        "artifact.build_wheels",
        "artifact.locate",
        "artifact.remove_bundle",
        "artifact.read_checksum",
        "archive.download",
        "archive.verify_sha256",
        "archive.verify_checksums_file",
        "archive.extract_tar",
        "artifact.materialize_cli",
        "cgroup.select_concurrency",
        "pip.install_requirements",
        "pip.install_package",
        "pip.uninstall",
        "pip.build_wheel",
        "pip.install_wheels",
        "pip.capture_metadata",
        "fs.remove_tree",
        "fs.remove_file",
        "fs.assert_dir",
        "fs.ensure_dir",
        "fs.prune_files",
        "cmake.configure",
        "cmake.build",
        "smoke.test",
        "publish.install",
        "publish.record_update",
        "publish.forget",
        "publish.finalize",
    }
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

# Required and optional keys of an engine definition file. Optional keys exist
# so template expansion can supply any subset; a fully expanded definition
# still must provide every required key.
_IDENTITY_KEYS = ("schema_version", "id", "extends")
_CAPABILITY_KEYS = (
    "label",
    "repository_source",
    "install_kind",
    "runtime_kind",
    "scanner_kind",
    "active_path_fields",
    "artifact_formats",
    "package_kinds",
    "tasks",
    "input_modalities",
    "output_modalities",
    "supports_embeddings",
    "experimental",
    "maturity_surfaces",
)
ENGINE_DEFINITION_KEYS: FrozenSet[str] = frozenset(_IDENTITY_KEYS + _CAPABILITY_KEYS)

TEMPLATE_KEYS: FrozenSet[str] = frozenset(_CAPABILITY_KEYS)


class DefinitionError(ValueError):
    """A definition or template document failed schema validation."""


# ---------------------------------------------------------------------------
# YAML loading with duplicate-key rejection
# ---------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of last-wins."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    seen: List[Any] = []
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in seen:
                raise DefinitionError(f"duplicate key {key!r} in YAML mapping")
        except TypeError:  # unhashable key: let YAML surface it naturally
            continue
        seen.append(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml_strict(text: str, *, source: str) -> Any:
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except DefinitionError as e:
        raise DefinitionError(f"{source}: {e}") from None
    except yaml.YAMLError as e:
        raise DefinitionError(f"{source}: malformed YAML: {e}") from None


def load_yaml_file_strict(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return load_yaml_strict(fh.read(), source=os.fspath(path))


# ---------------------------------------------------------------------------
# Field validators. Each takes (name, value, where) and returns a normalized
# value or raises DefinitionError with the offending path in `where`.
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str, where: str) -> None:
    if not condition:
        raise DefinitionError(f"{where}: {message}")


def _check_str(name: str, value: Any, where: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{name} must be a non-empty string",
        where,
    )
    return value


def _check_kind(name: str, value: Any, vocab: FrozenSet[str], where: str) -> str:
    value = _check_str(name, value, where)
    _require(
        value in vocab,
        f"{name} {value!r} is not a known kind (known: {sorted(vocab)})",
        where,
    )
    return value


def _check_token_list(name: str, value: Any, where: str) -> Tuple[str, ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a list",
        where,
    )
    items = tuple(str(v) for v in value)
    _require(len(items) > 0, f"{name} must not be empty", where)
    for item in items:
        _require(bool(_TOKEN_RE.match(item)), f"{name} contains invalid token {item!r}", where)
    _require(len(set(items)) == len(items), f"{name} contains duplicates", where)
    return items


def _check_pattern_list(name: str, value: Any, where: str) -> Tuple[str, ...]:
    """Glob-style release-asset patterns: non-empty strings, no duplicates."""
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a list",
        where,
    )
    items = tuple(_check_str(name, item, where) for item in value)
    _require(len(items) > 0, f"{name} must not be empty", where)
    _require(len(set(items)) == len(items), f"{name} contains duplicates", where)
    return items


def _check_bool(name: str, value: Any, where: str) -> bool:
    _require(isinstance(value, bool), f"{name} must be a boolean", where)
    return value


def _check_maturity(value: Any, where: str) -> Tuple[Tuple[str, str], ...]:
    _require(isinstance(value, Mapping), "maturity_surfaces must be a mapping", where)
    pairs: List[Tuple[str, str]] = []
    for surface, status in value.items():
        _check_str("maturity surface", surface, where)
        _require(
            isinstance(status, str) and status in MATURITY_STATUSES,
            f"maturity surface {surface!r} has unknown status {status!r} "
            f"(known: {sorted(MATURITY_STATUSES)})",
            where,
        )
        pairs.append((surface, status))
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Typed condition vocabulary (used by Phase 3+ recipes; defined + tested now)
# ---------------------------------------------------------------------------

CONDITION_OPS: FrozenSet[str] = frozenset(
    {"all", "any", "equals", "version_range", "exists", "capability_contains"}
)


def validate_condition(node: Any, where: str) -> None:
    """Validate one node of the typed condition algebra.

    Shape (exactly one operator key per mapping)::

        {all: [<condition>, ...]} | {any: [<condition>, ...]}
        {equals: {field: <dotted.path>, value: <scalar>}}
        {version_range: {field: <dotted.path>, min: <version>, max: <version>}}   # either bound optional
        {exists: {field: <dotted.path>}}
        {capability_contains: {field: <dotted.path>, value: <token>}}

    Expressions reference request/context/step outputs by dotted field path;
    Python or Jinja evaluation is intentionally not expressible.
    """
    _require(isinstance(node, Mapping), "condition must be a mapping", where)
    _require(
        len(node) == 1, f"condition must have exactly one operator key, got {sorted(node)}", where
    )
    ((op, body),) = node.items()
    _require(op in CONDITION_OPS, f"unknown condition operator {op!r}", where)
    if op in ("all", "any"):
        _require(
            isinstance(body, Sequence) and not isinstance(body, (str, bytes)),
            f"{op} expects a list of conditions",
            where,
        )
        _require(len(body) > 0, f"{op} expects at least one condition", where)
        for index, child in enumerate(body):
            validate_condition(child, f"{where}.{op}[{index}]")
        return
    _require(isinstance(body, Mapping), f"{op} expects a mapping body", where)
    _require("field" in body, f"{op} requires 'field'", where)
    _check_str("field", body["field"], where)
    allowed = {
        "equals": {"field", "value"},
        "version_range": {"field", "min", "max"},
        "exists": {"field"},
        "capability_contains": {"field", "value"},
    }[op]
    unknown = set(body) - allowed
    _require(not unknown, f"{op} has unknown operands {sorted(unknown)}", where)
    if op in ("equals", "capability_contains"):
        _require("value" in body, f"{op} requires 'value'", where)
    if op == "version_range":
        for bound in ("min", "max"):
            if bound in body:
                _check_str(bound, body[bound], where)


def resolve_field(path: str, context: Mapping[str, Any]) -> Any:
    """Resolve a dotted field path against an evaluation context.

    A flat key matching the full path wins (contexts may be flat:
    ``{"host.cuda_version": "12.8"}``); otherwise each segment indexes into
    nested mappings. A missing segment resolves to ``None``.
    """
    key = str(path)
    if key in context:
        return context[key]
    current: Any = context
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _coerce_version(value: Any):
    try:
        from packaging.version import Version

        return Version(str(value))
    except Exception:
        return None


def _absent(value: Any) -> bool:
    return value is None or (isinstance(value, (str, Sequence, Mapping)) and len(value) == 0)


def evaluate_condition(node: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """Evaluate a validated condition against a context mapping.

    Total by design: evaluation of a well-formed condition never raises —
    unknown fields resolve to ``None``, unparseable versions are False — so
    recipe steps conservatively SKIP rather than crash an install.
    """
    ((op, body),) = node.items()
    if op == "all":
        return all(evaluate_condition(child, context) for child in body)
    if op == "any":
        return any(evaluate_condition(child, context) for child in body)
    value = resolve_field(body["field"], context)
    if op == "exists":
        return not _absent(value)
    if op == "equals":
        expected = body["value"]
        if isinstance(value, bool) or isinstance(expected, bool):
            return isinstance(value, bool) and value == expected
        return value == expected or (
            value is not None and not isinstance(value, Mapping) and str(value) == str(expected)
        )
    if op == "version_range":
        current = _coerce_version(value)
        if current is None:
            return False
        for bound, ok in (("min", lambda c, b: c >= b), ("max", lambda c, b: c <= b)):
            if bound in body:
                edge = _coerce_version(body[bound])
                if edge is None or not ok(current, edge):
                    return False
        return True
    if op == "capability_contains":
        needle = str(body["value"])
        if isinstance(value, Mapping):
            return needle in {str(v) for v in value.values()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return needle in {str(v) for v in value}
        if isinstance(value, str):
            return needle in [token for token in re.split(r"[,\s]+", value) if token]
        return False
    raise DefinitionError(f"cannot evaluate unknown condition operator {op!r}")


def validate_asset_selectors(selectors: Any, where: str) -> List[Dict[str, Any]]:
    """Validate release-asset selector lists: unique names, at most one default.

    An ambiguous artifact-selector list (duplicate names or multiple defaults)
    makes release resolution host-order dependent, so it is rejected at load.
    """
    _require(
        isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)),
        "asset selectors must be a list",
        where,
    )
    normalized: List[Dict[str, Any]] = []
    seen_names: set = set()
    defaults = 0
    for index, selector in enumerate(selectors):
        item_where = f"{where}[{index}]"
        _require(isinstance(selector, Mapping), "asset selector must be a mapping", item_where)
        unknown = set(selector) - {"name", "patterns", "default"}
        _require(not unknown, f"asset selector has unknown keys {sorted(unknown)}", item_where)
        name = _check_str("name", selector.get("name"), item_where)
        _require(name not in seen_names, f"duplicate asset selector name {name!r}", item_where)
        seen_names.add(name)
        patterns = _check_pattern_list("patterns", selector.get("patterns"), item_where)
        _require(bool(patterns), f"asset selector {name!r} needs at least one pattern", item_where)
        is_default = _check_bool("default", selector.get("default", False), item_where)
        defaults += int(is_default)
        _require(defaults <= 1, "asset selectors allow at most one default", item_where)
        normalized.append({"name": name, "patterns": list(patterns), "default": is_default})
    _require(len(normalized) > 0, "asset selectors must not be empty", where)
    return normalized


# ---------------------------------------------------------------------------
# Schema v2 section validators (settings, model sources, discovery, profiles,
# recipes). Validated at load; consumed by the Phase 3/6 executors.
# ---------------------------------------------------------------------------

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_scalar(name: str, value: Any, where: str) -> Any:
    _require(
        isinstance(value, (str, int, float, bool)) or value is None,
        f"{name} must be a scalar value",
        where,
    )
    return value


def _check_argv_list(name: str, value: Any, where: str) -> Tuple[str, ...]:
    """argv tokens are arbitrary strings (module paths, -c snippets, globs)."""
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a list",
        where,
    )
    return tuple(_check_str(name, item, where) for item in value)


def _validate_settings_defaults(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    _require(isinstance(value, Mapping), "settings_defaults must be a mapping", where)
    pairs: List[Tuple[str, Any]] = []
    for name, default in value.items():
        _check_str("settings_defaults key", name, where)
        pairs.append((name, _check_scalar(f"settings_defaults.{name}", default, where)))
    return tuple(sorted(pairs))


def _validate_source_build(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Native CMake source-build policy (repository/artifact/build choices)."""
    _require(isinstance(value, Mapping), "source_build must be a mapping", where)
    allowed = {
        "repository_name",
        "source_dir",
        "binary_name",
        "build_examples_required",
        "blas_supported",
        "stable_release_versions",
    }
    unknown = set(value) - allowed
    _require(not unknown, f"source_build has unknown keys {sorted(unknown)}", where)
    for key in ("repository_name", "source_dir", "binary_name"):
        if key in value:
            _check_str(f"source_build.{key}", value[key], where)
    for key in ("build_examples_required", "blas_supported", "stable_release_versions"):
        if key in value:
            _check_bool(f"source_build.{key}", value[key], where)
    return tuple((key, value[key]) for key in sorted(value))


def _validate_build_defaults(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Per-engine build-settings defaults (built on top of the catalog defaults)."""
    _require(isinstance(value, Mapping), "build_defaults must be a mapping", where)
    pairs: List[Tuple[str, Any]] = []
    for name, default in value.items():
        _check_str("build_defaults key", name, where)
        pairs.append((name, _check_scalar(f"build_defaults.{name}", default, where)))
    return tuple(sorted(pairs))


def _validate_model_sources(value: Any, where: str) -> Tuple[str, ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "model_sources must be a list",
        where,
    )
    items = tuple(
        _check_kind("model source kind", item, MODEL_SOURCE_KINDS, where) for item in value
    )
    _require(len(set(items)) == len(items), "model_sources contains duplicates", where)
    return items


CATALOG_PROVIDER_KINDS: FrozenSet[str] = frozenset({"huggingface", "audio_cpp"})


def _validate_catalog(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Model-catalog surface: which catalog providers serve this engine."""
    _require(isinstance(value, Mapping), "catalog must be a mapping", where)
    unknown = set(value) - {"providers"}
    _require(not unknown, f"catalog has unknown keys {sorted(unknown)}", where)
    providers = value.get("providers")
    _require(
        isinstance(providers, Sequence)
        and not isinstance(providers, (str, bytes))
        and bool(providers),
        "catalog.providers must be a non-empty list",
        where,
    )
    checked = tuple(
        _check_kind("catalog provider", item, CATALOG_PROVIDER_KINDS, where) for item in providers
    )
    _require(len(set(checked)) == len(checked), "catalog.providers contains duplicates", where)
    return tuple((key, value[key]) for key in sorted(value))


def _validate_upstream(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Upstream identity: the canonical repository fork detection compares to."""
    _require(isinstance(value, Mapping), "upstream must be a mapping", where)
    unknown = set(value) - {"repository_url"}
    _require(not unknown, f"upstream has unknown keys {sorted(unknown)}", where)
    url = _check_str("upstream.repository_url", value.get("repository_url"), where)
    _require(
        url.startswith(("https://", "http://", "git@", "ssh://")),
        "upstream.repository_url must be a remote URL",
        where,
    )
    return tuple((key, value[key]) for key in sorted(value))


def _validate_release_assets(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Prebuilt-archive release policy consumed by the native release host."""
    _require(isinstance(value, Mapping), "release_assets must be a mapping", where)
    allowed = {
        "repository",
        "repository_url",
        "manifest_asset",
        "checksum_asset",
        "max_walkback",
    }
    unknown = set(value) - allowed
    _require(not unknown, f"release_assets has unknown keys {sorted(unknown)}", where)
    for key in ("repository", "repository_url", "manifest_asset", "checksum_asset"):
        if key in value:
            _check_str(f"release_assets.{key}", value[key], where)
    if "max_walkback" in value:
        walkback = value["max_walkback"]
        _require(
            isinstance(walkback, int) and not isinstance(walkback, bool) and walkback > 0,
            "release_assets.max_walkback must be a positive integer",
            where,
        )
    return tuple((key, value[key]) for key in sorted(value))


def _validate_data_dirs(value: Any, where: str) -> Tuple[str, ...]:
    """Engine-owned data directories created under the studio data root."""
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "data_dirs must be a list",
        where,
    )
    items: List[str] = []
    for raw in value:
        item = _check_str("data_dirs item", raw, where)
        _require(
            not item.startswith(("/", "\\")) and ".." not in item.split("/"),
            f"data_dirs item {item!r} must be a relative path below the data root",
            where,
        )
        _require(item not in items, f"data_dirs contains duplicate {item!r}", where)
        items.append(item)
    return tuple(items)


_MODEL_COMPAT_KEYS = frozenset({"default_for_formats", "prepared_bundle_formats"})


def _validate_model_compat(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Model-compatibility declarations: default-for-format + bundle formats."""
    _require(isinstance(value, Mapping), "model_compat must be a mapping", where)
    unknown = set(value) - _MODEL_COMPAT_KEYS
    _require(not unknown, f"model_compat has unknown keys {sorted(unknown)}", where)
    for key_name, raw in value.items():
        _require(
            isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and bool(raw),
            f"model_compat.{key_name} must be a non-empty list",
            where,
        )
        for item in raw:
            _check_str(f"model_compat.{key_name} item", item, where)
        _require(
            len(set(raw)) == len(raw),
            f"model_compat.{key_name} contains duplicates",
            where,
        )
    return tuple((key, value[key]) for key in sorted(value))


def _validate_discovery(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    _require(isinstance(value, Mapping), "discovery must be a mapping", where)
    unknown = set(value) - {
        "mode",
        "parser",
        "probe_argv_prefix",
        "cwd",
        "cuda_env",
        "probe_variants",
        "env",
        "runtime_probes",
        "primary_runtime",
    }
    _require(not unknown, f"discovery has unknown keys {sorted(unknown)}", where)
    mode = _check_kind("discovery.mode", value.get("mode") or "probes", DISCOVERY_MODES, where)
    out: List[Tuple[str, Any]] = [("mode", mode)]
    if "parser" in value:
        parser = _check_kind("discovery.parser", value["parser"], SCANNER_KINDS, where)
        out.append(("parser", parser))
    else:
        _require(mode == "adapter", "discovery.parser is required for probe mode", where)
        parser = ""
    if mode == "probes":
        _require(
            bool(value.get("probe_argv_prefix") or value.get("probe_variants")),
            "discovery probe mode requires probe_argv_prefix or probe_variants",
            where,
        )
    if "primary_runtime" in value:
        name = value["primary_runtime"]
        _require(
            isinstance(name, str) and bool(_TOKEN_RE.match(name)),
            f"discovery.primary_runtime is not a token: {name!r}",
            where,
        )
        out.append(("primary_runtime", name))
    if "probe_argv_prefix" in value:
        out.append(
            (
                "probe_argv_prefix",
                list(
                    _check_argv_list(
                        "discovery.probe_argv_prefix", value["probe_argv_prefix"], where
                    )
                ),
            )
        )
    if "cwd" in value:
        out.append(
            (
                "cwd",
                _check_kind("discovery.cwd", value["cwd"], DISCOVERY_CWDS, where),
            )
        )
    if "cuda_env" in value:
        # bool: compose the managed CUDA env from the row's recorded version.
        # string: the DEFAULT CUDA version when the row records none.
        cuda = value["cuda_env"]
        if isinstance(cuda, str):
            out.append(("cuda_env", _check_str("discovery.cuda_env", cuda, where)))
        else:
            out.append(("cuda_env", _check_bool("discovery.cuda_env", cuda, where)))
    if "env" in value:
        env = value["env"]
        _require(isinstance(env, Mapping), "discovery.env must be a mapping", where)
        unknown = set(env) - {"defaults", "required"}
        _require(not unknown, f"discovery.env has unknown keys {sorted(unknown)}", where)
        defaults = env.get("defaults", {})
        _require(isinstance(defaults, Mapping), "discovery.env.defaults must map", where)
        for env_name, env_value in defaults.items():
            _require(
                isinstance(env_name, str) and bool(_ENV_NAME_RE.match(env_name)),
                f"discovery.env.defaults contains invalid name {env_name!r}",
                where,
            )
            _check_str(f"discovery.env.defaults.{env_name}", env_value, where)
        required = env.get("required", [])
        _require(
            isinstance(required, Sequence) and not isinstance(required, (str, bytes)),
            "discovery.env.required must be a list",
            where,
        )
        for req in required:
            if isinstance(req, Mapping):
                unknown = set(req) - {"name", "error"}
                _require(
                    not unknown,
                    f"discovery.env.required has unknown keys {sorted(unknown)}",
                    where,
                )
                _require(
                    isinstance(req.get("name"), str) and bool(_ENV_NAME_RE.match(req["name"])),
                    f"discovery.env.required contains invalid name {req.get('name')!r}",
                    where,
                )
                _check_str("discovery.env.required.error", req.get("error"), where)
                continue
            _require(
                isinstance(req, str) and bool(_ENV_NAME_RE.match(req)),
                f"discovery.env.required contains invalid name {req!r}",
                where,
            )
        out.append(
            (
                "env",
                {
                    "defaults": {str(k): str(v) for k, v in defaults.items()},
                    "required": [
                        (
                            {"name": str(item.get("name")), "error": str(item.get("error") or "")}
                            if isinstance(item, Mapping)
                            else {"name": str(item), "error": ""}
                        )
                        for item in required
                    ],
                },
            )
        )
    if "runtime_probes" in value:
        probes = value["runtime_probes"]
        _require(
            isinstance(probes, Mapping) and bool(probes),
            "discovery.runtime_probes must be a non-empty mapping",
            where,
        )
        normalized_probes: Dict[str, Any] = {}
        for name, raw_probe in sorted(probes.items()):
            pwhere = f"{where}.discovery.runtime_probes.{name}"
            _require(
                isinstance(name, str) and bool(_TOKEN_RE.match(name)),
                f"invalid runtime probe name {name!r}",
                pwhere,
            )
            _require(isinstance(raw_probe, Mapping), "runtime probe must be a mapping", pwhere)
            unknown = set(raw_probe) - {
                "parser",
                "argv_prefix",
                "cwd",
                "unavailable_marker",
            }
            _require(not unknown, f"runtime probe has unknown keys {sorted(unknown)}", pwhere)
            probe_parser = _check_kind(
                "runtime probe parser", raw_probe.get("parser"), SCANNER_KINDS, pwhere
            )
            _require("argv_prefix" in raw_probe, "runtime probe requires 'argv_prefix'", pwhere)
            argv = _check_argv_list("runtime probe argv_prefix", raw_probe["argv_prefix"], pwhere)
            _require(bool(argv), "runtime probe argv_prefix must not be empty", pwhere)
            probe: Dict[str, Any] = {"parser": probe_parser, "argv_prefix": list(argv)}
            if "cwd" in raw_probe:
                probe["cwd"] = _check_kind(
                    "runtime probe cwd", raw_probe["cwd"], DISCOVERY_CWDS, pwhere
                )
            if "unavailable_marker" in raw_probe:
                probe["unavailable_marker"] = _check_str(
                    "runtime probe unavailable_marker", raw_probe["unavailable_marker"], pwhere
                )
            normalized_probes[name] = probe
        out.append(("runtime_probes", normalized_probes))
    if "probe_variants" in value:
        raw_variants = value["probe_variants"]
        _require(
            isinstance(raw_variants, Sequence)
            and not isinstance(raw_variants, (str, bytes))
            and bool(raw_variants),
            "discovery.probe_variants must be a non-empty list",
            where,
        )
        variants: List[Dict[str, Any]] = []
        for index, raw_variant in enumerate(raw_variants):
            vwhere = f"{where}.discovery.probe_variants[{index}]"
            _require(isinstance(raw_variant, Mapping), "probe variant must be a mapping", vwhere)
            unknown = set(raw_variant) - {"executable", "argv_prefix", "cwd"}
            _require(not unknown, f"probe variant has unknown keys {sorted(unknown)}", vwhere)
            variant: Dict[str, Any] = {}
            if "executable" in raw_variant:
                executable = raw_variant["executable"]
                _require(
                    isinstance(executable, Mapping), "probe variant executable must map", vwhere
                )
                unknown = set(executable) - {"kind", "name"}
                _require(
                    not unknown,
                    f"probe variant executable has unknown keys {sorted(unknown)}",
                    vwhere,
                )
                kind = _check_kind(
                    "probe variant executable.kind",
                    executable.get("kind"),
                    PROFILE_EXECUTABLE_KINDS,
                    vwhere,
                )
                name = _check_str("probe variant executable.name", executable.get("name"), vwhere)
                variant["executable"] = {"kind": kind, "name": name}
            _require("argv_prefix" in raw_variant, "probe variant requires 'argv_prefix'", vwhere)
            argv = _check_argv_list("probe variant argv_prefix", raw_variant["argv_prefix"], vwhere)
            _require(bool(argv), "probe variant argv_prefix must not be empty", vwhere)
            variant["argv_prefix"] = list(argv)
            if "cwd" in raw_variant:
                variant["cwd"] = _check_kind(
                    "probe variant cwd", raw_variant["cwd"], DISCOVERY_CWDS, vwhere
                )
            variants.append(variant)
        out.append(("probe_variants", variants))
    return tuple(out)


def _validate_profile(name: str, value: Any, where: str) -> Dict[str, Any]:
    item_where = f"{where}.runtime_profiles.{name}"
    _require(isinstance(value, Mapping), f"profile {name!r} must be a mapping", item_where)
    unknown = set(value) - PROFILE_KEYS
    _require(not unknown, f"profile {name!r} has unknown keys {sorted(unknown)}", item_where)

    if "launch_family" in value:
        # Family profile: the named mechanism composes the launch, so the
        # profile declares only the parts Studio still chooses (port flag).
        family = _check_kind(
            "launch_family", value.get("launch_family"), LAUNCH_FAMILIES, item_where
        )
        disallowed = set(value) - FAMILY_PROFILE_KEYS
        _require(
            not disallowed,
            f"profile {name!r} declaring launch_family {family!r} cannot also "
            f"declare {sorted(disallowed)}: the family mechanism composes those",
            item_where,
        )
        _require(
            "port_binding" in value,
            f"profile {name!r} requires 'port_binding'",
            item_where,
        )
        launch_args = value.get("launch_args")
        if launch_args is not None:
            _require(
                isinstance(launch_args, Mapping),
                "launch_args must be a mapping",
                item_where,
            )
            unknown = set(launch_args) - LAUNCH_ARG_KEYS
            _require(
                not unknown,
                f"launch_args has unknown keys {sorted(unknown)}",
                item_where,
            )
            for key, flag in launch_args.items():
                _check_str(f"launch_args.{key}", flag, item_where)
    else:
        for required in ("executable", "model_binding", "port_binding"):
            _require(required in value, f"profile {name!r} requires '{required}'", item_where)

        executable = value["executable"]
        _require(isinstance(executable, Mapping), "executable must be a mapping", item_where)
        unknown = set(executable) - {"kind", "name"}
        _require(not unknown, f"executable has unknown keys {sorted(unknown)}", item_where)
        kind = _check_kind(
            "executable.kind", executable.get("kind"), PROFILE_EXECUTABLE_KINDS, item_where
        )
        if kind == "venv_console_script":
            _check_str("executable.name", executable.get("name"), item_where)

        binding = value["model_binding"]
        _require(isinstance(binding, Mapping), "model_binding must be a mapping", item_where)
        unknown = set(binding) - {"mode", "flag", "source"}
        _require(not unknown, f"model_binding has unknown keys {sorted(unknown)}", item_where)
        mode = _check_kind(
            "model_binding.mode", binding.get("mode"), MODEL_BINDING_MODES, item_where
        )
        _check_kind(
            "model_binding.source", binding.get("source"), MODEL_BINDING_SOURCES, item_where
        )
        if mode == "flag":
            _check_str("model_binding.flag", binding.get("flag"), item_where)

    port_binding = value["port_binding"]
    _require(isinstance(port_binding, Mapping), "port_binding must be a mapping", item_where)
    unknown = set(port_binding) - {"flag"}
    _require(not unknown, f"port_binding has unknown keys {sorted(unknown)}", item_where)
    _check_str("port_binding.flag", port_binding.get("flag"), item_where)

    if "argv_prefix" in value:
        _check_argv_list("argv_prefix", value["argv_prefix"], item_where)
    if "default" in value:
        _check_bool("default", value["default"], item_where)
    if "description" in value:
        _check_str("description", value["description"], item_where)
    if "env" in value:
        env = value["env"]
        _require(isinstance(env, Mapping), "env must be a mapping", item_where)
        unknown = set(env) - {
            "required",
            "defaults",
            "prepend_bin_to_path",
            "cwd",
            "cuda",
            "caches",
        }
        _require(not unknown, f"env has unknown keys {sorted(unknown)}", item_where)
        if "launch_family" in value:
            # The family mechanism composes cwd/CUDA/required/path itself;
            # only the Studio-owned managed cache declarations belong here.
            disallowed_env = set(env) - {"caches"}
            _require(
                not disallowed_env,
                f"profile {name!r} declaring launch_family cannot also "
                f"declare env {sorted(disallowed_env)}: the family mechanism "
                "composes those",
                item_where,
            )
        if "caches" in env:
            # Managed JIT/kernel caches resolved to <install_dir>/caches/<var>.
            # The variable NAMES are data; Studio owns the path and the flush.
            caches = env["caches"]
            _require(
                isinstance(caches, list) and bool(caches),
                "env.caches must be a non-empty list of environment variable names",
                item_where,
            )
            for cache_name in caches:
                _require(
                    isinstance(cache_name, str) and bool(_ENV_NAME_RE.match(cache_name)),
                    f"env.caches contains invalid name {cache_name!r}",
                    item_where,
                )
        if "cuda" in env:
            # Studio-managed CUDA environment resolved per active install
            # (mirrors the installer's `build_env: cuda`). A string value is
            # the DEFAULT CUDA version used when the active row has none.
            cuda = env["cuda"]
            if isinstance(cuda, str):
                _check_str("env.cuda", cuda, item_where)
            else:
                _check_bool("env.cuda", cuda, item_where)
        for req in env.get("required", []):
            if isinstance(req, Mapping):
                unknown = set(req) - {"name", "error"}
                _require(
                    not unknown,
                    f"env.required entry has unknown keys {sorted(unknown)}",
                    item_where,
                )
                _require(
                    isinstance(req.get("name"), str) and bool(_ENV_NAME_RE.match(req["name"])),
                    f"env.required contains invalid name {req.get('name')!r}",
                    item_where,
                )
                _check_str("env.required.error", req.get("error"), item_where)
                continue
            _require(
                isinstance(req, str) and bool(_ENV_NAME_RE.match(req)),
                f"env.required contains invalid name {req!r}",
                item_where,
            )
        defaults = env.get("defaults", {})
        _require(isinstance(defaults, Mapping), "env.defaults must be a mapping", item_where)
        for env_name, env_value in defaults.items():
            _require(
                isinstance(env_name, str) and bool(_ENV_NAME_RE.match(env_name)),
                f"env.defaults contains invalid name {env_name!r}",
                item_where,
            )
            _check_str(f"env.defaults.{env_name}", env_value, item_where)
        if "prepend_bin_to_path" in env:
            _check_bool("env.prepend_bin_to_path", env["prepend_bin_to_path"], item_where)
        if "cwd" in env:
            _require(
                env["cwd"] in {"venv_root", "none"},
                f"env.cwd must be 'venv_root' or 'none', got {env['cwd']!r}",
                item_where,
            )
    if "scan" in value:
        scan = value["scan"]
        _require(isinstance(scan, Mapping), "scan must be a mapping", item_where)
        unknown = set(scan) - {"probe_argv_prefix", "unavailable_hint"}
        _require(not unknown, f"scan has unknown keys {sorted(unknown)}", item_where)
        if "probe_argv_prefix" in scan:
            _check_argv_list("scan.probe_argv_prefix", scan["probe_argv_prefix"], item_where)
        if "unavailable_hint" in scan:
            _check_str("scan.unavailable_hint", scan["unavailable_hint"], item_where)
    if "runtime_switch" in value:
        switch = value["runtime_switch"]
        _require(isinstance(switch, Mapping), "runtime_switch must be a mapping", item_where)
        unknown = set(switch) - {"config_key", "default", "variants"}
        _require(
            not unknown,
            f"runtime_switch has unknown keys {sorted(unknown)}",
            item_where,
        )
        _check_str("runtime_switch.config_key", switch.get("config_key"), item_where)
        _check_str("runtime_switch.default", switch.get("default"), item_where)
        variants = switch.get("variants")
        _require(
            isinstance(variants, Mapping) and bool(variants),
            "runtime_switch.variants must be a non-empty mapping",
            item_where,
        )
        for variant_name, variant in variants.items():
            variant_where = f"{item_where}.runtime_switch.variants.{variant_name}"
            _require(
                isinstance(variant_name, str) and bool(_TOKEN_RE.match(variant_name)),
                f"invalid runtime variant name {variant_name!r}",
                item_where,
            )
            _require(
                isinstance(variant, Mapping),
                "runtime variant must be a mapping",
                variant_where,
            )
            unknown = set(variant) - {"argv_prefix", "scanned_catalog"}
            _require(
                not unknown,
                f"runtime variant has unknown keys {sorted(unknown)}",
                variant_where,
            )
            if "argv_prefix" in variant:
                _check_argv_list("argv_prefix", variant["argv_prefix"], variant_where)
            if "scanned_catalog" in variant:
                _check_str("scanned_catalog", variant["scanned_catalog"], variant_where)
    return dict(value)


_FORK_OPTION_KEYS = frozenset(
    {
        "ccache",
        "variables",
        "create_dirs",
        "dependencies",
        "rustup_home_fallback",
        "required_tools",
        "toolchain",
        "concurrency",
        "caches",
        "require_cuda_nvcc",
        "cuda_error",
    }
)
_TRANSFORM_NAME_RE = re.compile(r"^[a-z0-9_-]+/[a-z0-9_-]+$")


def _validate_env_name_list(name: str, value: Any, where: str) -> None:
    names = _check_argv_list(name, value, where)
    for item in names:
        _require(
            bool(_ENV_NAME_RE.match(item)),
            f"{name} contains invalid environment name {item!r}",
            where,
        )


def _validate_fork_with(step_id: str, value: Any, where: str) -> None:
    """Validate the typed ``build.prepare_fork`` option mapping."""
    _require(isinstance(value, Mapping), f"step {step_id!r} 'with' must be a mapping", where)
    unknown = set(value) - _FORK_OPTION_KEYS
    _require(not unknown, f"step {step_id!r} 'with' has unknown keys {sorted(unknown)}", where)
    for key in ("ccache", "require_cuda_nvcc"):
        if key in value:
            _check_bool(f"build.prepare_fork.{key}", value[key], where)
    for key in ("cuda_error", "rustup_home_fallback"):
        if key in value:
            _check_str(f"build.prepare_fork.{key}", value[key], where)
    for key in ("create_dirs", "required_tools"):
        if key in value:
            _check_argv_list(f"build.prepare_fork.{key}", value[key], where)
    variables = value.get("variables", {})
    _require(isinstance(variables, Mapping), "build.prepare_fork.variables must map", where)
    for name, variable in variables.items():
        _require(
            isinstance(name, str) and bool(_ENV_NAME_RE.match(name)),
            f"build.prepare_fork.variables has invalid name {name!r}",
            where,
        )
        _check_scalar(f"build.prepare_fork.variables.{name}", variable, where)
    dependencies = value.get("dependencies", [])
    _require(
        isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes)),
        "build.prepare_fork.dependencies must be a list",
        where,
    )
    for index, dependency in enumerate(dependencies):
        dwhere = f"{where}.build.prepare_fork.dependencies[{index}]"
        _require(isinstance(dependency, Mapping), "dependency entry must be a mapping", dwhere)
        unknown = set(dependency) - {
            "path",
            "preserve_if_file",
            "require_files",
            "preserve_if_missing",
            "reason",
        }
        _require(not unknown, f"dependency entry has unknown keys {sorted(unknown)}", dwhere)
        _check_str("dependency path", dependency.get("path"), dwhere)
        if "preserve_if_file" in dependency:
            _check_str("dependency preserve_if_file", dependency["preserve_if_file"], dwhere)
        if "preserve_if_missing" in dependency:
            _check_bool("dependency preserve_if_missing", dependency["preserve_if_missing"], dwhere)
        if "require_files" in dependency:
            _check_argv_list("dependency require_files", dependency["require_files"], dwhere)
        if "reason" in dependency:
            _check_str("dependency reason", dependency["reason"], dwhere)
    toolchain = value.get("toolchain", {})
    _require(isinstance(toolchain, Mapping), "build.prepare_fork.toolchain must map", where)
    unknown = set(toolchain) - {"major_min", "major_max", "env_vars", "error"}
    _require(not unknown, f"toolchain has unknown keys {sorted(unknown)}", where)
    for key in ("major_min", "major_max"):
        if key in toolchain:
            _require(
                isinstance(toolchain[key], int) and not isinstance(toolchain[key], bool),
                f"toolchain.{key} must be an integer",
                where,
            )
    if "env_vars" in toolchain:
        _validate_env_name_list("toolchain.env_vars", toolchain["env_vars"], where)
    if "error" in toolchain:
        _check_str("toolchain.error", toolchain["error"], where)
    concurrency = value.get("concurrency", {})
    _require(isinstance(concurrency, Mapping), "build.prepare_fork.concurrency must map", where)
    unknown = set(concurrency) - {"memory_aware", "env_vars", "nvcc_threads"}
    _require(not unknown, f"concurrency has unknown keys {sorted(unknown)}", where)
    if "caches" in value:
        # Same contract as runtime_profiles.<name>.env.caches: the declared
        # variables point inside this install's managed cache root, so the
        # builder's precompile/smoke step fills what the server will read.
        caches = value["caches"]
        _require(
            isinstance(caches, list) and bool(caches),
            "build.prepare_fork.caches must be a non-empty list of environment variable names",
            where,
        )
        for cache_name in caches:
            _require(
                isinstance(cache_name, str) and bool(_ENV_NAME_RE.match(cache_name)),
                f"build.prepare_fork.caches contains invalid name {cache_name!r}",
                where,
            )
    if "memory_aware" in concurrency:
        _check_bool("concurrency.memory_aware", concurrency["memory_aware"], where)
    if "env_vars" in concurrency:
        _validate_env_name_list("concurrency.env_vars", concurrency["env_vars"], where)
    if "nvcc_threads" in concurrency:
        # An integer pins the per-compile nvcc parallelism; "auto" (or omitting
        # the key) derives it from the host budget and the architecture count.
        threads = concurrency["nvcc_threads"]
        _require(
            (isinstance(threads, int) and not isinstance(threads, bool))
            or (isinstance(threads, str) and threads.strip().lower() == "auto"),
            "concurrency.nvcc_threads must be an integer or 'auto'",
            where,
        )


def _validate_transform_with(step_id: str, value: Any, where: str) -> None:
    _require(isinstance(value, Mapping), f"step {step_id!r} 'with' must be a mapping", where)
    unknown = set(value) - {"transform"}
    _require(not unknown, f"step {step_id!r} 'with' has unknown keys {sorted(unknown)}", where)
    name = _check_str("vcs.apply_transform.transform", value.get("transform"), where)
    _require(
        bool(_TRANSFORM_NAME_RE.match(name)),
        f"transform asset name {name!r} must be '<dir>/<stem>'",
        where,
    )


def _progress_window(value: Any, where: str) -> None:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"{where} must be a two-integer list",
        where,
    )


def _validate_toolchain_with(step_id: str, value: Any, where: str) -> None:
    _require(isinstance(value, Mapping), f"step {step_id!r} 'with' must be a mapping", where)
    unknown = set(value) - {"kind", "installer_url", "progress_window", "output_key"}
    _require(not unknown, f"step {step_id!r} 'with' has unknown keys {sorted(unknown)}", where)
    _check_str("toolchain.ensure.kind", value.get("kind"), where)
    if "output_key" in value:
        _check_str("toolchain.ensure.output_key", value["output_key"], where)
    if "installer_url" in value:
        _check_str("toolchain.ensure.installer_url", value["installer_url"], where)
    if "progress_window" in value:
        _progress_window(value["progress_window"], "toolchain.ensure.progress_window")


def _validate_build_wheels_with(step_id: str, value: Any, where: str) -> None:
    _require(isinstance(value, Mapping), f"step {step_id!r} 'with' must be a mapping", where)
    unknown = set(value) - {
        "empty_error",
        "sub_builds",
        "main_progress_window",
        "main_error",
        "rust_hint",
    }
    _require(not unknown, f"step {step_id!r} 'with' has unknown keys {sorted(unknown)}", where)
    for key in ("empty_error", "main_error"):
        if key in value:
            _check_str(f"artifact.build_wheels.{key}", value[key], where)
    if "rust_hint" in value:
        _check_bool("artifact.build_wheels.rust_hint", value["rust_hint"], where)
    if "main_progress_window" in value:
        _progress_window(
            value["main_progress_window"], "artifact.build_wheels.main_progress_window"
        )
    sub_builds = value.get("sub_builds", [])
    _require(
        isinstance(sub_builds, Sequence) and not isinstance(sub_builds, (str, bytes)),
        "artifact.build_wheels.sub_builds must be a list",
        where,
    )
    for index, sub in enumerate(sub_builds):
        swhere = f"{where}.artifact.build_wheels.sub_builds[{index}]"
        _require(isinstance(sub, Mapping), "sub build must be a mapping", swhere)
        unknown = set(sub) - {"path", "progress_window", "error"}
        _require(not unknown, f"sub build has unknown keys {sorted(unknown)}", swhere)
        _check_str("sub build path", sub.get("path"), swhere)
        _check_str("sub build error", sub.get("error"), swhere)
        if "progress_window" in sub:
            _progress_window(sub["progress_window"], f"{swhere}.progress_window")


def _validate_release_assets_with(step_id: str, value: Any, where: str) -> None:
    _require(isinstance(value, Mapping), f"step {step_id!r} 'with' must be a mapping", where)
    unknown = set(value) - {"repository", "wheel_groups", "error"}
    _require(not unknown, f"step {step_id!r} 'with' has unknown keys {sorted(unknown)}", where)
    _check_str("prereqs.resolve_release_assets.repository", value.get("repository"), where)
    if "error" in value:
        _check_str("prereqs.resolve_release_assets.error", value["error"], where)
    groups = value.get("wheel_groups")
    _require(
        isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)) and bool(groups),
        "prereqs.resolve_release_assets.wheel_groups must be a non-empty list",
        where,
    )
    for index, group in enumerate(groups):
        gwhere = f"{where}.prereqs.resolve_release_assets.wheel_groups[{index}]"
        _require(isinstance(group, Mapping), "wheel group must be a mapping", gwhere)
        unknown = set(group) - {"prefixes", "required_prefixes"}
        _require(not unknown, f"wheel group has unknown keys {sorted(unknown)}", gwhere)
        _check_argv_list("wheel group prefixes", group.get("prefixes"), gwhere)
        if "required_prefixes" in group:
            _check_argv_list("wheel group required_prefixes", group["required_prefixes"], gwhere)


def _validate_recipes(value: Any, where: str) -> Tuple[Tuple[str, Tuple[Dict[str, Any], ...]], ...]:
    _require(isinstance(value, Mapping), "recipes must be a mapping", where)
    out: List[Tuple[str, Tuple[Dict[str, Any], ...]]] = []
    for recipe_name, steps in value.items():
        recipe_where = f"{where}.recipes.{recipe_name}"
        _require(
            isinstance(recipe_name, str) and bool(_TOKEN_RE.match(recipe_name)),
            f"invalid recipe name {recipe_name!r}",
            where,
        )
        _require(
            isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)),
            f"recipe {recipe_name!r} must be a list of steps",
            recipe_where,
        )
        _require(len(steps) > 0, f"recipe {recipe_name!r} must not be empty", recipe_where)
        seen_ids: set = set()
        produced: set = set()
        normalized_steps: List[Dict[str, Any]] = []
        for index, step in enumerate(steps):
            step_where = f"{recipe_where}[{index}]"
            _require(isinstance(step, Mapping), "recipe step must be a mapping", step_where)
            unknown = set(step) - RECIPE_STEP_KEYS
            _require(not unknown, f"recipe step has unknown keys {sorted(unknown)}", step_where)
            step_id = _check_str("id", step.get("id"), step_where)
            _require(
                bool(_TOKEN_RE.match(step_id)), f"step id {step_id!r} is not a token", step_where
            )
            _require(step_id not in seen_ids, f"duplicate step id {step_id!r}", step_where)
            seen_ids.add(step_id)
            uses, run = step.get("uses"), step.get("run")
            _require(
                (uses is None) != (run is None),
                f"step {step_id!r} needs exactly one of 'uses' or 'run'",
                step_where,
            )
            if uses is not None:
                _require(
                    isinstance(uses, str) and uses in INSTALL_PRIMITIVES,
                    f"step {step_id!r} references unknown primitive {uses!r} "
                    f"(known: {sorted(INSTALL_PRIMITIVES)})",
                    step_where,
                )
            else:
                _check_argv_list("run", run, step_where)
            if "with" in step:
                _require(
                    isinstance(step["with"], Mapping),
                    f"step {step_id!r} 'with' must be a mapping",
                    step_where,
                )
                if uses == "build.prepare_fork":
                    _validate_fork_with(step_id, step["with"], step_where)
                elif uses == "vcs.apply_transform":
                    _validate_transform_with(step_id, step["with"], step_where)
                elif uses == "toolchain.ensure":
                    _validate_toolchain_with(step_id, step["with"], step_where)
                elif uses == "artifact.build_wheels":
                    _validate_build_wheels_with(step_id, step["with"], step_where)
                elif uses == "prereqs.resolve_release_assets":
                    _validate_release_assets_with(step_id, step["with"], step_where)
                else:
                    for operand, operand_value in step["with"].items():
                        _check_str("with key", operand, step_where)
                        if isinstance(operand_value, Mapping):
                            _require(
                                not set(operand_value) - {"selector", "patterns", "default"},
                                f"step {step_id!r} 'with.{operand}' has unknown keys",
                                step_where,
                            )
                        elif isinstance(operand_value, Sequence) and not isinstance(
                            operand_value, (str, bytes)
                        ):
                            _check_argv_list(f"with.{operand}", operand_value, step_where)
                        else:
                            _check_scalar(f"with.{operand}", operand_value, step_where)
            if "when" in step:
                validate_condition(step["when"], f"{step_where}.when")
            if "cleanup" in step:
                # Always-run step: executes even after the body failed or was
                # cancelled (the plan's finally). Must be a plain boolean.
                _require(
                    isinstance(step["cleanup"], bool),
                    f"step {step_id!r} 'cleanup' must be a boolean",
                    step_where,
                )
            produces = step.get("produces", [])
            if produces:
                _check_token_list("produces", produces, step_where)
            for output in produces:
                _require(output not in produced, f"duplicate step output {output!r}", step_where)
            requires = step.get("requires", [])
            if requires:
                _check_token_list("requires", requires, step_where)
            for requirement in requires:
                _require(
                    requirement.startswith("input.") or requirement in produced,
                    f"step {step_id!r} requires undeclared output {requirement!r}",
                    step_where,
                )
            produced.update(produces)
            if "timeout_s" in step:
                _require(
                    isinstance(step["timeout_s"], int)
                    and not isinstance(step["timeout_s"], bool)
                    and step["timeout_s"] > 0,
                    f"step {step_id!r} timeout_s must be a positive integer",
                    step_where,
                )
            if "accepted_exit_codes" in step:
                codes = step["accepted_exit_codes"]
                _require(
                    isinstance(codes, Sequence)
                    and not isinstance(codes, (str, bytes))
                    and all(isinstance(c, int) and not isinstance(c, bool) for c in codes),
                    f"step {step_id!r} accepted_exit_codes must be a list of integers",
                    step_where,
                )
            normalized_steps.append(dict(step))
        out.append((recipe_name, tuple(normalized_steps)))
    return tuple(out)


# ---------------------------------------------------------------------------
# EngineDefinition: validated, immutable, expandable to a registry EngineSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineDefinition:
    id: str
    label: str
    repository_source: str
    install_kind: str
    runtime_kind: str
    scanner_kind: str
    active_path_fields: Tuple[str, ...]
    artifact_formats: Tuple[str, ...]
    package_kinds: Tuple[str, ...]
    tasks: Tuple[str, ...]
    input_modalities: Tuple[str, ...]
    output_modalities: Tuple[str, ...]
    supports_embeddings: bool
    experimental: bool
    maturity_surfaces: Tuple[Tuple[str, str], ...]
    source_path: str = ""
    schema_version: int = SCHEMA_VERSION
    settings_defaults: Tuple[Tuple[str, Any], ...] = ()
    model_sources: Tuple[str, ...] = ()
    discovery: Tuple[Tuple[str, Any], ...] = ()
    runtime_profiles: Tuple[Tuple[str, Mapping[str, Any]], ...] = ()
    recipes: Tuple[Tuple[str, Tuple[Mapping[str, Any], ...]], ...] = ()
    lifecycle: Tuple[Tuple[str, Any], ...] = ()
    # Native host mechanism (cmake_source/audio_cpp); empty for venv engines.
    native_host: str = ""
    # Frontend presentation family. Empty falls back to the generic card.
    ui_kind: str = ""
    launch_policy: Tuple[Tuple[str, Any], ...] = ()
    build_options: Tuple[Tuple[str, Any], ...] = ()
    protocol: Tuple[Tuple[str, Any], ...] = ()
    extensions: Tuple[Mapping[str, Any], ...] = ()
    request_defaults: Tuple[Tuple[str, Any], ...] = ()
    catalog: Tuple[Tuple[str, Any], ...] = ()
    upstream: Tuple[Tuple[str, Any], ...] = ()
    data_dirs: Tuple[str, ...] = ()
    model_compat: Tuple[Tuple[str, Any], ...] = ()
    build_defaults: Tuple[Tuple[str, Any], ...] = ()
    source_build: Tuple[Tuple[str, Any], ...] = ()
    release_assets: Tuple[Tuple[str, Any], ...] = ()

    def profile(self, name: str = "") -> Mapping[str, Any]:
        """Named runtime profile (default-marked when no name is given)."""
        profiles = dict(self.runtime_profiles)
        if name:
            if name not in profiles:
                raise DefinitionError(f"{self.id}: no runtime profile {name!r}")
            return profiles[name]
        for candidate, spec in profiles.items():
            if spec.get("default"):
                return spec
        if len(profiles) == 1:
            return next(iter(profiles.values()))
        raise DefinitionError(f"{self.id}: {len(profiles)} runtime profiles, none marked default")

    def to_spec(self) -> EngineSpec:
        profile = {}
        if self.runtime_profiles:
            try:
                profile = dict(self.profile(""))
            except DefinitionError:
                profile = {}
        return EngineSpec(
            id=self.id,
            label=self.label,
            repository_source=self.repository_source,
            install_kind=self.install_kind,
            runtime_kind=self.runtime_kind,
            scanner_kind=self.scanner_kind,
            active_path_fields=self.active_path_fields,
            artifact_formats=frozenset(self.artifact_formats),
            package_kinds=frozenset(self.package_kinds),
            tasks=frozenset(self.tasks),
            input_modalities=frozenset(self.input_modalities),
            output_modalities=frozenset(self.output_modalities),
            supports_embeddings=self.supports_embeddings,
            experimental=self.experimental,
            maturity_surfaces=self.maturity_surfaces,
            ui_kind=self.ui_kind,
            lifecycle=dict(self.lifecycle),
            launch_policy=dict(self.launch_policy),
            build_options=dict(self.build_options),
            protocol=dict(self.protocol),
            extensions=tuple(dict(item) for item in self.extensions),
            request_defaults=dict(self.request_defaults),
            catalog=dict(self.catalog),
            upstream=dict(self.upstream),
            data_dirs=tuple(self.data_dirs),
            model_compat=dict(self.model_compat),
            build_defaults=dict(self.build_defaults),
            settings_defaults=dict(self.settings_defaults),
            runtime_profile=profile,
        )

    def to_api_dict(self) -> Dict[str, Any]:
        return self.to_spec().to_api_dict()

    def _transform_asset_digests(self) -> Dict[str, str]:
        """Content digests for every transform asset this definition references.

        The recipe holds only the asset NAME; without this, editing an asset's
        operations (for example a CUDA pin) would leave the definition digest —
        and therefore the install/retry identity — unchanged.
        """
        from dataclasses import asdict

        from backend.engines.definitions.transforms import load_transform

        names = set()
        for _recipe, steps in self.recipes:
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                if step.get("uses") != "vcs.apply_transform":
                    continue
                name = str((step.get("with") or {}).get("transform") or "").strip()
                if name:
                    names.add(name)
        digests: Dict[str, str] = {}
        for name in sorted(names):
            asset = asdict(load_transform(name))
            raw = json.dumps(
                asset, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
            )
            digests[name] = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return digests

    @property
    def definition_digest(self) -> str:
        """Canonical sha256 over the FULL validated definition (recipes
        included — an install snapshot must freeze the effective recipe,
        not just registry identity). ``source_path`` is excluded because
        it is host-dependent; everything else that validation accepted is
        content. Referenced transform assets contribute their content digest
        so an asset-only edit changes installation identity too. Stable across
        loads, machines and dict ordering."""
        payload = {
            f.name: _canonical_definition_value(getattr(self, f.name))
            for f in fields(self)
            if f.name != "source_path"
        }
        asset_digests = self._transform_asset_digests()
        if asset_digests:
            payload["transform_assets"] = asset_digests
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical_definition_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical_definition_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_definition_value(v) for v in value]
    return value


def _defaults_applied(raw: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    merged = dict(raw)
    merged.setdefault("supports_embeddings", False)
    merged.setdefault("experimental", False)
    merged.setdefault("maturity_surfaces", {})
    return merged


def validate_definition(raw: Any, *, source: str) -> EngineDefinition:
    """Validate a fully expanded engine definition mapping (templates applied)."""
    where = source or "<definition>"
    _require(isinstance(raw, Mapping), "engine definition must be a mapping", where)
    version = raw.get("schema_version")
    _require(
        isinstance(version, int) and not isinstance(version, bool),
        "schema_version must be an integer",
        where,
    )
    _require(
        version in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported schema_version {version!r} (expected one of "
        f"{sorted(SUPPORTED_SCHEMA_VERSIONS)})",
        where,
    )
    allowed_keys = set(ENGINE_DEFINITION_KEYS)
    if version >= 2:
        allowed_keys |= set(V2_ONLY_KEYS)
    else:
        gated = sorted(key for key in V2_ONLY_KEYS if key in raw)
        _require(
            not gated,
            f"sections {gated} require schema_version 2",
            where,
        )
    unknown = set(raw) - allowed_keys
    _require(not unknown, f"unknown definition fields {sorted(unknown)}", where)
    engine_id = _check_str("id", raw.get("id"), where)
    _require(bool(_ID_RE.match(engine_id)), f"id {engine_id!r} must match {_ID_RE.pattern}", where)
    _require(
        "extends" not in raw or raw["extends"] is None,
        "extends must be resolved before validation",
        where,
    )

    raw = _defaults_applied(raw, where)
    try:
        return EngineDefinition(
            id=engine_id,
            label=_check_str("label", raw.get("label"), where),
            repository_source=_check_str("repository_source", raw.get("repository_source"), where),
            install_kind=_check_kind("install_kind", raw.get("install_kind"), INSTALL_KINDS, where),
            runtime_kind=_check_kind("runtime_kind", raw.get("runtime_kind"), RUNTIME_KINDS, where),
            scanner_kind=_check_kind("scanner_kind", raw.get("scanner_kind"), SCANNER_KINDS, where),
            active_path_fields=_check_token_list(
                "active_path_fields", raw.get("active_path_fields"), where
            ),
            artifact_formats=_check_token_list(
                "artifact_formats", raw.get("artifact_formats"), where
            ),
            package_kinds=_check_token_list("package_kinds", raw.get("package_kinds"), where),
            tasks=_check_token_list("tasks", raw.get("tasks"), where),
            input_modalities=_check_token_list(
                "input_modalities", raw.get("input_modalities"), where
            ),
            output_modalities=_check_token_list(
                "output_modalities", raw.get("output_modalities"), where
            ),
            supports_embeddings=_check_bool(
                "supports_embeddings", raw["supports_embeddings"], where
            ),
            experimental=_check_bool("experimental", raw["experimental"], where),
            maturity_surfaces=_check_maturity(raw["maturity_surfaces"], where),
            source_path=source,
            schema_version=version,
            settings_defaults=_validate_settings_defaults(raw.get("settings_defaults", {}), where),
            model_sources=_validate_model_sources(raw.get("model_sources", []), where),
            discovery=_validate_discovery(raw["discovery"], where)
            if raw.get("discovery") is not None
            else (),
            runtime_profiles=tuple(
                (name, _validate_profile(name, spec, where))
                for name, spec in sorted((raw.get("runtime_profiles") or {}).items())
            ),
            recipes=_validate_recipes(raw["recipes"], where)
            if raw.get("recipes") is not None
            else (),
            lifecycle=_validate_lifecycle(raw["lifecycle"], where)
            if raw.get("lifecycle") is not None
            else (),
            native_host=_check_kind("native_host", raw.get("native_host"), NATIVE_HOST_KINDS, where)
            if raw.get("native_host")
            else "",
            ui_kind=_check_kind("ui_kind", raw.get("ui_kind"), UI_KINDS, where)
            if raw.get("ui_kind")
            else "",
            launch_policy=_validate_launch_policy(raw["launch_policy"], where)
            if raw.get("launch_policy") is not None
            else (),
            build_options=_validate_build_options(raw["build_options"], where)
            if raw.get("build_options") is not None
            else (),
            protocol=_validate_protocol(raw["protocol"], where)
            if raw.get("protocol") is not None
            else (),
            extensions=_validate_extensions(raw["extensions"], where)
            if raw.get("extensions") is not None
            else (),
            request_defaults=_validate_request_defaults(raw["request_defaults"], where)
            if raw.get("request_defaults") is not None
            else (),
            catalog=_validate_catalog(raw["catalog"], where)
            if raw.get("catalog") is not None
            else (),
            upstream=_validate_upstream(raw["upstream"], where)
            if raw.get("upstream") is not None
            else (),
            data_dirs=_validate_data_dirs(raw["data_dirs"], where)
            if raw.get("data_dirs") is not None
            else (),
            model_compat=_validate_model_compat(raw["model_compat"], where)
            if raw.get("model_compat") is not None
            else (),
            release_assets=_validate_release_assets(raw["release_assets"], where)
            if raw.get("release_assets") is not None
            else (),
            build_defaults=_validate_build_defaults(raw["build_defaults"], where)
            if raw.get("build_defaults") is not None
            else (),
            source_build=_validate_source_build(raw["source_build"], where)
            if raw.get("source_build") is not None
            else (),
        )
    except KeyError as e:
        raise DefinitionError(f"{where}: missing required field {e}") from None
