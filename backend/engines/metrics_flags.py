"""Studio-managed engine flags for metrics and trustworthy logs.

These flags are not user configuration: Studio adds the metric flags and, for
audio.cpp, the framework log stream when the installed build's scanned
parameter catalog proves support. They make request usage, timing evidence,
and operational logs available without users having to discover each engine's
CLI surface, and custom arguments may not duplicate or override them. Flags
that redirect or silence the engine log are reserved as well.

The catalog gate is what makes "when available" safe for older builds: a flag
missing from the active version's scan is not passed.
"""

from __future__ import annotations

from typing import Iterable, Mapping

# Ordered by engine id. Keep this list to flags verified against the pinned
# upstream revisions and available in the active build's scanned catalog.
MANAGED_METRIC_ARGS: Mapping[str, tuple[str, ...]] = {
    "llama_cpp": ("--metrics", "--perf"),
    "ik_llama": ("--metrics", "--perf"),
    "unsloth_llama": ("--metrics", "--perf"),
    "1cat_vllm": (
        "--enable-prompt-tokens-details",
        "--enable-force-include-usage",
        "--kv-cache-metrics",
    ),
    "vllm": (
        "--enable-prompt-tokens-details",
        "--enable-force-include-usage",
        "--kv-cache-metrics",
    ),
    "sglang": (
        "--enable-cache-report",
        "--enable-metrics",
        "--stream-response-default-include-usage",
    ),
    "sglang_v100": (
        "--enable-cache-report",
        "--enable-metrics",
        "--stream-response-default-include-usage",
    ),
    "sglang_sxm2": (
        "--enable-cache-report",
        "--enable-metrics",
        "--stream-response-default-include-usage",
    ),
    "sglang_sm70": (
        "--enable-cache-report",
        "--enable-metrics",
        "--stream-response-default-include-usage",
    ),
}

# Log-producing flags Studio adds because they emit the framework progress and
# timing stream the UI expects to show. The scanned-catalog gate keeps this
# safe for builds that do not expose the flag.
MANAGED_LOG_ARGS: Mapping[str, tuple[str, ...]] = {
    "audio_cpp": ("--log",),
}

# Flags that redirect, truncate, or silence the engine log stream. Studio keeps
# them reserved so a saved model setting or custom argument cannot remove the
# output the log viewer and diagnostics depend on.
OWNED_LOG_ARGS: Mapping[str, tuple[str, ...]] = {
    "llama_cpp": ("--log-disable", "--log-file"),
    "ik_llama": ("--log-disable", "--log-file"),
    "unsloth_llama": ("--log-disable", "--log-file"),
    "vllm": ("--log-config-file", "--disable-log-stats"),
    "1cat_vllm": ("--log-config-file", "--disable-log-stats"),
    "sglang": ("--log-level", "--log-level-http", "--log-requests-target"),
    "sglang_v100": ("--log-level", "--log-level-http", "--log-requests-target"),
    "sglang_sxm2": ("--log-level", "--log-level-http", "--log-requests-target"),
    "sglang_sm70": ("--log-level", "--log-level-http", "--log-requests-target"),
    "lmdeploy": ("--log-level",),
    "audio_cpp": ("--log-file",),
}


def flags_of(meta: Mapping | None) -> set[str]:
    """Flag spellings declared by one catalog parameter."""
    if not isinstance(meta, Mapping):
        return set()
    flags = {str(name) for name in (meta.get("flags") or ()) if isinstance(name, str)}
    primary = meta.get("primary_flag")
    if isinstance(primary, str):
        flags.add(primary)
    return flags


def _declared_flags(param_index: Mapping[str, Mapping]) -> set[str]:
    flags: set[str] = set()
    for meta in param_index.values():
        flags.update(flags_of(meta))
    return flags


def strip_managed_config(
    config: Mapping[str, object], engine: str | None, param_index: Mapping[str, Mapping]
) -> dict:
    """Drop config keys whose emitted flag belongs to Studio's managed surface.

    Existing saved settings such as SGLang ``enable_metrics: true`` must not
    collide with the flags Studio itself appends. They are normalized away
    before emission; the managed pass then re-adds the canonical flag only when
    the active build supports it. Log destination and suppression flags are
    stripped too, because Studio must own the stream it displays. ``custom_args``
    is intentionally preserved so an explicit user flag is still rejected by
    the owned-flag check.
    """
    reserved = (
        set(MANAGED_METRIC_ARGS.get(str(engine or ""), ()))
        | set(MANAGED_LOG_ARGS.get(str(engine or ""), ()))
        | set(OWNED_LOG_ARGS.get(str(engine or ""), ()))
    )
    if not reserved:
        return dict(config)
    normalized = dict(config)
    for key, meta in (param_index or {}).items():
        if key == "custom_args":
            continue
        if flags_of(meta) & reserved:
            normalized.pop(key, None)
    return normalized


def managed_metric_args(engine: str | None, param_index: Mapping[str, Mapping]) -> list[str]:
    """Return managed flags supported by this engine's scanned catalog."""
    declared = _declared_flags(param_index or {})
    return [flag for flag in MANAGED_METRIC_ARGS.get(str(engine or ""), ()) if flag in declared]


def managed_log_args(engine: str | None, param_index: Mapping[str, Mapping]) -> list[str]:
    """Return log-producing flags supported by this engine's scanned catalog."""
    declared = _declared_flags(param_index or {})
    return [flag for flag in MANAGED_LOG_ARGS.get(str(engine or ""), ()) if flag in declared]


def managed_metric_flag_set(engine: str | None) -> frozenset[str]:
    """All metric flags Studio would own if the engine supports them.

    Used to reserve the surface even when a given build's catalog omits a flag,
    so a custom argument cannot silently change Studio's metrics contract.
    """
    return frozenset(MANAGED_METRIC_ARGS.get(str(engine or ""), ()))


def managed_log_flag_set(engine: str | None) -> frozenset[str]:
    """All log-producing flags Studio adds for this engine."""
    return frozenset(MANAGED_LOG_ARGS.get(str(engine or ""), ()))


def owned_log_flag_set(engine: str | None) -> frozenset[str]:
    """Log destination/suppression flags reserved from user control."""
    return frozenset(OWNED_LOG_ARGS.get(str(engine or ""), ()))


def owned_observability_flag_set(engine: str | None) -> frozenset[str]:
    """Every metric/log flag reserved from model config and custom arguments."""
    return (
        managed_metric_flag_set(engine) | managed_log_flag_set(engine) | owned_log_flag_set(engine)
    )


def flag_names(tokens: Iterable[str]) -> set[str]:
    return {str(token).split("=", 1)[0] for token in tokens if str(token).startswith("-")}
