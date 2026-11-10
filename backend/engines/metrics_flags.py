"""Studio-managed engine flags required for trustworthy metrics.

These flags are not user configuration: Studio adds them when the installed
build's scanned parameter catalog proves support. They make request usage and
operational metrics available without users having to discover each engine's
CLI surface, and custom arguments may not duplicate or override them.

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
    the active build supports it. ``custom_args`` is intentionally preserved so
    an explicit user flag is still rejected by the owned-flag check.
    """
    managed = set(MANAGED_METRIC_ARGS.get(str(engine or ""), ()))
    if not managed:
        return dict(config)
    normalized = dict(config)
    for key, meta in (param_index or {}).items():
        if key == "custom_args":
            continue
        if flags_of(meta) & managed:
            normalized.pop(key, None)
    return normalized


def managed_metric_args(engine: str | None, param_index: Mapping[str, Mapping]) -> list[str]:
    """Return managed flags supported by this engine's scanned catalog."""
    declared = _declared_flags(param_index or {})
    return [flag for flag in MANAGED_METRIC_ARGS.get(str(engine or ""), ()) if flag in declared]


def managed_metric_flag_set(engine: str | None) -> frozenset[str]:
    """All flags Studio would own if the engine supports them.

    Used to reserve the surface even when a given build's catalog omits a flag,
    so a custom argument cannot silently change Studio's metrics contract.
    """
    return frozenset(MANAGED_METRIC_ARGS.get(str(engine or ""), ()))


def flag_names(tokens: Iterable[str]) -> set[str]:
    return {str(token).split("=", 1)[0] for token in tokens if str(token).startswith("-")}
