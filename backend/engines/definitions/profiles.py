"""Schema v2 runtime-profile evaluation (shadow mode).

Compiles a definition's named runtime profile into the argv/env shape the
Phase 6 launch compiler will emit. The semantics here intentionally mirror
``backend/proxy/launch_spec.py`` (module invocation + model binding + owned
port flag) so equivalence tests can prove the declarative form reproduces
today's launch argv before any runtime code is replaced.

Nothing in ``backend/`` imports this module at runtime yet — enforced by
``backend/tests/test_engine_definitions_equivalence.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .schema import DefinitionError, EngineDefinition

PORT_SENTINEL = "{{port}}"


def compile_profile_argv(
    definition: EngineDefinition,
    *,
    model_ref: str,
    profile: str = "",
    port: str = PORT_SENTINEL,
    extra_argv: Sequence[str] = (),
) -> List[str]:
    """argv for one launch: argv_prefix + model binding + owned port flag.

    Model and port flags are engine-owned (mirrors launch_spec's
    ``_OWNED_PORT_FLAGS`` rejection of user-supplied duplicates); user
    parameters may only arrive through ``extra_argv``.
    """
    spec = definition.profile(profile)
    family = str(spec.get("launch_family") or "")
    if family:
        # Family profiles name a shared launch mechanism instead of declaring
        # an argv skeleton; this shadow evaluator cannot model one, and saying
        # so is better than a KeyError on the missing model_binding.
        raise DefinitionError(
            f"{definition.id}: profile {profile or 'default'!r} declares launch "
            f"family {family!r}, which is composed by the launch interpreter "
            "(backend/proxy), not by this evaluator"
        )
    if not str(model_ref or "").strip():
        raise DefinitionError(
            f"{definition.id}: profile {profile or 'default'!r} requires a model reference"
        )
    argv: List[str] = list(spec.get("argv_prefix", ()))
    binding = spec["model_binding"]
    if binding["mode"] == "positional":
        argv.append(str(model_ref))
    else:
        argv.extend([binding["flag"], str(model_ref)])
    argv.extend([spec["port_binding"]["flag"], str(port)])
    argv.extend(str(token) for token in extra_argv)
    return argv


def compile_profile_env(
    definition: EngineDefinition,
    provided: Mapping[str, str],
    *,
    profile: str = "",
) -> Dict[str, Any]:
    """Resolve a profile's env requirements against a provided environment.

    Returns merged defaults plus the required variables missing from
    ``provided``; the future executor turns ``missing`` into a typed
    preflight error (like launch_spec's SGLang-V100 CUDA_HOME check).
    """
    spec = definition.profile(profile)
    env = spec.get("env") or {}
    defaults = dict(env.get("defaults") or {})
    merged = {**defaults, **dict(provided)}
    missing = [name for name in env.get("required", ()) if not merged.get(name)]
    return {
        "env": merged,
        "missing_required": missing,
        "prepend_bin_to_path": bool(env.get("prepend_bin_to_path", False)),
        "cwd": (env.get("cwd") or "none"),
    }
