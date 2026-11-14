"""Profile-driven launch interpreter.

Engines whose launch behavior lives in
``backend/engines/definitions/*.yaml`` compile their ``LaunchSpec`` here
through the schema-v2 ``runtime_profiles`` evaluator
(``backend/engines/definitions/profiles.py``). Every Python-venv engine is
definition-owned and profile-driven; a profile may instead DECLARE a shared
launch mechanism (``launch_family``), which this interpreter delegates to
(``backend/proxy/gguf_launch.py``) so family behaviour — model-source
composition, companions, library path — is composed once, from a declaration
rather than from engine-id set membership. Family mechanisms without a
declaration (audio sidecars) keep their frozen compiler in
``backend/proxy/launch_spec.py`` (recorded departure).

Engine-id literals: none — behavior is keyed on definition presence and the
``executable.kind`` / ``model_binding.source`` vocabulary only.

Mirrored legacy semantics (byte-compatible shapes): executable resolution
mirrors the venv console-script and venv interpreter resolvers; cwd/env
composition reuses ``launch_spec._compile_env``; a missing ``env.required``
mirrors the managed-CUDA preflight (typed
``LaunchCompileError(field=NAME)``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.definitions.profiles import (
    PORT_SENTINEL,
    compile_profile_argv,
)
from backend.engines.definitions.schema import DefinitionError, EngineDefinition

_REGISTRY = None

# Legacy relative-path resolver base. Tests monkeypatch this instead of writing
# to a real /app, which does not exist (or is not writable) on CI runners.
APP_ROOT = "/app"

_VENV_EXECUTABLE_KINDS = frozenset({"venv_console_script", "venv_python"})


def _definitions():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_default_engine_definitions()
    return _REGISTRY


def profile_launch_definition(engine: Optional[str]) -> Optional[EngineDefinition]:
    """The definition backing a profile-driven launch, or None.

    A definition without a usable default profile returns None so the
    dispatcher falls through to its family compiler (native GGUF/audio) or to
    the historical "No runtime adapter registered" error for an unknown id.
    """
    if not engine:
        return None
    definition = _definitions().get(engine)
    if definition is None:
        return None
    try:
        definition.profile("")
    except DefinitionError:
        return None
    return definition


def definition_digest_for(engine: Optional[str]) -> Optional[str]:
    """Canonical digest of the shipped definition backing an engine, or None.

    A definition-owned engine's digest participates in launch revisions and
    manifests even when its argv is composed by a family compiler rather than a
    runtime profile. This keeps audio.cpp's launch identity tied to its YAML
    definition instead of relying only on the installed executable identity.
    """
    if not engine:
        return None
    definition = _definitions().get(engine)
    if definition is None:
        return None
    return definition.definition_digest


def profile_model_source(definition: EngineDefinition) -> str:
    spec = definition.profile("")
    if str(spec.get("launch_family") or ""):
        # A family mechanism composes the model SOURCE itself (file vs repo);
        # the proxy's model name stays the Studio model id, which is the alias
        # the mechanism emits. Reporting a binding source here would silently
        # rewrite every proxied request's model name.
        return "model_id"
    return str((spec.get("model_binding") or {}).get("source") or "huggingface_id")


def profile_owned_flags(definition: EngineDefinition) -> frozenset:
    """Flags the profile itself emits, so custom args may not repeat them.

    Derived from the YAML only: the model binding (when flag-mode), the
    port binding, and every flag-shaped token in ``argv_prefix`` (e.g. a
    declared ``--host``). Values in the prefix are not flags and are
    skipped.
    """
    spec = definition.profile("")
    owned = set()
    binding = spec.get("model_binding") or {}
    if str(binding.get("mode") or "") == "flag" and binding.get("flag"):
        owned.add(str(binding["flag"]))
    port = spec.get("port_binding") or {}
    if port.get("flag"):
        owned.add(str(port["flag"]))
    for token in spec.get("argv_prefix") or ():
        text = str(token)
        if text.startswith("-"):
            owned.add(text.split("=", 1)[0])
    return frozenset(owned)


def _resolve_executable(
    spec: Mapping[str, Any],
    engine: str,
    active: Mapping[str, Any],
) -> str:
    from backend.proxy.launch_spec import LaunchCompileError

    executable = spec.get("executable") or {}
    kind = str(executable.get("kind") or "")
    name = str(executable.get("name") or "")
    venv_path = str(active.get("venv_path") or "")
    if not venv_path:
        # Legacy rows (and pending source builds) may only record install_dir;
        # the venv lives at its `venv` child.
        install_dir = str(active.get("install_dir") or "")
        if install_dir:
            venv_path = os.path.join(install_dir, "venv")
    # Legacy resolvers normalize relative venv paths against the app root
    # (/app) before probing, and require the directory to exist.
    if venv_path and not os.path.isabs(venv_path):
        venv_path = os.path.join(APP_ROOT, venv_path)
    if venv_path and not os.path.isdir(venv_path):
        venv_path = ""
    exe = ""
    if not exe and kind == "venv_console_script" and venv_path and name:
        exe = os.path.join(venv_path, "bin", name)
    elif not exe and kind == "venv_python" and venv_path:
        exe = os.path.join(venv_path, "bin", name or "python")
    elif not exe and kind == "native_artifact" and name:
        exe = str(active.get(name) or "")
    # Legacy parity: a file that exists but is not executable is unusable.
    if not exe or not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        raise LaunchCompileError(
            f"{engine} executable unavailable",
            field="executable",
        )
    return exe


def compile_profile_launch(
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    stable_id: str,
    engine: str,
):
    """LaunchSpec from the engine's default runtime profile, or None.

    Returns None when the engine has no launchable definition, letting
    the dispatcher fall through to the legacy "No runtime adapter
    registered" error with its exact historical bytes.
    """
    from backend.engines.metrics_flags import (
        managed_log_args,
        managed_metric_args,
        owned_observability_flag_set,
        strip_managed_config,
    )
    from backend.proxy.commands import (
        _ALLOWED_NONCANONICAL_KEYS,
        _active_engine_param_index,
        _emit_structured_tokens,
        _model_attr,
    )
    from backend.proxy.launch_spec import (
        _COMPOSITIONAL_KEYS,
        _OWNED_PORT_FLAGS,
        PORT_PLACEHOLDER,
        LaunchCompileError,
        LaunchSpec,
        _active_row,
        _compile_env,
        _file_identity,
        _install_identity,
        _reject_owned_flags,
        _split_paths,
    )

    definition = profile_launch_definition(engine)
    if definition is None:
        return None
    spec = definition.profile("")

    family = str(spec.get("launch_family") or "")
    if family:
        # Declared mechanism: the family composes executable/cwd/argv/env/
        # identities, so this interpreter must not also evaluate a binding.
        from backend.proxy import gguf_launch

        if gguf_launch.launch_family_supported(family):
            return gguf_launch.compile_gguf_family_launch(
                definition=definition,
                spec=spec,
                model=model,
                config=config,
                stable_id=stable_id,
                engine=engine,
            )
        from backend.proxy.launch_spec import LaunchCompileError

        raise LaunchCompileError(
            f"{engine} declares no implementation for launch family {family!r}",
            field="launch_family",
        )

    active = _active_row(engine)
    executable = _resolve_executable(spec, engine, active)

    source = profile_model_source(definition)
    if source == "local_path":
        model_ref = _model_attr(model, "local_path") or _model_attr(model, "path")
    else:
        model_ref = _model_attr(model, "huggingface_id")
    if not model_ref:
        raise LaunchCompileError(
            f"{engine} model must have {source}",
            field="model",
        )

    try:
        param_index = _active_engine_param_index(engine)
    except Exception:
        param_index = {}
    switch = spec.get("runtime_switch") or {}
    prefix_tokens = [str(token) for token in (spec.get("argv_prefix") or ())]
    structured_config = dict(config)
    if switch:
        config_key = str(switch.get("config_key") or "")
        selected = str(structured_config.get(config_key) or switch.get("default") or "")
        variants = switch.get("variants") or {}
        variant = variants.get(selected) or {}
        prefix_tokens = [str(token) for token in (variant.get("argv_prefix") or prefix_tokens)]
        catalog = str(variant.get("scanned_catalog") or "")
        if catalog:
            # A variant can re-index parameters from its OWN scanned catalog
            # (SGLang diffusion); the default runtime's keys are then kept out
            # of argv unless the variant catalog also declares them.
            from backend.engines.params import param_index_from_entry
            from backend.proxy.commands import _active_engine_entry

            try:
                entry = _active_engine_entry(engine) or {}
            except Exception:
                entry = {}
            catalog_profile = (entry.get("runtime_profiles") or {}).get(catalog) or {}
            if catalog_profile.get("scan_error") or not catalog_profile.get("sections"):
                reason = catalog_profile.get("scan_error") or (
                    f"{catalog} parameters have not been scanned"
                )
                raise LaunchCompileError(
                    f"{definition.label} {catalog} is unavailable: {reason}. "
                    "Rescan engine parameters after installing the fork's "
                    "diffusion dependencies.",
                    field=config_key,
                )
            inactive = set(param_index)
            param_index = param_index_from_entry(catalog_profile)
            structured_config = {
                key: value
                for key, value in structured_config.items()
                if key not in inactive or key in param_index or key in _ALLOWED_NONCANONICAL_KEYS
            }
    # Studio owns the operational-metrics surface. Normalize saved settings
    # that would emit an owned flag (for example SGLang enable_metrics: true)
    # BEFORE emission, then append the canonical flag only when the active
    # build's scanned catalog proves support. Custom arguments are preserved so
    # an explicit user override is still rejected.
    managed_flags = managed_metric_args(engine, param_index)
    managed_log_flags = managed_log_args(engine, param_index)
    structured_config = strip_managed_config(structured_config, engine, param_index)
    structured = _emit_structured_tokens(
        structured_config,
        engine=engine,
        param_index=param_index,
    )
    prefix_tokens.extend(managed_flags)
    prefix_tokens.extend(managed_log_flags)
    # Custom arguments may not repeat what the PROFILE (i.e. the YAML the
    # engine ships) owns: bindings, prefix flags. Studio additionally reserves
    # the port/host CLI surface universally — it chooses the port, so no
    # engine may re-declare it through custom args.
    owned = set(profile_owned_flags(definition)) | set(_OWNED_PORT_FLAGS)
    owned |= owned_observability_flag_set(engine)
    owned |= {token.split("=", 1)[0] for token in prefix_tokens if token.startswith("-")}
    _reject_owned_flags(structured, owned=owned)

    binding = spec.get("model_binding") or {}
    port_binding = spec.get("port_binding") or {}
    argv_tokens: List[Any] = list(prefix_tokens)
    if str(binding.get("mode") or "positional") == "positional":
        argv_tokens.append(str(model_ref))
    else:
        argv_tokens.extend([str(binding.get("flag") or ""), str(model_ref)])
    argv_tokens.extend([str(port_binding.get("flag") or ""), PORT_SENTINEL])
    argv_tokens.extend(structured)
    argv: List[Any] = [
        dict(PORT_PLACEHOLDER) if token == PORT_SENTINEL else token for token in argv_tokens
    ]

    env_spec = spec.get("env") or {}
    required_env: List[tuple] = []
    for entry in env_spec.get("required") or ():
        if isinstance(entry, Mapping):
            required_env.append((str(entry.get("name") or ""), str(entry.get("error") or "")))
        else:
            required_env.append((str(entry), ""))
    engine_defaults = {
        str(key): str(value) for key, value in (env_spec.get("defaults") or {}).items()
    }
    bin_dir = os.path.dirname(executable)
    cached_vars = [str(name) for name in (env_spec.get("caches") or ())]
    if cached_vars:
        # Managed caches live inside the ACTIVE install so a rebuild starts
        # empty and two versions never share compiled kernels. The installer
        # points the same variables at the same paths, which is what lets a
        # server start reuse what the smoke/precompile step already produced.
        from backend.engines.caches import cache_environment, ensure_cache_dirs

        install_dir = str(active.get("install_dir") or "")
        if not install_dir:
            install_dir = str(active.get("venv_path") or "")
            if install_dir:
                install_dir = os.path.dirname(install_dir)
        if install_dir:
            ensure_cache_dirs(install_dir, cached_vars)
            for name, path in cache_environment(install_dir, cached_vars).items():
                engine_defaults.setdefault(name, path)
    required_path: List[str] = []
    kind = str((spec.get("executable") or {}).get("kind") or "")
    if bin_dir and (kind in _VENV_EXECUTABLE_KINDS or env_spec.get("prepend_bin_to_path")):
        required_path = [bin_dir]
    cwd = None
    if str(env_spec.get("cwd") or "none") == "venv_root" and bin_dir:
        venv_dir = os.path.dirname(bin_dir)
        cwd = venv_dir if venv_dir and os.path.isdir(venv_dir) else None

    required_ld: List[str] = []
    cuda_spec = env_spec.get("cuda")
    if cuda_spec:
        # Studio-managed CUDA environment for the ACTIVE install, composed
        # exactly like the venv family's compilers do (LD/PATH split, rest as
        # engine defaults). A string `cuda` value is the default version used
        # when the active row carries none (the fork's 12.8 fallback).
        cuda_version = str(active.get("cuda_version") or "")
        if not cuda_version and isinstance(cuda_spec, str):
            cuda_version = cuda_spec
        try:
            from backend.cuda_installer import get_cuda_installer

            cuda_env = get_cuda_installer().get_cuda_env(cuda_version or None)
        except Exception:
            cuda_env = {}
        required_ld = _split_paths(cuda_env.get("LD_LIBRARY_PATH"))
        required_path = _split_paths(cuda_env.get("PATH")) + required_path
        engine_defaults.update(
            {key: value for key, value in cuda_env.items() if key not in _COMPOSITIONAL_KEYS}
        )

    env = _compile_env(
        dict(config),
        engine=engine,
        required_ld=required_ld,
        required_path=required_path,
        engine_defaults=engine_defaults,
        active=active,
    )
    # Validate required variables against the environment the process will
    # ACTUALLY receive. ``registry._start`` builds it as
    # ``os.environ + env.set - env.unset``; validating against a different
    # mapping let a required variable pass here and be absent at launch.
    final_env = dict(os.environ)
    final_env.update({str(key): str(value) for key, value in env.set.items()})
    for key in env.unset:
        final_env.pop(key, None)
    missing = [(name, message) for name, message in required_env if not final_env.get(name)]
    if missing:
        name, message = missing[0]
        raise LaunchCompileError(
            message or f"{engine} requires {name}",
            field=name,
        )
    return LaunchSpec(
        model_id=stable_id,
        engine_id=engine,
        engine_install_id=_install_identity(active, executable),
        executable=executable,
        argv=argv,
        cwd=cwd,
        env=env,
        file_identities={"executable": _file_identity(executable)},
    )


def profile_executable(engine: str, active: Mapping[str, Any]) -> str:
    """Executable path the engine's default profile resolves for ``active``.

    Same resolver the launch compiler uses, so discovery scans the binary
    the launch would spawn. Raises ``LaunchCompileError(field="executable")``
    when the row cannot supply it. Definition-backed engines only (generated
    engines keep their Python-owned binary resolution).
    """
    from backend.proxy.launch_spec import LaunchCompileError

    definition = profile_launch_definition(engine)
    if definition is None:
        raise LaunchCompileError(
            f"{engine} has no runtime profile",
            field="executable",
        )
    spec = definition.profile("")
    return _resolve_executable(spec, engine, active)


def profile_variant_executable(
    executable: Mapping[str, Any], engine: str, active: Mapping[str, Any]
) -> str:
    """Resolve one declared discovery-probe executable.

    Reuses the launch executable resolver (same existence/executable checks and
    error type) so a scan probes a binary the launch would actually accept.
    """
    return _resolve_executable({"executable": dict(executable)}, engine, active)


def profile_scan_contract(engine: Optional[str]) -> Optional[dict]:
    """The YAML-declared discovery contract for a definition-backed engine.

    Returns ``None`` for ids with no shipped definition. Returns an empty dict
    when the definition exists but declares no ``discovery`` block, so callers
    can emit a typed "definition must declare discovery" error instead of
    guessing. ``mode: adapter`` explicitly keeps a shared scanner-kind adapter.

    ``probe_variants`` lets one definition declare ordered executable/argv
    combinations (e.g. a console script with a module-invocation fallback);
    ``cuda_env`` composes the managed CUDA environment for the probe.
    """
    if not engine:
        return None
    definition = _definitions().get(engine)
    if definition is None:
        return None
    discovery = dict(definition.discovery or ())
    if not discovery:
        return {}
    parser = str(discovery.get("parser") or "")
    prefix = [str(item) for item in (discovery.get("probe_argv_prefix") or [])]
    variants = [dict(item) for item in (discovery.get("probe_variants") or ())]
    env_spec = dict(discovery.get("env") or {})
    probes = {
        str(name): {
            "parser": str(dict(probe).get("parser") or ""),
            "argv_prefix": [str(item) for item in (dict(probe).get("argv_prefix") or [])],
            "cwd": str(dict(probe).get("cwd") or discovery.get("cwd") or "bin_dir"),
            "unavailable_marker": str(dict(probe).get("unavailable_marker") or ""),
        }
        for name, probe in dict(discovery.get("runtime_probes") or {}).items()
    }
    return {
        "mode": str(discovery.get("mode") or "probes"),
        "parser": parser,
        "primary_runtime": str(discovery.get("primary_runtime") or ""),
        "probe_argv_prefix": prefix,
        "probe_variants": variants,
        "cwd": str(discovery.get("cwd") or "bin_dir"),
        "cuda_env": discovery.get("cuda_env", False),
        "env_defaults": {
            str(key): str(value) for key, value in dict(env_spec.get("defaults") or {}).items()
        },
        "env_required": [
            (
                {"name": str(item.get("name") or ""), "error": str(item.get("error") or "")}
                if isinstance(item, Mapping)
                else {"name": str(item), "error": ""}
            )
            for item in (env_spec.get("required") or ())
        ],
        "runtime_probes": probes,
    }
