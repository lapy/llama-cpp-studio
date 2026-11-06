"""Shared native GGUF launch mechanism (declared as ``launch_family``).

Every native server engine used to be compiled by one frozen function in
``launch_spec`` selected by engine-id set membership. Its argv/env composition
is ENGINE-FAMILY behaviour, not per-engine data: resolve the active binary and
its invocation path, compose the model source (cached GGUF file vs
``--hf-repo``), attach the mmproj/draft companions, keep the Studio port
placeholder and model alias, append the scanned structured flags and the
managed library path, and record file identities.

This module is that mechanism, selected by a DECLARATION
(``runtime_profiles.<name>.launch_family``) instead of by engine id. The
profile supplies only what Studio still chooses per engine: the port flag and
an optional argv prefix. Everything else is family vocabulary and stays here.

Imports are deliberately FUNCTION-LOCAL: the launch harness patches these
resolver symbols exactly as it patches the frozen compiler this replaces
(``backend.proxy.commands._resolve_cuda_library_path``,
``_active_engine_param_index``, ``resolve_gguf_model_path``, the llama-family
binary/invocation resolvers), so a module-level ``from ... import`` would bind
copies and silently defeat those seams.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

LAUNCH_FAMILY = "gguf_llama_server"


def compile_gguf_family_launch(
    *,
    definition: Any,
    spec: Mapping[str, Any],
    model: Any,
    config: Mapping[str, Any],
    stable_id: str,
    engine: str,
):
    """Compile the declared GGUF-family launch into a ``LaunchSpec``.

    Behaviour-equivalent to the frozen compiler this replaces: same resolution
    order, same argv order (prefix, model tokens, port, alias, mmproj, draft,
    structured), same typed errors, same identities and environment. The active
    install row is read through ``launch_spec._active_row`` (the interpreter's
    tolerant read) instead of a bare store call.
    """
    from backend import data_store
    from backend.engines.llama_cpp.resolve import (
        SERVER_BINARY_NAME,
        get_active_binary_path_for_engine,
    )
    from backend.engines.llama_cpp.server_exec import resolve_llama_server_invocation_paths
    from backend.proxy.commands import (
        _active_engine_param_index,
        _emit_structured_tokens,
        _resolve_cuda_library_path,
        _resolve_draft_companion,
        _resolve_llama_model_source,
        _resolve_mmproj_path,
    )
    from backend.proxy.launch_spec import (
        _OWNED_PORT_FLAGS,
        PORT_PLACEHOLDER,
        LaunchCompileError,
        LaunchSpec,
        _active_row,
        _compile_env,
        _file_identity,
        _install_identity,
        _reject_owned_flags,
    )
    from backend.proxy.profile_launch import profile_owned_flags

    active = _active_row(engine)
    binary = get_active_binary_path_for_engine(data_store.get_store(), engine)
    if not binary:
        raise LaunchCompileError(
            f"No active {engine} {SERVER_BINARY_NAME} binary configured",
            field="executable",
        )
    if not os.path.isabs(binary):
        binary = os.path.join("/app", binary)
    if not os.path.isfile(binary):
        raise LaunchCompileError(
            f"{SERVER_BINARY_NAME} binary not found at: {binary}",
            field="executable",
        )
    executable, cwd = resolve_llama_server_invocation_paths(binary)

    model_path, hf_repo, hf_id = _resolve_llama_model_source(model)
    if not model_path and not hf_repo:
        raise LaunchCompileError(
            "Model path could not be resolved from HF metadata or runtime overlay",
            field="model",
        )

    launch_args = dict(spec.get("launch_args") or {})

    def arg(key: str, default: str) -> str:
        return str(launch_args.get(key) or default)

    prefix = [str(token) for token in (spec.get("argv_prefix") or ())]
    argv: List[Any] = list(prefix)
    identities: Dict[str, Dict[str, Any]] = {}
    if hf_repo:
        argv.extend([arg("hf_repo", "--hf-repo"), hf_repo])
    else:
        argv.extend([arg("model", "--model"), str(model_path or "")])
        identities["model"] = _file_identity(model_path)
    port_flag = str((spec.get("port_binding") or {}).get("flag") or "--port")
    argv.extend([port_flag, dict(PORT_PLACEHOLDER), arg("alias", "--alias"), stable_id])

    mmproj = _resolve_mmproj_path(model, hf_id, hf_repo)
    if mmproj:
        argv.extend([arg("mmproj", "--mmproj"), mmproj])
        identities["mmproj"] = _file_identity(mmproj)
    draft, draft_spec = _resolve_draft_companion(model, hf_id, hf_repo)

    structured = _emit_structured_tokens(
        dict(config),
        engine=engine,
        param_index=_active_engine_param_index(engine),
    )
    # Custom arguments may not repeat what the DECLARATION owns: the port flag
    # and the prefix's own flags. Studio additionally reserves the port/host
    # surface everywhere (it chooses the port), exactly as before.
    owned = set(profile_owned_flags(definition)) | set(_OWNED_PORT_FLAGS)
    owned |= {token.split("=", 1)[0] for token in prefix if token.startswith("-")}
    _reject_owned_flags(structured, owned=owned)

    if draft:
        argv.extend([arg("draft", "--model-draft"), draft])
        identities["draft"] = _file_identity(draft)
        spec_type_flag = arg("spec_type", "--spec-type")
        if spec_type_flag not in structured:
            argv.extend([spec_type_flag, draft_spec or "draft-mtp"])
    argv.extend(structured)

    library = _resolve_cuda_library_path(cwd)
    required_ld = [part for part in str(library).split(":") if part]
    env = _compile_env(
        config,
        engine=engine,
        required_ld=required_ld,
        required_path=[],
        engine_defaults={},
        active=active,
    )
    identities["executable"] = _file_identity(executable)
    return LaunchSpec(
        model_id=stable_id,
        engine_id=engine,
        engine_install_id=_install_identity(active, executable),
        executable=executable,
        argv=argv,
        cwd=cwd,
        env=env,
        file_identities=identities,
    )


def launch_family_supported(family: Optional[str]) -> bool:
    """True when this module implements the named declared launch family."""
    return str(family or "") == LAUNCH_FAMILY
