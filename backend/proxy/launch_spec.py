"""Pure launch and proxy specs for every registered inference engine.

The compiler emits structured argv before any shell rendering. Placeholders are
typed objects, never interpolated strings. Preview and manifest publication
share this output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from backend.engines.registry import VALID_ENGINE_IDS
from backend.feature_flags import launch_manifests_enabled

SCHEMA_VERSION = 1
PORT_PLACEHOLDER = {"runtime": "port"}
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_NAME_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|PRIVATE_KEY)",
    re.IGNORECASE,
)
_STUDIO_PREFIX = "LLAMA_STUDIO_"
_COMPOSITIONAL_KEYS = frozenset({"PATH", "LD_LIBRARY_PATH"})
_BASELINE_KEYS = (
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
_OWNED_PORT_FLAGS = frozenset({"--port", "--server-port", "--host"})


def _launch_policy(engine: str) -> Dict[str, Any]:
    """The engine's declared launch policy (R5), read from its spec.

    Generated engines declare it in ``backend/engines/registry.py`` and the
    exporter writes it into their definitions; authoritative YAML engines
    declare it in their own file. Absent means "no constraints".
    """
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(str(engine or ""))
    return dict(spec.launch_policy) if spec is not None else {}


def _cpu_supported(engine: str) -> bool:
    return bool(_launch_policy(engine).get("cpu_supported", True))


def _gpu_only_invariants(engine: str) -> Tuple[str, ...]:
    raw = _launch_policy(engine).get("gpu_only_invariants") or ()
    return tuple(str(item) for item in raw)


ArgvItem = Union[str, Dict[str, str]]


class LaunchCompileError(ValueError):
    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


@dataclass
class EnvSpec:
    set: Dict[str, str] = field(default_factory=dict)
    unset: List[str] = field(default_factory=list)


@dataclass
class LaunchSpec:
    model_id: str
    engine_id: str
    engine_install_id: str
    executable: str
    argv: List[ArgvItem]
    cwd: Optional[str]
    env: EnvSpec
    artifacts: Dict[str, Any] = field(default_factory=dict)
    file_identities: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def document(self) -> Dict[str, Any]:
        artifact_digests = {
            name: artifact_digest(payload) for name, payload in sorted(self.artifacts.items())
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "engine_id": self.engine_id,
            "engine_install_id": self.engine_install_id,
            "executable": self.executable,
            "argv": self.argv,
            "cwd": self.cwd,
            "env": {"set": self.env.set, "unset": list(self.env.unset)},
            "artifacts": artifact_digests,
            "file_identities": self.file_identities,
        }


@dataclass
class ProxyModelSpec:
    model_id: str
    catalog_id: str
    engine_id: str
    aliases: List[str]
    filters: Optional[Dict[str, Any]]
    use_model_name: Optional[str]
    health_endpoint: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


@dataclass
class CompiledModel:
    launch: LaunchSpec
    proxy: ProxyModelSpec
    revision: str
    engine_command: str
    launcher_command: str
    # Canonical digest of the authoritative definition that produced the
    # launch ("" for the generated engines, whose compilers are Python-owned).
    # It participates in the revision so a YAML/profile change is visible to
    # manifests, published pointers and staleness checks.
    profile_digest: str = ""


def compiler_engine_ids() -> frozenset:
    return frozenset(VALID_ENGINE_IDS)


def artifact_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def launch_revision(spec: LaunchSpec, *, profile_digest: str = "") -> str:
    """Revision hash of a compiled launch.

    ``profile_digest`` is added when the engine is backed by a YAML definition,
    including audio.cpp, whose argv is composed by the frozen audio family
    compiler. Any definition edit therefore changes the published revision and
    manifest identity.
    """
    body = spec.document()
    if profile_digest:
        body["profile_digest"] = profile_digest
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def manifest_document(
    spec: LaunchSpec, revision: str, *, profile_digest: str = ""
) -> Dict[str, Any]:
    """On-disk manifest. ``revision`` is excluded from the hashed body."""
    body = spec.document()
    body["artifacts_inline"] = spec.artifacts
    body["revision"] = revision
    if profile_digest:
        body["profile_digest"] = profile_digest
    return body


def _launch_profile_digest(engine: str) -> str:
    """Definition digest that owns this engine's launch, or "" (legacy)."""
    from backend.proxy import profile_launch

    return profile_launch.definition_digest_for(engine) or ""


def redact_env_value(name: str, value: str) -> str:
    if _SECRET_NAME_RE.search(name):
        return "***"
    return value


def redact_env_map(values: Mapping[str, str]) -> Dict[str, str]:
    return {key: redact_env_value(key, value) for key, value in values.items()}


def format_argv(executable: str, argv: Sequence[ArgvItem]) -> str:
    from backend.proxy.commands import _shell_join

    tokens = [executable]
    for item in argv:
        if isinstance(item, dict) and item.get("runtime") == "port":
            tokens.append("${PORT}")
        elif isinstance(item, dict) and item.get("artifact"):
            tokens.append("${artifact:" + str(item["artifact"]) + "}")
        else:
            tokens.append(str(item))
    return _shell_join(tokens)


def stable_launcher_argv(model_id: str) -> List[str]:
    from backend.proxy.manifests import model_runtime_dir

    launcher = os.path.abspath(os.path.join(os.path.dirname(__file__), "launcher.py"))
    return [
        _interpreter(),
        launcher,
        "--manifest-root",
        model_runtime_dir(model_id),
        "--port",
        "${PORT}",
    ]


def _interpreter() -> str:
    import sys

    return os.path.abspath(sys.executable)


def stable_launcher_command(model_id: str) -> str:
    from backend.proxy.commands import _shell_join

    return _shell_join(stable_launcher_argv(model_id))


def compile_model_runtime(model: Mapping[str, Any]) -> CompiledModel:
    from backend import data_store
    from backend.proxy.commands import (
        _model_attr,
        request_filters_and_aliases,
        model_capabilities,
    )
    from backend.models.config import effective_model_config_from_raw

    stable_id = data_store.resolve_model_id(model)
    if not stable_id:
        raise LaunchCompileError("Model has no Studio proxy id", field="model_id")
    config = effective_model_config_from_raw(model.get("config"))
    engine = str(config.get("engine") or "")
    if engine not in VALID_ENGINE_IDS:
        raise LaunchCompileError(
            f"No runtime adapter registered for engine {engine!r}",
            field="engine",
        )
    launch = _compile_launch(model, config, stable_id, engine)
    filters, aliases = request_filters_and_aliases(
        stable_id=stable_id,
        config=dict(config),
        model=model,
    )
    use_model_name = _use_model_name(model, config, stable_id, engine)
    proxy = ProxyModelSpec(
        model_id=stable_id,
        catalog_id=str(_model_attr(model, "id") or stable_id),
        engine_id=engine,
        aliases=list(aliases or []),
        filters=filters,
        use_model_name=use_model_name,
        capabilities=model_capabilities(engine, model=dict(model), config=dict(config)),
    )
    profile_digest = _launch_profile_digest(engine)
    revision = launch_revision(launch, profile_digest=profile_digest)
    return CompiledModel(
        launch=launch,
        proxy=proxy,
        revision=revision,
        engine_command=format_argv(launch.executable, launch.argv),
        launcher_command=stable_launcher_command(stable_id) if launch_manifests_enabled() else "",
        profile_digest=profile_digest,
    )


def annotate_preview(model: Mapping[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the native launch and routing contract to the command preview."""
    if not payload.get("ok"):
        return payload
    try:
        compiled = compile_model_runtime(model)
    except Exception as exc:
        payload["engine_preview_error"] = str(exc)
        return payload
    payload["engine_command"] = compiled.engine_command
    payload["engine_cwd"] = compiled.launch.cwd
    payload["engine_env"] = [
        f"{key}={redact_env_value(key, value)}"
        for key, value in sorted(compiled.launch.env.set.items())
    ]
    payload["engine_env_unset"] = list(compiled.launch.env.unset)
    payload["launch_revision"] = compiled.revision
    payload["engine_id"] = compiled.launch.engine_id
    if launch_manifests_enabled():
        payload["launcher_command"] = compiled.launcher_command
    try:
        from backend.proxy.manifests import LaunchManifestStore

        published = LaunchManifestStore().read_pointer(compiled.proxy.model_id)
        payload["published_revision"] = published.revision if published else None
    except Exception:
        payload["published_revision"] = None
    return payload


def _use_model_name(model, config, stable_id: str, engine: str) -> Optional[str]:
    from backend.proxy.commands import _model_attr

    if engine == "audio_cpp":
        return stable_id
    # Profile-backed engines follow the profile's model-binding source; engines
    # without a profile (native GGUF) keep the historical stable-id behaviour.
    from backend.proxy import profile_launch

    definition = profile_launch.profile_launch_definition(engine)
    if definition is not None:
        if profile_launch.profile_model_source(definition) == "huggingface_id":
            value = _model_attr(model, "huggingface_id")
            return str(value) if value else None
        return stable_id
    return stable_id


def _compile_launch(model, config: Mapping[str, Any], stable_id: str, engine: str) -> LaunchSpec:
    # Engines whose definition declares a runtime profile compile through the
    # profile interpreter FIRST (every Python-venv engine does, and a native
    # GGUF engine does once its profile declares a launch family). Engines
    # without a declaration keep the byte-frozen family compilers below.
    profiled = _compile_profile_launch(model, config, stable_id, engine)
    if profiled is not None:
        return profiled
    if engine == "audio_cpp":
        return _compile_audio(model, config, stable_id, engine)
    raise LaunchCompileError(
        f"No runtime adapter registered for engine {engine!r}",
        field="engine",
    )


def _compile_profile_launch(
    model, config: Mapping[str, Any], stable_id: str, engine: str
) -> Optional[LaunchSpec]:
    """Profile-driven launch for definition-backed engines (launch campaign R1).

    Called from two places: the GGUF branch (before the legacy GGUF
    implementation, so an engine whose profile declares a launch family — or
    any YAML-only engine that declares gguf artifacts — is profile-launched)
    and the tail of the dispatch chain. Native GGUF engines stay on the legacy
    compiler until their definition declares a family; definition-less ids
    return None so the historical "No runtime adapter registered" error keeps
    its bytes.
    """
    from backend.proxy import profile_launch

    return profile_launch.compile_profile_launch(model, config, stable_id, engine)


def _compile_audio(model, config, stable_id: str, engine: str = "") -> LaunchSpec:
    from backend import data_store

    engine = str(engine or config.get("engine") or "").strip()
    runtime_module = __import__(f"backend.engines.{engine}.runtime", fromlist=["runtime"])
    build_runtime = getattr(runtime_module, f"build_{engine}_runtime")

    runtime = build_runtime(
        data_store.get_store(),
        dict(model),
        dict(config),
        stable_id,
    )
    raw_argv = list(runtime.get("cmd_argv") or [])
    if not raw_argv:
        raise LaunchCompileError("audio.cpp command requires server arguments")
    executable = str(raw_argv[0])
    argv: List[ArgvItem] = []
    macros = runtime.get("macros") or {}
    for token in raw_argv[1:]:
        text = str(token)
        if text == "${PORT}":
            argv.append(dict(PORT_PLACEHOLDER))
        elif text.startswith("${") and text.endswith("}") and text[2:-1] in macros:
            argv.append({"artifact": "server.json"})
        else:
            argv.append(text)
    sidecar = runtime.get("sidecar") or {}
    active = _active_row(engine)
    env_lines = list(runtime.get("env") or [])
    engine_defaults = {}
    for line in env_lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in _COMPOSITIONAL_KEYS:
            continue
        engine_defaults[key] = value
    env = _compile_env(
        config,
        engine=engine,
        required_ld=_ld_from_lines(env_lines),
        required_path=[],
        engine_defaults=engine_defaults,
        active=active,
    )
    cwd = str(runtime.get("cmd_cwd") or "").strip() or None
    return LaunchSpec(
        model_id=stable_id,
        engine_id=engine,
        engine_install_id=_install_identity(active, executable),
        executable=executable,
        argv=argv,
        cwd=cwd if cwd and os.path.isdir(cwd) else None,
        env=env,
        artifacts={"server.json": sidecar},
        file_identities={"executable": _file_identity(executable)},
    )


def _active_row(engine: str) -> Dict[str, Any]:
    from backend import data_store

    try:
        return dict(data_store.get_store().get_active_engine_version(engine) or {})
    except Exception:
        return {}


def _install_identity(active: Mapping[str, Any], executable: str) -> str:
    identity = _file_identity(executable)
    payload = {
        "version": str(active.get("version") or ""),
        "source_commit": str(active.get("source_commit") or active.get("commit") or ""),
        "executable": identity,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _file_identity(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"path": "", "present": False}
    if not os.path.isfile(path):
        return {"path": path, "present": False}
    st = os.stat(path)
    return {
        "path": os.path.realpath(path),
        "present": True,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _reject_owned_flags(tokens: Sequence[Any], *, owned: Optional[set] = None) -> None:
    owned_flags = _OWNED_PORT_FLAGS if owned is None else owned
    for token in tokens:
        if not isinstance(token, str):
            continue
        flag = token.split("=", 1)[0]
        if flag in owned_flags:
            raise LaunchCompileError(
                f"{flag} is Studio-owned and cannot be repeated in custom arguments",
                field=flag,
            )


def _user_env(config: Mapping[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    raw = config.get("runtime_env")
    values: Dict[str, str] = {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise LaunchCompileError("runtime_env must be a mapping", field="runtime_env")
    for key, value in raw.items():
        name = str(key)
        _validate_env_name(name, field="runtime_env")
        if value is None:
            raise LaunchCompileError(
                f"{name} is null; use runtime_env_unset to remove it",
                field=name,
            )
        text = value if isinstance(value, str) else str(value)
        if "\x00" in text:
            raise LaunchCompileError(f"{name} contains a NUL byte", field=name)
        values[name] = text
    unset_raw = config.get("runtime_env_unset") or []
    if not isinstance(unset_raw, list):
        raise LaunchCompileError(
            "runtime_env_unset must be a list",
            field="runtime_env_unset",
        )
    unset: List[str] = []
    for item in unset_raw:
        name = str(item).strip()
        _validate_env_name(name, field="runtime_env_unset")
        if name in values:
            raise LaunchCompileError(
                f"{name} is both set and unset",
                field=name,
            )
        if name not in unset:
            unset.append(name)
    return values, unset


def _validate_env_name(name: str, *, field: str) -> None:
    if not name or not _ENV_NAME_RE.match(name) or "\x00" in name:
        raise LaunchCompileError(f"Invalid environment name {name!r}", field=field)
    if name.startswith(_STUDIO_PREFIX):
        raise LaunchCompileError(
            f"{name} is reserved by Studio",
            field=name,
        )


def _baseline_env() -> Dict[str, str]:
    found: Dict[str, str] = {}
    for key in _BASELINE_KEYS:
        if _SECRET_NAME_RE.search(key):
            continue
        value = os.environ.get(key)
        if value is None or "\x00" in value:
            continue
        found[key] = value
    return found


def _split_paths(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part for part in str(value).split(os.pathsep) if part]


def _compose_path(required: Sequence[str], user: Optional[str], baseline: Optional[str]) -> str:
    ordered: List[str] = []
    seen = set()
    chunks: List[str] = list(required)
    if user:
        chunks.extend(_split_paths(user))
    if baseline:
        chunks.extend(_split_paths(baseline))
    for part in chunks:
        if part in seen:
            continue
        seen.add(part)
        ordered.append(part)
    return os.pathsep.join(ordered)


def _compile_env(
    config: Mapping[str, Any],
    *,
    engine: str,
    required_ld: Sequence[str],
    required_path: Sequence[str],
    engine_defaults: Mapping[str, str],
    active: Mapping[str, Any],
) -> EnvSpec:
    user_set, user_unset = _user_env(config)
    if engine == "sglang_v100" and "PYTORCH_CUDA_ALLOC_CONF" in user_set:
        user_set.setdefault("PYTORCH_ALLOC_CONF", user_set["PYTORCH_CUDA_ALLOC_CONF"])
        user_set.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    _apply_gpu_contract(config, user_set, user_unset, engine=engine, active=active)
    merged: Dict[str, str] = {}
    baseline = _baseline_env()
    for key, value in baseline.items():
        if key in _COMPOSITIONAL_KEYS:
            continue
        merged[key] = value
    for key, value in engine_defaults.items():
        if key in _COMPOSITIONAL_KEYS:
            continue
        merged[key] = value
    for key, value in user_set.items():
        if key in _COMPOSITIONAL_KEYS:
            continue
        if key in {"CUDA_HOME", "CUDA_PATH"}:
            _validate_cuda_toolkit(value, field=key)
        merged[key] = value
    for key in user_unset:
        if key in _COMPOSITIONAL_KEYS:
            if required_ld and key == "LD_LIBRARY_PATH":
                raise LaunchCompileError(
                    "Cannot unset LD_LIBRARY_PATH while the engine requires library paths",
                    field=key,
                )
            if required_path and key == "PATH":
                raise LaunchCompileError(
                    "Cannot unset PATH while the engine requires executable paths",
                    field=key,
                )
        for required in _gpu_only_invariants(engine):
            if key == required:
                raise LaunchCompileError(
                    f"{engine} requires {key}",
                    field=key,
                )
        merged.pop(key, None)
    path_user = user_set.get("PATH")
    if "PATH" not in user_unset:
        composed = _compose_path(required_path, path_user, baseline.get("PATH"))
        if composed:
            merged["PATH"] = composed
    ld_user = user_set.get("LD_LIBRARY_PATH")
    if "LD_LIBRARY_PATH" not in user_unset:
        composed_ld = _compose_path(
            required_ld,
            ld_user,
            baseline.get("LD_LIBRARY_PATH"),
        )
        if composed_ld:
            merged["LD_LIBRARY_PATH"] = composed_ld
    unset = [key for key in user_unset if key not in merged]
    return EnvSpec(set=merged, unset=unset)


def _validate_cuda_toolkit(path: str, *, field: str) -> None:
    if not path.strip():
        raise LaunchCompileError(f"{field} cannot be an empty toolkit path", field=field)
    if not os.path.isdir(path):
        raise LaunchCompileError(
            f"{field} does not describe an installed CUDA toolkit: {path}",
            field=field,
        )
    if not (os.path.isdir(os.path.join(path, "bin")) or os.path.isdir(os.path.join(path, "lib64"))):
        raise LaunchCompileError(
            f"{field} is not a CUDA toolkit prefix: {path}",
            field=field,
        )


def _apply_gpu_contract(
    config: Mapping[str, Any],
    user_set: Dict[str, str],
    user_unset: List[str],
    *,
    engine: str,
    active: Mapping[str, Any],
) -> None:
    mode = str(config.get("gpu_mode") or "").strip().lower()
    devices = config.get("gpu_devices")
    raw = user_set.get("CUDA_VISIBLE_DEVICES")
    if mode and mode not in {"inherit", "selected", "cpu"}:
        raise LaunchCompileError(
            "gpu_mode must be inherit, selected, or cpu",
            field="gpu_mode",
        )
    if mode == "cpu":
        backend = str(config.get("backend") or active.get("build_config", {}).get("backend") or "")
        if not _cpu_supported(engine):
            raise LaunchCompileError(
                f"{engine} does not support CPU mode",
                field="gpu_mode",
            )
        if engine == "audio_cpp" and backend.lower() not in {"", "cpu"}:
            raise LaunchCompileError(
                "audio.cpp CPU mode conflicts with the selected backend",
                field="gpu_mode",
            )
        if raw not in (None, "", "-1"):
            raise LaunchCompileError(
                "CPU mode conflicts with CUDA_VISIBLE_DEVICES",
                field="CUDA_VISIBLE_DEVICES",
            )
        user_set.pop("CUDA_VISIBLE_DEVICES", None)
        if "CUDA_VISIBLE_DEVICES" not in user_unset:
            user_unset.append("CUDA_VISIBLE_DEVICES")
        return
    if mode == "selected":
        if not isinstance(devices, list) or not devices:
            raise LaunchCompileError(
                "Selected GPU mode requires an ordered non-empty gpu_devices list",
                field="gpu_devices",
            )
        ordered = [str(item).strip() for item in devices]
        if any(not item for item in ordered):
            raise LaunchCompileError("GPU identity is empty", field="gpu_devices")
        _reject_missing_devices(ordered)
        effective = _cuda_device_ids_for_engine(ordered, engine)
        joined = ",".join(effective)
        saved_joined = ",".join(ordered)
        if raw is not None and raw not in {saved_joined, joined}:
            raise LaunchCompileError(
                "gpu_devices conflicts with CUDA_VISIBLE_DEVICES",
                field="CUDA_VISIBLE_DEVICES",
            )
        user_set["CUDA_VISIBLE_DEVICES"] = joined
        # gpu_mode/gpu_devices are the authoritative representation. Older
        # saves and rapid UI mode changes can retain the CPU-mode unset marker;
        # selecting GPUs supersedes that stale marker.
        user_unset[:] = [name for name in user_unset if name != "CUDA_VISIBLE_DEVICES"]
        _validate_parallelism(config, ordered)
        return
    if mode == "inherit":
        if isinstance(devices, list) and devices:
            raise LaunchCompileError(
                "Inherit mode cannot also pin gpu_devices",
                field="gpu_devices",
            )
        if raw is None:
            baseline = _inherit_cuda_baseline()
            if baseline:
                user_set["CUDA_VISIBLE_DEVICES"] = baseline
        return
    if raw is not None:
        _validate_parallelism(config, [part for part in raw.split(",") if part.strip()])


def _cuda_device_ids_for_engine(ordered: Sequence[str], engine: str) -> List[str]:
    """Translate UUID selections for runtimes that only accept numeric CUDA ids."""
    if engine != "sglang_v100":
        return list(ordered)
    try:
        from backend.services.model_metadata import get_startup_gpu_list

        inventory = get_startup_gpu_list() or {}
    except Exception:
        return list(ordered)
    gpus = inventory.get("gpus") if isinstance(inventory, dict) else None
    if not isinstance(gpus, list):
        return list(ordered)
    uuid_to_index = {
        str(gpu["uuid"]): str(gpu["index"])
        for gpu in gpus
        if isinstance(gpu, dict) and gpu.get("uuid") and gpu.get("index") is not None
    }
    return [uuid_to_index.get(device, device) for device in ordered]


def _inherit_cuda_baseline() -> Optional[str]:
    """Deployment CUDA visibility for inherit mode.

    An unset value, ``all``, or a list of every device the process can see is
    not a selection. Copying that into the launch spec pins the model to every
    GPU. A narrower parent value is the deployment baseline.
    """
    baseline = os.environ.get("CUDA_VISIBLE_DEVICES")
    if baseline is None:
        return None
    text = baseline.strip()
    if not text or text.lower() == "all":
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts or _lists_every_visible_gpu(parts):
        return None
    return text


def _lists_every_visible_gpu(parts: Sequence[str]) -> bool:
    try:
        from backend.services.model_metadata import get_startup_gpu_list

        inventory = get_startup_gpu_list() or {}
    except Exception:
        return False
    gpus = inventory.get("gpus") if isinstance(inventory, dict) else None
    if not isinstance(gpus, list) or not gpus:
        return False
    indexes = []
    uuids = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        if gpu.get("index") is not None:
            indexes.append(str(gpu["index"]))
        if gpu.get("uuid"):
            uuids.append(str(gpu["uuid"]))
    selected = set(parts)
    if indexes and selected == set(indexes):
        return True
    if uuids and selected == set(uuids):
        return True
    return False


def _reject_missing_devices(ordered: Sequence[str]) -> None:
    try:
        from backend.services.model_metadata import get_startup_gpu_list

        inventory = get_startup_gpu_list() or {}
    except Exception:
        return
    gpus = inventory.get("gpus") if isinstance(inventory, dict) else None
    if not isinstance(gpus, list) or not gpus:
        if isinstance(inventory, dict) and inventory.get("cpu_only_mode"):
            raise LaunchCompileError(
                "Selected GPUs are not visible in this deployment",
                field="gpu_devices",
            )
        return
    known = set()
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        if gpu.get("uuid"):
            known.add(str(gpu["uuid"]))
        if gpu.get("index") is not None:
            known.add(str(gpu["index"]))
    missing = [item for item in ordered if item not in known]
    if missing:
        raise LaunchCompileError(
            "GPU identity is not visible in this deployment: " + ", ".join(missing),
            field="gpu_devices",
        )


def _validate_parallelism(config: Mapping[str, Any], devices: Sequence[str]) -> None:
    count = len(devices)
    if count <= 0:
        return
    for key in ("tensor_parallel_size", "tp", "pipeline_parallel_size"):
        raw = config.get(key)
        if raw in (None, ""):
            continue
        try:
            size = int(raw)
        except (TypeError, ValueError):
            continue
        if size > count:
            raise LaunchCompileError(
                f"{key}={size} exceeds the {count} selected GPU(s)",
                field=key,
            )
    raw_main = config.get("main_gpu")
    if raw_main not in (None, ""):
        try:
            main = int(raw_main)
        except (TypeError, ValueError):
            main = -1
        if main < 0 or main >= count:
            raise LaunchCompileError(
                "main_gpu is outside the effective CUDA device order",
                field="main_gpu",
            )


def _ld_from_lines(lines: Sequence[str]) -> List[str]:
    for line in lines:
        if line.startswith("LD_LIBRARY_PATH="):
            return _split_paths(line.split("=", 1)[1])
    return []
