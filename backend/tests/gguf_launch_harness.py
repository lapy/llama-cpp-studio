"""Shared harness for the native GGUF launch characterization/parity tests.

The production launch path binds its resolvers FUNCTION-LOCALLY (the frozen
compiler does, and so does the declared family mechanism), so this harness
patches the symbols where those late imports find them:

* ``backend.data_store.get_store`` and the llama-family active-binary resolver,
* ``backend.proxy.commands`` (param index, CUDA library path, model source),
* ``backend.models.hub.resolve_cached_model_path`` (companions),
* ``backend.proxy.launch_spec._baseline_env`` (a deterministic process env).

Nothing here asserts behaviour: it only builds install trees, model rows and
normalized projections so both implementations observe identical inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

from backend.engines.definitions.schema import validate_definition
from backend.engines.llama_cpp.resolve import SERVER_BINARY_NAME
from backend.engines.registry import (
    ENGINE_REGISTRY,
    GGUF_ENGINE_IDS,
    SHIPPED_ENGINE_IDS,
)
from backend.proxy import gguf_launch, profile_launch

BASELINE_ENV = {
    "HOME": "/home/studio",
    "PATH": "/usr/local/bin:/usr/bin",
    "LD_LIBRARY_PATH": "/opt/lib",
}

PARAM_INDEX = {
    "ctx_size": {"primary_flag": "--ctx-size", "value_kind": "scalar"},
    "stop": {"primary_flag": "--stop", "value_kind": "repeatable"},
}

STABLE_ID = "model-stable"

# The managed CUDA prefix and the build directory are joined with ":" by the
# production resolver; a fixed POSIX-looking pair keeps the expectation
# machine-independent (the real path is asserted through the captured call).
CUDA_LIB = "/opt/cuda/lib64"
BUILD_LIB = "/opt/build/bin"
LIBRARY_PATH = f"{CUDA_LIB}:{BUILD_LIB}"


def native_gguf_engines() -> tuple:
    """Native (non-venv) GGUF engines: a server runtime plus gguf artifacts."""
    return tuple(
        sorted(
            engine_id
            for engine_id in GGUF_ENGINE_IDS & SHIPPED_ENGINE_IDS
            if ENGINE_REGISTRY[engine_id].runtime_kind == "llama_server"
            and "binary_path" in ENGINE_REGISTRY[engine_id].active_path_fields
        )
    )


def _write_binary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("gguf", encoding="utf-8")
    return path


class LaunchHarness:
    """Fake installs + deterministic resolvers for one test."""

    def __init__(self, monkeypatch, tmp_path: Path, *, layout: str = "build_bin"):
        self.monkeypatch = monkeypatch
        self.root = Path(tmp_path)
        self.engines = native_gguf_engines()
        self.layout = layout
        self.binaries: Dict[str, Path] = {}
        self.rows: Dict[str, dict] = {}
        self.companions: Dict[str, str] = {}
        self.library_calls: list = []
        self.mechanism_calls: list = []
        self._model_path: Optional[str] = None
        self._build_installs()
        self._install_resolvers()

    # -- installs ---------------------------------------------------------
    def _binary_location(self, engine: str) -> Path:
        base = self.root / engine
        if self.layout == "sibling":
            # Stored path under ``bin`` while the real binary lives in build/bin.
            return base / "bin" / SERVER_BINARY_NAME
        return base / "build" / "bin" / SERVER_BINARY_NAME

    def _build_installs(self) -> None:
        for engine in self.engines:
            binary = _write_binary(self._binary_location(engine))
            self.binaries[engine] = str(binary)
            self.rows[engine] = {
                "version": f"{engine}-1",
                "source_commit": "abc123",
                "binary_path": str(binary),
                "build_config": {"backend": "cpu"},
            }
            if self.layout == "sibling":
                _write_binary(self.root / engine / "build" / "bin" / SERVER_BINARY_NAME)

    def _install_resolvers(self) -> None:
        monkeypatch = self.monkeypatch
        store = SimpleNamespace(
            get_active_engine_version=lambda engine: dict(self.rows.get(engine) or {})
        )
        monkeypatch.setattr("backend.data_store.get_store", lambda: store)
        monkeypatch.setattr(
            "backend.engines.llama_cpp.resolve.get_active_binary_path_for_engine",
            lambda _store, engine: self.binaries.get(engine),
        )
        monkeypatch.setattr("backend.proxy.commands._resolve_cuda_library_path", self._library)
        monkeypatch.setattr(
            "backend.proxy.commands._active_engine_param_index",
            lambda engine: dict(PARAM_INDEX),
        )
        monkeypatch.setattr("backend.proxy.launch_spec._baseline_env", lambda: dict(BASELINE_ENV))
        monkeypatch.setattr(
            "backend.proxy.launch_spec._active_row",
            lambda engine: dict(self.rows.get(engine) or {}),
        )
        monkeypatch.setattr(
            "backend.proxy.commands.resolve_gguf_model_path",
            lambda model: self._model_path if model.get("files") else None,
        )
        monkeypatch.setattr(
            "backend.proxy.commands.resolve_gguf_model_path_for_quant",
            lambda hf_id, quant: None,
        )
        monkeypatch.setattr(
            "backend.models.hub.resolve_cached_model_path",
            lambda hf_id, filename: self.companions.get(filename),
        )
        monkeypatch.setattr(
            "backend.services.model_metadata.get_startup_gpu_list",
            lambda: {
                "vendor": "nvidia",
                "cpu_only_mode": False,
                "gpus": [{"index": 0, "uuid": "GPU-0", "name": "A"}],
            },
        )
        # Observed once per harness: wrapping per call would stack wrappers and
        # record the same launch twice.
        original = gguf_launch.compile_gguf_family_launch

        def _observed(**kwargs):
            self.mechanism_calls.append(kwargs.get("engine"))
            return original(**kwargs)

        monkeypatch.setattr(gguf_launch, "compile_gguf_family_launch", _observed)

    def _library(self, build_dir: str) -> str:
        # Records the build directory the launch passed, so the test can assert
        # the executable/cwd contract while keeping the LD string portable.
        self.library_calls.append(str(build_dir))
        return LIBRARY_PATH

    # -- scenario inputs --------------------------------------------------
    def model_file(self, engine: str, name: str = "model.gguf") -> Path:
        return _write_gguf(self.root / "cache" / engine / name)

    def companion_file(self, engine: str, name: str) -> Path:
        return _write_gguf(self.root / "cache" / engine / name)

    def resolve_model_to(self, path: Optional[Path]) -> None:
        """Make model resolution return ``path`` (None = hf-repo fallback)."""
        self._model_path = str(path) if path is not None else None

    def resolve_companions(self, mapping: Mapping[str, Path]) -> None:
        """Companion filename -> cached path (a name absent here is not cached)."""
        self.companions = {name: str(path) for name, path in mapping.items()}

    def set_binary(self, engine: str, path: Optional[str]) -> None:
        """Override the active binary path (None = no active build)."""
        self.rows[engine]["binary_path"] = path or ""
        if path is None:
            self.binaries.pop(engine, None)
        else:
            # Stored verbatim: the launch compiler's own normalisation (the /app
            # fallback) must be what decides the resolved path.
            self.binaries[engine] = path

    # -- rows -------------------------------------------------------------
    def model(
        self,
        engine: str,
        *,
        huggingface_id: str = "org/model",
        quantization: Optional[str] = "Q4_K_M",
        files: Sequence[str] = ("model.gguf",),
        **extra: Any,
    ) -> dict:
        row: Dict[str, Any] = {
            "id": f"catalog-{engine}",
            "huggingface_id": huggingface_id,
            "config": {"engine": engine, "engines": {engine: {}}},
        }
        if quantization is not None:
            row["quantization"] = quantization
        if files:
            row["files"] = list(files)
        row.update(extra)
        return row

    # -- compilation ------------------------------------------------------
    def declare_family(self, engines: Optional[Sequence[str]] = None) -> None:
        table = {
            engine: family_definition(engine)
            for engine in (engines if engines is not None else self.engines)
        }
        self.monkeypatch.setattr(
            profile_launch,
            "_REGISTRY",
            SimpleNamespace(get=lambda engine_id: table.get(str(engine_id or ""))),
        )

    def declared(self, model: Mapping[str, Any], config: Mapping[str, Any], engine: str):
        self.declare_family()
        self.mechanism_calls = []
        spec = profile_launch.compile_profile_launch(model, config, STABLE_ID, engine)
        assert spec is not None
        # Non-vacuity: a parity assertion must never compare the frozen compiler
        # with itself. The DECLARED mechanism has to be the producer.
        assert self.mechanism_calls == [engine], "the declared mechanism did not run"
        return spec

    def dispatched(self, model: Mapping[str, Any], config: Mapping[str, Any], engine: str):
        """Compile through the public dispatch chain (profile branch first)."""
        from backend.proxy import launch_spec

        self.declare_family()
        return launch_spec._compile_launch(model, config, STABLE_ID, engine)


def family_definition(engine: str):
    """A validated definition whose default profile declares the GGUF family."""
    return validate_definition(
        {
            "schema_version": 2,
            "id": engine,
            "label": f"{engine} native GGUF family",
            "repository_source": "native GGUF family fixture",
            "install_kind": "cmake",
            "runtime_kind": "llama_server",
            "scanner_kind": "llama_help",
            "active_path_fields": ["binary_path"],
            "artifact_formats": ["gguf"],
            "package_kinds": ["single_file"],
            "tasks": ["text-generation"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "runtime_profiles": {
                "default": {
                    "default": True,
                    "launch_family": gguf_launch.LAUNCH_FAMILY,
                    "port_binding": {"flag": "--port"},
                }
            },
        },
        source=f"{engine}-family.yaml",
    )


def contract(spec) -> dict:
    """Machine-stable projection of a LaunchSpec (no host mtimes/realpaths)."""
    return {
        "engine_id": spec.engine_id,
        "model_id": spec.model_id,
        "executable": str(spec.executable),
        "cwd": str(spec.cwd) if spec.cwd is not None else None,
        "argv": list(spec.argv),
        "env_set": dict(spec.env.set),
        "env_unset": list(spec.env.unset),
        "artifacts": dict(spec.artifacts),
        "identity_fields": {
            name: sorted(payload) for name, payload in sorted(spec.file_identities.items())
        },
        "identity_present": {
            name: bool(payload.get("present"))
            for name, payload in sorted(spec.file_identities.items())
        },
        "install_identity": spec.engine_install_id,
    }


def path_list(*parts: str) -> str:
    """Compose a PATH/LD value the way the compiler does (os.pathsep, dedup)."""
    ordered = []
    for part in parts:
        if part and part not in ordered:
            ordered.append(part)
    return os.pathsep.join(ordered)
