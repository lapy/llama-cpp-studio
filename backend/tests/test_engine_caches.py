"""Every supported engine's compile/JIT caches are Studio-owned.

The contract has three parts:

* each shipped definition declares the baseline cache variables for its
  runtime family (``env.caches`` for profile launches, family profiles for
  native GGUF engines);
* every declared variable resolves under the ACTIVE install directory, never
  under ``$HOME`` or a temporary path;
* stock model-weight caches stay out of the engine install, because a
  per-version copy of a multi-GB download is not a cache Studio should own.

The research behind the baselines is recorded in ``backend/engines/caches.py``.
"""

from __future__ import annotations

import os

import pytest

from backend.engines.caches import (
    NATIVE_JIT_CACHES,
    PYTHON_JIT_CACHES,
    SGLANG_JIT_CACHES,
    VLLM_JIT_CACHES,
    cache_environment,
)
from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.proxy import launch_spec
from backend.tests.gguf_launch_harness import STABLE_ID, LaunchHarness

# Model-WEIGHT caches are deliberately not engine-scoped: they are Studio's
# model storage, and duplicating them per engine version would multiply disk
# use by the number of installed versions for no compiled-artifact benefit.
MODEL_WEIGHT_CACHES = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "MODELSCOPE_CACHE",
        "TORCH_HOME",
    }
)


def _all_declared_caches(definition) -> set[str]:
    declared: set[str] = set()
    for profile in dict(definition.runtime_profiles).values():
        declared.update((profile.get("env") or {}).get("caches") or ())
    return declared


def _family_baseline(definition) -> tuple[str, ...]:
    if definition.install_kind == "python_venv":
        baseline = list(PYTHON_JIT_CACHES)
        if definition.runtime_kind == "sglang":
            baseline.extend(SGLANG_JIT_CACHES)
        if definition.runtime_kind in {"vllm", "onecat_vllm"}:
            baseline.extend(VLLM_JIT_CACHES)
        return tuple(dict.fromkeys(baseline))
    if definition.install_kind in {"cmake", "prebuilt"}:
        # audio.cpp's frozen family compiler applies the native policy in
        # runtime.py; it has no runtime profile to declare it in.
        return NATIVE_JIT_CACHES
    return ()


def test_every_shipped_engine_declares_its_family_cache_baseline():
    registry = load_default_engine_definitions()
    assert registry.ids(), "no engine definitions loaded"
    for definition in registry.all():
        baseline = _family_baseline(definition)
        if not baseline:
            continue
        if definition.id == "audio_cpp":
            # Frozen audio family compiler: the managed policy is applied by
            # backend.engines.audio_cpp.runtime (pinned below).
            continue
        declared = _all_declared_caches(definition)
        missing = sorted(set(baseline) - declared)
        assert not missing, f"{definition.id} is missing managed caches {missing}"


def test_model_weight_caches_are_not_engine_scoped():
    registry = load_default_engine_definitions()
    for definition in registry.all():
        leaked = sorted(_all_declared_caches(definition) & MODEL_WEIGHT_CACHES)
        assert not leaked, f"{definition.id} must not scope model weights: {leaked}"


def test_every_declaration_resolves_under_the_install_directory(tmp_path):
    for definition in load_default_engine_definitions().all():
        for name in _all_declared_caches(definition):
            expected = os.path.join(str(tmp_path), "caches")
            path = cache_environment(str(tmp_path), [name])[name]
            assert os.path.dirname(path) == expected
            assert "$" not in path and path.startswith(str(tmp_path))


@pytest.mark.parametrize("engine", ["llama_cpp", "ik_llama", "unsloth_llama"])
def test_native_gguf_launch_applies_the_managed_cache_policy(engine, monkeypatch, tmp_path):
    """The family compiler resolves declared caches inside the active install."""
    harness = LaunchHarness(monkeypatch, tmp_path)
    install = tmp_path / "installs" / engine
    harness.rows[engine]["install_dir"] = str(install)
    model_file = harness.model_file(engine)
    harness.resolve_model_to(model_file)

    spec = launch_spec._compile_launch(harness.model(engine), {}, STABLE_ID, engine)

    assert spec.env.set["XDG_CACHE_HOME"] == str(install / "caches" / "xdg-cache-home")
    assert spec.env.set["CUDA_CACHE_PATH"] == str(install / "caches" / "cuda-cache-path")
    assert os.path.isdir(install / "caches" / "cuda-cache-path")
    assert "HOME" in spec.env.set


def test_audio_cpp_runtime_applies_the_native_cache_policy(tmp_path):
    from backend.engines.audio_cpp import runtime as audio_runtime

    install = tmp_path / "audio-install"
    install.mkdir()
    active = {
        "version": "v1",
        "install_dir": str(install),
        "server_binary_path": str(install / "bin" / "audiocpp_server"),
        "build_config": {"backend": "cpu"},
    }
    # CPU build keeps the CUDA toolkit resolver out of this unit test; the
    # managed cache policy is independent of the backend.
    env = dict(line.split("=", 1) for line in audio_runtime._runtime_env(active, {}) if "=" in line)

    assert env["XDG_CACHE_HOME"] == str(install / "caches" / "xdg-cache-home")
    assert env["CUDA_CACHE_PATH"] == str(install / "caches" / "cuda-cache-path")
    assert os.path.isdir(install / "caches" / "cuda-cache-path")


def test_managed_xdg_keeps_model_weights_in_the_shared_studio_cache(monkeypatch, tmp_path):
    """XDG_CACHE_HOME must not drag HF/Torch model weights into a version."""
    from backend.proxy import launch_spec

    monkeypatch.setattr("backend.paths.studio_data_dir", lambda: str(tmp_path / "studio"))
    monkeypatch.setattr(launch_spec, "_baseline_env", lambda: {"HOME": "/home/studio"})
    install = tmp_path / "install"
    env = launch_spec._compile_env(
        {},
        engine="fixture",
        required_ld=[],
        required_path=[],
        engine_defaults={"XDG_CACHE_HOME": str(install / "caches" / "xdg-cache-home")},
        active={},
    )
    assert env.set["HF_HOME"] == str(tmp_path / "studio" / "hf-cache")
    assert env.set["TORCH_HOME"] == str(tmp_path / "studio" / "hf-cache" / "torch")
    assert str(install) not in env.set["HF_HOME"]

    override = launch_spec._compile_env(
        {"runtime_env": {"HF_HOME": "/custom/hf"}},
        engine="fixture",
        required_ld=[],
        required_path=[],
        engine_defaults={"XDG_CACHE_HOME": str(install / "caches" / "xdg-cache-home")},
        active={},
    )
    assert override.set["HF_HOME"] == "/custom/hf"


def test_family_profile_may_only_declare_managed_caches():
    """A family profile cannot smuggle in env the mechanism would ignore."""
    base = {
        "schema_version": 2,
        "id": "familyfixture",
        "label": "Family fixture",
        "repository_source": "fixture",
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
                "launch_family": "gguf_llama_server",
                "port_binding": {"flag": "--port"},
                "env": {"caches": ["CUDA_CACHE_PATH"]},
            }
        },
    }
    validate_definition(dict(base), source="family.yaml")

    bad = dict(base)
    bad["runtime_profiles"] = {
        "default": {
            "default": True,
            "launch_family": "gguf_llama_server",
            "port_binding": {"flag": "--port"},
            "env": {"defaults": {"FOO": "bar"}},
        }
    }
    with pytest.raises(DefinitionError, match="cannot also declare env"):
        validate_definition(bad, source="family.yaml")


def test_venv_source_build_uses_the_runtime_profile_caches(tmp_path):
    """A source build fills the same managed caches the server will read."""
    from backend.engines.python.generic import GenericVenvInstaller

    definition = validate_definition(
        {
            "schema_version": 2,
            "id": "cachecraft",
            "label": "Cachecraft",
            "repository_source": "Example Org/Cachecraft",
            "install_kind": "python_venv",
            "runtime_kind": "vllm",
            "scanner_kind": "vllm_help",
            "active_path_fields": ["venv_path"],
            "artifact_formats": ["safetensors"],
            "package_kinds": ["hf_snapshot"],
            "tasks": ["text-generation"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "lifecycle": {
                "binary_name": "python",
                "root_name": "cachecraft",
                "log_name": "cachecraft.log",
            },
            "runtime_profiles": {
                "default": {
                    "default": True,
                    "executable": {"kind": "venv_python", "name": "python"},
                    "model_binding": {
                        "mode": "flag",
                        "flag": "--model",
                        "source": "huggingface_id",
                    },
                    "port_binding": {"flag": "--port"},
                    "env": {"caches": ["TRITON_CACHE_DIR", "CUDA_CACHE_PATH"]},
                }
            },
        },
        source="cachecraft.yaml",
    )
    installer = GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "cachecraft.log"),
        state_path=str(tmp_path / "cachecraft_state.json"),
        base_dir=str(tmp_path / "cachecraft"),
    )

    env = installer._compose_build_env()

    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "cachecraft" / "caches" / "triton-cache-dir")
    assert env["CUDA_CACHE_PATH"] == str(tmp_path / "cachecraft" / "caches" / "cuda-cache-path")
    assert os.path.isdir(tmp_path / "cachecraft" / "caches" / "cuda-cache-path")
