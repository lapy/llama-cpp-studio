"""Contracts for the ReVolta (Leonccaa/ReVolta) V100 inference engine.

ReVolta is a 1Cat-vLLM/vLLM-derived server distributed as the ``revolta``
wheel; the import name and CLI deliberately stay ``vllm``. It therefore joins
the existing ``vllm`` runtime mechanism, the ``vllm_help`` scanner and the
shared Python-venv install/build host instead of adding engine-specific code.
"""

from __future__ import annotations

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.plan import materialize_plan
from backend.engines.definitions.profiles import compile_profile_argv, compile_profile_env
from backend.engines.registry import ENGINE_REGISTRY
from backend.models.schema import compatible_engines_for_record
from backend.venv_install_settings import default_install_settings


def test_revolta_is_a_definition_owned_vllm_family_engine():
    definition = load_default_engine_definitions().definitions["revolta"]
    assert definition.install_kind == "python_venv"
    assert definition.runtime_kind == "vllm"
    assert definition.scanner_kind == "vllm_help"
    assert definition.artifact_formats == ("safetensors",)
    assert definition.data_dirs == ("revolta",)
    assert dict(definition.launch_policy)["gpu_archs"] == ("sm70",)
    assert dict(definition.launch_policy)["cpu_supported"] is False
    assert definition.experimental is True
    # One identity system: the registry entry is the validated definition.
    assert ENGINE_REGISTRY["revolta"].to_api_dict() == definition.to_api_dict()
    assert "revolta" in compatible_engines_for_record({"format": "safetensors"})


def test_revolta_install_defaults_come_from_the_definition():
    settings = default_install_settings("revolta")
    assert settings["source_repo"] == "https://github.com/Leonccaa/ReVolta.git"
    assert settings["source_branch"] == "main"
    assert settings["release_version"] == ""


def test_revolta_launch_profile_matches_the_documented_vllm_cli():
    definition = load_default_engine_definitions().definitions["revolta"]
    argv = compile_profile_argv(definition, model_ref="Org/ReVolta-Model", port="8123")
    assert argv == ["serve", "Org/ReVolta-Model", "--port", "8123"]
    env = compile_profile_env(definition, {"PATH": "/usr/bin"})
    # The SM70 environment is declared once: expandable segments require NCCL
    # graph registration off, and V100 JIT builds target 7.0 only.
    assert env["env"]["NCCL_GRAPH_REGISTER"] == "0"
    assert env["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["env"]["TORCH_CUDA_ARCH_LIST"] == "7.0"
    assert env["missing_required"] == []
    assert env["cwd"] == "venv_root"


def test_revolta_release_plan_installs_the_revolta_distribution():
    definition = load_default_engine_definitions().definitions["revolta"]
    plan = materialize_plan(definition, "release")
    assert plan.engine_id == "revolta"
    assert [step.primitive for step in plan.steps] == [
        "prereqs.resolve_version",
        "venv.create",
        "pip.install_package",
        "publish.finalize",
    ]
    install_step = plan.steps[2]
    assert install_step.with_["package"] == "revolta"
    assert "https://download.pytorch.org/whl/cu128" in install_step.with_["extra_args"]


def test_revolta_generic_lifecycle_surfaces(client, monkeypatch, tmp_path):
    from backend import data_store
    from backend.routes.engine_versions import register_definition_lifecycles

    import backend.main  # noqa: F401 - production startup registrations

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    # The app lifespan materializes YAML-only lifecycles; the test client does
    # not run lifespan, so run the same startup gate here.
    register_definition_lifecycles()

    listing = client.get("/api/engines")
    entry = next(engine for engine in listing.json()["engines"] if engine["id"] == "revolta")
    assert entry["label"] == "ReVolta"
    assert entry["ui_kind"] == "venv"
    assert entry["gpu_archs"] == ["sm70"]
    assert entry["generic_lifecycle"] is True
    assert "install" in entry["generic_verbs"]

    status = client.get("/api/engines/revolta/status")
    assert status.status_code == 200
    assert status.json()["installed"] is False

    settings = client.get("/api/engines/revolta/build-settings")
    assert settings.status_code == 200
    assert settings.json()["source_repo"] == "https://github.com/Leonccaa/ReVolta.git"
