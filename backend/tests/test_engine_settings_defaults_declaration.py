"""R73 pins: venv install-settings defaults are engine data.

`backend/venv_install_settings.py` used to keep a five-entry engine-id map
(`_ENGINE_DEFAULTS`). The frozen defaults now live on `EngineSpec.settings_defaults`,
are exported into the generated definitions, and the settings surface reads the
spec; only authoritative (YAML-only) registrations still land in the map.
"""

from __future__ import annotations

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.registry import get_engine_spec
from backend.venv_install_settings import (
    assert_venv_engine,
    coerce_install_settings,
    default_install_settings,
    register_engine_install_settings,
)

EXPECTED_DEFAULTS = {
    "lmdeploy": {
        "source_repo": "https://github.com/InternLM/lmdeploy.git",
        "source_branch": "main",
        "pip_version": "",
        "compat_sm70": False,
        "compat_torch_spec": "torch==2.8.0+cu126 torchvision==0.23.0+cu126",
        "compat_torch_index_url": "https://download.pytorch.org/whl/cu126",
    },
    "1cat_vllm": {
        "source_repo": "https://github.com/1CatAI/1Cat-vLLM.git",
        "source_branch": "main",
        "release_version": "",
    },
    "sglang": {
        "source_repo": "https://github.com/sgl-project/sglang.git",
        "source_branch": "main",
        "pip_version": "",
    },
    "sglang_v100": {
        "source_repo": "https://github.com/haohervchb/sglang-V100.git",
        "source_branch": "main",
    },
    "vllm": {
        "source_repo": "https://github.com/vllm-project/vllm.git",
        "source_branch": "main",
        "pip_version": "",
    },
}


def test_generated_engines_declare_the_historical_defaults():
    for engine_id, defaults in EXPECTED_DEFAULTS.items():
        spec = get_engine_spec(engine_id)
        assert spec is not None, engine_id
        assert dict(spec.settings_defaults) == defaults, engine_id
        assert default_install_settings(engine_id) == defaults
        assert "settings_defaults" not in spec.to_api_dict(), engine_id


def test_defaults_are_exported_with_the_definition():
    definitions = load_default_engine_definitions()
    for engine_id, defaults in EXPECTED_DEFAULTS.items():
        definition = definitions.get(engine_id)
        assert definition is not None, engine_id
        assert dict(definition.settings_defaults) == defaults, engine_id


def test_reads_the_spec_when_nothing_is_registered(monkeypatch):
    class FakeSpec:
        install_kind = "python_venv"
        settings_defaults = {
            "source_repo": "https://github.com/acme/tool.git",
            "source_branch": "dev",
            "pip_version": "",
        }

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())
    assert default_install_settings("tool_engine") == {
        "source_repo": "https://github.com/acme/tool.git",
        "source_branch": "dev",
        "pip_version": "",
    }
    assert coerce_install_settings("tool_engine", {"pip_version": "1.2"})["pip_version"] == "1.2"
    assert_venv_engine("tool_engine")


def test_unknown_engine_still_raises():
    with pytest.raises(ValueError, match="No install settings defaults"):
        default_install_settings("not-an-engine")
    with pytest.raises(ValueError, match="not a python_venv installer"):
        assert_venv_engine("audio_cpp")


def test_authoritative_registration_still_wins():
    register_engine_install_settings(
        "yamlcraft",
        {"source_repo": "https://github.com/acme/yamlcraft.git", "source_branch": "main"},
    )
    try:
        assert default_install_settings("yamlcraft") == {
            "source_repo": "https://github.com/acme/yamlcraft.git",
            "source_branch": "main",
        }
    finally:
        # Authoritative registrations are process-global; leave no residue.
        import backend.venv_install_settings as vis

        vis._ENGINE_DEFAULTS.pop("yamlcraft", None)
