"""R87 pins: declared YAML install defaults + declared bundle formats are honored.

Two reproduced defects:
1. `register_definition_lifecycles` derived install defaults from the
   `lifecycle` section and OVERWROTE `settings_defaults`, so a declared branch
   `stable` / version `1.2.3` became `main` / empty (wrong install target).
2. `models/schema._default_package_kind` checked `prepared_bundle_formats` but
   then restricted the result to `mixed`/`original`, so a declared
   `safetensors` bundle fell through to `hf_snapshot` (admitting engines through
   generic format inference).
"""

from __future__ import annotations

import pytest

from backend.engines.definitions.schema import validate_definition
from backend.engines.registry import EngineSpec
from backend.models.schema import _default_package_kind, compatible_engines_for_record
from backend.routes.engine_versions import definition_install_defaults
from backend.venv_install_settings import default_install_settings, register_engine_install_settings


def _raw(**overrides):
    doc = {
        "schema_version": 2,
        "id": "defaultscraft",
        "label": "Defaultscraft",
        "repository_source": "Example Org/Defaultscraft",
        "install_kind": "python_venv",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["venv_path"],
        "artifact_formats": ["safetensors"],
        "package_kinds": ["hf_snapshot"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "lifecycle": {
            "default_repo_url": "https://github.com/acme/defaultscraft.git",
            "default_branch": "main",
            "version_setting_key": "pip_version",
        },
    }
    doc.update(overrides)
    return doc


def test_declared_settings_defaults_win_over_lifecycle_fallbacks():
    definition = validate_definition(
        _raw(settings_defaults={"source_branch": "stable", "pip_version": "1.2.3"}),
        source="defaultscraft.yaml",
    )
    defaults = definition_install_defaults(definition)
    assert defaults["source_branch"] == "stable"
    assert defaults["pip_version"] == "1.2.3"
    # Unpinned keys still fall back to the lifecycle values.
    assert defaults["source_repo"] == "https://github.com/acme/defaultscraft.git"

    register_engine_install_settings("defaultscraft", defaults)
    try:
        effective = default_install_settings("defaultscraft")
        assert effective["source_branch"] == "stable"
        assert effective["pip_version"] == "1.2.3"
    finally:
        import backend.venv_install_settings as vis

        vis._ENGINE_DEFAULTS.pop("defaultscraft", None)


def test_lifecycle_only_definition_keeps_the_derived_defaults():
    definition = validate_definition(_raw(), source="defaultscraft.yaml")
    defaults = definition_install_defaults(definition)
    assert defaults == {
        "source_repo": "https://github.com/acme/defaultscraft.git",
        "source_branch": "main",
        "pip_version": "",
    }


@pytest.mark.asyncio
async def test_startup_registration_preserves_declared_defaults_for_dispatch(monkeypatch):
    """Producer→consumer: startup registration → settings surface → install call.

    The R87 defect lived in the JOIN between these components, so this drives
    the real registration function and the real dispatch resolver instead of
    calling the merge helper in isolation.
    """
    from dataclasses import replace

    from backend.engines.definitions import loader
    from backend.routes import engine_versions as ev
    import backend.venv_install_settings as vis

    definition = validate_definition(
        _raw(
            settings_defaults={"source_branch": "stable", "pip_version": "1.2.3"},
            lifecycle={
                "default_repo_url": "https://github.com/acme/defaultscraft.git",
                "default_branch": "main",
                "version_setting_key": "pip_version",
                "update_source": {"kind": "pypi", "package": "defaultscraft"},
            },
        ),
        source="defaultscraft.yaml",
    )
    real = loader.load_default_engine_definitions()
    registry = type(real)(definitions={**real.definitions, "defaultscraft": definition})
    monkeypatch.setattr(loader, "load_default_engine_definitions", lambda: registry)
    monkeypatch.setattr(ev, "LIFECYCLE", {})
    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", {})

    registered = ev.register_definition_lifecycles()
    assert "defaultscraft" in registered

    # The settings surface serves the DECLARED values...
    settings = vis.coerce_install_settings("defaultscraft", {})
    assert settings["source_branch"] == "stable"
    assert settings["pip_version"] == "1.2.3"

    # ...and the install dispatch resolves the same declared version.
    calls = []

    class _Installer:
        async def install_release(self, **kwargs):
            calls.append(kwargs)
            return {"message": "started", "task_id": "t-1"}

    lc = replace(ev.LIFECYCLE["defaultscraft"], get_installer=lambda: _Installer())
    result = await ev.lifecycle_install(lc, {})
    assert result["task_id"] == "t-1"
    assert calls == [{"version": "1.2.3", "force_reinstall": False}]


def _fake_spec(bundle_formats):
    return EngineSpec(
        id="bundlecraft",
        label="bundlecraft",
        repository_source="bundlecraft",
        install_kind="prebuilt",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"safetensors"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        model_compat={"prepared_bundle_formats": list(bundle_formats)},
    )


def test_declared_bundle_format_is_honored_for_any_format(monkeypatch):
    from backend.engines import registry as registry_module

    monkeypatch.setitem(registry_module.ENGINE_REGISTRY, "bundlecraft", _fake_spec(["safetensors"]))
    assert _default_package_kind({"engine": "bundlecraft"}, "safetensors") == "prepared_bundle"
    # A declared bundle is NOT a generic HF snapshot: compatibility stays explicit.
    record = {
        "engine": "bundlecraft",
        "artifact": {"format": "safetensors", "package_kind": "prepared_bundle"},
    }
    assert compatible_engines_for_record(record) == []
    # Even without an explicit package_kind, the declared format wins and the
    # record stays out of generic safetensors inference.
    assert (
        compatible_engines_for_record(
            {"engine": "bundlecraft", "artifact": {"format": "safetensors"}}
        )
        == []
    )
    # A plain safetensors record with no engine still infers by format (unchanged).
    assert "1cat_vllm" in compatible_engines_for_record({"artifact": {"format": "safetensors"}})


def test_undeclared_family_rule_stays_historical(monkeypatch):
    from backend.engines import registry as registry_module

    monkeypatch.setitem(registry_module.ENGINE_REGISTRY, "bundlecraft", _fake_spec([]))
    assert _default_package_kind({"family": "speech"}, "mixed") == "prepared_bundle"
    assert _default_package_kind({"family": "speech"}, "original") == "prepared_bundle"
    # The family rule does not generalize to other formats.
    assert _default_package_kind({"family": "speech"}, "safetensors") == "hf_snapshot"


def test_audio_declarations_still_hold():
    assert _default_package_kind({"engine": "audio_cpp"}, "mixed") == "prepared_bundle"
    assert _default_package_kind({"engine": "audio_cpp"}, "original") == "prepared_bundle"
    assert _default_package_kind({"engine": "audio_cpp"}, "gguf") == "single_file"
    assert _default_package_kind({"engine": "llama_cpp"}, "safetensors") == "hf_snapshot"


def test_declared_boolean_install_settings_round_trip():
    from backend.venv_install_settings import coerce_install_settings

    register_engine_install_settings(
        "compatcraft",
        {
            "source_repo": "https://github.com/acme/compatcraft.git",
            "source_branch": "main",
            "pip_version": "",
            "compat_sm70": False,
        },
    )
    assert coerce_install_settings("compatcraft", {"compat_sm70": True})["compat_sm70"] is True
    assert coerce_install_settings("compatcraft", {"compat_sm70": "true"})["compat_sm70"] is True
    assert coerce_install_settings("compatcraft", {})["compat_sm70"] is False
