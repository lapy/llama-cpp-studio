"""Launch campaign R1/R2 — the profile-driven launch interpreter.

Pins, in order of contractual weight:
1. the nine legacy adapters are UNREACHABLE through the new branch
   (dispatch order + GENERATED-engine gate — byte-preservation is
   structural, not aspirational: gate G2);
2. definition-less ids keep the EXACT historical "No runtime adapter
   registered" error bytes;
3. a definition-backed engine compiles a real LaunchSpec from its YAML
   runtime_profile (typed port placeholder, model binding both modes,
   cwd/env/PATH composition mirroring the legacy venv-family shapes);
4. env.required preflight is the typed LaunchCompileError(field=NAME)
   contract (the SGLang-V100 CUDA_HOME analogue, declaratively);
5. the interpreter module itself carries zero engine-id literals (G6).

No engine-id literals appear in this file on purpose: fixtures are a
synthetic definition ("profilecraft"), and legacy coverage iterates the
live GENERATED/derived id SETS — so this file also cannot grow the
literal ratchet.
"""

from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest

from backend.engines.definitions.schema import validate_definition
from backend.engines.registry import ENGINE_REGISTRY, GGUF_ENGINE_IDS
from backend.proxy import launch_spec, profile_launch

PROFILECRAFT_ID = "profilecraft"


def _raw_definition():
    return {
        "schema_version": 2,
        "id": PROFILECRAFT_ID,
        "label": "Profilecraft",
        "repository_source": "Example Org/Profilecraft",
        "install_kind": "python_venv",
        # runtime_kind is decorative today (zero production readers);
        # 'llama_server' is chosen from the vocabulary to keep this test
        # free of engine-id literals under the ratchet.
        "runtime_kind": "llama_server",
        "scanner_kind": "lmdeploy_help",
        "active_path_fields": ["venv_path"],
        "artifact_formats": ["safetensors"],
        "package_kinds": ["hf_snapshot"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "runtime_profiles": {
            "default": {
                "default": True,
                "executable": {"kind": "venv_console_script", "name": "profilecraft-server"},
                "argv_prefix": ["serve", "api"],
                "model_binding": {
                    "mode": "flag",
                    "flag": "--model",
                    "source": "huggingface_id",
                },
                "port_binding": {"flag": "--port"},
            }
        },
    }


def _definition(**profile_overrides):
    raw = _raw_definition()
    profile = dict(raw["runtime_profiles"]["default"])
    profile.update(profile_overrides)
    raw["runtime_profiles"] = {"default": profile}
    return validate_definition(raw, source="profilecraft.yaml")


def _install_fake(monkeypatch, *definitions):
    table = {definition.id: definition for definition in definitions}
    monkeypatch.setattr(
        profile_launch,
        "_REGISTRY",
        SimpleNamespace(get=lambda engine_id: table.get(str(engine_id or ""))),
    )
    return table


def _fake_venv(tmp_path, script_name="profilecraft-server"):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / script_name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    return str(tmp_path / "venv"), str(exe)


def _patch_active_row(monkeypatch, row):
    monkeypatch.setattr(launch_spec, "_active_row", lambda engine: dict(row))


# ---------------------------------------------------------------------------
# 1) The nine stay on their frozen compilers until a definition opts in (G2)
# ---------------------------------------------------------------------------


def test_unknown_engine_keeps_the_historical_error_bytes():
    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        launch_spec._compile_launch({"huggingface_id": "org/model"}, {}, "m", "nosuchengine")
    assert str(excinfo.value) == ("No runtime adapter registered for engine 'nosuchengine'")
    assert excinfo.value.field == "engine"


def test_definition_without_runtime_profiles_is_not_launchable(monkeypatch):
    raw = validate_definition(
        {
            "schema_version": 2,
            "id": PROFILECRAFT_ID,
            "label": "Profilecraft",
            "repository_source": "Example Org/Profilecraft",
            "install_kind": "python_venv",
            "runtime_kind": "llama_server",
            "scanner_kind": "lmdeploy_help",
            "active_path_fields": ["venv_path"],
            "artifact_formats": ["safetensors"],
            "package_kinds": ["hf_snapshot"],
            "tasks": ["text-generation"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
        source="profilecraft.yaml",
    )
    _install_fake(monkeypatch, raw)
    assert profile_launch.compile_profile_launch({}, {}, "m", PROFILECRAFT_ID) is None
    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        launch_spec._compile_launch({}, {}, "m", PROFILECRAFT_ID)
    assert str(excinfo.value) == (f"No runtime adapter registered for engine {PROFILECRAFT_ID!r}")


# ---------------------------------------------------------------------------
# 3) Happy path: YAML profile → real LaunchSpec
# ---------------------------------------------------------------------------


def test_console_script_profile_compiles_a_real_launch_spec(monkeypatch, tmp_path):
    definition = _definition()
    _install_fake(monkeypatch, definition)
    venv_path, exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"version": "9.9.9", "venv_path": venv_path})

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "model-stable", PROFILECRAFT_ID
    )

    assert spec is not None
    assert spec.engine_id == PROFILECRAFT_ID
    assert spec.model_id == "model-stable"
    assert spec.executable == exe
    # the port stays a TYPED placeholder — never an interpolated string
    assert spec.argv == [
        "serve",
        "api",
        "--model",
        "org/model",
        "--port",
        {"runtime": "port"},
    ]
    # venv-family PATH composition mirrors the legacy compilers
    assert os.path.dirname(exe) in (spec.env.set.get("PATH") or "")
    assert spec.cwd is None
    assert '"9.9.9"' in spec.engine_install_id
    assert spec.file_identities["executable"]["present"] is True
    # document() round-trips (the manifest consumes exactly this shape)
    document = spec.document()
    assert document["argv"][5] == {"runtime": "port"}


def test_declared_caches_resolve_inside_the_engine_install(monkeypatch, tmp_path):
    """`env.caches` points each JIT cache at <install_dir>/caches/<slug>."""
    definition = _definition(
        env={
            "cwd": "venv_root",
            "caches": ["TRITON_CACHE_DIR", "SGLANG_CACHE_DIR"],
        }
    )
    _install_fake(monkeypatch, definition)
    venv_path, exe = _fake_venv(tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    _patch_active_row(
        monkeypatch,
        {"version": "9.9.9", "venv_path": venv_path, "install_dir": str(install_dir)},
    )

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "model-stable", PROFILECRAFT_ID
    )

    triton = os.path.join(str(install_dir), "caches", "triton-cache-dir")
    sglang = os.path.join(str(install_dir), "caches", "sglang-cache-dir")
    assert spec.env.set["TRITON_CACHE_DIR"] == triton
    assert spec.env.set["SGLANG_CACHE_DIR"] == sglang
    # The compiler creates them, so a first start writes into a real directory.
    assert os.path.isdir(triton) and os.path.isdir(sglang)
    # An explicit default still wins over the managed path.
    pinned = _definition(
        env={"caches": ["TRITON_CACHE_DIR"], "defaults": {"TRITON_CACHE_DIR": "/opt/triton"}}
    )
    _install_fake(monkeypatch, pinned)
    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "model-stable", PROFILECRAFT_ID
    )
    assert spec.env.set["TRITON_CACHE_DIR"] == "/opt/triton"


def test_positional_binding_and_module_style_prefix(monkeypatch, tmp_path):
    definition = _definition(
        argv_prefix=["-m", "profilecraft.server"],
        executable={"kind": "venv_python"},
        model_binding={"mode": "positional", "source": "huggingface_id"},
        port_binding={"flag": "--server-port"},
    )
    _install_fake(monkeypatch, definition)
    venv_path, exe = _fake_venv(tmp_path, script_name="python")
    _patch_active_row(monkeypatch, {"venv_path": venv_path})

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
    )
    assert spec.argv == [
        "-m",
        "profilecraft.server",
        "org/model",
        "--server-port",
        {"runtime": "port"},
    ]


def test_missing_model_reference_is_typed(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})

    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch({}, {}, "m", PROFILECRAFT_ID)
    assert excinfo.value.field == "model"
    assert str(excinfo.value) == f"{PROFILECRAFT_ID} model must have huggingface_id"


def test_missing_executable_is_typed(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _definition())
    _patch_active_row(monkeypatch, {"venv_path": str(tmp_path / "nonexistent-venv")})

    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(
            {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
        )
    assert excinfo.value.field == "executable"
    assert str(excinfo.value) == f"{PROFILECRAFT_ID} executable unavailable"


# ---------------------------------------------------------------------------
# 4) env.required preflight (typed, CUDA_HOME-analogue semantics)
# ---------------------------------------------------------------------------

_REQUIRED_ENV = "PROFILECAMPAIGN_REQUIRED_TOKEN"


def _env_required_definition():
    return _definition(
        env={"required": [_REQUIRED_ENV], "defaults": {"PROFILECAMPAIGN_MODE": "fast"}}
    )


def test_missing_required_env_raises_typed_preflight(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _env_required_definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    monkeypatch.delenv(_REQUIRED_ENV, raising=False)

    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(
            {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
        )
    assert excinfo.value.field == _REQUIRED_ENV
    assert str(excinfo.value) == f"{PROFILECRAFT_ID} requires {_REQUIRED_ENV}"


def test_required_env_satisfied_and_defaults_land_in_env(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _env_required_definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    monkeypatch.setenv(_REQUIRED_ENV, "provided")

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
    )
    assert spec.env.set.get("PROFILECAMPAIGN_MODE") == "fast"


def test_user_config_env_satisfies_required(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _env_required_definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    monkeypatch.delenv(_REQUIRED_ENV, raising=False)

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"},
        {"runtime_env": {_REQUIRED_ENV: "from-user-config"}},
        "m",
        PROFILECRAFT_ID,
    )
    assert spec.env.set.get(_REQUIRED_ENV) == "from-user-config"


def test_required_env_defaults_satisfy_validation(monkeypatch, tmp_path):
    definition = _definition(
        env={
            "required": [_REQUIRED_ENV],
            "defaults": {_REQUIRED_ENV: "from-defaults"},
        }
    )
    _install_fake(monkeypatch, definition)
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    monkeypatch.delenv(_REQUIRED_ENV, raising=False)

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
    )
    assert spec.env.set.get(_REQUIRED_ENV) == "from-defaults"


def test_required_env_is_validated_against_the_final_launch_environment(monkeypatch, tmp_path):
    """A required variable present in the process env but UNSET by config
    must fail here: the launched process receives
    ``os.environ + env.set - env.unset``, so validating the pre-unset
    mapping let it pass and be absent at launch."""
    _install_fake(monkeypatch, _env_required_definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    monkeypatch.setenv(_REQUIRED_ENV, "from-process")

    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(
            {"huggingface_id": "org/model"},
            {"runtime_env_unset": [_REQUIRED_ENV]},
            "m",
            PROFILECRAFT_ID,
        )
    assert excinfo.value.field == _REQUIRED_ENV
    assert str(excinfo.value) == f"{PROFILECRAFT_ID} requires {_REQUIRED_ENV}"


# ---------------------------------------------------------------------------
# 5) cwd + PATH policy from the profile
# ---------------------------------------------------------------------------


def test_cwd_venv_root_resolves_from_the_executable_layout(monkeypatch, tmp_path):
    definition = _definition(
        env={"cwd": "venv_root", "prepend_bin_to_path": True},
    )
    _install_fake(monkeypatch, definition)
    venv_path, exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
    )
    # cwd derives from the EXECUTABLE's layout, dirname(dirname(exe)) —
    # mirroring the legacy venv-module compiler's rule.
    assert spec.cwd == venv_path
    assert os.path.dirname(exe) in (spec.env.set.get("PATH") or "")


# ---------------------------------------------------------------------------
# 6) proxy model-name contract follows the declared binding source
# ---------------------------------------------------------------------------


def test_use_model_name_follows_binding_source(monkeypatch):
    model = {"huggingface_id": "org/model"}
    _install_fake(monkeypatch, _definition())
    assert launch_spec._use_model_name(model, {}, "stable", PROFILECRAFT_ID) == "org/model"

    _install_fake(
        monkeypatch,
        _definition(model_binding={"mode": "flag", "flag": "--model", "source": "local_path"}),
    )
    assert launch_spec._use_model_name(model, {}, "stable", PROFILECRAFT_ID) == "stable"
    # unknown ids keep the historical default byte-for-byte
    assert launch_spec._use_model_name(model, {}, "stable", "nosuchengine") == "stable"


# ---------------------------------------------------------------------------
# 6) Studio-owned flags come from the PROFILE, not a fixed list
# ---------------------------------------------------------------------------


def test_owned_flags_are_profile_driven(monkeypatch, tmp_path):
    definition = _definition(argv_prefix=["serve", "api", "--host", "127.0.0.1"])
    _install_fake(monkeypatch, definition)
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})

    owned = profile_launch.profile_owned_flags(definition)
    # model binding + port binding + the prefix's own flag (host), and NOT
    # the prefix's positional tokens.
    assert owned == frozenset({"--model", "--port", "--host"})
    assert "serve" not in owned

    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(
            {"huggingface_id": "org/model"},
            {"custom_args": "--host 0.0.0.0"},
            "m",
            PROFILECRAFT_ID,
        )
    assert excinfo.value.field == "--host"


def test_custom_args_cannot_override_the_yaml_selected_model(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})

    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(
            {"huggingface_id": "org/model"},
            {"custom_args": "--model someone/else"},
            "m",
            PROFILECRAFT_ID,
        )
    assert excinfo.value.field == "--model"
    assert "Studio-owned" in str(excinfo.value)


def test_non_owned_custom_args_still_append(monkeypatch, tmp_path):
    _install_fake(monkeypatch, _definition())
    venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"},
        {"custom_args": "--log-level debug"},
        "m",
        PROFILECRAFT_ID,
    )
    assert spec.argv[-2:] == ["--log-level", "debug"]
    assert spec.argv[:4] == ["serve", "api", "--model", "org/model"]


# ---------------------------------------------------------------------------
# 7) GGUF dispatch: definition-backed engines are profile-launched
# ---------------------------------------------------------------------------


def test_gguf_declaring_definition_engine_uses_the_profile_compiler(monkeypatch, tmp_path):
    raw = _raw_definition()
    raw["artifact_formats"] = ["gguf"]
    definition = validate_definition(raw, source="profilecraft.yaml")
    _install_fake(monkeypatch, definition)
    venv_path, exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    # Declaring gguf artifacts must not select a family compiler; the
    # profile declaration is the only routing fact.
    from backend.engines import registry as registry_module

    registry_module.VALID_ENGINE_IDS.add(PROFILECRAFT_ID)
    try:
        compiled = launch_spec.compile_model_runtime(
            {
                "id": "ggufcraft-model",
                "huggingface_id": "org/model",
                "config": {"engine": PROFILECRAFT_ID, "engines": {PROFILECRAFT_ID: {}}},
            }
        )
    finally:
        registry_module.VALID_ENGINE_IDS.discard(PROFILECRAFT_ID)

    assert compiled.launch.engine_id == PROFILECRAFT_ID
    assert compiled.launch.executable == exe
    assert compiled.launch.argv == [
        "serve",
        "api",
        "--model",
        "org/model",
        "--port",
        launch_spec.PORT_PLACEHOLDER,
    ]


# ---------------------------------------------------------------------------
# 8) Launch revisions carry the definition digest (YAML changes are visible)
# ---------------------------------------------------------------------------


def _compiled(monkeypatch, tmp_path, definition, venv_path=None):
    _install_fake(monkeypatch, definition)
    if venv_path is None:
        venv_path, _exe = _fake_venv(tmp_path)
    _patch_active_row(monkeypatch, {"venv_path": venv_path})
    model = {
        "id": "revision-model",
        "huggingface_id": "org/model",
        "config": {"engine": PROFILECRAFT_ID, "engines": {PROFILECRAFT_ID: {}}},
    }
    from backend.engines import registry as registry_module

    registry_module.VALID_ENGINE_IDS.add(PROFILECRAFT_ID)
    try:
        return launch_spec.compile_model_runtime(model), venv_path
    finally:
        registry_module.VALID_ENGINE_IDS.discard(PROFILECRAFT_ID)


def test_launch_revision_includes_the_definition_digest(monkeypatch, tmp_path):
    first, venv_path = _compiled(monkeypatch, tmp_path, _definition())
    assert first.profile_digest.startswith("sha256:")
    assert first.revision == launch_spec.launch_revision(
        first.launch, profile_digest=first.profile_digest
    )
    # Recompiling the same definition against the same install is stable
    # (file identities are part of the launch document).
    same, _venv = _compiled(monkeypatch, tmp_path, _definition(), venv_path=venv_path)
    assert same.revision == first.revision

    # A definition change that leaves the COMPILED launch contract identical
    # (here: a label) must still change the revision — YAML edits participate.
    relabeled = _raw_definition()
    relabeled["label"] = "Profilecraft Relabeled"
    changed, _venv = _compiled(
        monkeypatch,
        tmp_path,
        validate_definition(relabeled, source="profilecraft.yaml"),
        venv_path=venv_path,
    )
    assert changed.profile_digest != first.profile_digest
    assert changed.launch.document() == first.launch.document()
    assert changed.revision != first.revision


def test_manifest_document_records_the_profile_digest(monkeypatch, tmp_path):
    compiled, _venv = _compiled(monkeypatch, tmp_path, _definition())
    document = launch_spec.manifest_document(
        compiled.launch, compiled.revision, profile_digest=compiled.profile_digest
    )
    assert document["profile_digest"] == compiled.profile_digest
    assert document["revision"] == compiled.revision


def test_every_manifest_document_call_site_passes_the_profile_digest():
    """Digest propagation is a sweep: every runtime manifest writer must pass
    the compiled digest (a missed call site silently drops it)."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for path in sorted(root.joinpath("backend").rglob("*.py")):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^[^\n]*manifest_document\(", text, re.MULTILINE):
            line = match.group(0)
            if "def manifest_document" in line:
                continue
            snippet = text[match.start() : match.start() + 300]
            head = snippet.split(")", 1)[0]
            if "profile_digest" not in head:
                offenders.append(
                    f"{path.relative_to(root)}:{text.count(chr(10), 0, match.start()) + 1}"
                )
    assert not offenders, f"manifest_document calls missing profile_digest: {offenders}"


def test_generated_engines_keep_their_historical_revision(
    monkeypatch,
):  # No digest → key absent from the hashed body, revisions byte-identical.
    spec = launch_spec.LaunchSpec(
        model_id="m",
        engine_id="lmdeploy",
        engine_install_id="i",
        executable="/bin/true",
        argv=["serve"],
        cwd=None,
        env=launch_spec.EnvSpec(),
    )
    assert launch_spec.launch_revision(spec) == launch_spec.launch_revision(spec, profile_digest="")
    document = launch_spec.manifest_document(spec, "r")
    assert "profile_digest" not in document


# ---------------------------------------------------------------------------
# 9) G6: the interpreter ships no engine-id literals (self-scan)
# ---------------------------------------------------------------------------


def test_interpreter_module_has_no_engine_id_literals():
    import inspect

    from backend.engines.registry import SHIPPED_ENGINE_IDS

    source = inspect.getsource(profile_launch)
    for engine_id in sorted(SHIPPED_ENGINE_IDS):
        assert engine_id not in source, engine_id


def test_definition_digest_reaches_every_definition_owned_engine():
    """Every YAML-backed launch carries its definition digest, including audio."""
    from backend.engines.definitions import load_default_engine_definitions
    from backend.engines.registry import DEFINITION_ENGINE_IDS

    definitions = load_default_engine_definitions()
    owned = {
        engine_id for engine_id in DEFINITION_ENGINE_IDS if definitions.get(engine_id) is not None
    }
    assert owned, "no definition-owned engines are loaded"
    for engine_id in owned:
        digest = profile_launch.definition_digest_for(engine_id)
        assert digest == definitions.get(engine_id).definition_digest
        assert launch_spec._launch_profile_digest(engine_id) == digest

    # audio.cpp composes argv through its frozen family compiler instead of a
    # runtime profile, but its launch identity must still follow the YAML.
    audio_definition = definitions.get("audio_cpp")
    assert audio_definition is not None
    assert not audio_definition.runtime_profiles
    assert profile_launch.definition_digest_for("audio_cpp") == audio_definition.definition_digest
    assert launch_spec._launch_profile_digest("audio_cpp") == audio_definition.definition_digest


def test_every_native_gguf_engine_follows_its_declaration():
    """Phase-6 boundary, direction-agnostic and cutover-safe.

    Native GGUF launch composition is FAMILY behaviour (model source file vs
    ``--hf-repo``, mmproj/draft companions, alias, shared library path), so an
    engine either DECLARES the shared mechanism — and is then compiled by the
    profile interpreter, never by the frozen compiler — or declares nothing and
    keeps the frozen compiler until it does. A profile without the mechanism
    would silently compose a different (venv) argv, so that shape must be
    unrepresentable: this pin is what makes a later cutover a reviewed diff
    instead of a silent reroute.
    """
    from backend.engines.definitions import load_default_engine_definitions
    from backend.engines.registry import GGUF_ENGINE_IDS, SHIPPED_ENGINE_IDS
    from backend.proxy import gguf_launch

    definitions = load_default_engine_definitions()
    native = [
        engine_id
        for engine_id in GGUF_ENGINE_IDS & SHIPPED_ENGINE_IDS
        if definitions.get(engine_id) is not None
    ]
    assert native, "expected native GGUF engines in the definition set"
    for engine_id in native:
        definition = definitions.get(engine_id)
        if not definition.runtime_profiles:
            assert ENGINE_REGISTRY[engine_id].runtime_profile == {}
            assert profile_launch.profile_launch_definition(engine_id) is None, engine_id
            continue
        spec = definition.profile("")
        assert spec.get("launch_family") == gguf_launch.LAUNCH_FAMILY, engine_id
        assert "model_binding" not in spec, engine_id
        assert "executable" not in spec, engine_id
        assert profile_launch.profile_launch_definition(engine_id) is not None, engine_id


# ---------------------------------------------------------------------------
# 10) The declared GGUF family mechanism (S1/S2 vocabulary)
# ---------------------------------------------------------------------------


def _family_definition():
    from backend.proxy import gguf_launch

    raw = _raw_definition()
    raw["install_kind"] = "cmake"
    raw["active_path_fields"] = ["binary_path"]
    raw["artifact_formats"] = ["gguf"]
    raw["runtime_profiles"] = {
        "default": {
            "default": True,
            "launch_family": gguf_launch.LAUNCH_FAMILY,
            "port_binding": {"flag": "--port"},
        }
    }
    return validate_definition(raw, source="profilecraft.yaml")


def test_family_profile_compiles_through_the_mechanism(monkeypatch, tmp_path):
    """A declared family profile produces the native argv, not a venv argv."""
    definition = _family_definition()
    _install_fake(monkeypatch, definition)
    binary = tmp_path / "build" / "bin" / "server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    model_file = tmp_path / "model.gguf"
    model_file.write_text("gguf", encoding="utf-8")

    import backend.proxy.commands as commands

    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: {"version": "9.9.9"})
    monkeypatch.setattr(
        "backend.engines.llama_cpp.resolve.get_active_binary_path_for_engine",
        lambda _store, _engine: str(binary),
    )
    monkeypatch.setattr(commands, "resolve_gguf_model_path", lambda _model: str(model_file))
    monkeypatch.setattr(commands, "_active_engine_param_index", lambda _engine: {})
    monkeypatch.setattr(commands, "_resolve_cuda_library_path", lambda build_dir: "/opt/lib")
    monkeypatch.setattr(launch_spec, "_baseline_env", lambda: {})

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model", "quantization": "Q4_K_M", "files": ["model.gguf"]},
        {},
        "m",
        PROFILECRAFT_ID,
    )
    assert spec is not None
    assert spec.executable == str(binary)
    assert spec.cwd == str(binary.parent)
    assert spec.argv == [
        "--model",
        str(model_file),
        "--port",
        dict(launch_spec.PORT_PLACEHOLDER),
        "--alias",
        "m",
    ]
    assert spec.file_identities.keys() == {"model", "executable"}


def test_family_profile_keeps_the_studio_model_name(monkeypatch):
    """Declaring a family must not rewrite the proxied model name."""
    definition = _family_definition()
    _install_fake(monkeypatch, definition)
    assert profile_launch.profile_model_source(definition) == "model_id"
    assert (
        launch_spec._use_model_name({"huggingface_id": "org/model"}, {}, "stable", PROFILECRAFT_ID)
        == "stable"
    )


def test_family_profile_owns_only_the_port_surface(monkeypatch):
    definition = _family_definition()
    _install_fake(monkeypatch, definition)
    # The mechanism owns the model flags; only the port is declared here.
    assert profile_launch.profile_owned_flags(definition) == frozenset({"--port"})


def test_unknown_launch_family_is_rejected_at_dispatch(monkeypatch):
    class _UnknownFamily:
        id = PROFILECRAFT_ID
        runtime_profiles = (
            (
                "default",
                {
                    "default": True,
                    "launch_family": "not_a_mechanism",
                    "port_binding": {"flag": "--port"},
                },
            ),
        )

        def profile(self, name=""):
            return dict(self.runtime_profiles[0][1])

    _install_fake(monkeypatch, _UnknownFamily())
    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(
            {"huggingface_id": "org/model"}, {}, "m", PROFILECRAFT_ID
        )
    assert excinfo.value.field == "launch_family"
    assert "not_a_mechanism" in str(excinfo.value)
