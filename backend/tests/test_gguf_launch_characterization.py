"""Frozen launch contract for the native GGUF family (characterization).

These pins describe what a native GGUF model launch IS, independent of which
implementation produces it: the tests compile through the public dispatch
chain (``launch_spec._compile_launch``), so they hold while the frozen compiler
owns the family and keep holding once a definition declares the shared
mechanism. Everything a proxy manifest, a preview or a process depends on is
frozen here: executable/cwd, argv order and the typed port placeholder, the
model alias, mmproj/draft slots, structured flags, environment composition,
file/install identities and the typed error bytes.

No engine-id literal appears below: the matrix runs over the registry-derived
native GGUF set, and the binary name comes from the engine home constant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.engines.llama_cpp.resolve import SERVER_BINARY_NAME
from backend.proxy import launch_spec
from backend.tests.gguf_launch_harness import (
    BASELINE_ENV,
    BUILD_LIB,
    CUDA_LIB,
    STABLE_ID,
    LaunchHarness,
    contract,
    path_list,
)


@pytest.fixture
def launch(monkeypatch, tmp_path):
    harness = LaunchHarness(monkeypatch, tmp_path)
    assert harness.engines, "no native GGUF engine is registered"
    return harness


def _compile(launch, model, config, engine):
    return launch_spec._compile_launch(model, config, STABLE_ID, engine)


def _frozen(launch, engine, *, argv, identities, env_extra=None):
    """The expected observable contract for one launch.

    ``env_extra`` is merged over the composed baseline: HOME always survives,
    PATH/LD are composed from the baseline plus the managed library path.
    """
    binary = str(launch.binaries[engine])
    env = {
        "HOME": BASELINE_ENV["HOME"],
        "PATH": path_list(BASELINE_ENV["PATH"]),
        "LD_LIBRARY_PATH": path_list(CUDA_LIB, BUILD_LIB, BASELINE_ENV["LD_LIBRARY_PATH"]),
    }
    env.update(env_extra or {})
    return {
        "engine_id": engine,
        "model_id": STABLE_ID,
        "executable": binary,
        "cwd": str(Path(binary).parent),
        "argv": argv,
        "env_set": env,
        "env_unset": [],
        "artifacts": {},
        "identity_fields": {name: ["mtime_ns", "path", "present", "size"] for name in identities},
        "identity_present": {name: True for name in identities},
    }


def _assert_contract(launch, spec, expected):
    observed = contract(spec)
    install_identity = observed.pop("install_identity")
    assert observed == expected
    payload = json.loads(install_identity)
    # The install identity ties version + source commit to the exact executable
    # bytes the launch will spawn (staleness/rollback depend on it).
    assert payload["version"] == launch.rows[spec.engine_id]["version"]
    assert payload["source_commit"] == launch.rows[spec.engine_id]["source_commit"]
    assert payload["executable"]["present"] is True
    assert payload["executable"]["path"] == os.path.realpath(str(launch.binaries[spec.engine_id]))


# ---------------------------------------------------------------------------
# Canonical launches
# ---------------------------------------------------------------------------


def test_canonical_contract_is_frozen(launch):
    for engine in launch.engines:
        model_file = launch.model_file(engine)
        launch.resolve_model_to(model_file)
        model = launch.model(engine)
        config = {"ctx_size": 2048, "stop": ["END", "END"], "runtime_env": {"FOO": "bar"}}
        spec = _compile(launch, model, config, engine)
        _assert_contract(
            launch,
            spec,
            _frozen(
                launch,
                engine,
                argv=[
                    "--model",
                    str(model_file),
                    "--port",
                    launch_spec.PORT_PLACEHOLDER,
                    "--alias",
                    STABLE_ID,
                    "--ctx-size",
                    "2048",
                    "--stop",
                    "END",
                    "--stop",
                    "END",
                ],
                identities=("model", "executable"),
                env_extra={"FOO": "bar"},
            ),
        )


def test_hf_repo_fallback_contract_is_frozen(launch):
    for engine in launch.engines:
        launch.resolve_model_to(None)
        # Companions are deliberately suppressed while the server fetches from
        # the hub itself (it resolves mmproj on its own in repo mode).
        launch.resolve_companions({"mmproj.gguf": launch.companion_file(engine, "mmproj.gguf")})
        model = launch.model(engine, files=())
        spec = _compile(launch, model, {}, engine)
        _assert_contract(
            launch,
            spec,
            _frozen(
                launch,
                engine,
                argv=[
                    "--hf-repo",
                    "org/model:q4_k_m",
                    "--port",
                    launch_spec.PORT_PLACEHOLDER,
                    "--alias",
                    STABLE_ID,
                ],
                identities=("executable",),
            ),
        )


def test_companion_slots_are_frozen(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        mmproj = launch.companion_file(engine, "mmproj-F16.gguf")
        draft = launch.companion_file(engine, "mtp.gguf")
        launch.resolve_companions({"mmproj-F16.gguf": mmproj, "mtp.gguf": draft})
        model = launch.model(engine, mmproj_filename="mmproj-F16.gguf", mtp_filename="mtp.gguf")
        spec = _compile(launch, model, {}, engine)
        argv = spec.argv
        assert argv[4:8] == ["--alias", STABLE_ID, "--mmproj", str(mmproj)]
        assert argv[8:12] == ["--model-draft", str(draft), "--spec-type", "draft-mtp"]
        assert spec.file_identities.keys() == {"model", "mmproj", "draft", "executable"}


def test_dflash_precedes_mtp_and_is_frozen(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        launch.resolve_companions(
            {
                "mtp.gguf": launch.companion_file(engine, "mtp.gguf"),
                "dflash.gguf": launch.companion_file(engine, "dflash.gguf"),
            }
        )
        model = launch.model(engine, mtp_filename="mtp.gguf", dflash_filename="dflash.gguf")
        spec = _compile(launch, model, {}, engine)
        assert spec.argv[-2:] == ["--spec-type", "draft-dflash"]
        assert spec.argv[-4] == "--model-draft"
        assert spec.argv[-3].endswith("dflash.gguf")


def test_dspark_precedes_dflash_and_mtp(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        launch.resolve_companions(
            {
                "dspark.gguf": launch.companion_file(engine, "dspark.gguf"),
                "dflash.gguf": launch.companion_file(engine, "dflash.gguf"),
                "mtp.gguf": launch.companion_file(engine, "mtp.gguf"),
            }
        )
        model = launch.model(
            engine,
            dspark_filename="dspark.gguf",
            dflash_filename="dflash.gguf",
            mtp_filename="mtp.gguf",
        )
        spec = _compile(launch, model, {}, engine)
        assert spec.argv[-2:] == ["--spec-type", "draft-dspark"]
        assert spec.argv[-4] == "--model-draft"
        assert spec.argv[-3].endswith("dspark.gguf")


def test_user_spec_type_suppresses_the_default_and_is_frozen(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        launch.resolve_companions({"mtp.gguf": launch.companion_file(engine, "mtp.gguf")})
        model = launch.model(engine, mtp_filename="mtp.gguf")
        spec = _compile(launch, model, {"custom_args": "--spec-type draft-simple"}, engine)
        assert spec.argv.count("--spec-type") == 1
        assert spec.argv[-2:] == ["--spec-type", "draft-simple"]


def test_structured_flags_precede_custom_args(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        config = {"ctx_size": 4096, "custom_args": "--log-level debug"}
        spec = _compile(launch, launch.model(engine), config, engine)
        # Scanned parameters are emitted in sorted key order, then custom args.
        assert spec.argv[-4:] == ["--ctx-size", "4096", "--log-level", "debug"]


def test_port_and_alias_placeholders_are_typed(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        spec = _compile(launch, launch.model(engine), {}, engine)
        assert spec.argv[2] == "--port"
        # A typed placeholder, never an interpolated string.
        assert spec.argv[3] == {"runtime": "port"}
        assert [token for token in spec.argv if isinstance(token, dict)] == [
            launch_spec.PORT_PLACEHOLDER
        ]
        assert spec.argv[4] == "--alias"
        assert spec.argv[5] == STABLE_ID
        assert "${PORT}" in launch_spec.format_argv(spec.executable, spec.argv)
        assert "$HOME" not in launch_spec.format_argv(spec.executable, spec.argv)


def test_sibling_build_bin_invocation_is_frozen(monkeypatch, tmp_path):
    harness = LaunchHarness(monkeypatch, tmp_path, layout="sibling")
    for engine in harness.engines:
        harness.resolve_model_to(harness.model_file(engine))
        spec = _compile(harness, harness.model(engine), {}, engine)
        # A stored path under ``bin`` launches the sibling build artifact, with
        # the build directory as cwd (shared libraries are resolved relatively).
        assert spec.executable == str(harness.root / engine / "build" / "bin" / SERVER_BINARY_NAME)
        assert spec.cwd == str(harness.root / engine / "build" / "bin")
        assert harness.library_calls[-1] == spec.cwd


def test_library_path_uses_the_invocation_cwd_and_keeps_path_unprepended(launch):
    for engine in launch.engines:
        launch.resolve_model_to(launch.model_file(engine))
        spec = _compile(launch, launch.model(engine), {}, engine)
        assert launch.library_calls[-1] == spec.cwd
        entries = spec.env.set["LD_LIBRARY_PATH"].split(os.pathsep)
        assert entries[:2] == [CUDA_LIB, BUILD_LIB]
        assert BASELINE_ENV["LD_LIBRARY_PATH"] in entries
        # The native family never prepends its bin directory to PATH.
        assert spec.env.set["PATH"] == path_list(BASELINE_ENV["PATH"])
        assert spec.env.unset == []


def test_proxy_model_name_is_the_studio_id(launch):
    for engine in launch.engines:
        assert launch_spec._use_model_name(launch.model(engine), {}, STABLE_ID, engine) == STABLE_ID


# ---------------------------------------------------------------------------
# Typed error bytes (LaunchSpec consumers surface these in previews)
# ---------------------------------------------------------------------------


_NO_OVERRIDE = object()


def _error(
    launch,
    engine,
    *,
    config=None,
    set_binary=_NO_OVERRIDE,
    quantization="Q4_K_M",
    files=("model.gguf",),
):
    launch.resolve_model_to(launch.model_file(engine))
    if set_binary is not _NO_OVERRIDE:
        launch.set_binary(engine, set_binary)
    model = launch.model(engine, quantization=quantization, files=files)
    with pytest.raises(launch_spec.LaunchCompileError) as error:
        _compile(launch, model, config or {}, engine)
    return error.value


def test_missing_binary_error_bytes_are_frozen(launch):
    for engine in launch.engines:
        error = _error(launch, engine, set_binary=None)
        assert error.field == "executable"
        assert str(error) == f"No active {engine} {SERVER_BINARY_NAME} binary configured"


def test_missing_binary_file_error_bytes_are_frozen(launch):
    for engine in launch.engines:
        missing = str(launch.root / engine / "build" / "bin" / "gone")
        error = _error(launch, engine, set_binary=missing)
        assert error.field == "executable"
        assert str(error) == f"{SERVER_BINARY_NAME} binary not found at: {missing}"


def test_unresolvable_model_error_bytes_are_frozen(launch):
    for engine in launch.engines:
        error = _error(launch, engine, quantization=None, files=())
        assert error.field == "model"
        assert str(error) == (
            "Model path could not be resolved from HF metadata or runtime overlay"
        )


def test_owned_port_error_bytes_are_frozen(launch):
    for engine in launch.engines:
        error = _error(launch, engine, config={"custom_args": "--port 1"})
        assert error.field == "--port"
        assert str(error) == ("--port is Studio-owned and cannot be repeated in custom arguments")


def test_unset_library_path_error_bytes_are_frozen(launch):
    for engine in launch.engines:
        error = _error(launch, engine, config={"runtime_env_unset": ["LD_LIBRARY_PATH"]})
        assert error.field == "LD_LIBRARY_PATH"
        assert str(error) == (
            "Cannot unset LD_LIBRARY_PATH while the engine requires library paths"
        )


def test_the_legacy_gguf_compiler_no_longer_exists():
    from backend.proxy import launch_spec

    assert not hasattr(launch_spec, "_compile_gguf")


def test_every_native_gguf_engine_declares_the_family_profile():
    from backend.engines.definitions import load_default_engine_definitions

    definitions = load_default_engine_definitions().definitions
    for engine in ("llama_cpp", "ik_llama", "unsloth_llama"):
        profile = definitions[engine].profile("")
        assert profile["launch_family"] == "gguf_llama_server", engine
        assert profile["port_binding"]["flag"] == "--port", engine
