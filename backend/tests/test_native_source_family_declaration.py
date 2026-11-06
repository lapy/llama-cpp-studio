"""R105 pins: the llama-versions routes use the DECLARED native source family."""

from __future__ import annotations

from backend.engines.registry import SHIPPED_ENGINE_IDS, native_source_engine_ids
from backend.routes.llama_versions import _is_native_source_engine


def test_native_source_family_is_derived_from_the_specs():
    assert set(native_source_engine_ids()) & set(SHIPPED_ENGINE_IDS) == {
        "ik_llama",
        "llama_cpp",
    }


def test_route_family_predicate_matches_the_declaration():
    assert _is_native_source_engine("llama_cpp") is True
    assert _is_native_source_engine("ik_llama") is True
    # audio.cpp is cmake but its runtime is audio_cpp; unsloth is prebuilt.
    for other in ("audio_cpp", "unsloth_llama", "lmdeploy", "vllm", "", "not-an-engine"):
        assert _is_native_source_engine(other) is False, other


def test_declared_family_matches_the_install_and_runtime_kinds():
    from backend.engines.registry import ENGINE_REGISTRY

    for engine_id in native_source_engine_ids():
        spec = ENGINE_REGISTRY[engine_id]
        assert spec.install_kind == "cmake", engine_id
        assert spec.runtime_kind == "llama_server", engine_id


def test_llama_server_family_matches_the_historical_tuple():
    """R106: scan/help membership is the declared ``llama_server`` family.

    Membership (not exact equality): hermetic fixture engines registered by other
    tests legitimately declare the same runtime, so this asserts the historical
    members are included and the non-members are excluded.
    """
    from backend.engines.registry import llama_server_engine_ids

    family = set(llama_server_engine_ids())
    assert {"llama_cpp", "ik_llama", "unsloth_llama"} <= family
    # audio.cpp has its own runtime and must NOT be treated as a llama engine.
    assert "audio_cpp" not in family
    # The source-build family is a subset of the runtime family.
    assert set(native_source_engine_ids()) <= family


def test_lifecycle_families_are_declared():
    from backend.engines.lifecycle import _NATIVE_SHARED_ROOT_ENGINES
    from backend.engines.registry import engine_ids_with_install_kind

    assert set(_NATIVE_SHARED_ROOT_ENGINES) == {"llama_cpp", "ik_llama"}
    assert {"audio_cpp", "ik_llama", "llama_cpp"} <= set(engine_ids_with_install_kind("cmake"))


def test_scanner_uses_the_declared_family():
    """The scanner's llama branch is driven by the declared runtime family."""
    import inspect

    from backend.engines.scan import scanner

    source = inspect.getsource(scanner)
    assert "llama_server_engine_ids()" in source
    assert 'in ("llama_cpp", "ik_llama", "unsloth_llama")' not in source


def test_nativecraft_policy_is_loaded_from_definition_data(tmp_path):
    """A third native fork is configured by data, not a new installer branch."""
    from types import SimpleNamespace

    from backend.engines.python.cmake_source import CmakeSourceInstaller

    installer = CmakeSourceInstaller(
        "nativecraft",
        manager=SimpleNamespace(llama_dir=str(tmp_path / "nativecraft")),
    )

    assert installer.engine_id == "nativecraft"
    assert installer._repository_name == "nativecraft"
    assert installer._source_dir == "nativecraft"
    assert installer._binary_name == "native-server"
    assert installer._build_examples_required is True
    assert installer._blas_supported is False
    assert installer._stable_release_versions is False

    plan = installer._recipe("source")
    assert [step.primitive for step in plan.steps] == [
        "vcs.workspace_checkout",
        "vcs.apply_patch",
        "cmake.configure",
        "cmake.build",
        "artifact.locate",
        "publish.finalize",
    ]
    assert set(installer._extra_plan_handlers()) >= {
        "vcs.workspace_checkout",
        "cmake.configure",
        "cmake.build",
        "publish.finalize",
    }

def test_native_host_selection_comes_from_the_definition():
    from backend.engines.adapters import get_native_installer
    from backend.engines.audio_cpp.installer import AudioCppInstaller
    from backend.engines.python.cmake_source import CmakeSourceInstaller

    assert isinstance(get_native_installer("llama_cpp"), CmakeSourceInstaller)
    assert isinstance(get_native_installer("ik_llama"), CmakeSourceInstaller)
    assert isinstance(get_native_installer("nativecraft"), CmakeSourceInstaller)
    assert isinstance(get_native_installer("audio_cpp"), AudioCppInstaller)
    assert get_native_installer("lmdeploy") is None
