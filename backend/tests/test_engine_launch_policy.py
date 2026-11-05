"""R5 pins: the launch policy is engine DATA, not engine-id sets in the proxy.

The launch compiler used to hardcode ``_CPU_REJECTED_ENGINES`` /
``_GPU_ONLY_INVARIANTS`` (launch_spec) and ``_RUNTIME_CONTEXT_KEYS`` (commands).
Those sets are now declared on the engine spec, exported into the generated
definitions, and read from the spec — so a definition-only engine carries the
same behavior without touching proxy code. These pins freeze the historical
values (moved verbatim) and the definition round-trip.
"""

from __future__ import annotations

import pytest

from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import ENGINE_REGISTRY, get_engine_spec

EXPECTED_POLICIES = {
    "llama_cpp": {"runtime_context_key": "ctx_size", "capcompat": True},
    "ik_llama": {"runtime_context_key": "ctx_size", "capcompat": True},
    "unsloth_llama": {"runtime_context_key": "ctx_size", "capcompat": True},
    "lmdeploy": {"cpu_supported": False, "runtime_context_key": "session_len"},
    "vllm": {
        "cpu_supported": False,
        "gpu_archs": ("sm75", "sm80", "sm86", "sm89", "sm90"),
        "runtime_context_key": "max_model_len",
        "capcompat": True,
    },
    "1cat_vllm": {
        "cpu_supported": False,
        "gpu_archs": ("sm70",),
        "runtime_context_key": "max_model_len",
        "capcompat": True,
    },
    "sglang": {
        "cpu_supported": False,
        "gpu_archs": ("sm75", "sm80", "sm86", "sm89", "sm90"),
        "runtime_context_key": "context_length",
    },
    "sglang_v100": {
        "cpu_supported": False,
        "gpu_archs": ("sm70",),
        "runtime_context_key": "context_length",
        "gpu_only_invariants": (
            "CUDA_HOME",
            "TORCH_CUDA_ARCH_LIST",
            "FLASHINFER_DISABLE_VERSION_CHECK",
        ),
    },
}


def _raw(launch_policy=None, schema_version=2):
    raw = {
        "schema_version": schema_version,
        "id": "policycraft",
        "label": "Policycraft",
        "repository_source": "Example Org/Policycraft",
        "install_kind": "python_venv",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["venv_path"],
        "artifact_formats": ["safetensors"],
        "package_kinds": ["hf_snapshot"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
    }
    if launch_policy is not None:
        raw["launch_policy"] = launch_policy
    return raw


def test_shipped_engines_declare_the_historical_launch_policy():
    for engine_id, expected in EXPECTED_POLICIES.items():
        spec = get_engine_spec(engine_id)
        assert spec is not None, engine_id
        assert dict(spec.launch_policy) == expected, engine_id


def test_engines_without_constraints_carry_no_policy():
    for engine_id in ("llama_cpp", "ik_llama", "unsloth_llama"):
        assert "cpu_supported" not in dict(get_engine_spec(engine_id).launch_policy)
    # audio.cpp declares CPU support (R77) so the frontend filter needs no id
    # special case; it is the only generated engine whose policy is that small.
    assert dict(get_engine_spec("audio_cpp").launch_policy) == {"cpu_supported": True}
    for engine_id in ("launchcraft", "launchforge", "yamlcraft"):
        spec = get_engine_spec(engine_id)
        # Authoritative engines register at app startup; before it they are
        # simply absent, which must not imply a policy either.
        assert spec is None or dict(spec.launch_policy) == {}, engine_id


def test_policy_is_not_part_of_the_descriptor_payload():
    for engine_id in ENGINE_REGISTRY:
        assert "launch_policy" not in get_engine_spec(engine_id).to_api_dict()


def test_launch_spec_reads_cpu_support_and_gpu_invariants_from_the_spec():
    from backend.proxy import launch_spec

    for engine_id in ("lmdeploy", "vllm", "1cat_vllm", "sglang", "sglang_v100"):
        assert launch_spec._cpu_supported(engine_id) is False, engine_id
    for engine_id in ("llama_cpp", "ik_llama", "unsloth_llama", "audio_cpp"):
        assert launch_spec._cpu_supported(engine_id) is True, engine_id

    assert launch_spec._gpu_only_invariants("sglang_v100") == (
        "CUDA_HOME",
        "TORCH_CUDA_ARCH_LIST",
        "FLASHINFER_DISABLE_VERSION_CHECK",
    )
    for engine_id in ("vllm", "sglang", "llama_cpp", "audio_cpp"):
        assert launch_spec._gpu_only_invariants(engine_id) == ()
    # Unknown engines keep the permissive defaults (no registry entry).
    assert launch_spec._cpu_supported("not-an-engine") is True
    assert launch_spec._gpu_only_invariants("not-an-engine") == ()


def test_capability_context_reads_the_key_from_the_spec():
    from backend.proxy.commands import _runtime_context_key

    assert _runtime_context_key("llama_cpp") == ("ctx_size",)
    assert _runtime_context_key("lmdeploy") == ("session_len",)
    assert _runtime_context_key("vllm") == ("max_model_len",)
    assert _runtime_context_key("sglang_v100") == ("context_length",)
    assert _runtime_context_key("audio_cpp") == ()
    assert _runtime_context_key(None) == ()
    assert _runtime_context_key("not-an-engine") == ()


def test_definition_round_trips_the_launch_policy():
    definition = validate_definition(
        _raw(
            {
                "cpu_supported": False,
                "runtime_context_key": "context_length",
                "gpu_only_invariants": ["CUDA_HOME"],
            }
        ),
        source="policycraft.yaml",
    )

    assert dict(definition.launch_policy) == {
        "cpu_supported": False,
        "gpu_only_invariants": ("CUDA_HOME",),
        "runtime_context_key": "context_length",
    }
    assert dict(definition.to_spec().launch_policy) == dict(definition.launch_policy)
    assert "launch_policy" not in definition.to_api_dict()


def test_unknown_policy_keys_are_rejected():
    with pytest.raises(DefinitionError, match="launch_policy has unknown keys"):
        validate_definition(_raw({"cpu_allowed": True}), source="policycraft.yaml")


def test_policy_types_are_validated():
    with pytest.raises(DefinitionError, match="cpu_supported"):
        validate_definition(_raw({"cpu_supported": "no"}), source="policycraft.yaml")
    with pytest.raises(DefinitionError, match="gpu_only_invariants"):
        validate_definition(_raw({"gpu_only_invariants": "CUDA_HOME"}), source="policycraft.yaml")
    with pytest.raises(DefinitionError, match="gpu_only_invariants"):
        validate_definition(
            _raw({"gpu_only_invariants": ["not a name"]}), source="policycraft.yaml"
        )


def test_schema_v1_rejects_the_policy_block():
    with pytest.raises(DefinitionError, match="schema_version 2"):
        validate_definition(
            _raw({"cpu_supported": False}, schema_version=1), source="policycraft.yaml"
        )
