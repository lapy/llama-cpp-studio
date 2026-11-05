"""R77 pins: the descriptor payload exposes `cpu_supported`.

The frontend hardware filter used to special-case the `audio_cpp` engine id
(`!formats.has('gguf') && descriptor.id !== 'audio_cpp'`). The payload now
carries the declared launch-policy fact as an explicit field, and the filter
reads it — no engine-id special case.
"""

from __future__ import annotations

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.registry import get_engine_spec

CPU_SUPPORTED = {"audio_cpp": True}
GPU_ONLY = {"lmdeploy", "vllm", "1cat_vllm", "sglang", "sglang_v100"}


def test_cpu_supported_is_an_explicit_descriptor_field():
    assert get_engine_spec("audio_cpp").to_api_dict()["cpu_supported"] is True
    for engine_id in GPU_ONLY:
        assert get_engine_spec(engine_id).to_api_dict()["cpu_supported"] is False, engine_id
    # gguf engines pass the CPU filter through their artifact format; the field
    # itself stays false unless the engine declares CPU support.
    for engine_id in ("llama_cpp", "ik_llama", "unsloth_llama"):
        payload = get_engine_spec(engine_id).to_api_dict()
        assert payload["cpu_supported"] is False, engine_id
        assert "gguf" in payload["artifact_formats"], engine_id


def test_the_launch_policy_block_stays_out_of_the_payload():
    for engine_id in ("audio_cpp", "vllm", "llama_cpp"):
        payload = get_engine_spec(engine_id).to_api_dict()
        assert "launch_policy" not in payload, engine_id
        assert "cpu_supported" in payload, engine_id


def test_audio_declares_cpu_support_in_its_definition():
    definition = load_default_engine_definitions().definitions["audio_cpp"]
    policy = dict(definition.launch_policy)
    assert policy.get("cpu_supported") is True
    spec = definition.to_spec()
    assert dict(spec.launch_policy).get("cpu_supported") is True
    assert spec.to_api_dict()["cpu_supported"] is True


def test_engines_without_a_declaration_default_to_gpu_only():
    assert get_engine_spec("llama_cpp").to_api_dict()["cpu_supported"] is False


def test_gpu_archs_are_declared_and_exposed():
    # R78: the sm70 filter reads this instead of matching engine ids.
    for engine_id in ("sglang_v100", "1cat_vllm"):
        spec = get_engine_spec(engine_id)
        assert list(dict(spec.launch_policy).get("gpu_archs") or ()) == ["sm70"], engine_id
        assert spec.to_api_dict()["gpu_archs"] == ["sm70"], engine_id
    vllm = get_engine_spec("vllm").to_api_dict()["gpu_archs"]
    sglang = get_engine_spec("sglang").to_api_dict()["gpu_archs"]
    assert "sm70" not in sglang
    assert "sm75" in sglang
    assert "sm70" not in vllm
    assert "sm75" in vllm
    for engine_id in ("llama_cpp", "audio_cpp"):
        assert get_engine_spec(engine_id).to_api_dict()["gpu_archs"] == [], engine_id
    definition = load_default_engine_definitions().definitions["sglang_v100"]
    assert list(dict(definition.launch_policy).get("gpu_archs") or ()) == ["sm70"]
