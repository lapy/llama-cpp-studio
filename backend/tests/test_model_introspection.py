from backend.model_introspection import GgufIntrospector


def test_context_length_prefers_canonical_arch_key_over_general_fallback():
    metadata = {
        "general.architecture": "qwen",
        "general.context_length": 4096,
        "general.model_max_length": 8192,
        "qwen.context_length": 2048,
    }
    introspector = GgufIntrospector(metadata=metadata, tensors={})
    info = introspector.build_model_info()
    assert info.context_length == 2048


def test_context_length_ignores_rope_original_context_fallback():
    metadata = {
        "general.architecture": "unknown_arch",
        "rope.scaling.original_context_length": 131072,
        "max_position_embeddings": 8192,
    }
    introspector = GgufIntrospector(metadata=metadata, tensors={})
    info = introspector.build_model_info()
    assert info.context_length == 8192


def test_parameter_count_parses_formatted_and_raw_values():
    metadata = {
        "general.parameters": "7B",
        "general.parameter_count": 6_000_000_000,
    }
    introspector = GgufIntrospector(metadata=metadata, tensors={})
    info = introspector.build_model_info()
    # 7B should win over 6B
    assert info.parameter_count_display in {"7B", "7.0B"}


def test_moe_detection_from_expert_keys():
    metadata = {
        "general.architecture": "glm4moe",
        "ffn.expert_count": 64,
        "ffn.num_experts_per_tok": 8,
    }
    introspector = GgufIntrospector(metadata=metadata, tensors={})
    info = introspector.build_model_info()
    assert info.is_moe is True
    assert info.expert_count == 64
    assert info.experts_used_count == 8


def test_layer_count_does_not_use_leading_dense_block_count_as_total():
    metadata = {
        "general.architecture": "unknown_arch",
        "unknown_arch.leading_dense_block_count": 2,
    }
    introspector = GgufIntrospector(metadata=metadata, tensors={})
    info = introspector.build_model_info()
    assert info.block_count == 0
    assert info.layer_count == 0


def test_vocab_and_embedding_from_tensors_when_metadata_missing():
    tensors = {
        "tok_embeddings.weight": {
            "shape": [32000, 4096],
            "type": 0,
            "offset": 0,
        }
    }
    introspector = GgufIntrospector(metadata={}, tensors=tensors)
    info = introspector.build_model_info()
    assert info.vocab_size == 32000
    assert info.embedding_length == 4096
    assert info.layer_count == 0
