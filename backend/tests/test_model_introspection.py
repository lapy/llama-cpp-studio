from backend.model_introspection import GgufIntrospector


def test_context_length_prefers_largest_and_uses_config_global():
    # global config prefers general.context_length / model_max_length / max_position_embeddings
    metadata = {
        "general.context_length": 4096,
        "general.model_max_length": 8192,
        "qwen.context_length": 2048,
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

