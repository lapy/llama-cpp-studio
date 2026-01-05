from backend.architecture_profiles import compute_layers_for_architecture


def test_glm4moe_profile_uses_block_and_nextn():
    metadata = {
        "general.architecture": "glm4moe",
        "glm4moe.block_count": 47,
        "glm4moe.nextn_predict_layers": 1,
    }
    result = compute_layers_for_architecture(
        architecture="glm4moe",
        metadata=metadata,
        base_block_count=47,
    )
    assert result["block_count"] == 47
    assert result["effective_layer_count"] == 48


def test_llama_like_profile_adds_output_head():
    metadata = {
        "general.architecture": "llama",
        "llama.block_count": 32,
    }
    result = compute_layers_for_architecture(
        architecture="llama",
        metadata=metadata,
        base_block_count=32,
    )
    assert result["block_count"] == 32
    assert result["effective_layer_count"] == 33


def test_qwen_family_profile_adds_output_head():
    metadata = {
        "general.architecture": "qwen2",
        "qwen2.block_count": 28,
    }
    result = compute_layers_for_architecture(
        architecture="qwen2",
        metadata=metadata,
        base_block_count=28,
    )
    assert result["block_count"] == 28
    assert result["effective_layer_count"] == 29


def test_generic_profile_uses_base_block_count_plus_one():
    metadata = {
        "general.architecture": "some-new-arch",
    }
    result = compute_layers_for_architecture(
        architecture="some-new-arch",
        metadata=metadata,
        base_block_count=40,
    )
    assert result["block_count"] == 40
    assert result["effective_layer_count"] == 41


def test_generic_profile_falls_back_to_32_when_no_block_count():
    metadata = {
        "general.architecture": "unknown-arch",
    }
    result = compute_layers_for_architecture(
        architecture="unknown-arch",
        metadata=metadata,
        base_block_count=0,
    )
    assert result["block_count"] == 0
    assert result["effective_layer_count"] == 32


