"""GGUF reader small pure helpers."""

from backend.gguf_reader import _compute_layers_for_architecture


def test_compute_layers_default():
    r = _compute_layers_for_architecture("llama", {}, 32)
    assert r["block_count"] == 32
    assert r["effective_layer_count"] == 33


def test_compute_layers_glm4moe_nextn():
    r = _compute_layers_for_architecture(
        "glm4moe",
        {"glm4moe.nextn_predict_layers": 3},
        40,
    )
    assert r["effective_layer_count"] == 43
