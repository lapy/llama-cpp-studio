"""R72 pins: upstream identity is engine data, not a map in repo_identity.

`CANONICAL_REPOSITORY_URLS` (nine engine-id entries outside the engine home) is
gone: each engine's spec declares `upstream.repository_url`, the exporter writes
it into the generated definition, and fork detection reads it from the spec.
"""

from __future__ import annotations

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import get_engine_spec
from backend.repo_identity import (
    canonical_repository_url,
    is_github_fork_for_engine,
    source_build_type_labels_for_engine,
)

EXPECTED_UPSTREAMS = {
    "llama_cpp": "https://github.com/ggerganov/llama.cpp.git",
    "ik_llama": "https://github.com/ikawrakow/ik_llama.cpp.git",
    "unsloth_llama": "https://github.com/unslothai/llama.cpp.git",
    "audio_cpp": "https://github.com/0xShug0/audio.cpp.git",
    "lmdeploy": "https://github.com/InternLM/lmdeploy.git",
    "1cat_vllm": "https://github.com/1CatAI/1Cat-vLLM.git",
    "sglang": "https://github.com/sgl-project/sglang.git",
    "sglang_v100": "https://github.com/haohervchb/sglang-V100.git",
    "vllm": "https://github.com/vllm-project/vllm.git",
}


def test_every_generated_engine_declares_its_upstream():
    for engine_id, url in EXPECTED_UPSTREAMS.items():
        spec = get_engine_spec(engine_id)
        assert spec is not None, engine_id
        assert dict(spec.upstream) == {"repository_url": url}, engine_id
        assert canonical_repository_url(engine_id) == url
        assert "upstream" not in spec.to_api_dict(), engine_id


def test_upstream_is_exported_with_the_definition():
    definitions = load_default_engine_definitions()
    for engine_id, url in EXPECTED_UPSTREAMS.items():
        definition = definitions.get(engine_id)
        assert definition is not None, engine_id
        assert dict(definition.upstream) == {"repository_url": url}, engine_id


def test_fork_detection_reads_the_declared_upstream(monkeypatch):
    class FakeSpec:
        upstream = {"repository_url": "https://github.com/upstream-owner/tool.git"}

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())
    assert canonical_repository_url("tool_engine") == ("https://github.com/upstream-owner/tool.git")
    assert is_github_fork_for_engine("tool_engine", "https://github.com/alice/tool.git")
    assert not is_github_fork_for_engine(
        "tool_engine", "https://github.com/upstream-owner/tool.git"
    )
    assert (
        source_build_type_labels_for_engine("tool_engine", "https://github.com/alice/tool.git")[
            "type"
        ]
        == "fork"
    )


def test_engine_without_an_upstream_is_not_a_fork(monkeypatch):
    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: None)
    assert canonical_repository_url("nonexistent") == ""
    assert is_github_fork_for_engine("nonexistent", "https://github.com/alice/x.git") is False
    assert (
        source_build_type_labels_for_engine("nonexistent", "https://github.com/alice/x.git")["type"]
        == "source"
    )


def test_definition_round_trips_upstream():
    raw = {
        "schema_version": 2,
        "id": "upstreamcraft",
        "label": "Upstreamcraft",
        "repository_source": "Example Org/Upstreamcraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary_path"],
        "artifact_formats": ["builtin"],
        "package_kinds": ["builtin"],
        "tasks": ["tts"],
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "upstream": {"repository_url": "https://github.com/acme/upstreamcraft.git"},
    }
    definition = validate_definition(raw, source="upstreamcraft.yaml")
    assert dict(definition.upstream) == raw["upstream"]
    assert dict(definition.to_spec().upstream) == raw["upstream"]
    assert "upstream" not in definition.to_api_dict()


def test_malformed_upstream_is_rejected():
    base = {
        "schema_version": 2,
        "id": "upstreamcraft",
        "label": "Upstreamcraft",
        "repository_source": "Example Org/Upstreamcraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary_path"],
        "artifact_formats": ["builtin"],
        "package_kinds": ["builtin"],
        "tasks": ["tts"],
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
    }
    with pytest.raises(DefinitionError, match="must be a remote URL"):
        validate_definition({**base, "upstream": {"repository_url": "acme/tool"}}, source="u.yaml")
    with pytest.raises(DefinitionError, match="unknown keys"):
        validate_definition(
            {**base, "upstream": {"url": "https://github.com/a/b.git"}}, source="u.yaml"
        )
    with pytest.raises(DefinitionError, match="schema_version 2"):
        validate_definition(
            {
                **base,
                "schema_version": 1,
                "upstream": {"repository_url": "https://github.com/a/b.git"},
            },
            source="u.yaml",
        )
