"""Schema/loader validation gates for YAML engine definitions (Phase 2).

Each test pins one startup-rejection guarantee from
docs/config-driven-engine-refactoring-plan.md: definitions are validated once
and rejected precisely — duplicate keys/IDs, unknown fields or kinds, missing
references, inheritance cycles, invalid conditions, ambiguous selectors.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest

from backend.engines.definitions import DEFINITIONS_DIR, load_engine_definitions
from backend.engines.definitions.loader import TEMPLATES_FILENAME
from backend.engines.definitions.schema import (
    DefinitionError,
    load_yaml_strict,
    validate_asset_selectors,
    validate_condition,
)

BASE_ENGINE = {
    "schema_version": 1,
    "id": "fixture_engine",
    "label": "Fixture",
    "repository_source": "Fixture",
    "install_kind": "python_venv",
    "runtime_kind": "vllm",
    "scanner_kind": "vllm_help",
    "active_path_fields": ["venv_path"],
    "artifact_formats": ["safetensors"],
    "package_kinds": ["hf_snapshot"],
    "tasks": ["text-generation"],
    "input_modalities": ["text"],
    "output_modalities": ["text"],
}


def _write_definitions(tmp_path, engine_doc: Dict, templates_doc=None):
    if templates_doc is None:
        templates_doc = {"schema_version": 1, "templates": {}}
    (tmp_path / TEMPLATES_FILENAME).write_text(_dump(templates_doc), encoding="utf-8")
    (tmp_path / f"{engine_doc['id']}.yaml").write_text(_dump(engine_doc), encoding="utf-8")
    return tmp_path


def _dump(document) -> str:
    import yaml

    return yaml.safe_dump(document, sort_keys=False)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def test_duplicate_yaml_keys_rejected():
    text = "id: a\nid: b\n"
    with pytest.raises(DefinitionError, match="duplicate key"):
        load_yaml_strict(text, source="mem")


def test_malformed_yaml_rejected():
    with pytest.raises(DefinitionError, match="malformed YAML"):
        load_yaml_strict("id: [unclosed", source="mem")


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------


def _load_one(tmp_path, mutate=None, templates=None):
    doc = dict(BASE_ENGINE)
    if mutate:
        mutate(doc)
    _write_definitions(tmp_path, doc, templates)
    return load_engine_definitions(str(tmp_path))


def test_valid_minimal_definition(tmp_path):
    registry = _load_one(tmp_path)
    definition = registry.get("fixture_engine")
    assert definition is not None
    assert definition.supports_embeddings is False
    assert definition.experimental is False
    assert definition.maturity_surfaces == ()


def test_unknown_field_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="unknown definition fields"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("recipe_hooks", ["x"]))


def test_missing_required_field_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="runtime_kind"):
        _load_one(tmp_path, mutate=lambda d: d.pop("runtime_kind"))


def test_unknown_install_kind_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="install_kind"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("install_kind", "docker"))


def test_unknown_runtime_kind_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="runtime_kind"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("runtime_kind", "trt_llm"))


def test_unknown_scanner_kind_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="scanner_kind"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("scanner_kind", "magic"))


def test_unsupported_schema_version_rejected(tmp_path):
    # v2 is supported (reference spike); a future version must still be
    # rejected rather than partially interpreted.
    with pytest.raises(DefinitionError, match="schema_version"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("schema_version", 99))


def test_list_typed_field_as_string_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="active_path_fields"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("active_path_fields", "venv_path"))


def test_duplicate_tokens_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="duplicates"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("tasks", ["a", "a"]))


def test_bool_field_as_int_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="supports_embeddings"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("supports_embeddings", 1))


def test_empty_list_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="must not be empty"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("artifact_formats", []))


def test_unknown_maturity_status_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="unknown status"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("maturity_surfaces", {"asr": "beta"}))


def test_invalid_engine_id_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="must match"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("id", "Bad ID"))


# ---------------------------------------------------------------------------
# Identity and template references
# ---------------------------------------------------------------------------


def test_filename_stem_must_match_id(tmp_path):
    (tmp_path / TEMPLATES_FILENAME).write_text(
        _dump({"schema_version": 1, "templates": {}}), encoding="utf-8"
    )
    (tmp_path / "other_name.yaml").write_text(_dump(BASE_ENGINE), encoding="utf-8")
    with pytest.raises(DefinitionError, match="does not match filename"):
        load_engine_definitions(str(tmp_path))


def test_same_id_in_two_files_rejected(tmp_path):
    """Duplicate ids are caught either directly or via the filename-stem rule;
    both rejections are startup errors, the stem rule simply fires first."""
    _write_definitions(tmp_path, BASE_ENGINE)
    (tmp_path / "fixture_engine_2.yaml").write_text(_dump(BASE_ENGINE), encoding="utf-8")
    with pytest.raises(DefinitionError, match="duplicate engine id|does not match filename"):
        load_engine_definitions(str(tmp_path))


def test_unknown_template_reference_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="extends unknown template"):
        _load_one(tmp_path, mutate=lambda d: d.__setitem__("extends", ["nonexistent"]))


def test_template_cycle_rejected(tmp_path):
    templates = {
        "schema_version": 1,
        "templates": {
            "a": {"extends": ["b"], "label": "A"},
            "b": {"extends": ["a"], "label": "B"},
        },
    }
    doc = {k: v for k, v in BASE_ENGINE.items() if k not in ("label",)}
    doc["extends"] = ["a"]
    with pytest.raises(DefinitionError, match="cycle"):
        _write_then(tmp_path, doc, templates)


def _write_then(tmp_path, engine_doc, templates):
    _write_definitions(tmp_path, engine_doc, templates)
    return load_engine_definitions(str(tmp_path))


def test_template_deterministic_merge(tmp_path):
    """Later extends override earlier; engine keys override all; maps merge per key."""
    templates = {
        "schema_version": 1,
        "templates": {
            "base": {
                "label": "From base",
                "maturity_surfaces": {"asr": "experimental", "tts": "stable"},
            },
            "second": {"label": "From second", "maturity_surfaces": {"asr": "stable"}},
        },
    }
    doc = {k: v for k, v in BASE_ENGINE.items() if k != "label"}
    doc["extends"] = ["base", "second"]
    doc["maturity_surfaces"] = {"tts": "specialized"}
    registry = _write_then(tmp_path, doc, templates)
    definition = registry.get("fixture_engine")
    assert definition.label == "From second"
    assert dict(definition.maturity_surfaces) == {"asr": "stable", "tts": "specialized"}


def test_shipped_definitions_load_and_match_directory():
    registry = load_engine_definitions(DEFINITIONS_DIR)
    # every shipped engine has exactly one definition file
    assert len(registry.ids()) == 11


# ---------------------------------------------------------------------------
# Condition vocabulary
# ---------------------------------------------------------------------------


def test_valid_conditions():
    validate_condition(
        {
            "all": [
                {"exists": {"field": "request.repo_url"}},
                {
                    "any": [
                        {"equals": {"field": "host.os", "value": "linux"}},
                        {
                            "capability_contains": {
                                "field": "gpu.compute_capabilities",
                                "value": "sm70",
                            }
                        },
                    ]
                },
                {"version_range": {"field": "toolchain.cmake", "min": "3.22", "max": "4.9"}},
            ]
        },
        "t",
    )


@pytest.mark.parametrize(
    "node",
    [
        {"all": []},
        {"magic": {"field": "x"}},
        {"equals": {"field": "x", "value": 1, "extra": 2}},
        {"equals": {"value": 1}},
        {"exists": {"field": "x", "value": 1}},
        {"all": {"field": "x"}},
        {"all": [{"bad_op": {"field": "x"}}]},
        "equals",
        {"equals": {"field": "x"}, "exists": {"field": "y"}},
    ],
)
def test_invalid_conditions_rejected(node):
    with pytest.raises(DefinitionError):
        validate_condition(node, "t")


# ---------------------------------------------------------------------------
# Asset selector vocabulary
# ---------------------------------------------------------------------------


def test_valid_asset_selectors():
    selectors = validate_asset_selectors(
        [
            {"name": "cuda12", "patterns": ["studio-*cuda12*.tar.gz"], "default": True},
            {"name": "cpu", "patterns": ["studio-*cpu*.tar.gz"]},
        ],
        "t",
    )
    assert selectors[0]["default"] is True
    assert selectors[1]["default"] is False


@pytest.mark.parametrize(
    "selectors",
    [
        [],
        [{"name": "a", "patterns": ["x*"]}, {"name": "a", "patterns": ["y*"]}],
        [
            {"name": "a", "patterns": ["x*"], "default": True},
            {"name": "b", "patterns": ["y*"], "default": True},
        ],
        [{"name": "a", "patterns": ["x*"], "checksum": "sha256:zz"}],
        [{"name": "a", "patterns": []}],
    ],
)
def test_ambiguous_asset_selectors_rejected(selectors):
    with pytest.raises(DefinitionError):
        validate_asset_selectors(selectors, "t")


# ---------------------------------------------------------------------------
# Loader path errors
# ---------------------------------------------------------------------------


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(DefinitionError, match="definitions directory not found"):
        load_engine_definitions(os.path.join(str(tmp_path), "nope"))
