"""R94 pins: the llama-family build-option catalog is engine data.

The table used to live inside ``llama_cpp/build_options.py`` and later a Python
data module. It is now declared by the authoritative engine definitions, is
carried on both llama-family specs from those definitions, and the runtime
module derives its option dataclasses from the spec block.
"""

from __future__ import annotations

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.llama_cpp import build_options as bo
from backend.engines.registry import get_engine_spec


def test_both_llama_family_specs_carry_the_catalog():
    definitions = load_default_engine_definitions().definitions
    for engine_id in ("llama_cpp", "ik_llama"):
        spec = get_engine_spec(engine_id)
        assert dict(spec.build_options) == dict(definitions[engine_id].build_options), engine_id
        assert "build_options" not in spec.to_api_dict(), engine_id


def test_the_definition_carries_the_catalog():
    definitions = load_default_engine_definitions().definitions
    assert dict(definitions["llama_cpp"].build_options) == dict(
        definitions["ik_llama"].build_options
    )
    assert dict(definitions["llama_cpp"].build_options)["options"]


def test_module_options_derive_from_the_spec_block():
    raw = [
        {
            "key": "opt",
            "field": "enable_opt",
            "kind": "bool",
            "default": False,
            "label": "Opt",
            "desc": "GGML_OPT",
            "category": "backends",
            "cmake": "GGML_OPT",
            "engines": ["ik_llama"],
        }
    ]
    block = {
        "build_types": ["Only"],
        "build_type_default": "Only",
        "primary_keys": ["opt"],
        "categories": [{"id": "backends", "label": "Backends", "collapsed": False}],
        "options": raw,
    }

    class FakeSpec:
        build_options = block

    monkeypatch = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch.setattr(
            "backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec()
        )
        assert bo._catalog() == block
    finally:
        monkeypatch.undo()


def test_catalog_payload_still_filters_per_engine():
    llama = bo.catalog_for_ui("llama_cpp")
    ik = bo.catalog_for_ui("ik_llama")
    llama_keys = {o["key"] for c in llama["categories"] for o in c["options"]}
    ik_keys = {o["key"] for c in ik["categories"] for o in c["options"]}
    assert "cuda" in llama_keys and "cuda" in ik_keys
    assert "iqk_mul_mat" in ik_keys and "iqk_mul_mat" not in llama_keys
    assert "blas" in llama_keys and "blas" not in ik_keys
    # Every option advertises its engine scope (payload contract preserved).
    for entry in next(c for c in llama["categories"] if c["id"] == "backends")["options"]:
        assert entry["engines"]
    assert llama["build_types"] == ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"]
    assert llama["defaults"]["build_type"] == "Release"


def test_union_defaults_cover_both_engines():
    defaults = bo.default_build_settings()
    # Union view keeps every option (stored configs are engine-agnostic).
    assert "blas" in defaults and "iqk_mul_mat" in defaults
    catalog = dict(get_engine_spec("llama_cpp").build_options)
    assert len(bo.BUILD_OPTIONS) == len(catalog["options"])
