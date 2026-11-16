"""Default fragments parsed out of engine help prose."""

from backend.engines.scan.default_fragments import (
    default_from_description,
    default_is_value,
    documented_default_token,
    is_documentation_default_artifact,
    normalize_default_fragment,
)


def test_prose_after_default_is_not_a_default():
    # The audio.cpp server help describes derived behavior, not a value.
    text = "optional backend implementation; default preserves existing dispatch"
    assert default_from_description(text) is None
    assert default_is_value("preserves", " existing dispatch", False) is False


def test_derived_default_prose_is_not_a_default():
    assert default_from_description("the default comes from the model spec") is None


def test_stated_defaults_are_parsed():
    assert default_from_description("server bind address; default 127.0.0.1") == "127.0.0.1"
    assert default_from_description("default cuda (rocm is an alias for hip)") == "cuda"
    assert default_from_description("fail after; default 300000, 0 disables") == "300000"
    assert default_from_description("threads (default: 4)") == "4"
    assert default_from_description("mode: default=auto") == "auto"
    assert default_from_description("keep the cache; default 'on'") == "on"
    assert default_from_description("default true") == "true"


def test_documented_token_keeps_the_form_and_the_tail():
    assert documented_default_token("default: 4") == ("4", "", True)
    assert documented_default_token("default preserves existing dispatch") == (
        "preserves",
        " existing dispatch",
        False,
    )
    assert documented_default_token("no such clause in this help") is None


def test_help_artifact_detection_matches_only_the_prose_value():
    prose = "optional backend implementation; default preserves existing dispatch"
    assert is_documentation_default_artifact(prose, "preserves") is True
    assert is_documentation_default_artifact(prose, "ggml") is False
    stated = "backend implementation; default: ggml"
    assert is_documentation_default_artifact(stated, "ggml") is False


def test_moved_normalizer_keeps_the_llama_help_semantics():
    # Kept verbatim from help_parsers: comma tails, quotes, "0 = disabled".
    assert normalize_default_fragment("40, 0 = disabled") == "40"
    assert normalize_default_fragment("'auto'") == "auto"
    assert normalize_default_fragment("EPLBConfig(window_size=1000, step=3000)") == (
        "EPLBConfig(window_size=1000, step=3000)"
    )
    assert normalize_default_fragment("   ") is None
