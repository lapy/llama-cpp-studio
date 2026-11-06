from backend.engines.metrics_flags import (
    MANAGED_METRIC_ARGS,
    managed_metric_args,
    managed_metric_flag_set,
    strip_managed_config,
)


def _index(*flags: str) -> dict:
    return {
        flag.lstrip("-").replace("-", "_"): {
            "flags": [flag, flag.replace("--enable", "--no-enable")],
            "primary_flag": flag,
        }
        for flag in flags
    }


def test_managed_flags_are_added_only_when_scanned_catalog_supports_them():
    supported = _index("--enable-cache-report", "--enable-metrics")
    assert managed_metric_args("sglang_v100", supported) == [
        "--enable-cache-report",
        "--enable-metrics",
    ]
    assert "--stream-response-default-include-usage" not in managed_metric_args(
        "sglang_v100", supported
    )


def test_managed_flags_are_not_added_for_unrelated_engines():
    index = _index(*MANAGED_METRIC_ARGS["1cat_vllm"])
    assert managed_metric_args("llama_cpp", index) == []
    assert managed_metric_args(None, index) == []


def test_managed_surface_is_reserved_even_when_build_omits_a_flag():
    assert "--enable-cache-report" in managed_metric_flag_set("sglang")
    assert "--enable-prompt-tokens-details" in managed_metric_flag_set("1cat_vllm")
    assert "--metrics" in managed_metric_flag_set("llama_cpp")
    assert managed_metric_flag_set("lmdeploy") == frozenset()


def test_existing_managed_settings_are_normalized_before_emission():
    index = {
        "enable_metrics": {
            "flags": ["--enable-metrics"],
            "primary_flag": "--enable-metrics",
        },
        "disable_cuda_graph": {
            "flags": ["--disable-cuda-graph"],
            "primary_flag": "--disable-cuda-graph",
        },
    }
    config = {"enable_metrics": True, "disable_cuda_graph": True, "custom_args": "--foo"}
    normalized = strip_managed_config(config, "sglang_v100", index)
    assert "enable_metrics" not in normalized
    assert normalized["disable_cuda_graph"] is True
    assert normalized["custom_args"] == "--foo"


def test_llama_family_metric_flags_are_managed_when_catalog_supports_them():
    index = _index("--metrics", "--perf")
    assert managed_metric_args("llama_cpp", index) == ["--metrics", "--perf"]
    assert managed_metric_args("unsloth_llama", index) == ["--metrics", "--perf"]
