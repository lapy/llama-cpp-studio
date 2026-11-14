from backend.engines.metrics_flags import (
    MANAGED_METRIC_ARGS,
    managed_log_args,
    managed_metric_args,
    managed_metric_flag_set,
    owned_log_flag_set,
    owned_observability_flag_set,
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


def test_vllm_kv_cache_metrics_are_managed_when_supported():
    index = _index(
        "--enable-prompt-tokens-details",
        "--enable-force-include-usage",
        "--kv-cache-metrics",
    )
    assert managed_metric_args("vllm", index) == [
        "--enable-prompt-tokens-details",
        "--enable-force-include-usage",
        "--kv-cache-metrics",
    ]
    assert managed_metric_args("vllm", _index("--enable-prompt-tokens-details")) == [
        "--enable-prompt-tokens-details"
    ]


def test_audio_log_stream_is_managed_when_catalog_supports_it():
    assert managed_log_args("audio_cpp", _index("--log", "--metrics")) == ["--log"]
    assert managed_log_args("audio_cpp", _index("--metrics")) == []
    assert managed_log_args("llama_cpp", _index("--log")) == []


def test_log_destination_settings_are_normalized_before_emission():
    index = {
        "log_disable": {
            "flags": ["--log-disable"],
            "primary_flag": "--log-disable",
        },
        "log_file": {
            "flags": ["--log-file"],
            "primary_flag": "--log-file",
        },
        "temperature": {
            "flags": ["--temperature"],
            "primary_flag": "--temperature",
        },
    }
    config = {
        "log_disable": True,
        "log_file": "/tmp/engine.log",
        "temperature": 0.4,
        "custom_args": "--log-file /tmp/other.log",
    }
    normalized = strip_managed_config(config, "llama_cpp", index)
    assert "log_disable" not in normalized
    assert "log_file" not in normalized
    assert normalized["temperature"] == 0.4
    assert normalized["custom_args"] == "--log-file /tmp/other.log"


def test_log_destination_surface_is_reserved_even_when_build_omits_a_flag():
    assert "--log-file" in owned_log_flag_set("llama_cpp")
    assert "--log-disable" in owned_observability_flag_set("llama_cpp")
    assert "--log-file" in owned_log_flag_set("audio_cpp")
    assert "--log-level" in owned_log_flag_set("lmdeploy")
    assert "--disable-log-stats" in owned_log_flag_set("vllm")
    assert "--log-requests-target" in owned_log_flag_set("sglang")
