"""Unit tests for --help parsers."""

import os

from backend.cli_help_parsers import (
    lmdeploy_params_to_sections,
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
)


def _param_by_key(params, key):
    return next(p for p in params if p["key"] == key)


def test_parse_llama_snippet_ctx_and_help():
    text = """
----- common params -----
-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    assert sections
    keys = {p["key"] for s in sections for p in s["params"]}
    assert "ctx_size" in keys
    assert "usage" in keys or "help" in keys


def test_parse_llama_strips_cuda_prologue_before_section():
    text = """ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8000 MiB):
  Device 0: Example GPU
----- common params -----

--version                               show version and build info
-c,    --ctx-size N                     size of the prompt context
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    keys = {p["key"] for s in sections for p in s["params"]}
    assert "ctx_size" in keys
    assert "version" in keys


def test_parse_lmdeploy_snippet_port_and_backend():
    text = """
options:
  --server-port SERVER_PORT
                        Server port. Default: 23333. Type: int
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str
"""
    raw = parse_lmdeploy_api_server_help(text)
    keys = {p["key"] for p in raw}
    assert "server_port" in keys
    assert "backend" in keys


def test_parse_lmdeploy_argparse_style_help():
    """Matches ``lmdeploy serve api_server --help`` (argparse: -h, --help + grouped sections)."""
    text = """
positional arguments:
  model_path            The path of a model. Type: str

options:
  -h, --help            show this help message and exit
  --server-name SERVER_NAME
                        Host ip for serving. Default: 0.0.0.0. Type: str
  --backend {pytorch,turbomind}
                        Set the inference backend. Default: turbomind. Type: str

PyTorch engine arguments:
  --tp TP               GPU number used in tensor parallelism. Default: 1. Type: int
  --session-len SESSION_LEN
                        The max session length. Default: None. Type: int

TurboMind engine arguments:
  --dtype {auto,float16,bfloat16}
                        data type for model weights. Default: auto. Type: str
"""
    raw = parse_lmdeploy_api_server_help(text)
    keys = {p["key"] for p in raw}
    assert "help" in keys
    assert "server_name" in keys
    assert "backend" in keys
    assert "tp" in keys
    assert "session_len" in keys
    assert "dtype" in keys
    sections = lmdeploy_params_to_sections(raw)
    ids = {s["id"] for s in sections}
    assert "pytorch_engine_arguments" in ids
    assert "turbomind_engine_arguments" in ids


def test_parse_llama_fixture_excerpt():
    """Excerpt in the style of real ``llama-server --help`` (CUDA prologue + ``-----`` sections); not full upstream text."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "llama_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    keys = {p["key"] for s in sections for p in s["params"]}
    assert len(keys) >= 10
    assert "ctx_size" in keys
    assert "version" in keys
    assert "threads" in keys
    assert "n_predict" in keys
    assert "n_gpu_layers" in keys
    assert "model" in keys
    assert "host" in keys
    assert "port" in keys
    assert "temperature" in keys
    assert "top_k" in keys
    assert "typical_p" in keys
    temperature = _param_by_key([p for s in sections for p in s["params"]], "temperature")
    assert temperature["primary_flag"] == "--temperature"
    assert "--temp" in temperature["flags"]
    assert temperature["scalar_type"] == "float"
    assert temperature["default"] == 0.8
    ids = {s["id"] for s in sections}
    assert ids >= {"common_params", "sampling_params", "example_specific_params"}


def test_parse_lmdeploy_fixture_excerpt():
    """Excerpt copied from a real ``lmdeploy serve api_server --help`` (argparse); not the full upstream text."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "lmdeploy_api_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    raw = parse_lmdeploy_api_server_help(text)
    keys = {p["key"] for p in raw}
    assert len(keys) >= 15
    assert "help" in keys
    assert "server_port" in keys
    assert "tp" in keys
    assert "session_len" in keys
    assert "tool_call_parser" in keys
    assert "vision_max_batch_size" in keys
    assert "speculative_algorithm" in keys
    allow_credentials = _param_by_key(raw, "allow_credentials")
    assert allow_credentials["value_kind"] == "flag"
    assert allow_credentials["type"] == "bool"
    allow_origins = _param_by_key(raw, "allow_origins")
    assert allow_origins["value_kind"] == "repeatable"
    assert allow_origins["multiple"] is True
    tool_call_parser = _param_by_key(raw, "tool_call_parser")
    assert tool_call_parser["value_kind"] == "enum"
    assert {opt["value"] for opt in tool_call_parser["options"]} >= {"internlm", "qwen3"}
    sections = lmdeploy_params_to_sections(raw)
    assert {s["id"] for s in sections} >= {
        "options",
        "pytorch_engine_arguments",
        "turbomind_engine_arguments",
        "vision_model_arguments",
        "speculative_decoding_arguments",
    }


def test_parse_llama_flag_only_and_paired_flags():
    text = """
----- templating params -----
--jinja                               use jinja template parsing
-kvo, --kv-offload, -nkvo, --no-kv-offload
                                      offload KV cache to GPU by default; use --no-kv-offload to disable
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    params = [p for s in sections for p in s["params"]]
    jinja = _param_by_key(params, "jinja")
    assert jinja["value_kind"] == "flag"
    assert jinja["primary_flag"] == "--jinja"
    kv_offload = _param_by_key(params, "kv_offload")
    assert kv_offload["value_kind"] == "flag"
    assert kv_offload["primary_flag"] == "--kv-offload"
    assert kv_offload["negative_flag"] == "--no-kv-offload"
