"""Unit tests for --help parsers."""

import os

from backend.cli_help_parsers import (
    lmdeploy_params_to_sections,
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
)
from backend.engine_param_catalog import embedding_mode_config_key_from_entry


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
    """Full verbatim ``llama-server --help`` (CUDA prologue + ``-----`` section banners)."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "llama_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    keys = [p["key"] for p in flat]
    assert len(keys) == len(set(keys)), "duplicate config keys after merge"
    assert len(flat) >= 200
    keys_set = set(keys)
    assert keys_set >= {
        "ctx_size",
        "version",
        "threads",
        "n_predict",
        "n_gpu_layers",
        "model",
        "host",
        "port",
        "temperature",
        "top_k",
        "typical_p",
        "rope_scaling",
        "pooling",
        "embeddings",
        "flash_attn",
        "samplers",
        "cache_list",
        "list_devices",
    }
    ids = {s["id"] for s in sections}
    assert ids == {"common_params", "sampling_params", "example_specific_params"}
    assert all(p.get("flags") for p in flat)
    assert all((p.get("description") or "").strip() for p in flat)

    temperature = _param_by_key(flat, "temperature")
    assert temperature["primary_flag"] == "--temperature"
    assert "--temp" in temperature["flags"]
    assert temperature["value_kind"] == "scalar"
    assert temperature["type"] == "float"
    assert temperature["scalar_type"] == "float"
    assert temperature["default"] == 0.8

    rope_scaling = _param_by_key(flat, "rope_scaling")
    assert rope_scaling["value_kind"] == "enum"
    assert rope_scaling["type"] == "select"
    assert {opt["value"] for opt in (rope_scaling.get("options") or [])} >= {
        "none",
        "linear",
        "yarn",
    }

    pooling = _param_by_key(flat, "pooling")
    assert pooling["value_kind"] == "enum"
    assert pooling["type"] == "select"
    assert {opt["value"] for opt in (pooling.get("options") or [])} >= {
        "none",
        "mean",
        "cls",
        "last",
        "rank",
    }

    flash_attn = _param_by_key(flat, "flash_attn")
    assert flash_attn["value_kind"] == "enum"
    assert flash_attn["type"] == "select"
    assert {opt["value"] for opt in (flash_attn.get("options") or [])} == {
        "on",
        "off",
        "auto",
    }

    samplers = _param_by_key(flat, "samplers")
    assert samplers["value_kind"] == "scalar"
    assert samplers["type"] == "string"

    embeddings = _param_by_key(flat, "embeddings")
    assert embeddings["value_kind"] == "flag"
    assert embeddings["type"] == "bool"
    assert set(embeddings["flags"]) >= {"--embedding", "--embeddings"}
    assert embedding_mode_config_key_from_entry({"sections": sections}) == "embeddings"

    escape = _param_by_key(flat, "escape")
    assert escape["value_kind"] == "flag"
    assert escape["type"] == "bool"
    assert "--no-escape" in escape["flags"]

    cache_list = _param_by_key(flat, "cache_list")
    assert cache_list["value_kind"] == "flag"
    assert cache_list["type"] == "bool"

    list_devices = _param_by_key(flat, "list_devices")
    assert list_devices["value_kind"] == "flag"
    assert list_devices["type"] == "bool"

    tools = _param_by_key(flat, "tools")
    assert tools["value_kind"] == "repeatable"
    assert tools["type"] == "list"
    assert tools["multiple"] is True

    override_tensor = _param_by_key(flat, "override_tensor")
    assert override_tensor["value_kind"] == "repeatable"
    assert override_tensor["type"] == "list"

    poll = _param_by_key(flat, "poll")
    assert poll["value_kind"] == "scalar"
    assert poll["type"] == "int"

    chat_template = _param_by_key(flat, "chat_template")
    assert chat_template["value_kind"] == "scalar"
    assert chat_template["type"] == "string"


def test_parse_lmdeploy_fixture_excerpt():
    """Full verbatim ``lmdeploy serve api_server --help`` (argparse layout)."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "lmdeploy_api_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    raw = parse_lmdeploy_api_server_help(text)
    keys_list = [p["key"] for p in raw]
    keys_set = set(keys_list)
    assert len(keys_list) == len(keys_set), "duplicate config keys after merge"
    assert len(raw) >= 60
    assert keys_set >= {
        "help",
        "server_port",
        "server_name",
        "tp",
        "session_len",
        "tool_call_parser",
        "vision_max_batch_size",
        "speculative_algorithm",
        "backend",
        "dtype",
        "device",
        "adapters",
        "rope_scaling_factor",
        "communicator",
    }
    assert all(p.get("flags") for p in raw)
    assert all((p.get("description") or "").strip() for p in raw)

    allow_credentials = _param_by_key(raw, "allow_credentials")
    assert allow_credentials["value_kind"] == "flag"
    assert allow_credentials["type"] == "bool"

    ssl = _param_by_key(raw, "ssl")
    assert ssl["value_kind"] == "flag"
    assert ssl["type"] == "bool"

    allow_origins = _param_by_key(raw, "allow_origins")
    assert allow_origins["value_kind"] == "repeatable"
    assert allow_origins["type"] == "list"
    assert allow_origins["multiple"] is True

    api_keys = _param_by_key(raw, "api_keys")
    assert api_keys["value_kind"] == "repeatable"
    assert api_keys["type"] == "list"

    adapters = _param_by_key(raw, "adapters")
    assert adapters["value_kind"] == "repeatable"
    assert adapters["type"] == "list"

    server_port = _param_by_key(raw, "server_port")
    assert server_port["value_kind"] == "scalar"
    assert server_port["type"] == "int"
    assert server_port["scalar_type"] == "int"

    backend = _param_by_key(raw, "backend")
    assert backend["value_kind"] == "enum"
    assert backend["type"] == "select"
    assert {opt["value"] for opt in (backend.get("options") or [])} == {
        "pytorch",
        "turbomind",
    }

    log_level = _param_by_key(raw, "log_level")
    assert log_level["value_kind"] == "enum"
    assert log_level["type"] == "select"
    assert {opt["value"] for opt in (log_level.get("options") or [])} >= {
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
    }

    device = _param_by_key(raw, "device")
    assert device["value_kind"] == "enum"
    assert device["type"] == "select"
    assert {opt["value"] for opt in (device.get("options") or [])} == {
        "cuda",
        "ascend",
        "maca",
        "camb",
    }

    quant_policy = _param_by_key(raw, "quant_policy")
    assert quant_policy["value_kind"] == "enum"
    assert quant_policy["type"] == "select"
    assert {opt["value"] for opt in (quant_policy.get("options") or [])} == {
        "0",
        "4",
        "8",
    }

    tool_call_parser = _param_by_key(raw, "tool_call_parser")
    assert tool_call_parser["value_kind"] == "enum"
    assert tool_call_parser["type"] == "select"
    assert {opt["value"] for opt in tool_call_parser["options"]} >= {
        "internlm",
        "qwen3",
        "llama3",
    }

    dtype = _param_by_key(raw, "dtype")
    assert dtype["value_kind"] == "enum"
    assert dtype["type"] == "select"
    assert {opt["value"] for opt in (dtype.get("options") or [])} == {
        "auto",
        "float16",
        "bfloat16",
    }

    async_opt = _param_by_key(raw, "async")
    assert async_opt["value_kind"] == "enum"
    assert async_opt["type"] == "select"
    assert {opt["value"] for opt in (async_opt.get("options") or [])} == {"0", "1"}

    speculative_algorithm = _param_by_key(raw, "speculative_algorithm")
    assert speculative_algorithm["value_kind"] == "enum"
    assert speculative_algorithm["type"] == "select"
    assert {opt["value"] for opt in (speculative_algorithm.get("options") or [])} >= {
        "eagle",
        "eagle3",
        "deepseek_mtp",
    }

    rope_scaling_factor = _param_by_key(raw, "rope_scaling_factor")
    assert rope_scaling_factor["value_kind"] == "scalar"
    assert rope_scaling_factor["type"] == "float"
    assert rope_scaling_factor["scalar_type"] == "float"

    communicator = _param_by_key(raw, "communicator")
    assert communicator["value_kind"] == "enum"
    assert communicator["type"] == "select"
    assert {opt["value"] for opt in (communicator.get("options") or [])} >= {
        "nccl",
        "native",
        "cuda-ipc",
    }

    vision_max_batch_size = _param_by_key(raw, "vision_max_batch_size")
    assert vision_max_batch_size["value_kind"] == "scalar"
    assert vision_max_batch_size["type"] == "int"

    sections = lmdeploy_params_to_sections(raw)
    assert {s["id"] for s in sections} == {
        "options",
        "pytorch_engine_arguments",
        "turbomind_engine_arguments",
        "vision_model_arguments",
        "speculative_decoding_arguments",
    }
    by_section = {s["id"]: s for s in sections}
    assert len(by_section["options"]["params"]) >= 20
    assert len(by_section["pytorch_engine_arguments"]["params"]) >= 25
    # Shared keys (e.g. ``dtype``, ``tp``) merge into one row; first section wins for placement.
    assert _param_by_key(raw, "dtype")["section_id"] == "pytorch_engine_arguments"
    assert "cp" in {
        p["key"] for p in by_section["turbomind_engine_arguments"]["params"]
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


def test_parse_llama_embedding_optional_plural_suffix_in_help_is_flag():
    """``--embedding(s)`` leaves ``(s)`` after flag extraction; treat as a boolean flag, not a scalar."""
    text = """
server:
         --embedding(s)           restrict to only support embedding use case (default: disabled)
"""
    sections = parse_llama_help_to_sections(text, "ik_llama")
    flat = [p for s in sections for p in s["params"]]
    emb = _param_by_key(flat, "embedding")
    assert emb["value_kind"] == "flag"
    assert emb["primary_flag"] == "--embedding"
    assert embedding_mode_config_key_from_entry({"sections": sections}) == "embedding"


def test_parse_ik_llama_help_sample_fixture():
    """
    Full verbatim ``ik_llama.cpp`` ``llama-server --help``: colon-style section headers (not
    ``-----`` banners) and ``--embedding(s)`` for embedding-only mode.
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "ik_llama_server_help_sample.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections = parse_llama_help_to_sections(text, "ik_llama")
    by_id = {s["id"]: s for s in sections}
    assert by_id.keys() >= {
        "general",
        "sampling",
        "template",
        "grammar",
        "embedding",
        "context_hacking",
        "perplexity",
        "parallel",
        "multi_modality",
        "backend",
        "model",
        "retrieval",
        "server",
        "logging",
        "export_lora",
    }
    flat = [p for s in sections for p in s["params"]]
    keys = [p["key"] for p in flat]
    assert len(keys) == len(set(keys)), "duplicate config keys after merge"
    assert len(flat) >= 200, "expected full help to yield hundreds of CLI params"
    keys_set = set(keys)
    assert "ctx_size" in keys_set
    assert "threads" in keys_set
    assert "temp" in keys_set
    pooling = _param_by_key(flat, "pooling")
    assert pooling["value_kind"] == "enum"
    assert {opt["value"] for opt in (pooling.get("options") or [])} >= {
        "none",
        "mean",
        "cls",
        "last",
    }
    embedding = _param_by_key(flat, "embedding")
    assert embedding["value_kind"] == "flag"
    assert embedding["primary_flag"] == "--embedding"
    assert "--embedding" in embedding["flags"]
    assert "embedding" in {p["key"] for p in by_id["server"]["params"]}
    entry = {"sections": sections}
    assert embedding_mode_config_key_from_entry(entry) == "embedding"

    assert "rope_scaling" in {p["key"] for p in by_id["context_hacking"]["params"]}
    assert all(p.get("flags") for p in flat)
    assert all((p.get("description") or "").strip() for p in flat)


def test_parse_llama_tensor_split_csv_stays_scalar():
    text = """
----- gpu params -----
-ts, --tensor-split N0,N1,N2,...      fraction of the model to offload to each GPU, comma-separated list of proportions
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    params = [p for s in sections for p in s["params"]]
    tensor_split = _param_by_key(params, "tensor_split")
    assert tensor_split["primary_flag"] == "--tensor-split"
    assert tensor_split["value_kind"] == "scalar"
    assert tensor_split["type"] == "string"
    assert tensor_split["multiple"] is False
