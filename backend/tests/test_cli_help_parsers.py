"""Unit tests for --help parsers."""

import os

from backend.cli_help_parsers import (
    lmdeploy_params_to_sections,
    parse_audio_cpp_help_to_sections,
    parse_audio_cpp_inspection,
    parse_audio_cpp_loader_list,
    parse_audio_cpp_loaders_json,
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
    parse_vllm_serve_help,
    vllm_params_to_sections,
)
from backend.engine_param_catalog import embedding_mode_config_key_from_entry
from backend.tests.help_parser_audit import (
    extract_lmdeploy_help_entries,
    extract_vllm_help_entries,
    verify_all_help_params,
)


def _param_by_key(params, key):
    return next(p for p in params if p["key"] == key)


def test_parse_audio_cpp_server_and_model_profiles():
    here = os.path.dirname(__file__)
    fixtures = os.path.join(here, "fixtures")
    with open(
        os.path.join(fixtures, "audio_cpp_server_help_sample.txt"),
        encoding="utf-8",
    ) as handle:
        server_sections = parse_audio_cpp_help_to_sections(
            handle.read(), source="server"
        )
    server_params = [
        param for section in server_sections for param in section["params"]
    ]
    server_index = {param["key"]: param for param in server_params}
    assert {"config", "host", "port", "backend", "device", "threads"} <= set(
        server_index
    )
    assert server_index["port"]["reserved"] is True
    assert server_index["threads"]["scope"] == "process"
    assert server_index["device"]["type"] == "int"
    assert server_index["threads"]["type"] == "int"
    assert server_index["backend"]["transport"] == "server_flag"


def test_parse_audio_cpp_server_help_skips_option_transport_docs():
    text = """
audiocpp_server --config <server.json> [--log] [--log-file <path>]
  --load-option key=value  Pass load options as key=value pairs
  --log  Enable framework logging
"""
    sections = parse_audio_cpp_help_to_sections(text, source="server")
    keys = {
        param["key"]
        for section in sections
        for param in section.get("params") or []
    }
    assert "key=value" not in keys
    assert "log" in keys

    here = os.path.dirname(__file__)
    fixtures = os.path.join(here, "fixtures")
    with open(
        os.path.join(fixtures, "audio_cpp_model_help_sample.txt"),
        encoding="utf-8",
    ) as handle:
        model_sections = parse_audio_cpp_help_to_sections(
            handle.read(), source="cli"
        )
    scoped = {
        (param["scope"], param["key"]): param
        for section in model_sections
        for param in section["params"]
    }
    assert ("load_option", "vevo2.whisper_model_path") in scoped
    assert ("session_option", "vevo2.weight_type") in scoped
    assert scoped[("request_option", "task_route")]["read_only"] is True


def test_parse_audio_cpp_inspection_and_loader_list():
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_inspect_sample.txt"),
        encoding="utf-8",
    ) as handle:
        inspection = parse_audio_cpp_inspection(handle.read())
    assert inspection["family"] == "vevo2"
    assert inspection["task_names"] == ["tts", "vc", "svc"]
    assert inspection["capabilities"]["supports_speaker_reference"] is True
    assert inspection["configs"][0]["id"] == "main"
    assert parse_audio_cpp_loader_list(
        "registered_loaders=3\nwhisper\nvevo2\nqwen3_tts\n"
    ) == ["whisper", "vevo2", "qwen3_tts"]


def test_parse_audio_cpp_loaders_json_and_inspect_json():
    loaders = parse_audio_cpp_loaders_json(
        {
            "loaders": [
                {
                    "family": "omnivoice",
                    "tasks": [{"id": "tts", "modes": ["offline"]}],
                    "instructions_policy": "soft_tags",
                    "api_endpoints": ["/v1/audio/speech"],
                }
            ]
        }
    )
    assert loaders["families"] == ["omnivoice"]
    assert loaders["family_tasks"]["omnivoice"] == ["tts"]
    assert loaders["family_policies"]["omnivoice"] == "soft_tags"
    assert parse_audio_cpp_loader_list(
        '{"loaders":[{"family":"demo_tts","tasks":["tts"]}]}'
    ) == ["demo_tts"]

    # Family-keyed map + schema_version wrapper
    mapped = parse_audio_cpp_loaders_json(
        {
            "schema_version": 1,
            "data": {
                "loaders": {
                    "qwen3_asr": {
                        "tasks": {"asr": ["offline"]},
                        "instruction_policy": "none",
                        "endpoints": "/v1/audio/transcriptions",
                    }
                }
            },
        }
    )
    assert mapped["families"] == ["qwen3_asr"]
    assert mapped["family_tasks"]["qwen3_asr"] == ["asr"]
    assert mapped["family_policies"]["qwen3_asr"] == "none"
    assert mapped["family_endpoints"]["qwen3_asr"] == ["/v1/audio/transcriptions"]

    inspection = parse_audio_cpp_inspection(
        '{"family":"omnivoice","tasks":[{"task":"tts","modes":["offline"]}],'
        '"instructions_policy":"soft_tags",'
        '"instructions_vocabulary":["female","british accent"],'
        '"preferred_api_endpoint":"/v1/audio/speech"}'
    )
    assert inspection["discovery_source"] == "json"
    assert inspection["instructions_policy"] == "soft_tags"
    assert inspection["instructions_vocabulary"] == ["female", "british accent"]
    assert inspection["preferred_api_endpoint"] == "/v1/audio/speech"
    assert inspection["task_names"] == ["tts"]

    wrapped = parse_audio_cpp_inspection(
        '{"schema_version":1,"data":{"family":"demo","tasks":["tts"],'
        '"instruction_policy":"openai_instruct"}}'
    )
    assert wrapped["family"] == "demo"
    assert wrapped["instructions_policy"] == "openai_instruct"


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
    assert ids == {
        "common_params",
        "sampling_params",
        "speculative_params",
        "example_specific_params",
    }
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
    assert samplers["value_kind"] == "semicolon_enum"
    assert samplers["type"] == "multiselect"
    assert samplers["default"] == [
        "penalties",
        "dry",
        "top_n_sigma",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature",
    ]

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

    chat_template_kwargs = _param_by_key(flat, "chat_template_kwargs")
    assert chat_template_kwargs["value_kind"] == "json_object"
    assert chat_template_kwargs["type"] == "json"
    assert chat_template_kwargs.get("options") in (None, [])
    assert chat_template_kwargs["primary_flag"] == "--chat-template-kwargs"

    cache_type_k_draft = _param_by_key(flat, "cache_type_k_draft")
    assert cache_type_k_draft["value_kind"] == "enum"
    assert cache_type_k_draft["type"] == "select"
    assert "--spec-draft-type-k" in cache_type_k_draft["flags"]
    assert cache_type_k_draft["default"] == "f16"
    assert {opt["value"] for opt in (cache_type_k_draft.get("options") or [])} == {
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "q4_0",
        "q4_1",
        "iq4_nl",
        "q5_0",
        "q5_1",
    }
    assert "(env:" not in cache_type_k_draft["description"].lower()
    assert "LLAMA_ARG" not in cache_type_k_draft["description"]

    log_verbosity = _param_by_key(flat, "log_verbosity")
    assert log_verbosity["default"] == 3
    assert log_verbosity["scalar_type"] == "int"

    flash_attn = _param_by_key(flat, "flash_attn")
    assert flash_attn["default"] == "auto"

    host = _param_by_key(flat, "host")
    assert host["value_kind"] == "scalar"
    assert host["type"] == "string"
    assert host["default"] == "127.0.0.1"

    no_host = _param_by_key(flat, "no_host")
    assert no_host["value_kind"] == "flag"
    assert no_host["primary_flag"] == "--no-host"

    mmproj = _param_by_key(flat, "mmproj")
    assert mmproj["value_kind"] == "scalar"
    assert mmproj["type"] == "string"

    no_mmproj = _param_by_key(flat, "no_mmproj")
    assert no_mmproj["value_kind"] == "flag"
    assert no_mmproj["primary_flag"] == "--no-mmproj"

    spec_ngram_mod_n_min = _param_by_key(flat, "spec_ngram_mod_n_min")
    assert spec_ngram_mod_n_min["value_kind"] == "scalar"
    assert spec_ngram_mod_n_min["type"] == "int"
    assert spec_ngram_mod_n_min["default"] == 48

    spec_ngram_mod_n_max = _param_by_key(flat, "spec_ngram_mod_n_max")
    assert spec_ngram_mod_n_max["value_kind"] == "scalar"
    assert spec_ngram_mod_n_max["type"] == "int"
    assert spec_ngram_mod_n_max["default"] == 64

    ctx_size = _param_by_key(flat, "ctx_size")
    assert ctx_size["default"] == 0

    sleep_idle_seconds = _param_by_key(flat, "sleep_idle_seconds")
    assert sleep_idle_seconds["default"] == -1

    cache_ram = _param_by_key(flat, "cache_ram")
    assert cache_ram["default"] == 8192

    kv_offload = _param_by_key(flat, "kv_offload")
    assert kv_offload["default"] is True

    cache_idle_slots = _param_by_key(flat, "cache_idle_slots")
    assert cache_idle_slots["default"] is True

    split_mode = _param_by_key(flat, "split_mode")
    assert split_mode["default"] == "layer"

    assert rope_scaling["default"] == "linear"

    spec_type = _param_by_key(flat, "spec_type")
    assert spec_type["value_kind"] == "csv_enum"
    assert spec_type["type"] == "multiselect"
    assert len(spec_type.get("options") or []) >= 5
    assert spec_type["default"] == ["none"]


def test_parse_llama_spec_type_csv_enum_multiselect():
    text = """
----- speculative params -----

--spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple
                                        comma-separated list of types of speculative decoding to use (default:
                                        none)
                                        (env: LLAMA_ARG_SPEC_TYPE)
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    row = _param_by_key(flat, "spec_type")
    assert row["value_kind"] == "csv_enum"
    assert row["type"] == "multiselect"
    assert row["multiple"] is True
    assert [o["value"] for o in row["options"]] == [
        "none",
        "draft-simple",
        "draft-eagle3",
        "draft-mtp",
        "ngram-simple",
    ]
    assert row["default"] == ["none"]


def test_parse_llama_samplers_semicolon_enum_multiselect():
    text = """
----- sampling params -----

--samplers SAMPLERS                     samplers that will be used for generation in the order, separated by
                                        ';'
                                        (default:
                                        penalties;dry;top_k;top_p;temperature)
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    row = _param_by_key(flat, "samplers")
    assert row["value_kind"] == "semicolon_enum"
    assert row["type"] == "multiselect"
    assert row["multiple"] is True
    assert [o["value"] for o in row["options"]] == [
        "penalties",
        "dry",
        "top_k",
        "top_p",
        "temperature",
    ]
    assert row["default"] == ["penalties", "dry", "top_k", "top_p", "temperature"]


def test_parse_llama_allowed_values_block():
    text = """
----- common params -----

--spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE
                                        KV cache data type for K for the draft model
                                        allowed values: f32, f16, bf16, q8_0
                                        (default: f16)
                                        (env: LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K)
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    row = _param_by_key(flat, "cache_type_k_draft")
    assert row["value_kind"] == "enum"
    assert row["type"] == "select"
    assert "--spec-draft-type-k" in row["flags"]
    assert row["default"] == "f16"
    assert [o["value"] for o in row["options"]] == ["f32", "f16", "bf16", "q8_0"]


def test_parse_llama_allowed_values_multiline():
    text = """
----- common params -----

--spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE
                                        KV cache data type for K for the draft model
                                        allowed values: f32, f16, bf16
                                        q8_0, q4_0, q4_1
                                        (default: f16)
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    row = _param_by_key(flat, "cache_type_k_draft")
    assert [o["value"] for o in row["options"]] == [
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "q4_0",
        "q4_1",
    ]


def test_parse_lmdeploy_fixture_excerpt():
    """Every flag in ``lmdeploy serve api_server --help`` is parsed and verified."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "lmdeploy_api_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()

    entries = extract_lmdeploy_help_entries(text)
    raw = parse_lmdeploy_api_server_help(text)
    issues = verify_all_help_params(entries, raw)

    assert len(entries) == 82
    assert len(raw) == 66
    assert not issues, ";\n".join(issues)

    sections = lmdeploy_params_to_sections(raw)
    assert {s["id"] for s in sections} == {
        "options",
        "pytorch_engine_arguments",
        "turbomind_engine_arguments",
        "vision_model_arguments",
        "speculative_decoding_arguments",
    }
    by_section = {s["id"]: s for s in sections}
    assert len(by_section["options"]["params"]) == 23
    assert len(by_section["pytorch_engine_arguments"]["params"]) == 32
    assert len(by_section["turbomind_engine_arguments"]["params"]) == 7
    assert len(by_section["vision_model_arguments"]["params"]) == 1
    assert len(by_section["speculative_decoding_arguments"]["params"]) == 3


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

    chat_template_kwargs = _param_by_key(flat, "chat_template_kwargs")
    assert chat_template_kwargs["value_kind"] == "json_object"
    assert chat_template_kwargs["type"] == "json"

    cache_type_k_draft = _param_by_key(flat, "cache_type_k_draft")
    assert cache_type_k_draft["value_kind"] == "enum"
    assert cache_type_k_draft["type"] == "select"
    assert len(cache_type_k_draft.get("options") or []) >= 9

    cache_type_k = _param_by_key(flat, "cache_type_k")
    assert cache_type_k["value_kind"] == "enum"
    assert cache_type_k["type"] == "select"
    assert len(cache_type_k.get("options") or []) >= 9

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


def test_parse_vllm_serve_help_snippet():
    text = """
options:
  --port PORT           Port number for the server. (default: 8000)
  --enable-auto-tool-choice, --no-enable-auto-tool-choice
                        Enable auto tool choice for supported models. (default: False)

ParallelConfig:
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas. (default: 1)
"""
    raw = parse_vllm_serve_help(text)
    keys = {p["key"] for p in raw}
    assert "port" in keys
    assert "enable_auto_tool_choice" in keys
    assert "tensor_parallel_size" in keys
    sections = vllm_params_to_sections(raw)
    assert {s["id"] for s in sections} == {"options", "parallelconfig"}


def test_parse_onecat_vllm_serve_help_fixture():
    """Every flag in ``vllm serve --help=all`` is parsed and verified."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "onecatvllm_serve_help_sample.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()

    entries = extract_vllm_help_entries(text)
    raw = parse_vllm_serve_help(text)
    issues = verify_all_help_params(entries, raw)

    assert len(entries) == 211
    assert len(raw) == 211
    assert not issues, ";\n".join(issues)

    sections = vllm_params_to_sections(raw)
    assert {s["id"] for s in sections} == {
        "options",
        "frontend",
        "modelconfig",
        "loadconfig",
        "attentionconfig",
        "structuredoutputsconfig",
        "parallelconfig",
        "cacheconfig",
        "multimodalconfig",
        "loraconfig",
        "observabilityconfig",
        "schedulerconfig",
        "compilationconfig",
        "vllmconfig",
    }
    by_section = {s["id"]: s for s in sections}
    assert len(by_section["options"]["params"]) == 8
    assert len(by_section["frontend"]["params"]) == 47
    assert len(by_section["modelconfig"]["params"]) == 38
    assert len(by_section["parallelconfig"]["params"]) == 35
    assert len(by_section["vllmconfig"]["params"]) == 10
