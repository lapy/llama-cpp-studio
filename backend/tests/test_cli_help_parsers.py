"""Unit tests for --help parsers."""

import json
import os

from backend.cli_help_parsers import (
    _attach_llama_sections,
    _extract_paren_default,
    infer_audio_cpp_family_tasks,
    lmdeploy_params_to_sections,
    parse_audio_cpp_help_to_sections,
    parse_audio_cpp_inspection,
    parse_audio_cpp_loader_family_tasks,
    parse_audio_cpp_loader_list,
    parse_audio_cpp_loaders_json,
    parse_llama_help_to_sections,
    parse_llama_server_help,
    parse_lmdeploy_api_server_help,
    parse_vllm_serve_help,
    vllm_params_to_sections,
)
from backend.engine_param_catalog import embedding_mode_config_key_from_entry
from backend.tests.help_parser_audit import (
    classify_llama_help_lines,
    classify_vllm_help_lines,
    extract_llama_help_entries,
    extract_lmdeploy_help_entries,
    extract_vllm_help_entries,
    llama_help_expected_rows,
    verify_all_help_params,
    verify_llama_help_line_by_line,
    verify_vllm_help_line_by_line,
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
    weight = scoped[("session_option", "vevo2.weight_type")]
    assert weight["type"] == "select"
    assert [opt["value"] for opt in weight["options"]] == [
        "native",
        "f32",
        "f16",
        "bf16",
        "q8_0",
    ]
    vocoder = scoped[("session_option", "vevo2.vocoder_weight_type")]
    assert vocoder["type"] == "select"
    assert [opt["value"] for opt in vocoder["options"]] == ["native", "f32", "f16"]
    mem_saver = scoped[("session_option", "vevo2.mem_saver")]
    assert mem_saver["type"] == "bool"
    assert scoped[("request_option", "task_route")]["read_only"] is True
    assert scoped[("request_option", "use_pitch_shift")]["type"] == "bool"


def test_parse_audio_cpp_help_legacy_session_option_prefix():
    """Older fixtures used --session-option; keep that transport form working."""
    text = """
family=demo
  Model session options:
    --session-option demo.weight_type <native|f32>  Weight storage type
    --session-option demo.mem_saver <true|false>  Memory saver
  Model load options:
    --load-option demo.extra_path <dir>  Extra path
"""
    scoped = {
        (param["scope"], param["key"]): param
        for section in parse_audio_cpp_help_to_sections(text, source="cli")
        for param in section["params"]
    }
    assert scoped[("session_option", "demo.weight_type")]["type"] == "select"
    assert scoped[("session_option", "demo.mem_saver")]["type"] == "bool"
    assert ("load_option", "demo.extra_path") in scoped


def test_parse_audio_cpp_help_captured_from_live_nemotron():
    """Captured ``audiocpp_cli --model … --help`` for nemotron_asr (bare keyed rows)."""
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_nemotron_help_live.txt"),
        encoding="utf-8",
    ) as handle:
        sections = parse_audio_cpp_help_to_sections(handle.read(), source="cli")
    scoped = {
        (param["scope"], param["key"]): param
        for section in sections
        for param in section["params"]
    }
    assert scoped[("session_option", "nemotron_asr.weight_type")]["type"] == "select"
    assert [opt["value"] for opt in scoped[("session_option", "nemotron_asr.weight_type")]["options"]] == [
        "native",
        "f32",
        "f16",
        "bf16",
        "q8_0",
    ]
    assert scoped[("session_option", "nemotron_asr.mem_saver")]["type"] == "bool"
    assert scoped[("request_option", "language")]["read_only"] is True
    assert scoped[("request_option", "keep_language_tags")]["type"] == "bool"


def test_parse_audio_cpp_help_qwen3_asr_has_no_session_options_in_cli():
    """Qwen3 ASR docs mention session options, but current CLI help omits them."""
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_qwen3_asr_help_live.txt"),
        encoding="utf-8",
    ) as handle:
        sections = parse_audio_cpp_help_to_sections(handle.read(), source="cli")
    session_keys = {
        param["key"]
        for section in sections
        for param in section["params"]
        if param.get("scope") == "session_option"
    }
    assert session_keys == set()
    scoped = {
        (param["scope"], param["key"]): param
        for section in sections
        for param in section["params"]
    }
    assert scoped[("model", "task")]["key"] == "task"


def test_parse_audio_cpp_server_help_live_pin():
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_server_help_live.txt"),
        encoding="utf-8",
    ) as handle:
        sections = parse_audio_cpp_help_to_sections(handle.read(), source="server")
    index = {
        param["key"]: param
        for section in sections
        for param in section["params"]
    }
    assert index["model_spec_override"]["reserved"] is True
    assert index["model_spec_override"]["primary_flag"] == "--model-spec-override"
    assert [opt["value"] for opt in index["backend"]["options"]] == [
        "cpu",
        "cuda",
        "hip",
        "rocm",
        "vulkan",
        "metal",
    ]
    assert index["backend"]["default"] == "cuda"
    assert index["ui"]["type"] == "bool"
    assert index["ui"]["negative_flag"] == "--no-ui"
    assert "serve" in (index["ui"]["description"] or "").lower()
    assert "no_ui" not in index
    assert index["busy_timeout_ms"]["type"] == "int"
    assert index["busy_timeout_ms"]["default"] == 300000
    assert index["idle_unload_ms"]["type"] == "int"
    assert index["idle_unload_ms"]["default"] == 0
    assert index["max_loaded_models"]["type"] == "int"
    assert index["port"]["type"] == "int"
    assert "-busy" not in (index["busy_timeout_ms"].get("flags") or [])
    assert "-ui" not in (index["ui"].get("flags") or [])


def test_parse_audio_cpp_cli_help_live_pin():
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_cli_help_live.txt"),
        encoding="utf-8",
    ) as handle:
        sections = parse_audio_cpp_help_to_sections(handle.read(), source="cli")
    index = {
        param["key"]: param
        for section in sections
        for param in section["params"]
    }
    assert "key" not in index
    assert [opt["value"] for opt in index["task"]["options"]] == [
        "vad",
        "asr",
        "diar",
        "sep",
        "gen",
        "tts",
        "clon",
        "vc",
        "s2s",
        "align",
        "vdes",
        "spk",
        "svc",
        "midi",
    ]
    assert [opt["value"] for opt in index["backend"]["options"]] == [
        "cpu",
        "cuda",
        "hip",
        "rocm",
        "vulkan",
        "metal",
        "best",
    ]
    assert index["mode"]["default"] == "offline"
    assert index["threads"]["default"] == 4
    assert index["model_spec_override"]["reserved"] is True
    assert index["model_spec_override"]["scope"] == "process"
    assert index["list_devices"]["scope"] == "process"
    assert index["load_option"]["reserved"] is True
    assert index["session_option"]["reserved"] is True
    assert index["request_option"]["reserved"] is True
    assert "json" not in index
    assert index["list_loaders"]["primary_flag"] == "--list-loaders"
    assert [opt["value"] for opt in index["text_chunk_mode"]["options"]] == [
        "default",
        "tag_aware",
        "japanese",
        "endline",
    ]
    assert index["text_chunk_mode"]["default"] is None
    assert index["do_sample"]["type"] == "bool"


def test_parse_audio_cpp_cli_transport_docs_are_not_keyed_options():
    text = """
  Global:
    --load-option key=value
    --session-option key=value
    --request-option key=value
    --family <name>
"""
    scoped = {
        (param["scope"], param["key"]): param
        for section in parse_audio_cpp_help_to_sections(text, source="cli")
        for param in section["params"]
    }
    assert ("model", "key") not in scoped
    assert scoped[("load_option", "load_option")]["reserved"] is True
    assert scoped[("session_option", "session_option")]["primary_flag"] == "--session-option"


def test_parse_audio_cpp_inspect_live_nemotron_and_qwen3():
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_nemotron_inspect_live.txt"),
        encoding="utf-8",
    ) as handle:
        nemotron = parse_audio_cpp_inspection(handle.read())
    assert nemotron["family"] == "nemotron_asr"
    assert nemotron["task_names"] == ["asr"]
    assert nemotron["tasks"][0]["modes"] == ["offline", "streaming"]
    assert nemotron["capabilities"]["supports_timestamps"] is True
    assert nemotron["discovery_source"] == "text"

    with open(
        os.path.join(here, "fixtures", "audio_cpp_qwen3_asr_inspect_live.txt"),
        encoding="utf-8",
    ) as handle:
        qwen = parse_audio_cpp_inspection(handle.read())
    assert qwen["family"] == "qwen3_asr"
    assert qwen["tasks"][0]["modes"] == ["offline", "streaming"]
    assert "Chinese" in qwen["languages"]


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


def test_bare_loader_list_infers_family_tasks():
    assert infer_audio_cpp_family_tasks("omnivoice") == ["tts"]
    assert infer_audio_cpp_family_tasks("vibevoice_asr") == ["asr"]
    assert infer_audio_cpp_family_tasks("qwen3_forced_aligner") == ["align"]
    assert infer_audio_cpp_family_tasks("miocodec") == ["codec"]
    assert infer_audio_cpp_family_tasks("miocodec_25hz_44k_v2") == ["codec"]
    assert infer_audio_cpp_family_tasks("miotts_1_7b") == ["tts"]
    assert infer_audio_cpp_family_tasks("supertonic") == ["tts"]
    mapped = parse_audio_cpp_loader_family_tasks(
        "registered_loaders=5\nomnivoice\nvibevoice_asr\nmiocodec\nsupertonic\nqwen3_forced_aligner\n"
    )
    assert mapped["omnivoice"] == ["tts"]
    assert mapped["vibevoice_asr"] == ["asr"]
    assert mapped["miocodec"] == ["codec"]
    assert mapped["supertonic"] == ["tts"]
    assert mapped["qwen3_forced_aligner"] == ["align"]


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

    voiced = parse_audio_cpp_inspection(
        '{"family":"pocket_tts","tasks":["tts"],"voices":["alba","cosette"]}'
    )
    assert voiced["voices"] == ["alba", "cosette"]


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
    classified = classify_llama_help_lines(text)
    assert classified[0]["role"] == "prologue"
    assert any(row["role"] == "section" for row in classified)
    assert all(row["role"] != "other" for row in classified)


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
        "load_mode",
        "n_cpu_ffn",
        "ui_config",
        "agent",
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
    assert tools["value_kind"] == "csv_enum"
    assert tools["type"] == "multiselect"
    assert tools["multiple"] is True
    assert tools["default"] == []
    assert {opt["value"] for opt in (tools.get("options") or [])} >= {
        "all",
        "read_file",
        "write_file",
        "get_info",
    }

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

    mmproj_auto = _param_by_key(flat, "mmproj_auto")
    assert mmproj_auto["value_kind"] == "flag"
    assert mmproj_auto["primary_flag"] == "--mmproj-auto"
    assert "--no-mmproj" in mmproj_auto["flags"]

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
    assert {opt["value"] for opt in (spec_type.get("options") or [])} >= {
        "none",
        "draft-simple",
        "draft-mtp",
        "draft-dflash",
        "draft-dspark",
        "ngram-simple",
    }
    assert spec_type["default"] == ["none"]

    load_mode = _param_by_key(flat, "load_mode")
    assert load_mode["value_kind"] == "enum"
    assert load_mode["type"] == "select"
    assert load_mode["default"] == "auto"
    assert {opt["value"] for opt in (load_mode.get("options") or [])} >= {
        "auto",
        "mmap",
        "mlock",
        "dio",
    }

    ui = _param_by_key(flat, "ui")
    assert ui["value_kind"] == "flag"
    assert ui["primary_flag"] == "--ui"
    assert ui["negative_flag"] == "--no-ui"
    assert set(ui["flags"]) >= {"--ui", "--webui", "--no-ui", "--no-webui"}

    ui_config = _param_by_key(flat, "ui_config")
    assert ui_config["value_kind"] == "json_object"
    assert ui_config["primary_flag"] == "--ui-config"
    assert "--webui-config" in ui_config["flags"]

    agent = _param_by_key(flat, "agent")
    assert agent["value_kind"] == "flag"
    assert agent["primary_flag"] == "--agent"
    assert agent["negative_flag"] == "--no-agent"

    timeout = _param_by_key(flat, "timeout")
    assert timeout["default"] == 3600

    cpu_strict = _param_by_key(flat, "cpu_strict")
    assert cpu_strict["value_kind"] == "enum"
    assert cpu_strict["type"] == "select"
    assert cpu_strict["default"] == 0
    assert [opt["value"] for opt in cpu_strict["options"]] == ["0", "1"]

    poll_section = next(
        section["id"]
        for section in sections
        if any(param["key"] == "poll" for param in section["params"])
    )
    assert poll_section == "common_params"


def test_extract_paren_default_allows_possessive_apostrophe():
    """``model's metadata`` must not be treated as an unclosed quoted string."""
    text = (
        "set custom jinja chat template (default: template taken from model's "
        "metadata) if suffix/prefix are specified (env: LLAMA_ARG_CHAT_TEMPLATE)"
    )
    assert _extract_paren_default(text) == "template taken from model's metadata"


def test_parse_llama_fixture_excerpt_audits_every_flag():
    """Every non-removed ``llama-server --help`` flag is parsed with consistent types."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "llama_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    entries = extract_llama_help_entries(text)
    raw = _attach_llama_sections(text, parse_llama_server_help(text, "llama_cpp"))
    issues = verify_all_help_params(entries, raw)
    line_issues = verify_llama_help_line_by_line(text, raw)
    assert len(entries) == len(raw)
    assert len(entries) >= 240
    assert not issues, ";\n".join(issues)
    assert not line_issues, ";\n".join(line_issues)


def test_parse_llama_fixture_classifies_every_line():
    """No help line is left unclassified; option/continuation/section cover the fixture."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "llama_server_help_excerpt.txt")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    classified = classify_llama_help_lines(text)
    assert all(row["role"] != "other" for row in classified)
    assert sum(1 for row in classified if row["role"] == "option") >= 240
    assert sum(1 for row in classified if row["role"] == "section") == 4
    assert len(classified) == len(text.splitlines())


def test_parse_llama_fixture_matches_expected_snapshot():
    """Line-by-line parse of llama-server --help matches the checked-in snapshot."""
    here = os.path.dirname(__file__)
    fixtures = os.path.join(here, "fixtures")
    with open(
        os.path.join(fixtures, "llama_server_help_excerpt.txt"), encoding="utf-8"
    ) as handle:
        text = handle.read()
    with open(
        os.path.join(fixtures, "llama_server_help_expected.json"), encoding="utf-8"
    ) as handle:
        expected = json.load(handle)
    raw = _attach_llama_sections(text, parse_llama_server_help(text, "llama_cpp"))
    actual = llama_help_expected_rows(raw)
    assert len(actual) == len(expected)
    by_key_expected = {row["key"]: row for row in expected}
    by_key_actual = {row["key"]: row for row in actual}
    assert set(by_key_actual) == set(by_key_expected)
    mismatches = [
        f"{key}: {by_key_actual[key]!r} != {by_key_expected[key]!r}"
        for key in sorted(by_key_expected)
        if by_key_actual[key] != by_key_expected[key]
    ]
    assert not mismatches, ";\n".join(mismatches)


def test_parse_llama_does_not_treat_description_flag_mentions_as_options():
    text = """
----- common params -----
-hf,   --hf-repo <user>/<model>[:quant]
                                        mmproj is also downloaded automatically if available. to disable, add
                                        --no-mmproj
                                        (default: unused)
--mmproj-auto, --no-mmproj, --no-mmproj-auto
                                        whether to use multimodal projector file (default: enabled)
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    keys = {p["key"] for p in flat}
    assert "no_mmproj" not in keys
    mmproj_auto = _param_by_key(flat, "mmproj_auto")
    assert "--no-mmproj" in mmproj_auto["flags"]
    hf_repo = _param_by_key(flat, "hf_repo")
    assert "--no-mmproj" in (hf_repo.get("description") or "")


def test_parse_llama_ui_aliases_and_hyphenated_short_negatives():
    text = """
----- example-specific params -----
-ag,   --agent, -no-ag, --no-agent      whether to enable CORS proxy (default: disabled)
--ui,  --webui, --no-ui, --no-webui     whether to enable the Web UI (default: enabled)
--ui-config, --webui-config JSON        JSON that provides default UI settings
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    agent = _param_by_key(flat, "agent")
    assert agent["value_kind"] == "flag"
    assert agent["negative_flag"] == "--no-agent"
    ui = _param_by_key(flat, "ui")
    assert ui["primary_flag"] == "--ui"
    assert "--webui" in ui["flags"]
    ui_config = _param_by_key(flat, "ui_config")
    assert ui_config["primary_flag"] == "--ui-config"
    assert ui_config["value_kind"] == "json_object"


def test_parse_llama_angle_pipe_zero_one_is_enum():
    text = """
----- common params -----
--cpu-strict <0|1>                      use strict CPU placement (default: 0)
--poll <0...100>                        use polling level to wait for work (0 - no polling, default: 50)
--poll-batch <0|1>                      use polling to wait for work (default: same as --poll)
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    cpu_strict = _param_by_key(flat, "cpu_strict")
    assert cpu_strict["value_kind"] == "enum"
    assert [opt["value"] for opt in cpu_strict["options"]] == ["0", "1"]
    assert cpu_strict["default"] == 0
    poll = _param_by_key(flat, "poll")
    assert poll["value_kind"] == "scalar"
    assert poll["type"] == "int"
    assert poll["default"] == 50
    poll_batch = _param_by_key(flat, "poll_batch")
    assert poll_batch["value_kind"] == "enum"
    assert [opt["value"] for opt in poll_batch["options"]] == ["0", "1"]


def test_parse_llama_load_mode_dash_list_enum():
    text = """
----- common params -----
-lm,   --load-mode MODE                 model loading mode (default: auto)
                                        - auto: mmap, unless a device does not support it
                                        - none: no special loading mode
                                        - mmap: memory-map model
                                        - mlock: force system to keep model in RAM
                                        - mmap+mlock: mmap + mlock
                                        - dio: use DirectIO if available
"""
    sections = parse_llama_help_to_sections(text, "llama_cpp")
    flat = [p for s in sections for p in s["params"]]
    row = _param_by_key(flat, "load_mode")
    assert row["value_kind"] == "enum"
    assert [o["value"] for o in row["options"]] == [
        "auto",
        "none",
        "mmap",
        "mlock",
        "mmap+mlock",
        "dio",
    ]
    assert row["default"] == "auto"


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
    line_issues = verify_vllm_help_line_by_line(text, raw)

    assert len(entries) == 211
    assert len(raw) == 211
    assert not issues, ";\n".join(issues)
    assert not line_issues, ";\n".join(line_issues)

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

    disable_log_stats = _param_by_key(raw, "disable_log_stats")
    assert disable_log_stats["value_kind"] == "flag"
    assert disable_log_stats["default"] is False

    config_format = _param_by_key(raw, "config_format")
    assert config_format["value_kind"] == "enum"
    assert [opt["value"] for opt in config_format["options"]] == [
        "auto",
        "hf",
        "mistral",
    ]

    middleware = _param_by_key(raw, "middleware")
    assert middleware["value_kind"] == "repeatable"


def test_parse_onecat_vllm_fixture_classifies_every_line():
    """No 1Cat-vLLM help line is left unclassified."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "onecatvllm_serve_help_sample.txt")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    classified = classify_vllm_help_lines(text)
    assert all(row["role"] != "other" for row in classified)
    assert sum(1 for row in classified if row["role"] == "option") == 211
    assert sum(1 for row in classified if row["role"] == "section") == 14
    assert sum(1 for row in classified if row["role"] == "footer") >= 1
    assert len(classified) == len(text.splitlines())


def test_parse_onecat_vllm_fixture_matches_expected_snapshot():
    """Line-by-line parse of vllm serve --help=all matches the checked-in snapshot."""
    here = os.path.dirname(__file__)
    fixtures = os.path.join(here, "fixtures")
    with open(
        os.path.join(fixtures, "onecatvllm_serve_help_sample.txt"), encoding="utf-8"
    ) as handle:
        text = handle.read()
    with open(
        os.path.join(fixtures, "onecatvllm_serve_help_expected.json"), encoding="utf-8"
    ) as handle:
        expected = json.load(handle)
    actual = llama_help_expected_rows(parse_vllm_serve_help(text))
    assert len(actual) == len(expected)
    by_key_expected = {row["key"]: row for row in expected}
    by_key_actual = {row["key"]: row for row in actual}
    assert set(by_key_actual) == set(by_key_expected)
    mismatches = [
        f"{key}: {by_key_actual[key]!r} != {by_key_expected[key]!r}"
        for key in sorted(by_key_expected)
        if by_key_actual[key] != by_key_expected[key]
    ]
    assert not mismatches, ";\n".join(mismatches)


def test_parse_vllm_inline_description_without_metavar_is_flag():
    text = """
options:
  --disable-log-stats   Disable logging statistics. (default: False)
  --headless            Run in headless mode. (default: False)
  -h, --help            show this help message and exit
"""
    raw = parse_vllm_serve_help(text)
    disable_log_stats = _param_by_key(raw, "disable_log_stats")
    assert disable_log_stats["value_kind"] == "flag"
    assert disable_log_stats["default"] is False
    headless = _param_by_key(raw, "headless")
    assert headless["value_kind"] == "flag"
    help_row = _param_by_key(raw, "help")
    assert help_row["value_kind"] == "flag"


def test_parse_vllm_quoted_choice_list_is_enum():
    text = """
ModelConfig:
  --config-format ['auto', 'hf', 'mistral']
                        The format of the model config to load. (default: auto)
  --model-impl ['auto', 'transformers', 'vllm']
                        Which implementation of the model to use. (default: auto)
"""
    raw = parse_vllm_serve_help(text)
    config_format = _param_by_key(raw, "config_format")
    assert config_format["value_kind"] == "enum"
    assert [opt["value"] for opt in config_format["options"]] == [
        "auto",
        "hf",
        "mistral",
    ]
    assert config_format["default"] == "auto"
    model_impl = _param_by_key(raw, "model_impl")
    assert model_impl["value_kind"] == "enum"
    assert [opt["value"] for opt in model_impl["options"]] == [
        "auto",
        "transformers",
        "vllm",
    ]


def test_parse_vllm_screaming_metavar_and_repeatable_middleware():
    text = """
options:
  --api-server-count API_SERVER_COUNT, -asc API_SERVER_COUNT
                        How many API server processes to run. (default: None)
Frontend:
  --middleware MIDDLEWARE
                        Additional ASGI middleware to apply to the app. We accept multiple --middleware arguments. (default: [])
  --port PORT           Port number. (default: 8000)
"""
    raw = parse_vllm_serve_help(text)
    api_server_count = _param_by_key(raw, "api_server_count")
    assert api_server_count["value_kind"] == "scalar"
    assert "API_SERVER_COUNT" not in (api_server_count.get("description") or "")
    middleware = _param_by_key(raw, "middleware")
    assert middleware["value_kind"] == "repeatable"
    port = _param_by_key(raw, "port")
    assert port["value_kind"] == "scalar"
    assert port["default"] == 8000
