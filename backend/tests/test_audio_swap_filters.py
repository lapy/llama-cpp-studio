"""llama-swap filter merge semantics for audio.cpp request defaults."""

import backend.llama_swap_config as swap_config


def test_yaml_filters_injects_speech_defaults_as_set_params():
    config = {
        "engine": "audio_cpp",
        "family": "omnivoice",
        "task": "tts",
        "speech_defaults": {"instructions": "warm narrator", "temperature": 0.7},
    }
    filters, aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-omnivoice",
        config=config,
    )
    assert filters["setParams"] == {
        "instructions": "warm narrator",
        "temperature": 0.7,
    }
    assert aliases == []


def test_yaml_filters_injects_transcription_defaults_with_prompt_text():
    config = {
        "engine": "audio_cpp",
        "family": "qwen3_asr",
        "task": "asr",
        "transcription_defaults": {
            "language": "en",
            "prompt": "Transcribe clearly.",
            "options": {"num_beams": 4},
        },
    }
    filters, _aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-asr",
        config=config,
    )
    assert filters["setParams"] == {
        "language": "en",
        "options": {"num_beams": 4, "text": "Transcribe clearly."},
    }


def test_yaml_filters_injects_task_defaults_for_generic_families():
    config = {
        "engine": "audio_cpp",
        "family": "ace_step",
        "task": "gen",
        "task_defaults": {
            "text": "upbeat pop",
            "duration_seconds": 30.0,
            "options": {"task_route": "text2music"},
        },
    }
    filters, _aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-gen",
        config=config,
    )
    assert filters["setParams"]["text"] == "upbeat pop"
    assert filters["setParams"]["duration_seconds"] == 30.0
    assert filters["setParams"]["options"]["task_route"] == "text2music"


def test_yaml_filters_request_defaults_coexist_with_set_params_by_id():
    config = {
        "engine": "audio_cpp",
        "family": "omnivoice",
        "task": "tts",
        "speech_defaults": {"temperature": 0.8},
        "set_params_by_id": [
            {"sub_id": "high", "params": {"temperature": 0.2}},
        ],
        "model_alias": "assistant-voice",
    }
    filters, aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-omnivoice",
        config=config,
    )
    assert filters["setParams"]["temperature"] == 0.8
    assert "assistant-voice:high" in filters["setParamsByID"]
    assert filters["setParamsByID"]["assistant-voice:high"]["temperature"] == 0.2
    assert "assistant-voice:high" in aliases


def test_yaml_filters_collects_model_alias_as_swap_alias():
    config = {
        "engine": "audio_cpp",
        "family": "omnivoice",
        "task": "tts",
        "model_alias": "my-voice",
        "swap_aliases": ["alt-voice"],
    }
    _filters, aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-omnivoice",
        config=config,
    )
    assert "my-voice" in aliases
    assert "alt-voice" in aliases


def test_yaml_filters_ignores_non_audio_engine_defaults():
    config = {
        "engine": "llama_cpp",
        "speech_defaults": {"instructions": "ignored"},
    }
    filters, aliases = swap_config._yaml_filters_and_aliases(
        stable_id="llama-model",
        config=config,
    )
    assert filters is None
    assert aliases == []
