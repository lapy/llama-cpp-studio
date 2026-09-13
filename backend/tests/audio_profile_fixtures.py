"""Shared fixtures and expectations for audio.cpp profile tests."""

from __future__ import annotations

# (task, family, defaults_key, api_endpoint)
DOC_PROFILED_FAMILIES = [
    ("tts", "chatterbox", "speech_defaults", "/v1/audio/speech"),
    ("clon", "chatterbox", "speech_defaults", "/v1/audio/speech"),
    ("tts", "neutts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "miotts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "moss_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "moss_tts_nano", "speech_defaults", "/v1/audio/speech"),
    ("clon", "moss_tts_nano", "speech_defaults", "/v1/audio/speech"),
    ("tts", "moss_tts_local", "speech_defaults", "/v1/audio/speech"),
    ("clon", "moss_tts_local", "speech_defaults", "/v1/audio/speech"),
    ("tts", "index_tts2", "speech_defaults", "/v1/audio/speech"),
    ("clon", "index_tts2", "speech_defaults", "/v1/audio/speech"),
    ("tts", "omnivoice", "speech_defaults", "/v1/audio/speech"),
    ("tts", "pocket_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "voxcpm2", "speech_defaults", "/v1/audio/speech"),
    ("tts", "higgs_audio_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "irodori_tts", "speech_defaults", "/v1/audio/speech"),
    ("vdes", "irodori_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "supertonic", "speech_defaults", "/v1/audio/speech"),
    ("vc", "chatterbox", "speech_defaults", "/v1/audio/speech"),
    ("tts", "vibevoice", "speech_defaults", "/v1/audio/speech"),
    ("tts", "qwen3_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "f5_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "habibi", "speech_defaults", "/v1/audio/speech"),
    ("clon", "habibi_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "chatterbox_turbo", "speech_defaults", "/v1/audio/speech"),
    ("vc", "chatterbox_turbo", "speech_defaults", "/v1/audio/speech"),
    ("tts", "confucius4_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "dots_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "dramabox", "speech_defaults", "/v1/audio/speech"),
    ("tts", "fish_audio", "speech_defaults", "/v1/audio/speech"),
    ("tts", "firered_audio", "speech_defaults", "/v1/audio/speech"),
    ("tts", "fireredtts3", "speech_defaults", "/v1/audio/speech"),
    ("tts", "glm_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "inflect_v2", "speech_defaults", "/v1/audio/speech"),
    ("tts", "magpie_tts", "speech_defaults", "/v1/audio/speech"),
    ("vdes", "moss_voicegen", "speech_defaults", "/v1/audio/speech"),
    ("tts", "outetts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "soprano_tts", "speech_defaults", "/v1/audio/speech"),
    ("tts", "vietneu_tts", "speech_defaults", "/v1/audio/speech"),
    ("asr", "citrinet", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "higgs_audio_stt", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "hviske", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "nemotron_asr", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "vibevoice", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "qwen3_asr", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "parakeet_tdt", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "fun_asr_nano", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "voxtral_realtime", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "kroko_asr", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "granite5asr", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "sense_asr", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "firered_audio", "transcription_defaults", "/v1/audio/transcriptions"),
    ("asr", "vibevoice_asr_streaming", "transcription_defaults", "/v1/audio/transcriptions"),
    ("gen", "ace_step", "task_defaults", "/audioapi/v1/tasks/run"),
    ("gen", "stable_audio", "task_defaults", "/audioapi/v1/tasks/run"),
    ("gen", "heartmula", "task_defaults", "/audioapi/v1/tasks/run"),
    ("gen", "midashenglm_gen", "task_defaults", "/audioapi/v1/tasks/run"),
    ("gen", "minimax_music3", "task_defaults", "/audioapi/v1/tasks/run"),
    ("gen", "minimax_h3", "task_defaults", "/audioapi/v1/tasks/run"),
    ("gen", "controlfoley", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vc", "seed_vc", "task_defaults", "/audioapi/v1/tasks/run"),
    ("svc", "seed_vc", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vc", "miocodec", "task_defaults", "/audioapi/v1/tasks/run"),
    ("s2s", "miocodec", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vc", "rvc", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vc", "meanvc2", "task_defaults", "/audioapi/v1/tasks/run"),
    ("s2s", "audiosr", "task_defaults", "/audioapi/v1/tasks/run"),
    ("s2s", "personaplex", "task_defaults", "/audioapi/v1/tasks/run"),
    ("tts", "vevo2", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vc", "vevo2", "task_defaults", "/audioapi/v1/tasks/run"),
    ("s2s", "vevo2", "task_defaults", "/audioapi/v1/tasks/run"),
    ("svc", "vevo2", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vad", "silero_vad", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vad", "marblenet_vad", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vad", "marblenet", "task_defaults", "/audioapi/v1/tasks/run"),
    ("diar", "sortformer_diar", "task_defaults", "/audioapi/v1/tasks/run"),
    ("diar", "sortformer", "task_defaults", "/audioapi/v1/tasks/run"),
    ("diar", "sortformer_diar_v2", "task_defaults", "/audioapi/v1/tasks/run"),
    ("sep", "htdemucs", "task_defaults", "/audioapi/v1/tasks/run"),
    ("sep", "mel_band_roformer", "task_defaults", "/audioapi/v1/tasks/run"),
    ("sep", "bs_roformer", "task_defaults", "/audioapi/v1/tasks/run"),
    ("align", "qwen3_forced_aligner", "task_defaults", "/v1/audio/alignments"),
    ("align", "mms_forced_aligner", "task_defaults", "/v1/audio/alignments"),
]

UNKNOWN_FAMILIES = [
    ("tts", "unknown_family"),
    ("asr", "whisper_large"),
    ("gen", "musicgen"),
    ("vc", "openvoice"),
    ("vad", "webrtc_vad"),
    ("sep", "spleeter"),
    ("align", "gentle"),
]

TTS_FAMILIES = [
    "chatterbox",
    "index_tts2",
    "neutts",
    "miotts",
    "moss_tts",
    "moss_tts_local",
    "moss_tts_nano",
    "omnivoice",
    "pocket_tts",
    "voxcpm2",
    "higgs_audio_tts",
    "irodori_tts",
    "supertonic",
    "vibevoice",
    "qwen3_tts",
    "f5_tts",
    "habibi",
    "habibi_tts",
    "chatterbox_turbo",
    "confucius4_tts",
    "dots_tts",
    "dramabox",
    "fish_audio",
    "firered_audio",
    "fireredtts3",
    "glm_tts",
    "inflect_v2",
    "magpie_tts",
    "moss_voicegen",
    "outetts",
    "soprano_tts",
    "vietneu_tts",
]

ASR_FAMILIES = [
    "citrinet",
    "citrinet_asr",
    "higgs_audio_stt",
    "hviske",
    "hviske_asr",
    "nemotron_asr",
    "vibevoice",
    "vibevoice_asr",
    "qwen3_asr",
    "parakeet_tdt",
    "fun_asr_nano",
    "voxtral_realtime",
    "kroko_asr",
    "granite5asr",
    "sense_asr",
    "firered_audio",
    "vibevoice_asr_streaming",
]

GEN_FAMILIES = [
    "ace_step",
    "stable_audio",
    "heartmula",
    "midashenglm_gen",
    "minimax_music3",
    "minimax_h3",
    "controlfoley",
]

VC_FAMILIES = [
    "seed_vc",
    "miocodec",
    "vevo2",
    "rvc",
    "meanvc2",
    "audiosr",
    "personaplex",
]

ANALYSIS_FAMILIES = [
    "silero_vad",
    "marblenet_vad",
    "marblenet",
    "sortformer_diar",
    "sortformer_diar_v2",
    "sortformer",
]

SEP_FAMILIES = ["htdemucs", "mel_band_roformer", "bs_roformer"]

ALIGN_FAMILIES = ["qwen3_forced_aligner", "mms_forced_aligner"]


def assert_profile_shape(profile: dict) -> None:
    assert isinstance(profile, dict)
    assert profile.get("label")
    assert isinstance(profile.get("workflows"), list)
    assert profile.get("summary")
    assert profile.get("api_hint") or profile.get("summary")


def assert_field_groups_shape(groups: list) -> None:
    assert isinstance(groups, list)
    for group in groups:
        assert group.get("id")
        assert group.get("label")
        assert isinstance(group.get("fields"), list)
        for field in group["fields"]:
            assert field.get("key")
            assert field.get("label")
            assert field.get("type")
