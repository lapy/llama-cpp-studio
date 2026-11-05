"""Shared fixtures and expectations for audio.cpp profile tests."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

# Live capture of ``audiocpp_cli --list-loaders --json`` shape from the v0.9.1
# line. The activation route gate parses this JSON before an engine version may
# become active, so tests that activate through the real gate need a CLI binary
# able to answer it.
V091_LIST_LOADERS_JSON = {
    "schema_version": 1,
    "loaders": [
        {
            "family": "omnivoice",
            "tasks": [{"id": "tts", "modes": ["offline", "streaming"]}],
            "api_endpoints": ["/v1/audio/speech"],
            "instructions_policy": "soft_tags",
        },
        {
            "family": "qwen3_asr",
            "tasks": [{"id": "asr", "modes": ["offline"]}],
            "api_endpoints": ["/v1/audio/transcriptions"],
        },
    ],
}

_FAKE_CLI_TEMPLATE = """#!/bin/sh
for arg in "$@"; do
  if [ "$arg" = "--json" ]; then
    printf '%s' {payload!r}
    exit 0
  fi
done
printf 'Usage: audiocpp_cli [options]\\n  --model <path>  Model path\\n'
"""


def write_fake_v091_cli(path: Path, *, payload: dict | None = None) -> Path:
    """Create an executable sh stand-in answering ``--list-loaders --json``."""
    body = _FAKE_CLI_TEMPLATE.format(payload=json.dumps(payload or V091_LIST_LOADERS_JSON))
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def fake_audio_binaries(tmp_path: Path) -> tuple[str, str]:
    """Executable server+cli pair accepted by the activation route gate."""
    server = tmp_path / "audiocpp_server"
    server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    server.chmod(server.stat().st_mode | stat.S_IXUSR | os.X_OK)
    cli = write_fake_v091_cli(tmp_path / "audiocpp_cli")
    return str(server), str(cli)


# One representative per task contract; family identity is engine-owned.
AUDIO_TASK_CONTRACTS = [
    ("tts", "unseen_tts", "speech_defaults", "/v1/audio/speech"),
    ("clon", "unseen_clone", "speech_defaults", "/v1/audio/speech"),
    ("vdes", "unseen_voice", "speech_defaults", "/v1/audio/speech"),
    ("asr", "unseen_asr", "transcription_defaults", "/v1/audio/transcriptions"),
    ("gen", "unseen_gen", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vc", "unseen_vc", "task_defaults", "/audioapi/v1/tasks/run"),
    ("svc", "unseen_vc", "task_defaults", "/audioapi/v1/tasks/run"),
    ("s2s", "unseen_vc", "task_defaults", "/audioapi/v1/tasks/run"),
    ("vad", "unseen_analysis", "task_defaults", "/audioapi/v1/tasks/run"),
    ("diar", "unseen_analysis", "task_defaults", "/audioapi/v1/tasks/run"),
    ("sep", "unseen_sep", "task_defaults", "/audioapi/v1/tasks/run"),
    ("align", "unseen_align", "task_defaults", "/v1/audio/alignments"),
]


def assert_profile_shape(profile: dict) -> None:
    assert isinstance(profile, dict)
    assert profile.get("label")
    assert isinstance(profile.get("workflows"), list)
    assert profile.get("summary")
