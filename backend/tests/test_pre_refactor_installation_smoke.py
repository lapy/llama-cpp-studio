"""Focused upgrade smoke: a pre-refactor installation state still works.

The refactor changed how engines are discovered, installed and launched, but it
did not promise a config migration. This test writes the state an installation
from before the refactor actually has — ``engines.yaml`` rows in their historical
shape, an ``active_version``, real paths on disk, and NO
``install_snapshots.yaml`` ledger — and then exercises the paths an operator
would use after upgrading:

* the rows are listed with their stored binary/venv paths intact;
* the generic and compatibility status surfaces agree;
* activation through the shared version route persists;
* a launch compiles from the OLD venv path;
* deleting an inactive version removes its row;
* an install with no snapshot is retryable (the new digest guard does not
  invent a snapshot and refuse an upgrade).

Nothing here rewrites or migrates ``engines.yaml``; the test also asserts the
document's version rows are unchanged after the reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend import data_store
from backend.data_store import DataStore


def _touch_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _old_row(version: str, **overrides):
    row = {"version": version, "type": "source", "install_type": "source"}
    row.update(overrides)
    return row


@pytest.fixture()
def old_install(tmp_path, monkeypatch):
    """The on-disk state of a pre-refactor installation."""
    config = tmp_path / "config"
    config.mkdir()

    llama_binary = _touch_executable(tmp_path / "llama-cpp" / "source-main" / "llama-server")
    ik_binary = _touch_executable(tmp_path / "ik-llama" / "source-old" / "llama-server")
    unsloth_binary = _touch_executable(tmp_path / "unsloth" / "b12000" / "llama-server")
    audio_binary = _touch_executable(tmp_path / "audio-cpp" / "source-0.1" / "audio-server")
    lmdeploy_venv = tmp_path / "venvs" / "lmdeploy"
    vllm_venv = tmp_path / "venvs" / "vllm"
    # A real pre-refactor venv reports its distribution version through its
    # interpreter; the fixture stands in for that metadata probe.
    _touch_executable(lmdeploy_venv / "bin" / "python", "#!/bin/sh\necho 0.6.0\n")
    _touch_executable(lmdeploy_venv / "bin" / "lmdeploy")
    _touch_executable(vllm_venv / "bin" / "python")

    documents = {
        "engines.yaml": {
            # No schema_version: this file predates the document schema.
            "llama_cpp": {
                "active_version": "source-main",
                "versions": [
                    _old_row(
                        "source-main",
                        binary_path=str(llama_binary),
                        source_ref="main",
                        source_ref_type="branch",
                        source_branch="main",
                        source_repo="https://example.test/llama.cpp.git",
                        build_config={"enable_cuda": False, "build_type": "Release"},
                        repository_source="llama.cpp",
                        build_status="ready",
                    )
                ],
            },
            "ik_llama": {
                "active_version": "source-old",
                "versions": [
                    _old_row(
                        "source-old",
                        binary_path=str(ik_binary),
                        source_ref="main",
                        source_ref_type="branch",
                        source_repo="https://example.test/ik_llama.cpp.git",
                        build_config={"enable_cuda": False, "build_type": "Release"},
                        repository_source="ik_llama.cpp",
                        build_status="ready",
                    )
                ],
            },
            "unsloth_llama": {
                "active_version": "b12000",
                "versions": [
                    _old_row(
                        "b12000",
                        type="release",
                        install_type="release",
                        binary_path=str(unsloth_binary),
                        install_dir=str(unsloth_binary.parent),
                        build_status="ready",
                    )
                ],
            },
            "audio_cpp": {
                "active_version": "source-0.1",
                "versions": [
                    _old_row(
                        "source-0.1",
                        binary_path=str(audio_binary),
                        install_dir=str(audio_binary.parent),
                        source_ref="main",
                        source_ref_type="branch",
                        build_status="ready",
                    )
                ],
            },
            "lmdeploy": {
                "active_version": "0.6.0",
                "versions": [
                    {
                        "version": "0.6.0",
                        "type": "pip",
                        "install_type": "pip",
                        "venv_path": str(lmdeploy_venv),
                        "build_status": "ready",
                    }
                ],
            },
            "vllm": {
                "active_version": None,
                "versions": [
                    {
                        "version": "0.8.0",
                        "type": "pip",
                        "install_type": "pip",
                        "venv_path": str(vllm_venv),
                        "build_status": "ready",
                    }
                ],
            },
            "cuda": {"installed_version": None, "install_path": None},
        },
        "settings.yaml": {
            "huggingface_token": "",
            "proxy_port": 2000,
            "proxy": {"backend": "studio", "allow_anonymous": True},
        },
        "models.yaml": {"schema_version": 2, "models": []},
    }
    for name, document in documents.items():
        (config / name).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    original = (config / "engines.yaml").read_text(encoding="utf-8")

    store = DataStore(config_dir=str(config))
    monkeypatch.setattr(data_store, "_store", store)
    return {
        "config": config,
        "store": store,
        "original": original,
        "llama_binary": llama_binary,
        "lmdeploy_venv": lmdeploy_venv,
        "vllm_venv": vllm_venv,
    }


def test_pre_refactor_rows_list_with_their_paths(old_install, client):
    response = client.get("/api/llama-versions")
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}

    assert rows["llama_cpp:source-main"]["binary_path"] == str(old_install["llama_binary"])
    assert rows["unsloth_llama:b12000"]["binary_path"].endswith("llama-server")
    assert rows["lmdeploy:0.6.0"]["venv_path"] == str(old_install["lmdeploy_venv"])

    # The read path did not rewrite the historical document.
    assert (old_install["config"] / "engines.yaml").read_text(
        encoding="utf-8"
    ) == old_install["original"]
    # No snapshot ledger is invented for an installation that never had one.
    assert not (old_install["config"] / "install_snapshots.yaml").exists()


def test_generic_and_compat_status_agree_for_an_old_row(old_install, client):
    generic = client.get("/api/engines/lmdeploy/status")
    compat = client.get("/api/lmdeploy/status")
    assert generic.status_code == 200, generic.text
    assert compat.status_code == 200, compat.text
    assert generic.json() == compat.json()
    assert generic.json()["installed"] is True
    assert generic.json()["version"] == "0.6.0"


def test_activation_and_launch_use_the_old_venv(old_install, client, stub_studio_proxy):
    activated = client.post(
        "/api/llama-versions/versions/activate",
        json={"version_id": "lmdeploy:0.6.0"},
    )
    assert activated.status_code == 200, activated.text
    assert old_install["store"].get_active_engine_version("lmdeploy")["version"] == "0.6.0"

    from backend.proxy import profile_launch

    spec = profile_launch.compile_profile_launch(
        {"huggingface_id": "org/model"}, {}, "model-stable", "lmdeploy"
    )
    assert spec is not None
    assert spec.executable == str(old_install["lmdeploy_venv"] / "bin" / "lmdeploy")
    assert "org/model" in spec.argv


def test_removing_an_inactive_old_version_cleans_the_row(old_install, client):
    removed = client.delete("/api/llama-versions/vllm:0.8.0")
    assert removed.status_code == 200, removed.text
    assert old_install["store"].get_engine_versions("vllm") == []
    # Other engines' rows are untouched.
    assert old_install["store"].get_engine_versions("lmdeploy")[0]["version"] == "0.6.0"


def test_retry_of_a_pre_refactor_install_has_no_snapshot_guard(old_install):
    from backend.engines.install_snapshots import (
        install_snapshot_for,
        retry_resolved_inputs,
    )

    assert install_snapshot_for(old_install["store"], "llama_cpp", "source-main") == {}
    # No recorded digest: the current definition is accepted instead of
    # refusing an install that predates the snapshot ledger.
    assert (
        retry_resolved_inputs(
            old_install["store"],
            "llama_cpp",
            "source-main",
            current_digest="sha256:" + "f" * 64,
            label="llama.cpp",
        )
        == {}
    )
    assert not (old_install["config"] / "install_snapshots.yaml").exists()
