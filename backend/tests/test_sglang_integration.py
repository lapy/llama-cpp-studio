"""Contracts for upstream SGLang and the SGLang-V100 engine variant."""

from pathlib import Path

import pytest

import backend.engine_param_scanner as engine_param_scanner
import backend.llama_swap_config as llama_swap_config
from backend.cli_help_parsers import (
    parse_sglang_launch_server_help,
    sglang_params_to_sections,
)
from backend.cuda_installer import CUDAInstaller
from backend.engine_registry import ENGINE_REGISTRY
from backend.model_schema import compatible_engines_for_record
from backend.progress_manager import get_progress_manager
from backend.sglang_manager import SglangManager
from backend.venv_install_settings import default_install_settings


_V100_INSTALLER_FIXTURE = """#!/usr/bin/env bash
set -Eeuo pipefail
log() { printf '%s\\n' "$*"; }
die() { printf '%s\\n' "$*" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
[[ -d "$REPO_ROOT/.git" ]] || die "not a checkout"
if [[ ${EUID} -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi
"${SUDO[@]}" apt-get update
if ! command -v conda >/dev/null 2>&1; then
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
conda activate sglang-v100
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0
# Use every CPU only when RAM can sustain that many compiler processes.
python -m pip install torch==2.9.1
python -m pip install -e "$REPO_ROOT/python[diffusion-v100]"
log "Complete. Run: conda activate sglang-v100"
"""


def test_sglang_engines_are_registered_for_hf_snapshots():
    assert ENGINE_REGISTRY["sglang"].runtime_kind == "sglang"
    assert ENGINE_REGISTRY["sglang_v100"].runtime_kind == "sglang"
    assert ENGINE_REGISTRY["sglang_v100"].experimental is True
    assert "sglang" in compatible_engines_for_record({"format": "safetensors"})
    assert "sglang_v100" in compatible_engines_for_record({"format": "safetensors"})


def test_sglang_install_defaults_distinguish_upstream_and_v100():
    upstream = default_install_settings("sglang")
    v100 = default_install_settings("sglang_v100")
    assert "sgl-project/sglang" in upstream["source_repo"]
    assert upstream["pip_version"] == ""
    assert "haohervchb/sglang-V100" in v100["source_repo"]
    assert "pip_version" not in v100


def test_sglang_help_parser_reserves_studio_owned_model_and_port():
    help_text = """usage: launch_server.py [-h] --model-path MODEL_PATH [--port PORT] [--tp-size TP_SIZE]

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Model path.
  --port PORT           Port (default: 30000)
  --tp-size TP_SIZE     Tensor parallel size (default: 1)
"""
    sections = sglang_params_to_sections(parse_sglang_launch_server_help(help_text))
    params = {row["key"]: row for section in sections for row in section["params"]}
    assert params["model_path"]["reserved"] is True
    assert params["port"]["reserved"] is True
    assert params["tp_size"]["reserved"] is False
    assert params["tp_size"]["default"] == 1


def test_v100_parameter_scan_uses_sm70_environment(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    python_bin = venv / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    python_bin.chmod(0o755)
    captured = {}

    def fake_help(argv, **kwargs):
        captured.update(kwargs.get("extra_env") or {})
        return "options:\n  --tp-size TP_SIZE  Tensor parallel size.\n", None

    class FakeCudaInstaller:
        def get_cuda_env(self, version=None):
            assert version == "12.8"
            return {"CUDA_HOME": "/studio/cuda-12.8"}

    import backend.cuda_installer as cuda_installer

    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: FakeCudaInstaller())
    monkeypatch.setattr(engine_param_scanner, "_run_help_argv", fake_help)
    result = engine_param_scanner.scan_sglang_version(
        {"venv_path": str(venv)}, "sglang_v100"
    )

    assert result["scan_error"] is None
    assert captured["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert captured["TORCH_CUDA_ARCH_LIST"] == "7.0"


def test_sglang_preview_builds_openai_server_command(monkeypatch):
    monkeypatch.setattr(
        llama_swap_config, "_resolve_sglang_bin", lambda engine: f"/opt/{engine}/bin/python"
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_param_index",
        lambda engine: {
            "tensor_parallel_size": {
                "primary_flag": "--tensor-parallel-size",
                "value_kind": "scalar",
            }
        },
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_resolve_sglang_cuda_env",
        lambda engine: {
            "CUDA_HOME": "/studio/cuda/cuda-12.8",
            "LD_LIBRARY_PATH": "/studio/cuda/cuda-12.8/lib64",
        },
    )
    preview = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "id": "model",
            "huggingface_id": "org/model",
            "format": "safetensors",
            "config": {
                "engine": "sglang_v100",
                "engines": {
                    "sglang_v100": {
                        "tensor_parallel_size": 4,
                        "swap_env": {"NCCL_P2P_LEVEL": "NVL"},
                    }
                },
            },
        }
    )
    assert preview["ok"] is True
    assert "-m sglang.launch_server" in preview["cmd"]
    assert "--model-path org/model" in preview["cmd"]
    assert "--port ${PORT}" in preview["cmd"]
    assert "--tensor-parallel-size 4" in preview["cmd"]
    assert sorted(preview["env"]) == [
        "CUDA_HOME=/studio/cuda/cuda-12.8",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        "LD_LIBRARY_PATH=/studio/cuda/cuda-12.8/lib64",
        "NCCL_P2P_LEVEL=NVL",
        "TORCH_CUDA_ARCH_LIST=7.0",
    ]


def test_v100_installer_uses_studio_python_and_cuda(tmp_path):
    manager = SglangManager(
        "sglang_v100",
        base_dir=str(tmp_path / "installs"),
        log_path=str(tmp_path / "sglang-v100.log"),
    )
    manager._prepare_versioned_paths("source")
    checkout = Path(manager._base_dir) / "source"
    script = checkout / "scripts" / "install_v100.sh"
    script.parent.mkdir(parents=True)
    (checkout / ".git").mkdir()
    script.write_text(_V100_INSTALLER_FIXTURE, encoding="utf-8")
    patched_path = Path(manager._write_v100_prefix_installer(str(checkout)))
    patched = patched_path.read_text(encoding="utf-8")
    assert patched_path.parent == checkout / "scripts"
    assert "SGLANG_STUDIO_PYTHON" in patched
    assert "SGLANG_STUDIO_CUDA_HOME" in patched
    assert "python -m pip install torch==2.9.1" in patched
    assert "apt-get" not in patched
    assert "conda" not in patched.lower()
    assert "/usr/local/cuda-12.8" not in patched


@pytest.mark.asyncio
async def test_v100_install_passes_studio_environment_to_fork(tmp_path, monkeypatch):
    manager = SglangManager(
        "sglang_v100",
        base_dir=str(tmp_path / "installs"),
        log_path=str(tmp_path / "sglang-v100.log"),
    )
    manager._prepare_versioned_paths("source")
    checkout = Path(manager._base_dir) / "source"
    script = checkout / "scripts" / "install_v100.sh"
    script.parent.mkdir(parents=True)
    (checkout / ".git").mkdir()
    script.write_text(_V100_INSTALLER_FIXTURE, encoding="utf-8")
    captured = {}

    async def fake_run(argv, operation, **kwargs):
        captured.update(kwargs.get("env") or {})
        return 0

    monkeypatch.setattr(manager, "_run_logged", fake_run)
    await manager._install_source_checkout(
        str(checkout),
        {
            "PATH": "/studio/venv/bin:/studio/cuda/bin:/usr/bin",
            "SGLANG_STUDIO_PYTHON": "/studio/venv/bin/python",
            "SGLANG_STUDIO_CUDA_HOME": "/studio/cuda/cuda-12.8",
            "CUDA_HOME": "/studio/cuda/cuda-12.8",
        },
    )

    assert captured["SGLANG_STUDIO_PYTHON"] == "/studio/venv/bin/python"
    assert captured["CUDA_HOME"] == "/studio/cuda/cuda-12.8"
    assert captured["HOME"] == str(Path(manager._base_dir) / "home")
    assert captured["SGLANG_V100_DEPS_DIR"] == str(
        Path(manager._base_dir) / "dependencies"
    )
    assert captured["CARGO_HOME"] == str(Path(manager._base_dir) / "cargo-home")
    assert captured["CARGO_TARGET_DIR"] == str(
        Path(manager._base_dir) / "cargo-target"
    )
    assert Path(captured["CARGO_HOME"]).is_dir()
    assert Path(captured["CARGO_TARGET_DIR"]).is_dir()


def test_v100_progress_uses_monotonic_integer_stages(tmp_path):
    manager = SglangManager(
        "sglang_v100",
        base_dir=str(tmp_path / "installs"),
        log_path=str(tmp_path / "sglang-v100.log"),
    )
    progress = 0
    observed = []
    for line in (
        "Downloading attrs-26.1.0.whl",
        "Downloading another dependency",
        "Building editable for sglang (pyproject.toml): started",
        "random compiler output",
        "[install_v100] Building lean SM70-only sglang-kernel",
        "[install_v100] Running SM70 smoke checks",
    ):
        progress, _stage = manager._progress_stage_for_line(line, progress)
        observed.append(progress)

    assert observed == sorted(observed)
    assert observed == [16, 16, 22, 22, 70, 93]
    assert all(isinstance(value, int) for value in observed)


@pytest.mark.asyncio
async def test_sglang_progress_update_cannot_move_backward(tmp_path):
    manager = SglangManager(
        "sglang_v100",
        base_dir=str(tmp_path / "installs"),
        log_path=str(tmp_path / "sglang-v100.log"),
    )
    task_id = get_progress_manager().create_task(
        "install", "Install SGLang V100"
    )
    manager._progress_task_id = task_id

    await manager._update_progress_task(64, "Building kernels")
    await manager._update_progress_task(8, "Source checkout ready")

    task = get_progress_manager().get_task(task_id)
    assert task["progress"] == 64.0
    assert task["message"] == "Source checkout ready"


def test_v100_build_environment_comes_from_studio_cuda(tmp_path, monkeypatch):
    manager = SglangManager(
        "sglang_v100",
        base_dir=str(tmp_path / "installs"),
        log_path=str(tmp_path / "sglang-v100.log"),
    )
    manager._prepare_versioned_paths("source")
    python_bin = Path(manager._venv_path) / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    cuda_path = tmp_path / "cuda" / "cuda-12.8"
    (cuda_path / "bin").mkdir(parents=True)
    (cuda_path / "bin" / "nvcc").write_text("", encoding="utf-8")

    class FakeCudaInstaller:
        def get_cuda_env(self, version=None):
            assert version == "12.8"
            return {
                "CUDA_HOME": str(cuda_path),
                "CUDA_PATH": str(cuda_path),
                "PATH": f"{cuda_path}/bin:/usr/bin",
                "LD_LIBRARY_PATH": f"{cuda_path}/lib64",
            }

    import backend.cuda_installer as cuda_installer
    import backend.sglang_manager as sglang_manager

    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: FakeCudaInstaller())
    monkeypatch.setattr(
        sglang_manager.shutil,
        "which",
        lambda tool, path=None: f"/usr/bin/{tool}",
    )
    env = manager._v100_build_environment()

    assert env["CUDA_HOME"] == str(cuda_path)
    assert env["SGLANG_STUDIO_CUDA_HOME"] == str(cuda_path)
    assert env["SGLANG_STUDIO_PYTHON"] == str(python_bin)
    assert env["PATH"].split(":", 1)[0] == str(python_bin.parent)
    assert env["TORCH_CUDA_ARCH_LIST"] == "7.0"


def test_cuda_version_selection_does_not_fall_back_to_current(tmp_path, monkeypatch):
    installer = CUDAInstaller(
        log_path=str(tmp_path / "cuda.log"),
        state_path=str(tmp_path / "cuda.json"),
        download_dir=str(tmp_path / "downloads"),
    )
    cuda_root = tmp_path / "cuda"
    cuda_12 = cuda_root / "cuda-12.8"
    cuda_13 = cuda_root / "cuda-13.0"
    for path in (cuda_12, cuda_13):
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "nvcc").write_text("", encoding="utf-8")
    (cuda_root / "current").symlink_to(cuda_13, target_is_directory=True)
    installer._cuda_install_dir = str(cuda_root)
    monkeypatch.setattr(
        installer,
        "_load_state",
        lambda: {
            "installations": {
                "12.8": {"path": str(cuda_12)},
                "13.0": {"path": str(cuda_13)},
            }
        },
    )

    assert installer._get_cuda_path("12.8") == str(cuda_12)
    assert installer._get_cuda_path("12.7") is None


def test_sglang_build_settings_routes(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)

    response = client.get("/api/sglang/build-settings")
    assert response.status_code == 200
    assert response.json()["source_branch"] == "main"

    response = client.put(
        "/api/sglang-v100/build-settings",
        json={
            "source_repo": "https://github.com/example/sglang-v100.git",
            "source_branch": "sm70",
        },
    )
    assert response.status_code == 200
    assert store.get_engine_build_settings("sglang_v100")["source_branch"] == "sm70"
