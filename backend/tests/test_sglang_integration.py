"""Contracts for upstream SGLang and the SGLang-V100 engine variant."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import backend.engines.scan.scanner as engine_param_scanner
import backend.proxy.commands as runtime_commands
from backend.engines.scan.help_parsers import (
    parse_sglang_launch_server_help,
    sglang_params_to_sections,
)
from backend.cuda_installer import CUDAInstaller
from backend.engines.definitions import load_default_engine_definitions, materialize_plan
from backend.engines.definitions.transforms import (
    TransformError,
    apply_transform,
    load_transform,
)
from backend.engines.python import fork_build
from backend.engines.python.fork_build import prepare_fork_build
from backend.engines.python.generic import GenericVenvInstaller
from backend.engines.python.log_streaming import progress_milestone
from backend.engines.registry import ENGINE_REGISTRY
from backend.engines.sglang import SglangManager
from backend.models.schema import compatible_engines_for_record
from backend.operations.progress import get_progress_manager
from backend.venv_install_settings import default_install_settings


def _v100_installer(tmp_path) -> GenericVenvInstaller:
    definition = load_default_engine_definitions().definitions["sglang_v100"]
    return GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "sglang-v100.log"),
        base_dir=str(tmp_path / "installs"),
    )


def _v100_milestones() -> list:
    lifecycle = dict(load_default_engine_definitions().definitions["sglang_v100"].lifecycle)
    return [dict(item) for item in lifecycle["log"]["progress_milestones"]]


def _v100_option(step_id: str) -> dict:
    plan = materialize_plan(load_default_engine_definitions().definitions["sglang_v100"], "source")
    return dict(plan.by_id(step_id).with_)


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
prepare_sparse_repo() {
  local destination=$1
  shift
  git -C "$destination" sparse-checkout set "$@"
}
prepare_sparse_repo \
  "LMDeploy/1Cat TurboMind" https://github.com/1CatAI/1Cat-vLLM.git \
  "$TURBOMIND_SOURCE_REV" "$TURBOMIND_SOURCE_DIR" \
  LICENSE csrc/core csrc/sm70_turbomind csrc/moe
export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON"
log "Building V100 Marlin GPTQ/AWQ kernels"
bash "$REPO_ROOT/scripts/setup_v100_marlin.sh"
log "Complete. Run: conda activate sglang-v100"
"""

_V100_MARLIN_SETUP_FIXTURE = """#!/usr/bin/env bash
set -euo pipefail
log() { printf '%s\\n' "$*"; }
REPO="${MARLIN_V100_REPO:-$HOME/marlin_v100}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export CMAKE_ARGS="${CMAKE_ARGS:-}"
log "building (MAX_JOBS=$MAX_JOBS) ... this takes ~15-40 min"
( cd "$REPO" && bash "$REPO/build.sh" )
"""

_V100_MARLIN_PATCH_LOOP = load_transform("sglang_v100/marlin_builder").operations[0]["search"]


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
    result = engine_param_scanner.scan_definition_engine_version(
        "sglang_v100", {"venv_path": str(venv)}
    )

    assert result["scan_error"] is None
    assert captured["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert captured["TORCH_CUDA_ARCH_LIST"] == "7.0"


def test_sglang_scan_keeps_diffusion_parameters_separate(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    python_bin = venv / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    python_bin.chmod(0o755)

    def fake_help(argv, **_kwargs):
        if "diffusion" in argv:
            return (
                "options:\n"
                "  --quantization QUANTIZATION\n"
                "                        Diffusion quantization method.\n"
                "  --enable-torch-compile ENABLE_TORCH_COMPILE\n"
                "  --text-encoder-cpu-offload [TEXT_ENCODER_CPU_OFFLOAD]\n"
                "                        Use CPU offload for text encoder. Enable if run out of memory.\n"
                "  --vae-cpu-offload [VAE_CPU_OFFLOAD]\n"
                "                        Use CPU offload for VAE. Enable if run out of memory.\n",
                None,
            )
        return "options:\n  --tp-size TP_SIZE  Tensor parallel size.\n", None

    monkeypatch.setattr(engine_param_scanner, "_run_help_argv", fake_help)
    result = engine_param_scanner.scan_definition_engine_version("sglang", {"venv_path": str(venv)})

    llm_keys = {row["key"] for section in result["sections"] for row in section["params"]}
    diffusion = result["runtime_profiles"]["diffusion"]
    diffusion_params = {
        row["key"]: row for section in diffusion["sections"] for row in section["params"]
    }
    assert llm_keys == {"tp_size"}
    assert diffusion["scan_error"] is None
    assert diffusion_params["quantization"]["value_kind"] == "scalar"
    assert not diffusion_params["quantization"].get("options")
    for key in ("text_encoder_cpu_offload", "vae_cpu_offload"):
        assert diffusion_params[key]["value_kind"] == "flag"
        assert diffusion_params[key]["type"] == "bool"
        assert diffusion_params[key]["default"] is False


def test_sglang_scan_reports_missing_optional_diffusion_support(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    python_bin = venv / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    python_bin.chmod(0o755)

    def fake_help(argv, **_kwargs):
        if "diffusion" in argv:
            return "Diffusion model support is not available. Install extras.\n", None
        return "options:\n  --tp-size TP_SIZE  Tensor parallel size.\n", None

    monkeypatch.setattr(engine_param_scanner, "_run_help_argv", fake_help)
    result = engine_param_scanner.scan_definition_engine_version("sglang", {"venv_path": str(venv)})

    assert result["scan_error"] is None
    assert "not available" in result["runtime_profiles"]["diffusion"]["scan_error"]


def test_v100_parameter_scan_uses_install_dir_venv(tmp_path, monkeypatch):
    install_dir = tmp_path / "20260918-165120-source"
    python_bin = install_dir / "venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    python_bin.chmod(0o755)

    class FakeCudaInstaller:
        def get_cuda_env(self, version=None):
            return {"CUDA_HOME": "/studio/cuda-12.8"}

    import backend.cuda_installer as cuda_installer

    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: FakeCudaInstaller())
    monkeypatch.setattr(
        engine_param_scanner,
        "_run_help_argv",
        lambda *a, **k: ("options:\n  --tp-size TP_SIZE  Tensor parallel size.\n", None),
    )
    result = engine_param_scanner.scan_definition_engine_version(
        "sglang_v100", {"install_dir": str(install_dir)}
    )
    assert result["scan_error"] is None
    assert result["binary_path"] == str(python_bin)


def _patch_profile_preview(monkeypatch, tmp_path, *, cuda_env, with_venv=True):
    """Phase 6: previews resolve through the exported profile (active row +
    Studio-managed CUDA env), not the legacy resolver symbols."""
    import stat

    install = tmp_path / "install"
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(python_bin.stat().st_mode | stat.S_IXUSR)

    import backend.cuda_installer as cuda_installer
    import backend.data_store as data_store

    class _Store:
        def get_active_engine_version(self, engine_id):
            return {"version": "v1", "venv_path": str(install), "cuda_version": "12.8"}

    class _Cuda:
        def get_cuda_env(self, version):
            return dict(cuda_env)

    monkeypatch.setattr(data_store, "get_store", lambda: _Store())
    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: _Cuda())


def test_sglang_preview_builds_openai_server_command(monkeypatch, tmp_path):
    _patch_profile_preview(
        monkeypatch,
        tmp_path,
        cuda_env={
            "CUDA_HOME": "/studio/cuda/cuda-12.8",
            "LD_LIBRARY_PATH": "/studio/cuda/cuda-12.8/lib64",
        },
    )
    monkeypatch.setattr(
        runtime_commands,
        "_active_engine_param_index",
        lambda engine: {
            "tensor_parallel_size": {
                "primary_flag": "--tensor-parallel-size",
                "value_kind": "scalar",
            }
        },
    )
    preview = runtime_commands.preview_model_command(
        {
            "id": "model",
            "huggingface_id": "org/model",
            "format": "safetensors",
            "config": {
                "engine": "sglang_v100",
                "engines": {
                    "sglang_v100": {
                        "tensor_parallel_size": 4,
                        "runtime_env": {"NCCL_P2P_LEVEL": "NVL"},
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
    assert set(preview["env"]) >= {
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "CUDA_HOME=/studio/cuda/cuda-12.8",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        "LD_LIBRARY_PATH=/studio/cuda/cuda-12.8/lib64",
        "NCCL_P2P_LEVEL=NVL",
        "TORCH_CUDA_ARCH_LIST=7.0",
    }


def test_sglang_diffusion_preview_uses_diffusion_cli_and_free_form_quantization(
    monkeypatch, tmp_path
):
    _patch_profile_preview(monkeypatch, tmp_path, cuda_env={"CUDA_HOME": "/studio/cuda/cuda-12.8"})
    monkeypatch.setattr(
        runtime_commands,
        "_active_engine_entry",
        lambda _engine: {
            "runtime_profiles": {
                "diffusion": {
                    "scan_error": None,
                    "sections": [
                        {
                            "params": [
                                {
                                    "key": "quantization",
                                    "primary_flag": "--quantization",
                                    "value_kind": "scalar",
                                }
                            ]
                        }
                    ],
                }
            }
        },
    )
    monkeypatch.setattr(
        runtime_commands,
        "_active_engine_param_index",
        lambda _engine: {
            "tensor_parallel_size": {
                "primary_flag": "--tensor-parallel-size",
                "value_kind": "scalar",
            }
        },
    )

    preview = runtime_commands.preview_model_command(
        {
            "id": "video-model",
            "huggingface_id": "org/video-model",
            "format": "safetensors",
            "config": {
                "engine": "sglang_v100",
                "engines": {
                    "sglang_v100": {
                        "sglang_runtime": "diffusion",
                        "quantization": "v100_w4a16_awq",
                        # Kept for switching back to LLM; must not reach diffusion.
                        "tensor_parallel_size": 8,
                    }
                },
            },
        }
    )

    assert preview["ok"] is True
    assert "from sglang.cli.main import main; main()" in preview["cmd"]
    assert "serve --model-type diffusion" in preview["cmd"]
    assert "--quantization v100_w4a16_awq" in preview["cmd"]
    assert "--tensor-parallel-size" not in preview["cmd"]


def test_v100_patches_cuda_ipc_tuple_access_for_nvcc_12_8(tmp_path):
    source = tmp_path / "source"
    ipc = source / "python" / "sglang" / "kernels" / "jit" / "csrc" / "distributed" / "ipc.cuh"
    ipc.parent.mkdir(parents=True)
    ipc.write_text(
        "const auto ipc_handle = to_ipc_handle(get<0>(pair));\nconst auto offset = get<1>(pair);\n",
        encoding="utf-8",
    )

    apply_transform("sglang_v100/cuda_ipc", str(source))
    patched = ipc.read_text(encoding="utf-8")

    assert "to_ipc_handle(pair.get<0>())" in patched
    assert "const auto offset = pair.get<1>();" in patched
    assert "get<0>(pair)" not in patched
    assert "get<1>(pair)" not in patched
    # Retrying an installation must remain idempotent.
    apply_transform("sglang_v100/cuda_ipc", str(source))


def test_v100_marlin_builder_strips_verbose_ptxas_and_exports_parallel_level(tmp_path):
    repo = tmp_path / "marlin-v100"
    repo.mkdir()
    (repo / "build.sh").write_text(
        '#!/usr/bin/env bash\n# ptxas_flag = "-Xptxas=-v"\n'
        'echo "parallel=$CMAKE_BUILD_PARALLEL_LEVEL"\n',
        encoding="utf-8",
    )
    (repo / "setup.py").write_text(
        "import os\n"
        'targets = ["_C", "_moe_C"]\n'
        'build_args = ["--build", ".", *[f"--target={target}" for target in targets]]\n',
        encoding="utf-8",
    )
    checkout = tmp_path / "sglang"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "setup_v100_marlin.sh").write_text(_V100_MARLIN_SETUP_FIXTURE, encoding="utf-8")

    apply_transform("sglang_v100/marlin_builder", str(checkout))
    env = os.environ.copy()
    env["MARLIN_V100_REPO"] = str(repo)
    env["PYTHON"] = sys.executable
    env["MAX_JOBS"] = "3"
    result = subprocess.run(
        ["bash", str(scripts / "setup_v100_marlin.sh")],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "-Xptxas=-v" not in (repo / "build.sh").read_text(encoding="utf-8")
    setup_py = (repo / "setup.py").read_text(encoding="utf-8")
    assert "--parallel" not in setup_py
    assert "CMAKE_BUILD_PARALLEL_LEVEL" in (scripts / "setup_v100_marlin.sh").read_text(
        encoding="utf-8"
    )
    assert "parallel=3" in result.stdout


def test_v100_marlin_builder_preserves_stale_patches_and_local_edits(tmp_path):
    repo = tmp_path / "marlin-v100"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "studio@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Studio"],
        cwd=repo,
        check=True,
    )
    (repo / "kernel.txt").write_text("hello\n", encoding="utf-8")
    (repo / "build.sh").write_text("#!/usr/bin/env bash\necho built\n", encoding="utf-8")
    (repo / "setup.py").write_text(
        'build_args = ["--build", ".", *[f"--target={target}" for target in targets]]\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "kernel.txt", "build.sh", "setup.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    origin = tmp_path / "marlin-origin"
    subprocess.run(
        ["git", "clone", "--bare", str(repo), str(origin)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=repo, check=True)
    (repo / "kernel.txt").write_text("patched\n", encoding="utf-8")
    patch = subprocess.check_output(["git", "diff"], cwd=repo, text=True)
    subprocess.run(["git", "checkout", "--", "kernel.txt"], cwd=repo, check=True)
    (repo / "kernel.txt").write_text("stale-old-patch\n", encoding="utf-8")
    (repo / "local-notes.txt").write_text("keep untracked work\n", encoding="utf-8")
    (repo / "build").mkdir()
    (repo / "build" / "cached-object.o").write_bytes(b"cached build")

    checkout = tmp_path / "sglang"
    scripts = checkout / "scripts"
    patches = checkout / "patches"
    scripts.mkdir(parents=True)
    patches.mkdir()
    (patches / "marlin-v100-qwen-sm70-tuning.patch").write_text(patch, encoding="utf-8")
    (scripts / "setup_v100_marlin.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'log() { printf "%s\\n" "$*"; }',
                'die() { printf "%s\\n" "$*" >&2; exit 1; }',
                'REPO="${MARLIN_V100_REPO}"',
                'MARLIN_V100_REF="${MARLIN_V100_REF}"',
                'PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches"',
                'SM70_PATCHES=("$PATCH_DIR/marlin-v100-qwen-sm70-tuning.patch")',
                "export MAX_JOBS=1",
                _V100_MARLIN_PATCH_LOOP,
                'log "building (MAX_JOBS=$MAX_JOBS) ... this takes ~15-40 min"',
                '( cd "$REPO" && bash "$REPO/build.sh" )',
                "",
            ]
        ),
        encoding="utf-8",
    )

    apply_transform("sglang_v100/marlin_builder", str(checkout))
    rewritten = (scripts / "setup_v100_marlin.sh").read_text(encoding="utf-8")
    assert "Preserved marlin_v100 checkout at" in rewritten
    assert "reset --hard" not in rewritten
    assert "clean -fd" not in rewritten
    assert 'die "SM70 compatibility patch does not apply cleanly: $SM70_PATCH"' not in rewritten

    env = os.environ.copy()
    env["MARLIN_V100_REPO"] = str(repo)
    env["MARLIN_V100_REF"] = pin
    env["PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", str(scripts / "setup_v100_marlin.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    assert (repo / "kernel.txt").read_text(encoding="utf-8") == "patched\n"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == pin
    backups = list(tmp_path.glob("marlin-v100.preserved-*/source"))
    assert len(backups) == 1
    preserved = backups[0]
    assert (preserved / "kernel.txt").read_text(encoding="utf-8") == "stale-old-patch\n"
    assert (preserved / "local-notes.txt").read_text(encoding="utf-8") == "keep untracked work\n"
    assert (preserved / "build" / "cached-object.o").read_bytes() == b"cached build"
    assert (preserved / ".git").is_dir()
    assert str(preserved) in result.stdout
    assert "Preserved marlin_v100 checkout at" in result.stdout
    assert "built" in result.stdout


def test_v100_progress_uses_monotonic_integer_stages(tmp_path):
    milestones = _v100_milestones()
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
        progress, _stage = progress_milestone(milestones, "SGLang V100", line, progress)
        observed.append(progress)

    assert observed == sorted(observed)
    assert observed == [16, 16, 22, 22, 70, 93]
    assert all(isinstance(value, int) for value in observed)


@pytest.mark.asyncio
async def test_sglang_progress_update_cannot_move_backward(tmp_path):
    manager = _v100_installer(tmp_path)
    task_id = get_progress_manager().create_task("install", "Install SGLang V100")
    manager._progress_task_id = task_id

    await manager._update_progress_task(64, "Building kernels")
    await manager._update_progress_task(8, "Source checkout ready")

    task = get_progress_manager().get_task(task_id)
    assert task["progress"] == 64.0
    assert task["message"] == "Source checkout ready"


@pytest.mark.asyncio
async def test_failed_command_replays_early_compiler_diagnostic(tmp_path):
    log_path = tmp_path / "sglang-v100.log"
    manager = SglangManager(
        "sglang_v100",
        base_dir=str(tmp_path / "installs"),
        log_path=str(log_path),
    )
    script = (
        "print('FAILED: marlin-kernel.o'); "
        "[print(f'ptxas info {i}') for i in range(250)]; "
        "raise SystemExit(1)"
    )

    returncode = await manager._run_logged([sys.executable, "-c", script], "install_source")

    output = log_path.read_text(encoding="utf-8")
    assert returncode == 1
    assert "--- captured failure context ---" in output
    assert output.count("FAILED: marlin-kernel.o") >= 2
    assert output.rfind("FAILED: marlin-kernel.o") > output.rfind("ptxas info 249")


@pytest.mark.asyncio
async def test_verbose_ptxas_lines_are_not_broadcast(tmp_path, monkeypatch):
    manager = _v100_installer(tmp_path)
    broadcasted = []

    async def fake_broadcast(line: str) -> None:
        broadcasted.append(line)

    monkeypatch.setattr(manager, "_broadcast_log_line", fake_broadcast)
    script_path = tmp_path / "compiler_output.py"
    script_path.write_text(
        "\n".join(
            [
                "print('[12/40] Building CUDA object marlin.cu.o')",
                "print('ptxas info    : 0 bytes gmem')",
                "print('    40 bytes stack frame, 0 bytes spill stores')",
                "print('ptxas fatal   : Out of memory')",
                "print('DONE')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    returncode = await manager._run_logged([sys.executable, str(script_path)], "install_source")

    log_text = Path(manager._log_path).read_text(encoding="utf-8")
    assert returncode == 0
    assert "ptxas info    : 0 bytes gmem" in log_text
    assert "bytes spill stores" in log_text
    assert any("Building CUDA object marlin.cu.o" in line for line in broadcasted)
    assert any(line == "DONE" for line in broadcasted)
    assert any("ptxas fatal" in line for line in broadcasted)
    assert not any(line.lstrip().startswith("ptxas info") for line in broadcasted)
    assert not any("bytes spill" in line.lower() for line in broadcasted)


@pytest.mark.asyncio
async def test_silent_compiler_gets_heartbeat(tmp_path, monkeypatch):
    manager = _v100_installer(tmp_path)
    manager._log_heartbeat_seconds = 0.2
    broadcasted = []

    async def fake_broadcast(line: str) -> None:
        broadcasted.append(line)

    monkeypatch.setattr(manager, "_broadcast_log_line", fake_broadcast)
    script_path = tmp_path / "slow_compile.py"
    script_path.write_text(
        "import time\ntime.sleep(0.55)\nprint('compiled')\n",
        encoding="utf-8",
    )

    returncode = await manager._run_logged([sys.executable, str(script_path)], "install_source")

    log_text = Path(manager._log_path).read_text(encoding="utf-8")
    assert returncode == 0
    assert "still compiling" in log_text
    assert any("still compiling" in line for line in broadcasted)
    assert any(line == "compiled" for line in broadcasted)


def test_v100_safe_jobs_keeps_parallelism_on_typical_hosts(monkeypatch):
    monkeypatch.setattr(fork_build, "_read_meminfo_available_kib", lambda: 20 * 1024 * 1024)
    monkeypatch.setattr(fork_build, "_read_cgroup_available_kib", lambda: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)

    # 20 GiB available used to become MAX_JOBS=1 with a 16 GiB reserve.
    assert fork_build.safe_jobs() == 4


def test_v100_safe_jobs_ignores_cgroup_page_cache(monkeypatch):
    monkeypatch.setattr(fork_build, "_read_meminfo_available_kib", lambda: 24 * 1024 * 1024)
    monkeypatch.setattr(fork_build, "_read_cgroup_available_kib", lambda: 20 * 1024 * 1024)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert fork_build.safe_jobs() == 4


@pytest.mark.asyncio
async def test_v100_build_environment_comes_from_studio_cuda(tmp_path, monkeypatch):
    manager = _v100_installer(tmp_path)
    manager._prepare_versioned_paths("source")
    python_bin = Path(manager._venv_path) / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    cuda_path = tmp_path / "cuda" / "cuda-12.8"
    (cuda_path / "bin").mkdir(parents=True)
    (cuda_path / "bin" / "nvcc").write_text("", encoding="utf-8")
    checkout = tmp_path / "source"
    checkout.mkdir()

    class FakeCudaInstaller:
        def get_cuda_env(self, version=None):
            return {
                "CUDA_HOME": str(cuda_path),
                "CUDA_PATH": str(cuda_path),
                "PATH": f"{cuda_path}/bin:/usr/bin",
                "LD_LIBRARY_PATH": f"{cuda_path}/lib64",
            }

        def status(self):
            return {"version": "12.8", "cuda_path": str(cuda_path)}

    import backend.cuda_installer as cuda_installer

    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: FakeCudaInstaller())
    monkeypatch.setattr(fork_build.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    monkeypatch.setattr(fork_build, "_compiler_major", lambda _compiler: 12)
    monkeypatch.setattr(fork_build, "safe_jobs", lambda: 3)
    monkeypatch.setenv("MAX_JOBS", "99")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "99")
    monkeypatch.setenv("NVCC_THREADS", "8")

    env = await prepare_fork_build(
        manager,
        _v100_option("prepare_build"),
        checkout=str(checkout),
        base_env=manager._compose_build_env(),
    )

    assert env["CUDA_HOME"] == str(cuda_path)
    assert env["SGLANG_STUDIO_CUDA_HOME"] == str(cuda_path)
    assert env["SGLANG_STUDIO_PYTHON"] == str(python_bin)
    assert env["SGLANG_V100_PYTHON"] == str(python_bin)
    assert env["PATH"].split(os.pathsep, 1)[0] == str(python_bin.parent)
    assert env["TORCH_CUDA_ARCH_LIST"] == "7.0"
    assert env["CUDAHOSTCXX"] == "/usr/bin/g++-12"
    assert env["MAX_JOBS"] == "3"
    assert env["CMAKE_BUILD_PARALLEL_LEVEL"] == "3"
    assert env["NVCC_THREADS"] == "1"


@pytest.mark.asyncio
async def test_v100_compiler_rejects_gcc_13(tmp_path, monkeypatch):
    manager = _v100_installer(tmp_path)
    manager._prepare_versioned_paths("source")
    python_bin = Path(manager._venv_path) / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        fork_build.shutil,
        "which",
        lambda tool, path=None: "/usr/bin/g++" if tool in {"g++", "c++"} else None,
    )
    monkeypatch.setattr(fork_build, "_compiler_major", lambda _compiler: 13)
    options = _v100_option("prepare_build")
    options["required_tools"] = []
    options["require_cuda_nvcc"] = False

    with pytest.raises(fork_build.ForkBuildError, match=r"GCC/G\+\+ 10–12"):
        await prepare_fork_build(
            manager,
            options,
            checkout=str(tmp_path / "source"),
            base_env={"PATH": "/usr/bin"},
        )


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


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/api/sglang-v100/install",
            {
                "source_repo": "https://github.com/third-party/sglang-v100.git",
                "source_branch": "custom-sm70",
            },
        ),
        (
            "/api/sglang-v100/install-source",
            {
                "repository_url": "https://github.com/third-party/sglang-v100.git",
                "ref": "custom-sm70",
            },
        ),
    ],
)
def test_v100_install_honors_explicit_third_party_fork(client, monkeypatch, endpoint, payload):
    """An explicit V100 fork must beat the maintained fork install default."""
    from dataclasses import replace

    from backend.routes.engine_versions import LIFECYCLE

    captured = {}

    class Installer:
        async def install_from_source(self, repo_url, branch):
            captured.update(repo_url=repo_url, branch=branch)
            return {"started": True}

    # The compatibility prefix is a thin delegate; inject at the lifecycle seam.
    monkeypatch.setitem(
        LIFECYCLE,
        "sglang_v100",
        replace(LIFECYCLE["sglang_v100"], get_installer=lambda: Installer()),
    )

    response = client.post(endpoint, json=payload)

    assert response.status_code == 200
    assert captured == {
        "repo_url": "https://github.com/third-party/sglang-v100.git",
        "branch": "custom-sm70",
    }
