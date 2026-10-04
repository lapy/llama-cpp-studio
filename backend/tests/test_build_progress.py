"""Tests for universal git / [x/y] / [N%] build progress parsing."""

import time

from backend.build_progress import (
    BuildProgressTracker,
    PipInstallProgressTracker,
    apply_build_step_progress,
    apply_cmake_stage,
    cmake_stage_start,
    cmake_stage_window,
    map_build_step_progress,
    map_git_phase_progress,
    parse_build_percent,
    parse_build_progress_ratio,
    parse_build_step_ratio,
    parse_git_progress,
    progress_from_install_log,
)


def test_parse_build_step_ratio_ninja_style():
    assert parse_build_step_ratio("[123/456] Building CXX object foo.cpp.o") == (123, 456)
    assert parse_build_step_ratio("[ 12/ 90 ] Linking CXX executable server") == (12, 90)
    assert parse_build_step_ratio("no counter here") is None


def test_parse_build_step_ratio_clamps_overshoot():
    assert parse_build_step_ratio("[10/5] weird") == (5, 5)


def test_parse_build_percent_makefile_style():
    assert parse_build_percent("[ 46%] Built target ggml-cuda") == 46
    assert parse_build_percent("[100%] Built target llama-server") == 100
    assert parse_build_percent("[  0%] Built target llama-common-base") == 0
    assert parse_build_percent("[7%] Provisioning UI assets") == 7
    assert parse_build_percent("[123/456] Building CXX object") is None
    assert parse_build_percent("no percent here") is None
    assert parse_build_percent("[101%] overshoot") is None
    # Git uses "46%" without brackets — must not match Makefile parser.
    assert parse_build_percent("Receiving objects:  46% (48213/104810)") is None


def test_parse_git_progress_from_install_log_samples():
    assert parse_git_progress("remote: Counting objects:  50% (99/197)") == (
        "counting",
        50,
        99,
        197,
    )
    assert parse_git_progress("remote: Compressing objects:  50% (53/106)") == (
        "compressing",
        50,
        53,
        106,
    )
    assert parse_git_progress(
        "Receiving objects:  46% (48213/104810), 153.47 MiB | 23.38 MiB/s"
    ) == ("receiving", 46, 48213, 104810)
    assert parse_git_progress("Resolving deltas:  50% (36700/73399)") == (
        "resolving",
        50,
        36700,
        73399,
    )
    assert parse_git_progress("Updating files: 100% (3258/3258), done.") == (
        "updating",
        100,
        3258,
        3258,
    )
    assert parse_git_progress("[50/100] Building CXX object") is None
    assert parse_git_progress("plain text") is None


def test_map_git_phase_progress_weights_receiving_heaviest():
    # Receiving owns most of the clone window.
    early = map_git_phase_progress("counting", 100, floor=2, ceil=18)
    mid = map_git_phase_progress("receiving", 50, floor=2, ceil=18)
    late = map_git_phase_progress("resolving", 100, floor=2, ceil=18)
    done = map_git_phase_progress("updating", 100, floor=2, ceil=18)
    assert early < mid < late <= done
    assert done == 18


def test_build_progress_tracker_advances_through_git_clone_phases():
    tracker = BuildProgressTracker(floor=2, ceil=18)
    values = []
    for line in (
        "remote: Counting objects: 100% (197/197), done.",
        "remote: Compressing objects: 100% (106/106), done.",
        "Receiving objects:  10% (10481/104810), 7.67 MiB | 15.32 MiB/s",
        "Receiving objects:  50% (52405/104810), 165.65 MiB | 23.67 MiB/s",
        "Receiving objects: 100% (104810/104810), 411.34 MiB | 23.86 MiB/s, done.",
        "Resolving deltas:   0% (0/73399)",
        "Resolving deltas: 100% (73399/73399), done.",
        "Updating files: 100% (3258/3258), done.",
    ):
        step = tracker.apply_line(line)
        assert step is not None
        values.append(step[0])
    assert values == sorted(values)
    assert values[0] >= 2
    assert values[-1] == 18
    # Resolving restart at 0% must not drop the bar after receiving finished.
    assert values[5] >= values[4]


def test_build_progress_tracker_set_window_resets_for_next_stage():
    tracker = BuildProgressTracker(floor=2, ceil=18, progress=18)
    tracker.set_window(28, 92, progress=28)
    step = tracker.apply_line("[243/486] Building CUDA object ggml-cuda.cu.o")
    assert step is not None
    progress, suffix = step
    assert suffix == "[243/486]"
    assert 28 <= progress <= 92
    assert progress == map_build_step_progress(243, 486, floor=28, ceil=92)


def test_parse_build_progress_ratio_prefers_ninja_then_percent():
    assert parse_build_progress_ratio("[50/100] step") == (50, 100)
    assert parse_build_progress_ratio("[ 46%] Built target ggml-cuda") == (46, 100)
    assert parse_build_progress_ratio("plain text") is None


def test_map_build_step_progress_scales_into_window():
    assert map_build_step_progress(0, 100, floor=70, ceil=90) == 70
    assert map_build_step_progress(50, 100, floor=70, ceil=90) == 80
    assert map_build_step_progress(100, 100, floor=70, ceil=90) == 90


def test_apply_build_step_progress_never_decreases():
    assert apply_build_step_progress(
        "[10/100] step",
        current_progress=85,
        floor=70,
        ceil=95,
    ) == (85, "[10/100]")
    assert apply_build_step_progress(
        "[90/100] step",
        current_progress=28,
        floor=28,
        ceil=92,
    ) == (86, "[90/100]")


def test_apply_build_step_progress_makefile_percent():
    # Asymptotic mapping: first-pass 100% only reaches mid-window so later
    # Makefile target restarts still have room (see BuildProgressTracker).
    assert apply_build_step_progress(
        "[ 46%] Built target ggml-cuda",
        current_progress=28,
        floor=28,
        ceil=92,
    ) == (45, "[46%]")
    assert apply_build_step_progress(
        "[100%] Built target llama-server",
        current_progress=45,
        floor=28,
        ceil=92,
    ) == (60, "[100%]")
    # Never decrease when cmake reprints an earlier percent.
    assert apply_build_step_progress(
        "[  7%] Built target ggml-cpu",
        current_progress=57,
        floor=28,
        ceil=92,
    ) == (57, "[7%]")


def test_build_progress_tracker_handles_makefile_target_restarts():
    tracker = BuildProgressTracker(floor=28, ceil=92)
    first_mid = tracker.apply_line("[ 50%] Built target audiocpp_cli")
    assert first_mid is not None
    assert first_mid[0] == 47
    assert first_mid[1] == "[50%]"
    first_done = tracker.apply_line("[100%] Built target audiocpp_cli")
    assert first_done == (60, "[100%]")
    # Second target restarts at 0% — overall bar must keep rising, not reset.
    second = tracker.apply_line("[  0%] Built target audiocpp_server")
    assert second == (60, "[0%]")
    done = tracker.apply_line("[100%] Built target audiocpp_server")
    assert done == (76, "[100%]")
    assert tracker.complete() == 92


def test_split_cmake_cli_warning_flags_moves_nvcc_flag():
    from backend.build_progress import (
        apply_relocated_cuda_warning_flags,
        is_cmake_warning_option,
        split_cmake_cli_warning_flags,
    )

    assert is_cmake_warning_option("-Wno-dev")
    assert is_cmake_warning_option("-Wdeprecated")
    assert is_cmake_warning_option("-WCMD_DEPRECATED")
    assert not is_cmake_warning_option("-Wno-deprecated-gpu-targets")
    kept, relocated = split_cmake_cli_warning_flags(
        [
            "cmake",
            "..",
            "-DGGML_CUDA=ON",
            "-Wno-deprecated-gpu-targets",
            "-Wno-dev",
        ]
    )
    assert kept == ["cmake", "..", "-DGGML_CUDA=ON", "-Wno-dev"]
    assert relocated == ["-Wno-deprecated-gpu-targets"]
    env = apply_relocated_cuda_warning_flags({}, relocated)
    assert "-Wno-deprecated-gpu-targets" in env["CMAKE_CUDA_FLAGS"]
    assert "-Wno-deprecated-gpu-targets" in env["CUDAFLAGS"]


def test_prefer_ninja_generator_appends_when_available(monkeypatch):
    from backend.build_progress import prefer_ninja_generator

    monkeypatch.setattr(
        "backend.build_progress.shutil.which",
        lambda name: "/usr/bin/ninja" if name == "ninja" else None,
    )
    assert prefer_ninja_generator(["cmake", "-S", "src", "-B", "build"]) == [
        "cmake",
        "-S",
        "src",
        "-B",
        "build",
        "-G",
        "Ninja",
    ]
    assert prefer_ninja_generator(["cmake", "-G", "Unix Makefiles", "-B", "build"]) == [
        "cmake",
        "-G",
        "Unix Makefiles",
        "-B",
        "build",
    ]


def test_cmake_build_stage_owns_most_of_the_bar():
    floor, ceil = cmake_stage_window("build")
    assert floor == 28
    assert ceil == 92
    assert (ceil - floor) >= 60
    assert cmake_stage_window("clone") == (2, 18)
    assert cmake_stage_start("configure") < floor
    assert cmake_stage_start("validate") >= ceil


def test_apply_cmake_stage_sets_shared_window():
    ctx = apply_cmake_stage({}, "build", base_message="Building llama.cpp")
    assert ctx["stage"] == "build"
    assert ctx["progress"] == 28
    assert ctx["progress_floor"] == 28
    assert ctx["progress_ceil"] == 92
    assert ctx["base_message"] == "Building llama.cpp"


def test_progress_from_install_log_prefers_build_ratio():
    progress, suffix = progress_from_install_log(
        "[50/100] Building CXX object",
        current_progress=10,
        log_count=3,
    )
    assert suffix == "[50/100]"
    assert progress == 56.0


def test_progress_from_install_log_accepts_makefile_percent():
    progress, suffix = progress_from_install_log(
        "[ 46%] Built target ggml-cuda",
        current_progress=10,
        log_count=3,
    )
    assert suffix == "[46%]"
    assert progress == 40.0


def test_progress_from_install_log_collecting_stays_in_resolve():
    progress, label = progress_from_install_log(
        "Collecting torch",
        current_progress=10,
        log_count=20,
    )
    assert label == "Resolving dependencies"
    assert 8.0 <= progress <= 20.0


def test_progress_from_install_log_unrecognized_creep_stays_low():
    progress, label = progress_from_install_log(
        "some compiler note",
        current_progress=0,
        log_count=40,
    )
    assert label == ""
    assert progress <= 8.0


def test_onecat_release_log_progression():
    """Progress follows the 1Cat-vLLM cu128 wheel install: upgrade, download, backtrack, install."""
    tracker = PipInstallProgressTracker()
    seen = []

    def feed(line: str):
        progress, message = tracker.observe(line)
        seen.append(progress)
        return progress, message

    progress, message = feed(
        "$ /app/data/1cat-vllm/venv/bin/python -m pip install --upgrade pip setuptools wheel"
    )
    assert message == "Upgrading pip"
    assert progress < 8

    feed("Downloading pip-26.2.1-py3-none-any.whl (1.8 MB)")
    progress, message = feed(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 12.5 MB/s eta 0:00:00"
    )
    assert "pip" in message
    assert progress < 10

    feed("Installing collected packages: setuptools, pip, packaging, wheel")
    progress, message = feed(
        "Successfully installed packaging-26.3 pip-26.2.1 setuptools-84.0.0 wheel-0.48.0"
    )
    assert message == "Pip tools ready"
    assert progress <= 10

    progress, message = feed(
        "$ /app/data/1cat-vllm/venv/bin/python -m pip install --prefer-binary "
        "--no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 "
        "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.1/"
        "1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl"
    )
    assert message == "Resolving dependencies"
    assert progress >= 8

    feed("Collecting 1cat-vllm==1.5.1")
    feed("Downloading 1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl (181.1 MB)")
    progress, message = feed(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.1/181.1 MB 67.5 MB/s  0:00:02"
    )
    assert "1cat-vllm" in message
    after_primary = progress

    feed(
        "Downloading torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl (916.9 MB)"
    )
    progress, message = feed(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 400.0/916.9 MB 105.2 MB/s  0:00:08"
    )
    assert "torch" in message
    assert "400.0/916.9 MB" in message
    assert "105.2 MB/s" in message
    partial = progress
    progress, message = feed(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 916.9/916.9 MB 105.2 MB/s  0:00:08"
    )
    assert progress > partial
    assert progress > after_primary
    after_torch = progress

    for name, size in (
        ("nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl", "706.8"),
        ("nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl", "594.3"),
        ("nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.whl", "193.1"),
        ("nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl", "267.5"),
        ("nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.whl", "288.2"),
        ("nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl", "287.2"),
        ("nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.whl", "322.3"),
        ("nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.whl", "139.1"),
    ):
        feed(f"Downloading {name} ({size} MB)")
        feed(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ {size}/{size} MB 100 MB/s  0:00:01")
    after_wave = tracker.progress
    assert after_torch < after_wave < 40

    progress, message = feed(
        "INFO: pip is looking at multiple versions of xgrammar to determine which "
        "version is compatible with other requirements. This could take a while."
    )
    assert message == "Resolving xgrammar versions"
    assert progress >= after_wave
    looking = progress

    progress, message = feed(
        "INFO: pip is still looking at multiple versions of cuda-python to determine "
        "which version is compatible with other requirements. This could take a while."
    )
    assert message == "Still resolving cuda-python versions"
    assert progress > looking

    progress, message = feed(
        "INFO: This is taking longer than usual. You might need to provide the "
        "dependency resolver with stricter constraints to reduce runtime."
    )
    assert message == "Dependency resolution is taking longer than usual"
    assert progress > looking
    stalled = progress
    progress, message = feed(
        "Downloading cuda_python-12.9.4-py3-none-any.whl.metadata (4.7 kB)"
    )
    assert message == "Dependency resolution is taking longer than usual"
    assert progress >= stalled

    feed("Downloading flashinfer_cubin-0.6.11.post2-py3-none-any.whl (360.9 MB)")
    progress, message = feed(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 360.9/360.9 MB 81.7 MB/s  0:00:04"
    )
    assert "flashinfer-cubin" in message
    assert progress >= stalled

    feed("Installing collected packages: z3-solver, torch, 1cat-vllm")
    assert tracker.message == "Installing packages"
    assert tracker.phase == "unpack"
    assert tracker.progress < 50

    progress, message = feed("Successfully uninstalled setuptools-84.0.0")
    assert message == "Installing packages"
    assert tracker.phase == "unpack"
    assert progress < 55
    unpack_start = progress
    tracker._unpack_started = time.monotonic() - 180
    progress, message = tracker.note_unpack_wait()
    seen.append(progress)
    assert message == "Installing packages"
    assert progress > unpack_start + 15
    assert progress < 94

    progress, message = feed(
        "Successfully installed 1cat-vllm-1.5.1 torch-2.10.0+cu128"
    )
    assert message == "Packages installed"
    assert 94 <= progress < 100
    assert seen == sorted(seen)


async def test_installer_broadcast_uses_pip_phases(tmp_path):
    from backend.engines.vllm import OneCatVllmManager
    from backend.operations.progress import get_progress_manager

    manager = OneCatVllmManager(
        log_path=str(tmp_path / "onecat.log"),
        state_path=str(tmp_path / "onecat_state.json"),
        base_dir=str(tmp_path / "1cat-vllm"),
    )
    task_id = get_progress_manager().create_task("install", "Install 1Cat-vLLM")
    manager._progress_task_id = task_id

    await manager._broadcast_log_line(
        "$ /app/venv/bin/python -m pip install --upgrade pip setuptools wheel"
    )
    task = get_progress_manager().get_task(task_id)
    assert task["message"] == "Upgrading pip"
    assert task["metadata"]["stage"] == "bootstrap"
    assert task["progress"] < 8

    await manager._broadcast_log_line(
        "Successfully installed packaging-26.3 pip-26.2.1 setuptools-84.0.0 wheel-0.48.0"
    )
    await manager._broadcast_log_line(
        "$ /app/venv/bin/python -m pip install --prefer-binary 1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl"
    )
    await manager._broadcast_log_line(
        "Downloading torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl (916.9 MB)"
    )
    await manager._broadcast_log_line(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 400.0/916.9 MB 105.2 MB/s  0:00:08"
    )
    task = get_progress_manager().get_task(task_id)
    assert task["message"].startswith("Downloading torch")
    assert "400.0/916.9 MB" in task["message"]
    assert task["metadata"]["stage"] == "download"
    assert task["progress"] >= 8

    await manager._broadcast_log_line(
        "INFO: pip is looking at multiple versions of cuda-python to determine which version is compatible."
    )
    await manager._broadcast_log_line(
        "INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints."
    )
    task = get_progress_manager().get_task(task_id)
    assert task["message"] == "Dependency resolution is taking longer than usual"
    assert task["metadata"]["stage"] == "backtrack"

    await manager._broadcast_log_line(
        "Installing collected packages: torch, 1cat-vllm"
    )
    await manager._broadcast_log_line("Successfully uninstalled setuptools-84.0.0")
    task = get_progress_manager().get_task(task_id)
    assert task["message"] == "Installing packages"
    assert task["metadata"]["stage"] == "unpack"
    assert task["progress"] < 55
    await manager._broadcast_log_line("Successfully installed 1cat-vllm-1.5.1 torch-2.10.0+cu128")
    task = get_progress_manager().get_task(task_id)
    assert task["message"] == "Packages installed"
    assert 94 <= task["progress"] < 100


def test_sample_install_log_replay_is_monotonic():
    """Replay key markers from /home/vlapy/llama-install.log across stages."""
    tracker = BuildProgressTracker(floor=2, ceil=18)
    last = 2
    for line in (
        "Receiving objects:  46% (48213/104810), 153.47 MiB | 23.38 MiB/s",
        "Receiving objects: 100% (104810/104810), 411.34 MiB | 23.86 MiB/s, done.",
        "Resolving deltas: 100% (73399/73399), done.",
        "Updating files: 100% (3258/3258), done.",
    ):
        step = tracker.apply_line(line)
        assert step is not None
        assert step[0] >= last
        last = step[0]
    tracker.complete()
    assert tracker.progress == 18

    tracker.set_window(28, 92, progress=28)
    last = 28
    for line in (
        "[1/486] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/hbm.cpp.o",
        "[243/486] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq.cu.o",
        "[400/486] Linking CXX executable tools/ui/llama-ui-embed",
        "[486/486] Linking CXX executable bin/llama-server",
    ):
        step = tracker.apply_line(line)
        assert step is not None
        assert step[0] >= last
        last = step[0]
    assert tracker.complete() == 92
