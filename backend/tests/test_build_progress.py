"""Tests for universal [x/y] and [N%] build progress parsing."""

from backend.build_progress import (
    apply_build_step_progress,
    apply_cmake_stage,
    cmake_stage_start,
    cmake_stage_window,
    map_build_step_progress,
    parse_build_percent,
    parse_build_progress_ratio,
    parse_build_step_ratio,
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
    from backend.build_progress import BuildProgressTracker

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


def test_progress_from_install_log_creep_stays_pre_build():
    progress, suffix = progress_from_install_log(
        "Collecting torch",
        current_progress=10,
        log_count=20,
    )
    assert suffix == ""
    assert progress <= 28.0
