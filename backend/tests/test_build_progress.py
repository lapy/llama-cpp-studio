"""Tests for compiler [x/y] build progress parsing."""

from backend.build_progress import (
    apply_build_step_progress,
    apply_cmake_stage,
    cmake_stage_start,
    cmake_stage_window,
    map_build_step_progress,
    parse_build_step_ratio,
    progress_from_install_log,
)


def test_parse_build_step_ratio_ninja_style():
    assert parse_build_step_ratio("[123/456] Building CXX object foo.cpp.o") == (123, 456)
    assert parse_build_step_ratio("[ 12/ 90 ] Linking CXX executable server") == (12, 90)
    assert parse_build_step_ratio("no counter here") is None


def test_parse_build_step_ratio_clamps_overshoot():
    assert parse_build_step_ratio("[10/5] weird") == (5, 5)


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


def test_progress_from_install_log_creep_stays_pre_build():
    progress, suffix = progress_from_install_log(
        "Collecting torch",
        current_progress=10,
        log_count=20,
    )
    assert suffix == ""
    assert progress <= 28.0
