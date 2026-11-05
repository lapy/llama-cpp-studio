"""Condition evaluator + shadow install-plan executor tests (Phase 3 seed).

Executor semantics are pinned against the REAL reference plans
(lmdeploy release, sglang_v100 source) with fake primitive runners, so the
Phase 3 cutover inherits a proven execution contract.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace

import pytest

from backend.engines.definitions import (
    REFERENCE_DEFINITIONS_DIR,
    PrimitiveExecutionError,
    evaluate_condition,
    execute_plan,
    load_engine_definitions,
    materialize_plan,
)
from backend.engines.definitions.executor import (
    EXECUTION_CANCELLED,
    EXECUTION_COMPLETED,
    EXECUTION_FAILED,
    STEP_CANCELLED,
    STEP_CLEANED,
    STEP_CLEANUP_FAILED,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_SKIPPED,
    StepOutcome,
)
from backend.engines.definitions.plan import InstallPlan
from backend.engines.definitions.schema import validate_definition


@pytest.fixture(scope="module")
def reference():
    return load_engine_definitions(REFERENCE_DEFINITIONS_DIR)


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------


def test_exists_flat_and_nested_and_absent():
    assert evaluate_condition({"exists": {"field": "a.b"}}, {"a": {"b": "x"}})
    assert evaluate_condition({"exists": {"field": "a.b"}}, {"a.b": "x"})
    assert not evaluate_condition({"exists": {"field": "a.b"}}, {"a": {"b": ""}})
    assert not evaluate_condition({"exists": {"field": "ghost"}}, {})


def test_equals_typed_and_string_equivalence():
    assert evaluate_condition({"equals": {"field": "f", "value": "12.8"}}, {"f": "12.8"})
    assert evaluate_condition({"equals": {"field": "f", "value": "12.8"}}, {"f": 12.8})
    assert not evaluate_condition({"equals": {"field": "f", "value": "1"}}, {"f": True})
    assert not evaluate_condition({"equals": {"field": "f", "value": True}}, {"f": 1})
    assert evaluate_condition({"equals": {"field": "f", "value": True}}, {"f": True})


def test_version_range_inclusive_bounds():
    node = {"version_range": {"field": "v", "min": "12.8", "max": "12.8"}}
    assert evaluate_condition(node, {"v": "12.8"})
    assert not evaluate_condition(node, {"v": "12.4"})
    assert not evaluate_condition(node, {"v": "13.0"})
    assert not evaluate_condition(node, {"v": None})
    assert not evaluate_condition(node, {"v": "garbage"})
    assert evaluate_condition({"version_range": {"field": "v", "min": "2.0"}}, {"v": "2.1"})


def test_capability_contains_over_shapes():
    assert evaluate_condition(
        {"capability_contains": {"field": "c", "value": "g++-12"}}, {"c": ["gcc-12", "g++-12"]}
    )
    assert evaluate_condition(
        {"capability_contains": {"field": "c", "value": "sm_70"}}, {"c": "sm_75,sm_70"}
    )
    assert evaluate_condition(
        {"capability_contains": {"field": "c", "value": "x"}}, {"c": {"k": "x"}}
    )
    assert not evaluate_condition({"capability_contains": {"field": "c", "value": "x"}}, {})


def test_all_any_nesting_and_produced_output_visibility():
    node = {
        "all": [
            {"exists": {"field": "venv"}},
            {"any": [{"equals": {"field": "runtime", "value": "llm"}}, {"exists": {"field": "x"}}]},
        ]
    }
    assert evaluate_condition(node, {"venv": "/venv", "runtime": "llm"})
    assert not evaluate_condition(node, {"runtime": "llm"})
    assert not evaluate_condition(node, {"venv": "/venv", "runtime": "diffusion"})


def test_evaluator_is_total_on_valid_nodes():
    # Missing data must make steps skip, never crash an install.
    assert not evaluate_condition({"version_range": {"field": "a.b.c", "min": "1"}}, {})
    assert not evaluate_condition({"capability_contains": {"field": "a", "value": "z"}}, {"a": 5})


# ---------------------------------------------------------------------------
# execute_plan over the real reference plans
# ---------------------------------------------------------------------------


def _auto_runner(calls):
    def runner(primitive, step, requirements, context):
        calls.append((step.id, primitive, dict(requirements)))
        return {name: f"{name}-from-{step.id}" for name in step.produces}

    return runner


@pytest.mark.asyncio
async def test_lmdeploy_release_plan_executes_in_order_with_output_flow(reference):
    plan = materialize_plan(reference.get("lmdeploy"), "release")
    calls = []
    execution = await execute_plan(
        plan,
        run_primitive=_auto_runner(calls),
        context={"input.engine_root": "/data/engines/lmdeploy"},
    )
    assert execution.status == EXECUTION_COMPLETED
    assert [c[0] for c in calls] == [
        "resolve_version",
        "create_venv",
        "install_package",
        "capture_metadata",
        "publish",
    ]
    install_call = next(c for c in calls if c[0] == "install_package")
    assert install_call[2] == {
        "venv": "venv-from-create_venv",
        "version": "version-from-resolve_version",
    }
    assert execution.outputs["version"] == "version-from-resolve_version"


def _v100_runner(calls, *, healthy_host: bool):
    """Fake primitives where host.inspect yields the real host facts that
    recipe conditions evaluate against (managed CUDA, compilers, git)."""

    def runner(primitive, step, requirements, context):
        calls.append((step.id, primitive, dict(requirements)))
        outputs = {name: f"{name}-from-{step.id}" for name in step.produces}
        if step.id == "inspect_host":
            outputs["host"] = (
                {"managed_cuda_version": "12.8", "compilers": ["g++-12"], "vcs": ["git"]}
                if healthy_host
                else {"managed_cuda_version": "11.0", "compilers": [], "vcs": []}
            )
        return outputs

    return runner


@pytest.mark.asyncio
async def test_sglang_v100_source_plan_full_walkthrough(reference):
    plan = materialize_plan(reference.get("sglang_v100"), "source")
    calls = []
    execution = await execute_plan(
        plan,
        run_primitive=_v100_runner(calls, healthy_host=True),
        context={"input.engine_root": "/data/engines/sglang_v100"},
    )
    assert execution.status == EXECUTION_COMPLETED
    assert [c[0] for c in calls] == [step.id for step in plan.steps]
    assert all(o.status == STEP_COMPLETED for o in execution.outcomes)
    # The managed-CUDA gate from the reference recipe evaluated true.
    assert "cuda" in execution.outputs


@pytest.mark.asyncio
async def test_conditions_skip_steps_and_dependents_report_missing_output(reference):
    definition = reference.get("sglang_v100")
    recipes = dict(definition.recipes)
    steps = [copy.deepcopy(step) for step in recipes["source"]]
    clone = next(step for step in steps if step["id"] == "clone_repository")
    clone["when"] = {"capability_contains": {"field": "host.vcs", "value": "git"}}
    broken = dict(definition.recipes, source=tuple(steps))
    mutated = replace(definition, recipes=tuple(broken.items()))
    plan = materialize_plan(mutated, "source")

    calls = []
    execution = await execute_plan(
        plan,
        run_primitive=_v100_runner(calls, healthy_host=False),
        context={"input.engine_root": "/x"},
    )
    assert execution.status == EXECUTION_FAILED
    outcomes = {o.id: o.status for o in execution.outcomes}
    assert outcomes["require_cuda_12_8"] == STEP_SKIPPED  # host says CUDA 11.0
    assert outcomes["require_gcc12"] == STEP_SKIPPED
    assert outcomes["clone_repository"] == STEP_SKIPPED  # no git capability
    assert outcomes["checkout_branch"] == STEP_FAILED
    assert "never produced" in execution.failed_step.error

    # With a healthy host the same plan completes.
    execution = await execute_plan(
        plan,
        run_primitive=_v100_runner(calls, healthy_host=True),
        context={"input.engine_root": "/x"},
    )
    assert execution.status == EXECUTION_COMPLETED


@pytest.mark.asyncio
async def test_primitive_failure_aborts_and_reports_exit_code(reference):
    plan = materialize_plan(reference.get("lmdeploy"), "release")

    def runner(primitive, step, requirements, context):
        if step.id == "install_package":
            raise PrimitiveExecutionError("pip exited non-zero", exit_code=2)
        return {name: "x" for name in step.produces}

    execution = await execute_plan(plan, run_primitive=runner, context={"input.engine_root": "/x"})
    assert execution.status == EXECUTION_FAILED
    assert execution.failed_step.id == "install_package"
    assert execution.failed_step.exit_code == 2
    assert [o.id for o in execution.outcomes] == [
        "resolve_version",
        "create_venv",
        "install_package",
    ]


@pytest.mark.asyncio
async def test_missing_declared_output_is_a_failure(reference):
    plan = materialize_plan(reference.get("lmdeploy"), "release")

    def runner(primitive, step, requirements, context):
        return {}  # lies about producing anything

    execution = await execute_plan(plan, run_primitive=runner, context={"input.engine_root": "/x"})
    assert execution.status == EXECUTION_FAILED
    assert "did not produce declared outputs" in execution.failed_step.error


@pytest.mark.asyncio
async def test_cancellation_stops_before_next_step(reference):
    plan = materialize_plan(reference.get("sglang_v100"), "source")
    calls = []
    polls = {"n": 0}

    def cancel():
        polls["n"] += 1
        return polls["n"] > 2  # allow two steps, then the user cancels

    execution = await execute_plan(
        plan,
        run_primitive=_v100_runner(calls, healthy_host=True),
        context={"input.engine_root": "/x"},
        cancel=cancel,
    )
    assert execution.status == EXECUTION_CANCELLED
    assert len(calls) == 2
    assert not any(o.status == STEP_FAILED for o in execution.outcomes)


@pytest.mark.asyncio
async def test_progress_hook_sees_every_outcome(reference):
    plan = materialize_plan(reference.get("lmdeploy"), "release")
    seen = []
    await execute_plan(
        plan,
        run_primitive=_auto_runner([]),
        context={"input.engine_root": "/x"},
        on_step=lambda outcome: seen.append((outcome.id, outcome.status)),
    )
    assert seen == [(step.id, STEP_COMPLETED) for step in plan.steps]


@pytest.mark.asyncio
async def test_raw_run_step_passes_argv_to_runner():
    definition = validate_definition(
        {
            "schema_version": 2,
            "id": "x",
            "label": "X",
            "repository_source": "X",
            "install_kind": "prebuilt",
            "runtime_kind": "llama_server",
            "scanner_kind": "llama_help",
            "active_path_fields": ["binary"],
            "artifact_formats": ["gguf"],
            "package_kinds": ["binary"],
            "tasks": ["text-generation"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "recipes": {"release": [{"id": "s", "run": ["git", "--version"], "produces": ["v"]}]},
        },
        source="x.yaml",
    )
    plan = materialize_plan(definition, "release")
    captured = {}

    def runner(primitive, step, requirements, context):
        captured["primitive"] = primitive
        captured["run"] = step.run
        return {"v": "git version 2.x"}

    execution = await execute_plan(plan, run_primitive=runner)
    assert execution.status == EXECUTION_COMPLETED
    assert captured == {"primitive": None, "run": ("git", "--version")}


@pytest.mark.asyncio
async def test_empty_plan_completes_trivially():
    plan = InstallPlan(engine_id="x", recipe="r", steps=())
    execution = await execute_plan(plan, run_primitive=lambda *a: {})
    assert execution.status == EXECUTION_COMPLETED
    assert execution.outcomes == ()
    assert execution.outputs == {}


# ---------------------------------------------------------------------------
# finally steps: the recipe-level teardown the native build cutovers need
# ---------------------------------------------------------------------------


def _teardown_definition(steps):
    return validate_definition(
        {
            "schema_version": 2,
            "id": "x",
            "label": "X",
            "repository_source": "X",
            "install_kind": "cmake",
            "runtime_kind": "llama_server",
            "scanner_kind": "llama_help",
            "active_path_fields": ["binary"],
            "artifact_formats": ["gguf"],
            "package_kinds": ["binary"],
            "tasks": ["text-generation"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "recipes": {"build": list(steps)},
        },
        source="x.yaml",
    )


def _teardown_plan(*steps):
    return materialize_plan(_teardown_definition(steps), "build")


_BODY = [
    {"id": "build", "uses": "env.compose", "produces": ["artifacts"]},
    {"id": "publish", "uses": "publish.install", "requires": ["artifacts"]},
    {"id": "wipe", "uses": "fs.remove_tree", "requires": ["input.root"], "cleanup": True},
]


def _runner(fail_on=()):
    """Auto-succeeding runner; steps named in ``fail_on`` raise instead."""
    calls = []

    def runner(primitive, step, requirements, context):
        calls.append(step.id)
        if step.id in fail_on:
            raise PrimitiveExecutionError(f"{step.id} exploded")
        return {name: f"{name}-from-{step.id}" for name in step.produces}

    return runner, calls


@pytest.mark.asyncio
async def test_cleanup_step_runs_after_body_failure():
    plan = _teardown_plan(*_BODY)
    runner, calls = _runner(fail_on={"publish"})
    execution = await execute_plan(plan, run_primitive=runner, context={"input.root": "/x"})
    assert execution.status == EXECUTION_FAILED
    assert calls == ["build", "publish", "wipe"]  # 'publish' failed, 'wipe' still ran
    assert execution.failed_step.id == "publish"
    assert [(o.id, o.status) for o in execution.outcomes] == [
        ("build", STEP_COMPLETED),
        ("publish", STEP_FAILED),
        ("wipe", STEP_CLEANED),
    ]


@pytest.mark.asyncio
async def test_cleanup_step_runs_after_cancellation():
    plan = _teardown_plan(*_BODY)
    runner, calls = _runner()
    cancelled = {"now": False}

    def cancel():
        return cancelled["now"]

    def runner_then_cancel(primitive, step, requirements, context):
        out = runner(primitive, step, requirements, context)
        if step.id == "build":
            cancelled["now"] = True
        return out

    execution = await execute_plan(
        plan, run_primitive=runner_then_cancel, context={"input.root": "/x"}, cancel=cancel
    )
    assert execution.status == EXECUTION_CANCELLED
    assert calls == ["build", "wipe"]  # publish never started, cleanup did
    assert execution.cancelled_at == "publish"
    assert not [o for o in execution.outcomes if o.status == STEP_FAILED]
    assert [(o.id, o.status) for o in execution.outcomes] == [
        ("build", STEP_COMPLETED),
        ("publish", STEP_CANCELLED),
        ("wipe", STEP_CLEANED),
    ]


@pytest.mark.asyncio
async def test_cleanup_cannot_mask_the_primary_failure():
    plan = _teardown_plan(*_BODY)
    runner, calls = _runner(fail_on={"publish", "wipe"})
    execution = await execute_plan(plan, run_primitive=runner, context={"input.root": "/x"})
    assert execution.status == EXECUTION_FAILED
    assert execution.failed_step.id == "publish"
    assert [o.id for o in execution.cleanup_failures] == ["wipe"]
    assert [(o.id, o.status) for o in execution.outcomes][-1] == ("wipe", STEP_CLEANUP_FAILED)
    assert calls == ["build", "publish", "wipe"]


@pytest.mark.asyncio
async def test_cleanup_failure_alone_fails_the_execution():
    plan = _teardown_plan(*_BODY)
    runner, calls = _runner(fail_on={"wipe"})
    execution = await execute_plan(plan, run_primitive=runner, context={"input.root": "/x"})
    # No body failure, so the finally's own error is the only thing to report.
    assert execution.status == EXECUTION_FAILED
    assert execution.failed_step.id == "wipe"
    assert execution.failed_step.error == "wipe exploded"
    assert calls == ["build", "publish", "wipe"]


@pytest.mark.asyncio
async def test_cleanup_never_outranks_a_cancellation():
    plan = _teardown_plan(*_BODY)
    runner, calls = _runner(fail_on={"wipe"})
    execution = await execute_plan(
        plan,
        run_primitive=runner,
        context={"input.root": "/x"},
        cancel=lambda: True,
    )
    assert execution.status == EXECUTION_CANCELLED
    assert execution.failed_step is None  # a caller must still see 'cancelled'
    assert execution.cancelled_at == "build"
    assert [o.id for o in execution.cleanup_failures] == ["wipe"]


@pytest.mark.asyncio
async def test_cleanup_steps_missing_inputs_are_skipped_not_failed():
    steps = [
        {"id": "fetch", "uses": "env.compose", "produces": ["artifacts"]},
        {"id": "wipe", "uses": "fs.remove_tree", "requires": ["artifacts"], "cleanup": True},
    ]
    plan = _teardown_plan(*steps)
    runner, calls = _runner(fail_on={"fetch"})
    execution = await execute_plan(plan, run_primitive=runner)
    assert execution.status == EXECUTION_FAILED
    assert execution.failed_step.id == "fetch"
    # artifacts were never produced: the finally skips instead of failing.
    assert [(o.id, o.status) for o in execution.outcomes] == [
        ("fetch", STEP_FAILED),
        ("wipe", STEP_SKIPPED),
    ]
    assert "never produced" in execution.outcomes[-1].error


@pytest.mark.asyncio
async def test_non_cleanup_steps_after_a_failure_stay_skipped():
    steps = [
        {"id": "boom", "uses": "exec.run_logged"},
        {"id": "later", "uses": "publish.install"},
        {"id": "wipe", "uses": "fs.remove_tree", "cleanup": True},
    ]
    plan = _teardown_plan(*steps)
    runner, calls = _runner(fail_on={"boom"})
    execution = await execute_plan(plan, run_primitive=runner)
    assert calls == ["boom", "wipe"]  # 'later' never runs
    assert [o.id for o in execution.outcomes] == ["boom", "wipe"]


@pytest.mark.asyncio
async def test_cleanup_step_respects_its_condition():
    steps = [
        {"id": "boom", "uses": "exec.run_logged"},
        {
            "id": "wipe",
            "uses": "fs.remove_tree",
            "cleanup": True,
            "when": {"equals": {"field": "input.root", "value": "/never"}},
        },
    ]
    plan = _teardown_plan(*steps)
    runner, calls = _runner(fail_on={"boom"})
    execution = await execute_plan(plan, run_primitive=runner, context={"input.root": "/x"})
    assert calls == ["boom"]
    assert [(o.id, o.status) for o in execution.outcomes] == [
        ("boom", STEP_FAILED),
        ("wipe", STEP_SKIPPED),
    ]


@pytest.mark.asyncio
async def test_cleanup_flag_survives_materialization():
    plan = _teardown_plan(*_BODY)
    assert [step.cleanup for step in plan.steps] == [False, False, True]


@pytest.mark.parametrize("value", ["true", 1, None, {}])
def test_cleanup_flag_must_be_a_boolean(value):
    with pytest.raises(Exception, match="must be a boolean"):
        _teardown_definition([{"id": "wipe", "uses": "fs.remove_tree", "cleanup": value}])


@pytest.mark.asyncio
async def test_completed_execution_is_untouched_by_cleanup_steps():
    steps = [
        {"id": "build", "uses": "env.compose"},
        {"id": "wipe", "uses": "fs.remove_tree", "cleanup": True},
    ]
    plan = _teardown_plan(*steps)
    runner, calls = _runner()
    seen = []
    execution = await execute_plan(
        plan, run_primitive=runner, context={"input.root": "/x"}, on_step=seen.append
    )
    assert execution.status == EXECUTION_COMPLETED
    assert calls == ["build", "wipe"]
    assert seen == [
        StepOutcome(id="build", status=STEP_COMPLETED, outputs={}),
        StepOutcome(id="wipe", status=STEP_CLEANED, outputs={}),
    ]


# ---------------------------------------------------------------------------
# Cleanup guarantee (2026-11-02 review finding #3): ANY handler failure must
# reach the teardown pass — before this tranche only PrimitiveExecutionError
# did; a plain RuntimeError propagated out of execute_plan with zero cleanup
# (recorded RED-first: the exception itself was the old behavior).
# Unexpected errors INSIDE the teardown pass must not abort the remaining
# cleanup steps either.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_handler_error_still_runs_teardown_then_propagates():
    plan = _teardown_plan(*_BODY)
    calls = []
    seen = []

    def runner(primitive, step, requirements, context):
        calls.append(step.id)
        if step.id == "publish":
            raise RuntimeError("boom")
        return {name: f"{name}-from-{step.id}" for name in step.produces}

    with pytest.raises(RuntimeError, match="boom"):  # propagation unchanged for parity callers
        await execute_plan(
            plan,
            run_primitive=runner,
            context={"input.root": "/x"},
            on_step=seen.append,
        )
    assert calls == ["build", "publish", "wipe"]  # teardown ran despite the foreign error
    assert [(o.id, o.status) for o in seen] == [
        ("build", STEP_COMPLETED),
        ("publish", STEP_FAILED),
        ("wipe", STEP_CLEANED),
    ]
    assert next(o for o in seen if o.id == "publish").error == "RuntimeError: boom"


@pytest.mark.asyncio
async def test_unexpected_teardown_error_propagates_after_remaining_cleanups_run():
    steps = [
        {"id": "build", "uses": "env.compose", "produces": ["artifacts"]},
        {
            "id": "first_wipe",
            "uses": "fs.remove_tree",
            "requires": ["input.root"],
            "cleanup": True,
        },
        {
            "id": "second_wipe",
            "uses": "fs.remove_tree",
            "requires": ["input.root"],
            "cleanup": True,
        },
    ]
    plan = _teardown_plan(*steps)
    calls = []
    seen = []

    def runner(primitive, step, requirements, context):
        calls.append(step.id)
        if step.id == "build":
            raise PrimitiveExecutionError("build exploded")
        if step.id == "first_wipe":
            raise TypeError("cleanup is broken")
        return {}

    with pytest.raises(TypeError, match="cleanup is broken"):
        await execute_plan(
            plan,
            run_primitive=runner,
            context={"input.root": "/x"},
            on_step=seen.append,
        )
    assert calls == ["build", "first_wipe", "second_wipe"]
    assert [(o.id, o.status) for o in seen] == [
        ("build", STEP_FAILED),
        ("first_wipe", STEP_CLEANUP_FAILED),
        ("second_wipe", STEP_CLEANED),
    ]
    assert next(o for o in seen if o.id == "first_wipe").error == "TypeError: cleanup is broken"


@pytest.mark.asyncio
async def test_task_cancellation_still_runs_the_cleanup_pass():
    """P2: CancelledError must not bypass the recipe's cleanup steps.

    The predicate-based cancellation path (`cancel=`) always ran them; actual
    asyncio task cancellation used to skip them because `Exception` does not
    catch CancelledError.
    """
    started = asyncio.Event()
    cleaned: list = []

    async def runner(primitive, step, requirements, context):
        if step.id == "build":
            started.set()
            await asyncio.Event().wait()  # cancelled here
            return {name: "x" for name in step.produces}
        if step.id == "wipe":
            cleaned.append(step.id)
            return {}
        return {name: "x" for name in step.produces}

    task = asyncio.create_task(
        execute_plan(
            _teardown_plan(*_BODY),
            run_primitive=runner,
            context={"input.root": "/tmp/x"},
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["wipe"]
