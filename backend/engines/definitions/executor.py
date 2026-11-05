"""Install-plan executor (Phase 3) — runs definition-driven installs.

Executes an :class:`~backend.engines.definitions.plan.InstallPlan`:
deterministic step order, typed conditions gate each step, step outputs flow
into later steps' requirements, awaitable primitive handlers, and progress
hooks for operation reporting. Two teardown primitives support the native
engines queued behind it:

- ``cancel=`` is polled before every body step (never between cleanup
  steps); hitting it records a ``cancelled`` outcome at that step.
- steps flagged ``cleanup: true`` are the recipe's finally: after the body
  failed or was cancelled, non-cleanup steps are skipped and the remaining
  cleanup steps still run, in declaration order. Missing inputs there are
  ``skipped``, failures are ``cleanup-failed`` — they never overwrite the
  primary failure, and never outrank a cancellation.

Operation-level cancellation (task/process kill) still belongs to the
supervisor; this predicate is the engine's between-stages check.
First consumers of the teardown keys arrive with the native-engine cutovers.
Every engine on the cutover allow-list in
``backend/tests/test_engine_definitions_equivalence.py`` is a definition-driven
runtime reader.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .plan import InstallPlan, PlannedStep
from .schema import evaluate_condition

STEP_COMPLETED = "completed"
STEP_SKIPPED = "skipped"
STEP_FAILED = "failed"
STEP_CANCELLED = "cancelled"
STEP_CLEANED = "cleaned"
STEP_CLEANUP_FAILED = "cleanup-failed"

EXECUTION_COMPLETED = "completed"
EXECUTION_FAILED = "failed"
EXECUTION_CANCELLED = "cancelled"


class PrimitiveExecutionError(Exception):
    """A primitive failed; the message reaches the operation log verbatim."""

    def __init__(self, message: str, *, exit_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class StepOutcome:
    id: str
    status: str
    outputs: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class PlanExecution:
    status: str
    outcomes: Tuple[StepOutcome, ...]
    outputs: Mapping[str, Any]

    @property
    def failed_step(self) -> Optional[StepOutcome]:
        """The failure a caller must report: the body failure wins; a
        cleanup failure only surfaces when nothing else failed (status stays
        failed), and never during a cancellation."""
        for outcome in self.outcomes:
            if outcome.status == STEP_FAILED:
                return outcome
        if self.status == EXECUTION_FAILED:
            for outcome in self.outcomes:
                if outcome.status == STEP_CLEANUP_FAILED:
                    return outcome
        return None

    @property
    def cleanup_failures(self) -> Tuple[StepOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == STEP_CLEANUP_FAILED)

    @property
    def cancelled_at(self) -> Optional[str]:
        for outcome in self.outcomes:
            if outcome.status == STEP_CANCELLED:
                return outcome.id
        return None


PrimitiveRunner = Callable[
    [Optional[str], PlannedStep, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]
"""(primitive, step, requirement_values, context) -> step outputs."""


async def execute_plan(
    plan: InstallPlan,
    *,
    run_primitive: PrimitiveRunner,
    context: Mapping[str, Any] = {},
    cancel: Optional[Callable[[], bool]] = None,
    on_step: Optional[Callable[[StepOutcome], None]] = None,
) -> PlanExecution:
    produced: Dict[str, Any] = {}
    outcomes: list = []
    status = EXECUTION_COMPLETED

    def finish(final_status: str) -> PlanExecution:
        return PlanExecution(status=final_status, outcomes=tuple(outcomes), outputs=dict(produced))

    def emit(outcome: StepOutcome) -> None:
        outcomes.append(outcome)
        if on_step is not None:
            on_step(outcome)

    def wire(step: PlannedStep) -> Tuple[Dict[str, Any], Optional[str]]:
        values: Dict[str, Any] = {}
        for requirement in step.requires:
            if requirement.startswith("input."):
                values[requirement] = context.get(requirement)
                continue
            if requirement not in produced:
                return values, requirement
            values[requirement] = produced[requirement]
        return values, None

    async def invoke(step: PlannedStep, values: Mapping[str, Any]):
        outputs = run_primitive(step.primitive, step, values, {**dict(context), **produced})
        if inspect.isawaitable(outputs):
            outputs = await outputs
        return dict(outputs or {})

    teardown_from: Optional[int] = None  # index into plan.steps for the finally pass
    foreign_error: Optional[BaseException] = None  # non-primitive failure, re-raised after cleanup
    cancelled_error: Optional[BaseException] = None  # task cancellation, re-raised after cleanup
    for index, step in enumerate(plan.steps):
        if cancel is not None and cancel():
            emit(StepOutcome(id=step.id, status=STEP_CANCELLED))
            status = EXECUTION_CANCELLED
            teardown_from = index
            break

        evaluation_context = {**dict(context), **produced}
        if step.when is not None and not evaluate_condition(step.when, evaluation_context):
            emit(StepOutcome(id=step.id, status=STEP_SKIPPED))
            continue

        requirement_values, missing = wire(step)
        failed: Optional[StepOutcome] = None
        if missing is not None:
            failed = StepOutcome(
                id=step.id,
                status=STEP_FAILED,
                error=f"required output {missing!r} was never produced",
            )
        else:
            try:
                outputs = await invoke(step, requirement_values)
            except PrimitiveExecutionError as error:
                failed = StepOutcome(
                    id=step.id,
                    status=STEP_FAILED,
                    error=str(error),
                    exit_code=error.exit_code,
                )
            except asyncio.CancelledError as error:
                # Task cancellation is a BaseException in 3.8+: the recipe's
                # cleanup pass MUST still run (temporary build resources), then
                # the cancellation propagates unchanged.
                cancelled_error = error
                status = EXECUTION_CANCELLED
                teardown_from = index
                break
            except Exception as error:  # noqa: BLE001
                # The cleanup guarantee (2026-11-02 review finding #3): ANY
                # handler failure — not just PrimitiveExecutionError — runs
                # the teardown pass below. Propagation is UNCHANGED: the
                # original exception is re-raised verbatim after cleanup so
                # legacy-parity callers keep owning its wrapping.
                foreign_error = error
                failed = StepOutcome(
                    id=step.id,
                    status=STEP_FAILED,
                    error=f"{type(error).__name__}: {error}",
                )
            else:
                absent_produces = [name for name in step.produces if name not in outputs]
                if absent_produces:
                    failed = StepOutcome(
                        id=step.id,
                        status=STEP_FAILED,
                        error=f"step did not produce declared outputs {absent_produces}",
                    )
                else:
                    produced.update(outputs)
                    emit(
                        StepOutcome(
                            id=step.id,
                            status=STEP_CLEANED if step.cleanup else STEP_COMPLETED,
                            outputs=outputs,
                        )
                    )
        if failed is not None:
            if step.cleanup:
                failed = StepOutcome(
                    id=failed.id,
                    status=STEP_CLEANUP_FAILED,
                    error=failed.error,
                    exit_code=failed.exit_code,
                )
            emit(failed)
            status = EXECUTION_FAILED
            teardown_from = index + 1
            break

    if teardown_from is not None:
        # The recipe's finally: cleanup steps run in declaration order after
        # a body failure or cancellation; nothing else may execute, and this
        # pass is not cancellable (it IS the cleanup).
        for step in plan.steps[teardown_from:]:
            if not step.cleanup:
                continue
            evaluation_context = {**dict(context), **produced}
            if step.when is not None and not evaluate_condition(step.when, evaluation_context):
                emit(StepOutcome(id=step.id, status=STEP_SKIPPED))
                continue
            requirement_values, missing = wire(step)
            if missing is not None:
                emit(
                    StepOutcome(
                        id=step.id,
                        status=STEP_SKIPPED,
                        error=f"required output {missing!r} was never produced",
                    )
                )
                continue
            try:
                # Shielded: a cancellation arriving during teardown must not
                # strand the remaining cleanup steps; it is re-raised below.
                outputs = await asyncio.shield(invoke(step, requirement_values))
            except asyncio.CancelledError as error:
                if cancelled_error is None:
                    cancelled_error = error
                emit(
                    StepOutcome(
                        id=step.id,
                        status=STEP_CLEANUP_FAILED,
                        error="cleanup interrupted by cancellation",
                    )
                )
                continue
            except PrimitiveExecutionError as error:
                emit(
                    StepOutcome(
                        id=step.id,
                        status=STEP_CLEANUP_FAILED,
                        error=str(error),
                        exit_code=error.exit_code,
                    )
                )
                continue
            except Exception as error:  # noqa: BLE001
                # One broken cleanup must not strand the remaining ones;
                # the first foreign error is still re-raised after the pass.
                if foreign_error is None:
                    foreign_error = error
                emit(
                    StepOutcome(
                        id=step.id,
                        status=STEP_CLEANUP_FAILED,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            absent_produces = [name for name in step.produces if name not in outputs]
            if absent_produces:
                emit(
                    StepOutcome(
                        id=step.id,
                        status=STEP_CLEANUP_FAILED,
                        error=f"step did not produce declared outputs {absent_produces}",
                    )
                )
                continue
            produced.update(outputs)
            emit(StepOutcome(id=step.id, status=STEP_CLEANED, outputs=outputs))

    if cancelled_error is not None:
        raise cancelled_error

    if foreign_error is not None:
        # Cleanup is guaranteed; propagation is NOT changed: the caller
        # (legacy parity or recipe runner) keeps owning this exception.
        raise foreign_error

    return finish(status)
