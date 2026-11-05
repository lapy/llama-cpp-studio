"""Typed install plans materialized from schema v2 recipes (Phase 3 seed).

Turns a validated recipe into an ordered, typed :class:`InstallPlan` — the
shape the Phase 3 executor consumes. The loader guarantees requires-before-
produces, unique step ids, and known primitives, so materialization only
re-checks those invariants (defense in depth against hand-built registries)
and binds step inputs. **Nothing executes an InstallPlan yet**; no installer
code imports this module (shadow-mode guard test enforces it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .schema import DefinitionError, EngineDefinition, INSTALL_PRIMITIVES, validate_condition


@dataclass(frozen=True)
class PlannedStep:
    id: str
    primitive: Optional[str]
    run: Tuple[str, ...]
    with_: Mapping[str, Any]
    when: Optional[Mapping[str, Any]]
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]
    timeout_s: Optional[int]
    accepted_exit_codes: Tuple[int, ...]
    cleanup: bool = False
    """True = always-run step: executes even after the body failed or was
    cancelled (the recipe's finally, for native build teardown)."""

    @property
    def is_raw_command(self) -> bool:
        return self.run != ()


@dataclass(frozen=True)
class InstallPlan:
    engine_id: str
    recipe: str
    steps: Tuple[PlannedStep, ...]

    @property
    def required_inputs(self) -> Tuple[str, ...]:
        """input.* context fields the plan consumes, in first-use order."""
        seen: list = []
        for step in self.steps:
            for requirement in step.requires:
                if requirement.startswith("input.") and requirement not in seen:
                    seen.append(requirement)
        return tuple(seen)

    def by_id(self, step_id: str) -> PlannedStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise DefinitionError(f"{self.engine_id}: plan has no step {step_id!r}")


def _to_step(step: Mapping[str, Any], where: str) -> PlannedStep:
    uses, run = step.get("uses"), step.get("run")
    if (uses is None) == (run is None):
        raise DefinitionError(f"{where}: step needs exactly one of 'uses' or 'run'")
    if uses is not None and uses not in INSTALL_PRIMITIVES:
        raise DefinitionError(f"{where}: unknown primitive {uses!r}")
    if run is not None:
        run = tuple(str(token) for token in run)
    when = step.get("when")
    if when is not None:
        validate_condition(when, f"{where}.when")
    return PlannedStep(
        id=str(step["id"]),
        primitive=uses,
        run=tuple(run or ()),
        with_=dict(step.get("with") or {}),
        when=when,
        requires=tuple(step.get("requires") or ()),
        produces=tuple(step.get("produces") or ()),
        timeout_s=step.get("timeout_s"),
        accepted_exit_codes=tuple(step.get("accepted_exit_codes") or ()),
        cleanup=bool(step.get("cleanup", False)),
    )


def materialize_plan(definition: EngineDefinition, recipe: str) -> InstallPlan:
    """Materialize one named recipe of a validated definition."""
    recipes = dict(definition.recipes)
    if recipe not in recipes:
        raise DefinitionError(
            f"{definition.id}: no recipe {recipe!r} (declared: {sorted(recipes)})"
        )
    steps_raw: Sequence[Mapping[str, Any]] = recipes[recipe]
    produced: set = set()
    seen_ids: set = set()
    steps = []
    for raw in steps_raw:
        where = f"{definition.id}.recipes.{recipe}.{raw.get('id')}"
        step = _to_step(raw, where)
        if step.id in seen_ids:
            raise DefinitionError(f"{where}: duplicate step id")
        seen_ids.add(step.id)
        for requirement in step.requires:
            if requirement.startswith("input."):
                continue
            if requirement not in produced:
                raise DefinitionError(f"{where}: requires {requirement!r} before it is produced")
        produced.update(step.produces)
        steps.append(step)
    return InstallPlan(engine_id=definition.id, recipe=recipe, steps=tuple(steps))
