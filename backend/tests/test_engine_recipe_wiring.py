"""Recipe wiring invariant: every ``*_from``/``requires`` names a producer.

This is the harness that would have caught the 1Cat `toolchain_env` contract
mismatch class of bug WITHOUT running an install: for every shipped
definition and every recipe, each requirement and every ``with`` reference
ending in ``_from`` must name an earlier step output or an ``input.*``
context key.
"""

from __future__ import annotations

import pytest

from backend.engines.definitions import load_default_engine_definitions, materialize_plan

RECIPES = ("release", "source", "sync", "remove", "remove_version", "build")  # fallback only


def _definitions():
    registry = load_default_engine_definitions()
    return [(engine_id, registry.get(engine_id)) for engine_id in registry.ids()]


def _plans():
    plans = []
    for engine_id, definition in _definitions():
        if definition is None:
            continue
        recipe_names = (
            [name for name, _steps in definition.recipes]
            if getattr(definition, "recipes", None)
            else list(RECIPES)
        )
        for recipe in recipe_names:
            try:
                plan = materialize_plan(definition, recipe)
            except Exception as exc:
                raise AssertionError(
                    f"{engine_id}/{recipe}: definition does not materialize: {exc}"
                ) from exc
            if plan is not None:
                plans.append((engine_id, recipe, plan))
    return plans


@pytest.mark.parametrize(
    "engine_id,recipe,plan",
    _plans(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_recipe_references_resolve(engine_id, recipe, plan):
    produced = set()
    for step in plan.steps:
        for requirement in step.requires:
            requirement = str(requirement)
            if requirement.startswith("input."):
                continue
            assert requirement in produced, (
                f"{engine_id}/{recipe}: step {step.id!r} requires {requirement!r} "
                f"which no earlier step produces"
            )
        for key, value in (step.with_ or {}).items():
            if not str(key).endswith("_from"):
                continue
            reference = str(value)
            if reference.startswith("input.") or reference in {
                "checkout",
                "venv",
                "workspace",
            }:
                continue
            assert reference in produced or reference.startswith("settings."), (
                f"{engine_id}/{recipe}: step {step.id!r} {key}={reference!r} "
                f"is not produced or an input"
            )
        produced.update(str(name) for name in step.produces)
