"""Typed, validated engine definitions (Phase 2 of the config-driven engine
refactoring). See schema.py for scope and validation guarantees."""

from backend.engines.definitions.loader import (
    DEFINITIONS_DIR,
    REFERENCE_DEFINITIONS_DIR,
    EngineDefinitionRegistry,
    load_default_engine_definitions,
    load_engine_definitions,
)
from backend.engines.definitions.executor import (
    PlanExecution,
    PrimitiveExecutionError,
    StepOutcome,
    execute_plan,
)
from backend.engines.definitions.plan import InstallPlan, PlannedStep, materialize_plan
from backend.engines.definitions.primitives import (
    SHARED_PRIMITIVES,
    make_engine_runner,
    run_shared_primitive,
)
from backend.engines.definitions.profiles import compile_profile_argv, compile_profile_env
from backend.engines.definitions.schema import (
    INSTALL_PRIMITIVES,
    SUPPORTED_SCHEMA_VERSIONS,
    DefinitionError,
    EngineDefinition,
    evaluate_condition,
)

__all__ = [
    "DEFINITIONS_DIR",
    "INSTALL_PRIMITIVES",
    "REFERENCE_DEFINITIONS_DIR",
    "SUPPORTED_SCHEMA_VERSIONS",
    "DefinitionError",
    "EngineDefinition",
    "EngineDefinitionRegistry",
    "InstallPlan",
    "PlanExecution",
    "PlannedStep",
    "PrimitiveExecutionError",
    "SHARED_PRIMITIVES",
    "StepOutcome",
    "compile_profile_argv",
    "compile_profile_env",
    "evaluate_condition",
    "execute_plan",
    "load_default_engine_definitions",
    "load_engine_definitions",
    "make_engine_runner",
    "materialize_plan",
    "run_shared_primitive",
]
