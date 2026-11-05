"""Engine definition directory loader: template expansion + immutable registry.

Loads ``backend/engines/definitions/*.yaml`` (engine files) and
``_templates.yaml`` (named templates), expands inheritance deterministically
(map merge, list replacement, left-to-right ``extends``, engine keys last),
validates each expansion via :mod:`backend.engines.definitions.schema`, and
freezes the result. Loading errors raise :class:`DefinitionError` with the
offending file path. See the schema module docstring for phase scope.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.engines.definitions.schema import (
    TEMPLATE_KEYS,
    DefinitionError,
    EngineDefinition,
    load_yaml_file_strict,
    validate_definition,
)

DEFINITIONS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_FILENAME = "_templates.yaml"
# schema_version 2 reference definitions (reference-spike deliverable). The
# default loader scans *.yaml at this level only, so this subdirectory is
# never picked up at runtime; tests load it explicitly.
REFERENCE_DEFINITIONS_DIR = os.path.join(DEFINITIONS_DIR, "reference")


@dataclass(frozen=True)
class EngineDefinitionRegistry:
    """Immutable map of engine id -> validated EngineDefinition."""

    definitions: Mapping[str, EngineDefinition]

    def get(self, engine_id: Optional[str]) -> Optional[EngineDefinition]:
        return self.definitions.get(str(engine_id or ""))

    def ids(self) -> Tuple[str, ...]:
        return tuple(self.definitions)

    def all(self) -> List[EngineDefinition]:
        return list(self.definitions.values())


def _merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic merge: mappings deep-merge per key; lists/scalars replace."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def _load_templates(templates_path: str) -> Dict[str, Dict[str, Any]]:
    doc = load_yaml_file_strict(templates_path)
    where = templates_path
    if not isinstance(doc, dict):
        raise DefinitionError(f"{where}: template document must be a mapping")
    if "schema_version" in doc and doc["schema_version"] != 1:
        raise DefinitionError(f"{where}: unsupported template schema_version")
    templates = doc.get("templates")
    if templates is None:
        templates = {}
    if not isinstance(templates, dict):
        raise DefinitionError(f"{where}: 'templates' must be a mapping")
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, body in templates.items():
        if not isinstance(name, str) or not isinstance(body, dict):
            raise DefinitionError(f"{where}: template {name!r} must map a name to a mapping")
        unknown = set(body) - (TEMPLATE_KEYS | {"extends"})
        if unknown:
            raise DefinitionError(
                f"{where}: template {name!r} has unknown fields {sorted(unknown)}"
            )
        normalized[name] = body
    return normalized


def _template_body(
    name: str, templates: Mapping[str, Dict[str, Any]], chain: List[str], where: str
) -> Dict[str, Any]:
    if name in chain:
        raise DefinitionError(f"{where}: template inheritance cycle: {' -> '.join(chain + [name])}")
    body = templates.get(name)
    if body is None:
        raise DefinitionError(f"{where}: extends unknown template {name!r}")
    expanded: Dict[str, Any] = {}
    parents = body.get("extends")
    if parents is not None:
        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
            raise DefinitionError(f"{where}: template {name!r} 'extends' must be a list of names")
        for parent in parents:
            expanded = _merge(expanded, _template_body(parent, templates, chain + [name], where))
    expanded = _merge(expanded, {k: v for k, v in body.items() if k != "extends"})
    return expanded


def _expand_engine(
    doc: Mapping[str, Any], templates: Mapping[str, Dict[str, Any]], where: str
) -> Dict[str, Any]:
    extends = doc.get("extends")
    expanded: Dict[str, Any] = {}
    if extends is not None:
        if isinstance(extends, str):
            extends = [extends]
        if not isinstance(extends, list) or not all(isinstance(e, str) for e in extends):
            raise DefinitionError(f"{where}: 'extends' must be a list of template names")
        for name in extends:
            expanded = _merge(
                expanded, _template_body(name, templates, [f"<engine {where}>"], where)
            )
    own = {k: v for k, v in doc.items() if k != "extends"}
    return _merge(expanded, own)


def _validate_transform_assets(definition: EngineDefinition) -> None:
    """Fail startup when a recipe names a missing/invalid transform asset."""
    from backend.engines.definitions.transforms import TransformError, load_transform

    for recipe_name, steps in definition.recipes:
        for step in steps:
            if step.get("uses") != "vcs.apply_transform":
                continue
            name = str((step.get("with") or {}).get("transform") or "")
            try:
                load_transform(name)
            except TransformError as exc:
                raise DefinitionError(
                    f"{definition.source_path}: recipe {recipe_name!r} step "
                    f"{step.get('id')!r}: {exc}"
                ) from exc


def load_engine_definitions(directory: str = DEFINITIONS_DIR) -> EngineDefinitionRegistry:
    """Load, expand, and validate every engine definition in ``directory``."""
    directory = os.fspath(directory)
    templates_path = os.path.join(directory, TEMPLATES_FILENAME)
    if not os.path.isdir(directory):
        raise DefinitionError(f"definitions directory not found: {directory}")
    templates = _load_templates(templates_path) if os.path.exists(templates_path) else {}

    filenames = sorted(
        name
        for name in os.listdir(directory)
        if name.endswith(".yaml") and name != TEMPLATES_FILENAME and not name.startswith("_")
    )
    definitions: Dict[str, EngineDefinition] = {}
    for filename in filenames:
        path = os.path.join(directory, filename)
        doc = load_yaml_file_strict(path)
        if not isinstance(doc, dict):
            raise DefinitionError(f"{path}: engine definition must be a mapping")
        expanded = _expand_engine(doc, templates, path)
        definition = validate_definition(expanded, source=path)
        _validate_transform_assets(definition)
        stem = os.path.splitext(filename)[0]
        if stem != definition.id:
            raise DefinitionError(
                f"{path}: id {definition.id!r} does not match filename stem {stem!r}"
            )
        if definition.id in definitions:
            raise DefinitionError(
                f"{path}: duplicate engine id {definition.id!r} "
                f"(already defined by {definitions[definition.id].source_path})"
            )
        definitions[definition.id] = definition
    return EngineDefinitionRegistry(definitions=definitions)


_default_registry: Optional[EngineDefinitionRegistry] = None


def load_default_engine_definitions() -> EngineDefinitionRegistry:
    """Process-wide singleton: definitions load and validate once at startup."""
    global _default_registry
    if _default_registry is None:
        _default_registry = load_engine_definitions()
    return _default_registry
