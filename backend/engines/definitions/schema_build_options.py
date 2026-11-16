"""Static build-option catalog validation for engine definitions.

Extracted from :mod:`backend.engines.definitions.schema` (module-size ratchet):
the catalog is one cohesive unit - its accepted keys, kinds, categories and
cross-references - so it lives here instead of growing the 2.5k-line schema
module.

The generic validation primitives (``_require``, ``_check_str``, ...) stay in
``schema``: they are used by every other definition block, and importing them
at module scope here would be circular because ``schema`` imports this module.
The import is therefore deferred to call time, which happens once per loaded
definition.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

_BUILD_OPTION_KINDS = frozenset({"bool", "str", "enum", "int"})
_BUILD_OPTION_KEYS = frozenset(
    {
        "key",
        "field",
        "kind",
        "default",
        "label",
        "desc",
        "category",
        "cmake",
        "requires",
        "enum_values",
        "special",
        # R94: which engines of a shared family expose this entry in the UI.
        "engines",
        # False marks an upstream knob Studio catalogues but cannot build, so the
        # UI disables it and validation refuses it instead of failing in CMake.
        "supported",
    }
)
_BUILD_CATEGORY_KEYS = frozenset({"id", "label", "collapsed", "requires", "engines"})
_BUILD_REQUIRES_PSEUDO = frozenset({"model_set_custom", "cuda_or_hip"})


def validate_build_options(value: Any, where: str) -> Tuple[Tuple[str, Any], ...]:
    """Static build-option catalog: types, defaults, categories, references."""
    from backend.engines.definitions.schema import (
        _TOKEN_RE,
        _check_bool,
        _check_kind,
        _check_str,
        _check_token_list,
        _require,
    )

    _require(isinstance(value, Mapping), "build_options must be a mapping", where)
    unknown = set(value) - {
        "build_types",
        "build_type_default",
        "primary_keys",
        "categories",
        "options",
    }
    _require(not unknown, f"build_options has unknown keys {sorted(unknown)}", where)

    build_types = value.get("build_types")
    _require(
        isinstance(build_types, Sequence)
        and not isinstance(build_types, (str, bytes))
        and bool(build_types),
        "build_options.build_types must be a non-empty list",
        where,
    )
    for item in build_types:
        _check_str("build_options.build_types item", item, where)
    build_type_default = value.get("build_type_default")
    _require(
        isinstance(build_type_default, str) and build_type_default in build_types,
        "build_options.build_type_default must be one of build_types",
        where,
    )

    primary_keys = value.get("primary_keys") or []
    _require(
        isinstance(primary_keys, Sequence) and not isinstance(primary_keys, (str, bytes)),
        "build_options.primary_keys must be a list",
        where,
    )
    for item in primary_keys:
        _check_str("build_options.primary_keys item", item, where)

    categories = value.get("categories")
    _require(
        isinstance(categories, Sequence)
        and not isinstance(categories, (str, bytes))
        and bool(categories),
        "build_options.categories must be a non-empty list",
        where,
    )
    category_ids = []
    for index, category in enumerate(categories):
        cwhere = f"{where}.build_options.categories[{index}]"
        _require(isinstance(category, Mapping), "category must be a mapping", cwhere)
        unknown = set(category) - _BUILD_CATEGORY_KEYS
        _require(not unknown, f"category has unknown keys {sorted(unknown)}", cwhere)
        category_id = _check_str("id", category.get("id"), cwhere)
        _require(bool(_TOKEN_RE.match(category_id)), f"invalid category id {category_id!r}", cwhere)
        _check_str("label", category.get("label"), cwhere)
        if "collapsed" in category:
            _check_bool("collapsed", category["collapsed"], cwhere)
        if "requires" in category:
            _check_str("requires", category["requires"], cwhere)
        if "engines" in category:
            _check_token_list("engines", category["engines"], cwhere)
        category_ids.append(category_id)
    _require(
        len(set(category_ids)) == len(category_ids),
        "duplicate build-option category id",
        where,
    )

    options = value.get("options")
    _require(
        isinstance(options, Sequence) and not isinstance(options, (str, bytes)) and bool(options),
        "build_options.options must be a non-empty list",
        where,
    )
    seen_keys = []
    seen_fields = []
    for index, option in enumerate(options):
        owhere = f"{where}.build_options.options[{index}]"
        _require(isinstance(option, Mapping), "option must be a mapping", owhere)
        unknown = set(option) - _BUILD_OPTION_KEYS
        _require(not unknown, f"option has unknown keys {sorted(unknown)}", owhere)
        key = _check_str("key", option.get("key"), owhere)
        field = _check_str("field", option.get("field"), owhere)
        for token in (key, field):
            _require(bool(_TOKEN_RE.match(token)), f"invalid build-option token {token!r}", owhere)
        _require(key not in seen_keys, f"duplicate build-option key {key!r}", owhere)
        _require(field not in seen_fields, f"duplicate build-option field {field!r}", owhere)
        seen_keys.append(key)
        seen_fields.append(field)
        kind = _check_kind("kind", option.get("kind"), _BUILD_OPTION_KINDS, owhere)
        _check_str("label", option.get("label"), owhere)
        _check_str("desc", option.get("desc"), owhere)
        category = _check_str("category", option.get("category"), owhere)
        _require(category in category_ids, f"unknown build-option category {category!r}", owhere)
        default = option.get("default")
        if kind == "bool":
            _check_bool("default", default, owhere)
        elif kind == "int":
            _require(
                isinstance(default, int) and not isinstance(default, bool),
                "default must be an integer",
                owhere,
            )
            _require(default >= 0, "default must be non-negative", owhere)
        else:
            # String defaults may be empty (e.g. a free-text CMake-args field).
            _require(isinstance(default, str), "default must be a string", owhere)
        for key_name in ("cmake", "requires", "special"):
            if key_name in option:
                _check_str(key_name, option[key_name], owhere)
        if "supported" in option:
            _check_bool("supported", option["supported"], owhere)
        if "engines" in option:
            _check_token_list("engines", option["engines"], owhere)
        if kind == "enum":
            enum_values = option.get("enum_values")
            _require(
                isinstance(enum_values, Sequence)
                and not isinstance(enum_values, (str, bytes))
                and bool(enum_values),
                "enum option requires a non-empty enum_values list",
                owhere,
            )
            for item in enum_values:
                _check_str("enum_values item", item, owhere)
            _require(default in enum_values, "default must be one of enum_values", owhere)
        elif "enum_values" in option:
            _require(False, "enum_values is only allowed for kind 'enum'", owhere)

    for index, category in enumerate(categories):
        requires = category.get("requires")
        if requires:
            _require(
                requires in seen_keys,
                f"category requires unknown build-option {requires!r}",
                f"{where}.build_options.categories[{index}]",
            )
    for index, option in enumerate(options):
        requires = option.get("requires")
        if requires:
            _require(
                requires in seen_keys or requires in _BUILD_REQUIRES_PSEUDO,
                f"option requires unknown build-option {requires!r}",
                f"{where}.build_options.options[{index}]",
            )
    return tuple((key, value[key]) for key in sorted(value))


__all__ = ["validate_build_options"]
