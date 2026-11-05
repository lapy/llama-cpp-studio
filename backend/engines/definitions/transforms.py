"""Versioned file-transform assets: applicability-checked, idempotent rewrites.

Engine forks often need to adapt an upstream checkout (bootstrap scripts,
metadata pins, generated build wrappers). Those rewrites must NOT live as
engine-specific Python: they are declared as versioned assets here and applied
by ONE primitive (``vcs.apply_transform``).

Asset shape (strict; unknown fields are rejected)::

    schema_version: 1
    name: sglang_v100/pyproject       # must equal the asset path stem
    description: ...
    file: python/pyproject.toml       # path inside the checkout
    optional_file: false              # true: a missing file is a no-op
    output: scripts/install_studio.sh # optional; defaults to `file`
    executable: false
    marker: "# already transformed"   # optional idempotence marker
    operations:
      - kind: replace_all             # every search must occur exactly once
        replacements: [{search: ..., replace: ...}]
      - kind: replace_all_or_none     # skip if none present; fail if partial
        replacements: [{search: ..., replace: ...}]
      - kind: replace                 # optional: true skips a missing search
        search: ...
        replace: ...
        optional: false
        skip_if_contains: ...
      - kind: replace_region          # inclusive start .. before end
        start: ...
        end: ...
        replace: ...
        skip_if_contains: ...
      - kind: insert_after
        marker: ...
        text: ...
        skip_if_contains: ...

Applicability failures raise :class:`TransformError` naming the asset, the
operation and the upstream text, so a changed fork layout is a descriptive
failure instead of a silent bad build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_OPERATION_KINDS = frozenset(
    {"replace_all", "replace_all_or_none", "replace", "replace_region", "insert_after"}
)


class TransformError(RuntimeError):
    """A transform asset is invalid or does not apply to this checkout."""


@dataclass(frozen=True)
class TransformAsset:
    name: str
    description: str
    file: str
    optional_file: bool
    output: str
    executable: bool
    marker: str
    prepend_marker: bool
    operations: Tuple[Mapping[str, Any], ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransformError(message)


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str), f"{where} must be a string")
    _require(allow_empty or bool(value), f"{where} must not be empty")
    return value


def _relative_path(value: Any, where: str) -> str:
    path = _string(value, where)
    _require(not os.path.isabs(path), f"{where} must be relative to the checkout")
    _require(".." not in path.split("/"), f"{where} must not escape the checkout")
    return path


def _replacements(value: Any, where: str) -> List[Dict[str, str]]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{where} list")
    _require(bool(value), f"{where} must not be empty")
    out = []
    for index, item in enumerate(value):
        iwhere = f"{where}[{index}]"
        _require(isinstance(item, Mapping), f"{iwhere} must be a mapping")
        unknown = set(item) - {"search", "replace"}
        _require(not unknown, f"{iwhere} has unknown keys {sorted(unknown)}")
        out.append(
            {
                "search": _string(item.get("search"), f"{iwhere}.search"),
                "replace": _string(item.get("replace"), f"{iwhere}.replace", allow_empty=True),
            }
        )
    return out


def _validate_operation(raw: Any, index: int, where: str) -> Dict[str, Any]:
    iwhere = f"{where}.operations[{index}]"
    _require(isinstance(raw, Mapping), f"{iwhere} must be a mapping")
    kind = _string(raw.get("kind"), f"{iwhere}.kind")
    _require(kind in _OPERATION_KINDS, f"{iwhere} has unknown kind {kind!r}")
    allowed = {
        "replace_all": {"kind", "replacements"},
        "replace_all_or_none": {"kind", "replacements"},
        "replace": {"kind", "search", "replace", "optional", "skip_if_contains"},
        "replace_region": {"kind", "start", "end", "replace", "skip_if_contains"},
        "insert_after": {"kind", "marker", "text", "skip_if_contains"},
    }[kind]
    unknown = set(raw) - allowed
    _require(not unknown, f"{iwhere} has unknown keys {sorted(unknown)}")
    operation: Dict[str, Any] = {"kind": kind}
    if kind in {"replace_all", "replace_all_or_none"}:
        operation["replacements"] = _replacements(raw.get("replacements"), f"{iwhere}.replacements")
    elif kind == "replace":
        operation["search"] = _string(raw.get("search"), f"{iwhere}.search")
        operation["replace"] = _string(raw.get("replace"), f"{iwhere}.replace", allow_empty=True)
        operation["optional"] = bool(raw.get("optional", False))
        if "skip_if_contains" in raw:
            operation["skip_if_contains"] = _string(
                raw["skip_if_contains"], f"{iwhere}.skip_if_contains"
            )
    elif kind == "replace_region":
        operation["start"] = _string(raw.get("start"), f"{iwhere}.start")
        operation["end"] = _string(raw.get("end"), f"{iwhere}.end")
        operation["replace"] = _string(raw.get("replace"), f"{iwhere}.replace", allow_empty=True)
        if "skip_if_contains" in raw:
            operation["skip_if_contains"] = _string(
                raw["skip_if_contains"], f"{iwhere}.skip_if_contains"
            )
    else:  # insert_after
        operation["marker"] = _string(raw.get("marker"), f"{iwhere}.marker")
        operation["text"] = _string(raw.get("text"), f"{iwhere}.text")
        if "skip_if_contains" in raw:
            operation["skip_if_contains"] = _string(
                raw["skip_if_contains"], f"{iwhere}.skip_if_contains"
            )
    return operation


def load_transform(name: str, *, assets_dir: Optional[str] = None) -> TransformAsset:
    """Load and strictly validate one transform asset (``dir/stem``)."""
    _require(bool(name) and ".." not in name, f"invalid transform name {name!r}")
    base = assets_dir or ASSETS_DIR
    path = os.path.join(base, *name.split("/")) + ".yaml"
    if not os.path.isfile(path):
        raise TransformError(f"transform asset not found: {name}")
    with open(path, "r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise TransformError(f"{name}: invalid YAML ({exc})") from exc
    _require(isinstance(raw, Mapping), f"{name}: transform asset must be a mapping")
    unknown = set(raw) - {
        "schema_version",
        "name",
        "description",
        "file",
        "optional_file",
        "output",
        "executable",
        "marker",
        "prepend_marker",
        "operations",
    }
    _require(not unknown, f"{name}: unknown fields {sorted(unknown)}")
    _require(raw.get("schema_version") == 1, f"{name}: schema_version must be 1")
    declared_name = _string(raw.get("name"), f"{name}.name")
    _require(declared_name == name, f"{name}: name must equal the asset path stem")
    file_path = _relative_path(raw.get("file"), f"{name}.file")
    output = _relative_path(raw["output"], f"{name}.output") if "output" in raw else file_path
    operations = raw.get("operations")
    _require(
        isinstance(operations, Sequence) and not isinstance(operations, (str, bytes)),
        f"{name}.operations must be a list",
    )
    normalized = tuple(
        _validate_operation(item, index, name) for index, item in enumerate(operations)
    )
    return TransformAsset(
        name=name,
        description=str(raw.get("description") or ""),
        file=file_path,
        optional_file=bool(raw.get("optional_file", False)),
        output=output,
        executable=bool(raw.get("executable", False)),
        marker=str(raw.get("marker") or ""),
        prepend_marker=bool(raw.get("prepend_marker", False)),
        operations=normalized,
    )


def _apply_operation(asset: TransformAsset, text: str, operation: Mapping[str, Any]) -> str:
    kind = operation["kind"]
    marker = f"{asset.name}: {kind}"

    def fail(detail: str) -> None:
        raise TransformError(f"{marker}: {detail}")

    skip = operation.get("skip_if_contains")
    if skip and skip in text:
        return text

    if kind == "replace_all":
        for replacement in operation["replacements"]:
            search = replacement["search"]
            if text.count(search) != 1:
                fail(f"expected exactly one occurrence of {search[:80]!r}")
            text = text.replace(search, replacement["replace"], 1)
        return text

    if kind == "replace_all_or_none":
        replacements = operation["replacements"]
        present = [replacement["search"] in text for replacement in replacements]
        if not any(present):
            return text
        if not all(present):
            fail("only part of the expected upstream text is present")
        for replacement in replacements:
            search = replacement["search"]
            if text.count(search) != 1:
                fail(f"ambiguous occurrence of {search[:80]!r}")
            text = text.replace(search, replacement["replace"], 1)
        return text

    if kind == "replace":
        search = operation["search"]
        if search not in text:
            if operation.get("optional"):
                return text
            fail(f"expected upstream text is missing: {search[:80]!r}")
        if text.count(search) != 1:
            fail(f"ambiguous upstream text: {search[:80]!r}")
        return text.replace(search, operation["replace"], 1)

    if kind == "replace_region":
        start_marker = operation["start"]
        end_marker = operation["end"]
        if text.count(start_marker) != 1 or text.count(end_marker) != 1:
            fail("region markers are missing or ambiguous")
        start = text.index(start_marker)
        end = text.index(end_marker)
        if end <= start:
            fail("region markers are out of order")
        return text[:start] + operation["replace"] + text[end:]

    marker_text = operation["marker"]
    if text.count(marker_text) != 1:
        fail("insertion marker is missing or ambiguous")
    index = text.index(marker_text) + len(marker_text)
    return text[:index] + "\n" + operation["text"] + text[index:]


def apply_transform(name: str, checkout: str) -> Optional[str]:
    """Apply one asset to a checkout; returns the written file or None."""
    asset = load_transform(name)
    source = os.path.join(checkout, asset.file)
    if not os.path.isfile(source):
        if asset.optional_file:
            return None
        raise TransformError(f"{asset.name}: required file is missing: {asset.file}")
    with open(source, "r", encoding="utf-8") as handle:
        original = handle.read()
    text = original
    if not (asset.marker and asset.marker in text):
        for operation in asset.operations:
            text = _apply_operation(asset, text, operation)
        if asset.marker and asset.prepend_marker:
            text = f"{asset.marker}\n{text}"
    destination = os.path.join(checkout, asset.output)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if text != original or destination != source:
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(text)
    if asset.executable:
        os.chmod(destination, 0o755)
    return destination
