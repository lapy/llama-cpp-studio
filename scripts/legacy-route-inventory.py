#!/usr/bin/env python3
"""R102: inventory the legacy per-engine compatibility routes.

The serial refactor kept ``/api/{segment}/{verb}`` routers alive as compatibility
delegates while the generic ``/api/engines/{id}/{verb}`` router became the
frontend's path. Removing the delegates is a deliberate API-surface cutover, so it
needs a CURRENT blast-radius record: which modules, which endpoints, which
frontend descriptors would lose their rollback path, and which tests cover them.

Supports both shapes the codebase uses:
- modules that mount ``python_engine_router`` (possibly several times);
- modules that hand-write ``@router.<verb>("/<prefix>/<endpoint>")`` handlers.

Writes ``docs/legacy-route-inventory.md``; ``--check`` verifies it is current.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "backend" / "routes"
REGISTRY = ROOT / "frontend" / "src" / "engines" / "registry.js"
TESTS = ROOT / "backend" / "tests"
DOC = ROOT / "docs" / "legacy-route-inventory.md"

FACTORY_FILE = ROUTES / "engine_versions.py"
HTTP_VERBS = {"get", "post", "put", "delete", "patch"}
# Shared factory, the generic router itself, and llama/audio (own surfaces).
NOT_LEGACY = {"engine_versions.py", "llama_versions.py", "audio_cpp_versions.py"}


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _constants(text: str) -> dict:
    """Simple module-level string constants (used by hand-written modules)."""
    values = {}
    for match in re.finditer(r'^(_?[A-Z][A-Z0-9_]*)\s*=\s*"([^"]+)"', text, re.M):
        values[match.group(1)] = match.group(2)
    return values


def _resolve_expr(node, constants: dict) -> str:
    literal = _literal(node)
    if literal is not None:
        return literal
    name = getattr(node, "id", None)
    if name and name in constants:
        return constants[name]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("{…}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _resolve_expr(node.left, constants) + _resolve_expr(node.right, constants)
    return ""


def module_routes(path: Path) -> list[dict]:
    """(prefix, endpoints, engine ids) mounted by one route module."""
    text = path.read_text(encoding="utf-8")
    constants = _constants(text)
    tree = ast.parse(text)
    mounts = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name == "python_engine_router":
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            engine_id = _literal(kwargs.get("engine_id"))
            prefix = _literal(kwargs.get("url_prefix"))
            if engine_id and prefix:
                mounts.append({"engine_id": engine_id, "prefix": prefix, "wired": True})
        elif name == "definition_compat_router" and len(node.args) >= 2:
            # Converted engines mount the SAME lifecycle record through the
            # definition-derived factory; still a legacy compatibility prefix.
            engine_id = _resolve_expr(node.args[0], constants)
            prefix = _resolve_expr(node.args[1], constants)
            if engine_id and prefix:
                mounts.append({"engine_id": engine_id, "prefix": prefix, "wired": True})
        elif name in HTTP_VERBS and node.args:
            route = _resolve_expr(node.args[0], constants)
            match = re.match(r"^/([a-z0-9\-]+)/([a-z0-9\-]+)$", route)
            if match:
                mounts.append(
                    {
                        "engine_id": None,
                        "prefix": "/" + match.group(1),
                        "endpoint": match.group(2),
                        "wired": False,
                    }
                )
    grouped: dict = {}
    for mount in mounts:
        prefix = mount["prefix"]
        entry = grouped.setdefault(
            prefix, {"engine_ids": set(), "endpoints": set(), "factory": False}
        )
        if mount["engine_id"]:
            entry["engine_ids"].add(mount["engine_id"])
            entry["factory"] = True
        if mount.get("endpoint"):
            entry["endpoints"].add(mount["endpoint"])
    return [{"prefix": prefix, **info} for prefix, info in sorted(grouped.items())]


def factory_endpoints() -> list[str]:
    text = FACTORY_FILE.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'f"\{prefix\}/([a-z0-9\-]+)"', text)))


def _descriptor_blocks(text: str) -> dict:
    """Engine descriptor bodies, found with a brace scanner (regex was lossy)."""
    blocks = {}
    marker = "ENGINE_DESCRIPTORS = Object.freeze({"
    start = text.index(marker) + len(marker)
    depth = 1
    index = start
    while index < len(text) and depth > 0:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    body = text[start : index - 1]
    for match in re.finditer(r"(?:^|\n)  ['\"]?(\w+)['\"]?: Object\.freeze\(\{", body):
        engine_id = match.group(1)
        block_start = match.end()
        block_depth = 1
        cursor = block_start
        while cursor < len(body) and block_depth > 0:
            if body[cursor] == "{":
                block_depth += 1
            elif body[cursor] == "}":
                block_depth -= 1
            cursor += 1
        blocks[engine_id] = body[block_start : cursor - 1]
    return blocks


def generic_verb_facts() -> dict:
    """Native engines served by the generic router through NATIVE_GENERIC_VERBS."""
    text = REGISTRY.read_text(encoding="utf-8")
    match = re.search(r"NATIVE_GENERIC_VERBS = Object\.freeze\(\{(.*?)\n\}\)", text, re.S)
    if not match:
        return {}
    facts = {}
    for block in re.finditer(r"(\w+): Object\.freeze\(\[(.*?)\]\)", match.group(1), re.S):
        verbs = re.findall(r"'([^']+)'", block.group(2))
        facts[block.group(1)] = verbs
    return facts


def descriptor_facts() -> dict:
    blocks = _descriptor_blocks(REGISTRY.read_text(encoding="utf-8"))
    native = generic_verb_facts()
    facts = {}
    for engine_id, body in blocks.items():
        segment = re.search(r"apiSegment:\s*'([^']+)'", body)
        if not segment and engine_id not in native:
            continue
        facts[engine_id] = {
            "apiSegment": segment.group(1) if segment else "—",
            "genericApi": bool(re.search(r"genericApi:\s*true", body)),
            "nativeGenericVerbs": native.get(engine_id, []),
        }
    return facts


def covered_by_tests(module_name: str, prefixes: list[str]) -> list[str]:
    stem = "backend.routes." + module_name.removesuffix(".py")
    hits = []
    for path in sorted(TESTS.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if stem in text or any(f"api{prefix}/" in text for prefix in prefixes):
            hits.append(path.name)
    return hits


def build() -> str:
    modules = {
        path.name: module_routes(path)
        for path in sorted(ROUTES.glob("*_versions.py"))
        if path.name not in NOT_LEGACY
    }
    endpoints = factory_endpoints()
    descriptors = descriptor_facts()
    lines = [
        "# Legacy per-engine route inventory (generated)",
        "",
        "Generated by `scripts/legacy-route-inventory.py` (`--check` runs in the",
        "gates). These modules mount the LEGACY compatibility prefixes; the",
        "frontend uses the generic `/api/engines/{id}/{verb}` routes for",
        "`genericApi` descriptors and `NATIVE_GENERIC_VERBS` for the native",
        "engines. Deleting a module is an API-surface cutover: use this table as the",
        "blast radius and update it when the decision lands.",
        "",
        "Generic routing column: `venv` = the descriptor declares `genericApi`;",
        "`native` = the engine is served through `NATIVE_GENERIC_VERBS`; `no` = the",
        "legacy prefix is still the only surface for that engine.",
        "",
        "Shared factory endpoint suffixes (`python_engine_router`): "
        + ", ".join(f"`{e}`" for e in endpoints),
        "",
        "| module | engine(s) | legacy prefix | endpoints | frontend segment | generic routing | covered by tests |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows_written = 0
    for module_name, mounts in modules.items():
        for mount in mounts:
            prefix = mount["prefix"]
            engines = sorted(mount["engine_ids"]) or [_engine_from_module(module_name)]
            endpoint_count = len(endpoints) if mount["factory"] else len(mount["endpoints"])
            segments = []
            routing = []
            for engine_id in engines:
                descriptor = descriptors.get(engine_id, {})
                segments.append(descriptor.get("apiSegment", "—"))
                if descriptor.get("nativeGenericVerbs"):
                    routing.append("native")
                elif descriptor.get("genericApi"):
                    routing.append("venv")
                else:
                    routing.append("no")
            tests = covered_by_tests(module_name, [prefix])
            lines.append(
                "| `{}` | {} | `/api{}` | {} | `{}` | {} | {} |".format(
                    module_name,
                    ", ".join(f"`{engine}`" for engine in engines),
                    prefix,
                    endpoint_count,
                    ", ".join(sorted(set(segments))),
                    ", ".join(sorted(set(routing))),
                    ", ".join(f"`{name}`" for name in tests) or "—",
                )
            )
            rows_written += 1
    lines.append("")
    lines.append(f"_{rows_written} legacy mounts across {len(modules)} modules._")
    lines.append("")
    return "\n".join(lines)


def _engine_from_module(module_name: str) -> str:
    """Engine id implied by a hand-written module name (stem minus ``_versions``)."""
    return module_name.removesuffix(".py").removesuffix("_versions")


def main(argv: list[str]) -> int:
    document = build()
    if "--check" in argv:
        if not DOC.exists() or DOC.read_text(encoding="utf-8") != document:
            print("legacy route inventory is stale: run scripts/legacy-route-inventory.py")
            return 1
        print("legacy route inventory current")
        return 0
    DOC.write_text(document, encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
