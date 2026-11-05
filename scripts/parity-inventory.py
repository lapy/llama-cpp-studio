#!/usr/bin/env python3
"""Generate the frozen engine-behavior inventory (Phase 1 of the config-driven
engine refactoring, docs/config-driven-engine-refactoring-plan.md).

The plan requires an inventory of current routes, engine-ID literals, and
per-engine operation surfaces that parity claims can be measured against
("one row per feature, implementation reference"). Hand-maintaining that list
makes it stale on the next commit, so this tool derives it from the live app,
the registry, and the source tree, and emits:

    docs/engine-parity-inventory.json   machine-readable (gate inputs compare against this)
    docs/engine-parity-inventory.md     human review format

Re-run after any behavior migration and commit the regenerated files:

    python scripts/parity-inventory.py

Exit code 1 with --check means the committed inventory is stale.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engines.registry import ENGINE_REGISTRY  # noqa: E402

JSON_PATH = ROOT / "docs" / "engine-parity-inventory.json"
MD_PATH = ROOT / "docs" / "engine-parity-inventory.md"

# Path fragments and literal spellings used to attribute hardcoded behavior to
# an engine. Both snake_case ids and the hyphenated/cased spellings found in
# routes, URLs, and frontend code are covered.
ENGINE_TOKENS: dict = {
    "llama_cpp": ["llama_cpp", "llamaCpp", "llama-versions", "llama-server"],
    # /api/llama-versions/* serves the whole llama-server family (engine is a
    # request-body field, not a path segment), so parity rows must attribute
    # those endpoints to each family member.
    "ik_llama": ["ik_llama", "ikLlama", "ik-llama", "llama-versions"],
    "unsloth_llama": ["unsloth_llama", "unslothLlama", "unsloth"],
    "lmdeploy": ["lmdeploy", "lmDeploy"],
    "vllm": ["vllm", "vLLM"],
    "1cat_vllm": ["1cat_vllm", "1cat-vllm", "onecat", "oneCat", "onecat_vllm", "onecatvllm"],
    "sglang": ["sglang", "sglangV100"],
    "sglang_v100": ["sglang_v100", "sglangV100", "v100"],
    "audio_cpp": ["audio_cpp", "audioCpp", "audio-cpp"],
}

SCAN_KIND_ATTRIBUTION = {
    "llama_help": ["llama_cpp", "ik_llama", "unsloth_llama"],
    "lmdeploy_help": ["lmdeploy"],
    "vllm_help": ["vllm", "1cat_vllm"],
    "sglang_help": ["sglang", "sglang_v100"],
    "audio_cpp": ["audio_cpp"],
}

# The per-engine lifecycle operations the adapter must reproduce (plan's
# ConfiguredEngineAdapter surface). Matching is path-SEGMENT equality, not
# substring: cuda-install/cuda-status/cuda-cancel are CUDA system
# capabilities mounted under an engine prefix, not the engine lifecycle, and
# segment matching also captures the real aliases (build-source, build-cancel,
# scan-engine-params) instead of substring collisions.
OPERATION_KEYWORDS: dict = {
    "install_release": {"install", "install-release", "build"},
    "install_source": {"install-source", "build-source"},
    "retry": {"retry"},
    "sync_source": {"sync", "sync-source"},
    "check_updates": {"check-updates", "updates", "update-check", "update"},
    "activate": {"activate"},
    "delete": {"delete", "remove"},
    "status": {"status"},
    "logs": {"logs", "log"},
    "cancel": {"cancel", "build-cancel"},
    "scan": {"scan", "rescan", "scan-engine-params"},
    "contract": {"contract"},
    "settings": {"settings", "build-settings", "build-options", "build-config"},
    "launch_preview": {"cmd", "command", "launch", "preview"},
}


def _lifecycle_segments(path: str) -> list:
    """Path segments excluding CUDA system-feature scopes (cuda, cuda-*)."""
    segments = [s for s in path.split("/") if s]
    out = []
    for segment in segments[1:]:  # drop the /api or /v1 root
        lowered = segment.lower()
        if lowered == "cuda" or lowered.startswith("cuda-"):
            return out  # everything under a cuda scope is a CUDA capability
        out.append(lowered)
    return out


def _engine_for_path(path: str) -> set:
    engines = set()
    lowered = path.lower()
    for engine_id, tokens in ENGINE_TOKENS.items():
        if any(token.lower() in lowered for token in tokens):
            engines.add(engine_id)
    return engines


def _response_schema_name(responses: dict) -> str:
    """Resolve the success-response schema name from an OpenAPI operation."""
    for code in sorted(responses):
        if not code.startswith("2"):
            continue
        content = responses[code].get("content", {})
        media = content.get("application/json") or next(iter(content.values()), {})
        schema = media.get("schema")
        if schema is None:
            return "empty" if not content else "media"
        seen = 0
        while "$ref" in schema and seen < 5:
            return schema["$ref"].split("/")[-1]
        if schema.get("type") == "array" and seen < 5:
            items = schema.get("items", {})
            if "$ref" in items:
                return items["$ref"].split("/")[-1] + "[]"
            return f"{items.get('type', 'object')}[]"
        if schema.get("type") == "object":
            return "object" if not schema.get("properties") else "object{...}"
        return str(schema.get("type", "unknown"))
    return "unknown"


def collect_routes(spec: dict) -> list:
    # The app defers router flattening (custom _IncludedRouter objects); the
    # OpenAPI document is the authoritative flattened surface — the same
    # contract the frontend consumes (frontend/src/api/openapi.json).
    http_methods = {"get", "put", "post", "delete", "patch", "head", "options"}
    routes = []
    for path, operations in spec.get("paths", {}).items():
        if not (path.startswith("/api") or path.startswith("/v1")):
            continue
        methods = sorted(m for m in operations if m.lower() in http_methods)
        if not methods:
            continue
        responses = {
            m.upper(): _response_schema_name(operations[m].get("responses", {})) for m in methods
        }
        routes.append(
            {
                "path": path,
                "methods": [m.upper() for m in methods],
                "responses": responses,
                "engines": sorted(_engine_for_path(path)),
            }
        )
    routes.sort(key=lambda r: (r["path"], r["methods"]))
    return routes


def collect_defaults() -> dict:
    """Per-engine persisted install/build defaults, from their live getters."""
    from backend.engines.audio_cpp.build_options import default_build_settings as audio_defaults
    from backend.engines.audio_cpp.manager import AUDIO_CPP_DEFAULT_REF
    from backend.engines.llama_cpp.build_options import default_build_settings as llama_defaults
    from backend.venv_install_settings import default_install_settings

    out: dict = {}
    for engine_id, spec in ENGINE_REGISTRY.items():
        entry: dict = {"source": None, "values": {}}
        try:
            if spec.install_kind == "python_venv":
                entry = {
                    "source": "backend/venv_install_settings.py:default_install_settings",
                    "values": default_install_settings(engine_id),
                }
            elif engine_id in ("llama_cpp", "ik_llama"):
                entry = {
                    "source": "backend/engines/llama_cpp/build_options.py:default_build_settings (shared llama-server family)",
                    "values": llama_defaults(),
                }
            elif engine_id == "audio_cpp":
                entry = {
                    "source": "backend/engines/audio_cpp/build_options.py:default_build_settings",
                    "values": {
                        "build_settings": audio_defaults(),
                        "source_ref_default": AUDIO_CPP_DEFAULT_REF,
                    },
                }
            else:
                entry = {
                    "source": None,
                    "values": {},
                    "note": "prebuilt installs have no persisted settings surface",
                }
        except ValueError:
            entry = {"source": None, "values": {}, "note": "no defaults registered"}
        out[engine_id] = entry
    return out


def collect_response_divergences(routes: list, operations: dict) -> dict:
    """Documented divergence sites: update-check and status response shapes."""
    by_key = {(r["path"], m): r["responses"][m] for r in routes for m in r["methods"]}

    def representative(endpoints: list, op: str, canonical: str):
        """Prefer the engine's own router (canonical /api/<engine-id>/ prefix,
        so family-token overlap like vllm vs 1cat-vllm cannot misattribute),
        then the endpoint whose last segment IS the operation keyword."""
        if not endpoints:
            return None
        keywords = OPERATION_KEYWORDS.get(op, set())
        for endpoint in endpoints:
            path = endpoint.partition(" ")[2]
            last_segment = path.rstrip("/").rpartition("/")[2].lower()
            if last_segment in keywords and canonical in path:
                return endpoint
        for endpoint in endpoints:
            path = endpoint.partition(" ")[2]
            last_segment = path.rstrip("/").rpartition("/")[2].lower()
            if last_segment in keywords:
                return endpoint
        return endpoints[0]

    out: dict = {}
    for engine_id, per_op in operations.items():
        canonical = f"/{engine_id.replace('_', '-')}/"
        shapes = {}
        for op in ("check_updates", "status", "install_release", "contract"):
            endpoint = representative(per_op.get(op, []), op, canonical)
            if not endpoint:
                continue
            methods_part, _, path = endpoint.partition(" ")
            method = methods_part.split(",")[0]
            shapes[op] = f"{method} {path} -> {by_key.get((path, method), '?')}"
        out[engine_id] = shapes
    return out


def collect_engine_literals() -> dict:
    """Every quoted/word-boundary engine-id literal outside backend/engines/."""
    patterns = {}
    for engine_id, tokens in ENGINE_TOKENS.items():
        alternatives = sorted(set(tokens), key=len, reverse=True)
        patterns[engine_id] = re.compile(
            r"(?<![\w.-])(" + "|".join(re.escape(t) for t in alternatives) + r")(?![\w-])"
        )
    literals: dict = defaultdict(lambda: defaultdict(list))
    trees = [
        ("backend", ROOT / "backend"),
        ("frontend", ROOT / "frontend" / "src"),
        ("frontend_e2e", ROOT / "frontend" / "e2e"),
        ("scripts", ROOT / "scripts"),
    ]
    skip_parts = {"__pycache__", "node_modules", "dist", "tests"}
    for tree, base in trees:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in {
                ".py",
                ".js",
                ".ts",
                ".vue",
                ".mjs",
                ".json",
            }:
                continue
            if skip_parts & set(path.parts):
                continue
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            # backend/engines/ is the definition home; literals there are
            # legitimate and tracked separately.
            in_engine_home = rel.startswith("backend/engines/") and tree == "backend"
            if tree == "backend" and rel.startswith("backend/engines/definitions/"):
                continue
            if tree == "frontend" and rel.endswith("api/openapi.json"):
                continue
            if tree == "backend" and rel.startswith("backend/engines/scan/"):
                # scanner dispatch and parsers are kind-keyed vocabulary, not
                # per-engine control flow; counted under engines_home.
                in_engine_home = True
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            lines = content.splitlines()
            for engine_id, pattern in patterns.items():
                hits = [i + 1 for i, line in enumerate(lines) if pattern.search(line)]
                # Longer-id double counting: v100 lines also match sglang,
                # onecat lines also match vllm; keep them in both columns but
                # record the parent-overlap so the plan checklist can dedupe.
                if not hits:
                    continue
                key = "engines_home" if in_engine_home else "outside_engines"
                literals[engine_id][key].append({"file": rel, "lines": hits, "count": len(hits)})
    return {
        engine_id: {
            key: sorted(entries, key=lambda e: -e["count"]) for key, entries in buckets.items()
        }
        for engine_id, buckets in literals.items()
    }


def collect_operations(routes: list) -> dict:
    ops: dict = {}
    for engine_id in ENGINE_REGISTRY:
        per_op = {}
        engine_routes = [r for r in routes if engine_id in r["engines"]]
        for op, keywords in OPERATION_KEYWORDS.items():
            endpoints = []
            for route in engine_routes:
                if any(segment in keywords for segment in _lifecycle_segments(route["path"])):
                    endpoints.append(",".join(route["methods"]) + " " + route["path"])
            if endpoints:
                per_op[op] = sorted(set(endpoints))
        ops[engine_id] = per_op
    return ops


def build_inventory() -> dict:
    # Imported lazily so other tools (the engine-literal ratchet) can reuse
    # the source-tree collectors in this module without booting the app.
    # Same lightweight app boot the OpenAPI exporter uses (no services started).
    os.environ.setdefault("STUDIO_DATA_DIR", tempfile.mkdtemp(prefix="studio-parity-"))
    os.environ.setdefault("STUDIO_ACCESS_MODE", "local")
    from backend.main import app

    spec = app.openapi()
    routes = collect_routes(spec)
    literals = collect_engine_literals()
    operations = collect_operations(routes)
    from backend.engines.scan.scanner import SCANNER_KIND_DISPATCH
    from backend.engines.definitions import load_default_engine_definitions

    definitions = load_default_engine_definitions()
    return {
        "schema_version": 2,
        "engine_count": len(ENGINE_REGISTRY),
        "registry": {
            engine_id: eng_spec.to_api_dict() for engine_id, eng_spec in ENGINE_REGISTRY.items()
        },
        "shadow_definitions_loaded": sorted(definitions.ids()),
        "scanner_kinds": {
            kind: {"dispatched": True, "engines": SCAN_KIND_ATTRIBUTION.get(kind, [])}
            for kind in sorted(SCANNER_KIND_DISPATCH)
        },
        "install_build_defaults": collect_defaults(),
        "routes": routes,
        "response_shape_divergences": collect_response_divergences(routes, operations),
        "engine_id_literals": literals,
        "operation_endpoints": operations,
    }


def render_markdown(inv: dict) -> str:
    lines = [
        "# Engine behavior parity inventory",
        "",
        "Derived by `scripts/parity-inventory.py`; regenerate and commit with any",
        "behavior migration. Inputs to the parity checklist for",
        "`docs/config-driven-engine-refactoring-plan.md` (Phase 1). See the JSON twin",
        "for machine consumption.",
        "",
        f"- Engines: {inv['engine_count']}",
        f"- `/api` routes: {len(inv['routes'])}",
        f"- Scanner kinds dispatched: {', '.join(inv['scanner_kinds'])}",
        f"- Shadow definitions loaded: {len(inv['shadow_definitions_loaded'])}",
        "",
        "## Hardcoded engine-ID literals (outside `backend/engines/` homes)",
        "",
        "The plan's Phase-1 checklist must consume every row: each file is a site",
        "the definition-driven code path must reach. Counts include both spellings",
        "where a specialized id nests in a family id (`sglang_v100` also matches",
        "`sglang`; `onecat` ids also match `vllm`).",
        "",
        "| Engine | Out-of-home literals | Files | Top files |",
        "| --- | --- | --- | --- |",
    ]
    for engine_id in sorted(ENGINE_REGISTRY):
        buckets = inv["engine_id_literals"].get(engine_id, {}).get("outside_engines", [])
        total = sum(entry["count"] for entry in buckets)
        top = ", ".join(f"`{e['file']}` ({e['count']})" for e in buckets[:4])
        lines.append(f"| `{engine_id}` | {total} | {len(buckets)} | {top} |")
    lines += [
        "",
        "## Per-engine operation endpoints",
        "",
        "Current route surface per lifecycle operation; the generic adapter must",
        "reproduce each endpoint family (or explicitly type a response-shape",
        "difference) before its legacy router can retire.",
        "",
        "| Engine | install | source | retry | sync | updates | activate | delete | status | logs | cancel | scan | other |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    op_keys = [
        "install_release",
        "install_source",
        "retry",
        "sync_source",
        "check_updates",
        "activate",
        "delete",
        "status",
        "logs",
        "cancel",
        "scan",
    ]
    for engine_id in ENGINE_REGISTRY:
        per_op = inv["operation_endpoints"].get(engine_id, {})
        cells = [str(len(per_op.get(op, []))) for op in op_keys]
        other = [op for op in per_op if op not in op_keys]
        cells.append(", ".join(other) if other else "0")
        lines.append(f"| `{engine_id}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Response-shape divergences (adapter unification inputs)",
        "",
        "The plan requires these shapes unified or explicitly typed per surface",
        "before legacy routers retire; today they differ per engine.",
        "",
        "| Engine | update-check | status | install | contract |",
        "| --- | --- | --- | --- | --- |",
    ]
    for engine_id in ENGINE_REGISTRY:
        shapes = inv["response_shape_divergences"].get(engine_id, {})
        cells = []
        for op in ("check_updates", "status", "install_release", "contract"):
            value = shapes.get(op)
            cells.append(f"`{value}`" if value else "—")
        lines.append(f"| `{engine_id}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Install/build default surfaces",
        "",
        "Persisted defaults each engine must keep producing after migration",
        "(full values in the JSON twin; keys listed here).",
        "",
        "| Engine | Default source | Keys |",
        "| --- | --- | --- |",
    ]
    for engine_id in ENGINE_REGISTRY:
        defaults = inv["install_build_defaults"].get(engine_id, {})
        keys = sorted((defaults.get("values") or {}).keys())
        shown = ", ".join(f"`{k}`" for k in keys) if keys else defaults.get("note", "—")
        source = defaults.get("source") or "—"
        lines.append(f"| `{engine_id}` | `{source}` | {shown} |")
    lines += [
        "",
        "## Route inventory",
        "",
        "| Path | Methods → success schema | Attributed engines |",
        "| --- | --- | --- |",
    ]
    for route in inv["routes"]:
        engines = ", ".join(route["engines"]) or "—"
        methods = ", ".join(f"{m} → {route['responses'][m]}" for m in route["methods"])
        lines.append(f"| `{route['path']}` | {methods} | {engines} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed files are stale")
    args = parser.parse_args()

    inventory = build_inventory()
    rendered_json = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    rendered_md = render_markdown(inventory)

    if args.check:
        for path, rendered in ((JSON_PATH, rendered_json), (MD_PATH, rendered_md)):
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != rendered:
                print(f"{path.relative_to(ROOT)} is stale; run scripts/parity-inventory.py")
                return 1
        print("engine parity inventory current")
        return 0

    JSON_PATH.write_text(rendered_json, encoding="utf-8", newline="\n")
    MD_PATH.write_text(rendered_md, encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)} and {MD_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
