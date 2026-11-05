"""Architecture audit for the converted engine tree (no dead compatibility logic)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _tree(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _class_methods(node: ast.ClassDef) -> set[str]:
    return {
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_no_runtime_branch_reads_the_exporter_tier():
    import backend.engines.registry as registry

    assert not hasattr(registry, "GENERATED_ENGINE_IDS")
    for relative in ("backend/routes/engines.py", "backend/routes/engine_versions.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "GENERATED_ENGINE_IDS" not in source, relative


def test_venv_engine_modules_and_installer_map_are_gone():
    for relative in (
        "backend/engines/lmdeploy",
        "backend/engines/vllm",
        "backend/engines/sglang",
    ):
        assert not list((ROOT / relative).glob("*.py")), relative
    assert not (ROOT / "backend/engines/unsloth_llama/installer.py").exists()
    import backend.engines.adapters as adapters

    assert not hasattr(adapters, "_INSTALLERS")
    assert not hasattr(adapters, "_NATIVE_HOSTS")

def test_compatibility_route_modules_are_thin_delegates():
    for relative in (
        "backend/routes/lmdeploy_versions.py",
        "backend/routes/vllm_versions.py",
        "backend/routes/onecat_vllm_versions.py",
        "backend/routes/sglang_versions.py",
    ):
        tree = _tree(relative)
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ], relative
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "python_engine_router" not in source, relative
        assert "definition_compat_router" in source, relative


def test_legacy_scan_adapters_are_gone():
    import backend.engines.scan.scanner as scanner

    for name in (
        "scan_sglang_version",
        "scan_onecat_vllm_version",
        "_scan_sglang_runtime",
    ):
        assert not hasattr(scanner, name), name
