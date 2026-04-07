"""
Guardrails for the models router refactor (architecture masterplan).

These checks target recurring merge regressions: a second copy of download/bundle
handlers was reintroduced several times, breaking tests and duplicating OpenAPI IDs.
"""

from pathlib import Path


def _models_py_text() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "routes" / "models.py").read_text(encoding="utf-8")


def test_models_router_single_bundle_endpoints():
    text = _models_py_text()
    assert text.count('@router.post("/safetensors/download-bundle")') == 1
    assert text.count('@router.post("/gguf/download-bundle")') == 1


def test_models_router_does_not_inline_download_tasks():
    """Download orchestration lives in backend.services.model_downloads."""
    text = _models_py_text()
    assert "async def download_model_task(" not in text
    assert "async def download_safetensors_bundle_task(" not in text
    assert "async def download_gguf_bundle_task(" not in text


def test_models_router_does_not_own_download_lock():
    text = _models_py_text()
    assert "download_lock" not in text
