"""Studio fields and catalog API payload shape."""

from backend.engine_param_catalog import registry_payload_from_entry
from backend.studio_engine_fields import studio_sections_for_engine


def test_studio_sections_llama_has_model_alias():
    sec = studio_sections_for_engine("llama_cpp")
    assert len(sec) == 1
    keys = {p["key"] for p in sec[0]["params"]}
    assert "model_alias" in keys


def test_studio_sections_lmdeploy_empty():
    assert studio_sections_for_engine("lmdeploy") == []


def test_registry_payload_includes_sections():
    entry = {
        "scanned_at": "2025-01-01T00:00:00Z",
        "scan_error": None,
        "sections": [
            {
                "id": "general",
                "label": "General",
                "params": [
                    {
                        "key": "ctx_size",
                        "label": "Ctx size",
                        "type": "int",
                        "flags": ["--ctx-size"],
                        "description": "x",
                    }
                ],
            }
        ],
    }
    studio = studio_sections_for_engine("llama_cpp")
    data = registry_payload_from_entry("llama_cpp", entry, studio, has_active_engine=True)
    assert data["engine"] == "llama_cpp"
    assert len(data["sections"]) >= 2
    all_keys = {p["key"] for s in data["sections"] for p in s.get("params") or []}
    assert "model_alias" in all_keys
    assert "ctx_size" in all_keys
    assert data["scan_pending"] is False


def test_registry_payload_scan_pending_without_entry():
    studio = studio_sections_for_engine("llama_cpp")
    data = registry_payload_from_entry("llama_cpp", None, studio, has_active_engine=True)
    assert data["scan_pending"] is True
