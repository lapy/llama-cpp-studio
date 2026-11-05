"""A version installed for a YAML/definition-registered engine must be listed.

The frontend builds its per-engine cards from `/api/llama-versions`; before
this, the endpoint enumerated only the original hardcoded engines, so a
successful YAML-engine install produced no card entry at all.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def _yaml_engine_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


def test_definition_registered_engine_versions_are_listed(client, _yaml_engine_store, monkeypatch):
    """The route's fallback branch lists engines defined only in YAML.

    The definitions registry is stubbed so the test does not depend on which
    fixture definitions the global loader cache happens to hold.
    """
    from types import SimpleNamespace

    import backend.routes.llama_versions as route

    class _Definition:
        repository_source = "Yaml Test"

    class _Registry:
        def ids(self):
            return ["yamltest"]

        def get(self, engine_id):
            return _Definition() if engine_id == "yamltest" else None

    monkeypatch.setattr(route, "load_default_engine_definitions", lambda: _Registry())
    _yaml_engine_store.add_engine_version(
        "yamltest",
        {
            "version": "v-test",
            "install_type": "pip",
            "venv_path": "/opt/venv",
            "install_dir": "/opt/install",
            "installed_at": "2026-01-01T00:00:00Z",
        },
    )

    response = client.get("/api/llama-versions")

    assert response.status_code == 200
    ids = {row.get("id") for row in response.json()}
    assert "yamltest:v-test" in ids
