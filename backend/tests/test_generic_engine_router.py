"""Generic /api/engines/{engine_id} router == legacy per-engine surfaces.

The delegation promise (docs/engine-surface-contract-matrix.md §7): both
surfaces run the SAME lifecycle_* callables, so for identical payloads they
must return byte-identical responses — including error statuses and detail
shapes. Equality is tested per verb with stub installers, so any future
divergence between the surfaces fails HERE, not in the frontend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.routes import engine_lifecycle
from backend.routes import engine_versions as ev


VENV_ENGINES = {
    "lmdeploy": "/api/lmdeploy",
    "vllm": "/api/vllm",
    "sglang": "/api/sglang",
    "sglang_v100": "/api/sglang-v100",
    "sglang_sxm2": "/api/sglang-sxm2",
    "1cat_vllm": "/api/1cat-vllm",
}

PYPI_PAYLOAD = {"info": {"version": "9.9.9"}, "releases": {"9.9.8": [], "9.9.9": []}}
COMMIT_PAYLOAD = {"sha": "f" * 40, "html_url": "https://github.com/x/y/commit/ffff"}
RELEASES_PAYLOAD = [{"tag_name": "v2.0.0", "prerelease": False, "name": "two"}]

PAYLOADS = {
    "lmdeploy": PYPI_PAYLOAD,
    "vllm": PYPI_PAYLOAD,
    "sglang": PYPI_PAYLOAD,
    "sglang_v100": COMMIT_PAYLOAD,
    "sglang_sxm2": COMMIT_PAYLOAD,
    "1cat_vllm": RELEASES_PAYLOAD,
}


class StubInstaller:
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.calls: list[tuple[str, dict]] = []

    def status(self):
        return {"engine": self.engine_id, "installed": True, "version": "9.9.9"}

    async def install_release(self, **kwargs):
        self.calls.append(("release", kwargs))
        return {"message": f"{self.engine_id} installation started", "task_id": "t-1"}

    async def install_from_source(self, **kwargs):
        self.calls.append(("source", kwargs))
        return {
            "message": f"{self.engine_id} source installation started",
            "task_id": "t-2",
            "repo": kwargs.get("repo_url"),
            "branch": kwargs.get("branch"),
        }

    async def remove(self, **kwargs):
        self.calls.append(("remove", kwargs))
        return {"message": f"{self.engine_id} removal started", "task_id": "t-3"}

    def cancel_task(self, task_id):
        self.calls.append(("cancel", {"task_id": task_id}))
        return {
            "ok": True,
            "terminated": False,
            "message": "Cancellation was requested. Termination has not been verified.",
            "task_id": task_id,
        }


def _stub(monkeypatch, engine_id: str) -> StubInstaller:
    from dataclasses import replace

    stub = StubInstaller(engine_id)
    # Every converted engine's compatibility prefix is now a thin delegate over
    # the shared LIFECYCLE record; inject the stub at that one seam.
    monkeypatch.setitem(
        ev.LIFECYCLE,
        engine_id,
        replace(ev.LIFECYCLE[engine_id], get_installer=lambda: stub),
    )
    return stub


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _fake_httpx(payload):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            if isinstance(payload, Exception):
                raise payload
            return _FakeResponse(payload)

    return SimpleNamespace(AsyncClient=_FakeClient)


def test_registry_covers_venv_and_native_engines():
    # Importing the app runs the five call-sites, populating LIFECYCLE.
    # The three native llama-family engines joined deliberately (verb
    # overrides, no installer machinery — see
    # tests/test_native_engine_lifecycle.py).
    import backend.main  # noqa: F401

    assert set(VENV_ENGINES) | {"llama_cpp", "ik_llama", "unsloth_llama"} <= set(ev.LIFECYCLE)
    # S4/R3: the authoritative fixtures may already be registered in this
    # process (lifespan ran in some earlier test) — bounded, order-proof.
    assert set(ev.LIFECYCLE) <= set(VENV_ENGINES) | {
        "llama_cpp",
        "ik_llama",
        "unsloth_llama",
        "launchcraft",
        "launchforge",
        "yamlcraft",
    }
    for engine_id in VENV_ENGINES:
        assert ev.LIFECYCLE[engine_id].verbs == {}  # venv runs the shared machinery
    assert ev.LIFECYCLE["sglang_v100"].update_source.kind == "github_commit"
    assert ev.LIFECYCLE["sglang_v100"].prefer_source_install is True
    assert ev.LIFECYCLE["sglang_sxm2"].update_source.kind == "github_commit"
    assert ev.LIFECYCLE["sglang_sxm2"].prefer_source_install is True
    assert ev.LIFECYCLE["1cat_vllm"].update_source.kind == "github_releases"
    assert ev.LIFECYCLE["1cat_vllm"].version_setting_key == "release_version"
    for engine in ("lmdeploy", "vllm", "sglang"):
        assert ev.LIFECYCLE[engine].update_source.kind == "pypi"
    # The per-engine status fallback overlays the factory shipped (matrix §3)
    # ride on the records, so both surfaces emit identical fallbacks.
    assert set(ev.LIFECYCLE["lmdeploy"].status_fallback) == {
        "installed_at",
        "removed_at",
        "log_path",
    }
    assert set(ev.LIFECYCLE["1cat_vllm"].status_fallback) == {
        "installed_at",
        "removed_at",
        "log_path",
        "release_tag",
    }
    assert ev.LIFECYCLE["vllm"].status_fallback is None


def test_generic_router_imports_no_engine_id_literals():
    # The delegation must dispatch on the registry, not on engine ids.
    import inspect

    source = inspect.getsource(engine_lifecycle)
    for needle in ("lmdeploy", "1cat_vllm", "sglang_v100", '"vllm"', '"sglang"'):
        assert needle not in source


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_status_identical_on_both_surfaces(client, monkeypatch, engine_id):
    _stub(monkeypatch, engine_id)
    legacy = client.get(f"{VENV_ENGINES[engine_id]}/status")
    generic = client.get(f"/api/engines/{engine_id}/status")
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_status_fallback_identical(client, monkeypatch, engine_id):
    def boom():
        raise RuntimeError("installer exploded")

    stub = _stub(monkeypatch, engine_id)
    monkeypatch.setattr(stub, "status", boom)
    legacy = client.get(f"{VENV_ENGINES[engine_id]}/status")
    generic = client.get(f"/api/engines/{engine_id}/status")
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()
    assert generic.json()["last_error"] == "installer exploded"


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_check_updates_identical(client, monkeypatch, engine_id):
    monkeypatch.setattr(ev, "httpx", _fake_httpx(PAYLOADS[engine_id]))
    legacy = client.get(f"{VENV_ENGINES[engine_id]}/check-updates")
    generic = client.get(f"/api/engines/{engine_id}/check-updates")
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()


def test_check_updates_failure_status_identical_per_engine(client, monkeypatch):
    # The per-engine divergence the matrix preserves: 502 (vllm) vs 500 (1cat).
    monkeypatch.setattr(ev, "httpx", _fake_httpx(RuntimeError("upstream down")))
    for engine_id, expected in (("vllm", 502), ("1cat_vllm", 500)):
        legacy = client.get(f"{VENV_ENGINES[engine_id]}/check-updates")
        generic = client.get(f"/api/engines/{engine_id}/check-updates")
        assert generic.status_code == legacy.status_code == expected
        assert generic.json() == legacy.json()


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_install_identical_including_prefer_source(client, monkeypatch, engine_id):
    stub = _stub(monkeypatch, engine_id)
    legacy = client.post(f"{VENV_ENGINES[engine_id]}/install", json={})
    generic = client.post(f"/api/engines/{engine_id}/install", json={})
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()
    # sglang_v100 and sglang_sxm2 (prefer_source_install) must schedule SOURCE
    # on both surfaces; everyone else schedules the release install.
    expected_verb = "source" if engine_id in {"sglang_v100", "sglang_sxm2"} else "release"
    assert [verb for verb, _ in stub.calls] == [expected_verb] * 2
    assert stub.calls[0][1] == stub.calls[1][1]


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_install_source_identical(client, monkeypatch, engine_id):
    _stub(monkeypatch, engine_id)
    legacy = client.post(
        f"{VENV_ENGINES[engine_id]}/install-source", json={"repo_url": "https://x/y"}
    )
    generic = client.post(
        f"/api/engines/{engine_id}/install-source", json={"repo_url": "https://x/y"}
    )
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_remove_identical_forwards_retire(client, monkeypatch, engine_id):
    stub = _stub(monkeypatch, engine_id)
    legacy = client.post(
        f"{VENV_ENGINES[engine_id]}/remove", json={"retire_launch_references": True}
    )
    generic = client.post(
        f"/api/engines/{engine_id}/remove", json={"retire_launch_references": True}
    )
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()
    assert stub.calls == [("remove", {"retire_references": True})] * 2


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_cancel_identical(client, monkeypatch, engine_id):
    _stub(monkeypatch, engine_id)
    legacy = client.post(f"{VENV_ENGINES[engine_id]}/cancel", json={"task_id": "t-9"})
    generic = client.post(f"/api/engines/{engine_id}/cancel", json={"task_id": "t-9"})
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()


def test_cancel_missing_task_id_identical_400(client, monkeypatch):
    _stub(monkeypatch, "vllm")
    legacy = client.post("/api/vllm/cancel", json={})
    generic = client.post("/api/engines/vllm/cancel", json={})
    assert generic.status_code == legacy.status_code == 400
    assert generic.json() == legacy.json() == {"detail": "task_id is required"}


@pytest.mark.parametrize("engine_id", sorted(VENV_ENGINES))
def test_build_settings_round_trip_identical(client, engine_id):
    legacy = client.get(f"{VENV_ENGINES[engine_id]}/build-settings")
    generic = client.get(f"/api/engines/{engine_id}/build-settings")
    assert generic.status_code == legacy.status_code == 200
    assert generic.json() == legacy.json()
    saved = client.put(f"/api/engines/{engine_id}/build-settings", json={"source_branch": "dev-x"})
    assert saved.status_code == 200
    assert saved.json()["source_branch"] == "dev-x"
    legacy_after = client.get(f"{VENV_ENGINES[engine_id]}/build-settings")
    assert legacy_after.json() == saved.json()


@pytest.mark.parametrize(
    ("method", "engine_id", "verb"),
    [
        ("get", "nope", "status"),
        ("get", "audio_cpp", "status"),
        ("post", "audio_cpp", "install"),
    ],
)
def test_unknown_or_non_venv_engines_404(client, method, engine_id, verb):
    # Unregistered engines 404. The native llama family is registered now
    # and answers 405 for its unimplemented verbs (pinned in
    # tests/test_native_engine_lifecycle.py); audio stays on its own router.
    response = getattr(client, method)(f"/api/engines/{engine_id}/{verb}")
    assert response.status_code == 404
    assert response.json() == {"detail": f"Unknown engine: {engine_id}"}


def test_openapi_documents_typed_contracts(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for verb in (
        "check-updates",
        "status",
        "build-settings",
        "install",
        "install-source",
        "remove",
        "cancel",
    ):
        assert f"/api/engines/{{engine_id}}/{verb}" in paths

    def ref(path, method):
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        return schema.get("$ref", "")

    assert ref("/api/engines/{engine_id}/status", "get").endswith("VenvEngineStatus")
    assert ref("/api/engines/{engine_id}/install", "post").endswith("InstallAccepted")
    assert ref("/api/engines/{engine_id}/cancel", "post").endswith("CancelAccepted")


def test_legacy_factory_paths_unchanged_in_openapi(client):
    # The refactor preserved the legacy surface: prefixes, operation ids.
    spec = client.get("/openapi.json").json()
    for engine, op in (
        ("/api/lmdeploy/status", "lmdeploy_status"),
        ("/api/vllm/remove", "vllm_remove"),
    ):
        assert engine in spec["paths"]
        assert spec["paths"][engine]["get" if "status" in engine else "post"]["operationId"] == op
