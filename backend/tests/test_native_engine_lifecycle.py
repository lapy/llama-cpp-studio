"""Native llama-family engines on the generic /api/engines/{engine_id} router.

The verb overrides registered in ``routes/llama_versions.py`` and
``routes/unsloth_llama_versions.py`` delegate to the SAME route functions
the legacy prefixes run, so for identical payloads both surfaces must
return byte-identical responses — including error statuses, error detail
shapes, and the response_model filtering on check-updates. Verbs the
native engines genuinely lack (per-engine status, engine-wide remove,
Unsloth build settings / source builds) answer 405 — the capability
contract is pinned here rather than faked with invented bodies.
"""

from __future__ import annotations

import pytest

from backend.routes import engine_versions as ev
from backend.routes import llama_versions as lv
from backend.routes import unsloth_llama_versions as uv
from backend.services.upstream_versions import UpstreamRequestError


def test_native_registration_shape():
    import backend.main  # noqa: F401

    assert set(ev.LIFECYCLE["llama_cpp"].verbs) == {
        "check_updates",
        "get_build_settings",
        "save_build_settings",
        "install",
        "install_source",
        "cancel",
    }
    assert set(ev.LIFECYCLE["ik_llama"].verbs) == set(ev.LIFECYCLE["llama_cpp"].verbs)
    assert ev.LIFECYCLE["llama_cpp"].url_prefix == "/api/llama-versions"
    assert ev.LIFECYCLE["unsloth_llama"].url_prefix == "/api/unsloth-llama"
    assert set(ev.LIFECYCLE["unsloth_llama"].verbs) == {"check_updates", "install", "cancel"}
    for engine_id in ("llama_cpp", "ik_llama", "unsloth_llama"):
        assert ev.LIFECYCLE[engine_id].get_installer is None
        assert ev.LIFECYCLE[engine_id].update_source is None


# ── check-updates ─────────────────────────────────────────────────────


UPSTREAM_FULL = {
    "latest_release": {
        "tag_name": "b9999",
        "published_at": "2026-01-01T00:00:00Z",
        "html_url": "https://github.com/x/y/releases/tag/b9999",
    },
    "latest_commit": {"sha": "a" * 40, "html_url": "https://github.com/x/y/commit/aaaa"},
    "latest_version": "b9999",  # extra key the legacy response_model drops
}


@pytest.mark.parametrize(
    "engine_id,legacy_params,expected_source",
    [
        ("llama_cpp", {}, None),
        ("ik_llama", {"source": "ik_llama"}, "ik_llama"),
    ],
)
def test_check_updates_identical(client, monkeypatch, engine_id, legacy_params, expected_source):
    seen = []

    async def fake(source):
        seen.append(source)
        return dict(UPSTREAM_FULL)

    monkeypatch.setattr(lv, "check_engine_updates", fake)
    legacy = client.get("/api/llama-versions/check-updates", params=legacy_params)
    generic = client.get(f"/api/engines/{engine_id}/check-updates")
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()
    # response_model filtering is byte-replicated on the generic surface
    assert "latest_version" not in generic.json()
    assert seen == [expected_source, expected_source]


def test_check_updates_upstream_error_identical(client, monkeypatch):
    async def boom(source):
        raise UpstreamRequestError(403, "github said no")

    monkeypatch.setattr(lv, "check_engine_updates", boom)
    legacy = client.get("/api/llama-versions/check-updates")
    generic = client.get("/api/engines/llama_cpp/check-updates")
    assert legacy.status_code == generic.status_code == 429
    assert generic.json() == legacy.json()


def test_check_updates_generic_500_identical(client, monkeypatch):
    async def boom(source):
        raise ValueError("kaboom")

    monkeypatch.setattr(lv, "check_engine_updates", boom)
    legacy = client.get("/api/llama-versions/check-updates")
    generic = client.get("/api/engines/ik_llama/check-updates")
    assert legacy.status_code == generic.status_code == 500
    assert generic.json() == legacy.json() == {"detail": "Update check failed: kaboom"}


# ── build-settings ────────────────────────────────────────────────────


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_build_settings_get_identical(client, engine_id):
    legacy = client.get("/api/llama-versions/build-settings", params={"engine": engine_id})
    generic = client.get(f"/api/engines/{engine_id}/build-settings")
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_build_settings_put_identical(client, engine_id):
    payload = {"tracking_ref": "refs/tags/b1234", "repository_url": "https://example/fork"}
    legacy = client.put(
        "/api/llama-versions/build-settings", params={"engine": engine_id}, json=payload
    )
    generic = client.put(f"/api/engines/{engine_id}/build-settings", json=payload)
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()
    stored = client.get(f"/api/engines/{engine_id}/build-settings").json()
    assert stored["tracking_ref"] == "refs/tags/b1234"


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_build_settings_missing_body_identical(client, engine_id):
    legacy = client.put("/api/llama-versions/build-settings", params={"engine": engine_id})
    generic = client.put(f"/api/engines/{engine_id}/build-settings")
    assert legacy.status_code == generic.status_code == 422
    assert generic.json() == legacy.json()


# ── install / install-source ──────────────────────────────────────────


@pytest.fixture
def captured_builds(monkeypatch):
    captured = []

    async def fake_ref(engine):
        return ("v1.2.3", "tag")

    async def fake_build(**kwargs):
        captured.append(kwargs)
        return {"task_id": "t-9", "status": "scheduled"}

    monkeypatch.setattr(lv, "resolve_build_ref", fake_ref)
    monkeypatch.setattr(lv, "_schedule_source_build", fake_build)
    return captured


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_identical(client, captured_builds, engine_id):
    body = {"engine": engine_id, "version_suffix": "x"}
    legacy = client.post("/api/llama-versions/update", json=body)
    generic = client.post(f"/api/engines/{engine_id}/install", json=body)
    assert legacy.status_code == generic.status_code == 200, legacy.text
    assert generic.json() == legacy.json()
    assert captured_builds[0] == captured_builds[1]
    assert captured_builds[0]["auto_activate"] is True


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_missing_body_identical(client, captured_builds, engine_id):
    legacy = client.post("/api/llama-versions/update")
    generic = client.post(f"/api/engines/{engine_id}/install")
    assert legacy.status_code == generic.status_code == 422
    assert generic.json() == legacy.json()


# ── path-derived engine identity (generic surface only) ─────────────────
# Review finding #1 (2026-11-02): the generic router must derive the build
# target from the URL PATH; conflicting body fields are rejected. Legacy
# routes keep their permissive behavior (incl. the absent-engine default
# quirk) for compatibility.

_EXPECTED_SOURCE = {"llama_cpp": "llama.cpp", "ik_llama": "ik_llama.cpp"}
_OTHER_ENGINE = {"llama_cpp": "ik_llama", "ik_llama": "llama_cpp"}


def test_install_generic_derives_target_from_path(client, captured_builds):
    """Recorded legacy bytes: the .get("engine", "llama_cpp") default made
    an empty body install llama.cpp REGARDLESS of intent. The generic
    surface no longer shares that quirk (deliberate divergence)."""
    legacy = client.post("/api/llama-versions/update", json={})
    assert legacy.status_code == 200
    assert captured_builds[0]["repository_source"] == "llama.cpp"  # legacy quirk kept
    generic = client.post("/api/engines/ik_llama/install", json={})
    assert generic.status_code == 200
    assert captured_builds[1]["repository_source"] == "ik_llama.cpp"  # PATH wins


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_body_identity_conflict(client, captured_builds, engine_id):
    other = _OTHER_ENGINE[engine_id]
    response = client.post(f"/api/engines/{engine_id}/install", json={"engine": other})
    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            f"Engine identity conflict: path engine '{engine_id}' "
            f"does not accept body engine '{other}'"
        )
    }
    assert captured_builds == []


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_source_generic_derives_repository_source(client, captured_builds, engine_id):
    generic = client.post(
        f"/api/engines/{engine_id}/install-source", json={"commit_sha": "deadbeef"}
    )
    assert generic.status_code == 200
    # Absent repository_source: generic derives from the path (the legacy
    # default quirk would have built llama.cpp here, even for ik_llama).
    assert captured_builds[0]["repository_source"] == _EXPECTED_SOURCE[engine_id]


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_source_repository_source_conflict(client, captured_builds, engine_id):
    other_source = _EXPECTED_SOURCE[_OTHER_ENGINE[engine_id]]
    response = client.post(
        f"/api/engines/{engine_id}/install-source",
        json={"commit_sha": "deadbeef", "repository_source": other_source},
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            f"Engine identity conflict: path engine '{engine_id}' "
            f"does not accept body repository_source '{other_source}'"
        )
    }
    assert captured_builds == []


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_source_identical(client, captured_builds, engine_id):
    body = {
        "commit_sha": "deadbeef",
        # Consistent body+path keeps legacy/generic equality for BOTH engines.
        "repository_source": _EXPECTED_SOURCE[engine_id],
        "auto_activate": True,
    }
    legacy = client.post("/api/llama-versions/build-source", json=body)
    generic = client.post(f"/api/engines/{engine_id}/install-source", json=body)
    assert legacy.status_code == generic.status_code == 200, legacy.text
    assert generic.json() == legacy.json()
    assert captured_builds[0] == captured_builds[1]


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_install_source_bad_request_identical(client, captured_builds, engine_id):
    body = {"repository_source": _EXPECTED_SOURCE[engine_id]}  # missing commit_sha
    legacy = client.post("/api/llama-versions/build-source", json=body)
    generic = client.post(f"/api/engines/{engine_id}/install-source", json=body)
    assert legacy.status_code == generic.status_code == 400
    assert generic.json() == legacy.json() == {"detail": "commit_sha is required"}


# ── cancel ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama"])
def test_cancel_identical(client, monkeypatch, engine_id):
    calls = []

    class FakeBuildTaskManager:
        @staticmethod
        def cancel(task_id):
            calls.append(task_id)
            return {"ok": True, "terminated": True, "task_id": task_id}

    monkeypatch.setattr(lv, "BuildTaskManager", FakeBuildTaskManager)
    legacy = client.post("/api/llama-versions/build-cancel", json={"task_id": "t1"})
    generic = client.post(f"/api/engines/{engine_id}/cancel", json={"task_id": "t1"})
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()
    assert calls == ["t1", "t1"]
    legacy = client.post("/api/llama-versions/build-cancel", json={})
    generic = client.post(f"/api/engines/{engine_id}/cancel", json={})
    assert legacy.status_code == generic.status_code == 400
    assert generic.json() == legacy.json() == {"detail": "task_id is required"}


# ── unsloth ───────────────────────────────────────────────────────────


class StubUnslothManager:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.installed_with = None

    async def latest_published_tag(self):
        return "b7777"

    async def install_release(self, tag_name=None):
        if self.fail:
            raise RuntimeError("another operation is running")
        self.installed_with = tag_name
        return {"task_id": "u-1", "tag": tag_name, "message": "install started"}

    def cancel_task(self, task_id):
        return {"ok": True, "task_id": task_id}


def test_unsloth_check_updates_identical(client, monkeypatch):
    monkeypatch.setattr(uv, "get_unsloth_llama_manager", lambda: StubUnslothManager())
    legacy = client.get("/api/unsloth-llama/check-updates")
    generic = client.get("/api/engines/unsloth_llama/check-updates")
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()
    assert generic.json()["latest_version"] == "b7777"


def test_unsloth_check_updates_error_identical(client, monkeypatch):
    class Boom(StubUnslothManager):
        async def latest_published_tag(self):
            raise ValueError("nope")

    monkeypatch.setattr(uv, "get_unsloth_llama_manager", lambda: Boom())
    legacy = client.get("/api/unsloth-llama/check-updates")
    generic = client.get("/api/engines/unsloth_llama/check-updates")
    assert legacy.status_code == generic.status_code == 500
    assert generic.json() == legacy.json()


def test_unsloth_install_identical(client, monkeypatch):
    stub = StubUnslothManager()
    monkeypatch.setattr(uv, "get_unsloth_llama_manager", lambda: stub)
    legacy = client.post("/api/unsloth-llama/install", json={"tag_name": "b7777"})
    generic = client.post("/api/engines/unsloth_llama/install", json={"tag_name": "b7777"})
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()
    assert stub.installed_with == "b7777"


def test_unsloth_install_conflict_identical(client, monkeypatch):
    monkeypatch.setattr(uv, "get_unsloth_llama_manager", lambda: StubUnslothManager(fail=True))
    legacy = client.post("/api/unsloth-llama/install", json={})
    generic = client.post("/api/engines/unsloth_llama/install", json={})
    assert legacy.status_code == generic.status_code == 409
    assert generic.json() == legacy.json()


def test_unsloth_cancel_identical(client, monkeypatch):
    monkeypatch.setattr(uv, "get_unsloth_llama_manager", lambda: StubUnslothManager())
    legacy = client.post("/api/unsloth-llama/cancel", json={"task_id": "u-9"})
    generic = client.post("/api/engines/unsloth_llama/cancel", json={"task_id": "u-9"})
    assert legacy.status_code == generic.status_code == 200
    assert generic.json() == legacy.json()
    legacy = client.post("/api/unsloth-llama/cancel", json={})
    generic = client.post("/api/engines/unsloth_llama/cancel", json={})
    assert legacy.status_code == generic.status_code == 400
    assert generic.json() == legacy.json() == {"detail": "task_id is required"}


# ── capability contract (405) ─────────────────────────────────────────


@pytest.mark.parametrize(
    "engine_id,verb,method",
    [
        ("llama_cpp", "status", "get"),
        ("ik_llama", "status", "get"),
        ("unsloth_llama", "status", "get"),
        ("unsloth_llama", "build-settings", "get"),
        ("unsloth_llama", "install-source", "post"),
    ],
)
def test_unsupported_verb_405(client, engine_id, verb, method):
    url = f"/api/engines/{engine_id}/{verb}"
    response = client.post(url, json={}) if method == "post" else client.get(url)
    assert response.status_code == 405
    assert response.json() == {"detail": f"{verb} is not available for engine '{engine_id}'"}


@pytest.mark.parametrize("engine_id", ["llama_cpp", "ik_llama", "unsloth_llama"])
def test_unsupported_remove_405(client, engine_id):
    response = client.post(f"/api/engines/{engine_id}/remove", json={})
    assert response.status_code == 405
    assert response.json() == {"detail": f"remove is not available for engine '{engine_id}'"}
