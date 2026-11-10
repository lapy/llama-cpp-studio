"""YAML campaign S1 — definition tiers, canonical digests, install ledger.

S1 is behavior-neutral groundwork: acceptance policy is UNCHANGED (the
startup refusal for non-generated engines still fires; every existing
byte-equality pin passing unchanged IS the baseline for that claim).
These pins cover the new additive surfaces only:

* ``EngineDefinition.definition_digest`` — canonical sha256 over the FULL
  validated definition (recipes included; host-dependent ``source_path``
  excluded).
* ``definition_tier`` — classification helper (no policy attached yet).
* The install-snapshot ledger in ``install_snapshots.yaml`` (rows stay
  scan-derived; the ledger is the only new bookkeeping).
* ``_record_install_snapshot`` wiring on the generic install dispatchers
  (response bytes untouched — the router/native equality pins own that).
"""

from __future__ import annotations

import copy

import pytest

from backend.data_store import DataStore
from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.definitions.schema import validate_definition
from backend.engines.definitions.startup_check import (
    definition_tier,
    verify_engine_definitions,
)
from backend.engines.registry import SHIPPED_ENGINE_IDS


def _raw(**overrides):
    doc = {
        "schema_version": 2,
        "id": "digesty",
        "label": "Digesty",
        "repository_source": "Digesty",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary"],
        "artifact_formats": ["gguf"],
        "package_kinds": ["binary"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "recipes": {
            "build": [
                {"id": "compose", "uses": "env.compose", "produces": ["env"]},
                {
                    "id": "wipe",
                    "uses": "fs.remove_tree",
                    "requires": ["input.root"],
                    "cleanup": True,
                },
            ]
        },
    }
    doc.update(overrides)
    return doc


def test_digest_is_canonical_and_source_path_independent():
    first = validate_definition(_raw(), source="/host/a/digesty.yaml")
    second = validate_definition(copy.deepcopy(_raw()), source="Z:\\other\\digesty.yaml")
    digest = first.definition_digest
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert digest == second.definition_digest  # host paths must not leak in


def test_digest_covers_the_effective_recipe_not_just_registry_identity():
    base = validate_definition(_raw(), source="digesty.yaml")
    # identical registry identity (recipes are NOT part of to_api_dict) but
    # a recipe step gains an accepted exit code — the snapshot must freeze
    # the effective recipe, so the digest has to move.
    tweaked = copy.deepcopy(_raw())
    tweaked["recipes"]["build"][1]["accepted_exit_codes"] = [0, 2]
    other = validate_definition(tweaked, source="digesty.yaml")
    assert base.to_api_dict() == other.to_api_dict()
    assert base.definition_digest != other.definition_digest


def test_digest_includes_referenced_transform_asset_content(monkeypatch):
    """An asset-only edit must change installation identity.

    The recipe stores only the transform NAME; the digest has to include the
    asset's effective content (operations/marker/file), not just the reference.
    """
    from dataclasses import replace

    from backend.engines.definitions import transforms

    definition = load_default_engine_definitions().definitions["sglang_v100"]
    baseline = definition.definition_digest
    original = transforms.load_transform

    def shifted(name, **kwargs):
        asset = original(name, **kwargs)
        if name == "sglang_v100/pyproject":
            return replace(asset, marker="changed-by-test")
        return asset

    monkeypatch.setattr(transforms, "load_transform", shifted)
    assert definition.definition_digest != baseline


def test_digests_are_unique_across_all_shipped_definitions():
    definitions = load_default_engine_definitions().definitions
    digests = {engine_id: d.definition_digest for engine_id, d in definitions.items()}
    # every shipped engine
    assert set(digests) == set(SHIPPED_ENGINE_IDS)
    assert len(set(digests.values())) == len(digests)


def test_tier_classification_and_startup_acceptance_owns_registration():
    assert all(definition_tier(engine_id) == "definition" for engine_id in SHIPPED_ENGINE_IDS)
    assert definition_tier("no_such_engine") == "authoritative"
    from backend.engines import registry as registry_module

    dict_copy = dict(registry_module.ENGINE_REGISTRY)
    sets = {
        name: set(getattr(registry_module, name))
        for name in (
            "VALID_ENGINE_IDS",
            "EMBEDDINGS_ENGINE_IDS",
            "NATIVE_ENGINE_IDS",
            "GGUF_ENGINE_IDS",
            "HF_SNAPSHOT_ENGINE_IDS",
        )
    }
    try:
        # Every shipped engine is registry-equivalent (the converted ones are
        # loaded by the import-time bootstrap).
        assert verify_engine_definitions() == len(SHIPPED_ENGINE_IDS)
    finally:
        registry_module.ENGINE_REGISTRY.clear()
        registry_module.ENGINE_REGISTRY.update(dict_copy)
        for name, snapshot in sets.items():
            target = getattr(registry_module, name)
            target.clear()
            target.update(snapshot)


def _store(tmp_path) -> DataStore:
    return DataStore(config_dir=str(tmp_path / "config"))


def test_install_snapshot_ledger_roundtrip_and_input_hygiene(tmp_path):
    store = _store(tmp_path)
    row = store.record_install_snapshot(
        "lmdeploy",
        definition_digest="sha256:" + "a" * 64,
        recipe_version="schema-v2",
        resolved_inputs={
            "version": "0.9.0",
            "confirm_operation_id": "op-1",  # tokens must not persist
            "confirm_state": "secret",
            "_internal": "nope",
        },
    )
    assert row["engine"] == "lmdeploy"
    assert row["resolved_inputs"] == {"version": "0.9.0"}
    assert isinstance(row["recorded_at"], int)

    store.record_install_snapshot("sglang", definition_digest="sha256:" + "b" * 64)

    assert [r["engine"] for r in store.list_install_snapshots("lmdeploy")] == ["lmdeploy"]
    assert len(store.list_install_snapshots()) == 2
    # durable: a fresh store instance reads the same ledger
    assert len(_store(tmp_path).list_install_snapshots()) == 2


def test_late_resolved_release_tag_updates_the_pending_row(tmp_path):
    store = _store(tmp_path)
    store.record_install_snapshot(
        "lmdeploy",
        definition_digest="sha256:" + "a" * 64,
        version="pending-v1",
        resolved_inputs={"install_type": "pip", "requested_version": ""},
    )
    store.record_install_snapshot(
        "lmdeploy",
        definition_digest="sha256:" + "a" * 64,
        version="pending-v1",
        phase="published",
        resolved_inputs={"install_type": "pip", "package_version": "0.9.0"},
    )

    updated = store.update_install_snapshot(
        "lmdeploy",
        version="pending-v1",
        resolved_inputs={"release_tag": "v0.9.0", "package_version": "v0.9.0"},
    )
    assert updated["phase"] == "scheduled"
    assert updated["resolved_inputs"]["release_tag"] == "v0.9.0"
    assert updated["resolved_inputs"]["package_version"] == "v0.9.0"

    rows = store.list_install_snapshots("lmdeploy")
    assert len(rows) == 2  # update-in-place, no duplicate pending row
    assert rows[0]["resolved_inputs"]["release_tag"] == "v0.9.0"
    # The published row is a separate historical fact and stays untouched.
    assert "release_tag" not in rows[1]["resolved_inputs"]


def test_removed_version_prunes_its_snapshots_only(tmp_path):
    store = _store(tmp_path)
    store.record_install_snapshot("llama_cpp", definition_digest="sha256:" + "a" * 64, version="v1")
    store.record_install_snapshot("llama_cpp", definition_digest="sha256:" + "b" * 64, version="v2")

    assert store.delete_install_snapshots("llama_cpp", "v1") == 1
    assert [row["version"] for row in store.list_install_snapshots("llama_cpp")] == ["v2"]
    # A missing association is a no-op, not an error.
    assert store.delete_install_snapshots("llama_cpp", "absent") == 0


@pytest.mark.asyncio
async def test_generic_install_records_snapshot_without_touching_the_response(
    tmp_path, monkeypatch
):
    from backend.routes import engine_versions as ev

    store = _store(tmp_path)
    monkeypatch.setattr(ev, "get_store", lambda: store)

    calls = []

    def fake_lc(payload=None) -> dict:
        calls.append(1)
        return {"task_id": "t-snap", "status": "scheduled"}

    lc = ev.EngineLifecycle(engine_id="lmdeploy", url_prefix="/api/x", verbs={"install": fake_lc})
    body = {"version": "0.9.0", "confirm_state": "secret"}

    result = await ev.lifecycle_install(lc, body)

    assert result == {"task_id": "t-snap", "status": "scheduled"}  # response untouched
    assert calls == [1]
    rows = store.list_install_snapshots("lmdeploy")
    assert len(rows) == 1
    real_digest = load_default_engine_definitions().definitions["lmdeploy"].definition_digest
    assert rows[0]["definition_digest"] == real_digest
    assert rows[0]["recipe_version"] == "schema-v2"
    assert rows[0]["phase"] == "scheduled" and rows[0]["version"] == ""
    # Resolved dispatch inputs: the requested version PLUS the defaults the
    # install actually used (a retry must be reproducible from the record).
    resolved = rows[0]["resolved_inputs"]
    assert resolved["version"] == "0.9.0"
    assert resolved["source_branch"] == "main"
    assert resolved["source_repo"].endswith(".git")
    assert "pip_version" in resolved
    # verb bodies are forwarded verbatim (hygiene lives in the ledger only)
    assert body == {"version": "0.9.0", "confirm_state": "secret"}


@pytest.mark.asyncio
async def test_snapshot_bookkeeping_failure_is_inert_to_the_install(tmp_path, monkeypatch):
    from backend.routes import engine_versions as ev

    class ExplodingStore:
        def record_install_snapshot(self, *args, **kwargs):
            raise RuntimeError("disk on fire")

    monkeypatch.setattr(ev, "get_store", lambda: ExplodingStore())

    lc = ev.EngineLifecycle(
        engine_id="lmdeploy",
        url_prefix="/api/x",
        verbs={"install": lambda payload=None: {"task_id": "t-ok"}},
    )
    assert await ev.lifecycle_install(lc, {}) == {"task_id": "t-ok"}
