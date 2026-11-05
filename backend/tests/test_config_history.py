import os

import pytest

from backend.config_history import (
    ConfigHistoryError,
    config_history_diff,
    list_config_history,
    restore_config_history_item,
)
from backend.data_store import DataStore
from backend.store_io import StoreDurabilityError


def test_mutations_create_private_redacted_history(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.update_settings({"huggingface_token": "hf_secret", "proxy_port": 3000})
    store.update_settings({"huggingface_token": "hf_new", "proxy_port": 4000})

    entries = list_config_history(store)
    assert len(entries) == 2
    assert entries[0]["document"] == "settings.yaml"
    comparison = config_history_diff(store, entries[0]["id"])
    changes = {item["path"]: item for item in comparison["changes"]}
    assert changes["huggingface_token"] == {
        "path": "huggingface_token",
        "before": "[redacted]",
        "after": "[redacted]",
    }
    assert changes["proxy_port"]["before"] == "3000"
    assert changes["proxy_port"]["after"] == "4000"
    history_dir = tmp_path / "config" / "history"
    assert os.stat(history_dir).st_mode & 0o777 == 0o700
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in history_dir.iterdir())


def test_restore_one_preference_is_revision_bound_and_keeps_credentials(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.update_settings({"huggingface_token": "hf_keep", "proxy_port": 3000})
    store.update_settings({"huggingface_token": "hf_keep", "proxy_port": 4000})
    entry = list_config_history(store)[0]
    comparison = config_history_diff(store, entry["id"])

    result = restore_config_history_item(
        store,
        entry["id"],
        kind="preference",
        item_id="proxy_port",
        expected_revision=comparison["current_revision"],
    )
    assert result["outcome"] == "completed"
    assert store.get_settings()["proxy_port"] == 3000
    assert store.get_settings()["huggingface_token"] == "hf_keep"
    with pytest.raises(ConfigHistoryError) as stale:
        restore_config_history_item(
            store,
            entry["id"],
            kind="preference",
            item_id="proxy_port",
            expected_revision=comparison["current_revision"],
        )
    assert stale.value.code == "HISTORY_STALE"


def test_history_failure_prevents_the_configuration_write(tmp_path, monkeypatch):
    from backend import config_history

    store = DataStore(config_dir=str(tmp_path / "config"))
    previous = store.get_settings()["proxy_port"]

    def fail_history(_path, _entry):
        raise StoreDurabilityError("history failed", phase="history", committed=False)

    monkeypatch.setattr(config_history, "_atomic_write", fail_history)
    with pytest.raises(StoreDurabilityError):
        store.update_settings({"proxy_port": 9876})
    assert store.get_settings()["proxy_port"] == previous


def test_history_api_lists_diffs_and_restores_one_item(client, monkeypatch, tmp_path):
    from backend import data_store

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    store.update_settings({"proxy_port": 3000})
    store.update_settings({"proxy_port": 4000})

    entries = client.get("/api/config-history")
    assert entries.status_code == 200
    entry_id = entries.json()[0]["id"]
    comparison = client.get(f"/api/config-history/{entry_id}")
    assert comparison.status_code == 200
    payload = comparison.json()
    restored = client.post(
        f"/api/config-history/{entry_id}/restore",
        json={
            "kind": "preference",
            "item_id": "proxy_port",
            "expected_revision": payload["current_revision"],
        },
    )
    assert restored.status_code == 200
    assert store.get_settings()["proxy_port"] == 3000
