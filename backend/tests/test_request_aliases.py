"""Named request overrides route to a real model without a second process identity."""

import json
from types import SimpleNamespace

import pytest

from backend.data_store import collect_model_aliases, find_model_name_conflicts
from backend.models.config import merge_model_config_put, normalize_model_config
from backend.proxy.commands import request_filters_and_aliases
from backend.proxy.studio.dataplane import _prepare_body
from backend.proxy.studio.manager import StudioProxyManager


def config():
    return {
        "engine": "llama_cpp",
        "model_alias": "assistant",
        "set_params_by_id": [
            {
                "sub_id": "thinking",
                "params": {"chat_template_kwargs.enable_thinking": True, "thinking_effort": "high"},
            },
            {
                "sub_id": "fast",
                "params": {"chat_template_kwargs.enable_thinking": False, "thinking_effort": "low"},
            },
        ],
    }


def test_named_overrides_share_model_and_leave_other_requests_unchanged():
    settings = config()
    filters, aliases = request_filters_and_aliases(stable_id="real", config=settings)
    assert aliases == ["assistant", "assistant:thinking", "assistant:fast"]
    manager = object.__new__(StudioProxyManager)
    manager._compiled_models = lambda: [
        (
            None,
            SimpleNamespace(
                proxy=SimpleNamespace(
                    model_id="real",
                    catalog_id="catalog",
                    aliases=aliases,
                    use_model_name="real",
                    filters=filters,
                    engine_id="llama_cpp",
                )
            ),
        )
    ]
    manager._routing = SimpleNamespace(default_model="", revision="routing")
    original = {
        "chat_template_kwargs": {"other": "preserved", "enable_thinking": None},
        "thinking_effort": "medium",
    }
    for name, thinking, effort in [
        ("assistant:thinking", True, "high"),
        ("assistant:fast", False, "low"),
        ("assistant", None, "medium"),
        ("real", None, "medium"),
    ]:
        resolved = manager._resolve(name)
        assert resolved.model_id == "real"
        body = json.loads(
            _prepare_body(json.dumps({"model": name, **original}).encode(), resolved, name)
        )
        assert body["model"] == "real"
        assert body["chat_template_kwargs"] == {"other": "preserved", "enable_thinking": thinking}
        assert body["thinking_effort"] == effort
    manager._routing.default_model = "assistant:thinking"
    default = manager._resolve("")
    body = json.loads(_prepare_body(json.dumps(original).encode(), default, ""))
    assert body["thinking_effort"] == "high"
    assert original["chat_template_kwargs"]["enable_thinking"] is None


def test_aliases_preserve_legacy_settings_and_can_be_removed_explicitly():
    stored = {"engine": "llama_cpp", "engines": {"llama_cpp": config()}}
    assert (
        normalize_model_config(stored)["engines"]["llama_cpp"]["set_params_by_id"]
        == config()["set_params_by_id"]
    )
    removed = merge_model_config_put(stored, {"set_params_by_id": []})
    assert removed["engines"]["llama_cpp"]["set_params_by_id"] == []
    assert collect_model_aliases(removed["engines"]["llama_cpp"], "real") == ["assistant"]


def test_request_alias_conflicts_with_another_models_api_name(monkeypatch, tmp_path):
    from backend.tests.test_api_routes import _install_temp_store

    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_model(
        {
            "id": "a",
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": config()},
            },
        }
    )
    store.add_model({"id": "b"})
    assert "assistant:thinking" in find_model_name_conflicts(store, "b", config())


@pytest.mark.parametrize("value", [[], {"model": "other"}, {"model?": "other"}])
def test_api_validation_rejects_invalid_override_objects(value):
    from fastapi import HTTPException
    from backend.routes.models import _validate_model_runtime_config

    with pytest.raises(HTTPException) as caught:
        _validate_model_runtime_config(
            None,
            {},
            {
                "engine": "llama_cpp",
                "engines": {
                    "llama_cpp": {"set_params_by_id": [{"sub_id": "fast", "params": value}]}
                },
            },
        )
    assert caught.value.status_code == 422


def test_native_legacy_filter_contract_and_plain_aliases_remain_supported():
    from backend.proxy.studio.filters import apply_request_filters
    from backend.data_store import collect_model_aliases

    filters = {
        "setParams": {"temperature?": 0.5},
        "setParamsByID": {"assistant:fast": {"enable_thinking": False}},
    }
    result = apply_request_filters(
        {"temperature": 0.8, "enable_thinking": True}, filters, "assistant:fast"
    )
    assert result == {"temperature": 0.8, "enable_thinking": False}
    assert collect_model_aliases(
        {"model_alias": "assistant", "swap_aliases": ["chat", "CHAT"]}, "real"
    ) == ["assistant", "chat"]


def test_native_round_robin_and_default_alias_can_target_request_presets():
    import threading
    from backend.proxy.studio.manager import RoutingSnapshot

    settings = config()
    filters, aliases = request_filters_and_aliases(stable_id="real", config=settings)
    manager = object.__new__(StudioProxyManager)
    manager._route_lock = threading.Lock()
    manager._route_cursor = {}
    manager._routing = RoutingSnapshot.build(
        {
            "default_model": "balanced",
            "routing_aliases": {"balanced": ["assistant:thinking", "assistant:fast"]},
        }
    )
    manager._compiled_models = lambda: [
        (
            None,
            SimpleNamespace(
                proxy=SimpleNamespace(
                    model_id="real",
                    catalog_id="catalog",
                    aliases=aliases,
                    use_model_name="real",
                    filters=filters,
                    engine_id="llama_cpp",
                )
            ),
        )
    ]
    resolved = [manager._resolve("") for _ in range(3)]
    assert [item.model_id for item in resolved] == ["real", "real", "real"]
    assert [item.request_name for item in resolved] == [
        "assistant:thinking",
        "assistant:fast",
        "assistant:thinking",
    ]
    efforts = [
        json.loads(_prepare_body(b'{"messages":[]}', item, ""))["thinking_effort"]
        for item in resolved
    ]
    assert efforts == ["high", "low", "high"]


def test_existing_native_model_header_is_still_recognized():
    from backend.proxy.studio.dataplane import _requested_model

    assert (
        _requested_model({"headers": [(b"x-llama-swap-model", b"assistant:fast")]}, b"{}")
        == "assistant:fast"
    )


@pytest.mark.parametrize("path", ["/api/runtime/apply-launch", "/api/llama-swap/apply-launch"])
def test_native_batch_apply_is_available_at_native_and_existing_urls(
    client, monkeypatch, tmp_path, path
):
    import asyncio
    from backend.tests.test_api_routes import _install_temp_store

    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_model({"id": "real"})
    manager = SimpleNamespace(_apply_lock=asyncio.Lock(), _client=lambda: "native-client")
    monkeypatch.setattr("backend.routes.runtime.get_studio_proxy_manager", lambda: manager)
    captured = {}

    async def apply(entries, **kwargs):
        captured.update(kwargs)
        assert manager._apply_lock.locked()
        assert entries[0]["catalog_id"] == "real"
        return {"results": [{"ok": True}], "stopped_early": False}

    monkeypatch.setattr("backend.services.model_runtime_apply.apply_many", apply)
    response = client.post(
        path,
        json={
            "mode": "next_start",
            "models": [{"catalog_id": "real", "idempotency_key": "batch"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["stopped_early"] is False
    assert captured["mode"] == "next_start"
    assert "real" in captured["models_by_id"]
