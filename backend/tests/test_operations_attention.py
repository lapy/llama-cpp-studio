"""GET /api/operations/attention: interrupted/unknown rows and safe actions.

The endpoint is a read-only projection. These tests pin what it lists, the
order it lists, what it deliberately omits, and that it offers no action the
backend cannot already serve.
"""

import time

from backend.tests.test_api_routes import _install_temp_store


def _operation(
    operation_id,
    status,
    *,
    updated_at=None,
    kind="build",
    resource_key="engine:llama_cpp",
    message=None,
    detail=None,
):
    return {
        "operation_id": operation_id,
        "kind": kind,
        "status": status,
        "resource_key": resource_key,
        "message": message if message is not None else f"{status} operation",
        "detail": dict(detail or {}),
        "updated_at": time.time() - 5 if updated_at is None else updated_at,
    }


def _seed(store, rows):
    for row in rows:
        store.upsert_operation(row)


def test_attention_lists_only_interrupted_and_unknown_newest_first(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    now = time.time()
    _seed(
        store,
        [
            _operation("running-1", "running", updated_at=now - 1),
            _operation("succeeded-1", "succeeded", updated_at=now - 2),
            _operation("failed-1", "failed", updated_at=now - 3),
            _operation("unknown-old", "unknown", updated_at=now - 40),
            _operation("interrupted-new", "interrupted", updated_at=now - 4),
        ],
    )

    response = client.get("/api/operations/attention")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [row["operation_id"] for row in body["operations"]] == [
        "interrupted-new",
        "unknown-old",
    ]
    # A read must not change the durable rows reconciliation stored.
    assert {row["operation_id"]: row["status"] for row in store.list_operations()}[
        "running-1"
    ] == "running"
    unknown = next(row for row in store.list_operations() if row["operation_id"] == "unknown-old")
    assert unknown["status"] == "unknown"


def test_attention_ordering_tiebreaks_by_operation_id(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    same_time = time.time() - 5
    _seed(
        store,
        [
            _operation("bbb", "unknown", updated_at=same_time),
            _operation("aaa", "interrupted", updated_at=same_time),
        ],
    )

    body = client.get("/api/operations/attention").json()

    assert [row["operation_id"] for row in body["operations"]] == ["aaa", "bbb"]


def test_attention_reports_reason_and_safe_retry_classification(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    recovery = {
        "kind": "build",
        "evidence": "unproven",
        "status": "unknown",
        "retry": "withheld",
        "replayed": 0,
        "message": "Whether this action finished could not be established.",
    }
    _seed(
        store,
        [
            _operation(
                "build-unknown",
                "unknown",
                message="Whether this action finished could not be established.",
                detail={"description": "Build llama.cpp", "recovery": recovery},
            ),
        ],
    )

    body = client.get("/api/operations/attention").json()
    row = body["operations"][0]

    assert row["task_id"] == "build-unknown"
    assert row["kind"] == "build"
    assert row["status"] == "unknown"
    assert row["description"] == "Build llama.cpp"
    assert row["message"] == "Whether this action finished could not be established."
    assert row["resource_key"] == "engine:llama_cpp"
    assert row["evidence"] == "unproven"
    assert row["retry"] == "withheld"
    assert row["needs_confirmation"] is True
    assert row["actions"] == ["view_logs"]
    assert row["recovery"]["status"] == "unknown"
    assert row["recovery"]["replayed"] == 0
    assert row["updated_at"] > 0


def test_attention_offers_dismiss_only_for_work_that_never_started(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed(
        store,
        [
            _operation("build-not-started", "interrupted", detail={"effect_started": False}),
            _operation("build-starting", "interrupted", detail={"effect_started": True}),
            _operation("build-unknown", "unknown", detail={"effect_started": True}),
            _operation("unknown-negative", "unknown", detail={"effect_started": False}),
        ],
    )

    rows = {
        row["operation_id"]: row
        for row in client.get("/api/operations/attention").json()["operations"]
    }

    not_started = rows["build-not-started"]
    assert not_started["evidence"] == "negative"
    assert not_started["retry"] == "safe"
    assert not_started["needs_confirmation"] is False
    assert not_started["actions"] == ["view_logs", "dismiss"]

    # An interrupted row without negative evidence is still an admission fence.
    starting = rows["build-starting"]
    assert starting["retry"] == "withheld"
    assert starting["needs_confirmation"] is True
    assert starting["actions"] == ["view_logs"]

    unknown = rows["build-unknown"]
    assert unknown["needs_confirmation"] is True
    assert "dismiss" not in unknown["actions"]

    # "unknown" is an admission fence even if its detail claims no side effect.
    unknown_negative = rows["unknown-negative"]
    assert unknown_negative["retry"] == "withheld"
    assert unknown_negative["needs_confirmation"] is True
    assert unknown_negative["actions"] == ["view_logs"]


def test_attention_skips_acknowledged_rows(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed(
        store,
        [
            _operation(
                "build-consumed",
                "unknown",
                detail={"confirmation_consumed": True},
            ),
            _operation("build-open", "unknown", detail={}),
        ],
    )

    body = client.get("/api/operations/attention").json()

    assert body["count"] == 1
    assert [row["operation_id"] for row in body["operations"]] == ["build-open"]


def test_attention_is_empty_when_nothing_is_interrupted(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed(store, [_operation("done-1", "succeeded")])

    body = client.get("/api/operations/attention").json()

    assert body == {"operations": [], "count": 0}
