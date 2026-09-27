"""Change plans, selective apply, rollback, and startup reconciliation.

Configuration applications serialize on the caller's lock. This module does
not unload unrelated models and never treats an unknown runtime as stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.proxy.manifests import LaunchManifestStore, ManifestStoreError
from backend.proxy.launch_spec import (
    LaunchCompileError,
    compile_model_runtime,
    manifest_document,
    project_stable_proxy_block,
)

TERMINAL = {"succeeded", "failed", "cancelled", "rolled_back", "interrupted"}


class ApplyRejected(Exception):
    def __init__(self, status: int, detail: Dict[str, Any]):
        super().__init__(detail.get("message") or detail.get("error") or "apply rejected")
        self.status = status
        self.detail = detail


class PreflightError(ApplyRejected):
    def __init__(self, failures: Sequence[str]):
        detail = {
            "error": "migration_preflight_failed",
            "message": "Migration preflight failed; the running deployment was left unchanged.",
            "failures": list(failures),
        }
        super().__init__(409, detail)


def build_apply_plan(
    models: Sequence[Mapping[str, Any]],
    *,
    disk_yaml: str,
    states: Optional[Mapping[str, str]] = None,
    yaml_differs: bool = False,
    runtime_known: bool = True,
    store: Optional[LaunchManifestStore] = None,
) -> Dict[str, Any]:
    """Classify saved models against the published manifests and on-disk proxy."""
    store = store or LaunchManifestStore()
    states = {} if states is None else states
    disk_models = _disk_models(disk_yaml)
    rows: List[Dict[str, Any]] = []
    for model in models:
        row = _classify_model(
            model,
            disk_models=disk_models,
            states=states,
            store=store,
            runtime_known=runtime_known,
        )
        if row is not None:
            rows.append(row)
    requires_proxy = bool(yaml_differs) or any(row["requires_proxy_reload"] for row in rows)
    for row in rows:
        if requires_proxy and row["action"] != "none":
            row["action"] = "global_proxy"
            row["requires_proxy_reload"] = True
            if "Reload proxy — affects all loaded models" not in row["reasons"]:
                row["reasons"].append("Reload proxy — affects all loaded models")
    actionable = [row for row in rows if row["action"] != "none"]
    fingerprint = json.dumps(
        {
            "yaml_differs": yaml_differs,
            "models": [
                {
                    "model_id": row["model_id"],
                    "desired": row.get("desired_revision"),
                    "published": row.get("published_revision"),
                    "action": row["action"],
                }
                for row in rows
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "plan_id": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
        "launch_manifests": True,
        "requires_proxy_reload": requires_proxy,
        "migration_required": bool(yaml_differs and requires_proxy),
        "models": actionable,
    }


def preflight_deployment(
    models: Sequence[Mapping[str, Any]],
    disk_yaml: str,
) -> List[Any]:
    """Compile every deployed model. Any failure aborts before service disruption."""
    disk_ids = set(_disk_models(disk_yaml))
    compiled = []
    failures: List[str] = []
    for model in models:
        model_id = _proxy_id(model)
        try:
            compiled.append(compile_model_runtime(model))
        except Exception as exc:
            if model_id and model_id in disk_ids:
                failures.append(f"{model_id}: {exc}")
    if failures:
        raise PreflightError(failures)
    deployed = {item.proxy.model_id for item in compiled}
    missing = sorted(model_id for model_id in disk_ids if model_id not in deployed)
    if missing:
        raise PreflightError(
            [f"{model_id}: deployed model could not be compiled" for model_id in missing]
        )
    return compiled


def publish_compiled(
    compiled: Sequence[Any],
    store: Optional[LaunchManifestStore] = None,
) -> Dict[str, Optional[str]]:
    """Stage generations, then move pointers. Returns the previous revision map."""
    store = store or LaunchManifestStore()
    previous: Dict[str, Optional[str]] = {}
    for item in compiled:
        document = manifest_document(item.launch, item.revision)
        store.stage(item.proxy.model_id, document)
        current = store.read_pointer(item.proxy.model_id)
        previous[item.proxy.model_id] = current.revision if current else None
    try:
        for item in compiled:
            store.publish(item.proxy.model_id, item.revision)
            store.garbage_collect(
                item.proxy.model_id,
                extra=[previous.get(item.proxy.model_id)] if previous.get(item.proxy.model_id) else [],
            )
    except Exception:
        restore_pointers(previous, store)
        raise
    return previous


def restore_pointers(
    previous: Mapping[str, Optional[str]],
    store: Optional[LaunchManifestStore] = None,
) -> None:
    store = store or LaunchManifestStore()
    for model_id, revision in previous.items():
        if not revision:
            pointer = os.path.join(store.model_dir(model_id), "active.json")
            try:
                os.remove(pointer)
            except OSError:
                pass
            continue
        try:
            store.publish(model_id, revision)
        except ManifestStoreError:
            continue


async def apply_model(
    model: Mapping[str, Any],
    *,
    mode: str,
    expected_desired_revision: Optional[str],
    expected_published_revision: Optional[str],
    check_published_revision: bool = False,
    idempotency_key: str,
    gateway: Any,
    yaml_differs: bool = False,
    disk_yaml: str = "",
    store: Optional[LaunchManifestStore] = None,
    plan_id: Optional[str] = None,
) -> Dict[str, Any]:
    if mode not in {"restart_now", "next_start"}:
        raise ApplyRejected(400, {"error": "invalid_mode", "message": "mode must be restart_now or next_start"})
    if not idempotency_key:
        raise ApplyRejected(400, {"error": "idempotency_key_required", "message": "idempotency_key is required"})
    store = store or LaunchManifestStore()
    fingerprint = _fingerprint(
        model,
        mode=mode,
        desired=expected_desired_revision,
        published=expected_published_revision,
    )
    existing = _read_idempotency(store, idempotency_key)
    if existing:
        if existing.get("fingerprint") != fingerprint:
            raise ApplyRejected(
                409,
                {
                    "error": "idempotency_conflict",
                    "message": "idempotency key was already used for a different apply",
                },
            )
        return existing["result"]

    try:
        compiled = compile_model_runtime(model)
    except LaunchCompileError as exc:
        raise ApplyRejected(
            400,
            {"error": "invalid_manifest", "message": str(exc), "field": exc.field},
        ) from exc
    if hasattr(gateway, "disk_yaml"):
        disk_yaml = gateway.disk_yaml() or disk_yaml
    disk_models = _disk_models(disk_yaml)
    state = await _gateway_state(gateway, compiled.proxy.model_id)
    row = _classify_model(
        model,
        disk_models=disk_models,
        states={compiled.proxy.model_id: state},
        store=store,
        compiled=compiled,
    )
    if yaml_differs or (row and row["requires_proxy_reload"]):
        raise ApplyRejected(
            409,
            {
                "error": "global_apply_required",
                "message": "This change rewrites proxy configuration and can stop every loaded model.",
                "plan_id": plan_id,
                "reasons": (row or {}).get("reasons") or [],
            },
        )
    published = store.read_pointer(compiled.proxy.model_id)
    published_revision = published.revision if published else None
    if expected_desired_revision and expected_desired_revision != compiled.revision:
        raise ApplyRejected(
            409,
            {
                "error": "revision_conflict",
                "message": "The saved configuration changed after this plan was reviewed.",
                "desired_revision": compiled.revision,
                "published_revision": published_revision,
            },
        )
    if check_published_revision and (expected_published_revision or None) != published_revision:
        raise ApplyRejected(
            409,
            {
                "error": "revision_conflict",
                "message": "The published revision changed after this plan was reviewed.",
                "desired_revision": compiled.revision,
                "published_revision": published_revision,
            },
        )
    if row and row["action"] == "none":
        result = _result(
            compiled,
            status="succeeded",
            phase="unchanged",
            message="No effective launch difference.",
            published_revision=published_revision,
            running_state=state,
        )
        _write_idempotency(store, idempotency_key, fingerprint, result)
        return result

    if state == "unknown" and mode == "restart_now":
        raise ApplyRejected(
            409,
            {
                "error": "runtime_state_unknown",
                "message": "Runtime state is unknown. Retry restart, or publish for the next start.",
            },
        )

    operation_id = uuid.uuid4().hex
    running_revision = verified_running_revision(store, compiled.proxy.model_id)
    restoration = running_revision or (published.previous_revision if published and state in {"running", "loading"} else published_revision)
    journal = {
        "operation_id": operation_id,
        "model_id": compiled.proxy.model_id,
        "catalog_id": compiled.proxy.catalog_id,
        "mode": mode,
        "phase": "validated",
        "desired_revision": compiled.revision,
        "published_revision": published_revision,
        "running_revision": running_revision,
        "restoration_revision": restoration,
        "was_running": state in {"running", "loading"},
        "state": state,
    }
    _write_journal(store, journal)
    _write_idempotency(
        store,
        idempotency_key,
        fingerprint,
        _result(compiled, status="running", phase="validated", message="Apply accepted", operation_id=operation_id, published_revision=published_revision, running_state=state),
    )

    if _cancelled(store, operation_id) and journal["phase"] == "validated":
        return _finish_cancel(store, journal, compiled, idempotency_key, fingerprint, before_stop=True)

    document = manifest_document(compiled.launch, compiled.revision)
    try:
        store.stage(compiled.proxy.model_id, document)
    except ManifestStoreError as exc:
        return _fail(store, journal, compiled, idempotency_key, fingerprint, f"Could not stage the launch generation: {exc}")

    was_running = state in {"running", "loading"}
    fd = store.acquire_gate(compiled.proxy.model_id, exclusive=True)
    try:
        if mode == "restart_now" and was_running:
            journal["phase"] = "stopping"
            _write_journal(store, journal)
            if _cancelled(store, operation_id):
                return _finish_cancel(
                    store,
                    journal,
                    compiled,
                    idempotency_key,
                    fingerprint,
                    before_stop=True,
                )
            hook = getattr(gateway, "before_stop", None)
            if hook is not None:
                await hook(operation_id)
            if _cancelled(store, operation_id):
                return _finish_cancel(
                    store,
                    journal,
                    compiled,
                    idempotency_key,
                    fingerprint,
                    before_stop=True,
                )
            try:
                await gateway.unload(compiled.proxy.model_id)
            except Exception as exc:
                return _fail(
                    store,
                    journal,
                    compiled,
                    idempotency_key,
                    fingerprint,
                    f"Could not unload the model: {exc}",
                )
        journal["phase"] = "publishing"
        _write_journal(store, journal)
        store.publish(compiled.proxy.model_id, compiled.revision)
    finally:
        store.release_gate(fd)
    store.garbage_collect(
        compiled.proxy.model_id,
        extra=[item for item in (published_revision, restoration, running_revision) if item],
    )
    journal["phase"] = "published"
    journal["published_revision"] = compiled.revision
    _write_journal(store, journal)

    if mode == "next_start" or not was_running:
        message = (
            "Published for the next start. The running process is still on the previous revision."
            if was_running
            else "Ready for next start. The model was left stopped."
        )
        return _succeed(
            store,
            journal,
            compiled,
            idempotency_key,
            fingerprint,
            phase="published",
            message=message,
            running_state="running" if was_running else "stopped",
        )

    journal["phase"] = "starting"
    _write_journal(store, journal)
    try:
        await gateway.load(compiled.proxy.model_id)
        ready = await _wait_until_ready(
            store,
            compiled.proxy.model_id,
            compiled.revision,
            timeout=getattr(gateway, "startup_timeout", 5.0),
        )
    except Exception as exc:
        ready = False
        start_error = str(exc)
    else:
        start_error = ""
    if not ready:
        rolled = await _rollback(
            store,
            gateway,
            journal,
            compiled,
            error=start_error or "replacement did not become ready",
        )
        result = _result(
            compiled,
            status="failed",
            phase=journal["phase"],
            message=rolled["message"],
            operation_id=operation_id,
            published_revision=rolled.get("published_revision"),
            running_state=rolled.get("running_state"),
            rollback=rolled,
        )
        journal["phase"] = "failed"
        journal["message"] = rolled["message"]
        _write_journal(store, journal)
        _write_idempotency(store, idempotency_key, fingerprint, result)
        return result

    return _succeed(
        store,
        journal,
        compiled,
        idempotency_key,
        fingerprint,
        phase="ready",
        message="Model restarted on the requested revision.",
        running_state="running",
    )


async def apply_many(
    entries: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    gateway: Any,
    models_by_id: Mapping[str, Mapping[str, Any]],
    yaml_differs: bool = False,
    disk_yaml: str = "",
    store: Optional[LaunchManifestStore] = None,
) -> Dict[str, Any]:
    """Apply launch-only models sequentially and stop after the first failure."""
    results = []
    for entry in entries:
        model_id = str(entry.get("catalog_id") or entry.get("model_id") or "")
        model = models_by_id.get(model_id)
        if model is None:
            results.append(
                {
                    "ok": False,
                    "catalog_id": model_id,
                    "error": "not_found",
                    "message": f"Model {model_id} is not in the catalog",
                }
            )
            break
        try:
            outcome = await apply_model(
                model,
                mode=mode,
                expected_desired_revision=entry.get("expected_desired_revision"),
                expected_published_revision=entry.get("expected_published_revision"),
                check_published_revision="expected_published_revision" in entry,
                idempotency_key=str(entry.get("idempotency_key") or ""),
                gateway=gateway,
                yaml_differs=yaml_differs,
                disk_yaml=disk_yaml,
                store=store,
            )
        except ApplyRejected as exc:
            results.append({"ok": False, "catalog_id": model_id, **exc.detail})
            break
        results.append({"ok": outcome.get("status") == "succeeded", "catalog_id": model_id, **outcome})
        if outcome.get("status") != "succeeded":
            break
    return {
        "results": results,
        "stopped_early": any(not row.get("ok") for row in results),
    }


def request_cancel(operation_id: str, store: Optional[LaunchManifestStore] = None) -> None:
    store = store or LaunchManifestStore()
    path = os.path.join(_ops_dir(store), f"{operation_id}.cancel")
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("cancel\n")


def reconcile_journals(store: Optional[LaunchManifestStore] = None) -> int:
    """Record interrupted applies without replaying a completed restart."""
    store = store or LaunchManifestStore()
    changed = 0
    directory = _ops_dir(store)
    if not os.path.isdir(directory):
        return 0
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                journal = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        phase = str(journal.get("phase") or "")
        if phase in TERMINAL or phase in {"ready", "published", "unchanged"}:
            continue
        pointer = None
        model_id = str(journal.get("model_id") or "")
        try:
            pointer = store.read_pointer(model_id) if model_id else None
        except ManifestStoreError:
            pointer = None
        journal["observed_published_revision"] = pointer.revision if pointer else None
        journal["observed_running_revision"] = (
            verified_running_revision(store, model_id) if model_id else None
        )
        journal["phase"] = "interrupted"
        journal["message"] = (
            "Apply was interrupted. Observed pointer and process identity were recorded; "
            "the restart was not replayed."
        )
        _write_json(path, journal)
        changed += 1
    return changed


def verified_running_revision(store: LaunchManifestStore, model_id: str) -> Optional[str]:
    receipts = store.read_receipts(model_id)
    for receipt in reversed(receipts):
        if _receipt_matches_process(receipt):
            return str(receipt.get("revision") or "") or None
    return None


def _classify_model(
    model: Mapping[str, Any],
    *,
    disk_models: Mapping[str, Any],
    states: Mapping[str, str],
    store: LaunchManifestStore,
    compiled: Any = None,
    runtime_known: bool = True,
) -> Optional[Dict[str, Any]]:
    model_id = _proxy_id(model)
    if not model_id:
        return None
    try:
        compiled = compiled or compile_model_runtime(model)
    except Exception as exc:
        if model_id not in disk_models:
            return None
        return {
            "catalog_id": str(model.get("id") or model_id),
            "model_id": model_id,
            "engine_id": "",
            "desired_revision": None,
            "published_revision": None,
            "action": "global_proxy",
            "reasons": [f"Deployed model cannot be compiled: {exc}"],
            "requires_proxy_reload": True,
            "running": False,
            "state": states.get(model_id, "unknown"),
        }
    disk_block = disk_models.get(compiled.proxy.model_id)
    reasons: List[str] = []
    proxy_changed = False
    if disk_block is None:
        proxy_changed = True
        reasons.append("Add or remove a model route")
    else:
        desired = _desired_block(compiled)
        if not _same_proxy(desired, disk_block):
            proxy_changed = True
            reasons.append("Proxy contract differs from the published llama-swap entry")
    pointer = None
    try:
        pointer = store.read_pointer(compiled.proxy.model_id)
    except ManifestStoreError as exc:
        proxy_changed = True
        reasons.append(f"Published pointer is unreadable: {exc}")
    published_revision = pointer.revision if pointer else None
    if not runtime_known:
        state = "unknown"
    else:
        state = states.get(compiled.proxy.model_id, "stopped")
    if proxy_changed:
        action = "global_proxy"
    elif published_revision == compiled.revision:
        action = "none"
        reasons = []
    elif state in {"running", "loading"}:
        action = "restart_now"
        reasons.append("Restart model " + compiled.proxy.model_id)
    else:
        action = "publish_next_start"
        reasons.append("Use new settings on next start")
    return {
        "catalog_id": compiled.proxy.catalog_id,
        "model_id": compiled.proxy.model_id,
        "engine_id": compiled.launch.engine_id,
        "desired_revision": compiled.revision,
        "published_revision": published_revision,
        "action": action,
        "reasons": reasons,
        "requires_proxy_reload": action == "global_proxy",
        "running": state in {"running", "loading"},
        "state": state,
        "plan_actions": ["restart_now", "publish_next_start"]
        if action in {"restart_now", "publish_next_start"}
        else [action],
    }


def _desired_block(compiled) -> Dict[str, Any]:
    block: Dict[str, Any] = {"cmd": "pending"}
    if compiled.proxy.use_model_name:
        block["useModelName"] = compiled.proxy.use_model_name
    if compiled.proxy.filters:
        block["filters"] = compiled.proxy.filters
    if compiled.proxy.aliases:
        block["aliases"] = compiled.proxy.aliases
    if compiled.proxy.health_endpoint:
        block["checkEndpoint"] = compiled.proxy.health_endpoint
    return project_stable_proxy_block(block, compiled.proxy.model_id)


def _same_proxy(desired: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return _canonical(_view(desired)) == _canonical(_view(actual))


def _view(block: Mapping[str, Any]) -> Dict[str, Any]:
    viewed: Dict[str, Any] = {}
    for key in (
        "cmd",
        "proxy",
        "useModelName",
        "filters",
        "aliases",
        "checkEndpoint",
        "capabilities",
        "env",
        "macros",
    ):
        value = block.get(key)
        if value in (None, "", [], {}):
            continue
        viewed[key] = value
    return viewed


def _disk_models(disk_yaml: str) -> Dict[str, Any]:
    if not disk_yaml.strip():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        parsed = yaml.safe_load(disk_yaml) or {}
    except Exception:
        return {}
    models = parsed.get("models") if isinstance(parsed, dict) else None
    if not isinstance(models, dict):
        return {}
    return {str(key): value for key, value in models.items() if isinstance(value, dict)}


def _proxy_id(model: Mapping[str, Any]) -> str:
    from backend import data_store

    try:
        return data_store.resolve_llama_swap_id(model)
    except Exception:
        return str(model.get("proxy_name") or model.get("id") or "")


async def _gateway_state(gateway: Any, model_id: str) -> str:
    if not hasattr(gateway, "state"):
        return "unknown"
    try:
        state = await gateway.state(model_id)
    except Exception:
        return "unknown"
    if state not in {"running", "stopped", "loading", "unknown"}:
        return "unknown"
    return state


async def _wait_until_ready(store, model_id: str, revision: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() <= deadline:
        if _revision_process_matches(store, model_id, revision):
            return True
        time.sleep(0.05)
    return _revision_process_matches(store, model_id, revision)


def _revision_process_matches(store, model_id: str, revision: str) -> bool:
    for receipt in store.read_receipts(model_id):
        if str(receipt.get("revision") or "") != revision:
            continue
        if _receipt_matches_process(receipt):
            return True
    return False


async def _rollback(store, gateway, journal, compiled, *, error: str) -> Dict[str, Any]:
    journal["phase"] = "rolling_back"
    _write_journal(store, journal)
    restoration = journal.get("restoration_revision")
    message = f"Apply failed: {error}."
    try:
        if hasattr(gateway, "unload"):
            await gateway.unload(compiled.proxy.model_id)
    except Exception as exc:
        message += f" Rollback unload failed: {exc}."
        return {
            "ok": False,
            "message": message + " The model was left stopped.",
            "published_revision": journal.get("published_revision"),
            "running_state": "stopped",
        }
    if not restoration:
        message += " No verified revision was available to restore. The model was left stopped."
        return {
            "ok": False,
            "message": message,
            "published_revision": store.read_pointer(compiled.proxy.model_id).revision
            if store.read_pointer(compiled.proxy.model_id)
            else None,
            "running_state": "stopped",
        }
    fd = store.acquire_gate(compiled.proxy.model_id, exclusive=True)
    try:
        store.publish(compiled.proxy.model_id, restoration)
    except ManifestStoreError as exc:
        message += f" Could not restore the previous generation: {exc}."
        return {"ok": False, "message": message, "published_revision": None, "running_state": "stopped"}
    finally:
        store.release_gate(fd)
    if journal.get("was_running"):
        try:
            await gateway.load(compiled.proxy.model_id)
        except Exception as exc:
            message += f" Restoration start failed: {exc}. The model was left stopped."
            journal["phase"] = "rolled_back"
            return {
                "ok": False,
                "message": message,
                "published_revision": restoration,
                "running_state": "stopped",
            }
    journal["phase"] = "rolled_back"
    message += " Restored the last verified revision. The requested change did not succeed."
    return {
        "ok": True,
        "message": message,
        "published_revision": restoration,
        "running_state": "running" if journal.get("was_running") else "stopped",
    }


def _receipt_matches_process(receipt: Mapping[str, Any]) -> bool:
    try:
        pid = int(receipt.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or not _pid_alive(pid):
        return False
    expected = str(receipt.get("start_ticks") or "")
    if not expected:
        return False
    return _start_ticks(pid) == expected


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_ticks(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            stat = handle.read()
    except OSError:
        return ""
    end = stat.rfind(")")
    if end < 0:
        return ""
    fields = stat[end + 2 :].split()
    if len(fields) < 20:
        return ""
    return fields[19]


def _result(compiled, **kwargs: Any) -> Dict[str, Any]:
    payload = {
        "model_id": compiled.proxy.model_id,
        "catalog_id": compiled.proxy.catalog_id,
        "engine_id": compiled.launch.engine_id,
        "desired_revision": compiled.revision,
        "published_revision": kwargs.get("published_revision"),
        "status": kwargs.get("status"),
        "phase": kwargs.get("phase"),
        "message": kwargs.get("message") or "",
        "operation_id": kwargs.get("operation_id"),
        "running_state": kwargs.get("running_state"),
    }
    if kwargs.get("rollback") is not None:
        payload["rollback"] = kwargs["rollback"]
    return payload


def _succeed(store, journal, compiled, key, fingerprint, *, phase, message, running_state):
    journal["phase"] = phase
    journal["message"] = message
    _write_journal(store, journal)
    result = _result(
        compiled,
        status="succeeded",
        phase=phase,
        message=message,
        operation_id=journal["operation_id"],
        published_revision=compiled.revision,
        running_state=running_state,
    )
    _write_idempotency(store, key, fingerprint, result)
    return result


def _fail(store, journal, compiled, key, fingerprint, message, rollback=False):
    journal["phase"] = "failed"
    journal["message"] = message
    _write_journal(store, journal)
    result = _result(
        compiled,
        status="failed",
        phase="failed",
        message=message,
        operation_id=journal["operation_id"],
        published_revision=journal.get("published_revision"),
        running_state=journal.get("state"),
    )
    _write_idempotency(store, key, fingerprint, result)
    return result


def _finish_cancel(store, journal, compiled, key, fingerprint, *, before_stop: bool):
    if before_stop:
        journal["phase"] = "cancelled"
        journal["message"] = "Cancelled before the model was stopped."
        status = "cancelled"
    else:
        journal["phase"] = "interrupted"
        journal["message"] = "Cancellation requested after the model was stopped; recovering to the published revision."
        status = "cancelled"
    _write_journal(store, journal)
    result = _result(
        compiled,
        status=status,
        phase=journal["phase"],
        message=journal["message"],
        operation_id=journal["operation_id"],
        published_revision=journal.get("published_revision"),
        running_state=journal.get("state"),
    )
    _write_idempotency(store, key, fingerprint, result)
    return result


def _fingerprint(model, *, mode: str, desired: Optional[str], published: Optional[str]) -> str:
    raw = json.dumps(
        {
            "catalog_id": str(model.get("id") or ""),
            "mode": mode,
            "desired": desired,
            "published": published,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ops_dir(store: LaunchManifestStore) -> str:
    path = os.path.join(os.path.dirname(store.root), "operations")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def _idempotency_path(store: LaunchManifestStore, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(_ops_dir(store), "idempotency", f"{digest}.json")


def _read_idempotency(store: LaunchManifestStore, key: str) -> Optional[Dict[str, Any]]:
    path = _idempotency_path(store, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_idempotency(store, key: str, fingerprint: str, result: Dict[str, Any]) -> None:
    path = _idempotency_path(store, key)
    _write_json(path, {"fingerprint": fingerprint, "result": result})


def _journal_path(store: LaunchManifestStore, operation_id: str) -> str:
    if not operation_id or any(ch not in "0123456789abcdef" for ch in operation_id):
        raise ApplyRejected(400, {"error": "invalid_operation", "message": "operation id is invalid"})
    return os.path.join(_ops_dir(store), f"{operation_id}.json")


def _write_journal(store, journal: Dict[str, Any]) -> None:
    _write_json(_journal_path(store, journal["operation_id"]), journal)


def _cancelled(store, operation_id: str) -> bool:
    return os.path.isfile(os.path.join(_ops_dir(store), f"{operation_id}.cancel"))


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class LlamaSwapRuntimeGateway:
    """Talks to the running proxy. Selective apply never calls unload-all."""

    def __init__(self, client: Any, *, startup_timeout: float = 30.0):
        self.client = client
        self.startup_timeout = startup_timeout

    async def state(self, model_id: str) -> str:
        response = await self.client.request("GET", "/running", timeout=5)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("running") if isinstance(payload, dict) else None
        for item in rows or []:
            if isinstance(item, dict) and item.get("model") == model_id:
                state = str(item.get("state") or "running")
                if state in {"running", "loading", "stopped"}:
                    return state
                return "running"
        return "stopped"

    async def unload(self, model_id: str) -> None:
        await self.client.unload_model(model_id)

    async def load(self, model_id: str) -> None:
        await self.client.load_model(model_id)
