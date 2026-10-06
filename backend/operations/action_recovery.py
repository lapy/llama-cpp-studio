"""Completion evidence for side-effecting actions.

A saved settings document is not evidence that a later action finished.
A missing result file is not evidence that it never started. Kinds that are
absent from this module stay unproven: a new kind is unknown until it records
proof that its side effect did not happen. Nothing here replays an action.

``effect_started: false`` is negative evidence only when the durable
transition to "may have started" happens before the request is sent or the
process is launched. A fresh stopped observation does not prove an earlier
request cannot still take effect.
"""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from typing import Mapping, Optional

_ACTIVE = frozenset({"queued", "running", "cancelling"})
_GATED_KINDS = frozenset({
    "build",
    "install",
    "install_source",
    "activate",
    "remove",
    "runtime_apply",
    "download",
    "audio_model_install",
    "audio_model_import",
    "model_start",
    "model_stop",
    "sync_source",
    "update",
    "uninstall",
})
PROXY_RUNTIME_KEY = "proxy:runtime"
CUDA_TOOLKIT_KEY = "cuda:toolkit"
# These kinds use the studio CUDA tree. They conflict with CUDA uninstall.
CUDA_DEPENDENT_KINDS = frozenset({
    "build",
    "install",
    "install_source",
    "sync_source",
    "update",
})


def keys_overlap(left: str, right: str) -> bool:
    """True when two resource keys name the same work or one contains the other.

    ``hf:org/model`` overlaps ``hf:org/model:file.gguf``. A colon is required,
    so ``hf:org/model`` does not overlap ``hf:org/model-extra``. A proxy-wide
    apply overlaps every per-model apply.
    """
    first = str(left or "")
    second = str(right or "")
    if not first or not second:
        return False
    if first == second or first.startswith(second + ":") or second.startswith(first + ":"):
        return True
    if first == PROXY_RUNTIME_KEY and second.startswith("runtime-apply:"):
        return True
    if second == PROXY_RUNTIME_KEY and first.startswith("runtime-apply:"):
        return True
    return False


def engine_installation_key(engine: str, path: str = "") -> str:
    """Engine-wide key, or one checkout/installation under that engine.

    ``engine:llama_cpp`` overlaps ``engine:llama_cpp:/opt/llama/source-main``.
    Two different installation paths of the same engine do not overlap.
    """
    name = str(engine or "").strip()
    location = str(path or "").strip()
    if not name:
        return location
    if not location:
        return f"engine:{name}"
    return f"engine:{name}:{location}"


def cuda_dependency(kind: str, metadata: Optional[Mapping] = None) -> Optional[str]:
    """Return the CUDA reservation a dependent operation must also hold."""
    body = metadata or {}
    if str(body.get("resource_key") or "") == CUDA_TOOLKIT_KEY:
        return None
    if str(kind or "") in CUDA_DEPENDENT_KINDS and body.get("engine"):
        return CUDA_TOOLKIT_KEY
    extra = body.get("depends_on")
    if isinstance(extra, str) and extra.strip():
        return extra.strip()
    return None


def resource_names(resource_key: str = "", depends_on: Optional[object] = None) -> list:
    """Every reservation name one operation holds."""
    names = []
    key = str(resource_key or "").strip()
    if key:
        names.append(key)
    if isinstance(depends_on, str) and depends_on.strip() and depends_on.strip() not in names:
        names.append(depends_on.strip())
    elif isinstance(depends_on, (list, tuple)):
        for item in depends_on:
            text = str(item or "").strip()
            if text and text not in names:
                names.append(text)
    return names


def huggingface_resource_key(repo_id: str, filename: Optional[str] = None) -> str:
    """One key for a Hugging Face repo, or for one file inside it."""
    repo = str(repo_id or "").strip()
    name = str(filename or "").strip()
    if name:
        return f"hf:{repo}:{name}"
    return f"hf:{repo}"
_confirmation: ContextVar[Optional[dict]] = ContextVar("action_confirmation", default=None)


def bind_action_confirmation(payload: Optional[Mapping]) -> None:
    """Bind a confirmation to this request. A missing pair clears any leftover."""
    body = payload or {}
    operation_id = body.get("confirm_operation_id")
    token = body.get("confirm_state")
    if operation_id and token:
        _confirmation.set({
            "operation_id": str(operation_id),
            "state_token": str(token),
        })
        return
    _confirmation.set(None)


def current_confirmation() -> Optional[dict]:
    return _confirmation.get()

# Explicit False means this process recorded the action before any side effect.
# Missing means the side effect may already have started.
_EFFECT_NOT_STARTED = "effect_started"

# Apply phases that are already settled and must not be rewritten.
SETTLED_APPLY_PHASES = frozenset({
    "ready",
    "unchanged",
    "succeeded",
    "failed",
    "cancelled",
    "rolled_back",
    "interrupted",
    "unknown",
})


def classify_action(
    kind: str,
    detail: Optional[Mapping] = None,
    observation: Optional[Mapping] = None,
) -> dict:
    """Return evidence, the durable status, and whether a new attempt is safe.

    ``retry`` is ``safe`` only when this process durably recorded that the
    side effect was not started. A verified running or stopped observation
    does not by itself authorize another attempt: the earlier request may
    still be in flight. ``unnecessary`` means the desired result is already
    observed, so nothing is sent. ``withheld`` means the attempt must not be
    repeated unless a confirmation names this operation and its current state.
    """
    payload = dict(detail or {})
    observed = dict(observation or {})
    evidence = _evidence(str(kind or "operation"), payload, observed)
    if evidence == "negative":
        status = "interrupted"
        retry = "safe"
        message = (
            "The action had not started. It was not run again. "
            "You can retry it."
        )
    elif evidence == "completed":
        status = "succeeded"
        retry = "unnecessary"
        message = (
            "The action's result was already observed. It was not run again."
        )
    else:
        status = "unknown"
        retry = _fresh_retry(str(kind or "operation"), observed)
        message = (
            "Whether this action finished could not be established. "
            "It was not run again. Confirm this operation and its current "
            "state before trying again."
        )
    return {
        "kind": str(kind or "operation"),
        "evidence": evidence,
        "status": status,
        "retry": retry,
        "replayed": 0,
        "message": message,
    }


def classify_apply_journal(
    journal: Mapping,
    *,
    published_revision: Optional[str],
    running_revision: Optional[str],
    observed_launch_id: Optional[str] = None,
) -> dict:
    """Classify a launch-apply journal from its phase, pointer, and receipt.

    A process receipt counts only when its launch id is the one this journal
    recorded for that generation. Another generation's live process is not
    evidence for this operation. Proxy reachability is not invented.
    """
    phase = str(journal.get("phase") or "")
    mode = str(journal.get("mode") or "")
    desired = _text(journal.get("desired_revision"))
    old_published = _text(journal.get("published_revision"))
    prior_launch = _text(journal.get("prior_launch_id"))
    operation_id = _text(journal.get("operation_id"))
    was_running = bool(journal.get("was_running"))
    pointer = _text(published_revision)
    running = _text(running_revision)
    observed_launch = _text(observed_launch_id)
    pointer_unchanged = pointer == old_published
    pointer_desired = bool(desired) and pointer == desired
    same_process = bool(prior_launch) and observed_launch == prior_launch
    this_generation = (
        bool(operation_id)
        and observed_launch == operation_id
        and bool(desired)
        and running == desired
    )
    publish_is_enough = mode != "restart_now" or not was_running

    if phase == "validated" and pointer_unchanged:
        evidence = "negative"
    elif phase == "stopping" and pointer_unchanged and same_process:
        evidence = "negative"
    elif phase == "publishing" and pointer_unchanged and (not was_running or same_process):
        evidence = "negative"
    elif phase in {"publishing", "published"} and pointer_desired and publish_is_enough:
        evidence = "completed"
    elif phase == "starting" and this_generation:
        evidence = "completed"
    else:
        evidence = "unproven"

    decision = classify_action(
        "runtime_apply",
        {"phase": phase, "evidence": evidence},
        {"forced_evidence": evidence},
    )
    return decision


class ActionAdmissionError(RuntimeError):
    """A build or install cannot start until the prior attempt is settled."""

    def __init__(self, decision: Mapping) -> None:
        super().__init__(str(decision.get("message") or "Action cannot start"))
        self.detail = dict(decision)


def state_token(row: Mapping) -> str:
    """Identity of one unresolved row. A newer write produces a different token."""
    detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
    raw = json.dumps(
        {
            "operation_id": str(row.get("operation_id") or ""),
            "resource_key": str(row.get("resource_key") or ""),
            "status": str(row.get("status") or ""),
            "updated_at": row.get("updated_at"),
            "effect_started": detail.get("effect_started", None),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resources_conflict(row: Mapping, requested: list) -> bool:
    detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
    existing = resource_names(str(row.get("resource_key") or ""), detail.get("depends_on"))
    return any(keys_overlap(left, right) for left in existing for right in requested)


def admit_action(
    rows: list,
    resource_key: str,
    *,
    confirm_operation_id: Optional[str] = None,
    confirm_state: Optional[str] = None,
    depends_on: Optional[object] = None,
) -> dict:
    """Decide whether a new build or install may be admitted for one resource.

    An active or unknown attempt is rejected. Durable ``effect_started: false``
    allows a retry. An unproven attempt needs a confirmation of that operation
    id and this state token; a token from an older state does not match.
    """
    requested = resource_names(resource_key, depends_on)
    relevant = [
        row for row in rows
        if _resources_conflict(row, requested)
    ]
    active = [row for row in relevant if str(row.get("status") or "") in _ACTIVE]
    if active:
        row = active[-1]
        return _denied(
            row,
            code="ACTION_IN_FLIGHT",
            message="Another attempt for this resource is still in progress. It was not started again.",
        )
    unknown = [row for row in relevant if str(row.get("status") or "") == "unknown"]
    interrupted = [
        row for row in relevant if str(row.get("status") or "") == "interrupted"
    ]
    negative = [
        row for row in interrupted
        if isinstance(row.get("detail"), dict) and row["detail"].get("effect_started") is False
    ]
    unproven_interrupted = [row for row in interrupted if row not in negative]
    if unknown or unproven_interrupted:
        unresolved = unknown or unproven_interrupted
    elif negative:
        row = negative[-1]
        return {
            "admit": True,
            "code": "ACTION_NOT_STARTED",
            "retry": "safe",
            "operation_id": str(row.get("operation_id") or ""),
            "state_token": state_token(row),
            "message": "The earlier attempt durably recorded that it never started.",
        }
    else:
        unresolved = []
    if unresolved:
        row = unresolved[-1]
        token = state_token(row)
        if (
            confirm_operation_id
            and confirm_state
            and confirm_operation_id == str(row.get("operation_id") or "")
            and confirm_state == token
        ):
            return {
                "admit": True,
                "code": "ACTION_CONFIRMED",
                "retry": "confirmed",
                "operation_id": str(row.get("operation_id") or ""),
                "state_token": token,
                "message": "The confirmation matches this unresolved attempt and its current state.",
            }
        return _denied(
            row,
            code="ACTION_RETRY_WITHHELD",
            message=(
                "Prior work may already have happened. "
                "Confirm this operation and its current state before trying again."
            ),
            state_token_value=token,
        )
    return {
        "admit": True,
        "code": "ACTION_CLEAR",
        "retry": "safe",
        "operation_id": None,
        "state_token": None,
        "message": "No unresolved attempt holds this resource.",
    }


def open_action(rows: list, resource_key: str) -> Optional[dict]:
    """Latest reconciled start/stop row that still governs a retry."""
    matches = [
        row for row in rows
        if str(row.get("resource_key") or "") == resource_key
        and str(row.get("status") or "") in {"unknown", "interrupted"}
    ]
    if not matches:
        return None
    return matches[-1]


def _evidence(kind: str, detail: Mapping, observation: Mapping) -> str:
    forced = observation.get("forced_evidence")
    if forced in {"negative", "completed", "unproven"}:
        return str(forced)
    if detail.get(_EFFECT_NOT_STARTED) is False:
        return "negative"
    if kind in {"model_start", "model_stop"} and observation.get("quality") == "verified":
        return _power_evidence(kind, observation)
    # A saved settings document never completes the action that follows it.
    if kind in {"model_start", "model_stop"}:
        return _power_evidence(kind, observation)
    if kind == "runtime_apply":
        return "unproven"
    if kind in {"build", "install", "install_source"}:
        return "unproven"
    return "unproven"


def _power_evidence(kind: str, observation: Mapping) -> str:
    if observation.get("quality") != "verified":
        return "unproven"
    state = str(observation.get("state") or "")
    if kind == "model_start" and state == "running":
        return "completed"
    if kind == "model_stop" and state == "stopped":
        return "completed"
    # Stopped does not prove an earlier start request has finished, and a
    # still-running model does not prove an earlier stop request has finished.
    return "unproven"


def _fresh_retry(kind: str, observation: Mapping) -> str:
    """Unproven work stays withheld until a bound confirmation arrives.

    A verified observation can show the desired result is already present.
    It cannot authorize a second attempt while an earlier request may still
    take effect.
    """
    del kind, observation
    return "withheld"


def _denied(row: Mapping, *, code: str, message: str, state_token_value: Optional[str] = None) -> dict:
    token = state_token_value if state_token_value is not None else state_token(row)
    return {
        "admit": False,
        "code": code,
        "retry": "withheld",
        "operation_id": str(row.get("operation_id") or ""),
        "state_token": token,
        "message": message,
    }


def _text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
