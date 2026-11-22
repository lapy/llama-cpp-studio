"""GPU-aware admission for the native Studio proxy.

A start that cannot fit is refused with a truthful reason before a process is
spawned. The decision comes from the compiled LaunchSpec's device selection
(``CUDA_VISIBLE_DEVICES``), never engine-id dispatch, and it never probes NVML
on the event loop: the probe runs on the shared blocking pool, and a probe
failure admits with an ``unknown`` decision instead of inventing a rejection.

Estimate precedence (see :mod:`backend.proxy.studio.vram_estimates`):

1. ``settings`` override (high confidence)
2. last uncontended ready-time used-memory delta (medium confidence)
3. model file-size weights-only lower bound (low confidence)
4. ``unknown``

``enforce`` refuses only a known deficit from an override or an observation.
Derived-only deficits warn, and unknown/probe failure admits. ``off`` is the
default and performs no probe and records no decision, so the previous
behavior is byte-compatible.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from backend.logging_config import get_logger
from backend.proxy.launch_spec import launch_device_selection
from backend.proxy.studio.registry import RegistryError
from backend.proxy.studio.vram_estimates import (
    DEFAULT_SAFETY_MARGIN_MIB,
    ESTIMATE_OBSERVED,
    ESTIMATE_SETTINGS,
    ESTIMATE_UNKNOWN,
    Estimate,
    normalize_mode,
    normalize_overrides,
    normalize_safety_margin,
    resolve_estimate,
)

logger = get_logger(__name__)

MIB = 1024 * 1024
#: A repeated admission within this window reuses one probe. Memory changes at
#: engine load timescales, so this is short enough to stay truthful and long
#: enough that a burst of starts does not run nvidia-smi once per model.
PROBE_TTL_SECONDS = 2.0
PROBE_TIMEOUT_SECONDS = 8.0
INSUFFICIENT_VRAM = "INSUFFICIENT_VRAM"

_ACTIVITY_TYPES = {
    "warned": "vram_warning",
    "rejected": "vram_rejected",
    "unknown": "vram_unverified",
}


class VramAdmissionError(RegistryError):
    """A start refused because the selected GPU devices cannot hold the model."""

    code = INSUFFICIENT_VRAM
    http_status = 409

    def __init__(self, decision: Mapping[str, Any]) -> None:
        self.decision: Dict[str, Any] = dict(decision)
        super().__init__(str(self.decision.get("reason") or "insufficient VRAM"))

    def payload(self) -> Dict[str, Any]:
        """HTTP body for a refused start: ``{error, code, detail}``."""
        return {"error": str(self), "code": self.code, "detail": dict(self.decision)}


@dataclass(frozen=True)
class _Reservation:
    token: str
    model: str
    devices: Tuple[int, ...]
    mib: int
    basis: str
    at: float


@dataclass(frozen=True)
class _Selection:
    devices: List[Dict[str, Any]]
    error: Optional[str] = None
    message: str = ""


def _iso(timestamp: Optional[float] = None) -> str:
    moment = datetime.fromtimestamp(
        timestamp if timestamp is not None else time.time(), timezone.utc
    )
    return moment.isoformat()


async def _default_probe() -> Mapping[str, Any]:
    from backend.gpu_detector import probe_gpu_memory

    return await probe_gpu_memory()


class VramAdmission:
    """Per-model VRAM estimates, reservations, probe state, and decisions.

    One instance belongs to the proxy manager for the life of the process, so
    the observed estimates and the last decision survive a proxy restart while
    a full Studio restart begins with no observations.
    """

    def __init__(
        self,
        *,
        model_info: Optional[Callable[[str], Any]] = None,
        probe: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._model_info = model_info
        self._probe_impl = probe or _default_probe
        self._lock = threading.RLock()
        self._activity: Optional[Callable[[str, Mapping[str, Any]], None]] = None
        self._mode = "off"
        self._margin = DEFAULT_SAFETY_MARGIN_MIB
        self._overrides: Dict[str, int] = {}
        self._observed: Dict[str, Dict[str, Any]] = {}
        self._reservations: Dict[str, _Reservation] = {}
        self._last_admission: Optional[Dict[str, Any]] = None
        self._probe_result: Optional[Dict[str, Any]] = None
        self._probe_at = 0.0
        self._probe_state: Dict[str, Any] = {
            "status": "never",
            "vendor": None,
            "at": None,
            "devices": [],
            "error": None,
            "tooling": "unknown",
        }

    # -- configuration ---------------------------------------------------

    def bind_activity(self, emitter: Optional[Callable[[str, Mapping[str, Any]], None]]) -> None:
        """Route admission activities to the registry's activity ring."""
        self._activity = emitter

    def apply_settings(self, settings: Optional[Mapping[str, Any]]) -> None:
        """Live-apply the admission fields of a proxy settings mapping."""
        source = settings if isinstance(settings, Mapping) else {}
        mode = normalize_mode(source.get("vram_admission"))
        margin = normalize_safety_margin(
            source.get("vram_safety_margin_mib", DEFAULT_SAFETY_MARGIN_MIB)
        )
        overrides = normalize_overrides(source.get("vram_estimates"))
        with self._lock:
            self._mode = mode
            self._margin = margin
            self._overrides = overrides

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    # -- observation -----------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Current admission configuration, probe, reservations and decision."""
        with self._lock:
            observed = {
                name: {
                    "mib": int(value.get("mib") or 0),
                    "at": _iso(value.get("at")),
                    "devices": list(value.get("devices") or []),
                }
                for name, value in self._observed.items()
            }
            reservations = [
                {
                    "model": item.model,
                    "devices": list(item.devices),
                    "mib": item.mib,
                    "basis": item.basis,
                    "at": _iso(item.at),
                }
                for item in self._reservations.values()
            ]
            return {
                "mode": self._mode,
                "safety_margin_mib": self._margin,
                "estimates_configured": dict(self._overrides),
                "observed_estimates": observed,
                "probe": dict(self._probe_state),
                "reservations": reservations,
                "last_admission": dict(self._last_admission) if self._last_admission else None,
            }

    def catalog_estimate(self, model_id: str, file_size_bytes: Any = None) -> Optional[Dict]:
        """Estimate document for a catalog row, or ``None`` when unknown."""
        with self._lock:
            overrides = dict(self._overrides)
            observed = {name: dict(value) for name, value in self._observed.items()}
        return resolve_estimate(
            model_id,
            overrides=overrides,
            observed=observed,
            file_size_bytes=file_size_bytes,
        ).document()

    # -- admission -------------------------------------------------------

    async def admit(self, spec: Any) -> Optional[Dict[str, Any]]:
        """Check one LaunchSpec before its process is spawned.

        Returns an opaque token for :meth:`release`/:meth:`observe_ready`, or
        ``None`` when admission is off or skipped. Raises
        :class:`VramAdmissionError` only for a known deficit in enforce mode.
        """
        if self._mode == "off":
            return None
        model_id = str(getattr(spec, "model_id", "") or "")
        engine_id = str(getattr(spec, "engine_id", "") or "")
        decision: Dict[str, Any] = {
            "decision": "unknown",
            "model": model_id,
            "engine": engine_id,
            "mode": self._mode,
            "estimate_mib": None,
            "estimate_source": ESTIMATE_UNKNOWN,
            "safety_margin_mib": self._margin,
            "devices": [],
            "available_mib": None,
            "deficit_mib": None,
            "reason": "",
        }
        probe = await self._probe()
        gpus = [dict(item) for item in probe.get("gpus") or [] if isinstance(item, Mapping)]
        if not gpus:
            return self._finish_without_devices(decision, probe)
        vendor = str(probe.get("vendor") or "")
        if vendor and vendor != "nvidia":
            return self._finish(
                decision,
                "skipped",
                f"GPU vendor {vendor!r} is not covered by VRAM admission "
                "(NVIDIA only in this release).",
                activity=None,
            )

        selection = self._select_devices(spec, gpus)
        if selection.error == "cpu":
            return self._finish(decision, "skipped", selection.message, activity=None)
        if selection.error:
            return self._finish(
                decision,
                "unknown",
                selection.message,
                activity="vram_unverified",
                token=self._make_token(decision, []),
            )

        file_size = await self._file_size(model_id)
        with self._lock:
            overrides = dict(self._overrides)
            observed = {name: dict(value) for name, value in self._observed.items()}
        estimate = resolve_estimate(
            model_id,
            overrides=overrides,
            observed=observed,
            file_size_bytes=file_size,
        )
        decision["estimate_mib"] = estimate.mib
        decision["estimate_source"] = estimate.source

        devices, available = self._device_rows(selection.devices, self._margin)
        decision["devices"] = devices
        decision["available_mib"] = available

        if not estimate.known:
            return self._finish(
                decision,
                "unknown",
                "No VRAM estimate is known for this model; admitting without a VRAM decision.",
                activity="vram_unverified",
                token=self._make_token(decision, selection.devices),
            )

        deficit = int(estimate.mib) - available
        decision["deficit_mib"] = max(0, deficit)
        if deficit <= 0:
            return self._finish(
                decision,
                "admitted",
                f"Estimate {estimate.mib} MiB ({estimate.source}) fits {available} MiB available.",
                activity=None,
                token=self._make_token(decision, selection.devices, reserve=estimate),
            )
        shortfall = (
            f"Estimate {estimate.mib} MiB ({estimate.source}) exceeds {available} MiB "
            f"available after the {self._margin} MiB safety margin"
        )
        if estimate.derived_only:
            return self._finish(
                decision,
                "warned",
                f"{shortfall}; the weights-only bound is not conclusive, so the start is admitted.",
                activity="vram_warning",
                token=self._make_token(decision, selection.devices, reserve=estimate),
            )
        if self._mode != "enforce":
            return self._finish(
                decision,
                "warned",
                f"{shortfall}; warn mode admits the start.",
                activity="vram_warning",
                token=self._make_token(decision, selection.devices, reserve=estimate),
            )
        self._finish(
            decision,
            "rejected",
            f"{shortfall}; refusing to start {model_id or 'the model'}.",
            activity="vram_rejected",
        )
        raise VramAdmissionError(decision)

    # -- reservations and ready-time observation -------------------------

    def release(self, token: Any) -> None:
        """Release a start reservation. Idempotent: tokens may be released twice."""
        token_id = token.get("id") if isinstance(token, Mapping) else token
        if not token_id:
            return
        with self._lock:
            self._reservations.pop(str(token_id), None)

    async def observe_ready(self, generation: Any, token: Any) -> None:
        """Measure the ready-time memory delta and remember it as an estimate."""
        if self._mode == "off" or not isinstance(token, Mapping):
            return
        baselines = token.get("baselines") or {}
        if not isinstance(baselines, Mapping) or not baselines:
            self.release(token)
            return
        probe = await self._probe(force=True)
        by_index = {
            int(item["index"]): item
            for item in probe.get("gpus") or []
            if isinstance(item, Mapping) and item.get("index") is not None
        }
        contended = bool(token.get("contended")) or self._has_overlap(
            token.get("devices") or (), exclude=str(token.get("id") or "")
        )
        rows: List[Dict[str, Any]] = []
        observed_mib = 0
        for index, baseline in baselines.items():
            device = by_index.get(int(index))
            if device is None:
                continue
            used_mib = int(device.get("used_mib") or 0)
            delta = max(0, used_mib - int(baseline or 0))
            observed_mib += delta
            rows.append({"index": int(index), "used_mib": used_mib, "delta_mib": delta})
        if rows:
            try:
                generation.vram = {
                    "estimate_mib": token.get("estimate_mib"),
                    "estimate_source": token.get("estimate_source"),
                    "observed_mib": observed_mib,
                    "quality": "contended" if contended else "observed",
                    "devices": rows,
                    "observed_at": _iso(),
                }
            except Exception:  # noqa: BLE001 - a display field must never fail a start
                logger.debug("Could not attach VRAM observation to generation", exc_info=True)
            model_id = str(token.get("model") or "")
            if observed_mib > 0 and not contended and model_id:
                with self._lock:
                    self._observed[model_id] = {
                        "mib": observed_mib,
                        "at": time.time(),
                        "devices": [row["index"] for row in rows],
                    }
        self.release(token)

    # -- selection and arithmetic ---------------------------------------

    def _select_devices(self, spec: Any, gpus: List[Dict[str, Any]]) -> _Selection:
        """Map the LaunchSpec's CUDA_VISIBLE_DEVICES onto probe devices."""
        selection = launch_device_selection(spec)
        if selection is None:
            return _Selection([], "cpu", "CPU mode: the launch does not select a GPU.")
        if selection == ("all",):
            return _Selection(
                list(gpus), None, "Inherit mode: every GPU visible to the deployment."
            )

        by_index = {int(item["index"]): item for item in gpus if item.get("index") is not None}
        uuid_index = self._uuid_index_map(gpus)
        selected: List[Dict[str, Any]] = []
        unmapped: List[str] = []
        for token in selection:
            if token.lstrip("-").isdigit():
                device = by_index.get(int(token))
            else:
                index = uuid_index.get(token) or uuid_index.get(token.lower())
                device = by_index.get(int(index)) if index is not None else None
            if device is None:
                unmapped.append(token)
            elif device not in selected:
                selected.append(device)
        if unmapped:
            return _Selection(
                [],
                "unmappable",
                "GPU identity is not visible in the cached GPU inventory: " + ", ".join(unmapped),
            )
        if not selected:
            return _Selection([], "cpu", "No GPU is selected by the launch spec.")
        return _Selection(selected, None, "")

    @staticmethod
    def _uuid_index_map(gpus: List[Dict[str, Any]]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        rows: List[Any] = list(gpus)
        try:
            from backend.services.model_metadata import get_startup_gpu_list

            inventory = get_startup_gpu_list() or {}
            rows.extend(inventory.get("gpus") or [])
        except Exception:  # noqa: BLE001 - inventory is an accelerator, not a dependency
            pass
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            uuid_value = item.get("uuid")
            index = item.get("index")
            if not uuid_value or index is None:
                continue
            try:
                mapping.setdefault(str(uuid_value), int(index))
                mapping.setdefault(str(uuid_value).lower(), int(index))
            except (TypeError, ValueError):
                continue
        return mapping

    def _device_rows(
        self, devices: List[Dict[str, Any]], margin: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        reserved = self._reserved_by_device()
        rows: List[Dict[str, Any]] = []
        available = 0
        for device in devices:
            index = int(device.get("index") or 0)
            free_mib = max(0, int(device.get("free_mib") or 0))
            reserved_mib = int(reserved.get(index, 0.0))
            usable = max(0, free_mib - margin - reserved_mib)
            available += usable
            rows.append(
                {
                    "index": index,
                    "uuid": str(device.get("uuid") or ""),
                    "name": str(device.get("name") or ""),
                    "total_mib": int(device.get("total_mib") or 0),
                    "free_mib": free_mib,
                    "used_mib": int(device.get("used_mib") or 0),
                    "reserved_mib": reserved_mib,
                    "margin_mib": margin,
                    "available_mib": usable,
                }
            )
        return rows, int(available)

    def _reserved_by_device(self) -> Dict[int, float]:
        """Split each in-flight reservation evenly across its device set.

        The estimate is per model, not per device, and a multi-GPU start may
        spread the model unevenly. An even split keeps the aggregate arithmetic
        exact (sum of available = free - margins - reservations) instead of
        double-counting the whole estimate on every device.
        """
        totals: Dict[int, float] = defaultdict(float)
        with self._lock:
            reservations = list(self._reservations.values())
        for item in reservations:
            if not item.devices:
                continue
            share = item.mib / len(item.devices)
            for index in item.devices:
                totals[int(index)] += share
        return {index: int(math.floor(value)) for index, value in totals.items()}

    def _has_overlap(self, devices: Any, *, exclude: str = "") -> bool:
        wanted = {int(index) for index in devices or ()}
        if not wanted:
            return False
        with self._lock:
            reservations = list(self._reservations.values())
        return any(
            item.token != exclude and wanted.intersection(item.devices) for item in reservations
        )

    def _make_token(
        self,
        decision: Mapping[str, Any],
        devices: List[Dict[str, Any]],
        *,
        reserve: Optional[Estimate] = None,
    ) -> Optional[Dict[str, Any]]:
        if not devices:
            return None
        indexes = tuple(int(device["index"]) for device in devices)
        token_id = uuid.uuid4().hex
        contended = self._has_overlap(indexes)
        token: Dict[str, Any] = {
            "id": token_id,
            "model": str(decision.get("model") or ""),
            "devices": indexes,
            "baselines": {
                int(device["index"]): int(device.get("used_mib") or 0) for device in devices
            },
            "estimate_mib": decision.get("estimate_mib"),
            "estimate_source": decision.get("estimate_source"),
            "contended": contended,
            "at": time.time(),
        }
        if reserve is not None and reserve.mib:
            with self._lock:
                self._reservations[token_id] = _Reservation(
                    token=token_id,
                    model=token["model"],
                    devices=indexes,
                    mib=int(reserve.mib),
                    basis=reserve.source,
                    at=token["at"],
                )
        return token

    # -- decisions -------------------------------------------------------

    def _finish_without_devices(
        self, decision: Dict[str, Any], probe: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        tooling = str(probe.get("tooling") or "none")
        error = probe.get("error")
        if tooling == "none" and not error:
            return self._finish(
                decision,
                "skipped",
                "No NVIDIA GPU is present; VRAM admission is skipped (NVIDIA only in this release).",
                activity=None,
            )
        return self._finish(
            decision,
            "unknown",
            "GPU memory probe failed; admitting without a VRAM decision."
            + (f" ({error})" if error else ""),
            activity="vram_unverified",
        )

    def _finish(
        self,
        decision: Dict[str, Any],
        outcome: str,
        message: str,
        *,
        activity: Optional[str],
        token: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        decision["decision"] = outcome
        decision["reason"] = message
        # Generic API error renderers (Studio's own included) look for
        # detail.message; keep the truthful sentence there as well.
        decision["message"] = message
        decision["at"] = _iso()
        with self._lock:
            self._last_admission = dict(decision)
        if activity and self._activity is not None:
            try:
                self._activity(activity, self._activity_payload(decision))
            except Exception:  # noqa: BLE001 - observation must never break a start
                logger.debug("Could not record VRAM admission activity", exc_info=True)
        return token

    @staticmethod
    def _activity_payload(decision: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "model": decision.get("model"),
            "engine": decision.get("engine"),
            "decision": decision.get("decision"),
            "estimate_mib": decision.get("estimate_mib"),
            "estimate_source": decision.get("estimate_source"),
            "available_mib": decision.get("available_mib"),
            "deficit_mib": decision.get("deficit_mib"),
            "reason": decision.get("reason"),
            "devices": [
                {
                    "index": device.get("index"),
                    "free_mib": device.get("free_mib"),
                    "reserved_mib": device.get("reserved_mib"),
                    "available_mib": device.get("available_mib"),
                }
                for device in decision.get("devices") or []
                if isinstance(device, Mapping)
            ],
        }

    async def _file_size(self, model_id: str) -> Any:
        if self._model_info is None or not model_id:
            return None
        try:
            row = await self._model_info(model_id)
        except Exception:  # noqa: BLE001 - a lookup failure means an unknown estimate
            logger.debug("Could not resolve model size for %s", model_id, exc_info=True)
            return None
        if not isinstance(row, Mapping):
            return None
        return row.get("file_size")

    # -- probe -----------------------------------------------------------

    async def _probe(self, *, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            cached = self._probe_result
            cached_at = self._probe_at
        if cached is not None and not force and time.monotonic() - cached_at < PROBE_TTL_SECONDS:
            return dict(cached)
        try:
            raw = await asyncio.wait_for(self._probe_impl(), PROBE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raw = {
                "vendor": None,
                "gpus": [],
                "error": f"probe timed out after {PROBE_TIMEOUT_SECONDS:g}s",
                "tooling": "error",
            }
        except Exception as exc:  # noqa: BLE001 - a probe failure admits as unknown
            raw = {
                "vendor": None,
                "gpus": [],
                "error": f"{type(exc).__name__}: {exc}",
                "tooling": "error",
            }
        result = dict(raw) if isinstance(raw, Mapping) else {}
        result["gpus"] = [
            dict(item) for item in result.get("gpus") or [] if isinstance(item, Mapping)
        ]
        with self._lock:
            self._probe_result = result
            self._probe_at = time.monotonic()
            self._probe_state = self._probe_document(result)
        return dict(result)

    @staticmethod
    def _probe_document(result: Mapping[str, Any]) -> Dict[str, Any]:
        gpus = [dict(item) for item in result.get("gpus") or [] if isinstance(item, Mapping)]
        tooling = str(result.get("tooling") or "unknown")
        error = result.get("error")
        if gpus:
            status = "ok"
        elif tooling == "none" and not error:
            status = "cpu"
        elif error:
            status = "failed"
        else:
            status = "cpu"
        return {
            "status": status,
            "vendor": result.get("vendor"),
            "at": _iso(),
            "devices": gpus,
            "error": error,
            "tooling": tooling,
        }
