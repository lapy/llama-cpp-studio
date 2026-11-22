"""VRAM estimate resolution for proxy admission (pure, no probes).

An estimate is always reported with its evidence, because the evidence decides
whether a deficit is a refusal or a warning:

``settings``
    An operator override in proxy settings. High confidence.
``observed``
    The last uncontended ready-time used-memory delta for that model. Medium
    confidence: it is a measurement, but of one particular load.
``file_size``
    The model payload size, a weights-only lower bound. Low confidence: real
    residency adds KV cache, compute buffers and runtime overhead.
``unknown``
    Nothing comparable, including a model whose size Studio cannot see.

Everything here is synchronous and side-effect free so the manager's blocking
pool can resolve catalog estimates while the event loop keeps running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

MIB = 1024 * 1024

ESTIMATE_SETTINGS = "settings"
ESTIMATE_OBSERVED = "observed"
ESTIMATE_FILE_SIZE = "file_size"
ESTIMATE_UNKNOWN = "unknown"

#: Confidence rank; lower is stronger evidence. Only the first two are strong
#: enough to reject a start in ``enforce`` mode.
ESTIMATE_CONFIDENCE = {
    ESTIMATE_SETTINGS: 0,
    ESTIMATE_OBSERVED: 1,
    ESTIMATE_FILE_SIZE: 2,
    ESTIMATE_UNKNOWN: 3,
}

VRAM_MODES = ("off", "warn", "enforce")
DEFAULT_VRAM_MODE = "off"
DEFAULT_SAFETY_MARGIN_MIB = 1024
MAX_SAFETY_MARGIN_MIB = 1_048_576
MIN_ESTIMATE_MIB = 1
MAX_ESTIMATE_MIB = 4_194_304
MAX_ESTIMATE_ENTRIES = 500


@dataclass(frozen=True)
class Estimate:
    """A per-model VRAM estimate and the evidence behind it."""

    mib: Optional[int]
    source: str

    @property
    def known(self) -> bool:
        return self.mib is not None

    @property
    def derived_only(self) -> bool:
        return self.source == ESTIMATE_FILE_SIZE

    @property
    def rejectable(self) -> bool:
        """Whether a deficit from this estimate may refuse a start."""
        return self.source in (ESTIMATE_SETTINGS, ESTIMATE_OBSERVED)

    def document(self) -> Optional[Dict[str, Any]]:
        if self.mib is None:
            return None
        return {"mib": int(self.mib), "source": self.source}


UNKNOWN_ESTIMATE = Estimate(None, ESTIMATE_UNKNOWN)


def weights_only_lower_bound_mib(file_size_bytes: Any) -> Optional[int]:
    """Model payload size in MiB, rounded up, or ``None`` for an unusable size."""
    try:
        size = int(file_size_bytes)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    return max(MIN_ESTIMATE_MIB, int(math.ceil(size / MIB)))


def normalize_overrides(raw: Any) -> Dict[str, int]:
    """Return valid operator overrides, dropping entries the route would reject.

    Persisted settings are user data: an older or hand-edited document must not
    make the proxy fail to start. Invalid entries are ignored at read time and
    the settings route still validates every new write.
    """
    if not isinstance(raw, Mapping):
        return {}
    overrides: Dict[str, int] = {}
    for model_id, value in raw.items():
        name = str(model_id).strip()
        if not name or len(overrides) >= MAX_ESTIMATE_ENTRIES:
            continue
        try:
            mib = int(value)
        except (TypeError, ValueError):
            continue
        if not MIN_ESTIMATE_MIB <= mib <= MAX_ESTIMATE_MIB:
            continue
        overrides[name] = mib
    return overrides


def normalize_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    return mode if mode in VRAM_MODES else DEFAULT_VRAM_MODE


def normalize_safety_margin(raw: Any, default: int = DEFAULT_SAFETY_MARGIN_MIB) -> int:
    try:
        margin = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, min(MAX_SAFETY_MARGIN_MIB, margin))


def resolve_estimate(
    model_id: str,
    *,
    overrides: Optional[Mapping[str, int]] = None,
    observed: Optional[Mapping[str, Any]] = None,
    file_size_bytes: Any = None,
) -> Estimate:
    """Resolve one model's estimate in the documented precedence order."""
    name = str(model_id or "")
    override = (overrides or {}).get(name)
    if isinstance(override, int) and not isinstance(override, bool) and override > 0:
        return Estimate(override, ESTIMATE_SETTINGS)
    observation = (observed or {}).get(name)
    observed_mib = observation.get("mib") if isinstance(observation, Mapping) else observation
    try:
        observed_value = int(observed_mib) if observed_mib is not None else 0
    except (TypeError, ValueError):
        observed_value = 0
    if observed_value > 0:
        return Estimate(observed_value, ESTIMATE_OBSERVED)
    lower_bound = weights_only_lower_bound_mib(file_size_bytes)
    if lower_bound is not None:
        return Estimate(lower_bound, ESTIMATE_FILE_SIZE)
    return UNKNOWN_ESTIMATE
