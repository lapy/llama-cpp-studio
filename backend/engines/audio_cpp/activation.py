"""audio.cpp work that runs after the shared activation lock is held."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


def capability_delta(previous: Optional[dict], current: Optional[dict]) -> Dict[str, Any]:
    from backend.engines.scan.scanner import compute_audio_cpp_capability_delta

    return compute_audio_cpp_capability_delta(previous, current)


def is_audio_cpp_model(model: dict) -> bool:
    if not isinstance(model, dict):
        return False
    engines = model.get("compatible_engines") or []
    if "audio_cpp" in engines:
        return True
    config = model.get("config") if isinstance(model.get("config"), dict) else {}
    if str(config.get("engine") or "").strip() == "audio_cpp":
        return True
    engines_cfg = config.get("engines") if isinstance(config.get("engines"), dict) else {}
    return isinstance(engines_cfg.get("audio_cpp"), dict)


def models_affected_by_delta(
    store, delta: Optional[dict], *, contract_changed: bool = False
) -> List[dict]:
    """Return lightweight model rows whose saved family/task intersect *delta*."""
    delta = delta or {}
    families = {
        str(item).strip().lower()
        for item in [
            *(delta.get("added_families") or []),
            *(delta.get("removed_families") or []),
        ]
        if str(item).strip()
    }
    tasks = {
        str(item).strip().lower()
        for item in [
            *(delta.get("added_tasks") or []),
            *(delta.get("removed_tasks") or []),
        ]
        if str(item).strip()
    }
    affected: List[dict] = []
    for model in store.list_models() or []:
        if not isinstance(model, dict):
            continue
        config = model.get("config") if isinstance(model.get("config"), dict) else {}
        engine = str(config.get("engine") or model.get("engine") or "").strip()
        engines = config.get("engines") if isinstance(config.get("engines"), dict) else {}
        audio_cfg = engines.get("audio_cpp") if isinstance(engines.get("audio_cpp"), dict) else {}
        if engine != "audio_cpp" and not audio_cfg:
            continue
        family = str(
            audio_cfg.get("family") or model.get("family") or ""
        ).strip().lower()
        task = str(audio_cfg.get("task") or "").strip().lower()
        intersects = (family and family in families) or (task and task in tasks)
        if intersects or (contract_changed and not families and not tasks):
            affected.append(
                {
                    "id": model.get("id"),
                    "name": model.get("name") or model.get("display_name") or model.get("id"),
                    "family": family or None,
                    "task": task or None,
                    "last_reviewed_fingerprint": audio_cfg.get(
                        "last_reviewed_fingerprint"
                    ),
                }
            )
    return affected


async def rescan_model_profiles(store, row: dict) -> List[dict]:
    """Force-refresh per-model session/request option profiles after activate."""
    from backend.engines.scan.scanner import scan_audio_cpp_model_profile

    results: List[dict] = []
    for model in store.list_models() or []:
        if not is_audio_cpp_model(model):
            continue
        model_id = str(model.get("id") or "")
        try:
            profile = await asyncio.to_thread(
                scan_audio_cpp_model_profile, store, row, model, force=True
            )
            results.append(
                {
                    "id": model_id,
                    "ok": not bool((profile or {}).get("scan_error")),
                    "scan_error": (profile or {}).get("scan_error"),
                    "fingerprint": (profile or {}).get("fingerprint"),
                }
            )
        except Exception as exc:
            logger.warning(
                "audio.cpp model profile rescan failed for %s: %s", model_id, exc
            )
            results.append(
                {"id": model_id, "ok": False, "scan_error": str(exc), "fingerprint": None}
            )
    return results


async def finish_audio_activation(store, row: dict) -> dict:
    """Scan, refresh model profiles, and restart the proxy for an active row."""
    from backend.engines.params import get_version_entry

    previous_entry = get_version_entry(store, "audio_cpp", str(row.get("version") or ""))
    scan_entry = None
    try:
        from backend.engines.scan.scanner import scan_engine_version

        scan_entry = await asyncio.to_thread(
            scan_engine_version, store, "audio_cpp", row
        )
    except Exception as exc:
        logger.warning("audio.cpp parameter scan failed after activation: %s", exc)

    profile_rescans: List[dict] = []
    try:
        profile_rescans = await rescan_model_profiles(store, row)
    except Exception as exc:
        logger.warning("audio.cpp model profile rescans failed after activation: %s", exc)

    try:
        from backend.proxy.llama_swap.manager import (
            get_llama_swap_manager,
            mark_swap_config_stale,
        )

        mark_swap_config_stale()
        await get_llama_swap_manager().start_proxy()
    except Exception as exc:
        logger.warning("Could not start llama-swap after audio.cpp activation: %s", exc)

    delta = (
        (scan_entry or {}).get("capability_delta")
        if isinstance(scan_entry, dict) and (scan_entry or {}).get("capability_delta")
        else capability_delta(
            previous_entry, scan_entry if isinstance(scan_entry, dict) else None
        )
    )
    contract_changed = bool(
        isinstance(scan_entry, dict) and scan_entry.get("contract_changed")
    )
    return {
        "message": f"Activated audio.cpp version {row['version']}",
        "capability_delta": delta,
        "contract_fingerprint": (scan_entry or {}).get("contract_fingerprint")
        if isinstance(scan_entry, dict)
        else None,
        "contract_changed": contract_changed,
        "affected_models": models_affected_by_delta(
            store, delta, contract_changed=contract_changed
        ),
        "profiles_rescanned": profile_rescans,
    }
