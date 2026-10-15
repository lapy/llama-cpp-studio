"""audio.cpp work that runs after the shared activation lock is held."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

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


def _model_family_task(model: dict) -> Tuple[str, str]:
    config = model.get("config") if isinstance(model.get("config"), dict) else {}
    engines_cfg = config.get("engines") if isinstance(config.get("engines"), dict) else {}
    audio_cfg = (
        engines_cfg.get("audio_cpp")
        if isinstance(engines_cfg.get("audio_cpp"), dict)
        else {}
    )
    family = str(audio_cfg.get("family") or model.get("family") or "").strip().lower()
    task = str(audio_cfg.get("task") or model.get("task") or "").strip().lower()
    return family, task


def _profile_cache_is_fresh(store, row: dict, model: dict) -> bool:
    """True when a fingerprint-matching profile without a scan error is cached."""
    from backend.engines.params import get_model_profile_entry
    from backend.engines.scan.scanner import audio_cpp_model_profile_fingerprint

    try:
        fingerprint = audio_cpp_model_profile_fingerprint(row, model)
        cached = get_model_profile_entry(
            store, "audio_cpp", str(row.get("version") or ""), fingerprint
        )
    except Exception as exc:
        logger.debug("audio.cpp profile cache probe failed: %s", exc)
        return False
    return bool(cached and not cached.get("scan_error"))


def plan_profile_rescans(store, row: dict, delta: Optional[dict], *, contract_changed: bool) -> List[Tuple[dict, bool]]:
    """Select which audio models need a profile rescan after activation.

    Returns ``(model, force)`` pairs. A changed contract invalidates every
    profile; otherwise only models whose family/task intersects the capability
    delta, or whose cached profile is missing/failed, are refreshed.
    """
    delta = delta or {}
    touched_families = {
        str(item).strip().lower()
        for item in [
            *(delta.get("added_families") or []),
            *(delta.get("removed_families") or []),
        ]
        if str(item).strip()
    }
    touched_tasks = {
        str(item).strip().lower()
        for item in [
            *(delta.get("added_tasks") or []),
            *(delta.get("removed_tasks") or []),
        ]
        if str(item).strip()
    }
    plan: List[Tuple[dict, bool]] = []
    for model in store.list_models() or []:
        if not is_audio_cpp_model(model):
            continue
        family, task = _model_family_task(model)
        intersects = (family and family in touched_families) or (
            task and task in touched_tasks
        )
        if contract_changed:
            plan.append((model, True))
        elif intersects:
            plan.append((model, True))
        elif not _profile_cache_is_fresh(store, row, model):
            # Cold or previously failed profile: a non-forced scan fills the cache.
            plan.append((model, False))
    return plan


RESCAN_CONCURRENCY = 4


async def rescan_model_profiles(
    store, row: dict, plan: Optional[List[Tuple[dict, bool]]] = None
) -> List[dict]:
    """Refresh selected per-model profiles with bounded concurrency.

    ``plan`` is ``(model, force)`` pairs from :func:`plan_profile_rescans`;
    when omitted every audio model is force-rescanned (explicit rescan APIs).
    """
    from backend.engines.scan.scanner import scan_audio_cpp_model_profile

    if plan is None:
        plan = [
            (model, True)
            for model in store.list_models() or []
            if is_audio_cpp_model(model)
        ]
    semaphore = asyncio.Semaphore(RESCAN_CONCURRENCY)

    async def _one(model: dict, force: bool) -> dict:
        model_id = str(model.get("id") or "")
        async with semaphore:
            try:
                profile = await asyncio.to_thread(
                    scan_audio_cpp_model_profile, store, row, model, force=force
                )
            except Exception as exc:
                logger.warning(
                    "audio.cpp model profile rescan failed for %s: %s", model_id, exc
                )
                return {"id": model_id, "ok": False, "scan_error": str(exc), "fingerprint": None}
        return {
            "id": model_id,
            "ok": not bool((profile or {}).get("scan_error")),
            "scan_error": (profile or {}).get("scan_error"),
            "fingerprint": (profile or {}).get("fingerprint"),
        }

    if not plan:
        return []
    return list(await asyncio.gather(*(_one(model, force) for model, force in plan)))


def _binary_identity(row: dict) -> str:
    """Cheap identity of the installed engine binaries + specs, no probing."""
    import hashlib

    parts: List[str] = [str(row.get("source_commit") or row.get("source_ref") or "")]
    for field in ("cli_binary_path", "server_binary_path"):
        path = str(row.get(field) or "").strip()
        try:
            stat = os.stat(path)
            parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(f"{path}:missing")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def assert_activation_supported(store, row: dict) -> None:
    """Route gate: audio.cpp builds must answer ``--list-loaders --json``.

    Studio hard-requires modern engine builds (v0.9.1+): the composed contract
    is derived from JSON probes only, so an old build must be refused *before*
    it becomes active. A cached successful JSON scan for identical binaries
    skips the probe.
    """
    from fastapi import HTTPException

    from backend.engines.params import get_version_entry

    entry = get_version_entry(store, "audio_cpp", str(row.get("version") or ""))
    if _scan_can_be_skipped(store, row, entry):
        return

    from backend.engines.scan.scanner import (
        _abs_audio_path,
        _audio_cpp_workdir,
        _run_audio_cpp_loaders,
    )

    cli_path = _abs_audio_path(str(row.get("cli_binary_path") or ""))
    if not cli_path or not os.path.isfile(cli_path):
        raise HTTPException(
            status_code=400,
            detail=f"audio.cpp CLI binary is missing: {cli_path or '(unset)'}",
        )
    payload, error = _run_audio_cpp_loaders(cli_path, cwd=_audio_cpp_workdir(row, cli_path))
    if payload is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "This audio.cpp build does not provide --list-loaders --json "
                f"(v0.9.1 or newer is required). Update audio.cpp. ({error or 'no JSON'})"
            ),
        )


def _scan_can_be_skipped(store, row: dict, previous_entry: Optional[dict]) -> bool:
    """True when a prior successful scan covers identical binaries and specs."""
    if not isinstance(previous_entry, dict) or previous_entry.get("scan_error"):
        return False
    if not (previous_entry.get("capabilities") or {}).get("families"):
        return False
    stored = str(previous_entry.get("binary_identity") or "").strip()
    return bool(stored) and stored == _binary_identity(row)


async def finish_audio_activation(store, row: dict, on_stage=None) -> dict:
    """Scan, refresh model profiles, and restart the proxy for an active row."""
    from backend.engines.params import get_version_entry

    def _stage(message: str) -> None:
        if on_stage is not None:
            try:
                on_stage(message)
            except Exception:
                pass

    previous_entry = get_version_entry(store, "audio_cpp", str(row.get("version") or ""))
    scan_entry = None
    skipped_scan = False
    try:
        if _scan_can_be_skipped(store, row, previous_entry):
            scan_entry = previous_entry
            skipped_scan = True
            _stage("Engine unchanged - reusing cached scan")
            logger.info(
                "audio.cpp scan skipped: engine binaries and specs unchanged"
            )
        else:
            from backend.engines.scan.scanner import scan_engine_version

            _stage("Probing engine help and loaders")
            scan_entry = await asyncio.to_thread(
                scan_engine_version, store, "audio_cpp", row
            )
            if isinstance(scan_entry, dict) and not scan_entry.get("scan_error"):
                scan_entry = dict(scan_entry)
                scan_entry["binary_identity"] = _binary_identity(row)
                from backend.engines.params import upsert_version_entry

                try:
                    upsert_version_entry(
                        store, "audio_cpp", str(row.get("version") or ""), scan_entry
                    )
                except Exception as exc:
                    logger.debug("audio.cpp identity persist failed: %s", exc)
    except Exception as exc:
        logger.warning("audio.cpp parameter scan failed after activation: %s", exc)

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

    profile_rescans: List[dict] = []
    try:
        plan = plan_profile_rescans(store, row, delta, contract_changed=contract_changed)
        if plan:
            _stage(f"Refreshing {len(plan)} model profile(s)")
        profile_rescans = await rescan_model_profiles(store, row, plan)
    except Exception as exc:
        logger.warning("audio.cpp model profile rescans failed after activation: %s", exc)

    _stage("Restarting proxy")

    try:
        from backend.proxy.llama_swap.manager import (
            get_llama_swap_manager,
            mark_swap_config_stale,
        )

        mark_swap_config_stale()
        await get_llama_swap_manager().start_proxy()
    except Exception as exc:
        logger.warning("Could not start llama-swap after audio.cpp activation: %s", exc)

    return {
        "message": f"Activated audio.cpp version {row['version']}",
        "capability_delta": delta,
        "scan_skipped_unchanged": skipped_scan,
        "contract_fingerprint": (scan_entry or {}).get("contract_fingerprint")
        if isinstance(scan_entry, dict)
        else None,
        "contract_changed": contract_changed,
        "affected_models": models_affected_by_delta(
            store, delta, contract_changed=contract_changed
        ),
        "profiles_rescanned": profile_rescans,
    }
