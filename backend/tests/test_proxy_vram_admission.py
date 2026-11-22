"""GPU-aware admission contract: estimates, devices, decisions, reservations.

The probe is faked with structured GPU payloads; no test touches NVML, a live
engine, or the public network. Registry tests assert a denial never reaches
``subprocess.Popen``.
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from backend.proxy.launch_spec import EnvSpec, LaunchSpec
from backend.proxy.studio import registry as registry_module
from backend.proxy.studio.registry import ProcessRegistry
from backend.proxy.studio.vram import VramAdmission, VramAdmissionError
from backend.proxy.studio.vram_estimates import (
    ESTIMATE_FILE_SIZE,
    ESTIMATE_OBSERVED,
    ESTIMATE_SETTINGS,
    Estimate,
    normalize_overrides,
    resolve_estimate,
    weights_only_lower_bound_mib,
)

MIB = 1024 * 1024
GIB = 1024 * MIB
requires_process_groups = pytest.mark.skipif(
    not hasattr(os, "killpg"), reason="Studio supervises POSIX process groups"
)


def gpu(index=0, *, free=24576, used=0, total=24576, uuid="", name="Fake GPU"):
    return {
        "index": index,
        "uuid": uuid,
        "name": name,
        "total_mib": total,
        "free_mib": free,
        "used_mib": used,
    }


def probe_with(*gpus, tooling="nvml", error=None):
    async def probe():
        return {
            "vendor": "nvidia" if gpus else None,
            "device_count": len(gpus),
            "gpus": [dict(item) for item in gpus],
            "cpu_only_mode": not gpus,
            "tooling": tooling,
            "source": "fake",
            "error": error,
        }

    return probe


def raising_probe(message="NVML is unavailable"):
    async def probe():
        raise RuntimeError(message)

    return probe


def spec(model="demo", devices="0", *, unset=False):
    values = {} if devices is None else {"CUDA_VISIBLE_DEVICES": devices}
    return LaunchSpec(
        model_id=model,
        engine_id="llama_cpp",
        engine_install_id="fake-build",
        executable=sys.executable,
        argv=["-u", "-c", "import time; time.sleep(60)"],
        cwd=None,
        env=EnvSpec(
            set=values,
            unset=["CUDA_VISIBLE_DEVICES"] if unset else [],
        ),
    )


def model_info(file_size=0):
    async def lookup(model_id):
        return {"id": model_id, "file_size": file_size}

    return lookup


def admission(
    *,
    gpus=(gpu(),),
    probe=None,
    file_size=0,
    mode="enforce",
    margin=1024,
    overrides=None,
):
    instance = VramAdmission(
        model_info=model_info(file_size),
        probe=probe if probe is not None else probe_with(*gpus),
    )
    instance.apply_settings(
        {
            "vram_admission": mode,
            "vram_safety_margin_mib": margin,
            "vram_estimates": dict(overrides or {}),
        }
    )
    activities = []
    instance.bind_activity(lambda kind, payload: activities.append({"type": kind, **payload}))
    return instance, activities


def test_estimate_precedence_is_settings_then_observed_then_file_size():
    observed = {"demo": {"mib": 9000}}
    file_size = 4 * GIB
    assert resolve_estimate(
        "demo", overrides={"demo": 12000}, observed=observed, file_size_bytes=file_size
    ) == Estimate(12000, ESTIMATE_SETTINGS)
    assert resolve_estimate("demo", observed=observed, file_size_bytes=file_size) == Estimate(
        9000, ESTIMATE_OBSERVED
    )
    assert resolve_estimate("demo", file_size_bytes=file_size) == Estimate(4096, ESTIMATE_FILE_SIZE)
    assert resolve_estimate("demo") == Estimate(None, "unknown")


def test_weights_only_lower_bound_rounds_up_and_ignores_missing_sizes():
    assert weights_only_lower_bound_mib(0) is None
    assert weights_only_lower_bound_mib(None) is None
    assert weights_only_lower_bound_mib("not a size") is None
    assert weights_only_lower_bound_mib(1) == 1
    assert weights_only_lower_bound_mib(3 * GIB + 1) == 3073


def test_overrides_drop_invalid_entries_without_raising():
    assert normalize_overrides({"demo": 4096, " ": 1, "zero": 0, "huge": 10**9}) == {"demo": 4096}
    assert normalize_overrides(["not", "a", "mapping"]) == {}


def test_launch_device_selection_reads_the_compiled_env_contract():
    from backend.proxy.launch_spec import launch_device_selection

    assert launch_device_selection(spec(devices="0,GPU-abc")) == ("0", "GPU-abc")
    assert launch_device_selection(spec(devices=None)) == ("all",)
    assert launch_device_selection(spec(devices="-1")) is None
    assert launch_device_selection(spec(devices=None, unset=True)) is None


@pytest.mark.asyncio
async def test_off_mode_does_not_probe_or_record_a_decision():
    probes = []

    async def probe():
        probes.append(1)
        return {
            "vendor": "nvidia",
            "gpus": [gpu()],
            "tooling": "nvml",
            "error": None,
        }

    instance = VramAdmission(model_info=model_info(), probe=probe)
    instance.apply_settings({"vram_admission": "off"})
    assert await instance.admit(spec()) is None
    assert probes == []
    snapshot = instance.snapshot()
    assert snapshot["mode"] == "off"
    assert snapshot["last_admission"] is None


@pytest.mark.asyncio
async def test_enforce_rejects_a_settings_deficit_with_a_structured_payload():
    instance, activities = admission(
        gpus=[gpu(free=8192)], mode="enforce", overrides={"demo": 16000}
    )
    with pytest.raises(VramAdmissionError) as caught:
        await instance.admit(spec())
    payload = caught.value.payload()
    assert payload["code"] == "INSUFFICIENT_VRAM"
    assert payload["error"] == caught.value.decision["reason"]
    detail = payload["detail"]
    assert detail["decision"] == "rejected"
    assert detail["estimate_mib"] == 16000
    assert detail["estimate_source"] == ESTIMATE_SETTINGS
    assert detail["available_mib"] == 8192 - 1024
    assert detail["deficit_mib"] == 16000 - (8192 - 1024)
    assert detail["devices"][0]["free_mib"] == 8192
    assert detail["devices"][0]["margin_mib"] == 1024
    assert [item["type"] for item in activities] == ["vram_rejected"]
    assert activities[0]["model"] == "demo"
    assert instance.snapshot()["last_admission"]["decision"] == "rejected"


@pytest.mark.asyncio
async def test_enforce_warns_but_admits_a_derived_only_deficit():
    instance, activities = admission(gpus=[gpu(free=8192)], mode="enforce", file_size=16 * GIB)
    token = await instance.admit(spec())
    assert token is not None
    decision = instance.snapshot()["last_admission"]
    assert decision["decision"] == "warned"
    assert decision["estimate_source"] == ESTIMATE_FILE_SIZE
    assert decision["deficit_mib"] > 0
    assert [item["type"] for item in activities] == ["vram_warning"]
    # A warned start still reserves its estimate so concurrent starts see it.
    assert instance.snapshot()["reservations"][0]["mib"] == 16384


@pytest.mark.asyncio
async def test_warn_mode_admits_a_known_deficit():
    instance, activities = admission(gpus=[gpu(free=8192)], mode="warn", overrides={"demo": 16000})
    token = await instance.admit(spec())
    assert token is not None
    assert instance.snapshot()["last_admission"]["decision"] == "warned"
    assert [item["type"] for item in activities] == ["vram_warning"]


@pytest.mark.asyncio
async def test_unknown_estimate_and_probe_failure_admit_with_unknown():
    unknown, unknown_activities = admission(gpus=[gpu(free=8192)], mode="enforce")
    token = await unknown.admit(spec())
    assert token is not None
    assert unknown.snapshot()["last_admission"]["decision"] == "unknown"
    assert [item["type"] for item in unknown_activities] == ["vram_unverified"]
    assert unknown.snapshot()["probe"]["status"] == "ok"

    failed, failed_activities = admission(probe=raising_probe(), mode="enforce")
    assert await failed.admit(spec()) is None
    decision = failed.snapshot()["last_admission"]
    assert decision["decision"] == "unknown"
    assert "NVML is unavailable" in decision["reason"]
    assert failed.snapshot()["probe"]["status"] == "failed"
    assert [item["type"] for item in failed_activities] == ["vram_unverified"]


@pytest.mark.asyncio
async def test_cpu_and_absent_nvidia_tooling_skip_quietly():
    cpu, cpu_activities = admission(mode="enforce", overrides={"demo": 16000})
    assert await cpu.admit(spec(devices=None, unset=True)) is None
    assert cpu.snapshot()["last_admission"]["decision"] == "skipped"
    assert cpu_activities == []

    absent, absent_activities = admission(gpus=(), probe=probe_with(tooling="none"), mode="enforce")
    assert await absent.admit(spec()) is None
    assert absent.snapshot()["last_admission"]["decision"] == "skipped"
    assert absent_activities == []


@pytest.mark.asyncio
async def test_device_selection_maps_numeric_ids_uuids_and_rejects_unknown_identity():
    gpus = [
        gpu(0, uuid="GPU-aaaa", free=8192),
        gpu(1, uuid="GPU-bbbb", free=8192),
    ]
    instance, _ = admission(gpus=gpus, mode="warn", overrides={"demo": 6000})
    await instance.admit(spec(devices="1"))
    assert [item["index"] for item in instance.snapshot()["last_admission"]["devices"]] == [1]

    await instance.admit(spec(devices="GPU-bbbb"))
    assert [item["index"] for item in instance.snapshot()["last_admission"]["devices"]] == [1]

    await instance.admit(spec(devices="GPU-missing"))
    unknown = instance.snapshot()["last_admission"]
    assert unknown["decision"] == "unknown"
    assert "GPU-missing" in unknown["reason"]


@pytest.mark.asyncio
async def test_inherit_mode_preflights_every_visible_device():
    instance, _ = admission(
        gpus=[gpu(0, free=8192), gpu(1, free=8192)], mode="warn", overrides={"demo": 12000}
    )
    await instance.admit(spec(devices=None))
    decision = instance.snapshot()["last_admission"]
    assert [item["index"] for item in decision["devices"]] == [0, 1]
    assert decision["available_mib"] == 2 * (8192 - 1024)
    assert decision["decision"] == "admitted"


@pytest.mark.asyncio
async def test_in_flight_reservation_blocks_overcommit_and_releases():
    instance, _ = admission(
        gpus=[gpu(free=10000)], mode="enforce", margin=0, overrides={"a": 8000, "b": 8000}
    )
    first = await instance.admit(spec("a"))
    assert first is not None
    reservation = instance.snapshot()["reservations"][0]
    assert (reservation["model"], reservation["devices"], reservation["mib"]) == ("a", [0], 8000)
    with pytest.raises(VramAdmissionError):
        await instance.admit(spec("b"))
    instance.release(first)
    assert instance.snapshot()["reservations"] == []
    assert await instance.admit(spec("b")) is not None


@pytest.mark.asyncio
async def test_ready_observation_records_an_observed_estimate_and_running_row():
    samples = []

    async def probe():
        used = 1000 if not samples else 9000
        samples.append(used)
        return await probe_with(gpu(free=24576 - used, used=used))()

    instance, _ = admission(probe=probe, mode="warn", file_size=2000)
    token = await instance.admit(spec())
    assert instance.snapshot()["last_admission"]["estimate_source"] == ESTIMATE_FILE_SIZE
    assert instance.snapshot()["reservations"]

    generation = SimpleNamespace(vram=None)
    await instance.observe_ready(generation, token)

    assert generation.vram["observed_mib"] == 8000
    assert generation.vram["quality"] == "observed"
    assert generation.vram["estimate_source"] == ESTIMATE_FILE_SIZE
    assert instance.snapshot()["reservations"] == []
    assert instance.snapshot()["observed_estimates"]["demo"]["mib"] == 8000
    assert instance.catalog_estimate("demo", 2000) == {"mib": 8000, "source": ESTIMATE_OBSERVED}


@pytest.mark.asyncio
async def test_contended_ready_observation_is_displayed_but_not_trusted():
    samples = []

    async def probe():
        used = 1000 if not samples else 9000
        samples.append(used)
        return await probe_with(gpu(free=24576 - used, used=used))()

    instance, _ = admission(probe=probe, mode="warn", overrides={"a": 4000, "b": 4000})
    token = await instance.admit(spec("a"))
    from backend.proxy.studio.vram import _Reservation

    instance._reservations["other"] = _Reservation(
        token="other", model="b", devices=(0,), mib=4000, basis=ESTIMATE_SETTINGS, at=0.0
    )
    generation = SimpleNamespace(vram=None)
    await instance.observe_ready(generation, token)
    assert generation.vram["quality"] == "contended"
    assert "a" not in instance.snapshot()["observed_estimates"]


@pytest.mark.asyncio
async def test_registry_denial_never_spawns_a_process(monkeypatch):
    instance, _ = admission(gpus=[gpu(free=4096)], mode="enforce", overrides={"demo": 16000})
    spawned = []

    def fail_popen(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("a refused start must not spawn a process")

    monkeypatch.setattr(registry_module.subprocess, "Popen", fail_popen)
    factory_calls = []

    async def factory(model_id):
        factory_calls.append(model_id)
        return spec(model_id), "revision", None

    registry = ProcessRegistry(factory, admission=instance)
    with pytest.raises(VramAdmissionError):
        await registry.acquire("demo")
    assert factory_calls == ["demo"]
    assert spawned == []
    assert registry.snapshot() == []
    # A refusal is visible in the same activity ring as lifecycle events.
    activities = registry.activities.values()
    assert [item["type"] for item in activities if item["type"] == "vram_rejected"] == [
        "vram_rejected"
    ]
    assert activities[-1]["model"] == "demo"
    assert activities[-1]["deficit_mib"] > 0


@pytest.mark.asyncio
async def test_registry_releases_the_reservation_when_the_spawn_fails():
    instance, _ = admission(gpus=[gpu()], mode="enforce", overrides={"demo": 4000})

    async def factory(model_id):
        return (
            LaunchSpec(
                model_id=model_id,
                engine_id="llama_cpp",
                engine_install_id="fake-build",
                executable="/nonexistent/studio-test-engine",
                argv=[],
                cwd=None,
                env=EnvSpec(),
            ),
            "revision",
            None,
        )

    registry = ProcessRegistry(factory, admission=instance)
    try:
        with pytest.raises(FileNotFoundError):
            await registry.acquire("demo")
        assert instance.snapshot()["reservations"] == []
    finally:
        await registry.close()


@requires_process_groups
@pytest.mark.asyncio
async def test_successful_start_reports_running_vram_and_releases_at_ready(monkeypatch):
    samples = []

    async def probe():
        used = 1000 if not samples else 9000
        samples.append(used)
        return await probe_with(gpu(free=24576 - used, used=used))()

    instance, _ = admission(probe=probe, mode="warn", overrides={"demo": 4000})

    async def factory(model_id):
        return spec(model_id), "revision", None

    registry = ProcessRegistry(factory, admission=instance)
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0))
    try:
        generation = await registry.acquire("demo")
        for _ in range(100):
            if instance.snapshot()["reservations"] == [] and generation.vram:
                break
            await asyncio.sleep(0.01)
        row = registry.snapshot()[0]
        assert row["vram"]["observed_mib"] == 8000
        assert instance.snapshot()["reservations"] == []
        assert instance.snapshot()["observed_estimates"]["demo"]["mib"] == 8000
        registry.release(generation)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_snapshot_reports_configuration_probe_and_decision_fields():
    instance, _ = admission(gpus=[gpu()], mode="warn", overrides={"demo": 4096})
    await instance.admit(spec())
    snapshot = instance.snapshot()
    assert snapshot["mode"] == "warn"
    assert snapshot["safety_margin_mib"] == 1024
    assert snapshot["estimates_configured"] == {"demo": 4096}
    assert snapshot["probe"]["status"] == "ok"
    assert snapshot["probe"]["devices"][0]["free_mib"] == 24576
    assert snapshot["last_admission"]["decision"] == "admitted"
    assert {"model", "decision", "estimate_mib", "estimate_source", "devices"} <= set(
        snapshot["last_admission"]
    )
