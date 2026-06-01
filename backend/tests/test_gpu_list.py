"""Tests for lightweight GPU list probing and startup cache."""

import asyncio

from backend import gpu_detector
from backend.services import model_metadata


def test_collect_gpu_list_cpu_only_when_disabled(monkeypatch):
    monkeypatch.setattr(gpu_detector, "_gpu_detection_disabled", True)
    monkeypatch.setattr(gpu_detector, "_gpu_disable_reason", "test")

    data = gpu_detector._collect_gpu_list()

    assert data["cpu_only_mode"] is True
    assert data["gpus"] == []
    assert data["reason"] == "test"


def test_detect_nvidia_gpu_list_via_smi_parses_index_and_name(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert "--query-gpu=index,name" in cmd
        return type(
            "Result",
            (),
            {"stdout": "0, NVIDIA GeForce RTX 4090\n1, NVIDIA GeForce RTX 3080\n"},
        )()

    monkeypatch.setattr(gpu_detector, "_resolve_nvidia_smi", lambda: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_detector.subprocess, "run", fake_run)

    data = gpu_detector._detect_nvidia_gpu_list_via_smi()

    assert data["vendor"] == "nvidia"
    assert data["device_count"] == 2
    assert data["gpus"] == [
        {"index": 0, "name": "NVIDIA GeForce RTX 4090"},
        {"index": 1, "name": "NVIDIA GeForce RTX 3080"},
    ]
    assert data["cpu_only_mode"] is False


def test_warm_gpu_list_cache_stores_probe_result(monkeypatch):
    async def fake_probe():
        return {
            "vendor": "nvidia",
            "device_count": 1,
            "gpus": [{"index": 0, "name": "Test GPU"}],
            "cpu_only_mode": False,
        }

    monkeypatch.setattr(model_metadata, "probe_gpu_list", fake_probe)
    model_metadata._gpu_list_cache = None

    result = asyncio.run(model_metadata.warm_gpu_list_cache())

    assert result["device_count"] == 1
    assert model_metadata.get_startup_gpu_list() == result
