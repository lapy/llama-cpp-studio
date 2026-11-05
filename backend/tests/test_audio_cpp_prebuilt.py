"""audio.cpp release archives install only when the host CUDA is new enough."""

import hashlib
import io
import tarfile

from backend.engines.audio_cpp.prebuilt import (
    HostTarget,
    extract_archive,
    prebuilt_skip_reason,
    select_prebuilt,
)


def _asset(name: str) -> dict:
    return {
        "name": name,
        "browser_download_url": f"https://example.test/{name}",
    }


def _host(**overrides) -> HostTarget:
    values = {
        "cuda": (12, 8),
        "sms": (86,),
        "has_nvidia": True,
        "avx512": False,
        "linux_x64": True,
    }
    values.update(overrides)
    return HostTarget(**values)


CUDA_128 = _asset("audio-v0.9.1-bin-ubuntu-x64-cuda12.8-colab.tar.gz")
CUDA_124 = _asset("audio-v0.9.1-bin-ubuntu-x64-cuda12.4.tar.gz")
CPU = _asset("audio-v0.9.1-bin-ubuntu-x64-cpu.tar.gz")
CPU_PORTABLE = _asset("audio-v0.9.1-bin-ubuntu-x64-cpu-portable.tar.gz")


def test_same_host_cuda_installs_the_matching_package():
    plan = select_prebuilt([CUDA_128, CPU_PORTABLE], _host(cuda=(12, 8)))
    assert plan is not None
    assert plan.asset_name.endswith("cuda12.8-colab.tar.gz")
    assert plan.package_cuda == "12.8"
    assert plan.cli_asset_name.endswith("cpu-portable.tar.gz")
    assert "matches" in plan.reason


def test_newer_host_cuda_installs_the_older_package():
    plan = select_prebuilt([CUDA_128, CPU_PORTABLE], _host(cuda=(13, 0)))
    assert plan is not None
    assert plan.package_cuda == "12.8"
    assert plan.cli_url.endswith("cpu-portable.tar.gz")
    assert "newer" in plan.reason


def test_cuda_prebuilt_without_a_cpu_cli_archive_builds_from_source():
    host = _host(cuda=(12, 8))
    assert select_prebuilt([CUDA_128], host) is None
    assert "audiocpp_cli" in prebuilt_skip_reason([CUDA_128], host)


def test_older_host_cuda_builds_from_source():
    host = _host(cuda=(12, 4))
    assert select_prebuilt([CUDA_128], host) is None
    reason = prebuilt_skip_reason([CUDA_128], host)
    assert "older than package CUDA 12.8" in reason


def test_highest_package_the_host_can_run_wins():
    plan = select_prebuilt(
        [CUDA_124, CUDA_128, CPU_PORTABLE],
        _host(cuda=(13, 0), sms=()),
    )
    assert plan is not None
    assert plan.package_cuda == "12.8"


def test_uncovered_gpu_builds_from_source():
    host = _host(cuda=(12, 8), sms=(120,))
    assert select_prebuilt([CUDA_128], host) is None
    assert "covers this GPU" in prebuilt_skip_reason([CUDA_128], host)


def test_cpu_host_prefers_the_portable_archive():
    plan = select_prebuilt(
        [CPU, CPU_PORTABLE],
        _host(cuda=None, sms=(), has_nvidia=False, avx512=False),
    )
    assert plan is not None
    assert plan.asset_name.endswith("cpu-portable.tar.gz")


def test_avx512_cpu_host_uses_the_native_archive():
    plan = select_prebuilt(
        [CPU, CPU_PORTABLE],
        _host(cuda=None, sms=(), has_nvidia=False, avx512=True),
    )
    assert plan is not None
    assert plan.asset_name.endswith("cpu.tar.gz")
    assert "portable" not in plan.asset_name.split("x64-", 1)[1]


def test_extract_finds_server_and_checks_its_digest(tmp_path):
    server = b"audiocpp-server"
    digest = hashlib.sha256(server).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = io.BytesIO(server)
        info = tarfile.TarInfo("audiocpp_server")
        info.size = len(server)
        archive.addfile(info, payload)
        sums = f"{digest}  audiocpp_server\n".encode()
        sums_info = tarfile.TarInfo("SHA256SUMS")
        sums_info.size = len(sums)
        archive.addfile(sums_info, io.BytesIO(sums))
    archive_path = tmp_path / "audio.tar.gz"
    archive_path.write_bytes(buffer.getvalue())

    found = extract_archive(str(archive_path), str(tmp_path / "install"))
    assert found["server_binary_path"].endswith("audiocpp_server")
    assert found["cli_binary_path"] == ""
