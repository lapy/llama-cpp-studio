"""audio.cpp release archives install only when the host CUDA is new enough."""

import hashlib
import io
import os
import tarfile

import pytest

from backend.engines.audio_cpp import prebuilt as prebuilt_mod
from backend.engines.audio_cpp.prebuilt import (
    HostTarget,
    assert_prebuilt_binaries_runnable,
    extract_archive,
    parse_sha256_digest,
    prebuilt_skip_reason,
    preflight_env,
    select_prebuilt,
    stream_release_download,
)


def _asset(name: str, digest: str | None = None) -> dict:
    asset = {
        "name": name,
        "browser_download_url": f"https://example.test/{name}",
    }
    if digest is not None:
        asset["digest"] = digest
    return asset


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
    reason = prebuilt_skip_reason([CUDA_128], host)
    # Name both sides of the mismatch: the archive's compute capabilities and
    # the host's, so an installed CUDA toolkit is not misread as the cause.
    assert "compute capability 7.5, 8.0, 8.6, 8.9, 9.0" in reason
    assert "this host runs 12.0" in reason
    assert "CUDA toolkit does not add architectures" in reason


def test_mixed_gpu_host_reports_the_uncovered_capability():
    # V100 (7.0) plus RTX 3090 (8.6): the colab archive covers 8.6 but not 7.0.
    host = _host(cuda=(12, 8), sms=(70, 86))
    assert select_prebuilt([CUDA_128], host) is None
    reason = prebuilt_skip_reason([CUDA_128], host)
    assert "this host runs 7.0, 8.6" in reason


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


# --- GitHub asset digest -------------------------------------------------------

SHA = "ab" * 32


def test_asset_digest_is_parsed_into_the_plan():
    cuda = _asset(
        "audio-v0.9.5-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
        f"sha256:{SHA}",
    )
    cpu = _asset("audio-v0.9.5-bin-ubuntu-x64-cpu-portable.tar.gz", "sha256:" + "cd" * 32)
    plan = select_prebuilt([cuda, cpu], _host(cuda=(12, 8)))
    assert plan is not None
    assert plan.sha256 == SHA
    assert plan.cli_sha256 == "cd" * 32


def test_missing_or_malformed_digest_is_left_empty():
    plan = select_prebuilt([CPU], _host(cuda=None, sms=(), has_nvidia=False))
    assert plan is not None and plan.sha256 == ""
    plan = select_prebuilt(
        [_asset(CPU["name"], "md5:not-a-sha256")],
        _host(cuda=None, sms=(), has_nvidia=False),
    )
    assert plan is not None and plan.sha256 == ""


def test_parse_sha256_digest_normalizes_github_shape():
    assert parse_sha256_digest(f"sha256:{SHA}") == SHA
    assert parse_sha256_digest(SHA.upper()) == SHA
    assert parse_sha256_digest("sha256:") == ""
    assert parse_sha256_digest(None) == ""


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeHttpClient:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def stream(self, method, url, headers=None):
        return _FakeStreamResponse(self._chunks)


def _patch_http(monkeypatch, payload: bytes) -> None:
    chunks = [payload[index : index + 7] for index in range(0, len(payload), 7)]
    monkeypatch.setattr(prebuilt_mod.httpx, "AsyncClient", lambda **kwargs: _FakeHttpClient(chunks))


@pytest.mark.asyncio
async def test_stream_download_verifies_the_published_digest(monkeypatch, tmp_path):
    payload = b"release-archive-bytes"
    _patch_http(monkeypatch, payload)
    dest = tmp_path / "audio.tar.gz"
    digest = await stream_release_download(
        "https://example.test/audio.tar.gz",
        str(dest),
        expected_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert dest.read_bytes() == payload


@pytest.mark.asyncio
async def test_stream_download_deletes_and_rejects_a_mismatch(monkeypatch, tmp_path):
    _patch_http(monkeypatch, b"tampered")
    dest = tmp_path / "audio.tar.gz"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        await stream_release_download(
            "https://example.test/audio.tar.gz", str(dest), expected_sha256=SHA
        )
    assert not dest.exists()


@pytest.mark.asyncio
async def test_stream_download_without_a_digest_is_recorded_as_computed(monkeypatch, tmp_path):
    payload = b"unverified-release"
    _patch_http(monkeypatch, payload)
    dest = tmp_path / "audio.tar.gz"
    digest = await stream_release_download("https://example.test/audio.tar.gz", str(dest))
    assert digest == hashlib.sha256(payload).hexdigest()
    assert dest.exists()


# --- runnability preflight -----------------------------------------------------


def _script(path, body: str):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_preflight_accepts_runnable_prebuilt_binaries(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    server = _script(install / "audiocpp_server", 'echo "audiocpp_server v0.9.3"\n')
    cli = _script(install / "audiocpp_cli", 'echo "Usage: audiocpp_cli [options]"\n')

    result = assert_prebuilt_binaries_runnable(str(server), str(cli), install_dir=str(install))
    assert result == {"server": "ok", "cli": "ok"}


def test_preflight_rejects_a_binary_that_cannot_exec_here(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    server = _script(
        install / "audiocpp_server",
        'echo "$0: /lib/x86_64-linux-gnu/libc.so.6: version GLIBC_2.43 not found" >&2\nexit 1\n',
    )
    cli = _script(install / "audiocpp_cli", 'echo "Usage: audiocpp_cli"\n')

    with pytest.raises(RuntimeError) as excinfo:
        assert_prebuilt_binaries_runnable(str(server), str(cli), install_dir=str(install))

    message = str(excinfo.value)
    assert "audiocpp_server binary cannot run on this host" in message
    assert "GLIBC_2.43 not found" in message
    assert "host glibc:" in message
    assert "build this release from source" in message


def test_preflight_rejects_a_missing_binary(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    cli = _script(install / "audiocpp_cli", 'echo "Usage: audiocpp_cli"\n')

    with pytest.raises(RuntimeError, match="audiocpp_server binary is missing"):
        assert_prebuilt_binaries_runnable("", str(cli), install_dir=str(install))


def test_preflight_env_is_clean_and_puts_the_install_dir_on_the_library_path(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    server = install / "audiocpp_server"
    cli = install / "audiocpp_cli"

    env = preflight_env(str(install), [str(server), str(cli)])

    assert "LD_PRELOAD" not in env
    assert str(install) in env["LD_LIBRARY_PATH"].split(os.pathsep)


def test_preflight_probes_see_the_install_dir_on_the_library_path(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    body = (
        f'case ":$LD_LIBRARY_PATH:" in *":{install}:"*) echo ok ;; '
        '*) echo "install dir missing from LD_LIBRARY_PATH" >&2; exit 3 ;; esac\n'
    )
    server = _script(install / "audiocpp_server", body)
    cli = _script(install / "audiocpp_cli", body)

    result = assert_prebuilt_binaries_runnable(str(server), str(cli), install_dir=str(install))
    assert result == {"server": "ok", "cli": "ok"}
