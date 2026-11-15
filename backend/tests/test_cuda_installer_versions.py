import pytest

from backend.cuda_errors import CudaVersionUnsupported
from backend.cuda_installer import CUDAInstaller


@pytest.fixture
def installer(tmp_path):
    return CUDAInstaller(
        log_path=str(tmp_path / "logs" / "cuda.log"),
        state_path=str(tmp_path / "config" / "cuda.json"),
        download_dir=str(tmp_path / "downloads"),
    )


@pytest.mark.asyncio
async def test_available_versions_select_latest_patch_per_supported_line(installer, monkeypatch):
    packages = [
        {"Package": "cuda-toolkit-13-1", "Version": "13.1.1-1"},
        {"Package": "cuda-toolkit-12-8", "Version": "12.8.1-1"},
        {"Package": "cuda-toolkit-12-8", "Version": "12.8.2-1"},
        {"Package": "cuda-toolkit-12-8", "Version": "12.8.2-2"},
        {"Package": "cuda-toolkit-12-7", "Version": "12.7.1-1"},
        {"Package": "cuda-runtime-12-8", "Version": "12.8.9-1"},
        {"Package": "cuda-toolkit-99-0", "Version": "99.0.1-1"},
    ]

    async def fake_packages(_ubuntu_version):
        return packages

    monkeypatch.setattr(installer, "_get_repo_packages", fake_packages)

    versions = await installer.get_available_versions()

    assert versions[:3] == ["13.1.1", "13.0", "12.9"]
    assert "12.8.2" in versions
    assert "12.8" not in versions
    assert "12.7.1" in versions
    assert "12.6" in versions  # offline fallback for a missing repository line
    assert "12.8.9" not in versions
    assert "99.0.1" not in versions


@pytest.mark.asyncio
async def test_available_versions_fall_back_when_repository_is_unavailable(installer, monkeypatch):
    async def no_packages(_ubuntu_version):
        return []

    monkeypatch.setattr(installer, "_get_repo_packages", no_packages)

    assert await installer.get_available_versions() == installer.SUPPORTED_VERSIONS


def test_patch_version_builds_exact_archive_url(installer, monkeypatch):
    monkeypatch.setattr(installer, "_get_archive_target_version", lambda: "24.04")

    url = installer._get_archive_url("12.8.2")

    assert "cuda-12-8-2-download-archive" in url
    assert "target_version=24.04" in url


@pytest.mark.parametrize("version", ["12", "12.x", "12.8.2.1", "v12.8.2", ""])
def test_cuda_version_parser_rejects_non_release_versions(installer, version):
    with pytest.raises(ValueError, match="Expected major.minor"):
        installer._cuda_version_parts(version)


def test_patch_version_uses_supported_release_line(installer):
    assert installer._cuda_release_line("12.8.2") == "12.8"
    assert installer._cuda_release_line("13.0.1") == "13.0"


@pytest.mark.asyncio
async def test_install_accepts_patch_and_new_minor_in_supported_major(installer, monkeypatch):
    class DownloadReached(Exception):
        pass

    async def fake_log(_line):
        return None

    async def fake_fetch(version):
        assert version == "13.1.1"
        raise DownloadReached

    async def fake_available_versions():
        return ["13.1.1", "13.0.3", "12.8.2"]

    monkeypatch.setattr(installer, "_get_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(installer, "_broadcast_log_line", fake_log)
    monkeypatch.setattr(installer, "_fetch_download_url", fake_fetch)
    monkeypatch.setattr(installer, "get_available_versions", fake_available_versions)

    with pytest.raises(DownloadReached):
        await installer.install("13.1.1")


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["13.999.999", "13.6.1", "12.8.999"])
async def test_install_rejects_undiscovered_versions(installer, monkeypatch, version):
    async def fake_available_versions():
        return ["13.5.0", "13.0.3", "12.8.2"]

    monkeypatch.setattr(installer, "_get_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(installer, "get_available_versions", fake_available_versions)

    with pytest.raises(CudaVersionUnsupported, match="Unsupported or unavailable"):
        await installer.install(version)


@pytest.mark.asyncio
async def test_install_keeps_static_release_line_as_offline_fallback(installer, monkeypatch):
    class DownloadReached(Exception):
        pass

    async def no_discovered_versions():
        return installer.SUPPORTED_VERSIONS

    async def fake_log(_line):
        return None

    async def fake_fetch(version):
        assert version == "12.8"
        raise DownloadReached

    monkeypatch.setattr(installer, "_get_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(installer, "get_available_versions", no_discovered_versions)
    monkeypatch.setattr(installer, "_broadcast_log_line", fake_log)
    monkeypatch.setattr(installer, "_fetch_download_url", fake_fetch)

    with pytest.raises(DownloadReached):
        await installer.install("12.8")
