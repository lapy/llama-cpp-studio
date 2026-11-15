"""Direct tests for NVIDIA CUDA metadata parsing and URL resolution.

The installer suites already pin delegation, discovery through a patched
``_get_repo_packages``, and the public API contract.  These tests exercise the
helpers that moved into ``backend.cuda_metadata`` with small in-memory
fixtures, so the apt stanza parser and archive-page URL patterns stay
independently testable without touching the network.
"""

import gzip

import pytest

from backend import cuda_metadata
from backend.cuda_installer import CUDAInstaller

PACKAGES_TEXT = """\
Package: cuda-toolkit-12-8
Version: 12.8.1-1
Filename: pool/cuda-toolkit-12-8_12.8.1-1_amd64.deb

Package: cuda-toolkit-12-8
Version: 12.8.2-1
Filename: pool/cuda-toolkit-12-8_12.8.2-1_amd64.deb
Description: ignored field
Homepage: https://example.test

Package: cuda-runtime-12-8
Version: 12.8.2-1
Filename: pool/cuda-runtime-12-8_12.8.2-1_amd64.deb
"""

RUNFILE_URL = (
    "https://developer.download.nvidia.com/compute/cuda/12.8.2/"
    "local_installers/cuda_12.8.2_570.86.10_linux.run"
)


def test_parse_packages_reads_a_packages_gz_payload():
    payload = gzip.compress(PACKAGES_TEXT.encode("utf-8"))

    packages = cuda_metadata.parse_packages(gzip.decompress(payload).decode("utf-8"))

    assert [package["Package"] for package in packages] == [
        "cuda-toolkit-12-8",
        "cuda-toolkit-12-8",
        "cuda-runtime-12-8",
    ]
    assert packages[1] == {
        "Package": "cuda-toolkit-12-8",
        "Version": "12.8.2-1",
        "Filename": "pool/cuda-toolkit-12-8_12.8.2-1_amd64.deb",
    }
    assert packages[2]["Package"] == "cuda-runtime-12-8"


def test_parse_packages_ignores_empty_payloads():
    assert cuda_metadata.parse_packages("") == []
    assert cuda_metadata.parse_packages("\n\n") == []


def test_resolve_available_versions_picks_the_latest_patch_per_supported_line():
    packages = [
        {"Package": "cuda-toolkit-12-8", "Version": "12.8.1-1"},
        {"Package": "cuda-toolkit-12-8", "Version": "12.8.2-1"},
        {"Package": "cuda-toolkit-12-9", "Version": "12.9.0-1"},
        {"Package": "cuda-toolkit-12-6", "Version": "12.6.3-1"},
        {"Package": "cuda-toolkit-11-7", "Version": "11.7.1-1"},
        {"Package": "cuda-toolkit-14-0", "Version": "14.0.0-1"},
        {"Package": "cuda-runtime-12-8", "Version": "12.9.9-1"},
    ]

    versions = cuda_metadata.resolve_available_versions(
        packages, supported_versions=["12.9", "12.8", "12.6"]
    )

    assert versions == ["12.9.0", "12.8.2", "12.6.3"]


def test_resolve_available_versions_keeps_unpublished_lines_as_fallbacks():
    assert cuda_metadata.resolve_available_versions([], supported_versions=["13.0", "12.8"]) == [
        "13.0",
        "12.8",
    ]


def test_select_repo_package_filters_and_picks_the_highest_version():
    packages = [
        {"Package": "libnccl2", "Version": "2.21.5-1+cuda12.8"},
        {"Package": "libnccl2", "Version": "2.22.3-1+cuda12.8"},
        {"Package": "libnccl2", "Version": "2.22.4-1+cuda12.6"},
        {"Package": "libnccl-dev", "Version": "2.22.3-1+cuda12.8"},
    ]

    selected = cuda_metadata.select_repo_package(
        packages, "libnccl2", version_prefix="2.22", version_contains="cuda12.8"
    )

    assert selected == {"Package": "libnccl2", "Version": "2.22.3-1+cuda12.8"}
    assert cuda_metadata.select_repo_package(packages, "missing") is None


def test_get_archive_url_pins_the_runfile_archive_query():
    assert cuda_metadata.get_archive_url("12.8.2", "24.04") == (
        "https://developer.nvidia.com/cuda-12-8-2-download-archive"
        "?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu"
        "&target_version=24.04&target_type=runfile_local"
    )
    assert cuda_metadata.get_archive_url("12.8", "22.04") == (
        "https://developer.nvidia.com/cuda-12-8-0-download-archive"
        "?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu"
        "&target_version=22.04&target_type=runfile_local"
    )


@pytest.mark.parametrize(
    ("ubuntu_version", "target_version"),
    [("ubuntu2404", "24.04"), ("ubuntu2204", "22.04"), ("ubuntu2004", "22.04")],
)
def test_get_archive_target_version_maps_repository_names(ubuntu_version, target_version):
    assert cuda_metadata.get_archive_target_version(ubuntu_version) == target_version


def test_extract_download_url_follows_an_escaped_href():
    html = (
        '{"Linux/x86_64/Ubuntu/24.04/runfile_local":'
        '{"href":\\u0022https://developer.download.nvidia.com/compute/cuda/12.8.2/'
        "local_installers/cuda_12.8.2_570.86.10_linux.run\\u0022}}"
    )

    assert cuda_metadata.extract_download_url(html, "12.8.2", "24.04") == RUNFILE_URL


def test_extract_download_url_builds_the_url_from_a_filename():
    html = (
        '{"Linux/x86_64/Ubuntu/24.04/runfile_local":{"filename":"cuda_12.8.2_570.86.10_linux.run"}}'
    )

    assert cuda_metadata.extract_download_url(html, "12.8.2", "24.04") == RUNFILE_URL


def test_extract_download_url_falls_back_to_a_bare_archive_url():
    html = f'<html><body><a href="{RUNFILE_URL}">download</a></body></html>'

    assert cuda_metadata.extract_download_url(html, "12.8.2", "24.04") == RUNFILE_URL


def test_extract_download_url_reports_a_page_without_a_runfile():
    with pytest.raises(RuntimeError, match="Could not find download URL for CUDA 12.8.2"):
        cuda_metadata.extract_download_url("<html>no runfile</html>", "12.8.2", "24.04")


@pytest.mark.asyncio
async def test_fetch_download_url_returns_the_cached_url_without_network():
    source = cuda_metadata.CudaMetadataSource()
    source._url_cache["12.8.2_linux_x86_64"] = "https://example.test/cuda.run"

    assert (
        await source.fetch_download_url("12.8.2", "https://unused.test", "24.04")
        == "https://example.test/cuda.run"
    )


@pytest.mark.asyncio
async def test_installer_metadata_helpers_delegate_to_the_source(tmp_path):
    class FakeMetadata:
        def __init__(self):
            self.repo_calls = []
            self.url_calls = []

        async def get_repo_packages(self, ubuntu_version):
            self.repo_calls.append(ubuntu_version)
            return [{"Package": "cuda-toolkit-12-8", "Version": "12.8.2-1"}]

        async def fetch_download_url(self, version, archive_url, target_version):
            self.url_calls.append((version, archive_url, target_version))
            return f"{archive_url}#{version}"

    installer = CUDAInstaller(
        log_path=str(tmp_path / "cuda.log"),
        state_path=str(tmp_path / "cuda.json"),
        download_dir=str(tmp_path / "downloads"),
    )
    metadata = FakeMetadata()
    installer._metadata = metadata

    assert installer._get_platform() == cuda_metadata.get_platform()
    assert installer._get_archive_target_version() in {"24.04", "22.04"}
    assert installer._cuda_version_parts("12.8.2") == (12, 8, 2)
    assert installer._cuda_release_line("12.8.2") == "12.8"
    assert installer._version_key("2.22.3-1") == (2, 22, 3, 1)
    assert installer._select_repo_package(
        [{"Package": "libnccl2", "Version": "2.22.3-1"}], "libnccl2"
    ) == {"Package": "libnccl2", "Version": "2.22.3-1"}
    assert await installer._get_repo_packages("ubuntu2404") == [
        {"Package": "cuda-toolkit-12-8", "Version": "12.8.2-1"}
    ]
    assert metadata.repo_calls == ["ubuntu2404"]

    archive_url = installer._get_archive_url("12.8.2")
    assert await installer._fetch_download_url("12.8.2") == f"{archive_url}#12.8.2"
    assert metadata.url_calls == [("12.8.2", archive_url, installer._get_archive_target_version())]
