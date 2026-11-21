"""Audio.cpp latest-release canary checks (no external network).

The canary script is loaded by path because its filename is hyphenated. Every
test uses a fake ``audiocpp_cli`` shell script, a synthetic release tarball, and
a loopback HTTP server (the repository conftest permits localhost sockets) so
asset selection, digest verification, probe classification, contract auditing,
and report shape stay deterministic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from backend.engines.audio_cpp.integration.contract_registry import registry_descriptor

ROOT = Path(__file__).resolve().parents[2]
CANARY_PATH = ROOT / "scripts" / "audio-cpp-canary.py"
V093_FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "audio_cpp_v093_list_loaders.json"

# The registry is the source of truth for the composed operation count; tests
# must not pin it or every registry growth breaks them.
REGISTRY_OPERATION_COUNT = len(registry_descriptor())


def _load_canary():
    spec = importlib.util.spec_from_file_location("audio_cpp_canary", CANARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


canary = _load_canary()

V091_PAYLOAD = {
    "schema_version": 1,
    "loaders": [
        {
            "family": "omnivoice",
            "tasks": [{"id": "tts", "modes": ["offline", "streaming"]}],
            "api_endpoints": ["/v1/audio/speech"],
            "instructions_policy": "soft_tags",
        },
        {
            "family": "qwen3_asr",
            "tasks": [{"id": "asr", "modes": ["offline"]}],
            "api_endpoints": ["/v1/audio/transcriptions"],
        },
    ],
}


def _load_v093_payload() -> dict:
    return json.loads(V093_FIXTURE.read_text(encoding="utf-8"))


def _cli_script(payload: dict) -> bytes:
    body = json.dumps(payload)
    return (f"#!/bin/sh\ncat <<'CANARY_JSON'\n{body}\nCANARY_JSON\n").encode()


def _abi_cli_script(required: str = "2.43") -> bytes:
    message = (
        "./audiocpp_cli: /lib/x86_64-linux-gnu/libm.so.6: "
        f"version `GLIBC_{required}' not found (required by ./audiocpp_cli)"
    )
    return (f"#!/bin/sh\ncat >&2 <<'CANARY_ABI'\n{message}\nCANARY_ABI\nexit 1\n").encode()


def _model_manager_script(payload: list) -> bytes:
    return (f"import json\nprint(json.dumps({json.dumps(payload)}))\n").encode()


def _write_tar(path: Path, entries: dict[str, tuple[bytes, int]]) -> bytes:
    with tarfile.open(path, "w:gz") as archive:
        for name, (content, mode) in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    return path.read_bytes()


def _build_release_archive(
    tmp_path: Path,
    *,
    payload: dict | None = None,
    cli: bytes | None = None,
    model_manager: list | None = None,
    name: str = "audio-v-test-bin-ubuntu-x64-cpu-portable.tar.gz",
) -> tuple[Path, str, int]:
    entries = {
        "audiocpp_cli": (cli if cli is not None else _cli_script(payload or V091_PAYLOAD), 0o755),
        "audiocpp_server": (b"#!/bin/sh\nexit 0\n", 0o755),
        "model_specs/example.json": (b"{}", 0o644),
    }
    if model_manager is not None:
        entries["tools/model_manager_v2.py"] = (_model_manager_script(model_manager), 0o644)
    path = tmp_path / name
    body = _write_tar(path, entries)
    return path, hashlib.sha256(body).hexdigest(), len(body)


def _release(
    *,
    tag: str,
    url: str,
    digest: str,
    size: int,
    name: str = "audio-v-test-bin-ubuntu-x64-cpu-portable.tar.gz",
    include_cuda: bool = True,
) -> dict:
    assets = []
    if include_cuda:
        assets.append(
            {
                "name": "audio-v-test-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
                "browser_download_url": url + "?cuda",
                "size": 10,
                "digest": None,
            }
        )
    assets.append(
        {
            "name": name,
            "browser_download_url": url,
            "size": size,
            "digest": f"sha256:{digest}",
        }
    )
    return {
        "tag": tag,
        "name": tag,
        "html_url": f"https://example.invalid/releases/tag/{tag}",
        "published_at": None,
        "prerelease": False,
        "assets": assets,
        "source": "test",
    }


class _AssetServer:
    def __init__(self) -> None:
        self.routes: dict[str, bytes] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                body = outer.routes.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def add(self, path: str, body: bytes) -> str:
        self.routes[path] = body
        return self.base_url + path

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def asset_server():
    server = _AssetServer()
    try:
        yield server
    finally:
        server.close()


def _run(tmp_path: Path, release: dict, **kwargs) -> dict:
    options = canary.CanaryOptions(
        tag=str(release.get("tag") or ""),
        report=str(tmp_path / "canary-report.json"),
        http_timeout=10.0,
        probe_timeout=20.0,
        **kwargs,
    )
    return canary.run_canary(options, release=release)


def _check(report: dict, check_id: str) -> dict:
    return next(check for check in report["checks"] if check["id"] == check_id)


class TestAssetSelection:
    def test_selects_cpu_portable_prebuilt(self):
        release = _release(
            tag="v-test",
            url="https://example.invalid/cpu.tar.gz",
            digest="a" * 64,
            size=10,
        )
        asset, reason = canary.select_cpu_asset(release)
        assert asset["name"] == "audio-v-test-bin-ubuntu-x64-cpu-portable.tar.gz"
        assert "cpu-portable" in reason

    def test_asset_override_selects_exact_name(self):
        release = _release(
            tag="v-test",
            url="https://example.invalid/cpu.tar.gz",
            digest="a" * 64,
            size=10,
        )
        asset, reason = canary.select_cpu_asset(
            release, asset_name="audio-v-test-bin-ubuntu-x64-cuda12.8-colab.tar.gz"
        )
        assert asset["name"].endswith("cuda12.8-colab.tar.gz")
        assert "override" in reason

    def test_unknown_asset_override_names_the_missing_asset(self):
        release = _release(
            tag="v-test",
            url="https://example.invalid/cpu.tar.gz",
            digest="a" * 64,
            size=10,
        )
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.select_cpu_asset(release, asset_name="audio-missing.tar.gz")
        assert excinfo.value.kind == "download_integrity"
        assert "audio-missing.tar.gz" in str(excinfo.value)

    def test_release_without_linux_cpu_asset_fails(self):
        release = {
            "tag": "v-test",
            "assets": [
                {
                    "name": "audio-v-test-bin-macos-arm64-metal.tar.gz",
                    "browser_download_url": "https://example.invalid/macos.tar.gz",
                }
            ],
        }
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.select_cpu_asset(release)
        assert excinfo.value.kind == "download_integrity"
        assert "no Linux x64 CPU prebuilt" in str(excinfo.value)


class TestAssetIntegrity:
    def test_digest_match_and_missing_digest(self, tmp_path):
        asset_path = tmp_path / "asset.bin"
        asset_path.write_bytes(b"payload")
        actual = hashlib.sha256(b"payload").hexdigest()

        digest, warning = canary.verify_asset_digest(
            str(asset_path), {"name": "a", "digest": f"sha256:{actual.upper()}"}
        )
        assert digest == actual
        assert warning == ""

        digest, warning = canary.verify_asset_digest(str(asset_path), {"name": "a"})
        assert digest == actual
        assert "no published digest" in warning

    def test_digest_mismatch_is_download_integrity(self, tmp_path):
        asset_path = tmp_path / "asset.bin"
        asset_path.write_bytes(b"payload")
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.verify_asset_digest(
                str(asset_path), {"name": "a", "digest": "sha256:" + "b" * 64}
            )
        assert excinfo.value.kind == "download_integrity"
        assert "mismatch" in str(excinfo.value)

    def test_unsupported_digest_algorithm_is_download_integrity(self, tmp_path):
        asset_path = tmp_path / "asset.bin"
        asset_path.write_bytes(b"payload")
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.verify_asset_digest(
                str(asset_path), {"name": "a", "digest": "sha512:" + "a" * 64}
            )
        assert excinfo.value.kind == "download_integrity"

    def test_declared_size_mismatch_is_download_integrity(self, tmp_path):
        asset_path = tmp_path / "asset.bin"
        asset_path.write_bytes(b"payload")
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.verify_asset_size(str(asset_path), {"name": "a", "size": 3})
        assert excinfo.value.kind == "download_integrity"
        assert "size mismatch" in str(excinfo.value)

    def test_download_refuses_non_http_url(self, tmp_path):
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.download_asset("file:///etc/passwd", str(tmp_path / "asset.bin"), timeout=5.0)
        assert excinfo.value.kind == "download_integrity"
        assert "non-HTTP" in str(excinfo.value)


class TestLoaderContract:
    def test_v091_payload_parses_and_audits_clean(self):
        payload, capabilities = canary.parse_loaders_payload(V091_PAYLOAD)
        assert payload["schema_version"] == 1
        assert capabilities["families"] == ["omnivoice", "qwen3_asr"]
        audit = canary.audit_contract(capabilities)
        assert audit["counts"] == {
            "families": 2,
            "tasks": 2,
            "endpoints": 2,
            "operations": REGISTRY_OPERATION_COUNT,
        }
        assert audit["drift"]["warnings"] == []
        assert all(check.status == "passed" for check in audit["checks"])

    def test_v093_fixture_handles_new_task_vocabulary_without_drift(self):
        payload = _load_v093_payload()
        _, capabilities = canary.parse_loaders_payload(payload)
        audit = canary.audit_contract(capabilities)
        descriptor = audit["descriptor"]

        assert audit["counts"] == {
            "families": 8,
            "tasks": 9,
            "endpoints": 3,
            "operations": REGISTRY_OPERATION_COUNT,
        }
        assert set(canary.REQUIRED_OPERATIONS) <= set(audit["operations"]["available"])
        assert audit["endpoints"]["unresolved"] == []
        assert descriptor["drift_warnings"] == []
        for task in ("wake", "cls", "midi", "turn", "diar", "gen"):
            assert descriptor["task_routing"][task] == "tasks_run"
        assert descriptor["family_routing"]["zipvoice"]["operation_id"] == "speech"
        assert descriptor["family_routing"]["whistle_asr"]["operation_id"] == "transcriptions"
        assert all(check.status == "passed" for check in audit["checks"])

    def test_schema_version_mismatch_is_contract_drift(self):
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.parse_loaders_payload({"schema_version": 2, "loaders": []})
        assert excinfo.value.kind == "contract_drift"
        assert "schema_version" in str(excinfo.value)

    def test_empty_loader_list_is_contract_drift(self):
        with pytest.raises(canary.CanaryFailure) as excinfo:
            canary.parse_loaders_payload({"schema_version": 1, "loaders": []})
        assert excinfo.value.kind == "contract_drift"
        assert "no families" in str(excinfo.value)

    def test_unknown_endpoint_is_reported_by_exact_name(self):
        payload = {
            "schema_version": 1,
            "loaders": [
                {
                    "family": "future_family",
                    "tasks": {"tts": ["offline"]},
                    "api_endpoints": ["/v1/future/route"],
                }
            ],
        }
        _, capabilities = canary.parse_loaders_payload(payload)
        audit = canary.audit_contract(capabilities)
        endpoint_check = next(
            check for check in audit["checks"] if check.id == "advertised_endpoints"
        )
        assert endpoint_check.status == "failed"
        assert endpoint_check.failure == "contract_drift"
        assert "/v1/future/route" in endpoint_check.message
        assert audit["endpoints"]["unresolved"] == [
            {"family": "future_family", "endpoint": "/v1/future/route"}
        ]
        drift_check = next(check for check in audit["checks"] if check.id == "drift_warnings")
        assert drift_check.status == "failed"
        assert "/v1/future/route" in drift_check.details["warnings"][0]

    def test_allow_drift_only_relaxes_the_drift_check(self):
        payload = {
            "schema_version": 1,
            "loaders": [
                {
                    "family": "future_family",
                    "tasks": {"tts": ["offline"]},
                    "api_endpoints": ["/v1/future/route"],
                }
            ],
        }
        _, capabilities = canary.parse_loaders_payload(payload)
        audit = canary.audit_contract(capabilities, allow_drift=True)
        drift_check = next(check for check in audit["checks"] if check.id == "drift_warnings")
        endpoint_check = next(
            check for check in audit["checks"] if check.id == "advertised_endpoints"
        )
        assert drift_check.status == "warning"
        assert drift_check.details["allow_drift"] is True
        assert endpoint_check.status == "failed"


class TestProbeClassification:
    def _run_script(self, path: Path, body: bytes) -> "canary.ProbeResult":
        path.write_bytes(body)
        path.chmod(0o755)
        return canary.run_cli_probe(str(path), cwd=str(path.parent), timeout=20.0)

    def test_glibc_loader_error_is_abi_unsupported(self, tmp_path):
        probe = self._run_script(tmp_path / "audiocpp_cli", _abi_cli_script())
        failure = canary.classify_probe_failure(
            probe, cli_path=str(tmp_path / "audiocpp_cli"), host_glibc="2.39"
        )
        assert failure is not None
        assert failure.kind == "abi_unsupported"
        assert failure.details["required_glibc"] == "2.43"
        assert failure.details["host_glibc"] == "2.39"
        assert "GLIBC_2.43" in failure.message
        assert "2.39" in failure.message

    def test_missing_elf_interpreter_is_abi_unsupported(self, tmp_path):
        cli = tmp_path / "audiocpp_cli"
        probe = self._run_script(cli, b"#!/nonexistent/interpreter\n")
        failure = canary.classify_probe_failure(probe, cli_path=str(cli), host_glibc="2.39")
        assert failure is not None
        assert failure.kind == "abi_unsupported"
        assert failure.details["host_glibc"] == "2.39"

    def test_non_abi_exit_is_probe_error(self, tmp_path):
        cli = tmp_path / "audiocpp_cli"
        probe = self._run_script(cli, b"#!/bin/sh\necho boom >&2\nexit 3\n")
        failure = canary.classify_probe_failure(probe, cli_path=str(cli), host_glibc="2.39")
        assert failure is not None
        assert failure.kind == "probe_error"
        assert "exited 3" in failure.message

    def test_no_output_is_probe_error(self, tmp_path):
        cli = tmp_path / "audiocpp_cli"
        probe = self._run_script(cli, b"#!/bin/sh\nexit 0\n")
        failure = canary.classify_probe_failure(probe, cli_path=str(cli), host_glibc="2.39")
        assert failure is not None
        assert failure.kind == "probe_error"


class TestReleasePageFallback:
    def test_parse_release_assets_html_pairs_each_digest(self):
        html = """
        <ul>
          <li><a href="/0xShug0/audio.cpp/releases/download/v9/cpu-portable.tar.gz">cpu</a>
            <span>sha256:{first}</span></li>
          <li><a href="/0xShug0/audio.cpp/releases/download/v9/cuda.tar.gz">cuda</a>
            <span>sha256:{second}</span></li>
        </ul>
        """.format(first="a" * 64, second="b" * 64)
        assets = canary.parse_release_assets_html(html)
        assert [asset["name"] for asset in assets] == ["cpu-portable.tar.gz", "cuda.tar.gz"]
        assert assets[0]["digest"] == "sha256:" + "a" * 64
        assert assets[1]["digest"] == "sha256:" + "b" * 64
        assert assets[0]["browser_download_url"].endswith(
            "/releases/download/v9/cpu-portable.tar.gz"
        )


class TestEndToEnd:
    def test_passing_run_reports_contract_counts_and_report_shape(self, tmp_path, asset_server):
        archive, digest, size = _build_release_archive(
            tmp_path,
            model_manager=[
                {
                    "family": "omnivoice",
                    "id": "omnivoice_test",
                    "target_directory": "omnivoice",
                    "repo": "audio-cpp/test",
                }
            ],
        )
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(
            tmp_path,
            _release(tag="v-test", url=url, digest=digest, size=size),
        )

        assert report["passed"] is True
        assert report["failure_kinds"] == []
        assert report["schema"] == "audio-cpp-canary/v1"
        assert report["release"]["tag"] == "v-test"
        assert report["asset"]["sha256"] == digest
        assert report["counts"] == {
            "families": 2,
            "tasks": 2,
            "endpoints": 2,
            "operations": REGISTRY_OPERATION_COUNT,
        }
        assert report["required_operations"]["missing"] == []
        assert report["model_manager"]["packages"] == 1
        assert report["model_manager"]["identity_fields"] is True
        for key in (
            "generated_at",
            "passed",
            "failure_kinds",
            "release",
            "asset",
            "host",
            "counts",
            "endpoints",
            "required_operations",
            "drift",
            "model_manager",
            "checks",
            "duration_seconds",
            "summary",
        ):
            assert key in report
        check_ids = [check["id"] for check in report["checks"]]
        for check_id in (
            "release",
            "asset_selection",
            "download",
            "asset_digest",
            "archive_extraction",
            "model_manager",
            "cli_probe",
            "loader_json",
            "contract_composition",
            "required_operations",
            "advertised_endpoints",
            "drift_warnings",
        ):
            assert check_id in check_ids
        assert all(check["failure"] is None for check in report["checks"])
        written = json.loads((tmp_path / "canary-report.json").read_text(encoding="utf-8"))
        assert written["passed"] is True

    def test_abi_failure_fails_the_run_and_reports_glibc(self, tmp_path, asset_server):
        archive, digest, size = _build_release_archive(tmp_path, cli=_abi_cli_script("2.43"))
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(tmp_path, _release(tag="v-abi", url=url, digest=digest, size=size))

        assert report["passed"] is False
        assert report["failure_kinds"] == ["abi_unsupported"]
        cli_check = _check(report, "cli_probe")
        assert cli_check["failure"] == "abi_unsupported"
        assert cli_check["details"]["required_glibc"] == "2.43"
        assert cli_check["details"]["host_glibc"] == canary.host_glibc_version()
        assert "abi_unsupported" in report["summary"]
        model_manager = _check(report, "model_manager")
        assert model_manager["status"] == "warning"

    def test_digest_mismatch_fails_before_extraction(self, tmp_path, asset_server):
        archive, _, size = _build_release_archive(tmp_path)
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(tmp_path, _release(tag="v-bad-digest", url=url, digest="c" * 64, size=size))

        assert report["passed"] is False
        assert report["failure_kinds"] == ["download_integrity"]
        assert _check(report, "asset_digest")["status"] == "failed"
        assert all(check["id"] != "archive_extraction" for check in report["checks"])

    def test_unknown_endpoint_fails_the_run_with_exact_name(self, tmp_path, asset_server):
        payload = {
            "schema_version": 1,
            "loaders": [
                {
                    "family": "future_family",
                    "tasks": {"tts": ["offline"]},
                    "api_endpoints": ["/v1/future/route"],
                }
            ],
        }
        archive, digest, size = _build_release_archive(tmp_path, payload=payload)
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(tmp_path, _release(tag="v-drift", url=url, digest=digest, size=size))

        assert report["passed"] is False
        assert report["failure_kinds"] == ["contract_drift"]
        assert report["endpoints"]["unresolved"] == [
            {"family": "future_family", "endpoint": "/v1/future/route"}
        ]
        endpoint_check = _check(report, "advertised_endpoints")
        assert endpoint_check["failure"] == "contract_drift"
        assert "/v1/future/route" in endpoint_check["message"]

    def test_missing_model_manager_script_is_only_a_warning(self, tmp_path, asset_server):
        archive, digest, size = _build_release_archive(tmp_path)
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(tmp_path, _release(tag="v-warn", url=url, digest=digest, size=size))

        assert report["passed"] is True
        model_manager = _check(report, "model_manager")
        assert model_manager["status"] == "warning"
        assert "not shipped" in model_manager["message"]

    def test_empty_model_manager_catalog_is_contract_drift(self, tmp_path, asset_server):
        archive, digest, size = _build_release_archive(tmp_path, model_manager=[])
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(tmp_path, _release(tag="v-empty-catalog", url=url, digest=digest, size=size))

        assert report["passed"] is False
        assert report["failure_kinds"] == ["contract_drift"]
        model_manager = _check(report, "model_manager")
        assert model_manager["failure"] == "contract_drift"
        assert "no package rows" in model_manager["message"]

    def test_v093_fixture_passes_end_to_end(self, tmp_path, asset_server):
        archive, digest, size = _build_release_archive(tmp_path, payload=_load_v093_payload())
        url = asset_server.add("/cpu-portable.tar.gz", archive.read_bytes())
        report = _run(tmp_path, _release(tag="v0.9.3", url=url, digest=digest, size=size))

        assert report["passed"] is True
        assert report["counts"]["families"] == 8
        assert report["counts"]["tasks"] == 9
        assert report["counts"]["endpoints"] == 3
