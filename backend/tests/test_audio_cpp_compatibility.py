"""Compatibility ceiling and upstream-contract seam for audio.cpp versions.

Covers the dependency-free version parser, the supported/unsupported/unknown
boundaries, the API fields the backend surfaces (status, contract, and the
version rows the Engines UI reads), and the env-gated audio.cpp#852 probe.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from backend.engines.audio_cpp.compatibility import (
    MINIMUM_ENGINE_VERSION,
    VALIDATED_ENGINE_VERSION,
    compatibility_fields,
    engine_compatibility,
    parse_engine_version,
)


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


@pytest.fixture
def store(client, monkeypatch, tmp_path):
    """An isolated store, matching the pattern in test_audio_cpp_versions_api."""
    return _install_temp_store(monkeypatch, tmp_path)


def _executable(path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def _fake_server(directory, stdout: str = "", *, argv_marker=None) -> str:
    """An executable that prints *stdout*, optionally recording its argv."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env python3", "import sys"]
    if argv_marker is not None:
        lines.append(f"open({str(argv_marker)!r}, 'w').write(' '.join(sys.argv[1:]))")
    lines.append(f"sys.stdout.write({stdout!r})")
    return _executable(directory / "audiocpp_server", "\n".join(lines) + "\n")


def _add_audio_version(store, tmp_path, version: str, **overrides):
    binaries = tmp_path / "binaries" / version.replace("/", "-")
    binaries.mkdir(parents=True, exist_ok=True)
    server = _executable(binaries / "audiocpp_server", "#!/bin/sh\nexit 0\n")
    cli = _executable(binaries / "audiocpp_cli", "#!/bin/sh\nexit 0\n")
    store.update_engine_build_settings(
        "audio_cpp",
        {
            "tracking_ref": "main",
            "repository_url": "https://github.com/0xShug0/audio.cpp.git",
            "backend": "cpu",
        },
    )
    row = {
        "version": version,
        "server_binary_path": server,
        "cli_binary_path": cli,
        "repository_source": "audio.cpp",
        **overrides,
    }
    store.add_engine_version("audio_cpp", row)
    store.set_active_engine_version("audio_cpp", version)
    return row


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("v0.9.5", "0.9.5"),
        ("release-0.9.4", "0.9.4"),
        ("0.9.3", "0.9.3"),
        ("V0.9.2", "0.9.2"),
        (" release-0.5.1 ", "0.5.1"),
        ("v0.9.5-cuda", "0.9.5"),
        ("release-0.9.4+local", "0.9.4"),
        ("v0.2.0-windows-prebuilt", "0.2.0"),
        ("release-0.9", "0.9.0"),
    ],
)
def test_parse_engine_version_spellings(raw, expected):
    assert parse_engine_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "main",
        "master",
        "source-main-1730000000",
        "linux-cpu-local",
        "a" * 40,
        "deadbeef",
        "release-",
        "v0.9.5.1",
        "release-0.9.4.1",
    ],
)
def test_parse_engine_version_rejects_non_releases(raw):
    assert parse_engine_version(raw) is None


# ---------------------------------------------------------------------------
# Ceiling boundaries
# ---------------------------------------------------------------------------


def test_validated_ceiling_is_pinned():
    # Changing the ceiling is a compatibility decision: update this pin, the
    # durable matrix, and the way the release was actually exercised.
    assert VALIDATED_ENGINE_VERSION == "0.9.5"
    assert MINIMUM_ENGINE_VERSION == "0.9.1"


def test_engine_at_or_inside_the_ceiling_is_supported():
    for raw in (
        "v0.9.5",
        "release-0.9.5",
        "v0.9.4",
        "release-0.9.4",
        "0.9.3",
        "v0.9.2",
        "v0.9.1",
    ):
        compat = engine_compatibility(raw)
        assert compat["supported"] is True, raw
        assert compat["status"] == "supported"
        assert compat["validated_through"] == VALIDATED_ENGINE_VERSION
        assert VALIDATED_ENGINE_VERSION in compat["note"]


def test_engine_newer_than_the_ceiling_is_unsupported():
    for raw in ("v0.9.6", "release-0.9.6", "v0.10.0", "v0.10.0-cuda"):
        compat = engine_compatibility(raw)
        assert compat["supported"] is False, raw
        assert compat["status"] == "unsupported"
        assert VALIDATED_ENGINE_VERSION in compat["note"]


def test_engine_older_than_activation_floor_is_unsupported():
    compat = engine_compatibility("release-0.5.1")
    assert compat["supported"] is False
    assert compat["status"] == "unsupported"
    assert MINIMUM_ENGINE_VERSION in compat["note"]


def test_unknown_engine_is_neither_supported_nor_unsupported():
    for raw in ("main", "source-main-1730000000", "a" * 40, None, ""):
        compat = engine_compatibility(raw)
        assert compat["supported"] is None, raw
        assert compat["status"] == "unknown", raw
        assert compat["validated_through"] == VALIDATED_ENGINE_VERSION
        assert compat["note"]


def test_compatibility_fields_are_api_shaped():
    fields = compatibility_fields("v0.10.0")
    assert fields["validated_through"] == VALIDATED_ENGINE_VERSION
    assert fields["engine_supported"] is False
    assert VALIDATED_ENGINE_VERSION in fields["engine_compat_note"]
    assert "engine_version" not in fields


# ---------------------------------------------------------------------------
# Backend surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "expected_supported"),
    [
        ("v0.9.5", True),
        ("release-0.9.4", True),
        ("v0.9.3", True),
        ("v0.10.0", False),
        ("source-main-1730000000", None),
    ],
)
def test_status_reports_engine_compatibility(client, store, tmp_path, version, expected_supported):
    _add_audio_version(store, tmp_path, version)

    r = client.get("/api/audio-cpp/status")
    assert r.status_code == 200
    data = r.json()
    assert data["engine_version"] == version
    assert data["validated_through"] == VALIDATED_ENGINE_VERSION
    assert data["engine_supported"] is expected_supported
    assert data["engine_compat_note"]
    assert data["upstream_contract_available"] is False


def test_status_without_active_engine_is_unknown(client, store, monkeypatch):
    monkeypatch.setattr(
        "backend.engines.audio_cpp.tracking.resolve_bootstrap_tracking_ref",
        lambda: "main",
    )

    r = client.get("/api/audio-cpp/status")
    assert r.status_code == 200
    data = r.json()
    assert data["installed"] is False
    assert data["active"] is None
    assert data["engine_version"] is None
    assert data["validated_through"] == VALIDATED_ENGINE_VERSION
    assert data["engine_supported"] is None
    assert "unknown" in data["engine_compat_note"].lower()


def test_contract_reports_engine_compatibility(client, store, tmp_path, monkeypatch):
    _add_audio_version(store, tmp_path, "v0.10.0")
    descriptor = {
        "schema": "audio-cpp.studio-contract/v0",
        "operations": [],
        "task_routing": {},
        "families": [],
        "family_routing": {},
        "drift_warnings": [],
        "upstream_contract": False,
        "descriptor_fingerprint": "a" * 64,
    }
    monkeypatch.setattr(
        "backend.routes.audio_cpp_versions.get_version_entry",
        lambda *_args, **_kwargs: {
            "contract": descriptor,
            "contract_fingerprint": "f" * 64,
            "contract_changed": False,
            "scanned_at": "2026-11-21T00:00:00Z",
        },
    )

    r = client.get("/api/audio-cpp/contract")
    assert r.status_code == 200
    data = r.json()
    assert data["engine_version"] == "v0.10.0"
    assert data["validated_through"] == VALIDATED_ENGINE_VERSION
    assert data["engine_supported"] is False
    assert VALIDATED_ENGINE_VERSION in data["engine_compat_note"]
    assert data["schema"] == "audio-cpp.studio-contract/v0"
    assert data["contract_fingerprint"] == "f" * 64


def test_contract_without_active_engine_still_returns_409(client, store):
    r = client.get("/api/audio-cpp/contract")
    assert r.status_code == 409
    assert "No composed audio.cpp contract is available" in r.json()["detail"]


def test_version_rows_expose_engine_compatibility(client, store, tmp_path):
    _add_audio_version(store, tmp_path, "v0.10.0")

    r = client.get("/api/llama-versions")
    assert r.status_code == 200
    row = next(item for item in r.json() if item["id"] == "audio_cpp:v0.10.0")
    assert row["engine_version"] == "v0.10.0"
    assert row["validated_through"] == VALIDATED_ENGINE_VERSION
    assert row["engine_supported"] is False
    assert VALIDATED_ENGINE_VERSION in row["engine_compat_note"]


# ---------------------------------------------------------------------------
# Upstream adapter (audio.cpp#852 detection seam)
# ---------------------------------------------------------------------------


def _contract_document() -> dict:
    return {
        "schema": "audio.cpp.contract/v1",
        "operations": [
            {"id": "speech", "native_path": "/v1/audio/speech", "method": "POST"},
        ],
    }


def test_upstream_contract_probe_is_off_by_default(monkeypatch, tmp_path):
    from backend.engines.audio_cpp.integration import upstream_adapter

    marker = tmp_path / "executed"
    binary = _fake_server(tmp_path, "{}", argv_marker=marker)
    monkeypatch.delenv(upstream_adapter.UPSTREAM_CONTRACT_ENV, raising=False)

    assert upstream_adapter.fetch_upstream_contract({"server_binary_path": binary}) is None
    assert not marker.exists(), "the probe must not exec when the env gate is off"


def test_upstream_contract_probe_uses_describe_server_json(monkeypatch, tmp_path):
    from backend.engines.audio_cpp.integration import upstream_adapter

    marker = tmp_path / "argv"
    binary = _fake_server(tmp_path, json.dumps(_contract_document()), argv_marker=marker)
    monkeypatch.setenv(upstream_adapter.UPSTREAM_CONTRACT_ENV, "1")

    document = upstream_adapter.fetch_upstream_contract({"server_binary_path": binary})

    assert document == _contract_document()
    assert marker.read_text(encoding="utf-8") == "--describe-server --json"


def test_upstream_contract_probe_returns_none_for_absent_or_broken_documents(monkeypatch, tmp_path):
    from backend.engines.audio_cpp.integration import upstream_adapter

    monkeypatch.setenv(upstream_adapter.UPSTREAM_CONTRACT_ENV, "1")

    # No binary, no checkout.
    assert upstream_adapter.fetch_upstream_contract({}) is None
    assert (
        upstream_adapter.fetch_upstream_contract({"server_binary_path": str(tmp_path / "nope")})
        is None
    )

    # A binary that prints non-JSON, or JSON without operations, is not a contract.
    garbage = _fake_server(tmp_path / "garbage", "not json")
    assert upstream_adapter.fetch_upstream_contract({"server_binary_path": garbage}) is None
    empty = _fake_server(tmp_path / "empty", json.dumps({"operations": []}))
    assert upstream_adapter.fetch_upstream_contract({"server_binary_path": empty}) is None


def test_upstream_contract_probe_reads_known_contract_file(monkeypatch, tmp_path):
    from backend.engines.audio_cpp.integration import upstream_adapter

    checkout = tmp_path / "checkout"
    contract_path = checkout / "contracts" / "server-contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(_contract_document()), encoding="utf-8")
    monkeypatch.setenv(upstream_adapter.UPSTREAM_CONTRACT_ENV, "1")

    document = upstream_adapter.fetch_upstream_contract({"source_path": str(checkout)})
    assert document == _contract_document()


def test_upstream_contract_probe_times_out_cleanly(monkeypatch, tmp_path):
    from backend.engines.audio_cpp.integration import upstream_adapter

    slow = _executable(tmp_path / "audiocpp_server", "#!/bin/sh\nsleep 30\n")
    monkeypatch.setenv(upstream_adapter.UPSTREAM_CONTRACT_ENV, "1")
    monkeypatch.setattr(upstream_adapter, "_PROBE_TIMEOUT_SECONDS", 0.2)

    assert upstream_adapter.fetch_upstream_contract({"server_binary_path": slow}) is None


def test_status_reports_upstream_contract_availability(client, store, tmp_path, monkeypatch):
    from backend.engines.audio_cpp.integration import upstream_adapter

    binary = _fake_server(tmp_path, json.dumps(_contract_document()))
    _add_audio_version(store, tmp_path, "v0.9.3", server_binary_path=binary)
    monkeypatch.setenv(upstream_adapter.UPSTREAM_CONTRACT_ENV, "1")

    r = client.get("/api/audio-cpp/status")
    assert r.status_code == 200
    assert r.json()["upstream_contract_available"] is True
