"""Launch campaign G5 — discovery is YAML-declared, not Python-owned.

Contract pinned here:
1. a definition-backed engine scans ITS runtime-profile executable with the
   ``discovery.probe_argv_prefix`` the YAML declares and parses with the
   ``discovery.parser`` kind it names (proven with a probe that only works
   when the declared prefix is used);
2. generated engines NEVER consult definitions — the legacy adapter
   dispatch still owns them (byte-frozen, gate G2);
3. a definition without a discovery block, or naming a parser with no
   parse-only adapter, yields a typed ``scan_error`` instead of guessing.

No engine-id literals: the fixture is a synthetic definition
("discoverycraft") and the legacy coverage iterates the live GENERATED set.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.engines.definitions.schema import validate_definition
from backend.engines.scan import scanner as scanner_mod
from backend.proxy import profile_launch

DISCOVERYCRAFT_ID = "discoverycraft"

HELP_TEXT = """discoverycraft-server 0.1.0

----- common params -----
--model PATH            path to the model served by this fixture
--port PORT             port to listen on
--max-batch-size N      maximum concurrent requests
"""


def _definition(**overrides):
    raw = {
        "schema_version": 2,
        "id": DISCOVERYCRAFT_ID,
        "label": "Discoverycraft",
        "repository_source": "Example Org/Discoverycraft",
        "install_kind": "python_venv",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["venv_path"],
        "artifact_formats": ["safetensors"],
        "package_kinds": ["hf_snapshot"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "discovery": {"parser": "llama_help", "probe_argv_prefix": ["--scan-help"]},
        "runtime_profiles": {
            "default": {
                "default": True,
                "executable": {"kind": "venv_console_script", "name": "discoverycraft-server"},
                "model_binding": {
                    "mode": "flag",
                    "flag": "--model",
                    "source": "huggingface_id",
                },
                "port_binding": {"flag": "--port"},
            }
        },
    }
    raw.update(overrides)
    return validate_definition(raw, source="discoverycraft.yaml")


def _install_definitions(monkeypatch, *definitions):
    table = {definition.id: definition for definition in definitions}
    monkeypatch.setattr(
        profile_launch,
        "_REGISTRY",
        SimpleNamespace(get=lambda engine_id: table.get(str(engine_id or ""))),
    )


def _fake_venv(tmp_path, help_text: str = HELP_TEXT, name: str = "discoverycraft-server"):
    """Venv layout with one console script; it records the argv it was given."""
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    marker = tmp_path / "argv.txt"
    script = bin_dir / name
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"open({str(marker)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
        "if '--scan-help' in sys.argv:\n"
        f"    sys.stdout.write({help_text!r})\n"
        "else:\n"
        "    print('usage: ' + sys.argv[0] + ' --model PATH --port PORT')\n"
    )
    script.chmod(0o755)
    return str(tmp_path / "venv"), script, marker


def test_definition_scan_uses_yaml_probe_and_profile_executable(tmp_path, monkeypatch):
    definition = _definition()
    _install_definitions(monkeypatch, definition)
    venv, script, marker = _fake_venv(tmp_path)

    entry = scanner_mod.scan_definition_engine_version(
        DISCOVERYCRAFT_ID, {"version": "0.1.0", "venv_path": venv}
    )

    assert entry["scan_error"] is None, entry
    assert entry["binary_path"] == str(script)
    # The probe argv came from YAML: the fixture only prints parseable help
    # for --scan-help, so a non-empty catalog proves the declared prefix ran.
    assert marker.read_text() == "--scan-help"
    params = [param for section in entry["sections"] for param in section["params"]]
    keys = {param["key"] for param in params}
    assert {"max_batch_size", "port"} <= keys, keys


def test_definition_without_discovery_is_a_typed_error(tmp_path, monkeypatch):
    _install_definitions(monkeypatch, _definition(discovery=None))
    venv, _script, _marker = _fake_venv(tmp_path)

    entry = scanner_mod.scan_definition_engine_version(
        DISCOVERYCRAFT_ID, {"version": "0.1.0", "venv_path": venv}
    )

    assert entry["scan_error"]
    assert "must declare discovery.parser" in entry["scan_error"]
    assert entry["sections"] == []


def test_parser_without_parse_only_adapter_is_a_typed_error(tmp_path, monkeypatch):
    _install_definitions(
        monkeypatch,
        _definition(discovery={"parser": "audio_cpp", "probe_argv_prefix": ["--scan-help"]}),
    )
    venv, _script, _marker = _fake_venv(tmp_path)

    entry = scanner_mod.scan_definition_engine_version(
        DISCOVERYCRAFT_ID, {"version": "0.1.0", "venv_path": venv}
    )

    assert entry["scan_error"]
    assert "no parse-only adapter" in entry["scan_error"]


class _FakeSession:
    def __init__(self):
        self.task_id = "scan-task"
        self._finished = False
        self.catalogs = []

    def set_stage(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def log_command(self, *args, **kwargs):
        return None

    def log_capture(self, *args, **kwargs):
        return None

    def log_catalog(self, entry):
        self.catalogs.append(entry)

    def finish_from_entry(self, entry, profile=None):
        self._finished = True

    def detach(self):
        return None

    def fail(self, message):
        self._finished = True


def _scan_dispatch_probe(monkeypatch, definition):
    """Drive scan_engine_version with a fake store/session and record dispatch."""
    session = _FakeSession()
    monkeypatch.setattr(scanner_mod, "get_param_scan_session", lambda: None)
    monkeypatch.setattr(scanner_mod, "start_param_scan", lambda *a, **k: session)
    monkeypatch.setattr(scanner_mod, "upsert_version_entry", lambda *a, **k: None)
    monkeypatch.setattr(scanner_mod, "_clear_llama_flags_cache", lambda: None)
    _install_definitions(monkeypatch, definition)

    store = SimpleNamespace(
        get_engine_versions=lambda engine: [],
        get_active_engine_version=lambda engine: {},
    )
    row = {"version": "0.1.0", "venv_path": "/nonexistent"}

    calls = {"legacy": 0, "definition": 0}

    def legacy(*args, **kwargs):
        calls["legacy"] += 1
        return {"scan_error": "legacy adapter ran", "sections": []}

    def definition_scan(*args, **kwargs):
        calls["definition"] += 1
        return {"scan_error": "definition scan ran", "sections": []}

    monkeypatch.setattr(
        scanner_mod,
        "SCANNER_KIND_DISPATCH",
        {kind: legacy for kind in scanner_mod.DEFINITION_HELP_PARSERS},
    )
    monkeypatch.setattr(scanner_mod, "scan_definition_engine_version", definition_scan)
    return store, row, calls


def test_definition_backed_engine_uses_the_definition_scan(monkeypatch):
    store, row, calls = _scan_dispatch_probe(monkeypatch, _definition())
    spec = SimpleNamespace(scanner_kind="llama_help")
    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine: spec)

    scanner_mod.scan_engine_version(store, DISCOVERYCRAFT_ID, row)

    assert calls == {"legacy": 0, "definition": 1}
