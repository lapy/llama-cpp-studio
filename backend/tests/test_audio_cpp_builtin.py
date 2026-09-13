"""Built-in utilities are advertised only after the engine inspect API accepts them."""

from types import SimpleNamespace

from backend.audio_cpp_artifact import BUILTIN_AUDIO_FAMILY
from backend.audio_cpp_builtin import discover_builtin_audio_packages


def test_builtin_discovery_requires_inspect_acceptance(tmp_path, monkeypatch):
    source = tmp_path / "src"
    assets = source / "assets" / "framework" / "audio_utilities" / "rnnoise"
    assets.mkdir(parents=True)
    (assets / "rnnoise.safetensors").write_bytes(b"x")
    (source / "model_specs").mkdir()
    (source / "model_specs" / f"{BUILTIN_AUDIO_FAMILY}.json").write_text(
        '{"schema_version": 1, "family": "builtin_audio_utils", "packages": [], '
        '"description": "Built-in denoise"}',
        encoding="utf-8",
    )
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o755)
    active = {
        "source_path": str(source),
        "cli_binary_path": str(cli),
    }
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        assert "--inspect" in argv and "--json" in argv
        assert "rnnoise" in argv
        return SimpleNamespace(
            returncode=0,
            stdout='{"family":"builtin_audio_utils","task_names":["s2s"],'
            '"tasks":[{"task":"s2s","modes":["offline"]}]}',
            stderr="",
        )

    monkeypatch.setattr("backend.audio_cpp_builtin.subprocess.run", fake_run)
    packages = discover_builtin_audio_packages(active, [BUILTIN_AUDIO_FAMILY])
    assert len(packages) == 1
    assert packages[0]["id"] == "builtin-rnnoise"
    assert packages[0]["source"]["model_id"] == "rnnoise"
    assert packages[0]["install_kind"] == "builtin"
    assert seen


def test_builtin_discovery_ignores_directory_names_the_engine_rejects(tmp_path, monkeypatch):
    source = tmp_path / "src"
    (source / "assets" / "framework" / "audio_utilities" / "mystery").mkdir(parents=True)
    (source / "model_specs").mkdir()
    (source / "model_specs" / f"{BUILTIN_AUDIO_FAMILY}.json").write_text(
        '{"schema_version": 1, "family": "builtin_audio_utils", "packages": []}',
        encoding="utf-8",
    )
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setattr(
        "backend.audio_cpp_builtin.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="unknown"),
    )
    assert discover_builtin_audio_packages(
        {"source_path": str(source), "cli_binary_path": str(cli)},
        [BUILTIN_AUDIO_FAMILY],
    ) == []
