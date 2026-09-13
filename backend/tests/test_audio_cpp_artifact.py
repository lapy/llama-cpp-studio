"""Canonical audio.cpp path / artifact contract."""

from backend.audio_cpp_artifact import (
    audio_model_ready,
    audio_model_path_ready,
    build_artifact_descriptor,
    build_builtin_artifact_descriptor,
    prefer_directory_model_path,
    resolve_audio_model_path,
    resolve_audio_bundle_root,
)


def test_prefer_directory_when_root_model_gguf_matches(tmp_path):
    root = tmp_path / "pkg"
    nested = root / "turbo"
    nested.mkdir(parents=True)
    gguf = nested / "weights.gguf"
    gguf.write_bytes(b"gguf")
    link = root / "model.gguf"
    link.symlink_to(gguf)

    assert prefer_directory_model_path(str(gguf), bundle_path=str(root)) == str(
        root.resolve()
    )
    assert resolve_audio_model_path(
        {"artifact": {"path": str(gguf), "bundle_path": str(root)}}
    ) == str(root.resolve())


def test_accepts_gguf_file_when_no_root_link(tmp_path):
    gguf = tmp_path / "alone.gguf"
    gguf.write_bytes(b"gguf")
    assert audio_model_path_ready(str(gguf))
    assert resolve_audio_model_path({"artifact": {"path": str(gguf)}}) == str(
        gguf.resolve()
    )


def test_build_artifact_descriptor_records_layout(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "model.gguf").write_bytes(b"g")
    artifact = build_artifact_descriptor(
        bundle_path=str(root),
        runtime_path=str(root),
        size=1,
    )
    assert artifact["layout"] == "directory"
    assert artifact["runtime_path"] == str(root.resolve())
    assert artifact["has_root_model_gguf"] is True
    assert artifact["package_kind"] == "prepared_bundle"


def test_builtin_identity_is_not_resolved_against_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = {"family": "builtin_audio_utils", "artifact": build_builtin_artifact_descriptor("rnnoise")}
    assert audio_model_ready(model)
    assert resolve_audio_model_path(model) == "rnnoise"
    assert resolve_audio_bundle_root(model) == ""
    assert not audio_model_path_ready("rnnoise")
    assert not audio_model_ready({"family": "other_family", "artifact": model["artifact"]})
    assert not audio_model_ready({"family": "builtin_audio_utils", "artifact": {"path": "rnnoise"}})


def test_invalid_builtin_identity_does_not_bypass_package_path_validation(tmp_path):
    model = {"family": "builtin_audio_utils", "artifact": {
        "package_kind": "builtin", "model_id": "../rnnoise", "path": str(tmp_path),
    }}
    assert not audio_model_ready(model)
    assert resolve_audio_model_path(model) == ""
