"""Canonical audio.cpp path / artifact contract."""

from backend.audio_cpp_artifact import (
    audio_model_path_ready,
    build_artifact_descriptor,
    prefer_directory_model_path,
    resolve_audio_model_path,
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
