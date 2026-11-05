"""Tests for GitHub fork detection used when labeling source builds."""

from backend.repo_identity import (
    github_owner_repo,
    is_github_fork,
    source_build_type_labels,
    source_build_type_labels_for_engine,
)


def test_github_owner_repo_https_and_ssh():
    assert github_owner_repo("https://github.com/ggerganov/llama.cpp.git") == (
        "ggerganov",
        "llama.cpp",
    )
    assert github_owner_repo("git@github.com:ikawrakow/ik_llama.cpp.git") == (
        "ikawrakow",
        "ik_llama.cpp",
    )


def test_same_owner_is_not_fork():
    assert not is_github_fork(
        "https://github.com/ggerganov/llama.cpp",
        "https://github.com/ggerganov/llama.cpp.git",
    )


def test_different_owner_is_fork():
    assert is_github_fork(
        "https://github.com/someone/llama.cpp.git",
        "https://github.com/ggerganov/llama.cpp.git",
    )


def test_source_build_type_labels_mark_fork():
    labels = source_build_type_labels(
        "https://github.com/alice/lmdeploy.git",
        "https://github.com/InternLM/lmdeploy.git",
    )
    assert labels["type"] == "fork"
    assert labels["install_type"] == "source"
    assert labels["is_fork"] is True


def test_patched_upstream_stays_patched():
    labels = source_build_type_labels(
        "https://github.com/ggerganov/llama.cpp.git",
        "https://github.com/ggerganov/llama.cpp.git",
        patches=True,
    )
    assert labels["type"] == "patched"
    assert labels["is_fork"] is False


def test_fork_wins_over_patched_display():
    labels = source_build_type_labels(
        "https://github.com/bob/llama.cpp.git",
        "https://github.com/ggerganov/llama.cpp.git",
        patches=True,
    )
    assert labels["type"] == "fork"
    assert labels["is_fork"] is True


def test_engine_helper_uses_canonical_map():
    labels = source_build_type_labels_for_engine(
        "audio_cpp",
        "https://github.com/other/audio.cpp.git",
    )
    assert labels["type"] == "fork"


def test_owner_compare_is_case_insensitive():
    assert not is_github_fork(
        "https://github.com/Ggerganov/llama.cpp.git",
        "https://github.com/ggerganov/llama.cpp.git",
    )


def test_www_github_host_parsed():
    assert github_owner_repo("https://www.github.com/InternLM/lmdeploy") == (
        "InternLM",
        "lmdeploy",
    )


def test_non_github_remote_vs_canonical_is_fork():
    assert is_github_fork(
        "https://gitlab.com/alice/llama.cpp.git",
        "https://github.com/ggerganov/llama.cpp.git",
    )


def test_empty_remote_is_not_fork():
    assert not is_github_fork("", "https://github.com/ggerganov/llama.cpp.git")


def test_same_owner_different_repo_name_is_not_fork():
    # Username matches canonical; repo rename alone is not treated as a fork.
    assert not is_github_fork(
        "https://github.com/ggerganov/llama.cpp-mirror.git",
        "https://github.com/ggerganov/llama.cpp.git",
    )
