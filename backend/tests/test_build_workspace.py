"""Recovery and isolation guarantees for the incremental build workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from backend.engines.build_workspace import (
    BuildWorkspace,
    WorkspaceBusy,
    WorkspaceError,
    WorkspaceGitError,
    WorkspaceIsolationError,
    ccache_environment,
    release_held,
    retarget_text_tree,
    workspace_key,
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit_file(repo, name, content, message):
    path = os.path.join(repo, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    _git(repo, "add", name)
    _git(repo, "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", message)


@pytest.fixture
def origin(tmp_path):
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _commit_file(repo, "src/main.c", "int main(void){return 0;}\n", "init")
    return repo


def test_workspace_key_changes_with_config_and_patches():
    base = workspace_key("llama_cpp", "https://example.com/llama.git", {"cuda": True})
    assert base == workspace_key("llama_cpp", "https://example.com/llama.git", {"cuda": True})
    assert base != workspace_key("llama_cpp", "https://example.com/llama.git", {"cuda": False})
    assert base != workspace_key("llama_cpp", "https://example.com/llama.git", {"cuda": True}, ["patch"])
    assert base != workspace_key("ik_llama", "https://example.com/llama.git", {"cuda": True})


def test_sync_preserves_build_dir_and_advances(origin, tmp_path):
    ws = BuildWorkspace.open("llama_cpp", str(origin), {"cuda": False}, root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        first = ws.sync_git(str(origin), "main")
        marker = os.path.join(ws.checkout_dir, "build", "kept.o")
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("keep")
        _commit_file(origin, "src/extra.c", "void extra(void){}\n", "extra")
        second = ws.sync_git(str(origin), "main")
        assert first != second
        assert os.path.isfile(marker)
        assert os.path.isfile(os.path.join(ws.checkout_dir, "src", "extra.c"))
        assert ws.read_state()["phase"] == "synced"
        assert ws.read_state()["head"] == second
    finally:
        ws.release()


def test_failed_sync_leaves_previous_checkout(origin, tmp_path):
    ws = BuildWorkspace.open("llama_cpp", str(origin), {"cuda": False}, root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        head = ws.sync_git(str(origin), "main")
        with pytest.raises(WorkspaceGitError):
            ws.sync_git(str(origin), "does-not-exist")
        assert subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ws.checkout_dir, text=True
        ).strip() == head
        assert os.path.isdir(ws.checkout_dir)
    finally:
        ws.release()


def test_stale_lock_is_recovered_and_live_lock_is_not(origin, tmp_path):
    root = tmp_path / "ws"
    first = BuildWorkspace.open("llama_cpp", str(origin), root=root)
    first.acquire()
    second = BuildWorkspace.open("llama_cpp", str(origin), root=root)
    with pytest.raises(WorkspaceBusy):
        second.acquire()
    lock_path = first._lock_path
    with open(lock_path, "r", encoding="utf-8") as handle:
        lock = json.load(handle)
    lock["pid"] = 2**22
    lock["cmdline"] = "gone"
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(lock, handle)
    first._held = False
    second.acquire()
    try:
        assert second._held
        assert second.read_state().get("recovered_from") in {"locked", "failed", ""}
    finally:
        second.release()


def test_publish_is_atomic_and_refuses_a_complete_snapshot(origin, tmp_path):
    ws = BuildWorkspace.open("audio_cpp", str(origin), root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        ws.sync_git(str(origin), "main")
        os.makedirs(os.path.join(ws.checkout_dir, "build", "bin"), exist_ok=True)
        binary = os.path.join(ws.checkout_dir, "build", "bin", "tool")
        with open(binary, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(binary, 0o755)
        dest = tmp_path / "versions" / "source-1"
        published = ws.publish_layout(
            {"source": ws.checkout_dir},
            str(dest),
            ["source/src/main.c"],
        )
        assert os.path.isfile(os.path.join(published, ".snapshot-complete"))
        assert not os.path.exists(str(dest) + ".incomplete")
        assert not os.path.isdir(os.path.join(os.path.dirname(dest), ".source-1.incomplete"))
        with pytest.raises(WorkspaceError):
            ws.publish_tree(ws.checkout_dir, published)
        assert os.path.isfile(os.path.join(published, "source", "src", "main.c"))
    finally:
        ws.release()


def test_publish_rejects_symlink_escape_and_leaves_no_install(origin, tmp_path):
    ws = BuildWorkspace.open("llama_cpp", str(origin), root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        ws.sync_git(str(origin), "main")
        outside = tmp_path / "secret"
        outside.write_text("nope", encoding="utf-8")
        os.symlink(outside, os.path.join(ws.checkout_dir, "leak"))
        dest = tmp_path / "versions" / "source-2"
        with pytest.raises(WorkspaceIsolationError):
            ws.publish_tree(ws.checkout_dir, str(dest))
        assert not dest.exists()
        hidden = tmp_path / "versions" / ".source-2.incomplete"
        assert not hidden.exists()
    finally:
        ws.release()


def test_retarget_stops_at_the_workspace_prefix(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    old = tmp_path / "workspace" / "source"
    new = tmp_path / "version" / "source"
    link = site / "pkg.egg-link"
    link.write_text(str(old) + "\n", encoding="utf-8")
    changed = retarget_text_tree(str(site), str(old), str(new))
    assert changed == 1
    assert link.read_text(encoding="utf-8").strip() == str(os.path.abspath(new))


def test_ccache_environment_skips_nvcc_when_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.engines.build_workspace.shutil.which",
        lambda name: "/usr/bin/ccache" if name == "ccache" else None,
    )
    monkeypatch.setattr(
        "backend.engines.build_workspace.ccache_supports_nvcc",
        lambda: False,
    )
    monkeypatch.setattr(
        "backend.engines.build_workspace.ccache_dir",
        lambda: str(tmp_path / "ccache"),
    )
    env = ccache_environment(str(tmp_path / "tree"), launchers=True, cuda=True)
    assert env["CCACHE_BASEDIR"] == str(tmp_path / "tree")
    assert env["CCACHE_SLOPPINESS"] == "file_macro,time_macros"
    assert env["CCACHE_COMPILERCHECK"] == "content"
    assert "CMAKE_C_COMPILER_LAUNCHER" in env
    assert "CMAKE_CUDA_COMPILER_LAUNCHER" not in env


def test_publish_rejects_a_binary_that_still_loads_the_workspace(origin, tmp_path):
    if not shutil.which("gcc") or not shutil.which("ldd"):
        pytest.skip("gcc and ldd are required")
    ws = BuildWorkspace.open("llama_cpp", str(origin), root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        ws.sync_git(str(origin), "main")
        lib_dir = os.path.join(ws.checkout_dir, "build", "bin")
        os.makedirs(lib_dir, exist_ok=True)
        lib = os.path.join(lib_dir, "libdemo.so")
        exe = os.path.join(lib_dir, "demo")
        src = os.path.join(lib_dir, "main.c")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write("void demo(void); int main(void){demo(); return 0;}\n")
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-x", "c", "-o", lib, "-"],
            input=b"void demo(void){}\n",
            check=True,
        )
        subprocess.run(
            ["gcc", "-o", exe, src, f"-Wl,-rpath,{lib_dir}", f"-L{lib_dir}", "-ldemo"],
            check=True,
        )
        dest = tmp_path / "versions" / "linked"
        if shutil.which("patchelf"):
            published = ws.publish_tree(ws.checkout_dir, str(dest), ["build/bin/demo"])
            assert os.path.isfile(os.path.join(published, "build", "bin", "demo"))
            linked = subprocess.check_output(
                ["ldd", os.path.join(published, "build", "bin", "demo")],
                text=True,
            )
            assert str(ws.path) not in linked
        else:
            with pytest.raises(WorkspaceIsolationError):
                ws.publish_tree(ws.checkout_dir, str(dest), ["build/bin/demo"])
            assert not dest.exists()
    finally:
        ws.release()


def test_publish_keeps_an_origin_binary_when_the_workspace_still_has_the_library(origin, tmp_path):
    if not shutil.which("gcc") or not shutil.which("ldd"):
        pytest.skip("gcc and ldd are required")
    ws = BuildWorkspace.open("llama_cpp", str(origin), root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        ws.sync_git(str(origin), "main")
        lib_dir = os.path.join(ws.checkout_dir, "build", "bin")
        os.makedirs(lib_dir, exist_ok=True)
        lib = os.path.join(lib_dir, "libdemo.so")
        exe = os.path.join(lib_dir, "demo")
        src = os.path.join(lib_dir, "main.c")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write("void demo(void); int main(void){demo(); return 0;}\n")
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-x", "c", "-o", lib, "-"],
            input=b"void demo(void){}\n",
            check=True,
        )
        subprocess.run(
            ["gcc", "-o", exe, src, "-Wl,-rpath,$ORIGIN", f"-L{lib_dir}", "-ldemo"],
            check=True,
        )
        dest = tmp_path / "versions" / "origin"
        published = ws.publish_tree(ws.checkout_dir, str(dest), ["build/bin/demo"])
        published_exe = os.path.join(published, "build", "bin", "demo")
        linked = subprocess.check_output(["ldd", published_exe], text=True)
        assert "libdemo.so" in linked
        assert str(ws.path) not in linked
        assert os.path.isfile(os.path.join(ws.checkout_dir, "build", "bin", "libdemo.so"))
    finally:
        ws.release()


def test_directory_lock_and_failed_release_are_recoverable(tmp_path):
    ws = BuildWorkspace.open("llama_cpp", "https://example.com/x.git", root=str(tmp_path / "ws"))
    os.makedirs(ws.path, exist_ok=True)
    os.makedirs(ws._lock_path)
    ws.acquire()
    try:
        assert ws._held
        assert not os.path.isdir(ws._lock_path)
        ws.note("building")
    finally:
        release_held(ws)
    assert not ws._held
    assert ws.read_state().get("phase") == "failed"
    again = BuildWorkspace.open("llama_cpp", "https://example.com/x.git", root=str(tmp_path / "ws"))
    again.acquire()
    try:
        assert again._held
        assert again.read_state().get("recovered_from") == "failed"
    finally:
        again.release()


def test_invalid_ref_is_rejected(tmp_path):
    ws = BuildWorkspace.open("llama_cpp", "https://example.com/x.git", root=str(tmp_path / "ws"))
    ws.acquire()
    try:
        with pytest.raises(WorkspaceGitError):
            ws.sync_git("https://example.com/x.git", "-evil")
        with pytest.raises(WorkspaceGitError):
            ws.sync_git("https://example.com/x.git", "feature/../../etc")
    finally:
        ws.release()
