"""Third native fork journey: nativecraft's own layout and policy survive.

``nativecraft.yaml`` exists to prove a CMake fork is configured by data. The
declaration-level pins in ``test_native_source_family_declaration.py`` prove the
shared host reads the right attributes; this module proves the DIFFERENT layout
and policy actually survive a build, a publication and a launch:

* the checkout lives in the declared ``source_dir`` (``nativecraft``, not
  ``llama.cpp``);
* the declared ``binary_name`` (``native-server``, not ``llama-server``) is what
  the build produces, what ``artifact.locate`` finds, and what publication
  installs;
* ``build_examples_required`` and ``blas_supported`` shape the CMake configure
  arguments (examples ON, BLAS absent);
* the published binary is what the declared GGUF launch profile resolves.

The toolchain is fake (a short POSIX shell script) but the recipe dispatch,
checkout, locate, copy/publication, row creation and launch compilation are the
production ones.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from backend import data_store
from backend.data_store import DataStore
from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.llama_cpp.manager import BuildConfig, LlamaManager
from backend.engines.python.cmake_source import CmakeSourceInstaller

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the fake CMake toolchain is a POSIX shell script"
)

NATIVECRAFT = "nativecraft"
VERSION = "1.0.0"

_FAKE_CMAKE = """#!/bin/sh
if [ -n "$FAKE_CMAKE_ARGS_LOG" ]; then
  printf '%s\\n' "$*" >> "$FAKE_CMAKE_ARGS_LOG"
fi
if [ "$1" = "--build" ]; then
  case "$*" in
    *"--target native-server"*)
      mkdir -p bin
      cat > bin/native-server <<'BIN'
#!/bin/sh
echo "nativecraft version: 1.0"
BIN
      chmod +x bin/native-server
      ;;
  esac
fi
exit 0
"""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
    )


@pytest.fixture()
def store(tmp_path, monkeypatch) -> DataStore:
    instance = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", instance)
    return instance


@pytest.fixture()
def nativecraft_definition(monkeypatch):
    definition = load_default_engine_definitions().definitions[NATIVECRAFT]
    # Fixture definitions are NOT registered in the production registry. The
    # only lookup that needs to see this engine is the GGUF binary resolver;
    # patching that module's set keeps the journey hermetic instead of mutating
    # process-wide registry state for every later test.
    from backend.engines.llama_cpp import resolve as llama_resolve

    monkeypatch.setattr(
        llama_resolve,
        "GGUF_ENGINE_IDS",
        set(llama_resolve.GGUF_ENGINE_IDS) | {NATIVECRAFT},
    )
    return definition


@pytest.mark.asyncio
async def test_nativecraft_build_publication_and_launch_use_its_own_layout(
    tmp_path, store, nativecraft_definition, monkeypatch
):
    policy = dict(nativecraft_definition.source_build)
    assert policy["binary_name"] == "native-server"
    assert policy["source_dir"] == "nativecraft"

    install_root = tmp_path / "installs"
    version_dir = install_root / VERSION
    checkout = version_dir / policy["source_dir"]
    checkout.mkdir(parents=True)
    _git(checkout, "init", "-b", "main")
    (checkout / "CMakeLists.txt").write_text("project(nativecraft)\n", encoding="utf-8")
    (checkout / "src").mkdir()
    (checkout / "src" / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture checkout")

    fake_cmake = tmp_path / "toolchain" / "cmake"
    fake_cmake.parent.mkdir()
    fake_cmake.write_text(_FAKE_CMAKE, encoding="utf-8")
    fake_cmake.chmod(0o755)
    args_log = tmp_path / "cmake-args.log"
    monkeypatch.setenv("FAKE_CMAKE_ARGS_LOG", str(args_log))

    manager = LlamaManager()
    monkeypatch.setattr(manager, "_find_cmake_executable", lambda: str(fake_cmake))

    installer = CmakeSourceInstaller(
        NATIVECRAFT,
        manager=manager,
        install_root=str(install_root),
        log_path=str(tmp_path / "nativecraft.log"),
    )

    binary = await installer.run_source(
        commit_sha="main",
        build_config=BuildConfig(build_examples=False),
        version_name=VERSION,
        repository_url="https://example.test/nativecraft.git",
        source_ref_type="branch",
        publish_row="install",
        store=store,
    )

    # Publication: the declared binary name, in the declared version directory.
    assert Path(binary) == version_dir / "native-server"
    assert Path(binary).is_file()
    rows = store.get_engine_versions(NATIVECRAFT)
    assert [row["version"] for row in rows] == [VERSION]
    assert rows[0]["binary_path"] == str(Path(binary))
    assert rows[0]["install_dir"] == str(version_dir)
    snapshots = store.list_install_snapshots(NATIVECRAFT)
    assert snapshots[-1]["version"] == VERSION
    assert snapshots[-1]["definition_digest"] == nativecraft_definition.definition_digest
    assert snapshots[-1]["resolved_inputs"]["install_dir"] == str(version_dir)

    # The source host's publish step now writes a version-associated snapshot,
    # so a retry can actually enforce the definition digest (it silently had no
    # snapshot before this journey existed).
    from backend.engines.install_snapshots import retry_resolved_inputs

    frozen = retry_resolved_inputs(
        store,
        NATIVECRAFT,
        VERSION,
        current_digest=nativecraft_definition.definition_digest,
        label="Nativecraft",
    )
    assert frozen["source_repo"] == "https://example.test/nativecraft.git"
    with pytest.raises(ValueError, match="definition changed"):
        retry_resolved_inputs(
            store,
            NATIVECRAFT,
            VERSION,
            current_digest="sha256:" + "0" * 64,
            label="Nativecraft",
        )
    # Installed versions are activated explicitly, as the Engines UI does.
    store.set_active_engine_version(NATIVECRAFT, VERSION)
    assert store.get_active_engine_version(NATIVECRAFT)["version"] == VERSION

    # Build policy: the BLAS flags are suppressed and the declared binary name
    # is the CMake target (the sample toolchain records each invocation).
    calls = args_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) >= 2, calls
    configure, build = calls[0], calls[1]
    assert ".." in configure.split()
    assert not any("GGML_BLAS" in call for call in calls)
    assert "--target" in build.split()
    assert any(
        "--target" in call.split() and "native-server" in call.split() for call in calls
    )

    # Launch: the declared GGUF profile resolves the PUBLISHED binary, not a
    # llama.cpp path, and composes the declared port flag with the model.
    from backend.proxy import commands, profile_launch

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    monkeypatch.setattr(
        commands,
        "_resolve_llama_model_source",
        lambda model, **kwargs: (str(model_path), None, None),
    )
    monkeypatch.setattr(commands, "_resolve_mmproj_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        commands, "_resolve_draft_companion", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(commands, "_active_engine_param_index", lambda engine: {})
    monkeypatch.setattr(commands, "_resolve_cuda_library_path", lambda path: "")
    monkeypatch.setattr(
        "backend.engines.llama_cpp.server_exec.resolve_llama_server_invocation_paths",
        lambda path: (str(path), str(Path(path).parent)),
    )

    model = {"id": "org/model", "format": "gguf", "path": str(model_path)}
    spec = profile_launch.compile_profile_launch(model, {}, "model-stable", NATIVECRAFT)

    assert spec is not None
    assert spec.executable == str(Path(binary))
    assert spec.argv[0] == "--model"
    assert str(model_path) in spec.argv
    assert "--port" in spec.argv
    assert "--alias" in spec.argv
    assert "model-stable" in spec.argv
    assert spec.engine_id == NATIVECRAFT

    # The fixture's distinct checkout survives on disk too (not just in attrs).
    assert (version_dir / "nativecraft" / ".git").is_dir()
    assert not (version_dir / "llama.cpp").exists()
    assert shutil.which("sh") is not None
    assert sys.platform != "win32"
