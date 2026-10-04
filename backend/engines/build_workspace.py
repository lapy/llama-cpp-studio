"""Incremental compile workspace that never aliases an installed engine version.

New source builds fetch and compile here, then publish a private snapshot into a
new version directory. The installed tree is a copy: its binaries must not load
libraries from the workspace, and its git checkout must not be a worktree of it.

A crash leaves the workspace in place so the next build can resume. Publishing
writes a hidden staging directory and renames it into place only after the
snapshot has been checked. Sync and retry of an existing version do not take
this lock and do not write here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from backend.git_https import git_argv
from backend.logging_config import get_logger
from backend.paths import studio_data_dir

logger = get_logger(__name__)

SNAPSHOT_SENTINEL = ".snapshot-complete"
WORKSPACE_DIRNAME = "build-workspaces"
_PRESERVE_ON_CLEAN = (
    "build",
    "build/",
    ".cache",
    ".cache/",
    "dist",
    "dist/",
    "target",
    "target/",
    ".venv",
    ".venv/",
    "dependencies",
    "dependencies/",
)
_ORIGIN_CMAKE = "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"


class WorkspaceError(RuntimeError):
    """The workspace could not be used without risking an installed version."""


class WorkspaceBusy(WorkspaceError):
    """Another live process holds the workspace lock."""


class WorkspaceGitError(WorkspaceError):
    """Fetching or resetting the workspace checkout failed."""


class WorkspaceIsolationError(WorkspaceError):
    """The snapshot still depends on the workspace and was not published."""


def workspace_root() -> str:
    """Directory for compile state. It is not an engine install root."""
    return os.path.join(studio_data_dir(), WORKSPACE_DIRNAME)


def ccache_dir() -> str:
    override = os.getenv("STUDIO_CCACHE_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(studio_data_dir(), "ccache")


def workspace_key(
    engine: str,
    repo_url: str,
    config: Optional[dict] = None,
    patches: Optional[Sequence[str]] = None,
) -> str:
    """Stable id for one engine, repository, build configuration, and patch set."""
    payload = {
        "engine": str(engine or "").strip(),
        "repo": str(repo_url or "").strip(),
        "config": config or {},
        "patches": [str(item) for item in (patches or [])],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def origin_cmake_args() -> List[str]:
    """Ask CMake to embed ``$ORIGIN`` so a copied binary finds sibling libraries."""
    return [_ORIGIN_CMAKE]


def ccache_supports_nvcc() -> bool:
    """True when this ccache is new enough to cache nvcc without a wrong cubin."""
    binary = shutil.which("ccache")
    if not binary:
        return False
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    text = f"{proc.stdout}\n{proc.stderr}"
    version = ""
    for token in text.replace(",", " ").split():
        parts = token.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            version = token
            break
    if not version:
        return False
    numbers = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    if len(numbers) < 2:
        return False
    return tuple(numbers[:2]) >= (4, 2)


def ccache_environment(
    base_dir: str,
    *,
    launchers: bool = True,
    cuda: bool = False,
) -> Dict[str, str]:
    """Environment that makes ccache hit across checkouts of the same tree.

    ``base_dir`` must be the parent that contains both the source and the build
    directory. CUDA is launched through ccache only when the installed ccache
    supports nvcc. ``CCACHE_COMPILERCHECK=content`` drops the cache when the
    compiler binary itself changes.
    """
    binary = shutil.which("ccache")
    if not binary or not base_dir:
        return {}
    cache = ccache_dir()
    os.makedirs(cache, exist_ok=True)
    env = {
        "CCACHE_DIR": cache,
        "CCACHE_BASEDIR": os.path.abspath(base_dir),
        "CCACHE_SLOPPINESS": "file_macro,time_macros",
        "CCACHE_COMPILERCHECK": "content",
        "CCACHE_MAXSIZE": os.getenv("STUDIO_CCACHE_MAXSIZE", "50G"),
        "CCACHE_NOHASHDIR": "",
    }
    if launchers:
        env["CMAKE_C_COMPILER_LAUNCHER"] = binary
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = binary
    if cuda and ccache_supports_nvcc():
        env["CMAKE_CUDA_COMPILER_LAUNCHER"] = binary
    return env


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _pid_alive(pid: int, expected_cmdline: str) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = _proc_cmdline(pid)
    if not cmdline:
        return False
    state = ""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            stat = handle.read()
        # pid (comm) state — comm may contain spaces and parentheses.
        state = stat.rsplit(")", 1)[-1].split(None, 1)[0]
    except OSError:
        return False
    if state in {"Z", "X"}:
        return False
    if expected_cmdline and cmdline != expected_cmdline:
        return False
    return True


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.realpath(path), os.path.realpath(root)]
        ) == os.path.realpath(root)
    except ValueError:
        return False


def _elf_kind(path: str) -> str:
    """Return ``exec``, ``dyn``, or empty when ``path`` is not a linked ELF."""
    import struct

    try:
        with open(path, "rb") as handle:
            data = handle.read(20)
    except OSError:
        return ""
    if len(data) < 18 or data[:4] != b"\x7fELF":
        return ""
    endian = "<" if data[5] == 1 else ">"
    kind = struct.unpack_from(endian + "H", data, 16)[0]
    return {2: "exec", 3: "dyn"}.get(kind, "")


def _is_linked_elf(path: str) -> bool:
    return _elf_kind(path) in {"exec", "dyn"}


def _is_elf(path: str) -> bool:
    return bool(_elf_kind(path)) or _raw_elf_magic(path)


def _raw_elf_magic(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def _read_rpath(binary: str) -> str:
    for argv in (["readelf", "-d", binary], ["objdump", "-p", binary]):
        if not shutil.which(argv[0]):
            continue
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        values: List[str] = []
        for line in proc.stdout.splitlines():
            if "RPATH" not in line and "RUNPATH" not in line:
                continue
            if "[" not in line or "]" not in line:
                continue
            values.append(line.split("[", 1)[1].rsplit("]", 1)[0].strip())
        if values:
            return ":".join(values)
    return ""


def _ensure_origin_rpath(binary: str) -> None:
    current = _read_rpath(binary)
    parts = [part for part in current.split(":") if part]
    if "$ORIGIN" in parts:
        return
    parts.insert(0, "$ORIGIN")
    _set_rpath(binary, ":".join(parts))


def _set_rpath(binary: str, value: str) -> None:
    tool = shutil.which("patchelf")
    if not tool:
        raise WorkspaceIsolationError(
            f"{binary} still points at the build workspace and patchelf is not installed, "
            "so the snapshot was not published. The previous engine version was left unchanged."
        )
    proc = subprocess.run(
        [tool, "--set-rpath", value, binary],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise WorkspaceIsolationError(
            f"Could not rewrite the library path of {binary}: {detail}"
        )


def _resolve_rpath_entry(binary: str, entry: str) -> str:
    origin = os.path.dirname(os.path.abspath(binary))
    expanded = entry.replace("${ORIGIN}", origin).replace("$ORIGIN", origin)
    if not os.path.isabs(expanded):
        expanded = os.path.join(origin, expanded)
    return os.path.realpath(expanded)


def repair_workspace_rpath(binary: str, forbidden_root: str) -> None:
    """Drop workspace entries from an ELF rpath and keep ``$ORIGIN`` plus the rest."""
    if not _is_linked_elf(binary):
        return
    current = _read_rpath(binary)
    parts = [part for part in current.split(":") if part]
    kept: List[str] = []
    removed = False
    for part in parts:
        if _is_within(_resolve_rpath_entry(binary, part), forbidden_root):
            removed = True
            continue
        if part not in kept:
            kept.append(part)
    if not removed:
        return
    if not any(part == "$ORIGIN" or part.startswith("$ORIGIN/") for part in kept):
        kept.insert(0, "$ORIGIN")
    _set_rpath(binary, ":".join(dict.fromkeys(kept)))


def assert_rpath_stays_outside(binary: str, forbidden_root: str) -> None:
    """Fail when a recorded rpath still resolves inside the workspace."""
    if not _is_linked_elf(binary):
        return
    for part in [item for item in _read_rpath(binary).split(":") if item]:
        if _is_within(_resolve_rpath_entry(binary, part), forbidden_root):
            raise WorkspaceIsolationError(
                f"{binary} still records a library path inside the build workspace ({part}). "
                "The snapshot was not published and the previous version was left unchanged."
            )


def _snapshot_library_dirs(snapshot_root: str) -> List[str]:
    dirs: List[str] = []
    if not snapshot_root or not os.path.isdir(snapshot_root):
        return dirs
    for dirpath, _, filenames in os.walk(snapshot_root):
        if any(name.endswith(".so") or ".so." in name for name in filenames):
            dirs.append(dirpath)
    return dirs


def assert_dynamic_libs_resolve(
    binary: str,
    forbidden_root: str,
    snapshot_root: str = "",
) -> None:
    """Fail when an ELF loads a library from the workspace.

    Libraries that exist only outside the snapshot (libcuda, libc, and the rest
    of the system) may be unresolved here. A library that was copied into the
    snapshot must resolve to that copy, never to the workspace that still holds
    the same file.
    """
    if not _is_linked_elf(binary) or not shutil.which("ldd"):
        return
    assert_rpath_stays_outside(binary, forbidden_root)
    env = os.environ.copy()
    lib_dirs = _snapshot_library_dirs(snapshot_root) if snapshot_root else []
    binary_dir = os.path.dirname(os.path.abspath(binary))
    if binary_dir not in lib_dirs:
        lib_dirs.insert(0, binary_dir)
    # Do not inherit the process library path. It can point at the workspace
    # and hide an isolation failure, or hide a library the snapshot actually ships.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)
    try:
        proc = subprocess.run(
            ["ldd", binary],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceIsolationError(f"Could not inspect {binary}: {exc}") from exc
    text = f"{proc.stdout}\n{proc.stderr}"
    if "not a dynamic executable" in text:
        return
    if proc.returncode != 0 and "not found" not in text:
        raise WorkspaceIsolationError(
            f"Could not inspect libraries for {binary}: {text.strip()}"
        )
    for line in proc.stdout.splitlines():
        if "=>" not in line:
            continue
        lib_name, rest = line.split("=>", 1)
        lib_name = os.path.basename(lib_name.strip())
        if "not found" in rest:
            if snapshot_root and _library_exists_under(snapshot_root, lib_name):
                raise WorkspaceIsolationError(
                    f"{os.path.basename(binary)} needs {lib_name}, which is in the snapshot "
                    "but did not resolve there. The snapshot was not published and the "
                    "previous version was left unchanged."
                )
            continue
        resolved = rest.split("(", 1)[0].strip()
        if resolved and _is_within(resolved, forbidden_root):
            raise WorkspaceIsolationError(
                f"{binary} loads {resolved} from the build workspace. "
                "The snapshot was not published and the previous version was left unchanged."
            )


def assert_symlinks_stay_inside(root: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                continue
            target = os.path.realpath(path)
            if not _is_within(target, root):
                raise WorkspaceIsolationError(
                    f"Snapshot symlink {path} points outside the snapshot at {target}"
                )


def site_packages_dir(venv_path: str) -> str:
    lib = os.path.join(venv_path, "lib")
    if not os.path.isdir(lib):
        return ""
    for name in sorted(os.listdir(lib)):
        if not name.startswith("python"):
            continue
        candidate = os.path.join(lib, name, "site-packages")
        if os.path.isdir(candidate):
            return candidate
    return ""


def retarget_text_tree(root: str, old: str, new: str) -> int:
    """Replace an absolute checkout prefix in text files. Returns files changed."""
    if not root or not os.path.isdir(root) or not old or old == new:
        return 0
    old_real = os.path.abspath(old)
    new_real = os.path.abspath(new)
    changed = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, "rb") as handle:
                    blob = handle.read()
            except OSError:
                continue
            if b"\x00" in blob[:8192]:
                continue
            text = blob.decode("utf-8", "surrogateescape")
            if old_real not in text and old not in text:
                continue
            updated = text.replace(old_real, new_real).replace(old, new_real)
            if updated == text:
                continue
            tmp = path + ".retarget"
            with open(tmp, "wb") as handle:
                handle.write(updated.encode("utf-8", "surrogateescape"))
            os.replace(tmp, path)
            changed += 1
    return changed


def assert_no_forbidden_text(root: str, forbidden: str) -> None:
    if not root or not os.path.isdir(root) or not forbidden:
        return
    needle = os.path.abspath(forbidden)
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, "rb") as handle:
                    blob = handle.read()
            except OSError:
                continue
            if b"\x00" in blob[:8192]:
                continue
            text = blob.decode("utf-8", "surrogateescape")
            if needle in text or forbidden in text:
                hits.append(path)
                if len(hits) >= 8:
                    break
        if len(hits) >= 8:
            break
    if hits:
        raise WorkspaceIsolationError(
            "Installed files still reference the build workspace: " + ", ".join(hits)
        )


def assert_outside_install_roots(path: str) -> None:
    from backend.engines.lifecycle import discover_engine_install_roots

    real = os.path.realpath(path)
    for root in discover_engine_install_roots().values():
        if root and _is_within(real, root):
            raise WorkspaceError(
                f"Refusing to place a build workspace inside an installed engine directory ({root})"
            )


class BuildWorkspace:
    """One persistent checkout and build directory for a single configuration."""

    def __init__(self, engine: str, key: str, *, root: Optional[str] = None) -> None:
        self.engine = str(engine or "").strip()
        self.key = str(key or "").strip()
        if not self.engine or not self.key or "/" in self.engine or "/" in self.key:
            raise WorkspaceError("Invalid workspace identity")
        base = os.path.abspath(root or workspace_root())
        self.path = os.path.join(base, self.engine, self.key)
        self.checkout_dir = os.path.join(self.path, "source")
        self.build_dir = os.path.join(self.path, "build")
        self._lock_path = os.path.join(self.path, "lock.json")
        self._state_path = os.path.join(self.path, "state.json")
        self._token = f"{os.getpid()}-{time.time_ns()}"
        self._held = False

    @classmethod
    def open(
        cls,
        engine: str,
        repo_url: str,
        config: Optional[dict] = None,
        patches: Optional[Sequence[str]] = None,
        *,
        root: Optional[str] = None,
    ) -> "BuildWorkspace":
        workspace = cls(engine, workspace_key(engine, repo_url, config, patches), root=root)
        if root is None:
            assert_outside_install_roots(workspace.path)
        return workspace

    def acquire(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        cmdline = _proc_cmdline(os.getpid())
        payload = {
            "pid": os.getpid(),
            "token": self._token,
            "cmdline": cmdline,
            "created_at": _utc(),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = None
        for _attempt in range(8):
            try:
                fd = os.open(self._lock_path, flags, 0o644)
                break
            except FileExistsError:
                self._steal_or_reject()
        if fd is None:
            raise WorkspaceBusy(
                f"Build workspace {self.engine}/{self.key} stayed locked after recovering a stale lock. "
                "The installed engine versions were not modified."
            )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        self._held = True
        previous = self.read_state()
        self.note(
            "locked",
            recovered_from=previous.get("phase") if previous.get("phase") not in {"ready", "idle", ""} else "",
        )

    def _steal_or_reject(self) -> None:
        try:
            with open(self._lock_path, "r", encoding="utf-8") as handle:
                lock = json.load(handle)
        except (OSError, json.JSONDecodeError):
            lock = {}
        pid = int(lock.get("pid") or 0)
        if _pid_alive(pid, str(lock.get("cmdline") or "")):
            raise WorkspaceBusy(
                f"Build workspace {self.engine}/{self.key} is in use by process {pid}. "
                "The installed engine versions were not modified. Retry after that build finishes."
            )
        logger.warning(
            "Recovering stale workspace lock %s (pid %s is gone)",
            self._lock_path,
            pid or "unknown",
        )
        try:
            if os.path.isdir(self._lock_path) and not os.path.islink(self._lock_path):
                shutil.rmtree(self._lock_path)
            else:
                os.remove(self._lock_path)
        except OSError as exc:
            raise WorkspaceBusy(
                f"Could not recover the workspace lock at {self._lock_path}: {exc}"
            ) from exc

    def release(self) -> None:
        if not self._held:
            return
        try:
            with open(self._lock_path, "r", encoding="utf-8") as handle:
                lock = json.load(handle)
        except (OSError, json.JSONDecodeError):
            lock = {}
        if lock.get("token") == self._token:
            try:
                os.remove(self._lock_path)
            except OSError as exc:
                logger.warning("Could not remove workspace lock %s: %s", self._lock_path, exc)
        self._held = False

    def read_state(self) -> dict:
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def note(self, phase: str, **extra: object) -> None:
        state = self.read_state()
        state.update(extra)
        state["phase"] = phase
        state["engine"] = self.engine
        state["key"] = self.key
        state["updated_at"] = _utc()
        _write_json(self._state_path, state)

    def sync_git(self, repo_url: str, ref: str) -> str:
        """Clone or fast-forward ``source`` to ``ref`` without deleting ``build/``."""
        repo_url = str(repo_url or "").strip()
        ref = _validate_ref(ref)
        if not repo_url:
            raise WorkspaceGitError("A repository URL is required")
        self.note("syncing", repo_url=repo_url, ref=ref)
        os.makedirs(self.path, exist_ok=True)
        git_dir = os.path.join(self.checkout_dir, ".git")
        if os.path.isdir(self.checkout_dir) and not os.path.isdir(git_dir) and not os.path.isfile(git_dir):
            abandoned = self.checkout_dir + ".abandoned-" + str(int(time.time()))
            os.rename(self.checkout_dir, abandoned)
            logger.warning("Moved an incomplete workspace checkout aside to %s", abandoned)
        if not os.path.isdir(git_dir):
            _git(["clone", repo_url, self.checkout_dir])
        origin = _git_output(["remote", "get-url", "origin"], cwd=self.checkout_dir).strip()
        if origin and origin != repo_url:
            raise WorkspaceGitError(
                f"Workspace origin is {origin}, not {repo_url}. "
                "Refusing to overwrite it. The installed versions were not modified."
            )
        fetch = _git(["fetch", "--prune", "origin", ref], cwd=self.checkout_dir, check=False)
        if fetch.returncode != 0 and ref == "master":
            fetch = _git(["fetch", "--prune", "origin", "main"], cwd=self.checkout_dir, check=False)
        if fetch.returncode != 0:
            fetch_all = _git(["fetch", "--prune", "--tags", "origin"], cwd=self.checkout_dir, check=False)
            if fetch_all.returncode != 0:
                detail = (fetch.stderr or fetch.stdout or fetch_all.stderr or "").strip()
                raise WorkspaceGitError(f"git fetch failed: {detail or 'unknown error'}")
            checkout = _git(["checkout", "--detach", ref], cwd=self.checkout_dir, check=False)
            if checkout.returncode != 0:
                alt = _git(
                    ["checkout", "--detach", f"origin/{ref}"],
                    cwd=self.checkout_dir,
                    check=False,
                )
                if alt.returncode != 0:
                    detail = (checkout.stderr or alt.stderr or "").strip()
                    raise WorkspaceGitError(f"git checkout failed: {detail or 'unknown error'}")
        else:
            _git(["checkout", "--detach", "FETCH_HEAD"], cwd=self.checkout_dir)
        _git(["reset", "--hard", "HEAD"], cwd=self.checkout_dir)
        clean_args = ["clean", "-fd", *_clean_excludes()]
        _git(clean_args, cwd=self.checkout_dir)
        if os.path.isfile(os.path.join(self.checkout_dir, ".gitmodules")):
            _git(["submodule", "update", "--init", "--recursive"], cwd=self.checkout_dir)
        head = _git_output(["rev-parse", "HEAD"], cwd=self.checkout_dir).strip()
        self.note("synced", head=head, ref=ref, repo_url=repo_url)
        return head

    def publish_tree(
        self,
        source: str,
        destination: str,
        required_relative: Sequence[str] = (),
    ) -> str:
        """Copy ``source`` to ``destination`` atomically. Return the destination path."""
        return self.publish_layout(
            {".": source},
            destination,
            required_relative=required_relative,
        )

    def publish_layout(
        self,
        parts: Dict[str, str],
        destination: str,
        required_relative: Sequence[str] = (),
    ) -> str:
        """Publish several directories as children of ``destination``.

        A key of ``.`` publishes that directory *as* ``destination``.
        Hidden staging is used so a crash cannot be mistaken for an install.
        """
        destination = os.path.abspath(destination)
        if os.path.isdir(destination) and _snapshot_complete(destination):
            raise WorkspaceError(
                f"{destination} is already a published engine snapshot and will not be replaced"
            )
        self.note("publishing", destination=destination)
        parent = os.path.dirname(destination)
        os.makedirs(parent, exist_ok=True)
        staging = os.path.join(parent, "." + os.path.basename(destination) + ".incomplete")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        try:
            if set(parts) == {"."}:
                src = os.path.abspath(parts["."])
                if not os.path.isdir(src):
                    raise WorkspaceError(f"Nothing to publish at {src}")
                _copy_tree(src, staging)
            else:
                os.makedirs(staging, exist_ok=False)
                for name, src in parts.items():
                    if name in {"", "."}:
                        raise WorkspaceError("A layout entry cannot replace the snapshot root")
                    src = os.path.abspath(src)
                    if not os.path.isdir(src):
                        raise WorkspaceError(f"Nothing to publish at {src}")
                    _copy_tree(src, os.path.join(staging, name))
            for relative in required_relative:
                if not os.path.exists(os.path.join(staging, relative)):
                    raise WorkspaceError(
                        f"Published snapshot is missing {relative}; the previous version was left unchanged"
                    )
            assert_symlinks_stay_inside(staging)
            _repair_published_binaries(staging, self.path)
            _write_sentinel(
                staging,
                {"engine": self.engine, "key": self.key, "published_at": _utc()},
            )
            if os.path.exists(destination):
                abandoned = os.path.join(
                    parent,
                    "." + os.path.basename(destination) + ".abandoned-" + str(int(time.time())),
                )
                os.rename(destination, abandoned)
                logger.warning("Moved incomplete install aside to %s", abandoned)
            os.rename(staging, destination)
            _fsync_dir(parent)
        except Exception:
            if os.path.exists(staging):
                shutil.rmtree(staging, ignore_errors=True)
            self.note("failed", error="publish failed")
            raise
        self.note("ready", destination=destination)
        return destination

    def seal_installed_source(self, snapshot_dir: str, venv_path: str) -> str:
        """Copy the checkout into the version and retarget the virtualenv at that copy.

        The virtualenv must not keep an editable install of the workspace. When the
        copy or the retarget cannot be proven, the snapshot is removed and the
        workspace is left as it was.
        """
        published = self.publish_tree(
            self.checkout_dir,
            snapshot_dir,
            required_relative=[".git"] if os.path.exists(os.path.join(self.checkout_dir, ".git")) else [],
        )
        site = site_packages_dir(venv_path)
        try:
            if site:
                retarget_text_tree(site, self.checkout_dir, published)
                retarget_text_tree(site, self.path, os.path.dirname(published))
                assert_no_forbidden_text(site, self.path)
                for dirpath, _, filenames in os.walk(site):
                    for name in filenames:
                        candidate = os.path.join(dirpath, name)
                        if _is_linked_elf(candidate):
                            repair_workspace_rpath(candidate, self.path)
                            assert_dynamic_libs_resolve(candidate, self.path, site)
        except Exception:
            self.note("failed", error="installed files still reference the workspace")
            hidden = os.path.join(
                os.path.dirname(published),
                "." + os.path.basename(published) + ".abandoned-" + str(int(time.time())),
            )
            try:
                os.rename(published, hidden)
            except OSError:
                shutil.rmtree(published, ignore_errors=True)
            raise
        return published


def release_held(workspace: Optional["BuildWorkspace"]) -> None:
    """Record an unfinished build, then drop the lock even if recording fails."""
    if workspace is None:
        return
    try:
        if workspace._held and workspace.read_state().get("phase") != "ready":
            workspace.note("failed")
    except Exception:
        logger.debug("Could not record workspace failure", exc_info=True)
    try:
        workspace.release()
    except Exception:
        logger.warning("Could not release workspace lock %s", workspace._lock_path, exc_info=True)


def _snapshot_complete(path: str) -> bool:
    return os.path.isfile(os.path.join(path, SNAPSHOT_SENTINEL))


def _write_sentinel(root: str, payload: dict) -> None:
    path = os.path.join(root, SNAPSHOT_SENTINEL)
    _write_json_atomic(path, payload)


def _write_json(path: str, payload: dict) -> None:
    _write_json_atomic(path, payload)


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_tree(src: str, dst: str) -> None:
    if os.path.exists(dst):
        raise WorkspaceError(f"Refusing to copy over existing path {dst}")
    cp = shutil.which("cp")
    if cp:
        proc = subprocess.run(
            [cp, "-a", src, dst],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        logger.warning("cp -a failed (%s); copying in Python", (proc.stderr or "").strip())
    shutil.copytree(src, dst, symlinks=True, copy_function=shutil.copy2)


def _repair_published_binaries(root: str, forbidden_root: str) -> None:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not _is_linked_elf(path):
                continue
            repair_workspace_rpath(path, forbidden_root)
            try:
                assert_dynamic_libs_resolve(path, forbidden_root, root)
            except WorkspaceIsolationError:
                _ensure_origin_rpath(path)
                assert_dynamic_libs_resolve(path, forbidden_root, root)


def _clean_excludes() -> List[str]:
    args: List[str] = []
    for pattern in _PRESERVE_ON_CLEAN:
        args.extend(["-e", pattern])
    return args


def _validate_ref(ref: str) -> str:
    text = str(ref or "").strip()
    if not text or "\x00" in text or text.startswith("-") or ".." in text.split("/"):
        raise WorkspaceGitError(f"Refusing to check out {ref!r}")
    if any(ch.isspace() for ch in text):
        raise WorkspaceGitError(f"Refusing to check out {ref!r}")
    return text


def _git(args: Sequence[str], *, cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        git_argv(*args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise WorkspaceGitError(detail or f"git {' '.join(args)} failed")
    return proc


def _git_output(args: Sequence[str], *, cwd: Optional[str] = None) -> str:
    proc = _git(args, cwd=cwd, check=True)
    return proc.stdout or ""


def _library_exists_under(root: str, basename: str) -> bool:
    if not root or not os.path.isdir(root) or not basename:
        return False
    for dirpath, _, filenames in os.walk(root):
        if basename in filenames:
            return True
    return False


def iter_elf_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if _is_elf(path):
                yield path
