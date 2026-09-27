#!/usr/bin/env python3
"""Replace this process with the engine described by one published manifest.

The launcher reads ``active.json`` once, resolves artifacts from that generation,
writes a receipt, drops the launch gate, and ``exec``s. It does not import the
Studio application, open a store, or use the network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--gate-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        port = int(args.port)
    except ValueError:
        _fail(2, "port must be an integer")
    if port < 1 or port > 65535:
        _fail(2, "port is outside 1-65535")
    root = os.path.realpath(args.manifest_root)
    if not os.path.isdir(root):
        _fail(3, "manifest root is not a directory")
    lock_fd = _acquire_shared(os.path.join(root, "launch.lock"), args.gate_timeout)
    try:
        pointer = _read_json(os.path.join(root, "active.json"))
        revision = _revision(pointer.get("revision"))
        generation = os.path.join(root, "generations", revision)
        _inside(root, generation)
        manifest_path = os.path.join(generation, "manifest.json")
        _inside(root, manifest_path)
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != 1:
            _fail(3, "unsupported manifest schema")
        if str(manifest.get("revision") or "") != revision:
            _fail(3, "manifest revision does not match the published pointer")
        executable = str(manifest.get("executable") or "")
        if not executable or not os.path.isfile(executable):
            _fail(3, "manifest executable is missing")
        argv = _substitute_argv(manifest.get("argv"), port=str(port), generation=generation, root=root)
        env = _environment(manifest.get("env"), port=str(port))
        cwd = manifest.get("cwd")
        if cwd:
            cwd = os.path.realpath(str(cwd))
            if not os.path.isdir(cwd):
                _fail(3, "manifest working directory is missing")
        launch_id = uuid.uuid4().hex
        _write_receipt(
            root,
            {
                "model_id": manifest.get("model_id"),
                "revision": revision,
                "launch_id": launch_id,
                "pid": os.getpid(),
                "start_ticks": _start_ticks(os.getpid()),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    finally:
        _release(lock_fd)
    if cwd:
        os.chdir(cwd)
    try:
        os.execve(executable, [executable, *argv], env)
    except OSError as exc:
        _fail(5, f"exec failed: {exc}")
    return 5


def _substitute_argv(raw, *, port: str, generation: str, root: str) -> list[str]:
    if not isinstance(raw, list):
        _fail(3, "manifest argv must be a list")
    argv: list[str] = []
    for item in raw:
        if isinstance(item, str):
            argv.append(item)
            continue
        if not isinstance(item, dict) or len(item) != 1:
            _fail(3, "manifest argv placeholder is invalid")
        if item.get("runtime") == "port":
            argv.append(port)
            continue
        artifact = item.get("artifact")
        if isinstance(artifact, str):
            path = os.path.join(generation, "artifacts", artifact)
            _inside(os.path.join(generation, "artifacts"), path)
            _inside(root, path)
            if not os.path.isfile(path):
                _fail(3, f"artifact {artifact} is missing")
            argv.append(os.path.realpath(path))
            continue
        _fail(3, "unsupported argv placeholder")
    return argv


def _environment(raw, *, port: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        _fail(3, "manifest env must be an object")
    values = raw.get("set") or {}
    unset = raw.get("unset") or []
    if not isinstance(values, dict) or not isinstance(unset, list):
        _fail(3, "manifest env set/unset is invalid")
    env: dict[str, str] = {}
    for key, value in values.items():
        name = str(key)
        if not name or "\x00" in name:
            _fail(3, "invalid environment name")
        if isinstance(value, dict) and value.get("runtime") == "port":
            env[name] = port
        elif isinstance(value, str):
            if "\x00" in value:
                _fail(3, "environment value contains NUL")
            env[name] = value
        else:
            _fail(3, f"environment value for {name} is not a literal")
    for key in unset:
        env.pop(str(key), None)
    return env


def _write_receipt(root: str, payload: dict) -> None:
    directory = os.path.join(root, "launches")
    os.makedirs(directory, mode=0o700, exist_ok=True)
    launch_id = str(payload["launch_id"])
    final = os.path.join(directory, f"{launch_id}.json")
    _inside(root, final)
    tmp = final + ".tmp"
    data = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    os.chmod(final, 0o600)


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(3, f"unreadable manifest file: {exc}")
    if not isinstance(payload, dict):
        _fail(3, "manifest file must be a JSON object")
    return payload


def _revision(value) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        _fail(3, "published revision is invalid")
    return text


def _inside(root: str, path: str) -> None:
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        _fail(3, "manifest path escapes its root")


def _start_ticks(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            stat = handle.read()
    except OSError:
        return ""
    end = stat.rfind(")")
    if end < 0:
        return ""
    fields = stat[end + 2 :].split()
    if len(fields) < 20:
        return ""
    return fields[19]


def _acquire_shared(path: str, timeout: float):
    import fcntl

    parent = os.path.dirname(path)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                _fail(4, "timed out waiting for the launch gate")
            time.sleep(0.02)


def _release(fd: int) -> None:
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _fail(code: int, message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3)
