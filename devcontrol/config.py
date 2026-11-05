"""Configuration and safety validation for the development stack.

The control plane is intentionally configured independently from the live
instance. Paths, the Compose project, image tags, and ports are validated here
before any command is run.
"""

from __future__ import annotations

import getpass
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .errors import ConfigError
from .locks import atomic_write_json, read_json

STACK_DIR = Path(__file__).resolve().parent.parent / "scripts" / "agent-dev"
DEFAULT_STATE_HOME = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / (
    "llama-cpp-studio-dev"
)
DEFAULT_REPO = Path.home() / "llama-cpp-studio-dev"
DEFAULT_DATA = Path.home() / "llama-cpp-studio-dev-data"
DEFAULT_WORKTREES = Path.home() / "llama-cpp-studio-worktrees"
DEFAULT_SHARED_MODELS = Path.home() / "llama-cpp-studio-shared-models"
DEFAULT_IMAGE = "llama-cpp-studio:agent-dev"
DEFAULT_BASE_IMAGE = "llama-cpp-studio:agent-dev-base"
DEFAULT_COMPOSE_PROJECT = "llama-cpp-studio-dev"
RESERVED_PORTS = {2000, 5173, 8080}
PROJECT_RE = re.compile(r"^llama-cpp-studio-dev(?:-[a-z0-9][a-z0-9-]*)?$")
WORKTREE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
GPU_RE = re.compile(r"^\d+(?:,\d+)*$")

CONFIG_FILENAME = "config.json"
ENV_FILENAME = "dev.env"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return default if value is None else value


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve(strict=False)


def _normalise_gpu(value: str) -> str:
    text = (value or "").strip().lower()
    if text in {"", "none", "off", "false", "0x0"}:
        return ""
    if not GPU_RE.match(text):
        raise ConfigError("DEV_GPU must be a comma-separated list of non-negative integers")
    parts = [item.strip() for item in text.split(",")]
    if any(not item for item in parts):
        raise ConfigError("DEV_GPU contains an empty device number")
    if len(set(parts)) != len(parts):
        raise ConfigError("DEV_GPU contains duplicate device numbers")
    return ",".join(parts)


def worktree_name_is_valid(name: str) -> bool:
    return bool(WORKTREE_RE.match(name or "")) and ".." not in name


@dataclass
class Config:
    state_home: Path = field(default_factory=lambda: _as_path(DEFAULT_STATE_HOME))
    repo: Path = field(default_factory=lambda: _as_path(DEFAULT_REPO))
    data: Path = field(default_factory=lambda: _as_path(DEFAULT_DATA))
    worktrees: Path = field(default_factory=lambda: _as_path(DEFAULT_WORKTREES))
    shared_models: Path = field(default_factory=lambda: _as_path(DEFAULT_SHARED_MODELS))
    image: str = DEFAULT_IMAGE
    base_image: str = DEFAULT_BASE_IMAGE
    backend_port: int = 18081
    frontend_port: int = 15173
    proxy_port: int = 12000
    gpu: str = ""
    compose_project: str = DEFAULT_COMPOSE_PROJECT
    runtime: str = "compose"
    active: str = "main"
    uid: str = "1000"
    gid: str = "1000"
    api_token: str = field(default="", repr=False)
    primeui_license: str = field(default="", repr=False)

    @property
    def initialized(self) -> bool:
        return (self.state_home / CONFIG_FILENAME).is_file()

    @property
    def config_path(self) -> Path:
        return self.state_home / CONFIG_FILENAME

    @property
    def env_path(self) -> Path:
        return self.state_home / ENV_FILENAME

    @property
    def journal_path(self) -> Path:
        return self.state_home / "operations.jsonl"

    @property
    def checkpoints_path(self) -> Path:
        return self.state_home / "checkpoints.json"

    @property
    def diagnostics_dir(self) -> Path:
        return self.state_home / "diagnostics"

    @property
    def locks_dir(self) -> Path:
        return self.state_home / "locks"

    @property
    def stack_dir(self) -> Path:
        return STACK_DIR

    @property
    def compose_file(self) -> Path:
        return self.stack_dir / "compose.yaml"

    @property
    def gpu_compose_file(self) -> Path:
        return self.stack_dir / "compose.gpu.yaml"

    @property
    def active_path(self) -> Path:
        if self.active == "main":
            return self.repo
        return self.worktrees / self.active

    @property
    def cuda_visible_devices(self) -> str:
        if not self.gpu:
            return ""
        # The NVIDIA runtime remaps selected devices; inside the container the
        # visible devices are always indexed from zero.
        return ",".join(str(index) for index in range(len(self.gpu.split(","))))

    @property
    def actor(self) -> str:
        explicit = (os.getenv("DEV_ACTOR") or "").strip()
        if explicit:
            return explicit
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        return f"{user}@{os.uname().nodename if hasattr(os, 'uname') else os.getenv('COMPUTERNAME', 'host')}"

    def compose_env(self) -> dict[str, str]:
        return {
            "DEV_REPO": str(self.repo),
            "DEV_CHECKOUT": str(self.active_path),
            "DEV_DATA": str(self.data),
            "DEV_SHARED_MODELS": str(self.shared_models),
            "DEV_BACKEND_PORT": str(self.backend_port),
            "DEV_FRONTEND_PORT": str(self.frontend_port),
            "DEV_PROXY_PORT": str(self.proxy_port),
            "DEV_API_TOKEN": self.api_token,
            "DEV_IMAGE": self.image,
            "DEV_BASE_IMAGE": self.base_image,
            "DEV_UID": self.uid,
            "DEV_GID": self.gid,
            "DEV_GPU": self.gpu,
            "DEV_CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
            "DEV_COMPOSE_PROJECT": self.compose_project,
            "VITE_PRIMEUI_LICENSE": self.primeui_license,
        }

    def env_file_text(self) -> str:
        lines = ["# Generated by devcontrol. Mode 0600; contains the dev management token.\n"]
        for key, value in self.compose_env().items():
            if "\n" in value or "\r" in value:
                raise ConfigError(f"{key} must not contain a newline")
            lines.append(f"{key}={value}\n")
        return "".join(lines)

    def save(self, *, include_env: bool = True) -> None:
        self.validate()
        payload = {
            "schema": 1,
            "state_home": str(self.state_home),
            "repo": str(self.repo),
            "data": str(self.data),
            "worktrees": str(self.worktrees),
            "shared_models": str(self.shared_models),
            "image": self.image,
            "base_image": self.base_image,
            "backend_port": self.backend_port,
            "frontend_port": self.frontend_port,
            "proxy_port": self.proxy_port,
            "gpu": self.gpu,
            "compose_project": self.compose_project,
            "runtime": self.runtime,
            "active": self.active,
            "uid": self.uid,
            "gid": self.gid,
        }
        atomic_write_json(self.config_path, payload)
        if include_env:
            from .locks import atomic_write_text

            atomic_write_text(self.env_path, self.env_file_text())

    def validate(self) -> None:
        self.state_home = _as_path(self.state_home)
        self.repo = _as_path(self.repo)
        self.data = _as_path(self.data)
        self.worktrees = _as_path(self.worktrees)
        self.shared_models = _as_path(self.shared_models)
        self.gpu = _normalise_gpu(self.gpu)
        self.image = (self.image or "").strip()
        self.base_image = (self.base_image or "").strip()
        if not self.image or not self.base_image:
            raise ConfigError("DEV_IMAGE and DEV_BASE_IMAGE must be non-empty")
        if self.image == self.base_image:
            raise ConfigError("DEV_IMAGE and DEV_BASE_IMAGE must differ")
        reserved_images = {
            "llama-cpp-studio",
            "llama-cpp-studio:latest",
            "llama-cpp-studio:dev",
        }
        for label, image in (("DEV_IMAGE", self.image), ("DEV_BASE_IMAGE", self.base_image)):
            if image in reserved_images:
                raise ConfigError(f"{label} must not reuse the live or local dev image: {image}")
        if self.runtime not in {"compose", "local"}:
            raise ConfigError("DEV_RUNTIME must be 'compose' or 'local'")
        if not PROJECT_RE.match(self.compose_project):
            raise ConfigError(
                "DEV_COMPOSE_PROJECT must be 'llama-cpp-studio-dev' or start with "
                "'llama-cpp-studio-dev-'"
            )
        if self.compose_project == "llama-cpp-studio":
            raise ConfigError("DEV_COMPOSE_PROJECT must not be the live project")
        ports = [self.backend_port, self.frontend_port, self.proxy_port]
        if any(not isinstance(port, int) or port < 1 or port > 65535 for port in ports):
            raise ConfigError("dev ports must be integers between 1 and 65535")
        if len(set(ports)) != len(ports):
            raise ConfigError("dev backend, frontend, and proxy ports must be distinct")
        reserved = sorted(set(ports) & RESERVED_PORTS)
        # In the self-contained image these are container-internal ports; only
        # SSH/Vite are published to the host, so they cannot collide with the
        # live instance.
        if reserved and self.runtime != "local":
            raise ConfigError(
                "dev ports must not overlap live/default Studio ports: "
                + ", ".join(str(port) for port in reserved)
            )
        if not worktree_name_is_valid(self.active):
            raise ConfigError(f"invalid active worktree name: {self.active!r}")
        if not self.uid or not self.gid:
            raise ConfigError("DEV_UID and DEV_GID must be set")
        self._assert_paths_distinct()

    def _assert_paths_distinct(self) -> None:
        named = {
            "repo": self.repo,
            "data": self.data,
            "worktrees": self.worktrees,
            "state_home": self.state_home,
            "shared_models": self.shared_models,
        }
        for label, path in named.items():
            if path == Path(path.anchor):
                raise ConfigError(f"{label} must not be the filesystem root")
            if str(path) == "/app":
                raise ConfigError(f"{label} must not be the production container path /app")
        # The live instance normally uses <repo>/data. Sharing that directory
        # would leak configuration and operation history, so keep every dev
        # root outside the checkout.
        for label, path in named.items():
            if label == "repo":
                continue
            if path == self.repo or path.is_relative_to(self.repo):
                raise ConfigError(f"{label} must be outside the development checkout")
            if self.repo.is_relative_to(path):
                raise ConfigError(f"{label} must not contain the development checkout")
        roots = {
            "data": self.data,
            "worktrees": self.worktrees,
            "state_home": self.state_home,
            "shared_models": self.shared_models,
        }
        items = list(roots.items())
        for index, (left_label, left) in enumerate(items):
            for right_label, right in items[index + 1 :]:
                if left == right:
                    raise ConfigError(f"{left_label} and {right_label} must be distinct")
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise ConfigError(
                        f"{left_label} and {right_label} must not overlap: {left} / {right}"
                    )

    @classmethod
    def defaults(cls, *, state_home: Optional[str | Path] = None) -> "Config":
        home = _as_path(state_home) if state_home else _as_path(DEFAULT_STATE_HOME)
        return cls(state_home=home)

    @classmethod
    def load(
        cls,
        *,
        state_home: Optional[str | Path] = None,
        overrides: Optional[dict[str, Any]] = None,
        require_initialized: bool = True,
    ) -> "Config":
        state = _as_path(state_home or _env("DEV_CONTROL_HOME") or DEFAULT_STATE_HOME)
        raw = read_json(state / CONFIG_FILENAME, {})
        if not isinstance(raw, dict):
            raw = {}
        if require_initialized and not raw:
            raise ConfigError(
                f"development stack is not initialized in {state}; run 'devcontrol bootstrap' first"
            )
        config = cls(state_home=state)
        path_keys = {
            "repo": "repo",
            "data": "data",
            "worktrees": "worktrees",
            "shared_models": "shared_models",
        }
        for key, raw_key in path_keys.items():
            if raw.get(raw_key):
                setattr(config, key, _as_path(raw[raw_key]))
        for key in (
            "image",
            "base_image",
            "backend_port",
            "frontend_port",
            "proxy_port",
            "gpu",
            "compose_project",
            "runtime",
            "active",
            "uid",
            "gid",
        ):
            if raw.get(key) is not None:
                setattr(config, key, raw[key])
        env_overrides = {
            "DEV_REPO": "repo",
            "DEV_DATA": "data",
            "DEV_WORKTREES": "worktrees",
            "DEV_SHARED_MODELS": "shared_models",
            "DEV_IMAGE": "image",
            "DEV_BASE_IMAGE": "base_image",
            "DEV_BACKEND_PORT": "backend_port",
            "DEV_FRONTEND_PORT": "frontend_port",
            "DEV_PROXY_PORT": "proxy_port",
            "DEV_GPU": "gpu",
            "DEV_COMPOSE_PROJECT": "compose_project",
            "DEV_RUNTIME": "runtime",
            "DEV_UID": "uid",
            "DEV_GID": "gid",
        }
        for env_name, key in env_overrides.items():
            value = os.getenv(env_name)
            if value is None or (value == "" and env_name != "DEV_GPU"):
                continue
            if key in {"backend_port", "frontend_port", "proxy_port"}:
                try:
                    value = int(value)
                except ValueError as exc:
                    raise ConfigError(f"{env_name} must be an integer") from exc
            elif key in {"repo", "data", "worktrees", "shared_models"}:
                value = _as_path(value)
            setattr(config, key, value)
        for key, value in (overrides or {}).items():
            if value is not None:
                setattr(config, key, value)
        config.api_token = _load_token(config)
        config.primeui_license = (os.getenv("VITE_PRIMEUI_LICENSE") or "").strip()
        config.validate()
        return config

    def assert_initialized(self) -> None:
        if not self.repo.is_dir() or not (self.repo / ".git").exists():
            raise ConfigError(
                f"development checkout is missing or not a git repository: {self.repo}"
            )


def _load_token(config: Config) -> str:
    explicit = (os.getenv("DEV_API_TOKEN") or "").strip()
    if explicit:
        return explicit
    try:
        with open(config.env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#") or "=" not in text:
                    continue
                key, value = text.split("=", 1)
                if key.strip() == "DEV_API_TOKEN":
                    return _unquote(value.strip())
    except OSError:
        return ""
    return ""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def new_token() -> str:
    return secrets.token_urlsafe(32)
