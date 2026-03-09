"""YAML-backed data store replacing SQLite."""

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

import yaml

from backend.logging_config import get_logger

logger = get_logger(__name__)


def _get_config_dir() -> str:
    """Return config directory (Docker: /app/data/config, local: data/config)."""
    if os.path.exists("/app/data"):
        return "/app/data/config"
    return os.path.abspath("data/config")


def generate_proxy_name(huggingface_id: str, quantization: Optional[str] = None) -> str:
    """
    Generate a proxy name for llama-swap using HuggingFace ID and optional quantization.
    """
    huggingface_slug = (
        huggingface_id.replace("/", "-").replace(" ", "-").replace(".", "-").lower()
    )
    if quantization:
        quantization_slug = quantization.replace(" ", "-").lower()
        return f"{huggingface_slug}.{quantization_slug}"
    return huggingface_slug


def _coerce_config(config_value: Optional[Any]) -> Dict[str, Any]:
    if not config_value:
        return {}
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            return {}
    return {}


def _model_value(model: Any, key: str, default: Any = None) -> Any:
    if isinstance(model, dict):
        return model.get(key, default)
    return getattr(model, key, default)


def normalize_proxy_alias(alias: Optional[str]) -> str:
    """Normalize a user-provided model alias into a safe exposed engine ID."""
    if alias is None:
        return ""

    normalized = str(alias).strip().lower()
    if not normalized:
        return ""

    normalized = normalized.replace("/", "-").replace("\\", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9._-]", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = normalized.strip("._-")
    return normalized


def resolve_proxy_name(model: Any) -> str:
    """Return the exposed runtime model ID for a stored model."""
    config = _coerce_config(_model_value(model, "config"))
    alias = normalize_proxy_alias(config.get("model_alias"))
    if alias:
        return alias

    existing = normalize_proxy_alias(_model_value(model, "proxy_name"))
    if existing:
        return existing

    return generate_proxy_name(
        _model_value(model, "huggingface_id", ""),
        _model_value(model, "quantization"),
    )


class DataStore:
    """Thread-safe YAML-backed data store replacing SQLite."""

    def __init__(self, config_dir: Optional[str] = None):
        self._config_dir = os.path.abspath(config_dir or _get_config_dir())
        self._lock = threading.Lock()
        self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        """Create config dir and default YAML files if they don't exist."""
        os.makedirs(self._config_dir, exist_ok=True)
        for filename, default in [
            ("models.yaml", {"models": []}),
            (
                "engines.yaml",
                {
                    "llama_cpp": {"active_version": None, "versions": []},
                    "ik_llama": {"active_version": None, "versions": []},
                    "lmdeploy": {
                        "installed": False,
                        "version": None,
                        "install_type": None,
                        "source_repo": None,
                        "source_branch": None,
                        "venv_path": None,
                    },
                    "cuda": {"installed_version": None, "install_path": None},
                },
            ),
            ("settings.yaml", {"huggingface_token": "", "proxy_port": 2000}),
        ]:
            path = os.path.join(self._config_dir, filename)
            if not os.path.exists(path):
                self._write_yaml(path, default)

    def _read_yaml(self, filename: str) -> dict:
        """Read and parse a YAML file. Returns empty dict on error."""
        path = os.path.join(self._config_dir, filename)
        with self._lock:
            if not os.path.exists(path):
                return {}
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
                return {}

    def _write_yaml(self, path: str, data: dict) -> None:
        """Atomic write: write to temp file then rename."""
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise e

    def _save_yaml(self, filename: str, data: dict) -> None:
        """Thread-safe write to a YAML file."""
        path = os.path.join(self._config_dir, filename)
        with self._lock:
            self._write_yaml(path, data)

    # --- Models ---

    def list_models(self) -> List[dict]:
        return self._read_yaml("models.yaml").get("models", [])

    def get_model(self, model_id: str) -> Optional[dict]:
        for m in self.list_models():
            if m.get("id") == model_id:
                return m
        return None

    def add_model(self, model: dict) -> dict:
        data = self._read_yaml("models.yaml")
        data.setdefault("models", []).append(model)
        self._save_yaml("models.yaml", data)
        return model

    def update_model(self, model_id: str, updates: dict) -> Optional[dict]:
        data = self._read_yaml("models.yaml")
        for m in data.get("models", []):
            if m.get("id") == model_id:
                m.update(updates)
                self._save_yaml("models.yaml", data)
                return m
        return None

    def delete_model(self, model_id: str) -> bool:
        data = self._read_yaml("models.yaml")
        models = data.get("models", [])
        new_models = [m for m in models if m.get("id") != model_id]
        if len(new_models) == len(models):
            return False
        data["models"] = new_models
        self._save_yaml("models.yaml", data)
        return True

    # --- Engines (llama_cpp, ik_llama) ---

    def get_engine_versions(self, engine: str) -> List[dict]:
        """engine is 'llama_cpp' or 'ik_llama'."""
        return self._read_yaml("engines.yaml").get(engine, {}).get("versions", [])

    def get_active_engine_version(self, engine: str) -> Optional[dict]:
        data = self._read_yaml("engines.yaml").get(engine, {})
        active = data.get("active_version")
        if not active:
            return None
        for v in data.get("versions", []):
            if v.get("version") == active:
                return v
        return None

    def add_engine_version(self, engine: str, version_data: dict) -> None:
        data = self._read_yaml("engines.yaml")
        data.setdefault(engine, {}).setdefault("versions", []).append(version_data)
        self._save_yaml("engines.yaml", data)

    def set_active_engine_version(self, engine: str, version: str) -> None:
        data = self._read_yaml("engines.yaml")
        data.setdefault(engine, {})["active_version"] = version
        self._save_yaml("engines.yaml", data)

    def delete_engine_version(self, engine: str, version: str) -> bool:
        data = self._read_yaml("engines.yaml")
        engine_data = data.get(engine, {})
        versions = engine_data.get("versions", [])
        new_versions = [v for v in versions if v.get("version") != version]
        if len(new_versions) == len(versions):
            return False
        engine_data["versions"] = new_versions
        if engine_data.get("active_version") == version:
            engine_data["active_version"] = None
        self._save_yaml("engines.yaml", data)
        return True

    def get_engine_build_settings(self, engine: str) -> Dict[str, Any]:
        """Return persisted build settings for the given engine (or empty dict)."""
        data = self._read_yaml("engines.yaml")
        return data.get(engine, {}).get("build_settings", {}) or {}

    def update_engine_build_settings(self, engine: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge and persist build settings for the given engine. Returns the stored settings."""
        if not isinstance(settings, dict):
            settings = {}
        data = self._read_yaml("engines.yaml")
        engine_data = data.setdefault(engine, {})
        existing = engine_data.get("build_settings") or {}
        merged = {**existing, **settings}
        engine_data["build_settings"] = merged
        self._save_yaml("engines.yaml", data)
        return merged

    # --- LMDeploy ---

    def get_lmdeploy_status(self) -> dict:
        return self._read_yaml("engines.yaml").get("lmdeploy", {})

    def update_lmdeploy(self, updates: dict) -> None:
        data = self._read_yaml("engines.yaml")
        data.setdefault("lmdeploy", {}).update(updates)
        self._save_yaml("engines.yaml", data)

    # --- CUDA ---

    def get_cuda_status(self) -> dict:
        return self._read_yaml("engines.yaml").get("cuda", {})

    def update_cuda(self, updates: dict) -> None:
        data = self._read_yaml("engines.yaml")
        data.setdefault("cuda", {}).update(updates)
        self._save_yaml("engines.yaml", data)

    # --- Settings ---

    def get_settings(self) -> dict:
        return self._read_yaml("settings.yaml")

    def update_settings(self, updates: dict) -> None:
        data = self._read_yaml("settings.yaml")
        data.update(updates)
        self._save_yaml("settings.yaml", data)


_store: Optional[DataStore] = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store
