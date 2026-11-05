"""Installer contract used by version routes and the adapter map."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class EngineInstaller(Protocol):
    """Public surface shared by every versioned engine installer."""

    @property
    def engine_id(self) -> str: ...

    @property
    def install_root(self) -> str: ...

    @property
    def log_path(self) -> str: ...

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    async def install_from_source(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    async def retry_existing_install(self, version_entry: Dict[str, Any]) -> Dict[str, Any]: ...

    async def sync_source(self, version_entry: Dict[str, Any]) -> Dict[str, Any]: ...

    async def remove(self, retire_references: bool = False) -> Dict[str, Any]: ...

    def status(self) -> Dict[str, Any]: ...

    def read_log_tail(self, max_bytes: int = 8192) -> str: ...

    def cancel_task(self, task_id: str) -> Dict[str, Any]: ...
