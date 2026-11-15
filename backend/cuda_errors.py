"""Typed failures for the Studio-managed CUDA installer API.

The Hardware panel drives install/activate/uninstall over HTTP and needs to
tell "not installed" apart from "incomplete", "busy", or "install failed"
without parsing English prose.  Every public failure therefore carries a
stable ``code`` and the ``http_status`` the route should answer with; the
message stays the same text users already saw.

Deliberately dependency-free so the installer and its tests can import it.
"""

from typing import Dict, Optional


class CudaInstallerError(Exception):
    """Base class for failures a CUDA API caller can act on."""

    code: str = "cuda_installer_error"
    http_status: int = 500
    default_message: str = "CUDA installer operation failed"

    def __init__(self, message: Optional[str] = None) -> None:
        self.message = str(message) if message else self.default_message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def to_detail(self) -> Dict[str, str]:
        """Stable error envelope for ``HTTPException.detail``."""
        return {"code": self.code, "message": self.message}


class CudaOperationBusy(CudaInstallerError):
    code = "cuda_operation_busy"
    http_status = 409
    default_message = "Another CUDA installer operation is already running"


class CudaVersionUnsupported(CudaInstallerError):
    code = "cuda_version_unsupported"
    http_status = 400
    default_message = "Unsupported or unavailable CUDA version"


class CudaVersionNotInstalled(CudaInstallerError):
    code = "cuda_version_not_installed"
    http_status = 404
    default_message = "CUDA installation not found"


class CudaInstallIncomplete(CudaInstallerError):
    code = "cuda_install_incomplete"
    http_status = 409
    default_message = "CUDA installation is missing or incomplete"


class CudaInstallFailed(CudaInstallerError):
    code = "cuda_install_failed"
    http_status = 500
    default_message = "CUDA installation failed"


class CudaUnsupportedPlatform(CudaInstallerError):
    code = "cuda_platform_unsupported"
    http_status = 400
    default_message = "CUDA installation is not supported on this platform"
