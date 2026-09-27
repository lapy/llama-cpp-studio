"""Long-running operation primitives shared by installers."""

from backend.operations.cancellable import CancelResult, CancellableOperationManager
from backend.operations.progress import ProgressManager, get_progress_manager

__all__ = [
    "CancelResult",
    "CancellableOperationManager",
    "ProgressManager",
    "get_progress_manager",
]
