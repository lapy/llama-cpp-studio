"""Typed failures surfaced by the development control plane."""

from __future__ import annotations

from typing import Any, Optional


class DevControlError(Exception):
    """An expected failure with a stable machine-readable code."""

    code = "DEV_CONTROL_ERROR"
    exit_code = 1

    def __init__(self, message: str, *, detail: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.detail = dict(detail or {})


class ConfigError(DevControlError):
    code = "CONFIG_INVALID"
    exit_code = 1


class GuardError(DevControlError):
    """A guard refused an operation before it had an effect."""

    code = "GUARD_REFUSED"
    exit_code = 3


class BusyError(GuardError):
    code = "OPERATION_BUSY"
    exit_code = 3

    def __init__(self, message: str, *, holder: Optional[dict[str, Any]] = None):
        super().__init__(message, detail={"holder": holder} if holder else None)
        self.holder = holder


class CommandError(DevControlError):
    code = "COMMAND_FAILED"
    exit_code = 1

    def __init__(
        self,
        message: str,
        *,
        argv: Optional[list[str]] = None,
        returncode: Optional[int] = None,
        stdout_tail: str = "",
        stderr_tail: str = "",
        timed_out: bool = False,
    ):
        detail = {
            "argv": list(argv or []),
            "returncode": returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "timed_out": timed_out,
        }
        super().__init__(message, detail=detail)
        self.argv = list(argv or [])
        self.returncode = returncode
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.timed_out = timed_out


class NotFoundError(DevControlError):
    code = "NOT_FOUND"
    exit_code = 1
