import logging
import sys
from typing import Any, Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def describe_error(exc: BaseException) -> str:
    """One line covering the exception, its structured detail, and its cause."""
    parts = []
    bodies = []
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text, body = _exception_text(current)
        if body and not any(body in earlier or earlier in body for earlier in bodies):
            parts.append(text)
            bodies.append(body)
        current = current.__cause__ or current.__context__
    return " caused by ".join(parts) if parts else type(exc).__name__


def log_api_error(logger: logging.Logger, request: Any, exc: BaseException) -> None:
    """Log an API failure with method, path, status, and the underlying reason."""
    method = getattr(request, "method", "?")
    url = getattr(request, "url", None)
    path = getattr(url, "path", "?")
    status = int(getattr(exc, "status_code", None) or getattr(exc, "status", None) or 500)
    text = describe_error(exc)
    cause = exc.__cause__ or exc.__context__
    traced = cause if cause is not None and cause.__traceback__ is not None else exc
    if status >= 500:
        logger.error(
            "%s %s failed (%s): %s",
            method,
            path,
            status,
            text,
            exc_info=traced,
        )
        return
    if status in (401, 403):
        logger.info("%s %s denied (%s): %s", method, path, status, text)
        return
    logger.warning("%s %s rejected (%s): %s", method, path, status, text)


def log_failure(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Log a non-HTTP failure with its traceback and structured detail."""
    logger.error("%s: %s", message, describe_error(exc), exc_info=exc)


def _exception_text(exc: BaseException) -> tuple[str, str]:
    """Return the log line and the comparable message body."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            rows = errors()
        except Exception:
            rows = None
        if isinstance(rows, list) and rows:
            rendered = []
            for item in rows:
                if not isinstance(item, dict):
                    rendered.append(str(item))
                    continue
                loc = ".".join(str(part) for part in item.get("loc") or [])
                msg = str(item.get("msg") or item.get("type") or "invalid")
                rendered.append(f"{loc}: {msg}" if loc else msg)
            body = "; ".join(rendered)
            return f"{type(exc).__name__}: {body}", body
    detail = getattr(exc, "detail", None)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    prefix = type(exc).__name__
    if status:
        prefix = f"{prefix} {status}"
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("error") or detail)
        failures = detail.get("failures") or []
        extra = ""
        if isinstance(failures, (list, tuple)) and failures:
            rendered = "; ".join(str(item) for item in failures)
            if rendered and rendered not in message:
                extra = f" ({rendered})"
        field = detail.get("field")
        if field and f"field={field}" not in message:
            extra += f" field={field}"
        body = f"{message}{extra}"
        return f"{prefix}: {body}", body
    if isinstance(detail, str) and detail.strip():
        body = detail.strip()
        return f"{prefix}: {body}", body
    message = str(exc).strip()
    name = type(exc).__name__
    if message and message != name:
        line = f"{prefix}: {message}" if status else f"{name}: {message}"
        return line, message
    return prefix, prefix
