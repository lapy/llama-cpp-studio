"""Single source of truth for the Studio application version."""

from pathlib import Path


def _read_version() -> str:
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    value = version_file.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("VERSION is empty")
    return value


APP_VERSION = _read_version()
