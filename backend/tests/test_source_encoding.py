"""Guard against UTF-8 text that was decoded as Windows-1252 and re-encoded.

Mojibake is user-visible (placeholders, comments, docs) and easy to reintroduce
when a file is read with the wrong codec and written back. The markers below are
the exact sequences produced by that round trip; legitimate text never needs
them, so the whole repository can be checked cheaply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".txt",
    ".vue",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "playwright-report",
    "test-results",
}

# Windows-1252 renderings of common UTF-8 punctuation and accents.
MOJIBAKE_SEQUENCES = (
    "\u00e2\u20ac\u201d",  # em dash
    "\u00e2\u20ac\u201c",  # en dash
    "\u00e2\u20ac\u00a6",  # ellipsis
    "\u00e2\u20ac\u2122",  # right single quote
    "\u00e2\u20ac\u0153",  # left double quote
    "\u00e2\u20ac\u009d",  # right double quote
    "\u00c3\u00a9",  # é
    "\u00c3\u00a8",  # è
    "\u00c3\u00bc",  # ü
    "\u00c3\u00b6",  # ö
    "\u00c3\u00a4",  # ä
    "\u00c3\u00b1",  # ñ
    "\u00c3\u00a7",  # ç
    "\u00c2\u00a0",  # non-breaking space
)

CHECKED_DIRS = ("backend", "docs", "frontend/e2e", "frontend/src", "scripts")


def _text_files():
    for relative in CHECKED_DIRS:
        root = REPO_ROOT / relative
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def test_repository_text_has_no_windows1252_mojibake():
    offenders: list[str] = []
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: not valid UTF-8")
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if "\ufffd" in line or any(marker in line for marker in MOJIBAKE_SEQUENCES):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()[:120]}"
                )

    assert not offenders, "mojibake found:\n" + "\n".join(offenders)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(pytest.main([__file__, "-q"]))
