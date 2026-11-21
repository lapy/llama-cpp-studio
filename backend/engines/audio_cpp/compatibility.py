"""audio.cpp engine-version compatibility ceiling.

Studio can install and run audio.cpp releases newer than the newest one its
integration surface was validated against, but it must not imply that newer
surface is supported. This module maps a Studio version label (``v0.9.5``,
``release-0.9.4``, ``0.9.3``, a source build of a branch or commit, ...) to one
of three states:

* ``supported`` - a release within the validated range;
* ``unsupported`` - a release outside it (newer than the ceiling, or older than
  the minimum build Studio can activate);
* ``unknown`` - not a release spelling (branch, commit, local build), so Studio
  makes no claim either way.

Nothing here blocks an install or an activation. The backend reports the
ceiling; the UI and docs decide how loudly to say it. The durable matrix lives
in ``docs/audio-cpp-compatibility.md``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# The newest upstream release whose Linux prebuilt and loader/inspect surface
# were actually exercised against Studio. Upstream v0.9.5 (2026-11-14) is newer
# and moves fast: v0.9.2 normalized route aliases/options, v0.9.3 changed
# ``instruct``/``instructions``, and v0.9.4/v0.9.5 Linux prebuilts require
# glibc 2.43. Those releases stay outside the validated set until this machine
# (or CI with an ubuntu:24.04-class image) actually exercises them.
VALIDATED_ENGINE_VERSION = "0.9.3"

# Studio activation hard-requires ``--list-loaders --json``
# (``activation.assert_activation_supported``), which upstream added in 0.9.1.
# Older releases cannot activate at all, so they are reported unsupported
# rather than "below the ceiling".
MINIMUM_ENGINE_VERSION = "0.9.1"

# Release spellings upstream/Studio use: ``v0.9.5``, ``release-0.9.4``,
# ``0.9.3``, with an optional ``-cuda``/``+local`` build suffix. A bare
# commit, branch, or local build name does not match and stays unknown.
_RELEASE_VERSION_RE = re.compile(
    r"^(?:release[-_])?v?(?P<number>\d+\.\d+(?:\.\d+)?)(?P<suffix>[-+].*)?$",
    re.IGNORECASE,
)


def parse_engine_version(value: Any) -> Optional[str]:
    """Return the canonical ``major.minor.patch`` release for *value*.

    ``v0.9.5``, ``release-0.9.4``, ``0.9.3``, ``v0.9.5-cuda`` and
    ``release-0.9.4+local`` all parse. A bare commit, a branch, a source-build
    label (``source-main-...``), or anything else returns ``None`` - the caller
    must not infer compatibility from it.
    """
    text = _clean(value)
    if not text:
        return None
    match = _RELEASE_VERSION_RE.fullmatch(text)
    if match is None:
        return None
    parts = [int(part) for part in match.group("number").split(".")]
    while len(parts) < 3:
        parts.append(0)
    return ".".join(str(part) for part in parts)


def engine_compatibility(version: Any) -> Dict[str, Any]:
    """Classify an installed/source engine version against the ceiling.

    Returns ``engine_version`` (the parsed canonical release, or the raw label
    when it is not a release), ``validated_through``, ``supported``
    (``True``/``False``/``None``), ``status`` (``supported``/``unsupported``/
    ``unknown``), and a short human-readable ``note``. An unknown version is
    never reported as supported and never reported as unsupported: it stays
    unknown.
    """
    label = _clean(version)
    parsed = parse_engine_version(label)
    validated = VALIDATED_ENGINE_VERSION

    if parsed is None:
        note = (
            "No audio.cpp engine version was supplied; compatibility is unknown."
            if not label
            else f"'{label}' is not an audio.cpp release version; compatibility is unknown, "
            "not assumed."
        )
        return _result(label, validated, None, "unknown", note)

    number = _version_tuple(parsed)
    if number > _version_tuple(validated):
        return _result(
            parsed,
            validated,
            False,
            "unsupported",
            f"audio.cpp {parsed} is newer than the newest validated release ({validated}); "
            "runs unblocked, but 0.9.4+ Linux prebuilts need glibc 2.43.",
        )
    if number < _version_tuple(MINIMUM_ENGINE_VERSION):
        return _result(
            parsed,
            validated,
            False,
            "unsupported",
            f"audio.cpp {parsed} predates the minimum build Studio can activate "
            f"({MINIMUM_ENGINE_VERSION}, structured loader JSON).",
        )
    return _result(
        parsed,
        validated,
        True,
        "supported",
        f"audio.cpp {parsed} is within the validated range ({MINIMUM_ENGINE_VERSION}-{validated}).",
    )


def compatibility_fields(version: Any) -> Dict[str, Any]:
    """API-shaped compatibility fields shared by status, contract, and rows."""
    compat = engine_compatibility(version)
    return {
        "validated_through": compat["validated_through"],
        "engine_supported": compat["supported"],
        "engine_compat_note": compat["note"],
    }


def _result(
    engine_version: str,
    validated_through: str,
    supported: Optional[bool],
    status: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "engine_version": engine_version,
        "validated_through": validated_through,
        "supported": supported,
        "status": status,
        "note": note,
    }


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _version_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


__all__ = [
    "MINIMUM_ENGINE_VERSION",
    "VALIDATED_ENGINE_VERSION",
    "compatibility_fields",
    "engine_compatibility",
    "parse_engine_version",
]
