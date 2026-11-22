"""Immutable routing snapshots for the native Studio proxy.

Extracted from the manager so the snapshot builder is testable on its own and
the manager stays within its module budget. A snapshot is published on every
routing change; accepted work keeps the snapshot it resolved against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class RoutingSnapshot:
    aliases: object
    default_model: str
    revision: str

    @classmethod
    def build(cls, settings):
        aliases = {
            str(alias): tuple(
                str(target).strip()
                for target in (targets if isinstance(targets, list) else [targets])
                if str(target).strip()
            )
            for alias, targets in (settings.get("routing_aliases") or {}).items()
        }
        default = str(settings.get("default_model") or "").strip()
        revision = hashlib.sha256(
            json.dumps({"aliases": aliases, "default": default}, sort_keys=True).encode()
        ).hexdigest()[:16]
        return cls(MappingProxyType(aliases), default, revision)
