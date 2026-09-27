"""Shared audio task-family profile lookup and field-group builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend.audio.fields import field_spec as default_field_spec


GroupMeta = Tuple[str, str, str]


@dataclass
class TaskProfileSet:
    """One task family: aliases, curated profiles, and grouped request fields."""

    tasks: frozenset[str]
    profiles: Dict[str, Dict[str, Any]]
    group_meta: Sequence[GroupMeta] = ()
    aliases: Dict[str, str] = field(default_factory=dict)
    field_spec: Callable[..., Dict[str, Any]] = default_field_spec

    def matches_task(self, task: Optional[str]) -> bool:
        return str(task or "").strip().lower() in self.tasks

    def canonical_family(self, family: Optional[str]) -> str:
        key = str(family or "").strip().lower()
        return self.aliases.get(key, key)

    def profile_for_family(self, family: Optional[str]) -> Optional[Dict[str, Any]]:
        key = self.canonical_family(family)
        if not key:
            return None
        return self.profiles.get(key)

    def field_groups(self, family: Optional[str]) -> List[Dict[str, Any]]:
        profile = self.profile_for_family(family) or {}
        hints = profile.get("field_hints") or {}
        groups: List[Dict[str, Any]] = []
        for group_id, label, description in self.group_meta:
            keys = profile.get(f"{group_id}_fields") or []
            fields = []
            for key in keys:
                spec = dict(self.field_spec(key))
                hint = hints.get(key)
                if hint:
                    spec["hint"] = hint
                fields.append(spec)
            if fields:
                groups.append(
                    {
                        "id": group_id,
                        "label": label,
                        "description": description,
                        "fields": fields,
                    }
                )
        return groups
