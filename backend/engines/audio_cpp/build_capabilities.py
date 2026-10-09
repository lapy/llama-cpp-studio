"""Runtime restrictions derived from the active audio.cpp installation."""

from typing import Optional


def server_ui_available(active: Optional[dict]) -> bool:
    """UI is opt-in; old/prebuilt installations without build evidence stay off."""
    settings = (active or {}).get("build_config") or {}
    value = settings.get("build_server_frontends")
    return value is True or isinstance(value, str) and value.strip().lower() in {"on", "true", "1"}


def constrain_ui_params(sections: list, active: Optional[dict]) -> list:
    if server_ui_available(active):
        return sections
    return [
        {
            **section,
            "params": [
                {
                    **row,
                    "supported": False,
                    "default": False,
                    "forced_value": False,
                    "description": "UI support was not enabled when this audio.cpp engine was installed.",
                }
                if row.get("key") in {"ui", "ui_management"}
                and row.get("scope", "process") == "process"
                else row
                for row in section.get("params") or []
            ],
        }
        for section in sections
    ]
