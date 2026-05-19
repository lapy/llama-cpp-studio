"""API for model configuration templates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from backend.data_store import get_store
from backend.model_config import config_api_response, normalize_model_config
from backend.model_config_templates import (
    apply_template_to_config,
    extract_template_config,
    new_template_record,
)

router = APIRouter()


class TemplateCreateBody(BaseModel):
    name: str
    description: str = ""
    config: Optional[Dict[str, Any]] = None
    include_routing: bool = False
    engines_scope: str = Field(default="all", pattern="^(all|active)$")
    source_model_id: Optional[str] = None


class TemplateUpdateBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


def _template_summary(item: dict) -> dict:
    cfg = item.get("config") or {}
    engines = cfg.get("engines") or {}
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "description": item.get("description", ""),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "source_model_id": item.get("source_model_id"),
        "include_routing": bool(item.get("include_routing")),
        "engines_scope": item.get("engines_scope", "all"),
        "engine": cfg.get("engine"),
        "engine_ids": sorted(engines.keys()),
    }


@router.get("")
async def list_templates() -> List[dict]:
    store = get_store()
    items = store.list_config_templates()
    return sorted(
        [_template_summary(t) for t in items],
        key=lambda x: (x.get("name") or "").lower(),
    )


@router.get("/{template_id}")
async def get_template(template_id: str) -> dict:
    store = get_store()
    item = store.get_config_template(template_id)
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return item


@router.post("")
async def create_template(body: TemplateCreateBody) -> dict:
    if not body.config:
        raise HTTPException(status_code=400, detail="config is required")
    try:
        record = new_template_record(
            name=body.name,
            description=body.description,
            config=body.config,
            source_model_id=body.source_model_id,
            include_routing=body.include_routing,
            engines_scope=body.engines_scope,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    store = get_store()
    store.add_config_template(record)
    return record


@router.put("/{template_id}")
async def update_template(template_id: str, body: TemplateUpdateBody) -> dict:
    store = get_store()
    existing = store.get_config_template(template_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Template not found")
    updates: Dict[str, Any] = {}
    if body.name is not None:
        trimmed = body.name.strip()
        if not trimmed:
            raise HTTPException(status_code=400, detail="Template name cannot be empty")
        updates["name"] = trimmed
    if body.description is not None:
        updates["description"] = body.description.strip()
    if not updates:
        return existing
    from backend.model_config_templates import _utc_now_iso

    updates["updated_at"] = _utc_now_iso()
    updated = store.update_config_template(template_id, updates)
    return updated or existing


@router.delete("/{template_id}")
async def delete_template(template_id: str) -> dict:
    store = get_store()
    if not store.delete_config_template(template_id):
        raise HTTPException(status_code=404, detail="Template not found")
    return {"ok": True, "id": template_id}
