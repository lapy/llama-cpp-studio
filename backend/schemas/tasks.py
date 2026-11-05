"""Typed envelopes for task events and upstream update checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskEvent(BaseModel):
    task_id: str
    type: str = ""
    description: str = ""
    progress: float = 0.0
    status: str = "running"
    message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[float] = None
    seq: Optional[int] = None


class TaskSnapshot(BaseModel):
    tasks: List[TaskEvent] = Field(default_factory=list)
    reason: str = "snapshot"


class CommitSummary(BaseModel):
    sha: Optional[str] = None
    commit_date: Optional[str] = None
    message: Optional[str] = None


class ReleaseSummary(BaseModel):
    tag_name: str
    published_at: Optional[str] = None
    html_url: Optional[str] = None


class UpdateCheckResponse(BaseModel):
    latest_release: Optional[ReleaseSummary] = None
    latest_commit: Optional[CommitSummary] = None


class SessionLogin(BaseModel):
    token: str
