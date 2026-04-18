"""Pydantic models for upload / scheduled PATCH APIs (shared by app and routers/uploads)."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class UploadInit(BaseModel):
    filename: str
    file_size: int
    content_type: str
    platforms: List[str]
    target_accounts: List[str] = []  # platform_tokens.id UUIDs — publish to specific accounts
    title: str = ""
    caption: str = ""
    hashtags: List[str] = []
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None
    schedule_mode: str = "immediate"  # immediate | scheduled | smart
    has_telemetry: bool = False
    use_ai: bool = False
    smart_schedule_days: int = 7  # How many days to spread uploads across
    duration_seconds: Optional[float] = None
    thumbnail_count: Optional[int] = None
    thumbnail_use_studio_engine: Optional[bool] = None
    thumbnail_use_pikzels: Optional[bool] = None
    thumbnail_use_persona: Optional[bool] = None
    thumbnail_persona_id: Optional[str] = None
    thumbnail_persona_strength: Optional[int] = Field(default=None, ge=0, le=100)


class SmartScheduleOnlyUpdate(BaseModel):
    """PATCH /api/scheduled/{id} - only smart_schedule (platform -> ISO datetime string)."""

    smart_schedule: Dict[str, str] = Field(..., description="Platform -> ISO datetime string")


class UploadUpdate(BaseModel):
    """PATCH /api/uploads/{id} - title, caption, hashtags, scheduled_time, smart_schedule."""

    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    scheduled_time: Optional[datetime] = None
    smart_schedule: Optional[Dict[str, str]] = Field(None, description="Platform -> ISO datetime string")


class CompleteUploadBody(BaseModel):
    """Optional metadata from upload page (single-file manual title/caption/hashtags)."""

    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    privacy: Optional[str] = None
    target_accounts: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
