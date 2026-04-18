"""Pydantic models for /api/admin/* request bodies (and related marketing/announcements)."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class AdminUserUpdate(BaseModel):
    subscription_tier: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    flex_enabled: Optional[bool] = None


class AdminWalletAdjust(BaseModel):
    wallet: str = Field(..., pattern="^(put|aic)$")
    mode: str = Field(..., pattern="^(add|subtract|set)$")
    amount: int = Field(..., ge=0, le=999999)
    reason: str = Field(..., min_length=3, max_length=200)


class AdminUpdateEmailIn(BaseModel):
    email: EmailStr


class AdminResetPasswordIn(BaseModel):
    temp_password: str = Field(min_length=8, max_length=128)


class AdminEmailJobRunRequest(BaseModel):
    job: Literal["trial_reminders", "monthly_user_digest", "weekly_admin_digest", "scheduled_publish_alerts", "all"]


class PromoTogglesBody(BaseModel):
    """PATCH /api/admin/settings/promo-toggles — omit a field to leave it unchanged."""
    promo_burst_week_enabled: Optional[bool] = None
    promo_referral_enabled: Optional[bool] = None


class MarketingCampaignIn(BaseModel):
    name: str = Field(..., min_length=3, max_length=160)
    objective: str = Field(..., min_length=3, max_length=400)
    channel: str = Field(..., pattern="^(in_app|email|discount|mixed)$")
    range: str = Field(default="30d", max_length=16)
    tiers: List[str] = []
    min_uploads_30d: int = Field(default=0, ge=0, le=10000)
    min_enterprise_fit_score: float = Field(default=0, ge=0, le=100)
    min_nudge_ctr_pct: float = Field(default=0, ge=0, le=100)
    require_no_revenue_7d: bool = False
    schedule_at: Optional[datetime] = None
    notes: Optional[str] = Field(default="", max_length=4000)


class MarketingCampaignStatusIn(BaseModel):
    status: str = Field(..., pattern="^(draft|scheduled|active|paused|completed|cancelled)$")


class MarketingAIGenerateIn(BaseModel):
    range: str = Field(default="30d", max_length=16)
    objective: str = Field(default="revenue_growth", max_length=120)
    tone: str = Field(default="executive_clear", max_length=80)
    offer_style: str = Field(default="value_first", max_length=80)
    channel_mix: str = Field(default="mixed", max_length=40)
    force_deploy: bool = False


class AnnouncementAudienceIn(BaseModel):
    """Frontend admin.html sends { type, tiers?, userIds? }."""
    type: str = "all"
    tiers: List[str] = Field(default_factory=list)
    userIds: List[str] = Field(default_factory=list)


class AnnouncementRequest(BaseModel):
    title: str
    body: str
    channels: Optional[List[str]] = None
    audience: Optional[AnnouncementAudienceIn] = None
    send_email: Optional[bool] = None
    send_discord_community: Optional[bool] = None
    send_user_webhooks: Optional[bool] = None
    target: str = "all"
    target_tiers: List[str] = Field(default_factory=list)
