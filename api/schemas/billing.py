"""Pydantic models for billing API routes."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CheckoutRequest(BaseModel):
    lookup_key: str
    kind: str = "subscription"  # subscription | topup | addon


class BillingSubscriptionActionRequest(BaseModel):
    action: Literal[
        "pause_payment_collection",
        "share_payment_update_link",
        "create_one_time_invoice",
        "cancel_subscription",
    ]
    amount_cents: Optional[int] = Field(default=None, ge=100, le=5000000)
    currency: str = "usd"
    description: Optional[str] = None


class UploadCostEstimateRequest(BaseModel):
    num_publish_targets: int = Field(default=1, ge=1, le=100)
    use_ai: bool = True
    use_hud: bool = False
    num_thumbnails: int = Field(default=1, ge=1, le=20)
    duration_seconds: Optional[float] = None
    file_size: Optional[int] = None
    has_telemetry: bool = False
