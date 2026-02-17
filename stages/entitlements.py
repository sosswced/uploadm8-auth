"""
UploadM8 Entitlements System
=============================
Defines tier-based entitlements and helper functions for the worker pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================
# Tier Configuration (mirrors PLAN_CONFIG from API server)
# ============================================================
TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "free": {
        "name": "Free", "price": 0,
        "put_daily": 1, "put_monthly": 30, "aic_monthly": 0,
        "max_accounts": 1,
        "watermark": True, "ads": True, "ai": False,
        "scheduling": False, "webhooks": False, "white_label": False,
        "excel": False, "priority": False, "flex": False,
        "hud": False,
    },
    "launch": {
        "name": "Launch", "price": 29,
        "put_daily": 5, "put_monthly": 600, "aic_monthly": 30,
        "max_accounts": 4,
        "watermark": True, "ads": False, "ai": True,
        "scheduling": False, "webhooks": False, "white_label": False,
        "excel": False, "priority": False, "flex": False,
        "hud": False,
    },
    "creator_pro": {
        "name": "Creator Pro", "price": 59,
        "put_daily": 10, "put_monthly": 1200, "aic_monthly": 300,
        "max_accounts": 8,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "excel": False, "priority": False, "flex": False,
        "hud": True,
    },
    "studio": {
        "name": "Studio", "price": 99,
        "put_daily": 25, "put_monthly": 3000, "aic_monthly": 1000,
        "max_accounts": 20,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "priority": False, "flex": False,
        "hud": True,
    },
    "agency": {
        "name": "Agency", "price": 199,
        "put_daily": 75, "put_monthly": 9000, "aic_monthly": 3000,
        "max_accounts": 60,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "priority": True, "flex": False,
        "hud": True,
    },
    "master_admin": {
        "name": "Admin", "price": 0,
        "put_daily": 9999, "put_monthly": 999999, "aic_monthly": 999999,
        "max_accounts": 999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "priority": True, "flex": True,
        "hud": True, "internal": True,
    },
    "friends_family": {
        "name": "Friends", "price": 0,
        "put_daily": 100, "put_monthly": 12000, "aic_monthly": 5000,
        "max_accounts": 80,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "priority": True, "flex": True,
        "hud": True, "internal": True,
    },
    "lifetime": {
        "name": "Lifetime", "price": 0,
        "put_daily": 100, "put_monthly": 12000, "aic_monthly": 5000,
        "max_accounts": 80,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "priority": True, "flex": True,
        "hud": True, "internal": True,
    },
}

STRIPE_LOOKUP_TO_TIER = {
    "uploadm8_launch_monthly": "launch",
    "uploadm8_creatorpro_monthly": "creator_pro",
    "uploadm8_studio_monthly": "studio",
    "uploadm8_agency_monthly": "agency",
}


# ============================================================
# Entitlements Dataclass
# ============================================================
@dataclass
class Entitlements:
    """Resolved entitlements for a user, combining tier + overrides."""

    tier: str = "free"
    tier_display: str = "Free"

    # Limits
    put_daily: int = 1
    put_monthly: int = 30
    aic_monthly: int = 0
    max_accounts: int = 1

    # Feature flags
    can_watermark: bool = True      # True = watermark IS applied (free tier)
    can_ai: bool = False
    can_schedule: bool = False
    can_webhooks: bool = False
    can_white_label: bool = False
    can_excel: bool = False
    can_priority: bool = False
    can_flex: bool = False
    can_burn_hud: bool = False
    show_ads: bool = True

    # Internal tier flag
    is_internal: bool = False


# ============================================================
# Builder Functions
# ============================================================
def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Build entitlements from a tier name."""
    cfg = TIER_CONFIG.get(tier.lower(), TIER_CONFIG["free"])
    return Entitlements(
        tier=tier.lower(),
        tier_display=cfg.get("name", tier.title()),
        put_daily=cfg.get("put_daily", 1),
        put_monthly=cfg.get("put_monthly", 30),
        aic_monthly=cfg.get("aic_monthly", 0),
        max_accounts=cfg.get("max_accounts", 1),
        can_watermark=cfg.get("watermark", True),
        can_ai=cfg.get("ai", False),
        can_schedule=cfg.get("scheduling", False),
        can_webhooks=cfg.get("webhooks", False),
        can_white_label=cfg.get("white_label", False),
        can_excel=cfg.get("excel", False),
        can_priority=cfg.get("priority", False),
        can_flex=cfg.get("flex", False),
        can_burn_hud=cfg.get("hud", False),
        show_ads=cfg.get("ads", True),
        is_internal=cfg.get("internal", False),
    )


def get_entitlements_from_user(user_record: dict, overrides: Optional[dict] = None) -> Entitlements:
    """
    Build entitlements from a user DB row + optional admin overrides.

    Args:
        user_record: Row from the users table (dict).
        overrides: Optional dict of per-user overrides from entitlement_overrides table.

    Returns:
        Fully resolved Entitlements.
    """
    tier = (user_record.get("subscription_tier") or "free").lower()
    ent = get_entitlements_for_tier(tier)

    # Apply per-user overrides (admin can grant features individually)
    if overrides:
        if overrides.get("can_ai") is not None:
            ent.can_ai = bool(overrides["can_ai"])
        if overrides.get("can_schedule") is not None:
            ent.can_schedule = bool(overrides["can_schedule"])
        if overrides.get("can_burn_hud") is not None:
            ent.can_burn_hud = bool(overrides["can_burn_hud"])
        if overrides.get("can_priority") is not None:
            ent.can_priority = bool(overrides["can_priority"])
        if overrides.get("can_flex") is not None:
            ent.can_flex = bool(overrides["can_flex"])
        if overrides.get("max_accounts") is not None:
            ent.max_accounts = int(overrides["max_accounts"])
        if overrides.get("can_watermark") is not None:
            ent.can_watermark = bool(overrides["can_watermark"])

    # flex_enabled on user record is a legacy per-user toggle
    if user_record.get("flex_enabled"):
        ent.can_flex = True

    return ent


def entitlements_to_dict(ent: Entitlements) -> dict:
    """Serialize entitlements to a JSON-safe dict."""
    return {
        "tier": ent.tier,
        "tier_display": ent.tier_display,
        "put_daily": ent.put_daily,
        "put_monthly": ent.put_monthly,
        "aic_monthly": ent.aic_monthly,
        "max_accounts": ent.max_accounts,
        "can_watermark": ent.can_watermark,
        "can_ai": ent.can_ai,
        "can_schedule": ent.can_schedule,
        "can_webhooks": ent.can_webhooks,
        "can_white_label": ent.can_white_label,
        "can_excel": ent.can_excel,
        "can_priority": ent.can_priority,
        "can_flex": ent.can_flex,
        "can_burn_hud": ent.can_burn_hud,
        "show_ads": ent.show_ads,
        "is_internal": ent.is_internal,
    }


# ============================================================
# Guard Helpers (used by upload presign, account connect, etc.)
# ============================================================
def can_user_upload(user_record: dict, overrides: Optional[dict] = None) -> bool:
    """Check if user's tier allows uploading."""
    ent = get_entitlements_from_user(user_record, overrides)
    return ent.put_daily > 0


def can_user_connect_platform(user_record: dict, current_count: int, overrides: Optional[dict] = None) -> bool:
    """Check if user can connect another platform account."""
    ent = get_entitlements_from_user(user_record, overrides)
    return current_count < ent.max_accounts


def get_tier_display_name(tier: str) -> str:
    """Get human-readable tier name."""
    cfg = TIER_CONFIG.get(tier.lower(), {})
    return cfg.get("name", tier.title())


def get_tier_from_lookup_key(lookup_key: str) -> str:
    """Convert Stripe lookup key to internal tier name."""
    return STRIPE_LOOKUP_TO_TIER.get(lookup_key, "free")
