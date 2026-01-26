"""
UploadM8 Entitlements Module
============================
Centralized tier gating - all subscription logic in one place.

Tier Features Matrix:
---------------------
| Tier        | Quota | Accounts | Captions | HUD | AI | Team | Priority | History |
|-------------|-------|----------|----------|-----|-----|------|----------|---------|
| Starter     | 10    | 1        | No       | No  | No  | 1    | No       | 7d      |
| Solo        | 60    | 2        | No       | Yes | No  | 1    | No       | 14d     |
| Creator     | 200   | 4        | Yes      | Yes | No  | 1    | Yes      | 30d     |
| Growth      | 500   | 8        | Yes      | Yes | Yes | 1    | Yes      | 30d     |
| Studio      | 1500  | 15       | Yes      | Yes | Yes | 3    | Yes      | 90d     |
| Agency      | 5000  | 40       | Yes      | Yes | Yes | 10   | Yes      | 365d    |
| Lifetime    | ∞     | 100      | Yes      | Yes | Yes | 5    | Yes      | 365d    |
| F&F         | ∞     | 100      | Yes      | Yes | Yes | 5    | Yes      | 365d    |
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Entitlements:
    """User entitlements based on subscription tier."""
    tier: str
    upload_quota: int
    unlimited_uploads: bool
    max_accounts: int
    can_generate_captions: bool
    can_burn_hud: bool
    can_use_ai_captions: bool
    team_seats: int
    priority_processing: bool
    history_days: int
    can_schedule: bool
    can_use_templates: bool
    can_export: bool
    can_use_webhooks: bool
    support_tier: str  # basic, standard, priority, sla


# Complete tier configuration with ALL advertised features
TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "starter": {
        "tier": "starter",
        "upload_quota": 10,
        "unlimited_uploads": False,
        "max_accounts": 1,
        "can_generate_captions": False,
        "can_burn_hud": False,
        "can_use_ai_captions": False,
        "team_seats": 1,
        "priority_processing": False,
        "history_days": 7,
        "can_schedule": False,
        "can_use_templates": False,
        "can_export": False,
        "can_use_webhooks": False,
        "support_tier": "basic",
    },
    "solo": {
        "tier": "solo",
        "upload_quota": 60,
        "unlimited_uploads": False,
        "max_accounts": 2,
        "can_generate_captions": False,
        "can_burn_hud": True,
        "can_use_ai_captions": False,
        "team_seats": 1,
        "priority_processing": False,
        "history_days": 14,
        "can_schedule": True,
        "can_use_templates": False,
        "can_export": False,
        "can_use_webhooks": False,
        "support_tier": "basic",
    },
    "creator": {
        "tier": "creator",
        "upload_quota": 200,
        "unlimited_uploads": False,
        "max_accounts": 4,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": False,
        "team_seats": 1,
        "priority_processing": True,
        "history_days": 30,
        "can_schedule": True,
        "can_use_templates": True,
        "can_export": False,
        "can_use_webhooks": False,
        "support_tier": "standard",
    },
    "growth": {
        "tier": "growth",
        "upload_quota": 500,
        "unlimited_uploads": False,
        "max_accounts": 8,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "team_seats": 1,
        "priority_processing": True,
        "history_days": 30,
        "can_schedule": True,
        "can_use_templates": True,
        "can_export": True,
        "can_use_webhooks": True,
        "support_tier": "standard",
    },
    "studio": {
        "tier": "studio",
        "upload_quota": 1500,
        "unlimited_uploads": False,
        "max_accounts": 15,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "team_seats": 3,
        "priority_processing": True,
        "history_days": 90,
        "can_schedule": True,
        "can_use_templates": True,
        "can_export": True,
        "can_use_webhooks": True,
        "support_tier": "priority",
    },
    "agency": {
        "tier": "agency",
        "upload_quota": 5000,
        "unlimited_uploads": False,
        "max_accounts": 40,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "team_seats": 10,
        "priority_processing": True,
        "history_days": 365,
        "can_schedule": True,
        "can_use_templates": True,
        "can_export": True,
        "can_use_webhooks": True,
        "support_tier": "sla",
    },
    "lifetime": {
        "tier": "lifetime",
        "upload_quota": 999999,
        "unlimited_uploads": True,
        "max_accounts": 100,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "team_seats": 5,
        "priority_processing": True,
        "history_days": 365,
        "can_schedule": True,
        "can_use_templates": True,
        "can_export": True,
        "can_use_webhooks": True,
        "support_tier": "priority",
    },
    "friends_family": {
        "tier": "friends_family",
        "upload_quota": 999999,
        "unlimited_uploads": True,
        "max_accounts": 100,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "team_seats": 5,
        "priority_processing": True,
        "history_days": 365,
        "can_schedule": True,
        "can_use_templates": True,
        "can_export": True,
        "can_use_webhooks": True,
        "support_tier": "priority",
    },
}


def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Get entitlements object for a subscription tier."""
    tier_lower = (tier or "starter").lower().strip()
    config = TIER_CONFIG.get(tier_lower, TIER_CONFIG["starter"])
    return Entitlements(**config)


def get_entitlements_from_user(user: dict) -> Entitlements:
    """Get entitlements from user record."""
    tier = user.get("subscription_tier", "starter")
    ent = get_entitlements_for_tier(tier)
    
    # Allow admin override of quota
    custom_quota = user.get("upload_quota")
    if custom_quota and custom_quota > ent.upload_quota:
        ent.upload_quota = custom_quota
    
    return ent


def can_user_upload(user: dict) -> tuple:
    """Check if user can upload based on quota. Returns (can_upload, message)."""
    ent = get_entitlements_from_user(user)
    
    if ent.unlimited_uploads:
        return True, "ok"
    
    used = user.get("uploads_this_month", 0)
    if used >= ent.upload_quota:
        return False, f"Monthly upload quota reached ({used}/{ent.upload_quota})"
    
    return True, "ok"


def can_user_connect_platform(user: dict, current_connected: int) -> tuple:
    """Check if user can connect another platform account. Returns (can_connect, message)."""
    ent = get_entitlements_from_user(user)
    
    if current_connected >= ent.max_accounts:
        return False, f"Account limit reached ({current_connected}/{ent.max_accounts}). Upgrade to connect more."
    
    return True, "ok"


def get_tier_display_name(tier: str) -> str:
    """Get human-readable tier name."""
    names = {
        "starter": "Starter (Free)",
        "solo": "Solo",
        "creator": "Creator",
        "growth": "Growth",
        "studio": "Studio",
        "agency": "Agency",
        "lifetime": "Lifetime",
        "friends_family": "Friends & Family",
    }
    return names.get(tier.lower(), tier.title())


def get_tier_from_lookup_key(lookup_key: str) -> str:
    """Map Stripe lookup key to tier name."""
    mapping = {
        "uploadm8_starter_monthly": "starter",
        "uploadm8_solo_monthly": "solo",
        "uploadm8_creator_monthly": "creator",
        "uploadm8_growth_monthly": "growth",
        "uploadm8_studio_monthly": "studio",
        "uploadm8_agency_monthly": "agency",
        "uploadm8_starter_yearly": "starter",
        "uploadm8_solo_yearly": "solo",
        "uploadm8_creator_yearly": "creator",
        "uploadm8_growth_yearly": "growth",
        "uploadm8_studio_yearly": "studio",
        "uploadm8_agency_yearly": "agency",
    }
    return mapping.get(lookup_key.strip().lower(), "starter")


def entitlements_to_dict(ent: Entitlements) -> dict:
    """Convert Entitlements to dictionary for API responses."""
    return {
        "tier": ent.tier,
        "tier_display": get_tier_display_name(ent.tier),
        "upload_quota": ent.upload_quota,
        "unlimited_uploads": ent.unlimited_uploads,
        "max_accounts": ent.max_accounts,
        "can_generate_captions": ent.can_generate_captions,
        "can_burn_hud": ent.can_burn_hud,
        "can_use_ai_captions": ent.can_use_ai_captions,
        "team_seats": ent.team_seats,
        "priority_processing": ent.priority_processing,
        "history_days": ent.history_days,
        "can_schedule": ent.can_schedule,
        "can_use_templates": ent.can_use_templates,
        "can_export": ent.can_export,
        "can_use_webhooks": ent.can_use_webhooks,
        "support_tier": ent.support_tier,
    }
