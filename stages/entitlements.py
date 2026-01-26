"""
UploadM8 Entitlements
=====================
Centralized tier gating logic. Stripe webhook is source of truth.
"""

from .context import Entitlements


# Tier definitions - single source of truth
TIER_CONFIG = {
    "starter": {
        "price": 0,
        "max_uploads": 10,
        "max_accounts": 1,
        "can_generate_captions": False,
        "can_burn_hud": False,
        "can_use_ai_captions": False,
        "priority_processing": False,
        "can_schedule": False,
        "can_export": False,
    },
    "solo": {
        "price": 9.99,
        "max_uploads": 60,
        "max_accounts": 2,
        "can_generate_captions": False,
        "can_burn_hud": True,
        "can_use_ai_captions": False,
        "priority_processing": False,
        "can_schedule": True,
        "can_export": False,
    },
    "creator": {
        "price": 19.99,
        "max_uploads": 200,
        "max_accounts": 4,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": False,
        "priority_processing": True,
        "can_schedule": True,
        "can_export": True,
    },
    "growth": {
        "price": 29.99,
        "max_uploads": 500,
        "max_accounts": 8,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "priority_processing": True,
        "can_schedule": True,
        "can_export": True,
    },
    "studio": {
        "price": 49.99,
        "max_uploads": 1500,
        "max_accounts": 15,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "priority_processing": True,
        "can_schedule": True,
        "can_export": True,
    },
    "agency": {
        "price": 99.99,
        "max_uploads": 5000,
        "max_accounts": 40,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "priority_processing": True,
        "can_schedule": True,
        "can_export": True,
    },
    "lifetime": {
        "price": 0,
        "max_uploads": 999999,
        "max_accounts": 999,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "priority_processing": True,
        "can_schedule": True,
        "can_export": True,
    },
    "friends_family": {
        "price": 0,
        "max_uploads": 999999,
        "max_accounts": 999,
        "can_generate_captions": True,
        "can_burn_hud": True,
        "can_use_ai_captions": True,
        "priority_processing": True,
        "can_schedule": True,
        "can_export": True,
    },
}


def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Get entitlements for a subscription tier."""
    tier = tier.lower().strip() if tier else "starter"
    config = TIER_CONFIG.get(tier, TIER_CONFIG["starter"])
    
    return Entitlements(
        tier=tier,
        can_generate_captions=config["can_generate_captions"],
        can_burn_hud=config["can_burn_hud"],
        can_use_ai_captions=config["can_use_ai_captions"],
        max_uploads_per_month=config["max_uploads"],
        max_accounts=config["max_accounts"],
        priority_processing=config["priority_processing"],
        can_schedule=config["can_schedule"],
        can_export=config["can_export"],
    )


def get_entitlements_from_user(user: dict) -> Entitlements:
    """Get entitlements from user database record."""
    # Check for unlimited first (lifetime, friends_family, admin grants)
    if user.get("unlimited_uploads"):
        tier = user.get("subscription_tier") or "lifetime"
        if tier in ("lifetime", "friends_family"):
            return get_entitlements_for_tier(tier)
        # Has unlimited but different tier - use unlimited config
        return get_entitlements_for_tier("lifetime")
    
    # Check for explicit upload_quota override
    tier = user.get("subscription_tier") or "starter"
    entitlements = get_entitlements_for_tier(tier)
    
    # Override max_uploads if explicitly set
    if user.get("upload_quota"):
        entitlements.max_uploads_per_month = user["upload_quota"]
    
    return entitlements


def can_user_upload(user: dict) -> tuple[bool, str]:
    """Check if user can upload. Returns (allowed, reason)."""
    if user.get("unlimited_uploads"):
        return True, "unlimited"
    
    quota = user.get("upload_quota", 10)
    used = user.get("uploads_this_month", 0)
    
    if used >= quota:
        return False, f"Monthly quota exceeded ({used}/{quota})"
    
    return True, f"OK ({used}/{quota})"


def tier_display_name(tier: str) -> str:
    """Get display name for a tier."""
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


def tier_price(tier: str) -> float:
    """Get monthly price for a tier."""
    config = TIER_CONFIG.get(tier.lower(), TIER_CONFIG["starter"])
    return config["price"]
