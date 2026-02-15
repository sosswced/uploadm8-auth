"""
UploadM8 Entitlements Module
============================
Centralized tier gating - ALL subscription logic in one place.
Server-authoritative: Frontend displays based on what backend allows.

Feature Flags (Master Admin Controllable):
- ads_enabled: Show ads in dashboard
- watermark_enabled: Burn UploadM8 watermark
- ai_captions_enabled: AI-generated captions
- ai_thumbnails_enabled: AI-generated thumbnails
- ai_hashtags_enabled: AI-generated hashtags
- max_uploads_month: Monthly upload limit
- max_hashtags: Maximum hashtags per upload
- max_accounts: Connected platform accounts
- priority_processing: Priority job queue
- white_label_enabled: Custom logo/branding
- smart_scheduling_enabled: AI-optimized posting times
- excel_export_enabled: Export data to Excel
- analytics_enabled: Full analytics access

Tier Matrix:
| Tier         | Quota  | Accts | AI Caption | AI Thumb | Hashtags | WM  | Ads | White Label |
|--------------|--------|-------|------------|----------|----------|-----|-----|-------------|
| free         | 5      | 1     | No         | No       | 2        | Yes | Yes | No          |
| starter      | 10     | 1     | No         | No       | 3        | Yes | Yes | No          |
| solo         | 60     | 2     | No         | No       | 5        | Yes | No  | No          |
| creator      | 200    | 4     | Yes        | Yes      | 15       | No  | No  | No          |
| growth       | 500    | 8     | Yes        | Yes      | 30       | No  | No  | No          |
| studio       | 1500   | 15    | Yes        | Yes      | 50       | No  | No  | Yes         |
| agency       | 5000   | 50    | Yes        | Yes      | ∞        | No  | No  | Yes         |
| lifetime     | ∞      | 100   | Yes        | Yes      | ∞        | No  | No  | Yes         |
| friends_fam  | ∞      | 100   | Yes        | Yes      | ∞        | No  | No  | Yes         |
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Entitlements:
    """Complete user entitlements based on subscription tier + overrides."""
    # Core tier info
    tier: str = "free"
    tier_display: str = "Free"
    
    # Upload limits
    upload_quota: int = 5
    unlimited_uploads: bool = False
    uploads_this_month: int = 0
    
    # Account limits
    max_accounts: int = 1
    max_accounts_per_platform: int = 1
    
    # AI Features
    ai_captions_enabled: bool = False
    ai_thumbnails_enabled: bool = False
    ai_hashtags_enabled: bool = False
    
    # Hashtags
    max_hashtags: int = 2
    unlimited_hashtags: bool = False
    always_use_hashtags: bool = False
    default_hashtag_count: int = 0
    
    # Branding
    show_watermark: bool = True
    show_ads: bool = True
    white_label_enabled: bool = False
    custom_logo_url: Optional[str] = None
    
    # Processing
    can_burn_hud: bool = False
    priority_processing: bool = False
    
    # Scheduling
    can_schedule: bool = False
    smart_scheduling_enabled: bool = False
    
    # Analytics & Export
    analytics_enabled: bool = False
    full_analytics_enabled: bool = False
    excel_export_enabled: bool = False
    
    # Team & History
    team_seats: int = 1
    history_days: int = 7
    
    # Templates & Webhooks
    can_use_templates: bool = False
    can_use_webhooks: bool = False
    
    # Support
    support_tier: str = "basic"  # basic, standard, priority, sla
    
    # Admin flags (can be toggled per-user)
    custom_overrides: Dict[str, Any] = field(default_factory=dict)


# Complete tier configuration
TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "free": {
        "tier": "free",
        "tier_display": "Free",
        "upload_quota": 5,
        "unlimited_uploads": False,
        "max_accounts": 1,
        "max_accounts_per_platform": 1,
        "ai_captions_enabled": False,
        "ai_thumbnails_enabled": False,
        "ai_hashtags_enabled": False,
        "max_hashtags": 2,
        "unlimited_hashtags": False,
        "show_watermark": True,
        "show_ads": True,
        "white_label_enabled": False,
        "can_burn_hud": False,
        "priority_processing": False,
        "can_schedule": False,
        "smart_scheduling_enabled": False,
        "analytics_enabled": False,
        "full_analytics_enabled": False,
        "excel_export_enabled": False,
        "team_seats": 1,
        "history_days": 7,
        "can_use_templates": False,
        "can_use_webhooks": False,
        "support_tier": "basic",
    },
    "starter": {
        "tier": "starter",
        "tier_display": "Starter",
        "upload_quota": 10,
        "unlimited_uploads": False,
        "max_accounts": 1,
        "max_accounts_per_platform": 1,
        "ai_captions_enabled": False,
        "ai_thumbnails_enabled": False,
        "ai_hashtags_enabled": False,
        "max_hashtags": 3,
        "unlimited_hashtags": False,
        "show_watermark": True,
        "show_ads": True,
        "white_label_enabled": False,
        "can_burn_hud": False,
        "priority_processing": False,
        "can_schedule": False,
        "smart_scheduling_enabled": False,
        "analytics_enabled": True,
        "full_analytics_enabled": False,
        "excel_export_enabled": False,
        "team_seats": 1,
        "history_days": 7,
        "can_use_templates": False,
        "can_use_webhooks": False,
        "support_tier": "basic",
    },
    "solo": {
        "tier": "solo",
        "tier_display": "Solo",
        "upload_quota": 60,
        "unlimited_uploads": False,
        "max_accounts": 2,
        "max_accounts_per_platform": 1,
        "ai_captions_enabled": False,
        "ai_thumbnails_enabled": False,
        "ai_hashtags_enabled": False,
        "max_hashtags": 5,
        "unlimited_hashtags": False,
        "show_watermark": True,
        "show_ads": False,
        "white_label_enabled": False,
        "can_burn_hud": True,
        "priority_processing": False,
        "can_schedule": True,
        "smart_scheduling_enabled": False,
        "analytics_enabled": True,
        "full_analytics_enabled": False,
        "excel_export_enabled": False,
        "team_seats": 1,
        "history_days": 14,
        "can_use_templates": False,
        "can_use_webhooks": False,
        "support_tier": "basic",
    },
    "creator": {
        "tier": "creator",
        "tier_display": "Creator",
        "upload_quota": 200,
        "unlimited_uploads": False,
        "max_accounts": 4,
        "max_accounts_per_platform": 2,
        "ai_captions_enabled": True,
        "ai_thumbnails_enabled": True,
        "ai_hashtags_enabled": True,
        "max_hashtags": 15,
        "unlimited_hashtags": False,
        "show_watermark": False,
        "show_ads": False,
        "white_label_enabled": False,
        "can_burn_hud": True,
        "priority_processing": True,
        "can_schedule": True,
        "smart_scheduling_enabled": True,
        "analytics_enabled": True,
        "full_analytics_enabled": True,
        "excel_export_enabled": False,
        "team_seats": 1,
        "history_days": 30,
        "can_use_templates": True,
        "can_use_webhooks": True,
        "support_tier": "standard",
    },
    "growth": {
        "tier": "growth",
        "tier_display": "Growth",
        "upload_quota": 500,
        "unlimited_uploads": False,
        "max_accounts": 8,
        "max_accounts_per_platform": 3,
        "ai_captions_enabled": True,
        "ai_thumbnails_enabled": True,
        "ai_hashtags_enabled": True,
        "max_hashtags": 30,
        "unlimited_hashtags": False,
        "show_watermark": False,
        "show_ads": False,
        "white_label_enabled": False,
        "can_burn_hud": True,
        "priority_processing": True,
        "can_schedule": True,
        "smart_scheduling_enabled": True,
        "analytics_enabled": True,
        "full_analytics_enabled": True,
        "excel_export_enabled": True,
        "team_seats": 1,
        "history_days": 30,
        "can_use_templates": True,
        "can_use_webhooks": True,
        "support_tier": "standard",
    },
    "studio": {
        "tier": "studio",
        "tier_display": "Studio",
        "upload_quota": 1500,
        "unlimited_uploads": False,
        "max_accounts": 15,
        "max_accounts_per_platform": 5,
        "ai_captions_enabled": True,
        "ai_thumbnails_enabled": True,
        "ai_hashtags_enabled": True,
        "max_hashtags": 50,
        "unlimited_hashtags": False,
        "show_watermark": False,
        "show_ads": False,
        "white_label_enabled": True,
        "can_burn_hud": True,
        "priority_processing": True,
        "can_schedule": True,
        "smart_scheduling_enabled": True,
        "analytics_enabled": True,
        "full_analytics_enabled": True,
        "excel_export_enabled": True,
        "team_seats": 3,
        "history_days": 90,
        "can_use_templates": True,
        "can_use_webhooks": True,
        "support_tier": "priority",
    },
    "agency": {
        "tier": "agency",
        "tier_display": "Agency",
        "upload_quota": 5000,
        "unlimited_uploads": False,
        "max_accounts": 50,
        "max_accounts_per_platform": 15,
        "ai_captions_enabled": True,
        "ai_thumbnails_enabled": True,
        "ai_hashtags_enabled": True,
        "max_hashtags": 9999,
        "unlimited_hashtags": True,
        "show_watermark": False,
        "show_ads": False,
        "white_label_enabled": True,
        "can_burn_hud": True,
        "priority_processing": True,
        "can_schedule": True,
        "smart_scheduling_enabled": True,
        "analytics_enabled": True,
        "full_analytics_enabled": True,
        "excel_export_enabled": True,
        "team_seats": 10,
        "history_days": 365,
        "can_use_templates": True,
        "can_use_webhooks": True,
        "support_tier": "sla",
    },
    "lifetime": {
        "tier": "lifetime",
        "tier_display": "Lifetime",
        "upload_quota": 999999,
        "unlimited_uploads": True,
        "max_accounts": 100,
        "max_accounts_per_platform": 25,
        "ai_captions_enabled": True,
        "ai_thumbnails_enabled": True,
        "ai_hashtags_enabled": True,
        "max_hashtags": 9999,
        "unlimited_hashtags": True,
        "show_watermark": False,
        "show_ads": False,
        "white_label_enabled": True,
        "can_burn_hud": True,
        "priority_processing": True,
        "can_schedule": True,
        "smart_scheduling_enabled": True,
        "analytics_enabled": True,
        "full_analytics_enabled": True,
        "excel_export_enabled": True,
        "team_seats": 5,
        "history_days": 365,
        "can_use_templates": True,
        "can_use_webhooks": True,
        "support_tier": "priority",
    },
    "friends_family": {
        "tier": "friends_family",
        "tier_display": "Friends & Family",
        "upload_quota": 999999,
        "unlimited_uploads": True,
        "max_accounts": 100,
        "max_accounts_per_platform": 25,
        "ai_captions_enabled": True,
        "ai_thumbnails_enabled": True,
        "ai_hashtags_enabled": True,
        "max_hashtags": 9999,
        "unlimited_hashtags": True,
        "show_watermark": False,
        "show_ads": False,
        "white_label_enabled": True,
        "can_burn_hud": True,
        "priority_processing": True,
        "can_schedule": True,
        "smart_scheduling_enabled": True,
        "analytics_enabled": True,
        "full_analytics_enabled": True,
        "excel_export_enabled": True,
        "team_seats": 5,
        "history_days": 365,
        "can_use_templates": True,
        "can_use_webhooks": True,
        "support_tier": "priority",
    },
}

# Admin/Master Admin always have everything
ADMIN_ENTITLEMENTS = {
    "tier": "admin",
    "tier_display": "Administrator",
    "upload_quota": 999999,
    "unlimited_uploads": True,
    "max_accounts": 999,
    "max_accounts_per_platform": 999,
    "ai_captions_enabled": True,
    "ai_thumbnails_enabled": True,
    "ai_hashtags_enabled": True,
    "max_hashtags": 9999,
    "unlimited_hashtags": True,
    "show_watermark": False,
    "show_ads": False,
    "white_label_enabled": True,
    "can_burn_hud": True,
    "priority_processing": True,
    "can_schedule": True,
    "smart_scheduling_enabled": True,
    "analytics_enabled": True,
    "full_analytics_enabled": True,
    "excel_export_enabled": True,
    "team_seats": 999,
    "history_days": 9999,
    "can_use_templates": True,
    "can_use_webhooks": True,
    "support_tier": "sla",
}


def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Get entitlements object for a subscription tier."""
    tier_lower = (tier or "free").lower().strip()
    config = TIER_CONFIG.get(tier_lower, TIER_CONFIG["free"])
    return Entitlements(**config)


def get_entitlements_from_user(user: dict, custom_overrides: dict = None) -> Entitlements:
    """
    Get entitlements from user record with all overrides applied.
    
    Priority:
    1. Role-based (admin/master_admin get everything)
    2. Custom overrides from user_entitlements table
    3. Tier-based defaults
    """
    role = user.get("role", "user")
    
    # Admins get everything
    if role in ("admin", "master_admin"):
        ent = Entitlements(**ADMIN_ENTITLEMENTS)
        return ent
    
    # Get tier-based entitlements
    tier = user.get("subscription_tier", "free")
    ent = get_entitlements_for_tier(tier)
    
    # Apply custom quota if set
    custom_quota = user.get("upload_quota")
    if custom_quota and custom_quota > ent.upload_quota:
        ent.upload_quota = custom_quota
    
    # Track current usage
    ent.uploads_this_month = user.get("uploads_this_month", 0)
    
    # Apply any custom overrides (from user_entitlements table)
    if custom_overrides:
        for key, value in custom_overrides.items():
            if hasattr(ent, key) and value is not None:
                setattr(ent, key, value)
        ent.custom_overrides = custom_overrides
    
    return ent


def can_user_upload(user: dict, ent: Entitlements = None) -> tuple:
    """Check if user can upload. Returns (can_upload, message)."""
    if ent is None:
        ent = get_entitlements_from_user(user)
    
    if ent.unlimited_uploads:
        return True, "ok"
    
    used = user.get("uploads_this_month", 0)
    if used >= ent.upload_quota:
        return False, f"Monthly upload quota reached ({used}/{ent.upload_quota}). Upgrade to upload more."
    
    return True, "ok"


def can_user_connect_platform(user: dict, platform: str, current_total: int, current_for_platform: int, ent: Entitlements = None) -> tuple:
    """
    Check if user can connect another platform account.
    Returns (can_connect, message)
    """
    if ent is None:
        ent = get_entitlements_from_user(user)
    
    # Check total account limit
    if current_total >= ent.max_accounts:
        return False, f"Total account limit reached ({current_total}/{ent.max_accounts}). Upgrade to connect more."
    
    # Check per-platform limit
    if current_for_platform >= ent.max_accounts_per_platform:
        return False, f"Per-platform limit reached ({current_for_platform}/{ent.max_accounts_per_platform}). Upgrade to connect more."
    
    return True, "ok"


def get_tier_display_name(tier: str) -> str:
    """Get human-readable tier name."""
    config = TIER_CONFIG.get(tier.lower(), {})
    return config.get("tier_display", tier.title())


def get_tier_from_lookup_key(lookup_key: str) -> str:
    """Map Stripe lookup key to tier name."""
    mapping = {
        "uploadm8_free_monthly": "free",
        "uploadm8_starter_monthly": "starter",
        "uploadm8_solo_monthly": "solo",
        "uploadm8_creator_monthly": "creator",
        "uploadm8_growth_monthly": "growth",
        "uploadm8_studio_monthly": "studio",
        "uploadm8_agency_monthly": "agency",
        "uploadm8_lifetime": "lifetime",
        # Yearly variants
        "uploadm8_starter_yearly": "starter",
        "uploadm8_solo_yearly": "solo",
        "uploadm8_creator_yearly": "creator",
        "uploadm8_growth_yearly": "growth",
        "uploadm8_studio_yearly": "studio",
        "uploadm8_agency_yearly": "agency",
    }
    return mapping.get(lookup_key.strip().lower(), "free")


def entitlements_to_dict(ent: Entitlements) -> dict:
    """Convert Entitlements to dictionary for API responses."""
    return {
        "tier": ent.tier,
        "tier_display": ent.tier_display,
        "upload_quota": ent.upload_quota,
        "unlimited_uploads": ent.unlimited_uploads,
        "uploads_this_month": ent.uploads_this_month,
        "max_accounts": ent.max_accounts,
        "max_accounts_per_platform": ent.max_accounts_per_platform,
        "ai_captions_enabled": ent.ai_captions_enabled,
        "ai_thumbnails_enabled": ent.ai_thumbnails_enabled,
        "ai_hashtags_enabled": ent.ai_hashtags_enabled,
        "max_hashtags": ent.max_hashtags,
        "unlimited_hashtags": ent.unlimited_hashtags,
        "always_use_hashtags": ent.always_use_hashtags,
        "default_hashtag_count": ent.default_hashtag_count,
        "show_watermark": ent.show_watermark,
        "show_ads": ent.show_ads,
        "white_label_enabled": ent.white_label_enabled,
        "can_burn_hud": ent.can_burn_hud,
        "priority_processing": ent.priority_processing,
        "can_schedule": ent.can_schedule,
        "smart_scheduling_enabled": ent.smart_scheduling_enabled,
        "analytics_enabled": ent.analytics_enabled,
        "full_analytics_enabled": ent.full_analytics_enabled,
        "excel_export_enabled": ent.excel_export_enabled,
        "team_seats": ent.team_seats,
        "history_days": ent.history_days,
        "can_use_templates": ent.can_use_templates,
        "can_use_webhooks": ent.can_use_webhooks,
        "support_tier": ent.support_tier,
    }


# Feature check helpers for worker
def should_burn_watermark(ent: Entitlements) -> bool:
    """Check if watermark should be burned onto video."""
    return ent.show_watermark


def should_generate_captions(ent: Entitlements, has_caption: bool) -> bool:
    """Check if AI captions should be generated."""
    return ent.ai_captions_enabled and not has_caption


def should_generate_thumbnails(ent: Entitlements) -> bool:
    """Check if AI thumbnails should be generated."""
    return ent.ai_thumbnails_enabled


def should_generate_hashtags(ent: Entitlements, has_hashtags: bool, always_hashtags: bool = False) -> bool:
    """Check if AI hashtags should be generated."""
    if not ent.ai_hashtags_enabled:
        return False
    if always_hashtags or ent.always_use_hashtags:
        return True
    return not has_hashtags


def get_max_hashtags(ent: Entitlements) -> int:
    """Get maximum allowed hashtags for user."""
    if ent.unlimited_hashtags:
        return 9999
    return ent.max_hashtags

# =====================================================================
# APPEND-ONLY PATCH: errors contract normalization
# Fixes ImportError: cannot import name 'StageError' from 'stages.errors'
# Ensures symbols exist: StageError, SkipStage, CancelRequested, ErrorCode
# =====================================================================

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict

# ---- ErrorCode (define only if missing) ----
try:
    ErrorCode  # type: ignore[name-defined]
except NameError:
    class ErrorCode(str, Enum):
        INTERNAL = "INTERNAL"
        CANCELLED = "CANCELLED"
        SKIPPED = "SKIPPED"
        VALIDATION = "VALIDATION"
        NOT_FOUND = "NOT_FOUND"
        UNAUTHORIZED = "UNAUTHORIZED"
        RATE_LIMIT = "RATE_LIMIT"
        UPSTREAM = "UPSTREAM"

# ---- StageError (define only if missing) ----
try:
    StageError  # type: ignore[name-defined]
except NameError:
    @dataclass
    class StageError(Exception):
        """Canonical pipeline error type."""
        code: Any = ErrorCode.INTERNAL
        message: str = "Stage error"
        stage: Optional[str] = None
        meta: Optional[Dict[str, Any]] = None
        retryable: bool = False

        def __str__(self) -> str:
            base = f"{self.code}: {self.message}"
            if self.stage:
                base = f"[{self.stage}] {base}"
            return base

# ---- SkipStage (define only if missing) ----
try:
    SkipStage  # type: ignore[name-defined]
except NameError:
    class SkipStage(Exception):
        """Raise to intentionally skip a stage without failing the pipeline."""
        def __init__(self, message: str = "Stage skipped", meta: Optional[Dict[str, Any]] = None):
            super().__init__(message)
            self.meta = meta or {}

# ---- CancelRequested (define only if missing) ----
try:
    CancelRequested  # type: ignore[name-defined]
except NameError:
    class CancelRequested(Exception):
        """Raise to stop processing due to explicit cancellation."""
        def __init__(self, message: str = "Cancel requested"):
            super().__init__(message)
