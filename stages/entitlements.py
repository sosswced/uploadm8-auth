"""
UploadM8 Entitlements System v2
================================
Single source of truth for ALL tier-based permissions.
Both app.py (PLAN_CONFIG) and the worker pipeline read from here.

PUBLIC TIERS:
  free          $0       validate the workflow
  creator_lite  $9.99    ship consistently, stop manual posting
  creator_pro   $19.99   weekend batching + higher AI precision [MOST POPULAR]
  studio        $49.99   turbo throughput + export-grade reporting
  agency        $99.99   built for agencies managing multiple clients

INTERNAL TIERS (not on pricing page):
  friends_family  full agency+ access, p0 priority, no limits
  lifetime        full agency+ access, p0 priority, no limits
  master_admin    everything, p0 priority, no limits

BACKWARD COMPAT:
  launch -> aliased to creator_lite values (no DB migration required at deploy)
  Run the SQL migration script when ready to clean up the database.

PRIORITY CLASS -> REDIS QUEUE ROUTING:
  p0  agency, friends_family, lifetime, master_admin  -> process:priority
  p1  studio                                          -> process:priority
  p2  creator_pro                                     -> process:priority
  p3  creator_lite / launch                           -> process:normal
  p4  free                                            -> process:normal

  BRPOP order: [process:priority, process:normal]
  Priority queue always drains before normal queue touches a worker slot.
  Agency uploads literally jump every free-tier upload in the system.

LOOKAHEAD HOURS:
  How far ahead the scheduler starts pre-processing a staged upload.
  Free = 2h: can only schedule uploads up to 2h out.
  Agency = 168h: uploads are pre-processed up to a full week early.

QUEUE DEPTH:
  Max staged + pending + queued uploads per user at once.
  Enforced at presign. Free = 25, Agency = unlimited (999999).

MAX CAPTION FRAMES:
  FFmpeg screenshots the caption AI analyzes per video.
  Always anchored at beginning (5%), middle (50%), end (95%).
  Additional frames fill between anchors up to this limit.
  Free = 3 (anchors only). Studio/Agency = 15 (dense AI scan).

MAX THUMBNAILS:
  Number of candidate thumbnail frames generated per video.
  The sharpest frame is auto-selected; all candidates stored for user pick.
  Free = 1 (single frame). Agency/internal = 20 (dense scan, best quality).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


# ============================================================
# Tier Configuration - SINGLE SOURCE OF TRUTH
# ============================================================
TIER_CONFIG: Dict[str, Dict[str, Any]] = {

    "free": {
        "name": "Free", "price": 0,
        "put_daily": 2, "put_monthly": 60, "aic_monthly": 0,
        "max_accounts": 4, "max_accounts_per_platform": 1,
        "watermark": True, "ads": True, "ai": False,
        "scheduling": False, "webhooks": False, "white_label": False,
        "excel": False, "hud": False, "flex": False,
        "priority_class": "p4", "queue_depth": 25, "lookahead_hours": 2,
        "max_caption_frames": 3, "ai_depth": "basic",
        "team_seats": 1, "analytics": "basic",
        "max_thumbnails": 1,
    },

    "creator_lite": {
        "name": "Creator Lite", "price": 9.99,
        "put_daily": 10, "put_monthly": 300, "aic_monthly": 100,
        "max_accounts": 8, "max_accounts_per_platform": 2,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "excel": False, "hud": False, "flex": False,
        "priority_class": "p3", "queue_depth": 500, "lookahead_hours": 12,
        "max_caption_frames": 5, "ai_depth": "enhanced",
        "team_seats": 1, "analytics": "standard",
        "max_thumbnails": 3,
    },

    "creator_pro": {
        "name": "Creator Pro", "price": 19.99,
        "put_daily": 20, "put_monthly": 900, "aic_monthly": 200,
        "max_accounts": 20, "max_accounts_per_platform": 5,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "excel": False, "hud": True, "flex": False,
        "priority_class": "p2", "queue_depth": 5000, "lookahead_hours": 24,
        "max_caption_frames": 8, "ai_depth": "advanced",
        "team_seats": 3, "analytics": "full",
        "max_thumbnails": 5,
    },

    "studio": {
        "name": "Studio", "price": 49.99,
        "put_daily": 50, "put_monthly": 3000, "aic_monthly": 1000,
        "max_accounts": 60, "max_accounts_per_platform": 15,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "excel": True, "hud": True, "flex": False,
        "priority_class": "p1", "queue_depth": 25000, "lookahead_hours": 48,
        "max_caption_frames": 15, "ai_depth": "max",
        "team_seats": 10, "analytics": "full_export",
        "max_thumbnails": 10,
    },

    "agency": {
        "name": "Agency", "price": 99.99,
        "put_daily": 200, "put_monthly": 10000, "aic_monthly": 3000,
        "max_accounts": 999, "max_accounts_per_platform": 999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "hud": True, "flex": True,
        "priority_class": "p0", "queue_depth": 999999, "lookahead_hours": 168,
        "max_caption_frames": 15, "ai_depth": "max",
        "team_seats": 25, "analytics": "full_export",
        "max_thumbnails": 15,
    },

    "friends_family": {
        "name": "Friends & Family", "price": 0,
        "put_daily": 999, "put_monthly": 999999, "aic_monthly": 999999,
        "max_accounts": 999, "max_accounts_per_platform": 999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "hud": True, "flex": True, "internal": True,
        "priority_class": "p0", "queue_depth": 999999, "lookahead_hours": 168,
        "max_caption_frames": 20, "ai_depth": "max",
        "team_seats": 999, "analytics": "full_export",
        "max_thumbnails": 20,
    },

    "lifetime": {
        "name": "Lifetime", "price": 0,
        "put_daily": 999, "put_monthly": 999999, "aic_monthly": 999999,
        "max_accounts": 999, "max_accounts_per_platform": 999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "hud": True, "flex": True, "internal": True,
        "priority_class": "p0", "queue_depth": 999999, "lookahead_hours": 168,
        "max_caption_frames": 20, "ai_depth": "max",
        "team_seats": 999, "analytics": "full_export",
        "max_thumbnails": 20,
    },

    "master_admin": {
        "name": "Administrator", "price": 0,
        "put_daily": 9999, "put_monthly": 999999, "aic_monthly": 999999,
        "max_accounts": 9999, "max_accounts_per_platform": 9999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "excel": True, "hud": True, "flex": True, "internal": True,
        "priority_class": "p0", "queue_depth": 999999, "lookahead_hours": 168,
        "max_caption_frames": 20, "ai_depth": "max",
        "team_seats": 9999, "analytics": "full_export",
        "max_thumbnails": 20,
    },
}

# Backward compat alias - 'launch' users resolve to creator_lite values.
# No DB migration needed until you run migration_tier_rename.sql.
TIER_CONFIG["launch"] = TIER_CONFIG["creator_lite"]

# Stripe price lookup_key -> internal tier slug
STRIPE_LOOKUP_TO_TIER: Dict[str, str] = {
    # New slugs
    "uploadm8_creatorlite_monthly":  "creator_lite",
    "uploadm8_creatorpro_monthly":   "creator_pro",
    "uploadm8_studio_monthly":       "studio",
    "uploadm8_agency_monthly":       "agency",
    # Legacy slugs - keep until Stripe products are renamed
    "uploadm8_launch_monthly":       "creator_lite",
    "uploadm8_creator_pro_monthly":  "creator_pro",
}

# ============================================================
# Topup / Add-on Products
# Maps Stripe price lookup_key -> {wallet, amount} metadata
# ============================================================
TOPUP_PRODUCTS = {
    "uploadm8_put_50":    {"wallet": "put",  "amount": 50},
    "uploadm8_put_100":   {"wallet": "put",  "amount": 100},
    "uploadm8_put_250":   {"wallet": "put",  "amount": 250},
    "uploadm8_put_500":   {"wallet": "put",  "amount": 500},
    "uploadm8_put_1000":  {"wallet": "put",  "amount": 1000},
    "uploadm8_aic_100":   {"wallet": "aic",  "amount": 100},
    "uploadm8_aic_250":   {"wallet": "aic",  "amount": 250},
    "uploadm8_aic_500":   {"wallet": "aic",  "amount": 500},
    "uploadm8_aic_1000":  {"wallet": "aic",  "amount": 1000},
    "uploadm8_aic_2500":  {"wallet": "aic",  "amount": 2500},
}

# Priority class routing sets
PRIORITY_QUEUE_CLASSES = {"p0", "p1", "p2"}   # -> process:priority / publish:priority
NORMAL_QUEUE_CLASSES   = {"p3", "p4"}          # -> process:normal  / publish:normal


# ============================================================
# Entitlements Dataclass
# ============================================================
@dataclass
class Entitlements:
    """Fully resolved entitlements for a user."""

    tier: str = "free"
    tier_display: str = "Free"

    # Wallet
    put_daily: int = 2
    put_monthly: int = 60
    aic_monthly: int = 0

    # Accounts
    max_accounts: int = 4
    max_accounts_per_platform: int = 1

    # Features
    can_watermark: bool = True          # True = watermark IS burned onto video
    can_ai: bool = False
    can_schedule: bool = False
    can_webhooks: bool = False
    can_white_label: bool = False
    can_excel: bool = False
    can_priority: bool = False
    can_flex: bool = False
    can_burn_hud: bool = False
    show_ads: bool = True

    # Queue / scheduler
    priority_class: str = "p4"         # p0 (highest) -> p4 (lowest)
    queue_depth: int = 25
    lookahead_hours: int = 2

    # AI
    max_caption_frames: int = 3
    ai_depth: str = "basic"

    # Thumbnails
    max_thumbnails: int = 1

    # Team / reporting
    team_seats: int = 1
    analytics: str = "basic"

    # Internal flag
    is_internal: bool = False

    # Per-user override audit trail
    custom_overrides: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Builder Functions
# ============================================================

def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Build Entitlements from a tier slug. Falls back to free on unknown slugs."""
    t = (tier or "free").lower().strip()
    cfg = TIER_CONFIG.get(t, TIER_CONFIG["free"])
    return Entitlements(
        tier=t,
        tier_display=cfg.get("name", tier.title()),
        put_daily=cfg.get("put_daily", 2),
        put_monthly=cfg.get("put_monthly", 60),
        aic_monthly=cfg.get("aic_monthly", 0),
        max_accounts=cfg.get("max_accounts", 4),
        max_accounts_per_platform=cfg.get("max_accounts_per_platform", 1),
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
        priority_class=cfg.get("priority_class", "p4"),
        queue_depth=cfg.get("queue_depth", 25),
        lookahead_hours=cfg.get("lookahead_hours", 2),
        max_caption_frames=cfg.get("max_caption_frames", 3),
        ai_depth=cfg.get("ai_depth", "basic"),
        max_thumbnails=cfg.get("max_thumbnails", 1),
        team_seats=cfg.get("team_seats", 1),
        analytics=cfg.get("analytics", "basic"),
        is_internal=cfg.get("internal", False),
    )


def get_entitlements_from_user(
    user_record: dict,
    overrides: Optional[dict] = None,
) -> Entitlements:
    """
    Build Entitlements from a users table row + optional per-user admin overrides.

    Priority order (highest wins):
      1. Role override: master_admin / admin role -> full admin entitlements
      2. Per-user overrides from entitlement_overrides table
      3. Tier defaults from TIER_CONFIG
    """
    # Role override - admin accounts always get full access regardless of subscription_tier
    role = (user_record.get("role") or "user").lower()
    if role in ("admin", "master_admin"):
        return get_entitlements_for_tier("master_admin")

    tier = (user_record.get("subscription_tier") or "free").lower()
    ent = get_entitlements_for_tier(tier)

    if overrides:
        _ov(ent, overrides, "can_ai",                    bool)
        _ov(ent, overrides, "can_schedule",              bool)
        _ov(ent, overrides, "can_burn_hud",              bool)
        _ov(ent, overrides, "can_priority",              bool)
        _ov(ent, overrides, "can_flex",                  bool)
        _ov(ent, overrides, "can_watermark",             bool)
        _ov(ent, overrides, "can_webhooks",              bool)
        _ov(ent, overrides, "can_white_label",           bool)
        _ov(ent, overrides, "max_accounts",              int)
        _ov(ent, overrides, "max_accounts_per_platform", int)
        _ov(ent, overrides, "queue_depth",               int)
        _ov(ent, overrides, "lookahead_hours",           int)
        _ov(ent, overrides, "max_caption_frames",        int)
        _ov(ent, overrides, "max_thumbnails",            int)
        _ov(ent, overrides, "priority_class",            str)
        ent.custom_overrides = dict(overrides)

    # Legacy per-user flex toggle on the users table row
    if user_record.get("flex_enabled"):
        ent.can_flex = True

    return ent


def _ov(ent: Entitlements, overrides: dict, key: str, cast) -> None:
    """Apply one override key to an Entitlements object if present and non-null."""
    val = overrides.get(key)
    if val is not None:
        try:
            setattr(ent, key, cast(val))
        except (ValueError, TypeError):
            pass


def entitlements_to_dict(ent: Entitlements) -> dict:
    """Serialize Entitlements to JSON-safe dict for API responses (/api/me, etc.)."""
    return {
        "tier":                      ent.tier,
        "tier_display":              ent.tier_display,
        "put_daily":                 ent.put_daily,
        "put_monthly":               ent.put_monthly,
        "aic_monthly":               ent.aic_monthly,
        "max_accounts":              ent.max_accounts,
        "max_accounts_per_platform": ent.max_accounts_per_platform,
        "can_watermark":             ent.can_watermark,
        "can_ai":                    ent.can_ai,
        "can_schedule":              ent.can_schedule,
        "can_webhooks":              ent.can_webhooks,
        "can_white_label":           ent.can_white_label,
        "can_excel":                 ent.can_excel,
        "can_priority":              ent.can_priority,
        "can_flex":                  ent.can_flex,
        "can_burn_hud":              ent.can_burn_hud,
        "show_ads":                  ent.show_ads,
        "priority_class":            ent.priority_class,
        "queue_depth":               ent.queue_depth,
        "lookahead_hours":           ent.lookahead_hours,
        "max_caption_frames":        ent.max_caption_frames,
        "ai_depth":                  ent.ai_depth,
        "max_thumbnails":            ent.max_thumbnails,
        "team_seats":                ent.team_seats,
        "analytics":                 ent.analytics,
        "is_internal":               ent.is_internal,
    }


# ============================================================
# Guard Helpers
# ============================================================

def can_user_upload(user_record: dict, overrides: Optional[dict] = None) -> bool:
    """Return True if user's tier allows any uploads."""
    ent = get_entitlements_from_user(user_record, overrides)
    return ent.put_daily > 0


def can_user_connect_platform(
    user_record: dict,
    current_total: int,
    current_for_platform: int = 0,
    overrides: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    Check if user can connect another platform account.

    Returns:
        (allowed: bool, reason: str)
    """
    ent = get_entitlements_from_user(user_record, overrides)
    if current_total >= ent.max_accounts:
        return False, (
            f"Total account limit reached ({current_total}/{ent.max_accounts}). "
            "Upgrade to connect more accounts."
        )
    if current_for_platform >= ent.max_accounts_per_platform:
        return False, (
            f"Per-platform limit reached ({current_for_platform}/"
            f"{ent.max_accounts_per_platform}). Upgrade to connect more."
        )
    return True, "ok"


def check_queue_depth(
    user_record: dict,
    current_pending: int,
    overrides: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    Check if user is within their queue depth limit.
    Call at presign time with count of staged + queued + pending uploads.

    Returns:
        (allowed: bool, reason: str)
    """
    ent = get_entitlements_from_user(user_record, overrides)
    if current_pending >= ent.queue_depth:
        return False, (
            f"Queue limit reached ({current_pending}/{ent.queue_depth} uploads in queue). "
            "Wait for existing uploads to process or upgrade your plan."
        )
    return True, "ok"


def get_priority_lane(priority_class: str) -> str:
    """Return 'priority' or 'normal' for a given priority_class string."""
    return "priority" if priority_class in PRIORITY_QUEUE_CLASSES else "normal"


def get_tier_display_name(tier: str) -> str:
    """Get human-readable display name for a tier slug."""
    cfg = TIER_CONFIG.get((tier or "free").lower(), {})
    return cfg.get("name", tier.title())


def get_tier_from_lookup_key(lookup_key: str) -> str:
    """Convert a Stripe price lookup_key to an internal tier slug."""
    return STRIPE_LOOKUP_TO_TIER.get((lookup_key or "").strip().lower(), "free")
