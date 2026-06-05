"""
UploadM8 Entitlements System v2
================================
Single source of truth for ALL tier-based permissions.
API (via routers) and the worker pipeline read tier/cost data from here.

PUBLIC TIERS (marketing one-liners — see docs/features-and-benefits.md):
  free (Starter) $0       prove the four-platform loop; no card
  creator_lite  $12/mo ($120/yr)   watermark-free weekly shipping + webhooks
  creator_pro   $29/mo ($290/yr)   priority lane, team seats [MOST POPULAR]
  studio        $79/mo ($790/yr)   turbo throughput + export-grade reporting
  agency        $199/mo ($1990/yr) client groups, white-label, flex, dedicated lane

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
  Free = 24h: schedule uploads up to a day out.
  Agency = 168h: uploads are pre-processed up to a full week early.

QUEUE DEPTH:
  Max staged + pending + queued uploads per user at once.
  Enforced at presign. Free = 10, Agency = 99,999 (effectively unlimited).

MAX CAPTION FRAMES:
  FFmpeg screenshots the caption AI analyzes per video.
  Always anchored at beginning (5%), middle (50%), end (95%).
  Additional frames fill between anchors up to this limit.
  Free = 3 (anchors only). Studio/Agency = 15 (dense AI scan).

MAX THUMBNAILS:
  Number of candidate thumbnail frames generated per video.
  The sharpest frame is auto-selected; all candidates stored for user pick.
  Free = 1 (single frame). Agency/internal = 20 (dense scan, best quality).

MAX PARALLEL UPLOADS:
  Plan-based cap for parallel batch upload mode (upload.html).
  Sequential mode always uses 1. Parallel mode uses min(selected, this cap).
  Free = 1, Creator Lite = 2, Creator Pro = 3, Studio = 4, Agency = 5, internal = 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Dict, Any, Tuple


# ============================================================
# Tier Configuration - SINGLE SOURCE OF TRUTH
# ============================================================
# Keys: per_platform -> max_accounts_per_platform, caption_frames -> max_caption_frames,
# parallel_uploads -> max_parallel_uploads. put_daily/priority_class added for worker.
TIER_CONFIG: Dict[str, Dict[str, Any]] = {

    "free": {
        "name": "Starter", "price": 0,
        "put_daily": 5, "put_monthly": 100, "aic_monthly": 80,
        "max_accounts": 4, "max_accounts_per_platform": 1, "per_platform": 1,
        "watermark": True, "ads": True, "ai": True,
        "scheduling": True, "webhooks": False, "white_label": False,
        "hud": False, "excel": False, "flex": False,
        "priority_class": "p4", "queue_depth": 10, "lookahead_hours": 24,
        "max_thumbnails": 3, "ai_depth": "basic",
        "max_caption_frames": 3, "caption_frames": 3,
        "max_parallel_uploads": 1, "parallel_uploads": 1,
        "custom_thumbnails": True, "ai_thumbnail_styling": False,
        "team_seats": 1, "analytics": "basic", "trial_days": 0,
    },
    "creator_lite": {
        "name": "Creator Lite", "price": 12, "price_annual": 120,
        "put_daily": 20, "put_monthly": 600, "aic_monthly": 220,
        "max_accounts": 10, "max_accounts_per_platform": 3, "per_platform": 3,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": False, "excel": False, "flex": False,
        "priority_class": "p3", "queue_depth": 100, "lookahead_hours": 12,
        "max_thumbnails": 5, "ai_depth": "enhanced",
        "max_caption_frames": 5, "caption_frames": 5,
        "max_parallel_uploads": 2, "parallel_uploads": 2,
        "custom_thumbnails": True, "ai_thumbnail_styling": False,
        "team_seats": 1, "analytics": "standard", "trial_days": 7,
    },
    "creator_pro": {
        "name": "Creator Pro", "price": 29, "price_annual": 290,
        "put_daily": 60, "put_monthly": 2000, "aic_monthly": 600,
        "max_accounts": 25, "max_accounts_per_platform": 6, "per_platform": 6,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": False, "excel": False, "flex": False,
        "priority_class": "p2", "queue_depth": 500, "lookahead_hours": 24,
        "max_thumbnails": 8, "ai_depth": "advanced",
        "max_caption_frames": 8, "caption_frames": 8,
        "max_parallel_uploads": 3, "parallel_uploads": 3,
        "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 3, "analytics": "full", "trial_days": 7,
    },
    "studio": {
        "name": "Studio", "price": 79, "price_annual": 790,
        "put_daily": 150, "put_monthly": 6500, "aic_monthly": 2000,
        "max_accounts": 75, "max_accounts_per_platform": 20, "per_platform": 20,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": False, "excel": True, "flex": False,
        "priority_class": "p1", "queue_depth": 2500, "lookahead_hours": 72,
        "max_thumbnails": 12, "ai_depth": "max",
        "max_caption_frames": 15, "caption_frames": 15,
        "max_parallel_uploads": 4, "parallel_uploads": 4,
        "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 10, "analytics": "full_export", "trial_days": 7,
    },
    "agency": {
        "name": "Agency", "price": 199, "price_annual": 1990,
        "put_daily": 500, "put_monthly": 20000, "aic_monthly": 7000,
        "max_accounts": 300, "max_accounts_per_platform": 100, "per_platform": 100,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": False, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 99999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max",
        "max_caption_frames": 15, "caption_frames": 15,
        "max_parallel_uploads": 6, "parallel_uploads": 6,
        "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 25, "analytics": "full_export", "trial_days": 7,
    },
    # ── Internal tiers (not sold via Stripe) ──
    "master_admin": {
        "name": "Admin", "price": 0,
        "put_daily": 9999, "put_monthly": 999999, "aic_monthly": 999999,
        "max_accounts": 999, "max_accounts_per_platform": 999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": False, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 999999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max", "max_caption_frames": 20,
        "max_parallel_uploads": 6, "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 9999, "analytics": "full_export", "internal": True,
    },
    "friends_family": {
        "name": "Friends & Family", "price": 0,
        "put_daily": 999, "put_monthly": 12000, "aic_monthly": 5000,
        "max_accounts": 80, "max_accounts_per_platform": 80,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": False, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 99999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max", "max_caption_frames": 15,
        "max_parallel_uploads": 6, "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 999, "analytics": "full_export", "internal": True,
    },
    "lifetime": {
        "name": "Lifetime", "price": 0,
        "put_daily": 999, "put_monthly": 12000, "aic_monthly": 5000,
        "max_accounts": 80, "max_accounts_per_platform": 80,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": False, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 99999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max", "max_caption_frames": 15,
        "max_parallel_uploads": 6, "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 999, "analytics": "full_export", "internal": True,
    },
    # ── Legacy alias (keep until migration complete) ──
    "launch": {
        "name": "Creator Lite (Legacy)", "price": 12, "price_annual": 120,
        "put_daily": 20, "put_monthly": 600, "aic_monthly": 200,
        "max_accounts": 10, "max_accounts_per_platform": 3,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": False, "excel": False, "flex": False,
        "priority_class": "p3", "queue_depth": 100, "lookahead_hours": 12,
        "max_thumbnails": 5, "ai_depth": "enhanced", "max_caption_frames": 5,
        "max_parallel_uploads": 2, "custom_thumbnails": True, "ai_thumbnail_styling": False,
        "team_seats": 1, "analytics": "standard", "trial_days": 7, "internal": False,
    },
}

# Stripe price lookup_key -> internal tier slug
STRIPE_LOOKUP_TO_TIER: Dict[str, str] = {
    "uploadm8_creatorlite_monthly": "creator_lite",
    "uploadm8_creator_lite_monthly": "creator_lite",
    "uploadm8_creatorlite_yearly": "creator_lite",
    "uploadm8_creator_lite_yearly": "creator_lite",
    "uploadm8_creatorpro_monthly":  "creator_pro",
    "uploadm8_creatorpro_yearly":   "creator_pro",
    "uploadm8_studio_monthly":      "studio",
    "uploadm8_studio_yearly":       "studio",
    "uploadm8_agency_monthly":      "agency",
    "uploadm8_agency_yearly":       "agency",
    "uploadm8_launch_monthly":      "launch",
    "uploadm8_launch_yearly":       "launch",
    "uploadm8_creator_pro_monthly": "creator_pro",
    "uploadm8_creator_pro_yearly":  "creator_pro",
}

# ============================================================
# Topup / Add-on Products
# Maps Stripe price lookup_key -> {wallet, amount} or bundle {wallet, put, aic}
# ============================================================
TOPUP_PRODUCTS = {
    "uploadm8_put_250":   {"wallet": "put", "amount": 250,   "price": 4.99,  "price_usd": 4.99},
    "uploadm8_put_500":   {"wallet": "put", "amount": 500,   "price": 7.99,  "price_usd": 7.99},
    "uploadm8_put_1000":  {"wallet": "put", "amount": 1000,  "price": 14.99, "price_usd": 14.99},
    "uploadm8_put_2500":  {"wallet": "put", "amount": 2500,  "price": 29.99, "price_usd": 29.99},
    "uploadm8_put_5000":  {"wallet": "put", "amount": 5000,  "price": 49.99, "price_usd": 49.99},
    "uploadm8_aic_250":   {"wallet": "aic", "amount": 250,   "price": 4.99,  "price_usd": 4.99},
    "uploadm8_aic_500":   {"wallet": "aic", "amount": 500,   "price": 7.99,  "price_usd": 7.99},
    "uploadm8_aic_1000":  {"wallet": "aic", "amount": 1000,  "price": 14.99, "price_usd": 14.99},
    "uploadm8_aic_2500":  {"wallet": "aic", "amount": 2500,  "price": 29.99, "price_usd": 29.99},
    "uploadm8_aic_5000":  {"wallet": "aic", "amount": 5000,  "price": 49.99, "price_usd": 49.99},
    "uploadm8_aic_10000": {"wallet": "aic", "amount": 10000, "price": 79.99, "price_usd": 79.99},
    "uploadm8_boost_small":  {"wallet": "bundle", "put": 200,  "aic": 100,   "price": 7.99,  "price_usd": 7.99},
    "uploadm8_boost_medium": {"wallet": "bundle", "put": 1000, "aic": 500,   "price": 29.99, "price_usd": 29.99},
    "uploadm8_boost_large":  {"wallet": "bundle", "put": 4000, "aic": 2000,  "price": 99.99, "price_usd": 99.99},
}

# Priority class routing sets
PRIORITY_QUEUE_CLASSES = {"p0", "p1", "p2"}   # -> process:priority / publish:priority
NORMAL_QUEUE_CLASSES   = {"p3", "p4"}          # -> process:normal  / publish:normal

# ============================================================
# Canonical Tier & Entitlement Schema (shared with frontend)
# ============================================================
# All valid subscription_tier values. Frontend and backend MUST use these slugs.
TIER_SLUGS = ("free", "creator_lite", "creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin", "launch")

# launch is legacy alias for creator_lite
TIER_ALIASES = {"launch": "creator_lite"}


def normalize_tier(tier: str) -> str:
    """Return canonical tier slug. Maps launch->creator_lite, unknown->free."""
    t = (tier or "free").lower().strip()
    if t in TIER_ALIASES:
        return TIER_ALIASES[t]
    return t if t in TIER_CONFIG else "free"


PUBLIC_TIER_SLUGS = ("free", "creator_lite", "creator_pro", "studio", "agency")
_PUBLIC_UPGRADE_LADDER = PUBLIC_TIER_SLUGS


def analytics_display_label(analytics: str) -> str:
    """Human label for analytics tier (pricing UI, tier-catalog.js)."""
    return {
        "basic": "Basic analytics",
        "standard": "Standard analytics",
        "full": "Full analytics",
        "full_export": "Full analytics + export",
    }.get((analytics or "basic").lower(), "Basic analytics")


def queue_lane_display_label(priority_class: str, slug: str = "") -> str:
    """Queue lane label derived from priority_class (and slug for marketing names)."""
    pc = (priority_class or "p4").lower()
    s = (slug or "").lower()
    if s == "agency" or pc == "p0":
        return "Dedicated"
    if s == "studio" or pc == "p1":
        return "Turbo"
    if s == "creator_pro" or pc == "p2":
        return "Priority"
    return "Standard"


def scheduling_window_display_label(lookahead_hours: int) -> str:
    """Scheduling lookahead as a short human phrase."""
    h = int(lookahead_hours or 0)
    if h >= 168:
        return "7 days"
    if h >= 72:
        return "3 days"
    if h >= 24:
        return "24 hours"
    if h >= 12:
        return "12 hours"
    if h > 0:
        return f"{h} hours"
    return "—"


def tier_cfg_to_api_dict(slug: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize one merged tier config row for API responses (entitlements + pricing)."""
    pa = cfg.get("price_annual")
    priority_class = cfg.get("priority_class", "p4")
    analytics = cfg.get("analytics", "basic")
    lookahead = int(cfg.get("lookahead_hours", 2) or 0)
    return {
        "slug": slug,
        "name": cfg.get("name", slug.replace("_", " ").title()),
        "price": float(cfg.get("price", 0)),
        "price_annual": float(pa) if pa is not None else None,
        "put_daily": cfg.get("put_daily", 2),
        "put_monthly": cfg.get("put_monthly", 0),
        "aic_monthly": cfg.get("aic_monthly", 0),
        "internal": bool(cfg.get("internal", False)),
        "max_accounts": cfg.get("max_accounts", 4),
        "max_accounts_per_platform": cfg.get("max_accounts_per_platform", 1),
        "per_platform": cfg.get("per_platform", cfg.get("max_accounts_per_platform", 1)),
        "queue_depth": cfg.get("queue_depth", 25),
        "lookahead_hours": lookahead,
        "trial_days": cfg.get("trial_days", 0),
        "team_seats": cfg.get("team_seats", 1),
        "analytics": analytics,
        "analytics_label": analytics_display_label(analytics),
        "ai_depth": cfg.get("ai_depth", "basic"),
        "priority_class": priority_class,
        "can_priority": priority_class in PRIORITY_QUEUE_CLASSES,
        "queue_lane_label": queue_lane_display_label(priority_class, slug),
        "scheduling_window_label": scheduling_window_display_label(lookahead),
        "max_thumbnails": cfg.get("max_thumbnails", 1),
        "max_caption_frames": cfg.get("max_caption_frames", 3),
        "caption_frames": cfg.get("caption_frames", cfg.get("max_caption_frames", 3)),
        "max_parallel_uploads": cfg.get("max_parallel_uploads", 1),
        "parallel_uploads": cfg.get("parallel_uploads", cfg.get("max_parallel_uploads", 1)),
        "watermark": bool(cfg.get("watermark", True)),
        "ads": bool(cfg.get("ads", True)),
        "ai": bool(cfg.get("ai", False)),
        "scheduling": bool(cfg.get("scheduling", False)),
        "webhooks": bool(cfg.get("webhooks", False)),
        "white_label": bool(cfg.get("white_label", False)),
        "excel": bool(cfg.get("excel", False)),
        "flex": bool(cfg.get("flex", False)),
        "custom_thumbnails": bool(cfg.get("custom_thumbnails", False)),
        "ai_thumbnail_styling": bool(cfg.get("ai_thumbnail_styling", False)),
    }


def get_next_public_upgrade_tier(tier: str) -> Optional[str]:
    """Next tier on the public pricing ladder, or None if already at the top or internal-only."""
    t = normalize_tier(tier)
    if t not in _PUBLIC_UPGRADE_LADDER:
        return None
    i = _PUBLIC_UPGRADE_LADDER.index(t)
    if i + 1 >= len(_PUBLIC_UPGRADE_LADDER):
        return None
    return _PUBLIC_UPGRADE_LADDER[i + 1]


def get_tier_display_name(tier: str) -> str:
    """Return human-readable tier name from TIER_CONFIG."""
    t = normalize_tier(tier)
    tier_cfg = get_effective_tier_config()
    return tier_cfg.get(t, tier_cfg["free"]).get("name", t.replace("_", " ").title())


# Entitlement keys returned by entitlements_to_dict — frontend uses these exact keys
ENTITLEMENT_KEYS = (
    "tier", "tier_display", "put_daily", "put_monthly", "aic_monthly",
    "max_accounts", "max_accounts_per_platform", "can_watermark", "can_ai",
    "can_schedule", "can_webhooks", "can_white_label", "can_excel", "can_priority",
    "can_flex", "show_ads", "priority_class", "queue_depth",
    "lookahead_hours", "max_caption_frames", "ai_depth", "max_thumbnails",
    "can_custom_thumbnails", "can_ai_thumbnail_styling", "max_parallel_uploads",
    "team_seats", "analytics", "trial_days", "is_internal", "allowed_ai_services",
)


def get_effective_tier_config() -> Dict[str, Dict[str, Any]]:
    """``TIER_CONFIG`` merged with master-admin DB overrides (``billing_catalog``)."""
    try:
        from services.billing_catalog import effective_tier_config

        return effective_tier_config()
    except Exception:
        return TIER_CONFIG


def get_effective_topup_products() -> Dict[str, Dict[str, Any]]:
    """``TOPUP_PRODUCTS`` merged with DB overrides."""
    try:
        from services.billing_catalog import effective_topup_products

        return effective_topup_products()
    except Exception:
        return TOPUP_PRODUCTS


def get_tiers_for_api() -> list:
    """Return tier metadata for /api/entitlements/tiers. Single source for frontend.
    Includes revenue tiers + internal (friends_family, lifetime, master_admin) + launch alias.
    Field names align with TIER_CONFIG and with ``frontend/js/tier-catalog.js`` ``mergeRow``."""
    all_slugs = (*PUBLIC_TIER_SLUGS, "friends_family", "lifetime", "master_admin", "launch")
    tier_cfg = get_effective_tier_config()
    return [tier_cfg_to_api_dict(slug, tier_cfg.get(slug, {})) for slug in all_slugs]


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
    can_custom_thumbnails: bool = False   # YouTube/FB/IG styled thumbnails allowed
    can_ai_thumbnail_styling: bool = False  # AI image edit vs template-only

    # Batch upload (upload.html parallel mode cap)
    max_parallel_uploads: int = 1

    # Team / reporting
    team_seats: int = 1
    analytics: str = "basic"

    # Trial
    trial_days: int = 0

    # Internal flag
    is_internal: bool = False

    # Pipeline services allowed for this tier (Whisper, music detection, etc.)
    # None = skip tier service filter (legacy/tests); empty frozenset = no services allowed.
    allowed_ai_services: Optional[FrozenSet[str]] = None

    # Per-user override audit trail
    custom_overrides: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Builder Functions
# ============================================================

def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Build Entitlements from a tier slug. Uses normalize_tier (launch->creator_lite, unknown->free)."""
    t = normalize_tier(tier)
    tier_cfg = get_effective_tier_config()
    cfg = tier_cfg.get(t, tier_cfg["free"])
    try:
        from services.billing_catalog import effective_tier_allowed_services

        allowed_services = effective_tier_allowed_services(t)
    except Exception:
        from stages.ai_service_costs import SERVICE_WEIGHTS

        allowed_services = frozenset(SERVICE_WEIGHTS.keys()) if cfg.get("ai") else frozenset()
    return Entitlements(
        tier=t,
        tier_display=cfg.get("name", t.replace("_", " ").title()),
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
        can_priority=cfg.get("priority_class", "p4") in PRIORITY_QUEUE_CLASSES,
        can_flex=cfg.get("flex", False),
        show_ads=cfg.get("ads", True),
        priority_class=cfg.get("priority_class", "p4"),
        queue_depth=cfg.get("queue_depth", 25),
        lookahead_hours=cfg.get("lookahead_hours", 2),
        max_caption_frames=cfg.get("max_caption_frames", 3),
        ai_depth=cfg.get("ai_depth", "basic"),
        max_thumbnails=cfg.get("max_thumbnails", 1),
        can_custom_thumbnails=cfg.get("custom_thumbnails", False),
        can_ai_thumbnail_styling=cfg.get("ai_thumbnail_styling", False),
        max_parallel_uploads=cfg.get("max_parallel_uploads", 1),
        team_seats=cfg.get("team_seats", 1),
        analytics=cfg.get("analytics", "basic"),
        trial_days=cfg.get("trial_days", 0),
        is_internal=cfg.get("internal", False),
        allowed_ai_services=allowed_services,
    )


def get_entitlements_from_user(
    user_record: dict,
    overrides: Optional[dict] = None,
) -> Entitlements:
    """
    Build Entitlements from a users table row + optional per-user admin overrides.

    Priority order (highest wins):
      1. Role override: master_admin role -> full internal master_admin entitlements
      2. Per-user overrides from entitlement_overrides table
      3. Tier defaults from TIER_CONFIG
    """
    # Role override — only master_admin (staff "admin" keeps subscription_tier quotas).
    role = str(user_record.get("role") or "user").strip().lower()
    if role == "master_admin":
        return get_entitlements_for_tier("master_admin")

    raw_tier = str(user_record.get("subscription_tier") or "free").strip().lower()
    tier = normalize_tier(raw_tier)
    ent = get_entitlements_for_tier(tier)

    if overrides:
        _ov(ent, overrides, "can_ai",                    bool)
        _ov(ent, overrides, "can_schedule",              bool)
        _ov(ent, overrides, "can_priority",              bool)
        _ov(ent, overrides, "can_flex",                  bool)
        _ov(ent, overrides, "can_watermark",             bool)
        _ov(ent, overrides, "can_webhooks",              bool)
        _ov(ent, overrides, "can_white_label",           bool)
        _ov(ent, overrides, "max_accounts",              int)
        _ov(ent, overrides, "max_accounts_per_platform", int)
        _ov(ent, overrides, "max_parallel_uploads",       int)
        _ov(ent, overrides, "queue_depth",               int)
        _ov(ent, overrides, "lookahead_hours",           int)
        _ov(ent, overrides, "max_caption_frames",        int)
        _ov(ent, overrides, "max_thumbnails",            int)
        _ov(ent, overrides, "can_custom_thumbnails",     bool)
        _ov(ent, overrides, "can_ai_thumbnail_styling", bool)
        _ov(ent, overrides, "priority_class",            str)
        ent.custom_overrides = dict(overrides)

    # Legacy per-user flex toggle on the users table row
    if user_record.get("flex_enabled"):
        ent.can_flex = True

    return ent


def wallet_bypass_for_user_record(user_record: dict | None) -> bool:
    """
    True when PUT/AIC reservations, debits, and refunds should not touch wallet balances.

    Staff roles (admin, master_admin) are exempt regardless of subscription_tier so
    support accounts are not blocked by test-wallet balances. Internal tiers
    (friends_family, lifetime, master_admin slug) remain exempt via ``is_internal``.
    """
    if not user_record:
        return False
    role = str(user_record.get("role") or "user").strip().lower()
    if role in ("admin", "master_admin"):
        return True
    ent = get_entitlements_from_user(user_record)
    return bool(getattr(ent, "is_internal", False))


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
        "show_ads":                  ent.show_ads,
        "priority_class":            ent.priority_class,
        "queue_depth":               ent.queue_depth,
        "lookahead_hours":           ent.lookahead_hours,
        "max_caption_frames":        ent.max_caption_frames,
        "ai_depth":                  ent.ai_depth,
        "max_thumbnails":            ent.max_thumbnails,
        "can_custom_thumbnails":     ent.can_custom_thumbnails,
        "can_ai_thumbnail_styling":  ent.can_ai_thumbnail_styling,
        "max_parallel_uploads":      ent.max_parallel_uploads,
        "team_seats":                ent.team_seats,
        "analytics":                 ent.analytics,
        "trial_days":                ent.trial_days,
        "is_internal":               ent.is_internal,
        "allowed_ai_services":       sorted(ent.allowed_ai_services) if ent.allowed_ai_services is not None else None,
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


def get_tier_from_lookup_key(lookup_key: str) -> str:
    """Convert a Stripe price lookup_key to an internal tier slug."""
    return STRIPE_LOOKUP_TO_TIER.get((lookup_key or "").strip().lower(), "free")


# ============================================================
# PUT/AIC Cost Formula — canonical, imported by API services/routers + worker.py
# ============================================================

# AIC base by ai_depth string
_AIC_DEPTH_BASE: Dict[str, int] = {
    "none":     0,
    "basic":    2,
    "enhanced": 3,
    "advanced": 4,
    "max":      6,
}


def compute_put_cost(
    num_platforms: int,
    is_priority: bool = False,
    num_thumbnails: int = 1,
    *,
    rules: Optional[Dict[str, int]] = None,
) -> int:
    """
    Deterministic PUT cost per upload job.
      base          = rules['base'] (default 10)
      + priority    if priority lane
      + per_extra_platform × extra platforms
      + per_extra_thumbnail × extra thumbnails
    """
    try:
        from services.billing_catalog import effective_put_cost_rules

        r = rules if rules is not None else effective_put_cost_rules()
    except Exception:
        from services.billing_catalog import PUT_COST_DEFAULTS

        r = rules if rules is not None else PUT_COST_DEFAULTS
    cost = int(r.get("base", 10))
    if is_priority:
        cost += int(r.get("priority_lane_addon", 5))
    cost += max(0, num_platforms - 1) * int(r.get("per_extra_platform", 2))
    cost += max(0, num_thumbnails - 1) * int(r.get("per_extra_thumbnail_beyond_first", 1))
    return cost


def compute_put_breakdown(
    num_platforms: int,
    is_priority: bool = False,
    num_thumbnails: int = 1,
    *,
    rules: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Line items that sum to compute_put_cost(...) — used for billing_breakdown / ledger meta.
    """
    try:
        from services.billing_catalog import effective_put_cost_rules

        r = rules if rules is not None else effective_put_cost_rules()
    except Exception:
        from services.billing_catalog import PUT_COST_DEFAULTS

        r = rules if rules is not None else PUT_COST_DEFAULTS
    lines = {
        "base": int(r.get("base", 10)),
        "priority_lane_addon": int(r.get("priority_lane_addon", 5)) if is_priority else 0,
        "extra_publish_targets": max(0, num_platforms - 1) * int(r.get("per_extra_platform", 2)),
        "extra_thumbnails_beyond_first": max(0, num_thumbnails - 1)
        * int(r.get("per_extra_thumbnail_beyond_first", 1)),
    }
    total = int(sum(lines.values()))
    return {"total": total, "lines": lines}


def compute_aic_cost(ai_depth: str, caption_frames: int) -> int:
    """
    Deterministic AIC cost per upload job.
      base by ai_depth: none=0, basic=2, enhanced=3, advanced=4, max=6
      +0 frames<=6, +1 frames 7-12, +2 frames 13-24, +3 frames 25-48
    """
    base = _AIC_DEPTH_BASE.get(ai_depth, 0)
    if base == 0:
        return 0
    if caption_frames <= 6:
        surcharge = 0
    elif caption_frames <= 12:
        surcharge = 1
    elif caption_frames <= 24:
        surcharge = 2
    else:
        surcharge = 3
    return base + surcharge


def compute_upload_cost(
    entitlements: "Entitlements",
    num_platforms: int,
    use_ai: bool = False,
    num_thumbnails: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Convenience wrapper: given entitlements + job params, return (put_cost, aic_cost).
    Applies entitlement caps before calculating — worker can call this to re-validate.
    """
    thumbs = min(
        num_thumbnails if num_thumbnails is not None else 1,
        entitlements.max_thumbnails,
    )
    is_priority = entitlements.priority_class in PRIORITY_QUEUE_CLASSES
    put = compute_put_cost(
        num_platforms=num_platforms,
        is_priority=is_priority,
        num_thumbnails=thumbs,
    )
    aic = 0
    if use_ai and entitlements.can_ai:
        aic = compute_aic_cost(
            ai_depth=entitlements.ai_depth,
            caption_frames=entitlements.max_caption_frames,
        )
    return put, aic


def ledger_pricing_reference(service_weights: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Serializable PUT/AIC rules and top-up SKUs for wallet UI (GET /api/wallet/ledger).
    When ``service_weights`` is provided (merged DB + code defaults), includes live pipeline catalog.
    """
    from stages.ai_service_costs import merge_service_weights_from_db, service_catalog

    topups: Dict[str, Any] = {}
    for key, prod in get_effective_topup_products().items():
        entry: Dict[str, Any] = {}
        for k in ("wallet", "amount", "put", "aic", "price_usd", "price"):
            if k in prod:
                entry[k] = prod[k]
        topups[key] = entry
    sw = merge_service_weights_from_db(service_weights)
    try:
        from services.billing_catalog import effective_put_cost_rules

        put_rules = effective_put_cost_rules()
    except Exception:
        from services.billing_catalog import PUT_COST_DEFAULTS

        put_rules = PUT_COST_DEFAULTS
    return {
        "topup_products": topups,
        "put_cost_rules": put_rules,
        "aic_cost_rules": {
            "model": "per_pipeline_service_weights",
            "note": "Upload AIC is the sum of enabled pipeline services × duration scaling (where applicable) + caption frame surcharge; see aic_pipeline_catalog.",
            "legacy_depth_formula": {
                "base_by_ai_depth": dict(_AIC_DEPTH_BASE),
                "frame_surcharge": "0 for <=6 frames, +1 for 7-12, +2 for 13-24, +3 for 25+",
            },
        },
        "aic_pipeline_catalog": service_catalog(sw),
    }
