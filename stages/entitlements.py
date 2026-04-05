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

PRIORITY CLASS -> REDIS QUEUE ROUTING:
  p0  agency, friends_family, lifetime, master_admin  -> process:priority
  p1  studio                                          -> process:priority
  p2  creator_pro                                     -> process:priority
  p3  creator_lite                                    -> process:normal
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

MAX PARALLEL UPLOADS:
  Plan-based cap for parallel batch upload mode (upload.html).
  Sequential mode always uses 1. Parallel mode uses min(selected, this cap).
  Free = 1, Creator Lite = 2, Creator Pro = 3, Studio = 4, Agency = 5, internal = 6.

AUDIO CONTEXT (Whisper Transcription):
  Runs as Stage 5.5 — after transcode, before thumbnail.
  Extracts audio via FFmpeg → sends to OpenAI Whisper → stores transcript in
  ctx.ai_transcript, which caption_stage injects into every GPT-4o prompt.
  Whisper may be disabled per user via `audio_transcription` / `audioTranscription` while
  keeping the rest of Stage 5.5 (YAMNet, ACRCloud, Hume, GPT classification).
  Users may opt out of the entire stage via `use_audio_context = False` in user_settings.
  Enforcement is in audio_stage.py (run_audio_context_stage); not gated on plan `can_ai`.
  Cost: ~$0.006/min via whisper-1 (platform cost, not billed as AIC tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

from stages.ai_service_costs import (
    billing_env_from_os,
    compute_aic_breakdown,
    effective_num_thumbnails,
)


# ============================================================
# Tier Configuration - SINGLE SOURCE OF TRUTH
# ============================================================
# Keys: per_platform -> max_accounts_per_platform, caption_frames -> max_caption_frames,
# parallel_uploads -> max_parallel_uploads. put_daily/priority_class added for worker.
TIER_CONFIG: Dict[str, Dict[str, Any]] = {

    "free": {
        "name": "Free", "price": 0,
        "put_daily": 4, "put_monthly": 120, "aic_monthly": 1200,
        "max_accounts": 16, "max_accounts_per_platform": 4, "per_platform": 4,
        "watermark": True, "ads": True, "ai": True,
        "scheduling": True, "webhooks": False, "white_label": False,
        "hud": False, "excel": False, "flex": False,
        "priority_class": "p4", "queue_depth": 10, "lookahead_hours": 4,
        "max_thumbnails": 3, "ai_depth": "basic",
        "max_caption_frames": 3, "caption_frames": 3,
        "max_parallel_uploads": 1, "parallel_uploads": 1,
        "custom_thumbnails": True, "ai_thumbnail_styling": False,
        "team_seats": 1, "analytics": "basic", "trial_days": 0,
        "audio_context": True,
    },
    "creator_lite": {
        "name": "Creator Lite", "price": 9.99,
        "put_daily": 17, "put_monthly": 500, "aic_monthly": 4500,
        "max_accounts": 40, "max_accounts_per_platform": 10, "per_platform": 10,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": False, "excel": False, "flex": False,
        "priority_class": "p3", "queue_depth": 100, "lookahead_hours": 12,
        "max_thumbnails": 5, "ai_depth": "enhanced",
        "max_caption_frames": 5, "caption_frames": 5,
        "max_parallel_uploads": 2, "parallel_uploads": 2,
        "custom_thumbnails": True, "ai_thumbnail_styling": False,
        "team_seats": 1, "analytics": "standard", "trial_days": 7,
        "audio_context": True,
    },
    "creator_pro": {
        "name": "Creator Pro", "price": 19.99,
        "put_daily": 60, "put_monthly": 1800, "aic_monthly": 13000,
        "max_accounts": 100, "max_accounts_per_platform": 25, "per_platform": 25,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": True, "excel": False, "flex": False,
        "priority_class": "p2", "queue_depth": 500, "lookahead_hours": 24,
        "max_thumbnails": 8, "ai_depth": "advanced",
        "max_caption_frames": 8, "caption_frames": 8,
        "max_parallel_uploads": 3, "parallel_uploads": 3,
        "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 3, "analytics": "full", "trial_days": 7,
        "audio_context": True,
    },
    "studio": {
        "name": "Studio", "price": 49.99,
        "put_daily": 234, "put_monthly": 7000, "aic_monthly": 45000,
        "max_accounts": 300, "max_accounts_per_platform": 75, "per_platform": 75,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": False,
        "hud": True, "excel": True, "flex": False,
        "priority_class": "p1", "queue_depth": 2500, "lookahead_hours": 72,
        "max_thumbnails": 12, "ai_depth": "max",
        "max_caption_frames": 15, "caption_frames": 15,
        "max_parallel_uploads": 4, "parallel_uploads": 4,
        "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 10, "analytics": "full_export", "trial_days": 7,
        "audio_context": True,
    },
    "agency": {
        "name": "Agency", "price": 99.99,
        "put_daily": 734, "put_monthly": 22000, "aic_monthly": 140000,
        "max_accounts": 1200, "max_accounts_per_platform": 300, "per_platform": 300,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": True, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 99999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max",
        "max_caption_frames": 15, "caption_frames": 15,
        "max_parallel_uploads": 6, "parallel_uploads": 6,
        "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 25, "analytics": "full_export", "trial_days": 7,
        "audio_context": True,
    },
    # ── Internal tiers (not sold via Stripe) ──
    "master_admin": {
        "name": "Admin", "price": 0,
        "put_daily": 9999, "put_monthly": 999999, "aic_monthly": 999999,
        "max_accounts": 999, "max_accounts_per_platform": 999,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": True, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 999999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max", "max_caption_frames": 20,
        "max_parallel_uploads": 6, "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 9999, "analytics": "full_export", "internal": True,
        "audio_context": True,
    },
    "friends_family": {
        "name": "Friends & Family", "price": 0,
        "put_daily": 999, "put_monthly": 12000, "aic_monthly": 5000,
        "max_accounts": 80, "max_accounts_per_platform": 80,
        "watermark": False, "ads": False, "ai": True,
        "scheduling": True, "webhooks": True, "white_label": True,
        "hud": True, "excel": True, "flex": True,
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
        "hud": True, "excel": True, "flex": True,
        "priority_class": "p0", "queue_depth": 99999, "lookahead_hours": 168,
        "max_thumbnails": 20, "ai_depth": "max", "max_caption_frames": 15,
        "max_parallel_uploads": 6, "custom_thumbnails": True, "ai_thumbnail_styling": True,
        "team_seats": 999, "analytics": "full_export", "internal": True,
    },
}

# ── Stripe Dashboard → Products → each Price → Metadata (Edit) ─────────────────
# Copy keys exactly; values must match entitlements / lookup keys in this file.
#
# Subscription Prices (lookup_key must match STRIPE_LOOKUP_TO_TIER):
#   kind              = subscription
#   tier              = creator_lite | creator_pro | studio | agency
#   put_month         = <int>   (same as TIER_CONFIG put_monthly for that tier)
#   aic_month         = <int>   (same as TIER_CONFIG aic_monthly)
#   flex_enabled      = true | false   (agency tier: true; others usually false)
#   price_usd         = <decimal>   (optional; for calculator / reporting)
#   priority_class    = p1 | p2 | p3 | p4   (optional; must match tier routing)
#
# Example (Creator Pro monthly): tier=creator_pro, put_month=1800, aic_month=13000,
#   flex_enabled=false, priority_class=p2, kind=subscription
#
# Top-up Prices (lookup_key must match TOPUP_PRODUCTS):
#   kind              = topup
#   wallet            = put | aic
#   amount            = <int>   (token pack size; must match TOPUP_PRODUCTS amount)
#   price_usd         = <decimal>   (optional)
#   promo             = true | false   (optional; Burst Week style promos)
#   expires_days      = 7   (optional; promo window hint for ops)
#
# Example (uploadm8_put_500): kind=topup, wallet=put, amount=500
#
# Ledger: checkout top-ups credit token_ledger with reason "topup" (legacy rows may say "topup_purchase").
STRIPE_LOOKUP_TO_TIER: Dict[str, str] = {
    "uploadm8_creator_lite_monthly": "creator_lite",
    "uploadm8_creatorlite_monthly": "creator_lite",
    "uploadm8_creatorpro_monthly":  "creator_pro",
    "uploadm8_studio_monthly":      "studio",
    "uploadm8_agency_monthly":      "agency",
    "uploadm8_launch_monthly":      "creator_lite",
    "uploadm8_creator_pro_monthly": "creator_pro",
}

# ============================================================
# Topup / Add-on Products
# Maps Stripe price lookup_key -> {wallet, amount} metadata
# ============================================================
TOPUP_PRODUCTS = {
    "uploadm8_put_250":  {"wallet": "put", "amount": 250,   "price": 4.99,  "price_usd": 4.99},
    "uploadm8_put_500":  {"wallet": "put", "amount": 500,   "price": 8.99,  "price_usd": 8.99},
    "uploadm8_put_1000": {"wallet": "put", "amount": 1000,  "price": 14.99, "price_usd": 14.99},
    "uploadm8_put_2500": {"wallet": "put", "amount": 2500,  "price": 29.99, "price_usd": 29.99},
    "uploadm8_put_5000": {"wallet": "put", "amount": 5000,  "price": 49.99, "price_usd": 49.99},
    "uploadm8_aic_500":  {"wallet": "aic", "amount": 500,   "price": 4.99,  "price_usd": 4.99},
    "uploadm8_aic_1000": {"wallet": "aic", "amount": 1000,  "price": 8.99,  "price_usd": 8.99},
    "uploadm8_aic_2500": {"wallet": "aic", "amount": 2500,  "price": 18.99, "price_usd": 18.99},
    "uploadm8_aic_5000": {"wallet": "aic", "amount": 5000,  "price": 34.99, "price_usd": 34.99},
    "uploadm8_aic_10000":{"wallet": "aic", "amount": 10000, "price": 59.99, "price_usd": 59.99},
}

# Priority class routing sets
PRIORITY_QUEUE_CLASSES = {"p0", "p1", "p2"}   # -> process:priority / publish:priority
NORMAL_QUEUE_CLASSES   = {"p3", "p4"}          # -> process:normal  / publish:normal

# ============================================================
# Canonical Tier & Entitlement Schema (shared with frontend)
# ============================================================
# All valid subscription_tier values. Frontend and backend MUST use these slugs.
TIER_SLUGS = ("free", "creator_lite", "creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin")

TIER_ALIASES: Dict[str, str] = {}


def normalize_tier(tier: str) -> str:
    """Return canonical tier slug. Unknown tiers fall back to free."""
    t = (tier or "free").lower().strip()
    if t in TIER_ALIASES:
        return TIER_ALIASES[t]
    return t if t in TIER_CONFIG else "free"


def get_tier_display_name(tier: str) -> str:
    """Return human-readable tier name from TIER_CONFIG."""
    t = normalize_tier(tier)
    return TIER_CONFIG.get(t, TIER_CONFIG["free"]).get("name", t.replace("_", " ").title())


# Public Stripe-sold tiers in upgrade order.
PUBLIC_UPGRADE_TIER_CHAIN = ("free", "creator_lite", "creator_pro", "studio", "agency")

# Subscription tiers counted as "on a paid plan" in admin KPI digests (excludes comped / staff tiers).
ADMIN_KPI_COUNTED_SUBSCRIPTION_TIERS = frozenset({"creator_lite", "creator_pro", "studio", "agency"})


def get_next_public_upgrade_tier(tier: str) -> Optional[str]:
    """Next tier in the public upgrade ladder, or None if already at top / non-public."""
    t = normalize_tier(tier)
    if t not in PUBLIC_UPGRADE_TIER_CHAIN:
        return None
    i = PUBLIC_UPGRADE_TIER_CHAIN.index(t)
    if i + 1 >= len(PUBLIC_UPGRADE_TIER_CHAIN):
        return None
    return PUBLIC_UPGRADE_TIER_CHAIN[i + 1]


# Human labels for /api/entitlements/tiers (marketing + success UIs)
_ANALYTICS_LABELS: Dict[str, str] = {
    "basic": "Basic analytics",
    "standard": "Standard analytics",
    "full": "Full account intel",
    "full_export": "Full + export-ready",
}

_PRIORITY_LANE_LABELS: Dict[str, str] = {
    "p0": "Dedicated (p0)",
    "p1": "Turbo (p1)",
    "p2": "Priority (p2)",
    "p3": "Standard (p3)",
    "p4": "Standard",
}


def _scheduling_window_phrase(lookahead_hours: Any) -> str:
    h = int(lookahead_hours or 0)
    if h >= 168:
        return "7 days"
    if h == 72:
        return "72 hours"
    if h == 24:
        return "24 hours"
    if h == 12:
        return "12 hours"
    if h == 4:
        return "4 hours"
    if h <= 0:
        return "this billing period"
    return f"{h} hours"


# Entitlement keys returned by entitlements_to_dict — frontend uses these exact keys
ENTITLEMENT_KEYS = (
    "tier", "tier_display", "put_daily", "put_monthly", "aic_monthly",
    "max_accounts", "max_accounts_per_platform", "can_watermark", "can_ai",
    "can_schedule", "can_webhooks", "can_white_label", "can_excel", "can_priority",
    "can_flex", "can_burn_hud", "show_ads", "priority_class", "queue_depth",
    "lookahead_hours", "max_caption_frames", "ai_depth", "max_thumbnails",
    "can_custom_thumbnails", "can_ai_thumbnail_styling", "max_parallel_uploads",
    "team_seats", "analytics", "trial_days", "is_internal", "can_audio_context",
)


def get_tiers_for_api() -> list:
    """Return tier metadata for /api/entitlements/tiers. Single source for frontend.
    Includes revenue tiers + internal (friends_family, lifetime, master_admin).
    app.js, wallet-tokens.js, and settings.html all consume this for consistent PUT/AIC/names."""
    all_slugs = ("free", "creator_lite", "creator_pro", "studio", "agency",
                 "friends_family", "lifetime", "master_admin")
    out = []
    for slug in all_slugs:
        cfg = TIER_CONFIG.get(slug, {})
        per_pf = cfg.get("max_accounts_per_platform", cfg.get("per_platform", 0))
        pc = str(cfg.get("priority_class", "p4"))
        ak = str(cfg.get("analytics", "basic"))
        lh = int(cfg.get("lookahead_hours", 0) or 0)
        out.append({
            "slug": slug,
            "name": cfg.get("name", slug.replace("_", " ").title()),
            "price": float(cfg.get("price", 0)),
            "put_monthly": cfg.get("put_monthly", 0),
            "aic_monthly": cfg.get("aic_monthly", 0),
            "max_accounts": cfg.get("max_accounts", 0),
            "max_accounts_per_platform": int(per_pf or 0),
            "queue_depth": cfg.get("queue_depth", 0),
            "lookahead_hours": lh,
            "internal": cfg.get("internal", False),
            "trial_days": int(cfg.get("trial_days", 0) or 0),
            "team_seats": int(cfg.get("team_seats", 1) or 1),
            "analytics": ak,
            "analytics_label": _ANALYTICS_LABELS.get(ak, ak.replace("_", " ").title()),
            "ai_depth": str(cfg.get("ai_depth", "basic")),
            "webhooks": bool(cfg.get("webhooks", False)),
            "white_label": bool(cfg.get("white_label", False)),
            "hud": bool(cfg.get("hud", False)),
            "excel": bool(cfg.get("excel", False)),
            "flex": bool(cfg.get("flex", False)),
            "watermark": bool(cfg.get("watermark", True)),
            "max_parallel_uploads": int(
                cfg.get("max_parallel_uploads", cfg.get("parallel_uploads", 1)) or 1
            ),
            "max_thumbnails": int(cfg.get("max_thumbnails", 1) or 1),
            "max_caption_frames": int(
                cfg.get("max_caption_frames", cfg.get("caption_frames", 3)) or 3
            ),
            "priority_class": pc,
            "queue_lane_label": _PRIORITY_LANE_LABELS.get(pc, pc),
            "scheduling_window_label": _scheduling_window_phrase(lh),
        })
    return out


# ============================================================
# Entitlements Dataclass
# ============================================================
@dataclass
class Entitlements:
    """Fully resolved entitlements for a user."""

    tier: str = "free"
    tier_display: str = "Free"

    # Wallet
    put_daily: int = 4
    put_monthly: int = 120
    aic_monthly: int = 1200

    # Accounts
    max_accounts: int = 16
    max_accounts_per_platform: int = 4

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
    queue_depth: int = 10
    lookahead_hours: int = 4

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

    # Audio context (Whisper transcription) — rides on can_ai; True for all tiers
    can_audio_context: bool = True

    # Per-user override audit trail
    custom_overrides: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Builder Functions
# ============================================================

def get_entitlements_for_tier(tier: str) -> Entitlements:
    """Build Entitlements from a tier slug. Uses normalize_tier (unknown->free)."""
    t = normalize_tier(tier)
    cfg = TIER_CONFIG.get(t, TIER_CONFIG["free"])
    return Entitlements(
        tier=t,
        tier_display=cfg.get("name", t.replace("_", " ").title()),
        put_daily=cfg.get("put_daily", 4),
        put_monthly=cfg.get("put_monthly", 120),
        aic_monthly=cfg.get("aic_monthly", 1200),
        max_accounts=cfg.get("max_accounts", 16),
        max_accounts_per_platform=cfg.get("max_accounts_per_platform", 4),
        can_watermark=cfg.get("watermark", True),
        can_ai=cfg.get("ai", False),
        can_schedule=cfg.get("scheduling", False),
        can_webhooks=cfg.get("webhooks", False),
        can_white_label=cfg.get("white_label", False),
        can_excel=cfg.get("excel", False),
        can_priority=cfg.get("priority_class", "p4") in PRIORITY_QUEUE_CLASSES,
        can_flex=cfg.get("flex", False),
        can_burn_hud=cfg.get("hud", False),
        show_ads=cfg.get("ads", True),
        priority_class=cfg.get("priority_class", "p4"),
        queue_depth=cfg.get("queue_depth", 10),
        lookahead_hours=cfg.get("lookahead_hours", 4),
        max_caption_frames=cfg.get("max_caption_frames", 3),
        ai_depth=cfg.get("ai_depth", "basic"),
        max_thumbnails=cfg.get("max_thumbnails", 1),
        can_custom_thumbnails=cfg.get("custom_thumbnails", False),
        can_ai_thumbnail_styling=cfg.get("ai_thumbnail_styling", False),
        can_audio_context=cfg.get("audio_context", True),
        max_parallel_uploads=cfg.get("max_parallel_uploads", 1),
        team_seats=cfg.get("team_seats", 1),
        analytics=cfg.get("analytics", "basic"),
        trial_days=cfg.get("trial_days", 0),
        is_internal=cfg.get("internal", False),
    )


def get_entitlements_from_user(
    user_record: dict,
    overrides: Optional[dict] = None,
) -> Entitlements:
    """
    Build Entitlements from a users table row + optional per-user admin overrides.

    Priority order (highest wins):
      1. Role override: master_admin role -> master_admin tier entitlements
      2. Per-user overrides from entitlement_overrides table
      3. Tier defaults from TIER_CONFIG (admin role uses normal tier quotas, not master_admin)
    """
    # Role override - only master_admin role gets full internal entitlements.
    # Admin role keeps subscription-tier entitlements for quota/account correctness.
    role = (user_record.get("role") or "user").lower()
    if role == "master_admin":
        return get_entitlements_for_tier("master_admin")

    raw_tier = (user_record.get("subscription_tier") or "free").lower()
    tier = normalize_tier(raw_tier)
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
        "can_custom_thumbnails":     ent.can_custom_thumbnails,
        "can_ai_thumbnail_styling":  ent.can_ai_thumbnail_styling,
        "can_audio_context":          ent.can_audio_context,
        "max_parallel_uploads":      ent.max_parallel_uploads,
        "team_seats":                ent.team_seats,
        "analytics":                 ent.analytics,
        "trial_days":                ent.trial_days,
        "is_internal":               ent.is_internal,
    }


# ============================================================
# Guard Helpers
# ============================================================

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


def should_generate_thumbnails(entitlements: Optional["Entitlements"]) -> bool:
    """Return True if tier allows thumbnail generation (base or styled)."""
    if not entitlements:
        return False
    return (getattr(entitlements, "max_thumbnails", 1) or 1) >= 1 or getattr(
        entitlements, "can_custom_thumbnails", False
    )


# ============================================================
# PUT/AIC Cost Formula — canonical, imported by app.py + worker.py
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
    hud_enabled: bool = False,
    is_priority: bool = False,
    num_thumbnails: int = 1,
) -> int:
    """
    Deterministic PUT cost per upload job.
      base          = 10
      +5  HUD burn (if enabled and allowed)
      +5  priority lane (p0-p2)
      +2  per extra platform beyond first
      +1  per extra thumbnail beyond 1
    """
    cost = 10
    if hud_enabled:
        cost += 5
    if is_priority:
        cost += 5
    cost += max(0, num_platforms - 1) * 2
    cost += max(0, num_thumbnails - 1)
    return cost


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
    use_hud: bool = False,
    num_thumbnails: Optional[int] = None,
    *,
    duration_seconds: Optional[float] = None,
    duration_hint: Optional[float] = None,
    file_size: Optional[int] = None,
    user_prefs: Optional[Dict[str, Any]] = None,
    has_telemetry: bool = False,
    billing_env: Optional[Dict[str, bool]] = None,
    return_breakdown: bool = False,
) -> Union[Tuple[int, int], Tuple[int, int, Dict[str, Any]]]:
    """
    Return (put_cost, aic_cost) using per-service AIC weights + duration scaling
    (see stages/ai_service_costs.py). PUT uses actual thumbnail count capped by tier.

    user_prefs: merged upload prefs (auto_captions, auto_thumbnails, use_audio_context, …).
    duration_seconds / file_size: used to estimate clip length before probe (optional).
    """
    prefs = dict(user_prefs or {})
    env = billing_env or billing_env_from_os()
    thumbs = effective_num_thumbnails(
        entitlements.max_thumbnails,
        prefs,
        use_ai,
        num_thumbnails,
    )
    is_priority = entitlements.priority_class in PRIORITY_QUEUE_CLASSES
    put = compute_put_cost(
        num_platforms=num_platforms,
        hud_enabled=(use_hud and entitlements.can_burn_hud),
        is_priority=is_priority,
        num_thumbnails=thumbs,
    )
    use_stack = bool(use_ai and entitlements.can_ai)
    aic, dbg = compute_aic_breakdown(
        can_ai=bool(entitlements.can_ai),
        user_prefs=prefs,
        use_ai_request=use_stack,
        has_telemetry=bool(has_telemetry),
        duration_seconds=float(duration_seconds or 0.0),
        file_size=file_size,
        duration_hint=duration_hint,
        max_caption_frames=int(entitlements.max_caption_frames or 0),
        num_thumbnails=thumbs,
        env=env,
    )
    if return_breakdown:
        return put, aic, dbg
    return put, aic