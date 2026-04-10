"""
UploadM8 Entitlements & Pricing routes — extracted from app.py.
Public endpoints for tier/entitlement schema and pricing info.
"""

from fastapi import APIRouter

from stages.entitlements import (
    TIER_CONFIG,
    TOPUP_PRODUCTS,
    TIER_SLUGS,
    ENTITLEMENT_KEYS,
    get_tiers_for_api,
)

router = APIRouter(tags=["entitlements"])


# ── Helper ───────────────────────────────────────────────────────
def _entitlements_tiers_payload():
    """Canonical tier list and entitlement schema. Keys match ENTITLEMENT_KEYS in stages/entitlements.py."""
    return {
        "tiers": get_tiers_for_api(),
        "tier_slugs": list(TIER_SLUGS),
        "entitlement_keys": list(ENTITLEMENT_KEYS),
    }


# ── Routes ───────────────────────────────────────────────────────
@router.get("/api/entitlements/tiers")
async def get_entitlements_tiers():
    """Canonical tier list. Frontend uses this as single source."""
    return _entitlements_tiers_payload()


@router.get("/api/entitlements")
async def get_entitlements():
    """Alias for /api/entitlements/tiers — backward compatibility."""
    return _entitlements_tiers_payload()


@router.get("/api/pricing")
async def get_public_pricing():
    """
    Public pricing and entitlements for landing page and billing UI.
    Returns tiers with PUT/AIC, perks, and top-up packs (with suggested prices).
    """
    STRIPE_LOOKUP = {
        "creator_lite": "uploadm8_creatorlite_monthly",
        "creator_pro": "uploadm8_creatorpro_monthly",
        "studio": "uploadm8_studio_monthly",
        "agency": "uploadm8_agency_monthly",
    }
    tiers = []
    for slug in ("free", "creator_lite", "creator_pro", "studio", "agency"):
        cfg = TIER_CONFIG.get(slug, {})
        tiers.append({
            "slug": slug,
            "name": cfg.get("name", slug.replace("_", " ").title()),
            "price": float(cfg.get("price", 0)),
            "put_monthly": cfg.get("put_monthly", 0),
            "aic_monthly": cfg.get("aic_monthly", 0),
            "max_accounts": cfg.get("max_accounts", 0),
            "lookahead_hours": cfg.get("lookahead_hours", 0),
            "queue_depth": cfg.get("queue_depth", 0),
            "max_thumbnails": cfg.get("max_thumbnails", 0),
            "stripe_lookup_key": STRIPE_LOOKUP.get(slug),
        })
    topups = []
    for lookup_key, meta in TOPUP_PRODUCTS.items():
        topups.append({
            "lookup_key": lookup_key,
            "wallet": meta.get("wallet", ""),
            "amount": meta.get("amount", 0),
            "price_usd": meta.get("price_usd") or meta.get("price"),
            "label": f"{meta.get('wallet', '').upper()} {meta.get('amount', 0)}",
        })
    return {"tiers": tiers, "topups": topups}
