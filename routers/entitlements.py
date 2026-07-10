"""
UploadM8 Entitlements & Pricing routes — extracted from app.py.
Public endpoints for tier/entitlement schema and pricing info.
"""

from fastapi import APIRouter, Query

import core.state
from stages.db import sanitize_watermark_burn_text
from stages.entitlements import (
    TIER_SLUGS,
    ENTITLEMENT_KEYS,
    PUBLIC_TIER_SLUGS,
    get_tiers_for_api,
    get_effective_topup_products,
)
from services.thumbnail_studio import format_library_rows
from services.pricing_surfaces import build_pricing_surfaces_snapshot

router = APIRouter(tags=["entitlements"])


# ── Helper ───────────────────────────────────────────────────────
def _entitlements_tiers_payload():
    """Canonical tier list and entitlement schema. Keys match ENTITLEMENT_KEYS in stages/entitlements.py."""
    wtxt = sanitize_watermark_burn_text(
        core.state.admin_settings_cache.get("watermark_burn_text")
    )
    return {
        "tiers": get_tiers_for_api(),
        "tier_slugs": list(TIER_SLUGS),
        "entitlement_keys": list(ENTITLEMENT_KEYS),
        # Shown on watermarked (free) tiers; actual burn-in enforced server-side on worker.
        "watermark_burn_text": wtxt,
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


@router.get("/api/entitlements/thumbnail-studio-niches")
async def get_thumbnail_studio_niches():
    """Canonical audience/niche list for Thumbnail Studio and settings UI."""
    from services.thumbnail_niches import niche_options_payload

    return {"niches": niche_options_payload()}


@router.get("/api/entitlements/thumbnail-studio-formats")
async def get_thumbnail_studio_layout_formats(niche: str = Query("")):
    """
    Thumbnail Studio layout chips. Lives on this router so it stays public with
    zero auth deps — same payload shape as the legacy ``/api/thumbnail-studio/formats``.

    Returns static presets plus procedural ``dyn-{niche}-{nnnn}`` rows scoped to
    ``niche`` (defaults to ``general``). When ``niche`` is set, matching rows are sorted
    first for convenience; ``format_key`` is the only key required downstream.
    """
    from services.thumbnail_niches import normalize_niche

    n = normalize_niche(niche)
    rows = list(format_library_rows(n if n else None))
    if n and n != "general":

        def _niche_rank(r: dict) -> tuple:
            hint = str(r.get("niche", "")).lower()
            rk = str(r.get("key") or "").lower()
            if hint == n or n in hint or rk.startswith(f"dyn-{n}-"):
                return (0, str(r.get("name") or ""))
            return (1, str(r.get("name") or ""))

        rows.sort(key=_niche_rank)
    return {"formats": rows}


def _topup_pricing_row(lookup_key: str, meta: dict) -> dict:
    wallet = meta.get("wallet", "")
    row = {
        "lookup_key": lookup_key,
        "wallet": wallet,
        "amount": int(meta.get("amount") or 0),
        "price_usd": meta.get("price_usd") or meta.get("price"),
        "put": int(meta.get("put") or 0),
        "aic": int(meta.get("aic") or 0),
    }
    if wallet == "bundle":
        row["label"] = f"{meta.get('put')} PUT + {meta.get('aic')} AIC"
    else:
        row["label"] = f"{wallet.upper()} {meta.get('amount', 0)}"
    return row


@router.get("/api/pricing")
async def get_public_pricing():
    """
    Public pricing and entitlements for landing page and billing UI.
    Returns tiers with PUT/AIC, perks, and top-up packs (with suggested prices).
    """
    from services.stripe_lookup_keys import STRIPE_MONTHLY_BY_TIER, STRIPE_YEARLY_BY_TIER

    by_slug = {row["slug"]: row for row in get_tiers_for_api()}
    tiers = []
    for slug in PUBLIC_TIER_SLUGS:
        row = dict(by_slug.get(slug, {}))
        row["stripe_lookup_key"] = STRIPE_MONTHLY_BY_TIER.get(slug)
        row["stripe_lookup_key_annual"] = STRIPE_YEARLY_BY_TIER.get(slug)
        tiers.append(row)
    topups = []
    for lookup_key, meta in get_effective_topup_products().items():
        topups.append(_topup_pricing_row(lookup_key, meta))
    wtxt = sanitize_watermark_burn_text(
        core.state.admin_settings_cache.get("watermark_burn_text")
    )
    return {
        "tiers": tiers,
        "topups": topups,
        "watermark_burn_text": wtxt,
        "debit_surfaces": build_pricing_surfaces_snapshot(),
    }
