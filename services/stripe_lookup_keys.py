"""Canonical Stripe price lookup_key maps for public subscription tiers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from stages.entitlements import PUBLIC_TIER_SLUGS

STRIPE_MONTHLY_BY_TIER: Dict[str, str] = {
    "creator_lite": "uploadm8_creatorlite_monthly",
    "creator_pro": "uploadm8_creatorpro_monthly",
    "studio": "uploadm8_studio_monthly",
    "agency": "uploadm8_agency_monthly",
}

STRIPE_YEARLY_BY_TIER: Dict[str, str] = {
    "creator_lite": "uploadm8_creatorlite_yearly",
    "creator_pro": "uploadm8_creatorpro_yearly",
    "studio": "uploadm8_studio_yearly",
    "agency": "uploadm8_agency_yearly",
}


def yearly_lookup_from_monthly(monthly_key: Optional[str]) -> Optional[str]:
    if not monthly_key:
        return None
    mk = str(monthly_key).strip()
    if mk.endswith("_monthly"):
        return mk[:-8] + "_yearly"
    return None


def subscription_stripe_keys(
    tier_slug: str,
    *,
    catalog_monthly_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (monthly_lookup_key, yearly_lookup_key) for a paid public tier."""
    slug = str(tier_slug or "").strip()
    if slug not in PUBLIC_TIER_SLUGS or slug == "free":
        return None, None
    monthly = (catalog_monthly_key or "").strip() or STRIPE_MONTHLY_BY_TIER.get(slug)
    yearly = STRIPE_YEARLY_BY_TIER.get(slug) or yearly_lookup_from_monthly(monthly)
    return monthly, yearly


async def subscription_keys_from_catalog(conn) -> Dict[str, Dict[str, Any]]:
    """Load subscription lookup keys from catalog_products (monthly rows)."""
    out: Dict[str, Dict[str, Any]] = {}
    rows = await conn.fetch(
        """
        SELECT tier_slug, lookup_key, stripe_product_id, price_usd, price_usd_yearly
        FROM catalog_products
        WHERE product_kind = 'subscription'
          AND tier_slug IS NOT NULL
          AND is_archived = FALSE
        """
    )
    for row in rows:
        slug = str(row["tier_slug"] or "").strip()
        if not slug:
            continue
        monthly, yearly = subscription_stripe_keys(slug, catalog_monthly_key=row["lookup_key"])
        out[slug] = {
            "tier_slug": slug,
            "catalog_lookup_key": row["lookup_key"],
            "stripe_lookup_key_monthly": monthly,
            "stripe_lookup_key_yearly": yearly,
            "stripe_product_id": row.get("stripe_product_id"),
            "price_usd": float(row["price_usd"]) if row.get("price_usd") is not None else None,
            "price_usd_yearly": float(row["price_usd_yearly"]) if row.get("price_usd_yearly") is not None else None,
        }
    for slug in PUBLIC_TIER_SLUGS:
        if slug == "free" or slug in out:
            continue
        monthly, yearly = subscription_stripe_keys(slug)
        out[slug] = {
            "tier_slug": slug,
            "catalog_lookup_key": monthly,
            "stripe_lookup_key_monthly": monthly,
            "stripe_lookup_key_yearly": yearly,
            "stripe_product_id": None,
            "price_usd": None,
            "price_usd_yearly": None,
        }
    return out
