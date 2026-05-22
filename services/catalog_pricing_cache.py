"""Load ``catalog_products`` rows into in-memory overlays for entitlement pricing.

Merge order (see billing_catalog): code defaults → DB catalog overlay →
``billing_catalog`` master-admin JSON overrides.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncpg

import core.state
from stages.entitlements import TIER_CONFIG, TOPUP_PRODUCTS
logger = logging.getLogger("uploadm8-api")

_TOPUP_KINDS = ("topup_put", "topup_aic", "topup_bundle")


def _optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _subscription_row_to_patch(row: asyncpg.Record) -> Dict[str, Any]:
    """Build a tier patch dict compatible with ``_merge_tier_dict``."""
    patch: Dict[str, Any] = {}
    if row.get("display_name"):
        patch["name"] = str(row["display_name"])
    pf = _optional_float(row.get("price_usd"))
    if pf is not None:
        patch["price"] = pf
    py = _optional_float(row.get("price_usd_yearly"))
    if py is not None:
        patch["price_annual"] = py

    for col, key in (
        ("put_daily", "put_daily"),
        ("put_monthly", "put_monthly"),
        ("aic_monthly", "aic_monthly"),
        ("max_accounts", "max_accounts"),
        ("max_accounts_per_platform", "max_accounts_per_platform"),
        ("queue_depth", "queue_depth"),
        ("lookahead_hours", "lookahead_hours"),
        ("trial_days", "trial_days"),
        ("team_seats", "team_seats"),
    ):
        iv = _optional_int(row.get(col))
        if iv is not None:
            patch[key] = iv

    for col in (
        "watermark",
        "ads",
        "webhooks",
        "white_label",
        "hud",
        "excel",
        "flex",
    ):
        if row.get(col) is not None:
            patch[col] = bool(row[col])

    if row.get("priority_class"):
        patch["priority_class"] = str(row["priority_class"])
    if row.get("ai_depth"):
        patch["ai_depth"] = str(row["ai_depth"])
    if row.get("analytics"):
        patch["analytics"] = str(row["analytics"])
    return patch


async def load_catalog_pricing_cache(conn: asyncpg.Connection) -> None:
    """Refresh ``core.state.catalog_pricing_cache`` from ``catalog_products``."""
    tier_overlay: Dict[str, Dict[str, Any]] = {}
    topup_overlay: Dict[str, Dict[str, Any]] = {}

    try:
        subs = await conn.fetch(
            """
            SELECT tier_slug, display_name, price_usd, price_usd_yearly,
                   put_daily, put_monthly, aic_monthly, max_accounts, max_accounts_per_platform,
                   queue_depth, lookahead_hours, trial_days, team_seats,
                   watermark, ads, webhooks, white_label, hud, excel, flex,
                   priority_class, ai_depth, analytics
            FROM catalog_products
            WHERE product_kind = 'subscription' AND tier_slug IS NOT NULL
              AND is_archived = FALSE
            """
        )
        for row in subs:
            slug = row["tier_slug"]
            if not slug or slug not in TIER_CONFIG:
                continue
            patch = _subscription_row_to_patch(row)
            if patch:
                tier_overlay[str(slug)] = patch

        tops = await conn.fetch(
            """
            SELECT lookup_key, price_usd, wallet, token_amount, put_monthly, aic_monthly, product_kind
            FROM catalog_products
            WHERE product_kind = ANY($1::text[]) AND is_archived = FALSE
            """,
            list(_TOPUP_KINDS),
        )
        for row in tops:
            lk = row["lookup_key"]
            if not lk or lk not in TOPUP_PRODUCTS:
                continue
            base = TOPUP_PRODUCTS[lk]
            price = _optional_float(row.get("price_usd"))
            amt = _optional_int(row.get("token_amount"))
            put_m = _optional_int(row.get("put_monthly"))
            aic_m = _optional_int(row.get("aic_monthly"))
            w = row.get("wallet")
            patch: Dict[str, Any] = {}
            if price is not None:
                patch["price"] = price
                patch["price_usd"] = price
            if (base.get("wallet") or "") == "bundle":
                if put_m is not None:
                    patch["put"] = put_m
                if aic_m is not None:
                    patch["aic"] = aic_m
            elif amt is not None:
                patch["amount"] = amt
            if w:
                patch["wallet"] = str(w).lower().strip()
            if patch:
                topup_overlay[str(lk)] = patch
    except Exception as exc:
        logger.warning("catalog_pricing_cache load failed: %s", exc)
        tier_overlay = {}
        topup_overlay = {}

    core.state.catalog_pricing_cache["tier_overlay"] = tier_overlay
    core.state.catalog_pricing_cache["topup_overlay"] = topup_overlay
    core.state.catalog_pricing_cache["loaded_at"] = datetime.now(timezone.utc).isoformat()
