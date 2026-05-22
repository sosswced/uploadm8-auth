"""
After master admin saves ``billing_catalog`` overrides, regenerate PNGs and optionally sync Stripe.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import core.state
from core.config import FRONTEND_STATIC_DIR, STRIPE_SECRET_KEY
from services.billing_catalog import (
    effective_tier_config,
    effective_topup_products,
    update_sync_status,
)
from services.catalog_publish import refresh_pricing_caches_after_catalog_sync
from services.product_card_art import generate_all
from services.stripe_catalog_sync import sync_stripe_catalog

logger = logging.getLogger("uploadm8-api")


def _images_dir() -> Path:
    raw = os.environ.get("PRODUCT_CARD_OUT_DIR", "").strip()
    if raw:
        return Path(raw)
    return FRONTEND_STATIC_DIR / "images"


def _cloud_icon_path() -> Path | None:
    raw = os.environ.get("PRODUCT_CARD_CLOUD_ICON", "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_file() else None
    cand = FRONTEND_STATIC_DIR / "images" / "cloud.png"
    return cand if cand.is_file() else None


def _generate_pngs_sync() -> None:
    tc = effective_tier_config()
    tp = effective_topup_products()
    out = _images_dir()
    cloud = _cloud_icon_path()
    generate_all(tc, tp, out, cloud, log=lambda m: logger.info("product_card: %s", m))


def _stripe_sync_sync() -> dict:
    tc = effective_tier_config()
    tp = effective_topup_products()
    image_dir = _images_dir()
    dry = os.environ.get("STRIPE_SYNC_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
    return sync_stripe_catalog(tc, tp, image_dir, dry_run=dry)


async def run_billing_catalog_sync_job() -> None:
    """Reload DB overrides, render cards, push Stripe; persist last_sync_* on billing_catalog."""
    detail: dict = {}
    err: str | None = None
    ok = True
    try:
        async with core.state.require_pool().acquire() as conn:
            await refresh_pricing_caches_after_catalog_sync(conn)

        await asyncio.to_thread(_generate_pngs_sync)
        detail["png_dir"] = str(_images_dir())

        disabled = os.environ.get("DISABLE_AUTO_STRIPE_SYNC", "").strip().lower() in ("1", "true", "yes")
        if disabled:
            detail["stripe"] = "skipped DISABLE_AUTO_STRIPE_SYNC"
        elif not STRIPE_SECRET_KEY:
            detail["stripe"] = "skipped no STRIPE_SECRET_KEY"
        else:
            stripe_result = await asyncio.to_thread(_stripe_sync_sync)
            detail["stripe"] = stripe_result
            if stripe_result.get("errors"):
                ok = False
                err = "; ".join(str(e) for e in stripe_result["errors"][:5])

    except Exception as e:
        ok = False
        err = str(e)
        logger.exception("billing catalog sync job failed")

    try:
        async with core.state.require_pool().acquire() as conn:
            await update_sync_status(conn, ok, err, detail)
            await refresh_pricing_caches_after_catalog_sync(conn)
    except Exception as e:
        logger.warning("could not persist catalog sync status: %s", e)
