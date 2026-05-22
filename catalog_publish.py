"""Reload in-memory pricing overlays from Postgres (single entry point).

Use ``refresh_pricing_caches_after_catalog_sync`` for:

- **API startup** (``app.py`` lifespan): hydrate ``catalog_products`` → tier/top-up overlays and
  ``billing_catalog`` so ``get_effective_tier_config()`` / ``/api/pricing`` match the DB without
  restart.
- **After catalog Stripe work**: per-product ``sync_product``, bulk ``sync_all``, admin PATCH
  reload, pricing-request resolve — same helper as startup.

Steps:

1. ``core.state.catalog_pricing_cache`` from ``catalog_products`` (subscription + top-up rows).
2. ``core.state.billing_catalog_cache`` from the ``billing_catalog`` singleton.

``stages/entitlements.get_effective_tier_config`` / ``get_effective_topup_products`` merge **code
defaults (``TIER_CONFIG`` / ``TOPUP_PRODUCTS``) → catalog overlay → billing_catalog JSON**
(billing wins on key collisions). Load order below matches that merge (catalog slice first, then
billing overrides).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

import core.state

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger("uploadm8-api")


async def refresh_pricing_caches_after_catalog_sync(
    conn: "asyncpg.Connection",
) -> Dict[str, Any]:
    """Reload in-memory overlays so ``GET /api/pricing`` matches ``catalog_products`` + billing overrides.

    Order (matches ``effective_tier_config`` / ``effective_topup_products`` merge):

    1. **catalog_products →** ``load_catalog_pricing_cache`` → ``core.state.catalog_pricing_cache``
       (tier + top-up overlays from the live table).
    2. **billing_catalog singleton →** ``load_billing_catalog_cache`` → ``core.state.billing_catalog_cache``
       (master-admin JSON overrides).

    After this returns successfully, ``/api/pricing`` reads current DB-backed list prices and caps
    without process restart.
    """
    detail: Dict[str, Any] = {
        "catalog_products_overlay_loaded": False,
        "billing_catalog_singleton_loaded": False,
        "catalog_pricing_loaded_at": None,
        "api_pricing_ready": False,
    }
    try:
        from services.catalog_pricing_cache import load_catalog_pricing_cache
        from services.billing_catalog import load_billing_catalog_cache

        await load_catalog_pricing_cache(conn)
        detail["catalog_products_overlay_loaded"] = True
        detail["catalog_pricing_loaded_at"] = core.state.catalog_pricing_cache.get("loaded_at")

        await load_billing_catalog_cache(conn)
        detail["billing_catalog_singleton_loaded"] = True

        detail["api_pricing_ready"] = True
        return detail
    except Exception as exc:
        logger.warning("refresh_pricing_caches_after_catalog_sync failed: %s", exc)
        detail["error"] = str(exc)
        return detail
