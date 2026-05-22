"""
UploadM8 mutable runtime state — connection pools, caches, encryption keys.
Everything here is set at startup or mutated during request handling.
Import from here (not app.py) when you need to read or write global state.
"""

from typing import Any, Dict, Optional, Set

import asyncpg
import redis.asyncio as aioredis

from core.config import BILLING_MODE

# ── Connection pools ──────────────────────────────────────────
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None

# ── Token encryption keys ─────────────────────────────────────
ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"

# ── Admin settings (hot-reloaded from DB) ─────────────────────
admin_settings_cache: Dict[str, Any] = {
    "demo_data_enabled": False,
    "billing_mode": BILLING_MODE,
    # Free-tier FFmpeg drawtext string (master admin editable; worker reads DB each job).
    "watermark_burn_text": "Upload M8",
}

# Master-admin catalog overrides (merged with stages/entitlements TIER_CONFIG + TOPUP_PRODUCTS).
billing_catalog_cache: Dict[str, Any] = {
    "tier_overrides": {},
    "topup_overrides": {},
    "put_cost_overrides": {},
    "tier_service_overrides": {},
    "last_sync_at": None,
    "last_sync_ok": None,
    "last_sync_error": None,
    "last_sync_detail": None,
}

# Rows from ``catalog_products`` (subscriptions + top-ups), merged before billing_catalog_cache.
catalog_pricing_cache: Dict[str, Any] = {
    "tier_overlay": {},
    "topup_overlay": {},
    "loaded_at": None,
}

# ── Schema introspection cache ────────────────────────────────
_UPLOADS_COLS: Optional[Set[str]] = None

# ── In-memory rate-limit buckets (replace with Redis later) ───
_RATE_BUCKETS: Dict[str, Dict[str, Any]] = {}


def require_pool() -> asyncpg.Pool:
    """Return the live asyncpg pool or raise HTTP 503 if it is not ready.

    The pool is set during FastAPI lifespan startup and torn down on shutdown.
    Requests that race with either edge (preboot, hot-reload, deploy, Ctrl+C)
    would otherwise hit ``AttributeError: 'NoneType' object has no attribute
    'acquire'``. Surface a clean ``503 Service Unavailable`` instead so we
    don't pollute Sentry with shutdown noise and clients can retry sensibly.
    """
    pool = db_pool
    if pool is None:
        # Local import to avoid pulling FastAPI into modules that only need state.
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Database pool not ready")
    return pool
