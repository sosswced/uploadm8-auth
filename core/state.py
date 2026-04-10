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
admin_settings_cache: Dict[str, Any] = {"demo_data_enabled": False, "billing_mode": BILLING_MODE}

# ── Schema introspection cache ────────────────────────────────
_UPLOADS_COLS: Optional[Set[str]] = None

# ── In-memory rate-limit buckets (replace with Redis later) ───
_RATE_BUCKETS: Dict[str, Dict[str, Any]] = {}
