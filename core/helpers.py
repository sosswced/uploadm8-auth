"""
UploadM8 shared helpers — pure utility functions extracted from app.py.
No mutable state lives here; anything that touches globals uses core.state.
"""

import json
import uuid
import hashlib
import time
import secrets
import re
import logging
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit

import asyncpg

import core.state
from stages.entitlements import get_entitlements_for_tier, entitlements_to_dict

logger = logging.getLogger("uploadm8-api")

# ---------------------------------------------------------------------------
# DB JSON CODECS (architectural cleanliness)
# Forces asyncpg to decode json/jsonb into Python objects.
# Keep _safe_json as a belt-and-suspenders fallback until schema is fully normalized.
# ---------------------------------------------------------------------------
async def _init_asyncpg_codecs(conn):
    try:
        await conn.set_type_codec(
            'json',
            encoder=lambda v: json.dumps(v, separators=(',', ':'), ensure_ascii=False),
            decoder=json.loads,
            schema='pg_catalog',
        )
    except Exception:
        pass
    try:
        await conn.set_type_codec(
            'jsonb',
            encoder=lambda v: json.dumps(v, separators=(',', ':'), ensure_ascii=False),
            decoder=json.loads,
            schema='pg_catalog',
        )
    except Exception:
        pass

async def _load_uploads_columns(pool):
    """Cache uploads column set to avoid UndefinedColumnError when schema drifts."""
    if core.state._UPLOADS_COLS is not None:
        return core.state._UPLOADS_COLS
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name
                 FROM information_schema.columns
                 WHERE table_schema='public' AND table_name='uploads'"""
        )
        core.state._UPLOADS_COLS = {r['column_name'] for r in rows}
    return core.state._UPLOADS_COLS

_COL_NAME_RE = __import__("re").compile(r"^[a-z_][a-z0-9_]*$")


def _safe_col(col: str, allowed: frozenset) -> str:
    """Validate a SQL column name against an explicit allowlist.
    Returns the name unchanged so it can be used inline in f-strings."""
    if col not in allowed:
        raise ValueError(f"Disallowed SQL column: {col!r}")
    return col


def _pick_cols(wanted, available):
    out = [c for c in wanted if c in available]
    for c in out:
        if not _COL_NAME_RE.match(c):
            raise ValueError(f"Invalid column name: {c!r}")
    return out


def _safe_json(v, default=None):
    """Parse JSON stored as text OR already-parsed objects. Defensive until schema is fully jsonb."""
    if v is None:
        return default
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return default


def _valid_uuid(s: str) -> bool:
    """Return True if s is a valid UUID string (avoids 500 when frontend sends 'undefined' etc)."""
    if not s or not isinstance(s, str) or len(s) != 36:
        return False
    try:
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False

# Sensitive query-param keys that must never appear in logs
_SENSITIVE_KEYS = {"access_token", "client_secret", "code", "refresh_token", "fb_exchange_token"}

def redact_url(url: str) -> str:
    """Strip sensitive query params from a URL before logging it."""
    try:
        parts = urlsplit(url)
        q = parse_qsl(parts.query, keep_blank_values=True)
        redacted = [(k, "***" if k in _SENSITIVE_KEYS else v) for k, v in q]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted), parts.fragment))
    except Exception:
        return "<url-redact-error>"

# Tier ranking for upgrade/downgrade detection (higher index = higher tier)
_TIER_RANK = {
    "free": 0, "launch": 1, "creator_lite": 1, "creator_pro": 2, "studio": 3,
    "agency": 4, "friends_family": 5, "lifetime": 6, "master_admin": 7,
}
def _tier_is_upgrade(old: str, new: str) -> bool:
    return _TIER_RANK.get(new, 0) >= _TIER_RANK.get(old, 0)

def get_plan(tier: str) -> dict:
    """
    Backward-compat shim — returns entitlements_to_dict() so existing callers
    that do plan.get("ai") / plan.get("watermark") keep working unchanged.
    """
    return entitlements_to_dict(get_entitlements_for_tier(tier))

# ============================================================
# Helpers
# ============================================================
def _now_utc(): return datetime.now(timezone.utc)
def _sha256_hex(s: str): return hashlib.sha256(s.encode()).hexdigest()
def _req_id(): return f"req_{int(time.time())}_{secrets.token_hex(4)}"
