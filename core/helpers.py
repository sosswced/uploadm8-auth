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
            encoder=lambda v: json.dumps(
                v, separators=(',', ':'), ensure_ascii=False, default=str
            ),
            decoder=json.loads,
            schema='pg_catalog',
        )
    except Exception:
        pass
    try:
        await conn.set_type_codec(
            'jsonb',
            encoder=lambda v: json.dumps(
                v, separators=(',', ':'), ensure_ascii=False, default=str
            ),
            decoder=json.loads,
            schema='pg_catalog',
        )
    except Exception:
        pass

async def ensure_uploads_columns_loaded(conn) -> set:
    """Populate ``core.state._UPLOADS_COLS`` from ``information_schema`` using ``conn`` (no acquire)."""
    if core.state._UPLOADS_COLS is not None:
        return core.state._UPLOADS_COLS
    rows = await conn.fetch(
        """SELECT column_name
             FROM information_schema.columns
             WHERE table_schema='public' AND table_name='uploads'"""
    )
    core.state._UPLOADS_COLS = {r["column_name"] for r in rows}
    return core.state._UPLOADS_COLS


async def _load_uploads_columns(pool):
    """Cache uploads column set to avoid UndefinedColumnError when schema drifts."""
    if core.state._UPLOADS_COLS is not None:
        return core.state._UPLOADS_COLS
    async with pool.acquire() as conn:
        return await ensure_uploads_columns_loaded(conn)

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


def coerce_hashtag_list(val: object) -> list[str]:
    """
    Flatten hashtag-ish values from DB/API (arrays, JSON strings, comma text)
    into raw string tokens (may still contain # or noise — pass through sanitize_hashtag_body).

    Also recovers individual tags from corrupted single-token forms that AI/legacy
    payloads have produced, e.g. ``'#"[\\"tester\\" #\\"qwe\\"]"'`` →
    ``["tester", "qwe"]`` instead of one ugly mash-up. This makes always_hashtags
    and any other hashtag column self-healing across the pipeline.
    """
    out: list[str] = []
    if val is None:
        return out
    if isinstance(val, dict):
        return out
    if isinstance(val, (list, tuple)):
        for item in val:
            out.extend(coerce_hashtag_list(item))
        return out
    s = str(val).strip()
    if not s:
        return out
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return coerce_hashtag_list(parsed)
        if isinstance(parsed, str):
            return coerce_hashtag_list(parsed)
        if isinstance(parsed, dict):
            return out
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        try:
            fixed = re.sub(r'"\s+"', '", "', s)
            fixed = re.sub(r"'\s+'", "', '", fixed)
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return coerce_hashtag_list(parsed)
        except Exception:
            quoted = re.findall(r'"([^"]{1,200})"', s)
            if quoted:
                return coerce_hashtag_list(quoted)
    # Self-heal AI/legacy blob like '#"[\"tester\" #\"qwe\"]"' that DB/AI may have
    # serialized into a single token. Pull out escaped/quoted words so each tag is
    # routable through sanitize_hashtag_body individually.
    if "[" in s and "]" in s and ('"' in s or "\\" in s):
        inner = re.findall(r'\\?"([^"\\\[\]]{1,80})\\?"', s)
        cleaned_inner = [t.strip() for t in inner if t and t.strip()]
        if cleaned_inner:
            return cleaned_inner
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def sanitize_hashtag_body(raw: str | None, *, max_len: int = 50) -> str:
    """
    Reduce a user/AI string to one safe hashtag token without leading '#'.

    Strips JSON/markup noise (brackets, quotes, commas) so pasted values like
    '#"[\"tester\"' become 'tester'. Uses Unicode-aware word characters so
    international tags still work. Returns '' if nothing usable remains.
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower().lstrip("#")
    if not s:
        return ""
    s = re.sub(r"[^\w]", "", s, flags=re.UNICODE)
    if not s:
        return ""
    return s[:max_len] if max_len > 0 else s


# AI sometimes pastes a JSON-ish hashtag array into prose; strip from captions/descriptions.
_STRAY_HASH_JSON_BLOB = re.compile(r'#\s*"\s*\[(?:\\"|[^\]])*?\]\s*"', re.DOTALL)
_STRAY_HASH_JSON_BRACKET = re.compile(r'#\s*\[(?:\\.|[^\]]){0,800}?\]', re.DOTALL)


def strip_stray_hashtag_json_blob(text: str) -> str:
    """Remove serialized JSON-array hashtag junk from caption/description text."""
    if not text:
        return ""
    cleaned = _STRAY_HASH_JSON_BLOB.sub(" ", text)
    cleaned = _STRAY_HASH_JSON_BRACKET.sub(" ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def coerce_jsonb_list(val: object) -> list:
    """Defensive read for a JSONB-list column.

    Worker pools historically lacked the JSONB codec and a few prior write paths
    double-encoded values, so columns that should be lists sometimes come back as
    JSON strings (e.g. ``'["tester","qwe"]'``). Parse them back into a real list
    so downstream code can index/iterate normally. Lists pass through untouched.
    """
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                # Double-encoded — peel one more layer.
                try:
                    inner = json.loads(parsed)
                    if isinstance(inner, list):
                        return inner
                except Exception:
                    pass
                return [parsed] if parsed else []
        except Exception:
            return [s]
    return []


def coerce_jsonb_dict(val: object, *, default: dict | None = None) -> dict:
    """Defensive read for a JSONB-object column. Mirrors :func:`coerce_jsonb_list`.

    JSON-string values are parsed back to a dict; non-dict / unparseable inputs
    return ``default`` (or ``{}``). Used for ``platform_hashtags`` /
    ``users.preferences`` / similar JSONB blobs that may arrive as text when a
    pool is missing the JSONB codec or the original write double-encoded.
    """
    fallback: dict = dict(default) if isinstance(default, dict) else {}
    if val is None:
        return fallback
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return fallback
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                try:
                    inner = json.loads(parsed)
                    if isinstance(inner, dict):
                        return inner
                except Exception:
                    pass
        except Exception:
            return fallback
    return fallback


def coerce_processed_assets_map(val: object) -> dict[str, str]:
    """Normalize ``uploads.processed_assets`` JSONB to ``{platform: r2_key}``."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return {
            str(k): str(v)
            for k, v in val.items()
            if v is not None and str(v).strip()
        }
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return {}
        try:
            return coerce_processed_assets_map(json.loads(s))
        except Exception:
            return {}
    if isinstance(val, list):
        out: dict[str, str] = {}
        for item in val:
            if not isinstance(item, dict):
                continue
            plat = item.get("platform") or item.get("name")
            key = item.get("r2_key") or item.get("key") or item.get("path")
            if plat and key:
                out[str(plat)] = str(key)
        return out
    return {}


def platform_hashtag_map_has_any_tags(m: object) -> bool:
    """True if a platform->tags dict has at least one non-empty tag list."""
    if not isinstance(m, dict):
        return False
    for v in m.values():
        if isinstance(v, list) and len(v) > 0:
            return True
    return False


def merge_platform_hashtag_overlay(base: object, incoming: object) -> dict[str, list]:
    """
    Merge ``users.preferences`` platform hashtag map onto DB (user_preferences) map.

    Used by the worker when loading settings: ``users.preferences`` often contains
    ``platformHashtags: {}`` from partial client payloads. Assigning that dict used
    to wipe per-platform lists that only existed in ``user_preferences.platform_hashtags``.

    Rules:
    - Keys are normalized to lowercase for stable lookup in ``get_effective_hashtags``.
    - If ``incoming`` is empty, return the base map unchanged.
    - Otherwise ``{**base, **incoming}`` so explicit ``[]`` on a platform still clears it.
    """
    def _as_lc_map(d: object) -> dict[str, list]:
        raw = _safe_json(d, {}) if isinstance(d, str) else d
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list] = {}
        for k, v in raw.items():
            nk = str(k).strip().lower()
            if isinstance(v, list):
                out[nk] = [str(x).strip() for x in v if str(x).strip()]
            elif v is None:
                out[nk] = []
            else:
                s = str(v).strip()
                out[nk] = [s] if s else []
        return out

    b = _as_lc_map(base)
    inc = _as_lc_map(incoming)
    if not inc:
        return b
    return {**b, **inc}


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
