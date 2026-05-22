"""
DB-backed overrides for public tier limits and top-up SKUs (merged with code defaults).

Singleton row ``billing_catalog`` (id=1). Cached in ``core.state.billing_catalog_cache``.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, FrozenSet, Optional, Set

import asyncpg

import core.state
from stages.entitlements import PUBLIC_TIER_SLUGS, TIER_CONFIG, TOPUP_PRODUCTS
from stages.ai_service_costs import SERVICE_WEIGHTS

logger = logging.getLogger("uploadm8-api")

_MERGEABLE_TIER_KEYS = frozenset(
    {
        "name",
        "price",
        "price_annual",
        "put_daily",
        "put_monthly",
        "aic_monthly",
        "max_accounts",
        "max_accounts_per_platform",
        "per_platform",
        "watermark",
        "ads",
        "ai",
        "scheduling",
        "webhooks",
        "white_label",
        "excel",
        "flex",
        "priority_class",
        "queue_depth",
        "lookahead_hours",
        "max_thumbnails",
        "ai_depth",
        "max_caption_frames",
        "caption_frames",
        "max_parallel_uploads",
        "parallel_uploads",
        "custom_thumbnails",
        "ai_thumbnail_styling",
        "team_seats",
        "analytics",
        "trial_days",
        "internal",
    }
)

_INT_TIER_KEYS = frozenset(
    {
        "put_daily",
        "put_monthly",
        "aic_monthly",
        "max_accounts",
        "max_accounts_per_platform",
        "per_platform",
        "queue_depth",
        "lookahead_hours",
        "trial_days",
        "team_seats",
        "max_thumbnails",
        "max_caption_frames",
        "caption_frames",
        "max_parallel_uploads",
        "parallel_uploads",
    }
)
_FLOAT_TIER_KEYS = frozenset({"price", "price_annual"})
_BOOL_TIER_KEYS = frozenset(
    {
        "watermark",
        "ads",
        "ai",
        "scheduling",
        "webhooks",
        "white_label",
        "excel",
        "flex",
        "custom_thumbnails",
        "ai_thumbnail_styling",
        "internal",
    }
)

_TOPUP_MERGE_KEYS = frozenset({"wallet", "amount", "put", "aic", "price", "price_usd"})

PUT_COST_DEFAULTS: Dict[str, int] = {
    "base": 10,
    "priority_lane_addon": 5,
    "per_extra_platform": 2,
    "per_extra_thumbnail_beyond_first": 1,
}
_PUT_COST_KEYS = frozenset(PUT_COST_DEFAULTS.keys())

_INTERNAL_FULL_SERVICE_TIERS = frozenset(
    {"master_admin", "friends_family", "lifetime"}
)


def validate_tier_service_overrides(raw: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    """Sparse per-tier pipeline service allow/deny overrides."""
    if not isinstance(raw, dict):
        raise ValueError("tier_service_overrides must be an object")
    out: Dict[str, Dict[str, bool]] = {}
    for tier_slug, patch in raw.items():
        slug = str(tier_slug or "").strip()
        if slug not in PUBLIC_TIER_SLUGS:
            raise ValueError(f"Unknown tier slug in tier_service_overrides: {slug!r}")
        if not isinstance(patch, dict):
            raise ValueError(f"tier_service_overrides[{slug}] must be an object")
        tier_out: Dict[str, bool] = {}
        for sid, val in patch.items():
            service_id = str(sid or "").strip()
            if service_id not in SERVICE_WEIGHTS:
                raise ValueError(f"Unknown service id {service_id!r} for tier {slug}")
            if isinstance(val, bool):
                tier_out[service_id] = val
            else:
                tier_out[service_id] = str(val).strip().lower() in ("1", "true", "yes", "on")
        if tier_out:
            out[slug] = tier_out
    return out


def merge_tier_service_overrides(
    overrides: Optional[Dict[str, Dict[str, bool]]],
) -> Dict[str, Dict[str, bool]]:
    return validate_tier_service_overrides(overrides or {})


def effective_tier_allowed_services(tier_slug: str) -> FrozenSet[str]:
    """
    Pipeline services a tier may use. Backward compatible: when ``ai`` is enabled,
    all ``SERVICE_WEIGHTS`` keys are allowed unless explicitly disabled in DB overrides.
    Internal tiers always receive the full service set.
    """
    slug = str(tier_slug or "free").strip().lower()
    if slug in _INTERNAL_FULL_SERVICE_TIERS:
        return frozenset(SERVICE_WEIGHTS.keys())

    tier_cfg = effective_tier_config().get(slug) or TIER_CONFIG.get(slug) or TIER_CONFIG["free"]
    if not tier_cfg.get("ai"):
        return frozenset()

    allowed: Set[str] = set(SERVICE_WEIGHTS.keys())
    overrides = core.state.billing_catalog_cache.get("tier_service_overrides") or {}
    tier_patch = overrides.get(slug) or {}
    if isinstance(tier_patch, dict):
        for sid, enabled in tier_patch.items():
            service_id = str(sid or "").strip()
            if service_id not in SERVICE_WEIGHTS:
                continue
            if enabled is False:
                allowed.discard(service_id)
            elif enabled is True:
                allowed.add(service_id)
    return frozenset(allowed)


def effective_tier_services_matrix() -> Dict[str, Dict[str, bool]]:
    """Effective allow/deny for each public tier × pipeline service (admin matrix UI)."""
    from stages.ai_service_costs import service_catalog

    rows = service_catalog()
    service_ids = [r["id"] for r in rows]
    matrix: Dict[str, Dict[str, bool]] = {}
    for slug in PUBLIC_TIER_SLUGS:
        allowed = effective_tier_allowed_services(slug)
        matrix[slug] = {sid: sid in allowed for sid in service_ids}
    return matrix


def merge_put_cost_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Code defaults overlaid with ``billing_catalog.put_cost_overrides``."""
    out = dict(PUT_COST_DEFAULTS)
    if not overrides:
        return out
    for k in _PUT_COST_KEYS:
        if k not in overrides:
            continue
        try:
            out[k] = max(0, min(5000, int(overrides[k])))
        except (TypeError, ValueError):
            continue
    return out


def effective_put_cost_rules() -> Dict[str, int]:
    bill = core.state.billing_catalog_cache.get("put_cost_overrides") or {}
    return merge_put_cost_overrides(bill)


def validate_put_cost_overrides(raw: Dict[str, Any]) -> Dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError("put_cost_overrides must be an object")
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if k not in _PUT_COST_KEYS:
            raise ValueError(f"Unknown put_cost key: {k!r}")
        out[k] = max(0, min(5000, int(v)))
    return out


def _coerce_tier_value(key: str, val: Any) -> Any:
    if key in _BOOL_TIER_KEYS:
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("1", "true", "yes", "on")
    if key in _INT_TIER_KEYS:
        return int(val)
    if key in _FLOAT_TIER_KEYS:
        return float(val)
    if key in ("name", "priority_class", "ai_depth", "analytics"):
        return str(val)
    return val


def _merge_tier_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if k not in _MERGEABLE_TIER_KEYS:
            continue
        try:
            out[k] = _coerce_tier_value(k, v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid tier override {k}={v!r}: {e}") from e
    return out


def merge_tier_config_with_overrides(
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Deep-copy ``TIER_CONFIG`` and apply per-slug shallow patches."""
    ovr = overrides or {}
    out = copy.deepcopy(TIER_CONFIG)
    for slug, patch in ovr.items():
        if slug not in out:
            continue
        if not isinstance(patch, dict):
            raise ValueError(f"tier_overrides[{slug}] must be an object")
        out[slug] = _merge_tier_dict(out[slug], patch)
    return out


def _merge_topup_entry(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in patch.items():
        if k not in _TOPUP_MERGE_KEYS:
            continue
        if k in ("amount", "put", "aic"):
            merged[k] = int(v)
        elif k in ("price", "price_usd"):
            merged[k] = float(v)
        elif k == "wallet":
            merged[k] = str(v).lower().strip()
        else:
            merged[k] = v
    return merged


def merge_topup_products_with_overrides(
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    ovr = overrides or {}
    out = copy.deepcopy(TOPUP_PRODUCTS)
    for lk, patch in ovr.items():
        if lk not in out:
            continue
        if not isinstance(patch, dict):
            raise ValueError(f"topup_overrides[{lk}] must be an object")
        out[lk] = _merge_topup_entry(out[lk], patch)
    return out


def effective_tier_config() -> Dict[str, Dict[str, Any]]:
    """``TIER_CONFIG`` ← ``catalog_products`` overlay ← ``billing_catalog`` tier_overrides."""
    cat = core.state.catalog_pricing_cache.get("tier_overlay") or {}
    bill = core.state.billing_catalog_cache.get("tier_overrides") or {}
    merged = copy.deepcopy(TIER_CONFIG)
    for slug, patch in cat.items():
        if slug in merged and isinstance(patch, dict):
            merged[slug] = _merge_tier_dict(merged[slug], patch)
    for slug, patch in bill.items():
        if slug in merged and isinstance(patch, dict):
            merged[slug] = _merge_tier_dict(merged[slug], patch)
    return merged


def tier_config_before_billing_overrides() -> Dict[str, Dict[str, Any]]:
    """``TIER_CONFIG`` merged with catalog overlay only (no ``billing_catalog`` tier_overrides)."""
    cat = core.state.catalog_pricing_cache.get("tier_overlay") or {}
    merged = copy.deepcopy(TIER_CONFIG)
    for slug, patch in cat.items():
        if slug in merged and isinstance(patch, dict):
            merged[slug] = _merge_tier_dict(merged[slug], patch)
    return merged


def effective_topup_products() -> Dict[str, Dict[str, Any]]:
    """``TOPUP_PRODUCTS`` ← ``catalog_products`` overlay ← ``billing_catalog`` topup_overrides."""
    cat = core.state.catalog_pricing_cache.get("topup_overlay") or {}
    bill = core.state.billing_catalog_cache.get("topup_overrides") or {}
    merged = copy.deepcopy(TOPUP_PRODUCTS)
    for lk, patch in cat.items():
        if lk in merged and isinstance(patch, dict):
            merged[lk] = _merge_topup_entry(merged[lk], patch)
    for lk, patch in bill.items():
        if lk in merged and isinstance(patch, dict):
            merged[lk] = _merge_topup_entry(merged[lk], patch)
    return merged


def topup_products_before_billing_overrides() -> Dict[str, Dict[str, Any]]:
    """``TOPUP_PRODUCTS`` merged with catalog overlay only (no billing DB topup_overrides)."""
    cat = core.state.catalog_pricing_cache.get("topup_overlay") or {}
    merged = copy.deepcopy(TOPUP_PRODUCTS)
    for lk, patch in cat.items():
        if lk in merged and isinstance(patch, dict):
            merged[lk] = _merge_topup_entry(merged[lk], patch)
    return merged


def validate_catalog_put(
    tier_overrides: Dict[str, Any],
    topup_overrides: Dict[str, Any],
) -> None:
    if not isinstance(tier_overrides, dict) or not isinstance(topup_overrides, dict):
        raise ValueError("tier_overrides and topup_overrides must be objects")
    for slug, patch in tier_overrides.items():
        if slug not in TIER_CONFIG:
            raise ValueError(f"Unknown tier slug in tier_overrides: {slug}")
        if not isinstance(patch, dict):
            raise ValueError(f"tier_overrides[{slug}] must be an object")
        for k in patch:
            if k not in _MERGEABLE_TIER_KEYS:
                raise ValueError(f"Unknown tier field {k!r} for {slug}")
        _merge_tier_dict(TIER_CONFIG[slug], patch)  # dry-run coerce
    for lk, patch in topup_overrides.items():
        if lk not in TOPUP_PRODUCTS:
            raise ValueError(f"Unknown topup lookup_key in topup_overrides: {lk}")
        if not isinstance(patch, dict):
            raise ValueError(f"topup_overrides[{lk}] must be an object")
        for k in patch:
            if k not in _TOPUP_MERGE_KEYS:
                raise ValueError(f"Unknown topup field {k!r} for {lk}")
        _merge_topup_entry(TOPUP_PRODUCTS[lk], patch)


async def fetch_catalog_row(conn: asyncpg.Connection) -> Optional[asyncpg.Record]:
    return await conn.fetchrow(
        """
        SELECT tier_overrides, topup_overrides, put_cost_overrides, tier_service_overrides,
               updated_at, updated_by,
               last_sync_at, last_sync_ok, last_sync_error, last_sync_detail
        FROM billing_catalog WHERE id = 1
        """
    )


async def load_billing_catalog_cache(conn: asyncpg.Connection) -> None:
    row = await fetch_catalog_row(conn)
    if not row:
        core.state.billing_catalog_cache["tier_overrides"] = {}
        core.state.billing_catalog_cache["topup_overrides"] = {}
        core.state.billing_catalog_cache["put_cost_overrides"] = {}
        core.state.billing_catalog_cache["tier_service_overrides"] = {}
        core.state.billing_catalog_cache["last_sync_at"] = None
        core.state.billing_catalog_cache["last_sync_ok"] = None
        core.state.billing_catalog_cache["last_sync_error"] = None
        core.state.billing_catalog_cache["last_sync_detail"] = None
        return
    core.state.billing_catalog_cache["tier_overrides"] = dict(row["tier_overrides"] or {})
    core.state.billing_catalog_cache["topup_overrides"] = dict(row["topup_overrides"] or {})
    core.state.billing_catalog_cache["put_cost_overrides"] = dict(row.get("put_cost_overrides") or {})
    core.state.billing_catalog_cache["tier_service_overrides"] = dict(row.get("tier_service_overrides") or {})
    core.state.billing_catalog_cache["last_sync_at"] = row["last_sync_at"]
    core.state.billing_catalog_cache["last_sync_ok"] = row["last_sync_ok"]
    core.state.billing_catalog_cache["last_sync_error"] = row["last_sync_error"]
    core.state.billing_catalog_cache["last_sync_detail"] = row["last_sync_detail"]


async def save_put_cost_overrides(
    conn: asyncpg.Connection,
    put_cost_overrides: Dict[str, Any],
    updated_by: Optional[str],
) -> None:
    validated = validate_put_cost_overrides(put_cost_overrides)
    await conn.execute(
        """
        UPDATE billing_catalog SET
            put_cost_overrides = $1::jsonb,
            updated_at = NOW(),
            updated_by = $2::uuid
        WHERE id = 1
        """,
        validated,
        updated_by if _valid_uuid(updated_by) else None,
    )
    core.state.billing_catalog_cache["put_cost_overrides"] = dict(validated)


async def save_catalog_overrides(
    conn: asyncpg.Connection,
    tier_overrides: Dict[str, Any],
    topup_overrides: Dict[str, Any],
    updated_by: Optional[str],
    tier_service_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    validate_catalog_put(tier_overrides, topup_overrides)
    svc_overrides = validate_tier_service_overrides(tier_service_overrides or {})
    await conn.execute(
        """
        INSERT INTO billing_catalog (id, tier_overrides, topup_overrides, tier_service_overrides, updated_at, updated_by)
        VALUES (1, $1::jsonb, $2::jsonb, $3::jsonb, NOW(), $4::uuid)
        ON CONFLICT (id) DO UPDATE SET
            tier_overrides = EXCLUDED.tier_overrides,
            topup_overrides = EXCLUDED.topup_overrides,
            tier_service_overrides = EXCLUDED.tier_service_overrides,
            updated_at = NOW(),
            updated_by = EXCLUDED.updated_by
        """,
        tier_overrides,
        topup_overrides,
        svc_overrides,
        updated_by if _valid_uuid(updated_by) else None,
    )
    core.state.billing_catalog_cache["tier_overrides"] = dict(tier_overrides)
    core.state.billing_catalog_cache["topup_overrides"] = dict(topup_overrides)
    core.state.billing_catalog_cache["tier_service_overrides"] = dict(svc_overrides)


def _valid_uuid(s: Optional[str]) -> bool:
    if not s:
        return False
    try:
        import uuid as _uuid

        _uuid.UUID(str(s))
        return True
    except Exception:
        return False


async def update_sync_status(
    conn: asyncpg.Connection,
    ok: bool,
    error: Optional[str],
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    await conn.execute(
        """
        UPDATE billing_catalog SET
            last_sync_at = NOW(),
            last_sync_ok = $1,
            last_sync_error = $2,
            last_sync_detail = $3::jsonb
        WHERE id = 1
        """,
        ok,
        error,
        json.dumps(detail or {}),
    )
