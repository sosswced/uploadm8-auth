"""
Master admin catalog management endpoints.

Flow when admin edits a product:
  1. PATCH /api/admin/catalog/products/{lookup_key}   -- updates DB row
  2. POST  /api/admin/catalog/products/{lookup_key}/sync  -- pushes to Stripe
     (auto-called from PATCH unless ?auto_sync=false)

Auto-sync behavior:
  - DB save fires the sync immediately
  - Sync regenerates card, uploads to Stripe Files, updates product,
    creates/migrates Price, mirrors card to R2 and local frontend dir
  - On any sync failure, DB save is preserved but flagged for retry
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import CATALOG_PRICING_APPROVAL_BYPASS, CATALOG_PRICING_APPROVAL_REQUIRED
import core.state
from core.deps import get_verified_user_id, require_master_admin_on_conn

logger = logging.getLogger("uploadm8.admin_catalog")

router = APIRouter(prefix="/api/admin/catalog", tags=["admin", "catalog"])


# =============================================================
# Pydantic models
# =============================================================


class CatalogProductRead(BaseModel):
    lookup_key: str
    stripe_product_id: Optional[str]
    product_kind: str
    tier_slug: Optional[str]
    display_name: str
    stripe_product_name: str
    statement_descriptor: str
    price_usd: float
    price_usd_yearly: Optional[float]
    wallet: Optional[str]
    token_amount: Optional[int]
    put_daily: Optional[int] = None
    put_monthly: Optional[int] = None
    aic_monthly: Optional[int] = None
    max_accounts: Optional[int] = None
    max_accounts_per_platform: Optional[int] = None
    queue_depth: Optional[int] = None
    lookahead_hours: Optional[int] = None
    trial_days: Optional[int] = None
    team_seats: Optional[int] = None
    watermark: Optional[bool] = None
    ads: Optional[bool] = None
    webhooks: Optional[bool] = None
    white_label: Optional[bool] = None
    hud: Optional[bool] = None
    excel: Optional[bool] = None
    flex: Optional[bool] = None
    priority_class: Optional[str] = None
    ai_depth: Optional[str] = None
    analytics: Optional[str] = None
    image_filename: str
    image_url: Optional[str]
    image_hash: Optional[str]
    last_synced_at: Optional[str]
    last_synced_by: Optional[str]
    is_archived: bool


_CATALOG_PRODUCT_SELECT = """
    lookup_key, stripe_product_id, product_kind, tier_slug,
    display_name, stripe_product_name, statement_descriptor,
    price_usd::float AS price_usd,
    price_usd_yearly::float AS price_usd_yearly,
    wallet, token_amount,
    put_daily, put_monthly, aic_monthly, max_accounts, max_accounts_per_platform,
    queue_depth, lookahead_hours, trial_days, team_seats,
    watermark, ads, webhooks, white_label, hud, excel, flex,
    priority_class, ai_depth, analytics,
    image_filename, image_url, image_hash,
    last_synced_at::text, last_synced_by, is_archived
"""


class CatalogProductCreate(BaseModel):
    lookup_key: str = Field(..., min_length=8, max_length=120, pattern=r"^uploadm8_[a-z0-9_]+$")
    product_kind: str = Field(..., pattern=r"^(subscription|topup_put|topup_aic|topup_bundle)$")
    display_name: str = Field(..., min_length=2, max_length=120)
    stripe_product_name: Optional[str] = Field(None, max_length=200)
    price_usd: float = Field(..., ge=0)
    price_usd_yearly: Optional[float] = Field(None, ge=0)
    tier_slug: Optional[str] = None
    wallet: Optional[str] = Field(None, pattern=r"^(put|aic|bundle)$")
    token_amount: Optional[int] = Field(None, ge=1)
    put_monthly: Optional[int] = Field(None, ge=0)
    aic_monthly: Optional[int] = Field(None, ge=0)
    sort_order: Optional[int] = Field(None, ge=0, le=9999)
    stripe_product_id: Optional[str] = None


class CatalogProductPatch(BaseModel):
    """Only the fields a master admin is permitted to edit live."""

    display_name: Optional[str] = None
    stripe_product_name: Optional[str] = None
    statement_descriptor: Optional[str] = None
    price_usd: Optional[float] = Field(None, ge=0)
    price_usd_yearly: Optional[float] = Field(None, ge=0)
    put_monthly: Optional[int] = Field(None, ge=0)
    aic_monthly: Optional[int] = Field(None, ge=0)
    put_daily: Optional[int] = Field(None, ge=0)
    max_accounts: Optional[int] = Field(None, ge=0)
    max_accounts_per_platform: Optional[int] = Field(None, ge=0)
    queue_depth: Optional[int] = Field(None, ge=0)
    lookahead_hours: Optional[int] = Field(None, ge=0)
    trial_days: Optional[int] = Field(None, ge=0)
    team_seats: Optional[int] = Field(None, ge=1)
    watermark: Optional[bool] = None
    ads: Optional[bool] = None
    webhooks: Optional[bool] = None
    white_label: Optional[bool] = None
    hud: Optional[bool] = None
    excel: Optional[bool] = None
    flex: Optional[bool] = None
    priority_class: Optional[str] = Field(None, pattern=r"^p[0-4]$")
    ai_depth: Optional[str] = Field(None, pattern=r"^(basic|enhanced|advanced|max)$")
    analytics: Optional[str] = None
    token_amount: Optional[int] = Field(None, ge=1)
    is_archived: Optional[bool] = None


class PricingRequestCreate(BaseModel):
    lookup_key: str
    proposed_patch: Dict[str, Any] = Field(default_factory=dict)


class PricingRequestResolve(BaseModel):
    status: str = Field(..., pattern=r"^(approved|rejected)$")
    resolution_notes: Optional[str] = None


MONEY_PATCH_KEYS = frozenset({"price_usd", "price_usd_yearly"})
_PATCHABLE_KEYS = frozenset(CatalogProductPatch.model_fields.keys())


def _actor(user: dict) -> str:
    em = user.get("email")
    if em:
        return f"admin:{em}"
    return f"admin:{user.get('id', 'unknown')}"


async def _apply_catalog_product_updates(
    conn: asyncpg.Connection,
    lookup_key: str,
    updates: Dict[str, Any],
    *,
    actor: str,
    log_note: str = "patch",
) -> None:
    cols: List[str] = []
    vals: List[Any] = []
    for i, (k, v) in enumerate(updates.items(), start=1):
        cols.append(f"{k} = ${i}")
        vals.append(v)
    vals.append(lookup_key)
    sql = f"""
        UPDATE catalog_products
        SET {", ".join(cols)}, updated_at = NOW()
        WHERE lookup_key = ${len(vals)}
        RETURNING id
    """
    row = await conn.fetchrow(sql, *vals)
    if not row:
        raise HTTPException(404, f"No product with lookup_key={lookup_key}")
    await conn.execute(
        """
        INSERT INTO catalog_sync_log
            (catalog_product_id, lookup_key, operation, status, stripe_response, actor)
        VALUES ($1, $2, 'db_update', 'success', $3::jsonb, $4)
        """,
        row["id"],
        lookup_key,
        json.dumps({"patched_fields": list(updates.keys()), "note": log_note}),
        actor,
    )


async def _reload_catalog_cache_from_conn(conn) -> None:
    from services.catalog_publish import refresh_pricing_caches_after_catalog_sync

    await refresh_pricing_caches_after_catalog_sync(conn)


# =============================================================
# READ
# =============================================================


@router.get("/products", response_model=List[CatalogProductRead])
async def list_catalog_products(
    include_archived: bool = False,
    include_internal: bool = False,
    user_id: str = Depends(get_verified_user_id),
):
    """Return all catalog rows, sorted by display order."""
    async with core.state.require_pool().acquire() as conn:
        await require_master_admin_on_conn(conn, user_id)
        where = []
        if not include_archived:
            where.append("is_archived = FALSE")
        if not include_internal:
            where.append("is_internal = FALSE")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = await conn.fetch(
            f"""
            SELECT {_CATALOG_PRODUCT_SELECT}
            FROM catalog_products
            {clause}
            ORDER BY sort_order
            """
        )
        return [CatalogProductRead(**dict(r)) for r in rows]


@router.get("/products/{lookup_key}", response_model=CatalogProductRead)
async def get_catalog_product(
    lookup_key: str,
    user_id: str = Depends(get_verified_user_id),
):
    async with core.state.require_pool().acquire() as conn:
        await require_master_admin_on_conn(conn, user_id)
        row = await conn.fetchrow(
            f"""
            SELECT {_CATALOG_PRODUCT_SELECT}
            FROM catalog_products WHERE lookup_key = $1
            """,
            lookup_key,
        )
        if not row:
            raise HTTPException(404, f"No product with lookup_key={lookup_key}")
        return CatalogProductRead(**dict(row))


@router.post("/products")
async def create_catalog_product(
    body: CatalogProductCreate,
    auto_sync: bool = Query(True, description="Push new product to Stripe immediately"),
    user_id: str = Depends(get_verified_user_id),
):
    """Insert a new catalog row (subscription or top-up SKU)."""
    lk = body.lookup_key.strip()
    kind = body.product_kind.strip()
    display = body.display_name.strip()
    stripe_name = (body.stripe_product_name or display).strip()
    stmt = "".join(c for c in stripe_name.upper() if c.isalnum())[:22] or "UPLOADM8"
    unit = "subscription" if kind == "subscription" else "token"
    wallet = body.wallet
    if kind == "topup_put":
        wallet = wallet or "put"
    elif kind == "topup_aic":
        wallet = wallet or "aic"
    elif kind == "topup_bundle":
        wallet = wallet or "bundle"
    image_filename = f"{lk.replace('uploadm8_', '')}.png"
    if kind == "topup_bundle":
        if body.put_monthly is None or body.aic_monthly is None:
            raise HTTPException(400, "topup_bundle requires put_monthly and aic_monthly")
        image_filename = f"topup_bundle_{body.put_monthly}_{body.aic_monthly}.png"
    elif kind.startswith("topup"):
        image_filename = f"topup_{wallet or 'put'}_{body.token_amount or 0}.png"

    async with core.state.require_pool().acquire() as conn:
        user = await require_master_admin_on_conn(conn, user_id)
        actor = _actor(user)
        exists = await conn.fetchval(
            "SELECT 1 FROM catalog_products WHERE lookup_key = $1", lk
        )
        if exists:
            raise HTTPException(409, f"lookup_key already exists: {lk}")
        sort_order = body.sort_order
        if sort_order is None:
            sort_order = int(
                await conn.fetchval(
                    "SELECT COALESCE(MAX(sort_order), 0) + 10 FROM catalog_products"
                )
                or 100
            )
        row = await conn.fetchrow(
            """
            INSERT INTO catalog_products (
                lookup_key, stripe_product_id, product_kind, tier_slug, sort_order,
                display_name, stripe_product_name, statement_descriptor, unit_label,
                price_usd, price_usd_yearly, wallet, token_amount,
                put_monthly, aic_monthly, image_filename
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12, $13,
                $14, $15, $16
            )
            RETURNING lookup_key
            """,
            lk,
            body.stripe_product_id,
            kind,
            body.tier_slug,
            sort_order,
            display,
            stripe_name,
            stmt,
            unit,
            body.price_usd,
            body.price_usd_yearly,
            wallet,
            body.token_amount,
            body.put_monthly,
            body.aic_monthly,
            image_filename,
        )
        pid = await conn.fetchval(
            "SELECT id FROM catalog_products WHERE lookup_key = $1", lk
        )
        await conn.execute(
            """
            INSERT INTO catalog_sync_log
                (catalog_product_id, lookup_key, operation, status, stripe_response, actor)
            VALUES ($1, $2, 'db_create', 'success', $3::jsonb, $4)
            """,
            pid,
            lk,
            json.dumps({"product_kind": kind, "display_name": display}),
            actor,
        )
        await _reload_catalog_cache_from_conn(conn)

        sync_result = None
        if auto_sync:
            try:
                from scripts.sync_stripe_catalog import sync_product

                sync_result = await sync_product(
                    conn, lookup_key=lk, actor=actor, dry_run=False
                )
            except Exception as exc:
                logger.exception("Auto-sync failed for new product %s", lk)
                return {
                    "created": True,
                    "lookup_key": lk,
                    "synced": False,
                    "error": str(exc),
                }

    return {
        "created": True,
        "lookup_key": lk,
        "synced": auto_sync,
        "sync_result": sync_result,
    }


# =============================================================
# PATCH (auto-syncs by default)
# =============================================================


@router.patch("/products/{lookup_key}")
async def update_catalog_product(
    lookup_key: str,
    patch: CatalogProductPatch,
    auto_sync: bool = Query(True, description="Push changes to Stripe + R2 immediately"),
    user_id: str = Depends(get_verified_user_id),
):
    """Update one catalog row. By default auto-syncs to Stripe + regen card."""
    updates = patch.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "No fields to update")

    if CATALOG_PRICING_APPROVAL_REQUIRED and not CATALOG_PRICING_APPROVAL_BYPASS:
        blocked = MONEY_PATCH_KEYS & set(updates.keys())
        if blocked:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "pricing_approval_required",
                    "message": (
                        "Money fields require an approved pricing request. "
                        "POST /api/admin/catalog/pricing-requests with proposed_patch, "
                        "then POST .../pricing-requests/{id}/resolve with status approved."
                    ),
                    "blocked_fields": sorted(blocked),
                    "lookup_key": lookup_key,
                },
            )

    async with core.state.require_pool().acquire() as conn:
        user = await require_master_admin_on_conn(conn, user_id)
        actor = _actor(user)
        cols = []
        vals: List[Any] = []
        for i, (k, v) in enumerate(updates.items(), start=1):
            cols.append(f"{k} = ${i}")
            vals.append(v)
        vals.append(lookup_key)
        sql = f"""
            UPDATE catalog_products
            SET {", ".join(cols)}, updated_at = NOW()
            WHERE lookup_key = ${len(vals)}
            RETURNING id, lookup_key
        """
        row = await conn.fetchrow(sql, *vals)
        if not row:
            raise HTTPException(404, f"No product with lookup_key={lookup_key}")

        await conn.execute(
            """
            INSERT INTO catalog_sync_log
                (catalog_product_id, lookup_key, operation, status, stripe_response, actor)
            VALUES ($1, $2, 'db_update', 'success', $3::jsonb, $4)
            """,
            row["id"],
            lookup_key,
            json.dumps({"patched_fields": list(updates.keys())}),
            actor,
        )

        await _reload_catalog_cache_from_conn(conn)

        sync_result = None
        if auto_sync:
            try:
                from scripts.sync_stripe_catalog import sync_product

                sync_result = await sync_product(
                    conn, lookup_key=lookup_key, actor=actor, dry_run=False
                )
            except Exception as exc:
                logger.exception("Auto-sync failed for %s", lookup_key)
                return {
                    "saved": True,
                    "synced": False,
                    "error": str(exc),
                    "message": (
                        "Saved to DB but Stripe sync failed. "
                        "Use POST /sync to retry."
                    ),
                }

        return {
            "saved": True,
            "synced": auto_sync,
            "patched_fields": list(updates.keys()),
            "sync_result": sync_result,
        }


# =============================================================
# Manual sync
# =============================================================


@router.post("/products/{lookup_key}/sync")
async def sync_catalog_product(
    lookup_key: str,
    dry_run: bool = False,
    user_id: str = Depends(get_verified_user_id),
):
    """Force-sync one product to Stripe. Idempotent; safe to call repeatedly."""
    async with core.state.require_pool().acquire() as conn:
        user = await require_master_admin_on_conn(conn, user_id)
        actor = _actor(user)
        try:
            from scripts.sync_stripe_catalog import sync_product

            return await sync_product(conn, lookup_key=lookup_key, actor=actor, dry_run=dry_run)
        except Exception as exc:
            logger.exception("Manual sync failed for %s", lookup_key)
            raise HTTPException(500, f"Sync failed: {exc}") from exc


@router.post("/sync-all")
async def sync_all_catalog(
    dry_run: bool = False,
    user_id: str = Depends(get_verified_user_id),
):
    """Force-sync everything. Use sparingly; the legacy PowerShell entry point."""
    async with core.state.require_pool().acquire() as conn:
        user = await require_master_admin_on_conn(conn, user_id)
        actor = _actor(user)
        from scripts.sync_stripe_catalog import sync_all

        return await sync_all(conn, actor=actor, dry_run=dry_run)


# =============================================================
# Sync log
# =============================================================


@router.get("/sync-log")
async def get_sync_log(
    lookup_key: Optional[str] = None,
    limit: int = Query(100, le=500),
    user_id: str = Depends(get_verified_user_id),
):
    async with core.state.require_pool().acquire() as conn:
        await require_master_admin_on_conn(conn, user_id)
        if lookup_key:
            rows = await conn.fetch(
                """
                SELECT * FROM catalog_sync_log
                WHERE lookup_key = $1
                ORDER BY created_at DESC LIMIT $2
                """,
                lookup_key,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT * FROM catalog_sync_log
                ORDER BY created_at DESC LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]


# =============================================================
# Ops: catalog pricing change requests (review queue)
# =============================================================


@router.get("/pricing-requests")
async def list_pricing_requests(
    status: Optional[str] = Query(None, description="open|approved|rejected"),
    limit: int = Query(100, le=500),
    user_id: str = Depends(get_verified_user_id),
):
    async with core.state.require_pool().acquire() as conn:
        await require_master_admin_on_conn(conn, user_id)
        if status:
            rows = await conn.fetch(
                """
                SELECT id, lookup_key, status, proposed_patch, actor_email,
                       resolution_notes, resolved_by_email,
                       resolved_at::text, created_at::text
                FROM catalog_pricing_requests
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                status,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, lookup_key, status, proposed_patch, actor_email,
                       resolution_notes, resolved_by_email,
                       resolved_at::text, created_at::text
                FROM catalog_pricing_requests
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]


@router.post("/pricing-requests")
async def create_pricing_request(
    body: PricingRequestCreate,
    user_id: str = Depends(get_verified_user_id),
):
    async with core.state.require_pool().acquire() as conn:
        user = await require_master_admin_on_conn(conn, user_id)
        email = user.get("email")
        rid = await conn.fetchval(
            """
            INSERT INTO catalog_pricing_requests
                (lookup_key, status, proposed_patch, actor_email)
            VALUES ($1, 'open', $2::jsonb, $3)
            RETURNING id
            """,
            body.lookup_key,
            body.proposed_patch or {},
            str(email) if email else None,
        )
        return {"id": str(rid), "status": "open"}


@router.post("/pricing-requests/{request_id}/resolve")
async def resolve_pricing_request(
    request_id: str,
    body: PricingRequestResolve,
    user_id: str = Depends(get_verified_user_id),
):
    sync_result: Optional[Any] = None
    async with core.state.require_pool().acquire() as conn:
        user = await require_master_admin_on_conn(conn, user_id)
        email = user.get("email")
        actor = _actor(user)
        async with conn.transaction():
            req = await conn.fetchrow(
                """
                SELECT id, lookup_key, status, proposed_patch
                FROM catalog_pricing_requests
                WHERE id = $1::uuid
                FOR UPDATE
                """,
                request_id,
            )
            if not req:
                raise HTTPException(404, "Request not found")
            if req["status"] != "open":
                raise HTTPException(400, "Request is not open")

            if body.status == "approved":
                patch_raw = dict(req["proposed_patch"] or {})
                patch = {k: v for k, v in patch_raw.items() if k in _PATCHABLE_KEYS}
                if not patch:
                    raise HTTPException(
                        400,
                        "proposed_patch must include at least one valid catalog column",
                    )
                lk = req["lookup_key"]
                await _apply_catalog_product_updates(
                    conn, lk, patch, actor=actor, log_note="pricing_request_approved"
                )
                from scripts.sync_stripe_catalog import sync_product

                sync_result = await sync_product(
                    conn, lookup_key=lk, actor=actor, dry_run=False
                )
                await _reload_catalog_cache_from_conn(conn)

            await conn.execute(
                """
                UPDATE catalog_pricing_requests SET
                    status = $2,
                    resolution_notes = $3,
                    resolved_by_email = $4,
                    resolved_at = NOW()
                WHERE id = $1::uuid
                """,
                request_id,
                body.status,
                body.resolution_notes,
                str(email) if email else None,
            )

    out: Dict[str, Any] = {"ok": True, "id": request_id, "status": body.status}
    if body.status == "approved":
        out["sync_result"] = sync_result
    return out
