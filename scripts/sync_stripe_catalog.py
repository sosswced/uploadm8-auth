"""
scripts/sync_stripe_catalog.py

Master sync orchestrator. Reads catalog_products from Postgres and pushes
everything (descriptions, metadata, prices, card images) to Stripe.

This is the SINGLE entrypoint for any catalog change. Master admin edits
the DB row (via admin UI or direct UPDATE), then runs this script (or it
runs automatically via the admin endpoint).

Behavior per product:
  1. Render description from services.catalog_descriptions (with current DB values)
  2. Regenerate card PNG with current DB values (via stages.product_card_renderer)
  3. SHA-256 the card bytes; if hash unchanged, skip image upload
  4. If hash changed: upload to Stripe Files + upload to R2 (for site display)
  5. UPDATE Stripe product: name, description, statement_descriptor,
     unit_label, tax_code, metadata, images[0]
  6. Find existing Stripe Price by lookup_key; if missing or amount changed,
     create new Price with the same lookup_key (transfer_lookup_key=true)
  7. Log to catalog_sync_log

Usage (CLI):
    python -m scripts.sync_stripe_catalog --all
    python -m scripts.sync_stripe_catalog --lookup-key uploadm8_creatorpro_monthly
    python -m scripts.sync_stripe_catalog --dry-run --all

Usage (programmatic — from admin endpoint):
    from scripts.sync_stripe_catalog import sync_product
    result = await sync_product(conn, lookup_key="uploadm8_creatorpro_monthly",
                                 actor="admin:earl@uploadm8.com")
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx

from services.catalog_publish import refresh_pricing_caches_after_catalog_sync

logger = logging.getLogger("uploadm8.catalog_sync")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STRIPE_BASE = "https://api.stripe.com/v1"
STRIPE_FILES_BASE = "https://files.stripe.com/v1"

# ---- Env config ----
STRIPE_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE", "")
FRONTEND_IMAGES_DIR = Path(os.environ.get(
    "UPLOADM8_FRONTEND_IMAGES_DIR", "/var/www/uploadm8/images"
))
TAX_CODE_DIGITAL_SAAS = "txcd_10103001"


# =============================================================
# Stripe helpers
# =============================================================

class StripeError(Exception):
    pass


async def _stripe_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    body: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {STRIPE_KEY}"}
    url = f"{STRIPE_BASE}/{path}"
    if method == "GET":
        r = await client.get(url, headers=headers)
    else:
        r = await client.post(url, headers=headers, data=body or {})
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {}).get("message", r.text)
        except Exception:
            err = r.text
        raise StripeError(f"{method} {path} -> {r.status_code}: {err}")
    return r.json()


async def _find_price_by_lookup_key(
    client: httpx.AsyncClient, lookup_key: str
) -> Optional[Dict[str, Any]]:
    q = f"lookup_key:'{lookup_key}'"
    res = await _stripe_request(client, "GET", f"prices/search?query={httpx.QueryParams({'q': q}).get('q')}")
    data = res.get("data") or []
    return data[0] if data else None


async def _upload_image_to_stripe(
    client: httpx.AsyncClient, image_bytes: bytes, filename: str
) -> str:
    """Upload PNG to Stripe Files API. Returns the file URL."""
    files = {
        "file": (filename, io.BytesIO(image_bytes), "image/png"),
        "purpose": (None, "product_image"),
    }
    r = await client.post(
        f"{STRIPE_FILES_BASE}/files",
        headers={"Authorization": f"Bearer {STRIPE_KEY}"},
        files=files,
    )
    if r.status_code >= 400:
        raise StripeError(f"Files upload {r.status_code}: {r.text[:200]}")
    return r.json()["url"]


async def _flatten_body(d: Dict[str, Any]) -> Dict[str, str]:
    """Convert nested dict to Stripe's bracketed form-encoded format."""
    out: Dict[str, str] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[f"{k}[{sk}]"] = str(sv)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                out[f"{k}[{i}]"] = str(item)
        elif isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = str(v)
    return out


# =============================================================
# Card rendering
# =============================================================

def _render_card(row: Dict[str, Any]) -> bytes:
    """Render a product card PNG for this catalog row. Returns PNG bytes."""
    from stages.product_card_renderer import render_card_bytes
    return render_card_bytes(row)


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =============================================================
# R2 upload (for site display)
# =============================================================

async def _upload_to_r2(image_bytes: bytes, key: str) -> Optional[str]:
    """Upload to R2 and return public URL. Falls back to None on failure."""
    try:
        try:
            from services.r2_client import r2_put_object_bytes
        except ImportError:
            from core.r2 import r2_put_object_bytes
        await r2_put_object_bytes(
            key=key, body=image_bytes, content_type="image/png"
        )
        if R2_PUBLIC_BASE:
            return f"{R2_PUBLIC_BASE.rstrip('/')}/{key}"
        return key
    except Exception as exc:
        logger.warning("R2 upload failed for %s: %s", key, exc)
        return None


# =============================================================
# Local filesystem mirror (so frontend serving and Stripe see same image)
# =============================================================

def _write_local_card(image_bytes: bytes, filename: str) -> None:
    if not FRONTEND_IMAGES_DIR.exists():
        FRONTEND_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_IMAGES_DIR / filename).write_bytes(image_bytes)


# =============================================================
# Audit log
# =============================================================

async def _log(
    conn: asyncpg.Connection,
    *,
    catalog_product_id: int,
    lookup_key: str,
    operation: str,
    status: str,
    stripe_response: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO catalog_sync_log
            (catalog_product_id, lookup_key, operation, status,
             stripe_response, error_message, actor)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        catalog_product_id, lookup_key, operation, status,
        json.dumps(stripe_response) if stripe_response else None,
        error_message, actor,
    )


# =============================================================
# Per-product sync
# =============================================================

async def sync_product(
    conn: asyncpg.Connection,
    *,
    lookup_key: str,
    actor: str = "manual",
    dry_run: bool = False,
    refresh_caches_after: bool = True,
) -> Dict[str, Any]:
    """Sync one catalog row to Stripe. Returns a summary dict.

    Steps:
      1. Load DB row
      2. Render description from services.catalog_descriptions
      3. Regenerate card PNG, hash it
      4. If hash changed (or no image_url), upload to Stripe Files + R2 + local
      5. Update Stripe Product with name/description/metadata/image
      6. Ensure Stripe Price exists at row.price_usd; if amount differs, mint new
      7. Update DB: last_synced_at, image_url, image_hash

    When ``refresh_caches_after`` is True (default), reloads catalog + billing in-memory caches
    so ``GET /api/pricing`` reflects this row. Bulk ``sync_all`` passes False and refreshes once
    at the end.
    """
    row = await conn.fetchrow(
        "SELECT * FROM catalog_products WHERE lookup_key = $1",
        lookup_key,
    )
    if not row:
        raise ValueError(f"No catalog_products row with lookup_key={lookup_key!r}")

    row_dict = dict(row)
    from services.catalog_descriptions import render_description
    description = render_description(row_dict)

    summary: Dict[str, Any] = {
        "lookup_key": lookup_key,
        "stripe_product_id": row["stripe_product_id"],
        "dry_run": dry_run,
        "operations": [],
    }

    # ---- 1. Regenerate card ----
    try:
        card_bytes = _render_card(row_dict)
    except Exception as exc:
        logger.error("Card render failed for %s: %s", lookup_key, exc)
        await _log(conn,
                   catalog_product_id=row["id"], lookup_key=lookup_key,
                   operation="card_regen", status="error",
                   error_message=str(exc), actor=actor)
        raise

    new_hash = _hash_bytes(card_bytes)
    image_changed = (new_hash != (row["image_hash"] or ""))
    summary["card_hash"] = new_hash
    summary["card_changed"] = image_changed

    if dry_run:
        summary["operations"].append("would_regen_card")

    # ---- 2. Stripe Files upload + R2 mirror + local mirror ----
    stripe_file_url: Optional[str] = None
    r2_url: Optional[str] = row["image_url"]

    if not dry_run and image_changed:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                stripe_file_url = await _upload_image_to_stripe(
                    client, card_bytes, row["image_filename"]
                )
                summary["operations"].append("uploaded_to_stripe_files")
            except StripeError as exc:
                await _log(conn,
                           catalog_product_id=row["id"], lookup_key=lookup_key,
                           operation="image_upload", status="error",
                           error_message=str(exc), actor=actor)
                raise

        r2_key = f"catalog-cards/{row['image_filename']}"
        r2_url = await _upload_to_r2(card_bytes, r2_key) or r2_url
        if r2_url:
            summary["operations"].append("uploaded_to_r2")

        _write_local_card(card_bytes, row["image_filename"])
        summary["operations"].append("wrote_local_mirror")

    # ---- 3. Update Stripe Product ----
    if not dry_run and row["stripe_product_id"]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            metadata = {
                "type": row["product_kind"],
                "lookup_key": lookup_key,
            }
            if row["product_kind"] == "subscription":
                metadata.update({
                    "tier": row["tier_slug"],
                    "put_monthly": str(row["put_monthly"] or 0),
                    "aic_monthly": str(row["aic_monthly"] or 0),
                    "max_accounts": str(row["max_accounts"] or 0),
                    "queue_depth": str(row["queue_depth"] or 0),
                    "lookahead_h": str(row["lookahead_hours"] or 0),
                })
            else:
                metadata.update({
                    "wallet": row["wallet"],
                    "amount": str(row["token_amount"] or 0),
                })

            product_body: Dict[str, Any] = {
                "name": row["stripe_product_name"],
                "description": description,
                "statement_descriptor": row["statement_descriptor"],
                "unit_label": row["unit_label"],
                "tax_code": row["tax_code"],
                "metadata": metadata,
            }
            if stripe_file_url:
                product_body["images"] = [stripe_file_url]

            flat = await _flatten_body(product_body)
            try:
                result = await _stripe_request(
                    client, "POST", f"products/{row['stripe_product_id']}",
                    body=flat,
                )
                summary["operations"].append("updated_product")
                await _log(conn,
                           catalog_product_id=row["id"], lookup_key=lookup_key,
                           operation="product_update", status="success",
                           stripe_response={"id": result.get("id")},
                           actor=actor)
            except StripeError as exc:
                await _log(conn,
                           catalog_product_id=row["id"], lookup_key=lookup_key,
                           operation="product_update", status="error",
                           error_message=str(exc), actor=actor)
                raise

            # ---- 4. Price reconciliation ----
            await _reconcile_price(
                client, conn, row_dict,
                summary=summary, actor=actor,
            )

    # ---- 5. Save state back to DB ----
    if not dry_run:
        await conn.execute(
            """
            UPDATE catalog_products
            SET image_hash = $1,
                image_url = COALESCE($2, image_url),
                last_synced_at = NOW(),
                last_synced_by = $3,
                updated_at = NOW()
            WHERE id = $4
            """,
            new_hash, r2_url, actor, row["id"],
        )

    if not dry_run and refresh_caches_after:
        try:
            summary["pricing_cache_refresh"] = await refresh_pricing_caches_after_catalog_sync(conn)
        except Exception as exc:
            logger.warning("pricing cache reload after sync failed: %s", exc)
            summary["pricing_cache_refresh"] = {"error": str(exc)}

    return summary


async def _reconcile_price(
    client: httpx.AsyncClient,
    conn: asyncpg.Connection,
    row: Dict[str, Any],
    *,
    summary: Dict[str, Any],
    actor: str,
) -> None:
    """Ensure a Stripe Price exists at row.price_usd for this lookup_key.

    Stripe Prices are immutable. If a Price exists with a different unit_amount,
    we DELETE its lookup_key (so it can be reused), archive the old price, and
    create a new Price with the same lookup_key. Old subscribers stay on the
    old price; new checkouts get the new one.
    """
    target_cents = int(round(float(row["price_usd"]) * 100))
    lookup_key = row["lookup_key"]

    existing = await _find_price_by_lookup_key(client, lookup_key)

    if existing and int(existing["unit_amount"] or 0) == target_cents:
        summary["operations"].append("price_unchanged")
        return

    if existing:
        # Remove its lookup_key + archive so we can mint a new one with the key
        await _stripe_request(client, "POST", f"prices/{existing['id']}",
                              body={"active": "false", "lookup_key": ""})
        summary["operations"].append("archived_old_price")

    # Create new
    new_body: Dict[str, Any] = {
        "product": row["stripe_product_id"],
        "unit_amount": str(target_cents),
        "currency": row["currency"],
        "lookup_key": lookup_key,
        "transfer_lookup_key": "true",
        "metadata": {"lookup_key": lookup_key},
    }
    if row["product_kind"] == "subscription":
        new_body["recurring"] = {"interval": "month"}

    flat = await _flatten_body(new_body)
    result = await _stripe_request(client, "POST", "prices", body=flat)
    summary["operations"].append(f"created_price:{result['id']}")
    await _log(conn,
               catalog_product_id=row["id"], lookup_key=lookup_key,
               operation="price_create", status="success",
               stripe_response={"id": result["id"], "unit_amount": target_cents},
               actor=actor)

    # Yearly price (only for subs that have one configured)
    yearly = row.get("price_usd_yearly")
    if row["product_kind"] == "subscription" and yearly:
        yearly_key = lookup_key.replace("_monthly", "_yearly")
        yearly_cents = int(round(float(yearly) * 100))
        existing_y = await _find_price_by_lookup_key(client, yearly_key)
        if existing_y and int(existing_y["unit_amount"] or 0) == yearly_cents:
            summary["operations"].append("yearly_price_unchanged")
            return
        if existing_y:
            await _stripe_request(client, "POST", f"prices/{existing_y['id']}",
                                  body={"active": "false", "lookup_key": ""})
            summary["operations"].append("archived_old_yearly_price")
        yearly_body = {
            "product": row["stripe_product_id"],
            "unit_amount": str(yearly_cents),
            "currency": row["currency"],
            "lookup_key": yearly_key,
            "transfer_lookup_key": "true",
            "recurring": {"interval": "year"},
            "metadata": {"lookup_key": yearly_key, "billing": "yearly"},
        }
        y_flat = await _flatten_body(yearly_body)
        y_result = await _stripe_request(client, "POST", "prices", body=y_flat)
        summary["operations"].append(f"created_yearly_price:{y_result['id']}")


# =============================================================
# Bulk sync
# =============================================================

async def sync_all(
    conn: asyncpg.Connection,
    *,
    actor: str = "manual",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sync every active catalog row (Stripe + cards + R2 + local), then reload pricing caches once.

    Per-product work uses ``sync_product(..., refresh_caches_after=False)`` so Stripe/R2/local
    and DB row updates run for each SKU without N reloads. After all rows:

    - ``load_catalog_pricing_cache`` — ``catalog_products`` → in-memory tier/top-up overlays
      (drives entitlements merge for list prices and caps).
    - ``load_billing_catalog_cache`` — ``billing_catalog`` singleton JSON overrides.

    ``GET /api/pricing`` then reads the merged effective config immediately (no restart).
    """
    rows = await conn.fetch(
        "SELECT lookup_key FROM catalog_products "
        "WHERE is_archived = FALSE AND is_internal = FALSE "
        "ORDER BY sort_order"
    )
    results: List[Dict[str, Any]] = []
    for r in rows:
        try:
            res = await sync_product(
                conn,
                lookup_key=r["lookup_key"],
                actor=actor,
                dry_run=dry_run,
                refresh_caches_after=False,
            )
            results.append(res)
            logger.info("Synced %s: %s", r["lookup_key"], res.get("operations"))
        except Exception as exc:
            logger.exception("Sync failed for %s: %s", r["lookup_key"], exc)
            results.append({
                "lookup_key": r["lookup_key"], "error": str(exc),
            })
    cache_detail: Dict[str, Any] = {}
    refreshed = False
    if not dry_run:
        try:
            cache_detail = await refresh_pricing_caches_after_catalog_sync(conn)
            refreshed = bool(
                cache_detail.get("catalog_products_overlay_loaded")
                and cache_detail.get("billing_catalog_singleton_loaded")
                and cache_detail.get("api_pricing_ready")
            )
        except Exception as exc:
            logger.warning("pricing cache reload after sync-all failed: %s", exc)
            cache_detail = {"error": str(exc)}
    return {
        "results": results,
        "in_memory_pricing_refreshed": refreshed,
        "pricing_cache_detail": cache_detail,
        "dry_run": dry_run,
    }


# =============================================================
# CLI entrypoint
# =============================================================

async def _main_cli() -> int:
    p = argparse.ArgumentParser(description="UploadM8 Stripe catalog sync")
    p.add_argument("--all", action="store_true", help="Sync all active products, then reload /api/pricing caches once")
    p.add_argument("--lookup-key", help="Sync only this lookup_key")
    p.add_argument("--dry-run", action="store_true", help="Render and hash, but don't call Stripe")
    p.add_argument("--actor", default=os.environ.get("USER", "manual"))
    args = p.parse_args()

    if not args.all and not args.lookup_key:
        p.print_help()
        return 1
    if not STRIPE_KEY:
        logger.error("STRIPE_SECRET_KEY env var not set")
        return 2
    if not DATABASE_URL:
        logger.error("DATABASE_URL env var not set")
        return 2

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if args.lookup_key:
            result = await sync_product(conn, lookup_key=args.lookup_key,
                                        actor=args.actor, dry_run=args.dry_run)
            print(json.dumps(result, indent=2, default=str))
        else:
            payload = await sync_all(conn, actor=args.actor, dry_run=args.dry_run)
            print(json.dumps(payload, indent=2, default=str))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main_cli()))
