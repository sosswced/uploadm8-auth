# Catalog sync and pricing governance

This document describes how UploadM8 keeps `catalog_products`, Stripe, caches, and frontends aligned.

## Runtime and env

- **Database**: `DATABASE_URL` — holds `catalog_products`, `catalog_sync_log`, `catalog_pricing_requests`.
- **Stripe**: `STRIPE_SECRET_KEY` — live or test; must match the mode you intend to reconcile.
- **Public assets**: `R2_PUBLIC_BASE` (and related R2 credentials) for mirrored catalog card images.
- **Local card mirror (optional)**: `UPLOADM8_FRONTEND_IMAGES_DIR` — where regenerated PNGs are written during sync.
- **Pricing approval** (optional):
  - `CATALOG_PRICING_APPROVAL_REQUIRED` — when truthy, direct `PATCH` on money fields (`price_usd`, `price_usd_yearly`) returns **409** unless bypassed.
  - `CATALOG_PRICING_APPROVAL_BYPASS` — when truthy, skips the approval gate (break-glass for automation or trusted operators).

## Admin API (`/api/admin/catalog`)

- **List / read products**: `GET /api/admin/catalog/products`, `GET /api/admin/catalog/products/{lookup_key}`.
- **Patch**: `PATCH /api/admin/catalog/products/{lookup_key}?auto_sync=true` (default).

### `auto_sync` query parameter

- `auto_sync=true` (default): after a successful DB update, the server runs a Stripe sync for that row (card, metadata, price migration as implemented in `scripts/sync_stripe_catalog.py`).
- `auto_sync=false`: DB-only update; use when you want to batch changes before a manual `POST …/sync` or `sync_stripe_catalog` run.

### Pricing approval flow

When `CATALOG_PRICING_APPROVAL_REQUIRED` is enabled and bypass is off:

1. `PATCH` that includes `price_usd` and/or `price_usd_yearly` receives **409** with a body that includes `detail.code == "pricing_approval_required"`.
2. Create a ticket: `POST /api/admin/catalog/pricing-requests` with `{ "lookup_key": "…", "proposed_patch": { … } }`.
3. Resolve: `POST /api/admin/catalog/pricing-requests/{id}/resolve` with `{ "status": "approved"|"rejected", "resolution_notes": "…" }`.
   - **approved**: applies allowed catalog columns from `proposed_patch`, runs `sync_product` for that `lookup_key`, then refreshes pricing + billing catalog caches.
   - **rejected**: closes the ticket without DB price changes.

## Scripts

- **`python -m scripts.disable_catalog_hud_and_sync`**: sets `catalog_products.hud = false` for all rows, then runs a full Stripe sync (same as deploy migration v1080 + `sync_stripe_catalog --all`).
- **`python -m scripts.sync_stripe_catalog`**: push catalog rows to Stripe; reloads catalog pricing cache and billing catalog cache after work.
- **`python -m scripts.stripe_reconcile_catalog`**: compare DB amounts to Stripe Prices by `lookup_key`. Add **`--apply`** to write DB from Stripe when a matching Price exists and cents differ (report-only by default).

## Canonical Stripe lookup keys

Subscription examples use the **`uploadm8_creatorlite_*`** spelling (no underscore between `creator` and `lite`). Legacy `uploadm8_creator_lite_*` keys may still map in entitlements for old subscriptions; new catalog rows should use the canonical keys.

## Startup and merge order (`app.py` lifespan)

On API boot, **`services.catalog_publish.refresh_pricing_caches_after_catalog_sync(conn)`** runs once (same helper as post–Stripe sync and billing-catalog background job tail):

1. `load_catalog_pricing_cache(conn)` — `catalog_products` → `core.state.catalog_pricing_cache` tier/top-up overlays (hydrates effective tier list prices and caps from the live table).
2. `load_billing_catalog_cache(conn)` — `billing_catalog` singleton → `core.state.billing_catalog_cache`.

Effective values used by `GET /api/pricing` and `stages/entitlements.get_effective_tier_config` / `get_effective_topup_products` merge **Python defaults (`TIER_CONFIG` / `TOPUP_PRODUCTS`) → catalog overlay → `billing_catalog` JSON**. On field collisions, **billing overrides win** over the catalog row.

## Bulk `POST /api/admin/catalog/sync-all`

Response JSON (same shape as `python -m scripts.sync_stripe_catalog --all`):

- **`results`**: array of per-product `sync_product` summaries (or `{ lookup_key, error }` on failure).
- **`in_memory_pricing_refreshed`**: `true` when both overlay steps succeeded (skipped when `dry_run=true`).
- **`pricing_cache_detail`**: object from `refresh_pricing_caches_after_catalog_sync` — e.g. `catalog_products_overlay_loaded`, `billing_catalog_singleton_loaded`, `catalog_pricing_loaded_at`, `api_pricing_ready`, or `error` on failure.
- **`dry_run`**: echo of the query flag.

Bulk sync runs Stripe + card + R2 + local for each row, then **one** reload: `catalog_products` → `catalog_pricing_cache`, then `billing_catalog` table → `billing_catalog_cache`, so `GET /api/pricing` is up to date without restarting the API.

## Frontend pricing

Marketing and billing pages should hydrate prices from **`GET /api/pricing`** (or tier-specific endpoints) rather than hardcoding USD strings, so catalog and entitlement overlays stay consistent.
