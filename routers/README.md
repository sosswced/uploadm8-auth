# UploadM8 FastAPI routers

## `app.py` is the wiring layer, not the API monolith

The FastAPI app is built in **`app.py`** (on the order of **~240 lines** of wiring): lifespan, middleware, exception handling, and **`include_router(...)`** calls. **SQL migrations** are **`migrations/runtime_migrations.py`** (applied from lifespan via `run_migrations(db_pool)`).

**Do not add new route handlers to `app.py`.** Add them to the right `routers/<domain>.py` (or a new router module) and register the router there.

Remaining structural debt is **large router modules** — not a monolithic `app.py`.

## Routers mounted on `app` (order)

Registration order in `app.py` (see “Router registration” section). Each router sets its own `APIRouter(prefix=...)` where applicable.

| Order | Module | Notes |
|------|--------|--------|
| 1 | `routers.auth` | `/api/auth/*` |
| 2 | `routers.me` | `/api/me`, settings, wallet UI APIs, account flows |
| 3 | `routers.uploads` | `/api/uploads/*` |
| 4 | `routers.scheduled` | Scheduled upload APIs |
| 5 | `routers.preferences` | User preferences |
| 6 | `routers.groups` | Account groups |
| 7 | `routers.oauth` | Platform OAuth |
| 8 | `routers.platforms` | Connected accounts |
| 9 | `routers.billing` | Checkout, Stripe customer portal |
| 10 | `routers.webhooks` | Stripe / platform webhooks |
| 11 | `routers.analytics` | Analytics / exports |
| 12 | `routers.admin` | Admin APIs |
| 13 | `routers.dashboard` | Dashboard |
| 14 | `routers.trill` | Trill / geo features |
| 15 | `routers.entitlements` | Entitlements |
| 16 | `routers.support` | Support |
| 17 | `routers.api_keys` | API keys |
| 18 | `routers.catalog` | Catalog |
| 19 | `routers.ops` | `/ready`, metrics, contract helpers (`/health` is on `app` itself) |
| 20 | `routers.thumbnail_studio_api` | Thumbnail studio |
| *last* | `routers.domain` | After `populate_domain_router()` — compat shell; mount **last** |

## Domain router (compat)

`populate_domain_router()` then `app.include_router(domain_router)` runs **last**.
Sub-registrars may be no-ops; prefer adding routes to the focused routers above.

## Schema migrations

- **Source:** `migrations/runtime_migrations.py` — `async def run_migrations(db_pool)`.
- **Invocation:** `app.py` lifespan → `await run_migrations(core.state.db_pool)` after pool creation.
- **CLI (optional):** `python tools/run_migrations_once.py` (loads `.env`, uses `DATABASE_URL`).

Do not add a second copy of the migration tuple list in `app.py`.

## CI / hygiene

`tools/lint_routers.py` enforces per-file line and DB-touch budgets using `tools/router_lint_baseline.json`. Regenerate baseline only when intentionally changing caps:  
`python tools/lint_routers.py --write-baseline`

## Frontend note

`frontend/js/html-safe.js` is intended for safe interpolation where `innerHTML` is used; prefer `textContent` or escaping helpers for any API-sourced strings.
