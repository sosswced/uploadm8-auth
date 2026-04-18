"""
UploadM8 API Server — production entrypoint (intentionally small).

Architecture (for audits / onboarding)
----------------------------------------
**This file is not a route monolith.** HTTP surface area lives in ``routers/*.py``.
Each module defines an ``APIRouter`` (with its own ``prefix`` where needed) and is
registered here with ``app.include_router(...)``. New endpoints belong in an existing router or a new
``routers/<domain>.py`` — do not grow this file with business handlers.

**Where complexity still concentrates**
- **Schema migrations** live in ``migrations/runtime_migrations.py`` (single source
  of truth). ``lifespan`` calls ``run_migrations(db_pool)`` after the pool exists.
- **Fat routers** (e.g. ``routers/admin.py``, ``routers/me.py``): the next
  decomposition target is *thin handlers +* ``services/``, not splitting ``app.py``.

**Contracts**
- Routers: validate input, call services, map errors, return responses.
- ``services/`` and ``core/``: orchestration, DB patterns, shared helpers.
- **Static UI:** when ``frontend/`` exists and ``SERVE_FRONTEND`` is not disabled, the app
  mounts it at ``/`` after all API routes so pages load from the same origin as ``/api/*``
  (see ``core.config.FRONTEND_STATIC_DIR``). HTML still uses ``js/api-base.js`` + ``auth-stack.js``
  to call the routers; browsers do not import Python ``routers/`` or ``core/`` directly.

``routers.domain`` is mounted last: ``populate_domain_router()`` then
``domain_router`` for backward‑compatible paths; real routes should live on the
focused routers above. See ``routers/README.md`` for mount order.
"""

import json
import logging
import pathlib

# Load .env before any config reads (needed for local dev when running via uvicorn)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import asyncpg
import stripe
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# ── Core imports ─────────────────────────────────────────────────────────────
import core.state
from core.config import (
    DATABASE_URL,
    REDIS_URL,
    STRIPE_SECRET_KEY,
    ALLOWED_ORIGINS_LIST,
    BOOTSTRAP_ADMIN_EMAIL,
    FRONTEND_STATIC_DIR,
    SERVE_FRONTEND_STATIC,
)
from core.helpers import _init_asyncpg_codecs, _load_uploads_columns, _now_utc, _req_id
from core.auth import init_enc_keys
from core.security import install_rate_limit_middleware
from core.sentry_init import init_sentry_for_api
from migrations.runtime_migrations import run_migrations

# ── Router imports ───────────────────────────────────────────────────────────
from routers.auth import router as auth_router
from routers.me import router as me_router
from routers.uploads import router as uploads_router
from routers.scheduled import router as scheduled_router
from routers.preferences import router as preferences_router
from routers.groups import router as groups_router
from routers.platforms import router as platforms_router
from routers.platform_avatar_redirect import router as platform_avatar_redirect_router
from routers.billing import router as billing_router
from routers.webhooks import router as webhooks_router
from routers.analytics import router as analytics_router
from routers.admin import router as admin_router
from routers.admin_contract import (
    admin_compat_router,
    marketing_router as admin_marketing_contract_router,
    ml_router as admin_ml_contract_router,
    public_marketing_router,
)
from routers.dashboard import router as dashboard_router
from routers.shell_bootstrap import router as shell_bootstrap_router
from routers.trill import router as trill_router, seed_trill_places
from routers.entitlements import router as entitlements_router
from routers.support import router as support_router
from routers.catalog import router as catalog_router
from routers.ops import router as ops_router
from routers.oauth import router as oauth_router
from routers.api_keys import router as api_keys_router
from routers.thumbnail_studio_api import router as thumbnail_studio_router
from routers.domain import populate_domain_router, router as domain_router

logger = logging.getLogger("uploadm8-api")
init_sentry_for_api()

# ============================================================
# App Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_enc_keys()
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY

    core.state.db_pool = await asyncpg.create_pool(
        DATABASE_URL, min_size=2, max_size=10, init=_init_asyncpg_codecs
    )
    await _load_uploads_columns(core.state.db_pool)
    logger.info("Database connected")

    await run_migrations(core.state.db_pool)

    if REDIS_URL:
        try:
            core.state.redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await core.state.redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis failed: {e}")

    try:
        async with core.state.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT settings_json FROM admin_settings WHERE id = 1")
            if row and row["settings_json"]:
                core.state.admin_settings_cache.update(json.loads(row["settings_json"]))
    except Exception:
        pass

    if BOOTSTRAP_ADMIN_EMAIL:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET role='master_admin', subscription_tier='master_admin' WHERE LOWER(email)=$1",
                BOOTSTRAP_ADMIN_EMAIL,
            )

    # Seed trill places for geo-targeting
    try:
        async with core.state.db_pool.acquire() as conn:
            await seed_trill_places(conn)
            logger.info("Trill places seeded")
    except Exception as e:
        logger.warning(f"Trill places seeding failed: {e}")

    yield

    if core.state.db_pool:
        await core.state.db_pool.close()
    if core.state.redis_client:
        await core.state.redis_client.close()


# ============================================================
# App creation
# ============================================================
app = FastAPI(title="UploadM8 API", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── CORS-safe exception handler ──────────────────────────────────────────────
# FastAPI's CORSMiddleware does NOT add Access-Control-Allow-Origin to 500
# responses when an unhandled exception propagates — the browser then reports
# a CORS error instead of the real HTTP 500.
@app.exception_handler(Exception)
async def _cors_safe_500_handler(request: Request, exc: Exception):
    """
    CORS-safe fallback for *unexpected* errors.

    Must not swallow Starlette ``HTTPException`` (401/403/429/503 from routes/deps) or
    ``RequestValidationError`` (422) — registering ``Exception`` matches those subclasses
    too, which previously turned every ``HTTPException`` into a misleading 500.
    """
    if isinstance(exc, StarletteHTTPException):
        return await http_exception_handler(request, exc)
    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)

    origin = request.headers.get("origin", "")
    allowed = ALLOWED_ORIGINS_LIST
    cors_origin = origin if origin in allowed else (allowed[0] if allowed else "*")

    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
        headers={
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
        },
    )


# ── Rate limiting ────────────────────────────────────────────────────────────
install_rate_limit_middleware(app)


# ── Request ID middleware ────────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or _req_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# ── Security headers middleware ──────────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "media-src 'self' blob: data: https:; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com data:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "script-src 'self' 'unsafe-inline' https:; "
        "connect-src 'self' https://api.stripe.com https://js.stripe.com "
        "https://uploadm8.com https://*.uploadm8.com "
        "https://cdn.jsdelivr.net https://cdn.sheetjs.com https://cdnjs.cloudflare.com "
        "https://*.r2.cloudflarestorage.com https://*.r2.dev;"
    )
    return resp


# ============================================================
# Router registration
# ============================================================
app.include_router(auth_router)
app.include_router(me_router)
app.include_router(uploads_router)
app.include_router(scheduled_router)
app.include_router(preferences_router)
app.include_router(groups_router)
app.include_router(oauth_router)
app.include_router(platforms_router)
app.include_router(platform_avatar_redirect_router)
app.include_router(billing_router)
app.include_router(webhooks_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(admin_marketing_contract_router)
app.include_router(admin_ml_contract_router)
app.include_router(admin_compat_router)
app.include_router(public_marketing_router)
app.include_router(dashboard_router)
app.include_router(shell_bootstrap_router)
app.include_router(trill_router)
app.include_router(entitlements_router)
app.include_router(support_router)
app.include_router(api_keys_router)
app.include_router(catalog_router)
app.include_router(ops_router)
app.include_router(thumbnail_studio_router)

populate_domain_router()
app.include_router(domain_router)

# ============================================================
# Health
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_utc().isoformat()}


# Static HTML/JS/CSS — same process as API (cookie auth, no cross-origin for local dev).
if SERVE_FRONTEND_STATIC and FRONTEND_STATIC_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_STATIC_DIR), html=True),
        name="frontend",
    )
