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

# Load .env before any config reads (needed for local dev when running via uvicorn) (needed for local dev when running via uvicorn)
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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("uploadm8-api")

# ============================================================
# PADUS + Gazetteer now live in PostgreSQL tables:
#   - padus_protected_areas  (656,986 rows, geometry+GIST index)
#   - gazetteer_places       (32,333 rows)
# Loaded once from local files via scripts/load_padus.py + load_gaz.py.
# Runtime migration v1059 ensures PostGIS is enabled and (if the table exists) a GiST index on geometry.
# No runtime download required.
# ============================================================

# ── Core imports ─────────────────────────────────────────────────────────────
import core.state
from core.config import (
    DATABASE_URL,
    DB_POOL_MAX,
    DB_POOL_MIN,
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
from routers.uploads_analytics import router as uploads_analytics_router
from routers.uploads_lifecycle import router as uploads_lifecycle_router
from routers.uploads_read import router as uploads_read_router
from routers.scheduled import router as scheduled_router
from routers.scheduling import router as scheduling_router
from routers.preferences import router as preferences_router
from routers.groups import router as groups_router
from routers.workspace import router as workspace_router
from routers.platforms import router as platforms_router
from routers.platform_avatar_redirect import router as platform_avatar_redirect_router
from routers.user_avatar_redirect import router as user_avatar_redirect_router
from routers.billing import router as billing_router
from routers.webhooks import router as webhooks_router
from routers.analytics import router as analytics_router
from routers.admin import router as admin_router
from routers.admin_catalog import router as admin_catalog_router
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
from routers.features import router as features_router
from routers.ops import router as ops_router
from routers.oauth import router as oauth_router
from routers.api_keys import router as api_keys_router
from routers.thumbnail_studio_api import router as thumbnail_studio_router
from routers.domain import populate_domain_router, router as domain_router
from routers.vehicles import router as vehicles_router

init_sentry_for_api()

# ============================================================
# App Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_enc_keys()
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
        # Pin the API version so SDK upgrades can't silently change response
        # shapes (e.g. the Basil move of current_period_* / invoice.subscription).
        stripe.api_version = "2025-03-31.basil"

    core.state.db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=DB_POOL_MIN,
        max_size=DB_POOL_MAX,
        init=_init_asyncpg_codecs,
        max_inactive_connection_lifetime=300,
        command_timeout=60,
    )
    await _load_uploads_columns(core.state.db_pool)
    logger.info("Database connected")

    await run_migrations(core.state.db_pool)

    try:
        from services.upload_funnel import set_funnel_db_pool

        set_funnel_db_pool(core.state.db_pool)
    except Exception as e:
        logger.debug("upload funnel db pool init skipped: %s", e)

    try:
        from services.ml_model_sync import sync_ml_models_from_hub
        from services.promo_targeting_model import reload_model as reload_promo_model
        from services.content_success_model import reload_model as reload_content_model

        sync_ml_models_from_hub()
        reload_promo_model()
        reload_content_model()
    except Exception as e:
        logger.debug("ml model reload skipped: %s", e)

    try:
        async with core.state.db_pool.acquire() as conn:
            from services.billing_service_weights import ensure_billing_weights_seeded

            await ensure_billing_weights_seeded(conn)
            logger.info("billing_service_weights seed check complete")
    except Exception as e:
        logger.warning("billing_service_weights seed failed: %s", e)

    if REDIS_URL:
        try:
            # Resilience tuning — fixes Windows WinError 10054 ("connection
            # forcibly closed") on enqueue_job after the socket goes idle:
            #   * health_check_interval pings idle conns so we discover a
            #     half-open socket BEFORE the next lpush.
            #   * socket_keepalive lets the OS reap dead peers quickly.
            #   * retry_on_error transparently reconnects + replays the command
            #     once when the underlying socket has been reset.
            from redis.asyncio.retry import Retry
            from redis.backoff import ExponentialBackoff
            from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

            core.state.redis_client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_keepalive=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                health_check_interval=30,
                retry_on_error=[RedisConnectionError, RedisTimeoutError],
                retry=Retry(ExponentialBackoff(cap=2, base=0.1), retries=3),
            )
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

    try:
        async with core.state.db_pool.acquire() as conn:
            from services.catalog_publish import refresh_pricing_caches_after_catalog_sync

            detail = await refresh_pricing_caches_after_catalog_sync(conn)
            if detail.get("api_pricing_ready"):
                logger.info(
                    "pricing caches hydrated (catalog_products → TIER_CONFIG overlays + billing_catalog): "
                    "loaded_at=%s",
                    detail.get("catalog_pricing_loaded_at"),
                )
            else:
                logger.warning("pricing cache hydrate incomplete: %s", detail)
    except Exception as e:
        logger.warning("pricing cache hydrate failed: %s", e)

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

    try:
        async with core.state.db_pool.acquire() as conn:
            from services.vehicle_catalog import ensure_makes_populated

            await ensure_makes_populated(conn)
            logger.info("vehicle makes catalog ready (popular seed, no request-path NHTSA)")
    except Exception as e:
        logger.warning("vehicle makes seed check failed: %s", e)

    trill_maintenance_task = None
    ml_engine_task = None
    marketing_automation_task = None
    if core.state.db_pool:
        import asyncio
        from services.trill_background import run_trill_maintenance_loop

        trill_maintenance_task = asyncio.create_task(
            run_trill_maintenance_loop(core.state.db_pool)
        )
        logger.info("Trill maintenance loop started")

        from services.ml_engine_background import run_ml_engine_loop

        ml_engine_task = asyncio.create_task(
            run_ml_engine_loop(core.state.db_pool, core.state.redis_client)
        )
        logger.info("ML engine loop started")

        from services.marketing_automation_background import run_marketing_automation_loop

        marketing_automation_task = asyncio.create_task(
            run_marketing_automation_loop(core.state.db_pool, core.state.redis_client)
        )
        logger.info("Marketing automation loop started (self-gates on MARKETING_AUTOMATION_ENABLED)")

    yield

    if trill_maintenance_task:
        import asyncio as _asyncio

        trill_maintenance_task.cancel()
        try:
            await trill_maintenance_task
        except _asyncio.CancelledError:
            pass

    if ml_engine_task:
        import asyncio as _asyncio

        ml_engine_task.cancel()
        try:
            await ml_engine_task
        except _asyncio.CancelledError:
            pass

    if marketing_automation_task:
        import asyncio as _asyncio

        marketing_automation_task.cancel()
        try:
            await marketing_automation_task
        except _asyncio.CancelledError:
            pass

    # Shutdown: pool/redis close can race with Ctrl+C / reload — never let teardown
    # exceptions escape lifespan (uvicorn already logs its own CancelledError on the
    # lifespan receive task; that is normal asyncio behaviour, not an app bug).
    if core.state.db_pool:
        try:
            await core.state.db_pool.close()
        except BaseException as e:
            logger.debug("lifespan: db_pool.close: %s", e)
    if core.state.redis_client:
        try:
            await core.state.redis_client.close()
        except BaseException as e:
            logger.debug("lifespan: redis close: %s", e)


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

    from core.db_pool import is_transient_db_error

    origin = request.headers.get("origin", "")
    allowed = ALLOWED_ORIGINS_LIST
    cors_origin = origin if origin in allowed else (allowed[0] if allowed else "*")
    cors_headers = {
        "Access-Control-Allow-Origin": cors_origin,
        "Access-Control-Allow-Credentials": "true",
    }

    if is_transient_db_error(exc):
        logger.warning(
            "Transient database error on %s %s: %s: %s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Database temporarily unavailable"},
            headers=cors_headers,
        )

    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )

    # Best-effort: record this 500 as an operational incident so the admin
    # incidents page surfaces every unhandled API failure. DB-only — no email
    # / Discord spam (worker errors and explicit notify_admin_error already
    # send those).
    try:
        if core.state.db_pool is not None:
            import traceback as _tb
            from services.ops_incidents import record_operational_incident

            uid = None
            try:
                state_user = getattr(request.state, "user", None)
                if isinstance(state_user, dict):
                    uid = state_user.get("id")
            except Exception:
                uid = None

            await record_operational_incident(
                core.state.db_pool,
                source="api",
                incident_type=f"api_500:{type(exc).__name__}"[:120],
                subject=f"{request.method} {request.url.path} → 500",
                body=f"{type(exc).__name__}: {exc}\n\n{_tb.format_exc()}"[:8000],
                details={
                    "method": request.method,
                    "path": str(request.url.path),
                    "query": str(request.url.query)[:1000],
                    "request_id": getattr(request.state, "request_id", None),
                    "user_agent": request.headers.get("user-agent", "")[:512],
                    "origin": origin[:200],
                },
                user_id=uid,
                # Alerts are rate-limited inside record_operational_incident
                # (one alert per source+incident_type per OPS_ALERT_DEDUPE_SECONDS,
                # default 15min) so an API outage producing thousands of 500s
                # only sends one Discord+email per error class per window.
                # Dedupe key bundles method+path so different broken endpoints
                # each get their own slot.
                dedupe_key=f"api_500:{type(exc).__name__}:{request.method}:{request.url.path}",
            )
    except Exception:
        pass

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


# ── Static asset cache + CDN preconnect (HTML responses) ─────────────────────
_STATIC_CACHE_EXTS = (
    ".js",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
)
_PRECONNECT_LINK = (
    "<https://fonts.googleapis.com>; rel=preconnect, "
    "<https://fonts.gstatic.com>; rel=preconnect; crossorigin, "
    "<https://cdnjs.cloudflare.com>; rel=preconnect"
)


@app.middleware("http")
async def static_cache_and_preconnect(request: Request, call_next):
    resp = await call_next(request)
    path = request.url.path.lower()
    if any(path.endswith(ext) for ext in _STATIC_CACHE_EXTS):
        resp.headers.setdefault(
            "Cache-Control",
            "public, max-age=86400, stale-while-revalidate=604800",
        )
    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/html" in content_type or path.endswith(".html") or path in ("", "/"):
        existing = resp.headers.get("Link", "")
        resp.headers["Link"] = (
            f"{existing}, {_PRECONNECT_LINK}" if existing else _PRECONNECT_LINK
        )
    return resp


# ── Security headers middleware ──────────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    # SAMEORIGIN: allow Settings guide iframe; still blocks third-party embeds.
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self'; "
        "default-src 'self'; "
        "media-src 'self' blob: data: https:; "
        "img-src 'self' data: https: blob:; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com data:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "script-src 'self' 'unsafe-inline' https:; "
        "connect-src 'self' https://api.stripe.com https://js.stripe.com "
        "https://uploadm8.com https://*.uploadm8.com "
        "https://cdn.jsdelivr.net https://cdn.sheetjs.com https://cdnjs.cloudflare.com "
        "https://unpkg.com "
        "https://*.r2.cloudflarestorage.com https://*.r2.dev;"
    )
    return resp


# ============================================================
# Router registration
# ============================================================
app.include_router(auth_router)
app.include_router(me_router)
app.include_router(uploads_read_router)
app.include_router(uploads_analytics_router)
app.include_router(uploads_lifecycle_router)
app.include_router(scheduled_router)
app.include_router(scheduling_router)
app.include_router(preferences_router)
app.include_router(groups_router)
app.include_router(workspace_router)
app.include_router(oauth_router)
app.include_router(platforms_router)
app.include_router(platform_avatar_redirect_router)
app.include_router(user_avatar_redirect_router)
app.include_router(billing_router)
app.include_router(webhooks_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(admin_catalog_router)
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
app.include_router(features_router)
app.include_router(ops_router)
app.include_router(thumbnail_studio_router)
app.include_router(vehicles_router)

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
    _SMART_LEGACY_REDIRECTS = {
        "ai-insights.html": "smart-insights.html",
        "ai-insights": "smart-insights.html",
        "admin-upload-ai-trace.html": "admin-upload-smart-trace.html",
        "admin-upload-ai-trace": "admin-upload-smart-trace.html",
        "ai-social-media-scheduler.html": "smart-social-media-scheduler.html",
        "ai-social-media-scheduler": "smart-social-media-scheduler.html",
        "ai-thumbnail-generator-for-youtube.html": "smart-thumbnail-generator-for-youtube.html",
        "ai-thumbnail-generator-for-youtube": "smart-thumbnail-generator-for-youtube.html",
    }

    for _legacy, _target in _SMART_LEGACY_REDIRECTS.items():

        def _make_legacy_redirect(dest=_target):
            async def _legacy_redirect_handler():
                return RedirectResponse(url=f"/{dest}", status_code=301)

            return _legacy_redirect_handler

        app.add_api_route(f"/{_legacy}", _make_legacy_redirect(), methods=["GET"])

    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_STATIC_DIR), html=True),
        name="frontend",
    )
