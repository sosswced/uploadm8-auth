"""
UploadM8 API Server - Production Build v4
Slim entrypoint: lifespan, migrations, middleware, router wiring.
All route handlers live in routers/, shared logic in core/.
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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Core imports ─────────────────────────────────────────────────────────────
import core.state
from core.config import (
    DATABASE_URL, REDIS_URL, STRIPE_SECRET_KEY,
    ALLOWED_ORIGINS_LIST, BOOTSTRAP_ADMIN_EMAIL,
)
from core.helpers import _init_asyncpg_codecs, _load_uploads_columns, _now_utc, _req_id
from core.auth import init_enc_keys
from core.security import install_rate_limit_middleware

# ── Router imports ───────────────────────────────────────────────────────────
from routers.auth import router as auth_router
from routers.me import router as me_router
from routers.uploads import router as uploads_router
from routers.scheduled import router as scheduled_router
from routers.preferences import router as preferences_router
from routers.groups import router as groups_router
from routers.platforms import router as platforms_router
from routers.billing import router as billing_router
from routers.webhooks import router as webhooks_router
from routers.analytics import router as analytics_router
from routers.admin import router as admin_router
from routers.dashboard import router as dashboard_router
from routers.trill import router as trill_router, seed_trill_places
from routers.entitlements import router as entitlements_router
from routers.support import router as support_router

logger = logging.getLogger("uploadm8-api")

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

    await run_migrations()

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
# Migrations
# ============================================================
async def run_migrations():
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW())")
        applied = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}

        migrations = [
            (1, """CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), email VARCHAR(255) UNIQUE NOT NULL, password_hash VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL, role VARCHAR(50) DEFAULT 'user', subscription_tier VARCHAR(50) DEFAULT 'free',
                stripe_customer_id VARCHAR(255), stripe_subscription_id VARCHAR(255), subscription_status VARCHAR(50),
                current_period_end TIMESTAMPTZ, flex_enabled BOOLEAN DEFAULT FALSE, timezone VARCHAR(100) DEFAULT 'UTC',
                avatar_url VARCHAR(512), status VARCHAR(50) DEFAULT 'active', last_active_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"""),
            (2, "CREATE TABLE IF NOT EXISTS refresh_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, token_hash VARCHAR(255) UNIQUE NOT NULL, expires_at TIMESTAMPTZ NOT NULL, revoked_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT NOW())"),
            (3, "CREATE TABLE IF NOT EXISTS platform_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, platform VARCHAR(50) NOT NULL, account_id VARCHAR(255), account_name VARCHAR(255), account_username VARCHAR(255), account_avatar VARCHAR(512), token_blob JSONB NOT NULL, is_primary BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (31, """
                ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ;
                CREATE INDEX IF NOT EXISTS idx_platform_tokens_user_platform_active ON platform_tokens(user_id, platform) WHERE revoked_at IS NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS ux_platform_tokens_active_identity ON platform_tokens(user_id, platform, account_id)
                    WHERE revoked_at IS NULL AND account_id IS NOT NULL AND account_id <> '';
            """),

            (4, """CREATE TABLE IF NOT EXISTS uploads (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                r2_key VARCHAR(512) NOT NULL, telemetry_r2_key VARCHAR(512), processed_r2_key VARCHAR(512), thumbnail_r2_key VARCHAR(512),
                filename VARCHAR(255) NOT NULL, file_size BIGINT, platforms VARCHAR(50)[] DEFAULT '{}',
                title VARCHAR(512), caption TEXT, hashtags TEXT[], privacy VARCHAR(50) DEFAULT 'public',
                status VARCHAR(50) DEFAULT 'pending', cancel_requested BOOLEAN DEFAULT FALSE,
                scheduled_time TIMESTAMPTZ, schedule_mode VARCHAR(50) DEFAULT 'immediate',
                processing_started_at TIMESTAMPTZ, processing_finished_at TIMESTAMPTZ, completed_at TIMESTAMPTZ,
                error_code VARCHAR(100), error_detail TEXT, platform_results JSONB,
                put_reserved INT DEFAULT 0, put_spent INT DEFAULT 0, aic_reserved INT DEFAULT 0, aic_spent INT DEFAULT 0,
                compute_seconds FLOAT DEFAULT 0, storage_bytes BIGINT DEFAULT 0, cost_attributed DECIMAL(10,4) DEFAULT 0,
                views BIGINT DEFAULT 0, likes BIGINT DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"""),
            (5, "CREATE TABLE IF NOT EXISTS user_settings (user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, discord_webhook VARCHAR(512), telemetry_enabled BOOLEAN DEFAULT TRUE, hud_enabled BOOLEAN DEFAULT TRUE, hud_position VARCHAR(50) DEFAULT 'bottom-left', speeding_mph INT DEFAULT 80, euphoria_mph INT DEFAULT 100, hud_speed_unit VARCHAR(10) DEFAULT 'mph', hud_color VARCHAR(20) DEFAULT '#FFFFFF', updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (6, """CREATE TABLE IF NOT EXISTS wallets (
                user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                put_balance INT DEFAULT 0, aic_balance INT DEFAULT 0,
                put_reserved INT DEFAULT 0, aic_reserved INT DEFAULT 0,
                last_refill_date DATE, created_at TIMESTAMPTZ DEFAULT NOW())"""),
            (7, """CREATE TABLE IF NOT EXISTS token_ledger (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                token_type VARCHAR(10) NOT NULL, platform VARCHAR(50), delta INT NOT NULL,
                reason VARCHAR(50) NOT NULL, upload_id UUID, stripe_event_id VARCHAR(255),
                meta JSONB, created_at TIMESTAMPTZ DEFAULT NOW())"""),
            (8, """CREATE TABLE IF NOT EXISTS announcements (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), title VARCHAR(500) NOT NULL, body TEXT NOT NULL,
                channels JSONB, target VARCHAR(50), target_tiers TEXT[],
                email_sent INT DEFAULT 0, discord_sent INT DEFAULT 0, webhook_sent INT DEFAULT 0,
                created_by UUID REFERENCES users(id), created_at TIMESTAMPTZ DEFAULT NOW())"""),
            (9, "CREATE TABLE IF NOT EXISTS admin_settings (id INT PRIMARY KEY DEFAULT 1, settings_json JSONB DEFAULT '{}', updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (10, "CREATE TABLE IF NOT EXISTS cost_tracking (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID, category VARCHAR(100) NOT NULL, operation VARCHAR(255), tokens INT, cost_usd DECIMAL(10,6), created_at TIMESTAMPTZ DEFAULT NOW())"),
            (11, "CREATE TABLE IF NOT EXISTS revenue_tracking (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID, amount DECIMAL(10,2) NOT NULL, source VARCHAR(100), stripe_event_id VARCHAR(255), plan VARCHAR(100), created_at TIMESTAMPTZ DEFAULT NOW())"),
            (12, "CREATE TABLE IF NOT EXISTS account_groups (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, name VARCHAR(100) NOT NULL, account_ids TEXT[] DEFAULT '{}', color VARCHAR(20) DEFAULT '#3b82f6', created_at TIMESTAMPTZ DEFAULT NOW())"),
            (13, "CREATE TABLE IF NOT EXISTS white_label_settings (user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, enabled BOOLEAN DEFAULT FALSE, logo_url VARCHAR(512), company_name VARCHAR(255), primary_color VARCHAR(20), created_at TIMESTAMPTZ DEFAULT NOW())"),
            (14, "INSERT INTO admin_settings (id, settings_json) VALUES (1, '{}') ON CONFLICT DO NOTHING"),
            (15, "CREATE INDEX IF NOT EXISTS idx_uploads_user_status ON uploads(user_id, status)"),
            (16, "CREATE INDEX IF NOT EXISTS idx_ledger_user ON token_ledger(user_id, created_at)"),
            (17, "CREATE INDEX IF NOT EXISTS idx_cost_tracking_date ON cost_tracking(created_at)"),
            (18, """CREATE TABLE IF NOT EXISTS user_color_preferences (
                user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                tiktok_color VARCHAR(20) DEFAULT '#000000',
                youtube_color VARCHAR(20) DEFAULT '#FF0000',
                instagram_color VARCHAR(20) DEFAULT '#E4405F',
                facebook_color VARCHAR(20) DEFAULT '#1877F2',
                accent_color VARCHAR(20) DEFAULT '#F97316',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW())"""),
            (19, "CREATE INDEX IF NOT EXISTS idx_uploads_scheduled ON uploads(user_id, scheduled_time) WHERE scheduled_time IS NOT NULL"),
            (20, "CREATE INDEX IF NOT EXISTS idx_uploads_user_scheduled_status ON uploads(user_id, status, scheduled_time)"),
            (21, "ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name VARCHAR(255)"),
            (22, "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name VARCHAR(255)"),
            (23, "ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS preferences_json JSONB DEFAULT '{}'"),
            (24, """CREATE TABLE IF NOT EXISTS user_preferences (
                user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                auto_captions BOOLEAN DEFAULT FALSE,
                auto_thumbnails BOOLEAN DEFAULT FALSE,
                thumbnail_interval INT DEFAULT 5,
                default_privacy VARCHAR(50) DEFAULT 'public',
                ai_hashtags_enabled BOOLEAN DEFAULT FALSE,
                ai_hashtag_count INT DEFAULT 5,
                ai_hashtag_style VARCHAR(50) DEFAULT 'mixed',
                hashtag_position VARCHAR(50) DEFAULT 'end',
                max_hashtags INT DEFAULT 15,
                always_hashtags JSONB DEFAULT '[]'::jsonb,
                blocked_hashtags JSONB DEFAULT '[]'::jsonb,
                platform_hashtags JSONB DEFAULT '{"tiktok":[],"youtube":[],"instagram":[],"facebook":[]}'::jsonb,
                email_notifications BOOLEAN DEFAULT TRUE,
                discord_webhook VARCHAR(512),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW())"""),
            (25, """DO $$
                BEGIN
                    -- Convert always_hashtags from TEXT[] to JSONB if it exists
                    IF EXISTS (SELECT 1 FROM information_schema.columns
                              WHERE table_name = 'user_preferences'
                              AND column_name = 'always_hashtags'
                              AND data_type = 'ARRAY') THEN
                        ALTER TABLE user_preferences
                        ALTER COLUMN always_hashtags TYPE JSONB
                        USING array_to_json(always_hashtags)::jsonb;
                    END IF;

                    -- Convert blocked_hashtags from TEXT[] to JSONB if it exists
                    IF EXISTS (SELECT 1 FROM information_schema.columns
                              WHERE table_name = 'user_preferences'
                              AND column_name = 'blocked_hashtags'
                              AND data_type = 'ARRAY') THEN
                        ALTER TABLE user_preferences
                        ALTER COLUMN blocked_hashtags TYPE JSONB
                        USING array_to_json(blocked_hashtags)::jsonb;
                    END IF;
                END $$;"""),
            (26, """-- Clean up corrupted hashtag data
                UPDATE user_preferences
                SET always_hashtags = '[]'::jsonb,
                    blocked_hashtags = '[]'::jsonb
                WHERE
                    (always_hashtags::text LIKE '%\\\\%' OR always_hashtags::text LIKE '%["%')
                    OR (blocked_hashtags::text LIKE '%\\\\%' OR blocked_hashtags::text LIKE '%["%');"""),
            # Trill Telemetry Migrations
            (100, """
                -- Trill analysis results
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS trill_score DECIMAL(5,2);
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS speed_bucket VARCHAR(50);
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS trill_metadata JSONB;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ai_generated_title TEXT;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ai_generated_caption TEXT;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ai_generated_hashtags TEXT[];
            """),
            (101, """
                -- Trill places (popular locations for geo-targeting)
                CREATE TABLE IF NOT EXISTS trill_places (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    state VARCHAR(2) NOT NULL,
                    lat DECIMAL(10,7) NOT NULL,
                    lon DECIMAL(10,7) NOT NULL,
                    popularity_score INT DEFAULT 0,
                    hashtags TEXT[] DEFAULT '{}',
                    is_protected BOOLEAN DEFAULT FALSE,
                    protected_name VARCHAR(255),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(name, state)
                );
                CREATE INDEX IF NOT EXISTS idx_trill_places_state ON trill_places(state);
                CREATE INDEX IF NOT EXISTS idx_trill_places_popularity ON trill_places(popularity_score DESC);
            """),
            (102, """
                -- User trill preferences
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_enabled BOOLEAN DEFAULT FALSE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_min_score INT DEFAULT 60;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_hud_enabled BOOLEAN DEFAULT FALSE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_ai_enhance BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_openai_model VARCHAR(50) DEFAULT 'gpt-4o-mini';
            """),

            (103, """CREATE TABLE IF NOT EXISTS support_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                name VARCHAR(255),
                email VARCHAR(255),
                subject VARCHAR(255),
                message TEXT NOT NULL,
                status VARCHAR(50) DEFAULT 'open',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )"""),

            (104, """CREATE TABLE IF NOT EXISTS admin_audit_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                admin_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                admin_email TEXT,
                action TEXT NOT NULL,
                details JSONB DEFAULT '{}'::jsonb,
                ip_address TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_admin_audit_log_user ON admin_audit_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_admin_audit_log_created ON admin_audit_log(created_at);

            CREATE TABLE IF NOT EXISTS email_changes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                old_email TEXT NOT NULL,
                new_email TEXT NOT NULL,
                changed_by_admin_id UUID REFERENCES users(id) ON DELETE SET NULL,
                verification_token TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS password_resets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                reset_by_admin_id UUID REFERENCES users(id) ON DELETE SET NULL,
                temp_password_hash TEXT NOT NULL,
                force_change BOOLEAN DEFAULT TRUE,
                expires_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );"""),

            (105, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_stage    VARCHAR(100);
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_progress  INT DEFAULT 0;
            """),

            (510, "ALTER TABLE account_groups ADD COLUMN IF NOT EXISTS description TEXT"),
            (511, "ALTER TABLE account_groups ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()"),
            (512, "UPDATE account_groups SET updated_at = NOW() WHERE updated_at IS NULL"),

            # ── Self-serve deletion audit trail ──────────────────────────────────
            (600, """
                CREATE TABLE IF NOT EXISTS account_deletion_log (
                    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id             TEXT NOT NULL,
                    user_email          TEXT NOT NULL,
                    user_name           TEXT,
                    requested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at        TIMESTAMPTZ,
                    r2_keys_deleted     INT DEFAULT 0,
                    tokens_revoked      INT DEFAULT 0,
                    stripe_cancelled    BOOLEAN DEFAULT FALSE,
                    rows_deleted        JSONB DEFAULT '{}'::jsonb,
                    initiated_by        TEXT DEFAULT 'self',
                    ip_address          TEXT,
                    notes               TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_deletion_log_user  ON account_deletion_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_deletion_log_reqat ON account_deletion_log(requested_at);
            """),

            (601, """
                CREATE TABLE IF NOT EXISTS platform_disconnect_log (
                    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id                 TEXT NOT NULL,
                    platform                TEXT NOT NULL,
                    account_id              TEXT,
                    account_name            TEXT,
                    revoked_at_provider     BOOLEAN DEFAULT FALSE,
                    provider_revoke_error   TEXT,
                    purged_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    initiated_by            TEXT DEFAULT 'self',
                    ip_address              TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_disconnect_log_user ON platform_disconnect_log(user_id);
            """),

            (602, """
                CREATE TABLE IF NOT EXISTS tiktok_webhook_events (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    client_key      TEXT,
                    event           TEXT NOT NULL,
                    create_time     BIGINT,
                    user_openid     TEXT,
                    content         JSONB,
                    raw_body        TEXT,
                    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sig_verified    BOOLEAN NOT NULL DEFAULT FALSE,
                    handling_notes  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_tt_webhook_event   ON tiktok_webhook_events(event);
                CREATE INDEX IF NOT EXISTS idx_tt_webhook_openid  ON tiktok_webhook_events(user_openid);
                CREATE INDEX IF NOT EXISTS idx_tt_webhook_created ON tiktok_webhook_events(processed_at);
            """),
            (603, """
                CREATE TABLE IF NOT EXISTS platform_metrics_cache (
                    user_id   UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    data      JSONB NOT NULL DEFAULT '{}'::jsonb
                );
            """),

            # ── Per-upload engagement metrics (comments + shares) ────────────────
            (604, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS comments BIGINT DEFAULT 0;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS shares   BIGINT DEFAULT 0;
            """),

            # ── Analytics auto-sync tracking ─────────────────────────────────────
            (605, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS analytics_synced_at TIMESTAMPTZ;
                CREATE INDEX IF NOT EXISTS idx_uploads_analytics_sync
                    ON uploads(status, analytics_synced_at)
                    WHERE status IN ('completed', 'succeeded', 'partial');
            """),

            # ── Comprehensive audit system ───────────────────────────────────────
            (700, """
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS event_category  VARCHAR(50)  DEFAULT 'ADMIN';
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS actor_user_id   UUID         REFERENCES users(id) ON DELETE SET NULL;
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS resource_type   VARCHAR(100);
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS resource_id     TEXT;
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS session_id      TEXT;
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS user_agent      TEXT;
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS severity        VARCHAR(20)  DEFAULT 'INFO';
                ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS outcome         VARCHAR(20)  DEFAULT 'SUCCESS';

                CREATE INDEX IF NOT EXISTS idx_audit_category   ON admin_audit_log(event_category);
                CREATE INDEX IF NOT EXISTS idx_audit_actor      ON admin_audit_log(actor_user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_resource   ON admin_audit_log(resource_type, resource_id);
                CREATE INDEX IF NOT EXISTS idx_audit_severity   ON admin_audit_log(severity);

                CREATE TABLE IF NOT EXISTS system_event_log (
                    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         UUID        REFERENCES users(id) ON DELETE SET NULL,
                    event_category  VARCHAR(50) NOT NULL,
                    action          TEXT        NOT NULL,
                    resource_type   VARCHAR(100),
                    resource_id     TEXT,
                    details         JSONB       DEFAULT '{}'::jsonb,
                    ip_address      TEXT,
                    user_agent      TEXT,
                    session_id      TEXT,
                    severity        VARCHAR(20) DEFAULT 'INFO',
                    outcome         VARCHAR(20) DEFAULT 'SUCCESS',
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_syslog_user       ON system_event_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_syslog_category   ON system_event_log(event_category);
                CREATE INDEX IF NOT EXISTS idx_syslog_action     ON system_event_log(action);
                CREATE INDEX IF NOT EXISTS idx_syslog_created    ON system_event_log(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_syslog_resource   ON system_event_log(resource_type, resource_id);
            """),

            # ── Country column for geo-analytics ─────────────────────────────────
            (701, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS country VARCHAR(2);
                CREATE INDEX IF NOT EXISTS idx_users_country ON users(country) WHERE country IS NOT NULL;
            """),

            (702, """
                ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS token_hash TEXT;
                ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS used_at TIMESTAMPTZ;
                CREATE INDEX IF NOT EXISTS idx_password_resets_token_hash ON password_resets(token_hash)
                    WHERE token_hash IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_password_resets_user_unused ON password_resets(user_id)
                    WHERE used_at IS NULL;
            """),

            (703, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS target_accounts TEXT[] DEFAULT '{}';
            """),
            (704, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS deletion_requested_at TIMESTAMPTZ;
                CREATE INDEX IF NOT EXISTS idx_users_deletion_requested ON users(deletion_requested_at) WHERE deletion_requested_at IS NOT NULL;
            """),
            (705, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS schedule_metadata JSONB;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS timezone VARCHAR(100) DEFAULT 'UTC';
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS user_preferences JSONB;
            """),
            (706, """
                CREATE TABLE IF NOT EXISTS kpi_sync_state (
                    id INT PRIMARY KEY DEFAULT 1,
                    last_stripe_sync_at TIMESTAMPTZ,
                    last_mailgun_sync_at TIMESTAMPTZ,
                    last_openai_sync_at TIMESTAMPTZ,
                    last_cf_sync_at TIMESTAMPTZ,
                    last_upstash_sync_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                INSERT INTO kpi_sync_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
            """),
            (707, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}';
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_font_family VARCHAR(100) DEFAULT 'Arial';
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_font_size INT DEFAULT 24;
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS ffmpeg_screenshot_interval INT DEFAULT 5;
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS auto_generate_thumbnails BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS auto_generate_captions BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS auto_generate_hashtags BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS default_hashtag_count INT DEFAULT 5;
                ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS always_use_hashtags BOOLEAN DEFAULT FALSE;
            """),
            (1030, "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS styled_thumbnails BOOLEAN DEFAULT TRUE"),
            (1031, "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'"),
        ]

        for version, sql in migrations:
            if version not in applied:
                try:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                    logger.info(f"Migration v{version} applied")
                except Exception as e:
                    logger.error(f"Migration v{version} failed: {e}")


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
async def _cors_safe_500_handler(request: Request, exc: Exception) -> JSONResponse:
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
        "img-src 'self' data: https:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "script-src 'self' 'unsafe-inline' https:; "
        "connect-src 'self' https://api.stripe.com https://js.stripe.com "
        "https://uploadm8.com https://*.uploadm8.com;"
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
app.include_router(platforms_router)
app.include_router(billing_router)
app.include_router(webhooks_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(dashboard_router)
app.include_router(trill_router)
app.include_router(entitlements_router)
app.include_router(support_router)


# ============================================================
# Health
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_utc().isoformat()}
