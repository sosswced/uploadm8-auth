"""Versioned SQL migrations applied once at API startup.

Single source of truth: edit this file only. ``app.py`` lifespan calls
``run_migrations(db_pool)`` — do not duplicate migration lists elsewhere.

v1066 creates ``catalog_products`` + ``catalog_sync_log`` and seeds catalog rows
(equivalent to ``scripts/001_catalog_products.sql``; that file is kept for manual
``psql -f`` repair and must stay in sync with v1066).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("uploadm8-api")

# v1083 — keep in one place; also used by ensure_subscription_tier_constraint().
SUBSCRIPTION_TIER_CHECK_SQL = """
    -- creator_lite renamed from launch in app/Stripe catalog; constraint may still list launch only.
    ALTER TABLE users DROP CONSTRAINT IF EXISTS users_subscription_tier_check;
    ALTER TABLE users ADD CONSTRAINT users_subscription_tier_check
        CHECK (subscription_tier = ANY (ARRAY[
            'free'::text,
            'launch'::text,
            'creator_lite'::text,
            'creator_pro'::text,
            'studio'::text,
            'agency'::text,
            'friends_family'::text,
            'lifetime'::text,
            'master_admin'::text
        ]));
"""


async def ensure_subscription_tier_constraint(conn) -> bool:
    """Idempotent repair: allow ``creator_lite`` on users.subscription_tier."""
    try:
        defn = await conn.fetchval(
            """
            SELECT pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'users' AND c.conname = 'users_subscription_tier_check'
            """
        )
    except Exception as e:
        logger.warning("ensure_subscription_tier_constraint: could not read constraint: %s", e)
        defn = None
    if defn and "creator_lite" in str(defn):
        return False
    await conn.execute(SUBSCRIPTION_TIER_CHECK_SQL)
    logger.info("ensure_subscription_tier_constraint: applied creator_lite tier check")
    return True


async def ensure_subscription_tier_constraint_pool(db_pool) -> bool:
    async with db_pool.acquire() as conn:
        return await ensure_subscription_tier_constraint(conn)

# v1066 — must match scripts/001_catalog_products.sql (DDL + INSERT … ON CONFLICT DO NOTHING).
CATALOG_PRODUCTS_BOOTSTRAP_SQL = r"""
CREATE TABLE IF NOT EXISTS catalog_products (
    id                   SERIAL PRIMARY KEY,
    lookup_key           TEXT NOT NULL UNIQUE,
    stripe_product_id    TEXT,
    product_kind         TEXT NOT NULL,
    tier_slug            TEXT,
    sort_order           INT  NOT NULL DEFAULT 100,

    display_name         TEXT NOT NULL,
    stripe_product_name  TEXT NOT NULL,
    statement_descriptor TEXT NOT NULL,
    unit_label           TEXT NOT NULL,
    tax_code             TEXT NOT NULL DEFAULT 'txcd_10103001',

    price_usd            NUMERIC(10,2) NOT NULL,
    price_usd_yearly     NUMERIC(10,2),
    currency             TEXT NOT NULL DEFAULT 'usd',

    wallet               TEXT,
    token_amount         INT,

    put_monthly          INT,
    aic_monthly          INT,
    put_daily            INT,
    max_accounts         INT,
    max_accounts_per_platform INT,
    queue_depth          INT,
    lookahead_hours      INT,
    trial_days           INT,
    team_seats           INT,

    watermark            BOOLEAN DEFAULT FALSE,
    ads                  BOOLEAN DEFAULT FALSE,
    webhooks             BOOLEAN DEFAULT FALSE,
    white_label          BOOLEAN DEFAULT FALSE,
    hud                  BOOLEAN DEFAULT FALSE,
    excel                BOOLEAN DEFAULT FALSE,
    flex                 BOOLEAN DEFAULT FALSE,
    priority_class       TEXT,
    ai_depth             TEXT,
    analytics            TEXT,

    image_filename       TEXT NOT NULL,
    image_url            TEXT,
    image_hash           TEXT,

    is_internal          BOOLEAN DEFAULT FALSE,
    is_archived          BOOLEAN DEFAULT FALSE,

    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_synced_at       TIMESTAMPTZ,
    last_synced_by       TEXT
);

CREATE INDEX IF NOT EXISTS idx_catalog_products_kind ON catalog_products(product_kind);
CREATE INDEX IF NOT EXISTS idx_catalog_products_tier ON catalog_products(tier_slug);
CREATE INDEX IF NOT EXISTS idx_catalog_products_active
    ON catalog_products(is_archived) WHERE is_archived = FALSE;


CREATE TABLE IF NOT EXISTS catalog_sync_log (
    id                BIGSERIAL PRIMARY KEY,
    catalog_product_id INT REFERENCES catalog_products(id) ON DELETE CASCADE,
    lookup_key        TEXT NOT NULL,
    operation         TEXT NOT NULL,
    status            TEXT NOT NULL,
    stripe_response   JSONB,
    error_message     TEXT,
    actor             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_sync_log_product ON catalog_sync_log(catalog_product_id, created_at DESC);


INSERT INTO catalog_products (
    lookup_key, stripe_product_id, product_kind, tier_slug, sort_order,
    display_name, stripe_product_name, statement_descriptor, unit_label,
    price_usd, price_usd_yearly, put_monthly, aic_monthly, put_daily, max_accounts,
    max_accounts_per_platform, queue_depth, lookahead_hours, trial_days, team_seats,
    watermark, ads, webhooks, white_label, hud, excel, flex,
    priority_class, ai_depth, analytics, image_filename
) VALUES
    ('uploadm8_creatorlite_monthly', 'prod_UDD0jSiHkf0s0n', 'subscription', 'creator_lite', 10,
     'Creator Lite', 'UploadM8 Creator Lite - Monthly Subscription', 'UPLOADM8 CREATOR LITE', 'subscription',
     12, 120, 600, 200, 20, 10, 3, 100, 12, 7, 1,
     FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE,
     'p3', 'enhanced', 'standard', 'sub_creator_lite.png'),

    ('uploadm8_creatorpro_monthly', 'prod_UDD01YK0Fa5SW5', 'subscription', 'creator_pro', 20,
     'Creator Pro', 'UploadM8 Creator Pro - Monthly Subscription', 'UPLOADM8 CREATOR PRO', 'subscription',
     29, 290, 2000, 600, 60, 25, 6, 500, 24, 7, 3,
     FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE,
     'p2', 'advanced', 'full', 'sub_creator_pro.png'),

    ('uploadm8_studio_monthly', 'prod_UDD0NSQKtdQM7D', 'subscription', 'studio', 30,
     'Studio', 'UploadM8 Studio - Monthly Subscription', 'UPLOADM8 STUDIO', 'subscription',
     79, 790, 6000, 2000, 150, 75, 20, 2500, 72, 7, 10,
     FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE,
     'p1', 'max', 'full_export', 'sub_studio.png'),

    ('uploadm8_agency_monthly', 'prod_UDD0JV6l2sbrmM', 'subscription', 'agency', 40,
     'Agency', 'UploadM8 Agency - Monthly Subscription', 'UPLOADM8 AGENCY', 'subscription',
     199, 1990, 20000, 7000, 500, 300, 100, 99999, 168, 7, 25,
     FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE,
     'p0', 'max', 'full_export', 'sub_agency.png')
ON CONFLICT (lookup_key) DO NOTHING;


INSERT INTO catalog_products (
    lookup_key, stripe_product_id, product_kind, sort_order,
    display_name, stripe_product_name, statement_descriptor, unit_label,
    price_usd, wallet, token_amount, image_filename
) VALUES
    ('uploadm8_put_250',  'prod_UDD0VgzKZur5Qy', 'topup_put', 70,
     'PUT 250',  'UploadM8 PUT 250 - Upload Token Top-Up',   'UPLOADM8 PUT 250',  'token',
     4.99,  'put', 250,  'topup_put_250.png'),

    ('uploadm8_put_500',  'prod_UDCtVBGtibAVxy', 'topup_put', 80,
     'PUT 500',  'UploadM8 PUT 500 - Upload Token Top-Up',   'UPLOADM8 PUT 500',  'token',
     7.99,  'put', 500,  'topup_put_500.png'),

    ('uploadm8_put_1000', 'prod_UDCtoSuZPbF0yl', 'topup_put', 90,
     'PUT 1000', 'UploadM8 PUT 1,000 - Upload Token Top-Up', 'UPLOADM8 PUT 1000', 'token',
     14.99, 'put', 1000, 'topup_put_1000.png'),

    ('uploadm8_put_2500', NULL, 'topup_put', 95,
     'PUT 2500', 'UploadM8 PUT 2,500 - Upload Token Top-Up', 'UPLOADM8 PUT 2500', 'token',
     29.99, 'put', 2500, 'topup_put_2500.png'),

    ('uploadm8_put_5000', NULL, 'topup_put', 96,
     'PUT 5000', 'UploadM8 PUT 5,000 - Upload Token Top-Up', 'UPLOADM8 PUT 5000', 'token',
     49.99, 'put', 5000, 'topup_put_5000.png')
ON CONFLICT (lookup_key) DO NOTHING;


INSERT INTO catalog_products (
    lookup_key, stripe_product_id, product_kind, sort_order,
    display_name, stripe_product_name, statement_descriptor, unit_label,
    price_usd, wallet, token_amount, image_filename
) VALUES
    ('uploadm8_aic_250',  NULL, 'topup_aic', 130,
     'AIC 250',  'UploadM8 AIC 250 - AI Credit Token Top-Up',   'UPLOADM8 AIC 250',   'token',
     4.99,  'aic', 250,  'topup_aic_250.png'),

    ('uploadm8_aic_500',  'prod_UDD0CCQzccfHTY', 'topup_aic', 140,
     'AIC 500',  'UploadM8 AIC 500 - AI Credit Token Top-Up',   'UPLOADM8 AIC 500',   'token',
     7.99,  'aic', 500,  'topup_aic_500.png'),

    ('uploadm8_aic_1000', 'prod_UDCt8PqZ9z1VnF', 'topup_aic', 150,
     'AIC 1000', 'UploadM8 AIC 1,000 - AI Credit Token Top-Up', 'UPLOADM8 AIC 1000', 'token',
     14.99, 'aic', 1000, 'topup_aic_1000.png'),

    ('uploadm8_aic_2500', NULL, 'topup_aic', 155,
     'AIC 2500', 'UploadM8 AIC 2,500 - AI Credit Token Top-Up', 'UPLOADM8 AIC 2500', 'token',
     29.99, 'aic', 2500, 'topup_aic_2500.png'),

    ('uploadm8_aic_5000', NULL, 'topup_aic', 156,
     'AIC 5000', 'UploadM8 AIC 5,000 - AI Credit Token Top-Up', 'UPLOADM8 AIC 5000', 'token',
     49.99, 'aic', 5000, 'topup_aic_5000.png'),

    ('uploadm8_aic_10000', NULL, 'topup_aic', 157,
     'AIC 10000', 'UploadM8 AIC 10,000 - AI Credit Token Top-Up', 'UPLOADM8 AIC 10000', 'token',
     79.99, 'aic', 10000, 'topup_aic_10000.png')
ON CONFLICT (lookup_key) DO NOTHING;
"""

async def run_migrations(db_pool):
    async with db_pool.acquire() as conn:
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
            (1032, """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    key_hash        VARCHAR(255) NOT NULL UNIQUE,
                    key_prefix      VARCHAR(12) NOT NULL,
                    name            VARCHAR(255) NOT NULL DEFAULT 'Default',
                    scopes          TEXT[] DEFAULT '{read}',
                    rate_limit      INT DEFAULT 100,
                    last_used_at    TIMESTAMPTZ,
                    expires_at      TIMESTAMPTZ,
                    revoked_at      TIMESTAMPTZ,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id) WHERE revoked_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash) WHERE revoked_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
            """),
            (1033, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN;
                COMMENT ON COLUMN users.email_verified IS 'NULL=legacy verified; false=pending signup verify; true=verified';

                CREATE TABLE IF NOT EXISTS signup_verifications (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR(64) NOT NULL UNIQUE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    used_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_signup_verifications_user ON signup_verifications(user_id);

                CREATE TABLE IF NOT EXISTS marketing_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    event_type VARCHAR(80) NOT NULL,
                    payload JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_marketing_events_created ON marketing_events(created_at);
            """),
            (1035, """
                CREATE TABLE IF NOT EXISTS platform_content_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    platform_token_id UUID REFERENCES platform_tokens(id) ON DELETE SET NULL,
                    platform VARCHAR(50) NOT NULL,
                    account_id TEXT NOT NULL DEFAULT '',
                    platform_video_id TEXT NOT NULL,
                    upload_id UUID REFERENCES uploads(id) ON DELETE SET NULL,
                    source VARCHAR(30) NOT NULL DEFAULT 'external',
                    content_kind VARCHAR(50),
                    title TEXT,
                    published_at TIMESTAMPTZ,
                    thumbnail_url TEXT,
                    platform_url TEXT,
                    duration_seconds INT,
                    views BIGINT DEFAULT 0,
                    likes BIGINT DEFAULT 0,
                    comments BIGINT DEFAULT 0,
                    shares BIGINT DEFAULT 0,
                    visibility VARCHAR(80),
                    presence VARCHAR(80),
                    metrics_synced_at TIMESTAMPTZ,
                    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, platform, account_id, platform_video_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pci_user ON platform_content_items (user_id);
                CREATE INDEX IF NOT EXISTS idx_pci_user_platform ON platform_content_items (user_id, platform);
                CREATE INDEX IF NOT EXISTS idx_pci_upload ON platform_content_items (upload_id)
                    WHERE upload_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS platform_content_sync_state (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    platform_token_id UUID NOT NULL REFERENCES platform_tokens(id) ON DELETE CASCADE,
                    platform VARCHAR(50) NOT NULL,
                    account_id TEXT NOT NULL DEFAULT '',
                    last_synced_at TIMESTAMPTZ,
                    next_cursor TEXT,
                    total_discovered INT DEFAULT 0,
                    total_linked INT DEFAULT 0,
                    status VARCHAR(50),
                    error_detail TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, platform_token_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pcss_user ON platform_content_sync_state (user_id);
            """),
            (1036, """
                CREATE TABLE IF NOT EXISTS studio_usage_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    operation VARCHAR(80) NOT NULL,
                    http_status INT,
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_studio_usage_created ON studio_usage_events(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_studio_usage_user_created ON studio_usage_events(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS marketing_campaigns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(500) NOT NULL DEFAULT 'Campaign',
                    objective TEXT,
                    channel VARCHAR(50) DEFAULT 'in_app',
                    status VARCHAR(50) DEFAULT 'draft',
                    estimated_audience INT DEFAULT 0,
                    schedule_at TIMESTAMPTZ,
                    targeting JSONB NOT NULL DEFAULT '{}'::jsonb,
                    notes TEXT,
                    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_marketing_campaigns_status ON marketing_campaigns(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS marketing_ai_decisions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    action VARCHAR(80) NOT NULL,
                    status VARCHAR(80),
                    objective TEXT,
                    range_key VARCHAR(32),
                    confidence_score DOUBLE PRECISION,
                    used_openai BOOLEAN NOT NULL DEFAULT FALSE,
                    truth_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                    plan_json JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_marketing_ai_decisions_created ON marketing_ai_decisions(created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_marketing_events_type_created
                    ON marketing_events(event_type, created_at DESC);
            """),
            (1037, """
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS range_key VARCHAR(32) DEFAULT '30d';
                UPDATE marketing_campaigns SET range_key = '30d' WHERE range_key IS NULL;

                CREATE TABLE IF NOT EXISTS marketing_automation_runs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at TIMESTAMPTZ,
                    status VARCHAR(32) NOT NULL DEFAULT 'running',
                    mode VARCHAR(48) NOT NULL DEFAULT 'touchpoints_v1',
                    segment_key VARCHAR(96),
                    users_evaluated INT DEFAULT 0,
                    users_messaged INT DEFAULT 0,
                    email_sent INT DEFAULT 0,
                    discord_sent INT DEFAULT 0,
                    in_app_written INT DEFAULT 0,
                    skipped_dedupe INT DEFAULT 0,
                    error_detail TEXT,
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS marketing_touchpoint_deliveries (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    channel VARCHAR(32) NOT NULL,
                    subject VARCHAR(500),
                    body_text TEXT,
                    body_html TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    scheduled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sent_at TIMESTAMPTZ,
                    error_detail TEXT,
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_mtd_user_chan_stat
                    ON marketing_touchpoint_deliveries(user_id, channel, status, scheduled_at DESC);
                CREATE INDEX IF NOT EXISTS idx_mtd_pending_sched
                    ON marketing_touchpoint_deliveries(scheduled_at)
                    WHERE status = 'pending';
            """),
            (1038, """
                ALTER TABLE marketing_touchpoint_deliveries
                    ADD COLUMN IF NOT EXISTS campaign_id UUID REFERENCES marketing_campaigns(id) ON DELETE SET NULL;
                CREATE INDEX IF NOT EXISTS idx_mtd_campaign_user
                    ON marketing_touchpoint_deliveries(campaign_id, user_id, created_at DESC);

                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS approved_by UUID REFERENCES users(id) ON DELETE SET NULL;
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS template_subject VARCHAR(500);
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS template_body_html TEXT;
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS template_body_text TEXT;
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS discord_message_text TEXT;

                CREATE TABLE IF NOT EXISTS user_marketing_consent (
                    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    email_marketing BOOLEAN NOT NULL DEFAULT TRUE,
                    discord_marketing BOOLEAN NOT NULL DEFAULT FALSE,
                    allow_pii_in_ml BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS marketing_suppressions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    channel VARCHAR(32) NOT NULL,
                    reason VARCHAR(160),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, channel)
                );
                CREATE INDEX IF NOT EXISTS idx_msup_user ON marketing_suppressions(user_id);

                CREATE TABLE IF NOT EXISTS marketing_admin_alerts (
                    alert_key VARCHAR(160) PRIMARY KEY,
                    last_sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS ml_outcome_labels (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    upload_id UUID REFERENCES uploads(id) ON DELETE SET NULL,
                    variant_id VARCHAR(128),
                    feature_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                    label_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_ml_outcome_user_created
                    ON ml_outcome_labels(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ml_outcome_variant
                    ON ml_outcome_labels(variant_id) WHERE variant_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS ml_model_promotion_audit (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_key VARCHAR(80) NOT NULL,
                    baseline_rate DOUBLE PRECISION,
                    model_rate DOUBLE PRECISION,
                    lift DOUBLE PRECISION,
                    promoted BOOLEAN NOT NULL DEFAULT FALSE,
                    sample_n INT,
                    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_ml_promotion_eval
                    ON ml_model_promotion_audit(evaluated_at DESC);

                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS studio_content_variant_id VARCHAR(128);
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS content_variant_meta JSONB NOT NULL DEFAULT '{}'::jsonb;
            """),
            (1039, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS output_artifacts JSONB NOT NULL DEFAULT '{}'::jsonb;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processed_assets JSONB;
            """),
            (1040, """
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS auth_security_alerts BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS digest_emails BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS scheduled_alert_emails BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS use_audio_context BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS audio_transcription BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_telemetry BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_audio_signals BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_music_detection BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_audio_summary BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_emotion_signals BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_caption_writer BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_thumbnail_designer BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_frame_inspector BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_speech_to_text BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_video_analyzer BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ai_service_scene_understanding BOOLEAN DEFAULT TRUE;
            """),
            (1041, """
                CREATE TABLE IF NOT EXISTS upload_quality_scores_daily (
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    day DATE NOT NULL,
                    platform VARCHAR(50) NOT NULL,
                    strategy_key VARCHAR(512) NOT NULL,
                    samples INT NOT NULL DEFAULT 0,
                    mean_engagement DOUBLE PRECISION,
                    mean_views DOUBLE PRECISION,
                    engagement_stddev DOUBLE PRECISION,
                    ci95_low DOUBLE PRECISION,
                    ci95_high DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, day, platform, strategy_key)
                );
                CREATE INDEX IF NOT EXISTS idx_uqsd_user_day
                    ON upload_quality_scores_daily(user_id, day DESC);
                CREATE INDEX IF NOT EXISTS idx_uqsd_user_strategy
                    ON upload_quality_scores_daily(user_id, strategy_key);
            """),
            (1042, """
                CREATE TABLE IF NOT EXISTS operational_incidents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source VARCHAR(50) NOT NULL,
                    incident_type VARCHAR(120) NOT NULL,
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    upload_id UUID,
                    subject TEXT NOT NULL,
                    body TEXT,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    screenshot_r2_key VARCHAR(512),
                    email_sent_at TIMESTAMPTZ,
                    discord_sent_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_operational_incidents_created
                    ON operational_incidents(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_operational_incidents_type
                    ON operational_incidents(incident_type);
                CREATE INDEX IF NOT EXISTS idx_operational_incidents_upload
                    ON operational_incidents(upload_id) WHERE upload_id IS NOT NULL;
            """),
            (1043, """
                CREATE INDEX IF NOT EXISTS idx_uploads_user_created_at
                    ON uploads (user_id, created_at DESC);
            """),
            (1044, """
                CREATE INDEX IF NOT EXISTS idx_pci_user_published_at
                    ON platform_content_items (user_id, published_at DESC)
                    WHERE published_at IS NOT NULL;
            """),
            (1045, """
                CREATE TABLE IF NOT EXISTS m8_publish_hour_priors (
                    platform VARCHAR(50) NOT NULL,
                    hour_utc SMALLINT NOT NULL,
                    prior_weight DOUBLE PRECISION NOT NULL,
                    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    training_run_id UUID NOT NULL,
                    train_row_count INT NOT NULL DEFAULT 0,
                    model_version VARCHAR(40) NOT NULL DEFAULT 'hgb-v1',
                    val_mae_log1p_views DOUBLE PRECISION,
                    PRIMARY KEY (platform, hour_utc)
                );
                CREATE INDEX IF NOT EXISTS idx_m8_publish_hour_priors_trained
                    ON m8_publish_hour_priors (trained_at DESC);
            """),
            (1046, """
                CREATE TABLE IF NOT EXISTS m8_model_runs (
                    id UUID PRIMARY KEY,
                    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_version VARCHAR(40) NOT NULL,
                    train_row_count INT NOT NULL DEFAULT 0,
                    val_mae_log1p_views DOUBLE PRECISION,
                    features_used JSONB NOT NULL DEFAULT '[]'::jsonb,
                    train_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                    metrics JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_m8_model_runs_trained
                    ON m8_model_runs (trained_at DESC);
            """),
            # Legacy DBs: users created before flex_enabled was in schema — presign/wallet read this column.
            (1047, "ALTER TABLE users ADD COLUMN IF NOT EXISTS flex_enabled BOOLEAN DEFAULT FALSE"),
            # Thumbnail Studio (Pikzels recreate, personas, cdn-preview keyed by variant row id).
            (1048, """
                CREATE TABLE IF NOT EXISTS creator_personas (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    image_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_creator_personas_user_created
                    ON creator_personas(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS creator_persona_images (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    persona_id UUID NOT NULL REFERENCES creator_personas(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    image_url TEXT NOT NULL,
                    quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_creator_persona_images_persona
                    ON creator_persona_images(persona_id);

                CREATE TABLE IF NOT EXISTS thumbnail_recreate_jobs (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    youtube_url TEXT NOT NULL,
                    youtube_video_id VARCHAR(32),
                    source_title VARCHAR(512) NOT NULL DEFAULT '',
                    topic VARCHAR(512) NOT NULL DEFAULT '',
                    niche VARCHAR(120) NOT NULL DEFAULT 'general',
                    closeness INT NOT NULL DEFAULT 55,
                    variant_count INT NOT NULL DEFAULT 6,
                    persona_id UUID REFERENCES creator_personas(id) ON DELETE SET NULL,
                    competitor_gap_mode BOOLEAN NOT NULL DEFAULT FALSE,
                    put_cost INT NOT NULL DEFAULT 0,
                    aic_cost INT NOT NULL DEFAULT 0,
                    breakdown_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_thumbnail_recreate_jobs_user_created
                    ON thumbnail_recreate_jobs(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS thumbnail_recreate_variants (
                    id UUID PRIMARY KEY,
                    job_id UUID NOT NULL REFERENCES thumbnail_recreate_jobs(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    rank_idx INT NOT NULL,
                    variant_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    selected BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_thumbnail_recreate_variants_job_rank
                    ON thumbnail_recreate_variants(job_id, rank_idx);
                CREATE INDEX IF NOT EXISTS idx_thumbnail_recreate_variants_user_job
                    ON thumbnail_recreate_variants(user_id, job_id);

                CREATE TABLE IF NOT EXISTS thumbnail_recreate_feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    job_id UUID NOT NULL REFERENCES thumbnail_recreate_jobs(id) ON DELETE CASCADE,
                    variant_id UUID REFERENCES thumbnail_recreate_variants(id) ON DELETE SET NULL,
                    event_type VARCHAR(64) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_thumbnail_recreate_feedback_job_created
                    ON thumbnail_recreate_feedback(job_id, created_at DESC);
            """),
            # Recover hashtag JSONB columns that were double-encoded by old write
            # paths (json.dumps() on top of the asyncpg JSONB codec). Symptom on
            # disk is a JSONB string value like `"[\"tester\",\"qwe\"]"` instead
            # of a real JSONB array. We parse the inner JSON back into the
            # proper structure when the outer value is a JSONB string of array/
            # object shape.
            # ── Admin email/digest jobs (trial reminders + monthly digest + admin digest)
            #
            # ``trial_reminder_sent`` lets the daily ``run_trial_reminders`` job avoid
            # re-emailing the same user. ``last_monthly_digest_sent_at`` is the
            # idempotency anchor for the per-user monthly digest job. ``admin_email_job_runs``
            # is a small ledger so the admin UI can show last-run status for each job.
            (1050, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_end TIMESTAMPTZ;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_reminder_sent TIMESTAMPTZ;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_monthly_digest_sent_at TIMESTAMPTZ;
                CREATE INDEX IF NOT EXISTS idx_users_trial_reminder_pending
                    ON users (trial_end)
                    WHERE subscription_status = 'trialing'
                      AND trial_reminder_sent IS NULL
                      AND trial_end IS NOT NULL;

                CREATE TABLE IF NOT EXISTS admin_email_job_runs (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job             VARCHAR(64) NOT NULL,
                    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at     TIMESTAMPTZ,
                    sent_count      INT NOT NULL DEFAULT 0,
                    skipped_count   INT NOT NULL DEFAULT 0,
                    error_count     INT NOT NULL DEFAULT 0,
                    triggered_by    VARCHAR(32) NOT NULL DEFAULT 'manual',
                    summary         JSONB NOT NULL DEFAULT '{}'::jsonb,
                    error_message   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_admin_email_job_runs_job_started
                    ON admin_email_job_runs (job, started_at DESC);
            """),
            (1049, """
                DO $$
                BEGIN
                    -- always_hashtags: jsonb string starting with '['  →  parse to array
                    UPDATE user_preferences
                       SET always_hashtags = (always_hashtags #>> '{}')::jsonb
                     WHERE jsonb_typeof(always_hashtags) = 'string'
                       AND (always_hashtags #>> '{}') ~ '^\\s*\\[';

                    -- blocked_hashtags
                    UPDATE user_preferences
                       SET blocked_hashtags = (blocked_hashtags #>> '{}')::jsonb
                     WHERE jsonb_typeof(blocked_hashtags) = 'string'
                       AND (blocked_hashtags #>> '{}') ~ '^\\s*\\[';

                    -- platform_hashtags: jsonb string starting with '{' → parse to object
                    UPDATE user_preferences
                       SET platform_hashtags = (platform_hashtags #>> '{}')::jsonb
                     WHERE jsonb_typeof(platform_hashtags) = 'string'
                       AND (platform_hashtags #>> '{}') ~ '^\\s*\\{';

                    -- users.preferences entire blob double-encoded (less common
                    -- but possible from /api/me/preferences before the fix).
                    UPDATE users
                       SET preferences = (preferences #>> '{}')::jsonb
                     WHERE jsonb_typeof(preferences) = 'string'
                       AND (preferences #>> '{}') ~ '^\\s*\\{';
                EXCEPTION WHEN OTHERS THEN
                    -- Never block startup over recovery; bad rows can still be
                    -- handled by the runtime coerce_jsonb_* helpers.
                    RAISE NOTICE 'hashtag JSONB recovery skipped: %', SQLERRM;
                END $$;
            """),
            # Must be a distinct version from the other 1050 block above; duplicate
            # version numbers caused this migration to be skipped after admin email 1050.
            (1051, """
                -- Pikzels multi-tenant hardening:
                -- - Track link state on UploadM8 personas so concurrent Link calls cannot
                --   create duplicate upstream pikzonalities for the same local row.
                -- - Track any generic Pikzels persona/style pikzonalities created through
                --   our proxy so future proxy calls can enforce user ownership.
                -- - Track best-effort remote delete failures after local persona deletion.
                ALTER TABLE creator_personas
                    ADD COLUMN IF NOT EXISTS link_status VARCHAR(24) NOT NULL DEFAULT 'local',
                    ADD COLUMN IF NOT EXISTS link_error TEXT,
                    ADD COLUMN IF NOT EXISTS link_started_at TIMESTAMPTZ,
                    ADD COLUMN IF NOT EXISTS link_completed_at TIMESTAMPTZ,
                    ADD COLUMN IF NOT EXISTS remote_delete_status VARCHAR(24),
                    ADD COLUMN IF NOT EXISTS remote_delete_error TEXT,
                    ADD COLUMN IF NOT EXISTS remote_delete_requested_at TIMESTAMPTZ;

                CREATE TABLE IF NOT EXISTS pikzels_user_assets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    kind VARCHAR(24) NOT NULL CHECK (kind IN ('persona', 'style')),
                    local_persona_id UUID REFERENCES creator_personas(id) ON DELETE SET NULL,
                    pikzels_pikzonality_id UUID NOT NULL,
                    name VARCHAR(255) NOT NULL DEFAULT '',
                    status VARCHAR(24) NOT NULL DEFAULT 'linked',
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_pikzels_user_assets_user_pkz
                    ON pikzels_user_assets(user_id, pikzels_pikzonality_id);
                CREATE INDEX IF NOT EXISTS idx_pikzels_user_assets_user_kind
                    ON pikzels_user_assets(user_id, kind, created_at DESC);

                CREATE TABLE IF NOT EXISTS pikzels_remote_deletes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    local_persona_id UUID,
                    pikzels_pikzonality_id UUID NOT NULL,
                    kind VARCHAR(24) NOT NULL DEFAULT 'persona',
                    status VARCHAR(24) NOT NULL DEFAULT 'pending',
                    http_status INT,
                    error_code VARCHAR(120),
                    error_message TEXT,
                    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_pikzels_remote_deletes_status
                    ON pikzels_remote_deletes(status, requested_at DESC);
            """),
            # Phase-3 video recognition tables: object/text/person/logo tracks
            # extracted by stages.video_intelligence_stage. Stored alongside
            # uploads so the thumbnail studio, ML feedback loop, analytics,
            # and admin KPI surfaces can query "what was actually in this clip"
            # without re-running Video Intelligence.
            #
            # Design choices:
            # - One row per detection (object track, text region, person
            #   segment, logo). Lets us index on (user_id, kind, description)
            #   for cheap "videos that contain a Tesla" / "uploads with logo X"
            #   queries.
            # - JSONB ``frames`` column for object_tracks captures bounding
            #   boxes at first/middle/last so the thumbnail keyframe selector
            #   can pick the most-on-screen moment without re-querying VI.
            # - Recognition_summaries is one-row-per-upload for fast widget
            #   rendering on the upload detail page and admin KPIs.
            (1052, """
                CREATE TABLE IF NOT EXISTS video_recognition (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    upload_id       UUID NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
                    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    kind            VARCHAR(24) NOT NULL CHECK (kind IN ('object', 'text', 'person', 'logo')),
                    description     VARCHAR(255) NOT NULL DEFAULT '',
                    confidence      DOUBLE PRECISION NOT NULL DEFAULT 0,
                    start_seconds   DOUBLE PRECISION NOT NULL DEFAULT 0,
                    end_seconds     DOUBLE PRECISION NOT NULL DEFAULT 0,
                    frames          JSONB NOT NULL DEFAULT '[]'::jsonb,
                    attributes      JSONB NOT NULL DEFAULT '{}'::jsonb,
                    raw             JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_video_recognition_upload
                    ON video_recognition(upload_id, kind, confidence DESC);
                CREATE INDEX IF NOT EXISTS idx_video_recognition_user_kind
                    ON video_recognition(user_id, kind, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_video_recognition_description
                    ON video_recognition(user_id, kind, lower(description));

                CREATE TABLE IF NOT EXISTS upload_recognition_summary (
                    upload_id              UUID PRIMARY KEY REFERENCES uploads(id) ON DELETE CASCADE,
                    user_id                UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    object_track_count     INT NOT NULL DEFAULT 0,
                    person_segment_count   INT NOT NULL DEFAULT 0,
                    text_detection_count   INT NOT NULL DEFAULT 0,
                    logo_count             INT NOT NULL DEFAULT 0,
                    top_objects            TEXT[] NOT NULL DEFAULT '{}',
                    top_logos              TEXT[] NOT NULL DEFAULT '{}',
                    top_text               TEXT[] NOT NULL DEFAULT '{}',
                    has_people             BOOLEAN NOT NULL DEFAULT FALSE,
                    coverage_seconds       DOUBLE PRECISION NOT NULL DEFAULT 0,
                    summary_text           TEXT NOT NULL DEFAULT '',
                    hydration_score        DOUBLE PRECISION NOT NULL DEFAULT 0,
                    raw_summary            JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_recognition_summary_user_recent
                    ON upload_recognition_summary(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_recognition_summary_top_objects
                    ON upload_recognition_summary USING gin(top_objects);
                CREATE INDEX IF NOT EXISTS idx_recognition_summary_top_logos
                    ON upload_recognition_summary USING gin(top_logos);
                CREATE INDEX IF NOT EXISTS idx_recognition_summary_hydration
                    ON upload_recognition_summary(hydration_score DESC);
            """),
            # Default new accounts to auto thumbnails + Trill on (NULL-safe GET still applies).
            (1053, """
                ALTER TABLE user_preferences ALTER COLUMN auto_thumbnails SET DEFAULT TRUE;
                ALTER TABLE user_preferences ALTER COLUMN trill_enabled SET DEFAULT TRUE;
            """),
            (1054, """
                ALTER TABLE marketing_campaigns ADD COLUMN IF NOT EXISTS promo_media JSONB NOT NULL DEFAULT '{}'::jsonb;
                ALTER TABLE announcements ADD COLUMN IF NOT EXISTS promo_media JSONB NOT NULL DEFAULT '{}'::jsonb;
                CREATE TABLE IF NOT EXISTS marketing_promo_media_runs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entity_kind VARCHAR(32) NOT NULL,
                    entity_id UUID,
                    variant_id VARCHAR(64) NOT NULL DEFAULT '',
                    http_status INT,
                    ok BOOLEAN NOT NULL DEFAULT FALSE,
                    detail TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_marketing_promo_media_runs_created
                    ON marketing_promo_media_runs(created_at DESC);
            """),
            (1055, """
                ALTER TABLE token_ledger ADD COLUMN IF NOT EXISTS ref_type VARCHAR(50);
                CREATE INDEX IF NOT EXISTS idx_token_ledger_user_created
                    ON token_ledger(user_id, created_at DESC);
            """),
            (1056, """
                CREATE TABLE IF NOT EXISTS wallet_disputes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    ledger_id UUID NOT NULL REFERENCES token_ledger(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    status VARCHAR(24) NOT NULL DEFAULT 'open',
                    note TEXT NOT NULL DEFAULT '',
                    admin_internal_note TEXT,
                    resolution_message TEXT,
                    operational_incident_id UUID REFERENCES operational_incidents(id) ON DELETE SET NULL,
                    user_email_sent_at TIMESTAMPTZ,
                    user_discord_sent_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    resolved_at TIMESTAMPTZ,
                    CONSTRAINT wallet_disputes_status_chk CHECK (
                        status IN ('open', 'in_review', 'resolved', 'rejected')
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_wallet_disputes_user_created
                    ON wallet_disputes(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_wallet_disputes_status
                    ON wallet_disputes(status, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_wallet_disputes_one_open_per_ledger
                    ON wallet_disputes(ledger_id)
                    WHERE status IN ('open', 'in_review');
            """),
            (1057, """
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_leaderboard_opt_in BOOLEAN NOT NULL DEFAULT FALSE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_map_sharing_opt_in BOOLEAN NOT NULL DEFAULT FALSE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_welcome_modal_seen_at TIMESTAMPTZ NULL;
            """),
            (1058, """
                CREATE TABLE IF NOT EXISTS marketing_campaign_audience (
                    campaign_id UUID NOT NULL REFERENCES marketing_campaigns(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    rendered_image_url TEXT,
                    variant_id VARCHAR(128),
                    rendered_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (campaign_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_mca_campaign_pending
                    ON marketing_campaign_audience(campaign_id)
                    WHERE rendered_image_url IS NULL;
            """),
            # PostGIS + PAD-US spatial index (table is optional; created by scripts/load_padus.py).
            (1059, """
                CREATE EXTENSION IF NOT EXISTS postgis;
                DO $padus_idx$
                BEGIN
                    IF to_regclass('public.padus_protected_areas') IS NOT NULL THEN
                        BEGIN
                            EXECUTE $sql$
                                CREATE INDEX IF NOT EXISTS idx_padus_protected_areas_geom_gist
                                ON public.padus_protected_areas USING GIST (geometry)
                            $sql$;
                        EXCEPTION
                            WHEN undefined_column THEN
                                NULL;
                        END;
                    END IF;
                END
                $padus_idx$;
            """),
            (1060, """
                CREATE TABLE IF NOT EXISTS billing_service_weights (
                    service_id TEXT PRIMARY KEY,
                    aic_weight INT NOT NULL CHECK (aic_weight >= 0 AND aic_weight <= 5000),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_by UUID REFERENCES users(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_billing_service_weights_updated
                    ON billing_service_weights(updated_at DESC);
            """),
            (1061, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS billing_breakdown JSONB;
            """),
            (1062, """
                CREATE TABLE IF NOT EXISTS vehicle_makes (
                    id SERIAL PRIMARY KEY,
                    nhtsa_make_id INT UNIQUE,
                    name TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vehicle_makes_name_lower
                    ON vehicle_makes (LOWER(name));
                CREATE TABLE IF NOT EXISTS vehicle_models (
                    id SERIAL PRIMARY KEY,
                    make_id INT NOT NULL REFERENCES vehicle_makes(id) ON DELETE CASCADE,
                    nhtsa_model_id INT,
                    name TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicle_models_make_nhtsa_uid
                    ON vehicle_models(make_id, nhtsa_model_id)
                    WHERE nhtsa_model_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_vehicle_models_make ON vehicle_models(make_id);
                CREATE INDEX IF NOT EXISTS idx_vehicle_models_name_lower
                    ON vehicle_models (LOWER(name));
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS default_vehicle_make_id INT REFERENCES vehicle_makes(id) ON DELETE SET NULL;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS default_vehicle_model_id INT REFERENCES vehicle_models(id) ON DELETE SET NULL;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS vehicle_make_id INT REFERENCES vehicle_makes(id) ON DELETE SET NULL;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS vehicle_model_id INT REFERENCES vehicle_models(id) ON DELETE SET NULL;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_public_name VARCHAR(64) NULL;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_public_name_pending VARCHAR(64) NULL;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_public_name_status VARCHAR(20) NOT NULL DEFAULT 'none';
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_public_name_rejection_reason TEXT NULL;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_public_name_reviewed_at TIMESTAMPTZ NULL;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_public_name_reviewed_by UUID NULL REFERENCES users(id) ON DELETE SET NULL;
            """),
            (1063, """
                ALTER TABLE m8_model_runs ADD COLUMN IF NOT EXISTS related_ops_incident_ids UUID[] NOT NULL DEFAULT '{}'::uuid[];
                CREATE INDEX IF NOT EXISTS idx_m8_model_runs_related_incidents
                    ON m8_model_runs USING gin (related_ops_incident_ids);
            """),
            (1064, """
                CREATE TABLE IF NOT EXISTS billing_catalog (
                    id INT PRIMARY KEY CHECK (id = 1),
                    tier_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
                    topup_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    last_sync_at TIMESTAMPTZ,
                    last_sync_ok BOOLEAN,
                    last_sync_error TEXT,
                    last_sync_detail JSONB
                );
                INSERT INTO billing_catalog (id) VALUES (1)
                ON CONFLICT (id) DO NOTHING;
            """),
            (1065, """
                CREATE TABLE IF NOT EXISTS marketing_approval_tickets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    campaign_id UUID NOT NULL REFERENCES marketing_campaigns(id) ON DELETE CASCADE,
                    status VARCHAR(24) NOT NULL DEFAULT 'open',
                    submitted_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    resolved_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    resolution_notes TEXT,
                    copy_snapshot_hash VARCHAR(128),
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    resolved_at TIMESTAMPTZ,
                    CONSTRAINT marketing_approval_tickets_status_chk CHECK (
                        status IN ('open', 'in_review', 'approved', 'rejected')
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_mat_campaign_created
                    ON marketing_approval_tickets(campaign_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_mat_status_created
                    ON marketing_approval_tickets(status, created_at DESC);

                INSERT INTO marketing_approval_tickets (
                    campaign_id, status, submitted_by, resolved_by, resolution_notes, created_at, resolved_at, copy_snapshot_hash
                )
                SELECT c.id, 'approved', c.approved_by, c.approved_by, 'legacy_backfill',
                       COALESCE(c.approved_at, c.updated_at), c.approved_at, NULL
                FROM marketing_campaigns c
                WHERE c.approved_at IS NOT NULL
                  AND LOWER(COALESCE(c.channel, '')) IN ('email', 'discord', 'mixed')
                  AND NOT EXISTS (
                      SELECT 1 FROM marketing_approval_tickets t
                      WHERE t.campaign_id = c.id AND t.status = 'approved' AND t.resolved_by IS NOT NULL
                  );
            """),
            (1066, CATALOG_PRODUCTS_BOOTSTRAP_SQL),
            (1068, """
                ALTER TABLE vehicle_makes ADD COLUMN IF NOT EXISTS consumer_vehicle BOOLEAN NOT NULL DEFAULT FALSE;
                CREATE INDEX IF NOT EXISTS idx_vehicle_makes_consumer_name
                    ON vehicle_makes (consumer_vehicle, LOWER(name))
                    WHERE consumer_vehicle = TRUE;
            """),
            (1070, """
                CREATE TABLE IF NOT EXISTS trill_badge_definitions (
                    id SERIAL PRIMARY KEY,
                    slug VARCHAR(64) NOT NULL UNIQUE,
                    title VARCHAR(120) NOT NULL,
                    description TEXT,
                    icon VARCHAR(64) NOT NULL DEFAULT 'fa-medal',
                    tier VARCHAR(16) NOT NULL DEFAULT 'bronze',
                    category VARCHAR(32) NOT NULL DEFAULT 'general',
                    sort_order INT NOT NULL DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                );
                CREATE TABLE IF NOT EXISTS trill_user_badges (
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    badge_id INT NOT NULL REFERENCES trill_badge_definitions(id) ON DELETE CASCADE,
                    earned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    PRIMARY KEY (user_id, badge_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trill_user_badges_user ON trill_user_badges(user_id);
                CREATE TABLE IF NOT EXISTS trill_seasons (
                    id SERIAL PRIMARY KEY,
                    slug VARCHAR(16) NOT NULL UNIQUE,
                    starts_at TIMESTAMPTZ NOT NULL,
                    ends_at TIMESTAMPTZ NOT NULL,
                    status VARCHAR(16) NOT NULL DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS trill_hall_of_fame (
                    id SERIAL PRIMARY KEY,
                    season_id INT NOT NULL REFERENCES trill_seasons(id) ON DELETE CASCADE,
                    rank INT NOT NULL,
                    user_id UUID NOT NULL,
                    driver_handle VARCHAR(64) NOT NULL,
                    best_trill_score NUMERIC(5,2),
                    category VARCHAR(32) NOT NULL DEFAULT 'overall',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS trill_rivals (
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    rival_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, rival_user_id)
                );
                CREATE TABLE IF NOT EXISTS trill_weekly_challenges (
                    id SERIAL PRIMARY KEY,
                    week_start DATE NOT NULL UNIQUE,
                    challenge_type VARCHAR(32) NOT NULL,
                    target_value NUMERIC(10,2) NOT NULL,
                    reward_put INT NOT NULL DEFAULT 0,
                    reward_aic INT NOT NULL DEFAULT 0,
                    title VARCHAR(120) NOT NULL,
                    description TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                );
                CREATE TABLE IF NOT EXISTS trill_challenge_completions (
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    challenge_id INT NOT NULL REFERENCES trill_weekly_challenges(id) ON DELETE CASCADE,
                    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    proof_upload_id UUID,
                    PRIMARY KEY (user_id, challenge_id)
                );
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_lb_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;
            """),
            (1069, """
                DELETE FROM vehicle_models WHERE make_id IN (
                    SELECT id FROM vehicle_makes WHERE COALESCE(nhtsa_make_id, 0) < 1
                );
                DELETE FROM user_preferences
                WHERE default_vehicle_make_id IN (
                    SELECT id FROM vehicle_makes WHERE COALESCE(nhtsa_make_id, 0) < 1
                );
                UPDATE uploads SET vehicle_make_id = NULL, vehicle_model_id = NULL
                WHERE vehicle_make_id IN (
                    SELECT id FROM vehicle_makes WHERE COALESCE(nhtsa_make_id, 0) < 1
                );
                DELETE FROM vehicle_makes WHERE COALESCE(nhtsa_make_id, 0) < 1;
            """),
            (1072, """
                CREATE TABLE IF NOT EXISTS trill_notifications (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    kind VARCHAR(48) NOT NULL,
                    rival_user_id UUID,
                    sort_key VARCHAR(32),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    meta JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_trill_notifications_user_kind_time
                    ON trill_notifications(user_id, kind, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_trill_hof_season_rank
                    ON trill_hall_of_fame(season_id, rank, category);
            """),
            (1071, """
                ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS last_oauth_reconnect_at TIMESTAMPTZ;
                ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ;
                CREATE INDEX IF NOT EXISTS idx_platform_tokens_user_last_used
                    ON platform_tokens (user_id, last_used_at DESC)
                    WHERE revoked_at IS NULL;
            """),
            (1073, """
                CREATE TABLE IF NOT EXISTS user_visual_entity_catalog (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    bucket          VARCHAR(32) NOT NULL,
                    entity_name     VARCHAR(200) NOT NULL,
                    normalized_name VARCHAR(200) NOT NULL,
                    seen_count      INT NOT NULL DEFAULT 1,
                    last_category   VARCHAR(64) NOT NULL DEFAULT 'general',
                    last_upload_id  UUID REFERENCES uploads(id) ON DELETE SET NULL,
                    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, bucket, normalized_name)
                );
                CREATE INDEX IF NOT EXISTS idx_user_visual_entity_user_bucket
                    ON user_visual_entity_catalog(user_id, bucket, last_seen_at DESC);
                CREATE INDEX IF NOT EXISTS idx_user_visual_entity_normalized
                    ON user_visual_entity_catalog(user_id, normalized_name);
            """),
            (1067, """
                CREATE TABLE IF NOT EXISTS catalog_pricing_requests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    lookup_key TEXT NOT NULL,
                    status VARCHAR(24) NOT NULL DEFAULT 'open',
                    proposed_patch JSONB NOT NULL DEFAULT '{}'::jsonb,
                    actor_email TEXT,
                    resolution_notes TEXT,
                    resolved_by_email TEXT,
                    resolved_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT catalog_pricing_requests_status_chk CHECK (
                        status IN ('open', 'approved', 'rejected')
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_catalog_pricing_requests_status_created
                    ON catalog_pricing_requests(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_catalog_pricing_requests_lookup
                    ON catalog_pricing_requests(lookup_key, created_at DESC);
            """),
            (1074, """
                ALTER TABLE billing_catalog
                    ADD COLUMN IF NOT EXISTS put_cost_overrides JSONB NOT NULL DEFAULT '{}'::jsonb;
            """),
            (1075, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS group_ids UUID[] DEFAULT '{}';
            """),
            (1076, """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    owner_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL DEFAULT 'My Workspace',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_workspaces_owner ON workspaces(owner_user_id);

                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL DEFAULT 'editor',
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    invited_at TIMESTAMPTZ,
                    joined_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (workspace_id, user_id),
                    CONSTRAINT workspace_members_role_chk CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
                    CONSTRAINT workspace_members_status_chk CHECK (status IN ('active', 'invited', 'removed'))
                );
                CREATE INDEX IF NOT EXISTS idx_workspace_members_user ON workspace_members(user_id, status);

                CREATE TABLE IF NOT EXISTS workspace_invites (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    email VARCHAR(255) NOT NULL,
                    token_hash VARCHAR(128) NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'editor',
                    expires_at TIMESTAMPTZ NOT NULL,
                    accepted_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT workspace_invites_role_chk CHECK (role IN ('editor', 'viewer', 'admin'))
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_invites_token ON workspace_invites(token_hash);
                CREATE INDEX IF NOT EXISTS idx_workspace_invites_email ON workspace_invites(workspace_id, LOWER(email));

                ALTER TABLE users ADD COLUMN IF NOT EXISTS active_workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL;
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS created_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL;
                CREATE INDEX IF NOT EXISTS idx_uploads_workspace ON uploads(workspace_id, created_at DESC);
            """),
            (1077, """
                INSERT INTO workspaces (id, owner_user_id, name, created_at)
                SELECT gen_random_uuid(), u.id, COALESCE(NULLIF(TRIM(u.name), ''), u.email, 'My Workspace'), COALESCE(u.created_at, NOW())
                FROM users u
                WHERE NOT EXISTS (
                    SELECT 1 FROM workspace_members wm
                    WHERE wm.user_id = u.id AND wm.role = 'owner' AND wm.status = 'active'
                );

                INSERT INTO workspace_members (workspace_id, user_id, role, status, joined_at)
                SELECT w.id, w.owner_user_id, 'owner', 'active', COALESCE(w.created_at, NOW())
                FROM workspaces w
                WHERE NOT EXISTS (
                    SELECT 1 FROM workspace_members wm
                    WHERE wm.workspace_id = w.id AND wm.user_id = w.owner_user_id
                );

                UPDATE users u SET active_workspace_id = w.id
                FROM workspaces w
                WHERE w.owner_user_id = u.id AND u.active_workspace_id IS NULL;

                UPDATE uploads up SET workspace_id = w.id, created_by_user_id = up.user_id
                FROM workspaces w
                WHERE w.owner_user_id = up.user_id AND up.workspace_id IS NULL;
            """),
            (1078, """
                ALTER TABLE billing_catalog
                    ADD COLUMN IF NOT EXISTS tier_service_overrides JSONB NOT NULL DEFAULT '{}'::jsonb;
            """),
            (1079, """
                INSERT INTO catalog_products (
                    lookup_key, stripe_product_id, product_kind, sort_order,
                    display_name, stripe_product_name, statement_descriptor, unit_label,
                    price_usd, wallet, put_monthly, aic_monthly, image_filename
                ) VALUES
                    ('uploadm8_boost_small', NULL, 'topup_bundle', 160,
                     'Boost Small', 'UploadM8 Boost Small - PUT + AIC Bundle', 'UPLOADM8 BOOST SM', 'token',
                     7.99, 'bundle', 200, 100, 'topup_bundle_small.png'),
                    ('uploadm8_boost_medium', NULL, 'topup_bundle', 161,
                     'Boost Medium', 'UploadM8 Boost Medium - PUT + AIC Bundle', 'UPLOADM8 BOOST MD', 'token',
                     29.99, 'bundle', 1000, 500, 'topup_bundle_medium.png'),
                    ('uploadm8_boost_large', NULL, 'topup_bundle', 162,
                     'Boost Large', 'UploadM8 Boost Large - PUT + AIC Bundle', 'UPLOADM8 BOOST LG', 'token',
                     99.99, 'bundle', 4000, 2000, 'topup_bundle_large.png')
                ON CONFLICT (lookup_key) DO NOTHING;
            """),
            (1080, """
                UPDATE catalog_products
                SET hud = FALSE, updated_at = NOW()
                WHERE hud IS TRUE;
            """),
            (1081, """
                CREATE INDEX IF NOT EXISTS idx_uploads_trill_scored_user_created
                    ON uploads (user_id, created_at DESC)
                    WHERE trill_score IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_uploads_user_vehicle_make_trill
                    ON uploads (user_id, vehicle_make_id, created_at DESC)
                    WHERE vehicle_make_id IS NOT NULL AND trill_score IS NOT NULL;
            """),
            (1082, """
                -- Normalize legacy flat trill_metadata into nested telemetry.* (see services.trill_access.backfill_trill_metadata_evidence)
                UPDATE uploads u
                SET
                    trill_metadata = COALESCE(u.trill_metadata, '{}'::jsonb) || jsonb_build_object(
                        'telemetry',
                        COALESCE(u.trill_metadata->'telemetry', '{}'::jsonb) || jsonb_strip_nulls(jsonb_build_object(
                            'mid_lat', COALESCE(
                                NULLIF(btrim(u.trill_metadata#>>'{telemetry,mid_lat}'), ''),
                                NULLIF(u.trill_metadata->>'place_lat', ''),
                                NULLIF(u.trill_metadata->>'start_lat', '')
                            ),
                            'mid_lon', COALESCE(
                                NULLIF(btrim(u.trill_metadata#>>'{telemetry,mid_lon}'), ''),
                                NULLIF(u.trill_metadata->>'place_lon', ''),
                                NULLIF(u.trill_metadata->>'start_lon', '')
                            ),
                            'start_lat', COALESCE(
                                NULLIF(btrim(u.trill_metadata#>>'{telemetry,start_lat}'), ''),
                                NULLIF(u.trill_metadata->>'start_lat', ''),
                                NULLIF(u.trill_metadata->>'place_lat', '')
                            ),
                            'start_lon', COALESCE(
                                NULLIF(btrim(u.trill_metadata#>>'{telemetry,start_lon}'), ''),
                                NULLIF(u.trill_metadata->>'start_lon', ''),
                                NULLIF(u.trill_metadata->>'place_lon', '')
                            ),
                            'max_speed_mmph', COALESCE(
                                NULLIF(btrim(u.trill_metadata#>>'{telemetry,max_speed_mph}'), ''),
                                NULLIF(u.trill_metadata->>'max_speed_mph', '')
                            ),
                            'total_distance_miles', COALESCE(
                                NULLIF(btrim(u.trill_metadata#>>'{telemetry,total_distance_miles}'), ''),
                                NULLIF(u.trill_metadata->>'distance_miles', '')
                            )
                        ))
                    ),
                    updated_at = NOW()
                WHERE u.trill_score IS NOT NULL
                  AND u.status = ANY(ARRAY['completed','succeeded','partial']::varchar[])
                  AND u.trill_metadata IS NOT NULL
                  AND (
                      u.trill_metadata ? 'place_lat'
                      OR u.trill_metadata ? 'place_lon'
                      OR u.trill_metadata ? 'start_lat'
                      OR u.trill_metadata ? 'start_lon'
                      OR u.trill_metadata ? 'max_speed_mph'
                      OR u.trill_metadata ? 'distance_miles'
                      OR u.trill_metadata ? 'trill_score'
                      OR (u.trill_metadata ? 'trill' AND NOT (u.trill_metadata ? 'telemetry'))
                  );
            """),
            (1083, SUBSCRIPTION_TIER_CHECK_SQL),
            (1084, """
                -- Stripe webhook idempotency hardening.
                -- 1. Collapse duplicate revenue rows that slipped in before the
                --    unique index existed (keeps the earliest physical row).
                DELETE FROM revenue_tracking a
                USING revenue_tracking b
                WHERE a.stripe_event_id IS NOT NULL
                  AND a.stripe_event_id = b.stripe_event_id
                  AND a.ctid > b.ctid;

                -- 2. One revenue row per Stripe event — arbiter for ON CONFLICT.
                CREATE UNIQUE INDEX IF NOT EXISTS ux_revenue_tracking_stripe_event
                    ON revenue_tracking (stripe_event_id)
                    WHERE stripe_event_id IS NOT NULL;

                -- 3. Monthly-refill dedup ledger (used by _do_monthly_refill).
                CREATE TABLE IF NOT EXISTS stripe_invoice_log (
                    invoice_id   TEXT PRIMARY KEY,
                    user_id      UUID,
                    tier_slug    TEXT,
                    put_credited INT  DEFAULT 0,
                    aic_credited INT  DEFAULT 0,
                    period_start TIMESTAMPTZ,
                    period_end   TIMESTAMPTZ,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );

                -- 4. Event-level idempotency for the billing webhook.
                CREATE TABLE IF NOT EXISTS processed_stripe_events (
                    event_id     TEXT PRIMARY KEY,
                    event_type   TEXT,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                );

                -- 5. Fast dedup lookups for top-up credits keyed on session id.
                CREATE INDEX IF NOT EXISTS idx_token_ledger_stripe_event
                    ON token_ledger (stripe_event_id)
                    WHERE stripe_event_id IS NOT NULL;
            """),
            (1085, """
                CREATE TABLE IF NOT EXISTS pikzels_thumbnail_analyses (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    upload_id UUID NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
                    main_score DOUBLE PRECISION,
                    subscores_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    suggestion TEXT NOT NULL DEFAULT '',
                    recommendation_status VARCHAR(24) NOT NULL DEFAULT 'open'
                        CHECK (recommendation_status IN ('open', 'saved', 'applied', 'dismissed', 'done')),
                    frame_source VARCHAR(32) NOT NULL DEFAULT 'primary_thumbnail',
                    title VARCHAR(200),
                    response_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    fix_image_url TEXT,
                    fix_r2_key VARCHAR(512),
                    fix_score DOUBLE PRECISION,
                    fix_subscores_json JSONB,
                    fix_response_json JSONB,
                    generated_titles_json JSONB,
                    persona_id VARCHAR(128),
                    parent_analysis_id UUID REFERENCES pikzels_thumbnail_analyses(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_pikzels_analyses_user_upload
                    ON pikzels_thumbnail_analyses(user_id, upload_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_pikzels_analyses_user_status
                    ON pikzels_thumbnail_analyses(user_id, recommendation_status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_pikzels_analyses_user_created
                    ON pikzels_thumbnail_analyses(user_id, created_at DESC);
            """),
            # Curated ML feature views: one tidy schema per learning loop ("bucket"),
            # governed by services/ml_feature_registry.py. Build scripts SELECT from
            # these so feature engineering (joins, age-normalization, leakage fixes)
            # lives in one documented place.
            (1087, """
                CREATE INDEX IF NOT EXISTS idx_mtd_user_segment_chan_sent
                    ON marketing_touchpoint_deliveries (user_id, channel, (COALESCE(meta->>'segment_key', '')))
                    WHERE status = 'sent';
            """),
            (1088, """
                -- marketing_events may predate v1033 (CREATE IF NOT EXISTS skipped payload column).
                ALTER TABLE marketing_events ADD COLUMN IF NOT EXISTS payload JSONB DEFAULT '{}'::jsonb;
                UPDATE marketing_events SET payload = '{}'::jsonb WHERE payload IS NULL;
            """),
            (1089, """
                ALTER TABLE uploads ADD COLUMN IF NOT EXISTS pipeline_manifest JSONB DEFAULT NULL;
                CREATE TABLE IF NOT EXISTS upload_funnel_events (
                    id BIGSERIAL PRIMARY KEY,
                    upload_id UUID NOT NULL,
                    event TEXT NOT NULL,
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    details JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_upload_funnel_events_upload_ts
                    ON upload_funnel_events (upload_id, ts);
            """),
            (1086, """
                CREATE OR REPLACE VIEW v_promo_targeting_features AS
                WITH uu30 AS (
                    SELECT u.user_id,
                        COUNT(*)::int AS uploads_30d,
                        AVG(COALESCE(u.views, 0))::double precision AS avg_views_30d,
                        AVG(CASE WHEN COALESCE(u.views, 0) > 0
                            THEN ((COALESCE(u.likes,0)+COALESCE(u.comments,0)+COALESCE(u.shares,0))::double precision / u.views::double precision) * 100.0
                            ELSE 0.0 END)::double precision AS avg_engagement_pct_30d
                    FROM uploads u
                    WHERE u.created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY u.user_id
                ),
                uu_prev30 AS (
                    SELECT u.user_id,
                        COUNT(*)::int AS uploads_prev30d,
                        AVG(COALESCE(u.views, 0))::double precision AS avg_views_prev30d
                    FROM uploads u
                    WHERE u.created_at >= NOW() - INTERVAL '60 days'
                      AND u.created_at <  NOW() - INTERVAL '30 days'
                    GROUP BY u.user_id
                ),
                pp30 AS (
                    SELECT pci.user_id,
                        COUNT(*)::int AS content_items_30d,
                        AVG(COALESCE(pci.views, 0))::double precision AS pci_avg_views_30d
                    FROM platform_content_items pci
                    WHERE pci.published_at >= NOW() - INTERVAL '30 days'
                    GROUP BY pci.user_id
                ),
                uact AS (
                    SELECT user_id, MAX(created_at) AS last_upload_at FROM uploads GROUP BY user_id
                ),
                mhist AS (
                    SELECT user_id, COUNT(*)::int AS prior_touchpoints,
                        MAX(COALESCE(sent_at, created_at)) AS last_touchpoint_at
                    FROM marketing_touchpoint_deliveries GROUP BY user_id
                ),
                mev AS (
                    SELECT user_id,
                        SUM(CASE WHEN event_type = 'campaign_email_open' THEN 1 ELSE 0 END)::int AS opens_all,
                        SUM(CASE WHEN event_type = 'clicked' THEN 1 ELSE 0 END)::int AS clicks_all
                    FROM marketing_events GROUP BY user_id
                ),
                tp AS (
                    SELECT mtd.id AS touchpoint_id, mtd.user_id, mtd.channel,
                        mtd.status AS delivery_status, mtd.created_at AS touchpoint_at,
                        COALESCE(mtd.sent_at, mtd.created_at) AS effective_sent_at
                    FROM marketing_touchpoint_deliveries mtd
                ),
                tp_rows AS (
                    SELECT
                        tp.touchpoint_id::text AS touchpoint_id,
                        tp.user_id,
                        'touchpoint'::text AS row_source,
                        0::int AS is_snapshot,
                        tp.channel::varchar AS channel,
                        tp.delivery_status::varchar AS delivery_status,
                        tp.touchpoint_at,
                        tp.effective_sent_at AS last_activity_at,
                        EXTRACT(DOW FROM tp.effective_sent_at)::int AS sent_dow_utc,
                        EXTRACT(HOUR FROM tp.effective_sent_at)::int AS sent_hour_utc,
                        COALESCE(w.put_balance, 0)::int AS put_balance,
                        COALESCE(w.aic_balance, 0)::int AS aic_balance,
                        COALESCE(u.subscription_tier, 'free')::varchar AS subscription_tier,
                        COALESCE(uu30.uploads_30d, 0)::int AS uploads_30d,
                        COALESCE(uu30.avg_views_30d, 0)::double precision AS avg_views_30d,
                        COALESCE(uu30.avg_engagement_pct_30d, 0)::double precision AS avg_engagement_pct_30d,
                        COALESCE(pp30.content_items_30d, 0)::int AS content_items_30d,
                        COALESCE(pp30.pci_avg_views_30d, 0)::double precision AS pci_avg_views_30d,
                        COALESCE(mhist.prior_touchpoints, 0)::int AS prior_touchpoints,
                        COALESCE(mev.opens_all, 0)::int AS opens_all,
                        COALESCE(mev.clicks_all, 0)::int AS clicks_all,
                        (EXTRACT(EPOCH FROM (NOW() - mhist.last_touchpoint_at)) / 86400.0)::double precision AS days_since_last_touchpoint,
                        (EXTRACT(EPOCH FROM (NOW() - u.created_at)) / 86400.0)::double precision AS account_age_days,
                        (EXTRACT(EPOCH FROM (NOW() - uact.last_upload_at)) / 86400.0)::double precision AS days_since_last_upload,
                        (COALESCE(uu30.uploads_30d, 0) - COALESCE(uu_prev30.uploads_prev30d, 0))::double precision AS uploads_trend_30d,
                        (COALESCE(uu30.avg_views_30d, 0) - COALESCE(uu_prev30.avg_views_prev30d, 0))::double precision AS views_trend_30d,
                        CASE WHEN (
                            EXISTS (SELECT 1 FROM revenue_tracking rt WHERE rt.user_id = tp.user_id AND rt.created_at >= tp.effective_sent_at AND rt.created_at < tp.effective_sent_at + INTERVAL '7 days' AND COALESCE(rt.amount,0) > 0)
                            OR EXISTS (SELECT 1 FROM marketing_events me WHERE me.user_id = tp.user_id AND me.created_at >= tp.effective_sent_at AND me.created_at < tp.effective_sent_at + INTERVAL '7 days' AND me.event_type = 'converted')
                            OR EXISTS (SELECT 1 FROM marketing_events me WHERE me.user_id = tp.user_id AND me.created_at >= tp.effective_sent_at AND me.created_at < tp.effective_sent_at + INTERVAL '7 days' AND me.event_type = 'clicked')
                        ) THEN 1 ELSE 0 END AS converted_7d,
                        CASE WHEN (
                            EXISTS (SELECT 1 FROM revenue_tracking rt WHERE rt.user_id = tp.user_id AND rt.created_at >= tp.effective_sent_at AND rt.created_at < tp.effective_sent_at + INTERVAL '7 days' AND COALESCE(rt.amount,0) > 0)
                            OR EXISTS (SELECT 1 FROM marketing_events me WHERE me.user_id = tp.user_id AND me.created_at >= tp.effective_sent_at AND me.created_at < tp.effective_sent_at + INTERVAL '7 days' AND me.event_type IN ('converted','clicked','campaign_email_open'))
                        ) THEN 1 ELSE 0 END AS engaged_7d,
                        COALESCE((SELECT SUM(COALESCE(rt2.amount,0))::double precision FROM revenue_tracking rt2 WHERE rt2.user_id = tp.user_id AND rt2.created_at >= tp.effective_sent_at AND rt2.created_at < tp.effective_sent_at + INTERVAL '7 days'), 0.0) AS revenue_7d
                    FROM tp
                    LEFT JOIN users u ON u.id = tp.user_id
                    LEFT JOIN wallets w ON w.user_id = tp.user_id
                    LEFT JOIN uu30 ON uu30.user_id = tp.user_id
                    LEFT JOIN uu_prev30 ON uu_prev30.user_id = tp.user_id
                    LEFT JOIN pp30 ON pp30.user_id = tp.user_id
                    LEFT JOIN uact ON uact.user_id = tp.user_id
                    LEFT JOIN mhist ON mhist.user_id = tp.user_id
                    LEFT JOIN mev ON mev.user_id = tp.user_id
                ),
                active_users AS (
                    SELECT DISTINCT u.id AS user_id
                    FROM users u
                    JOIN uploads up ON up.user_id = u.id
                    WHERE up.status IN ('completed','succeeded')
                      AND COALESCE(u.role, '') NOT IN ('master_admin')
                ),
                snap_rows AS (
                    SELECT
                        ('user:' || au.user_id::text)::text AS touchpoint_id,
                        au.user_id,
                        'snapshot'::text AS row_source,
                        1::int AS is_snapshot,
                        'snapshot'::varchar AS channel,
                        'active_user'::varchar AS delivery_status,
                        NOW() AS touchpoint_at,
                        uact.last_upload_at AS last_activity_at,
                        NULL::int AS sent_dow_utc,
                        NULL::int AS sent_hour_utc,
                        COALESCE(w.put_balance, 0)::int AS put_balance,
                        COALESCE(w.aic_balance, 0)::int AS aic_balance,
                        COALESCE(u.subscription_tier, 'free')::varchar AS subscription_tier,
                        COALESCE(uu30.uploads_30d, 0)::int AS uploads_30d,
                        COALESCE(uu30.avg_views_30d, 0)::double precision AS avg_views_30d,
                        COALESCE(uu30.avg_engagement_pct_30d, 0)::double precision AS avg_engagement_pct_30d,
                        COALESCE(pp30.content_items_30d, 0)::int AS content_items_30d,
                        COALESCE(pp30.pci_avg_views_30d, 0)::double precision AS pci_avg_views_30d,
                        COALESCE(mhist.prior_touchpoints, 0)::int AS prior_touchpoints,
                        COALESCE(mev.opens_all, 0)::int AS opens_all,
                        COALESCE(mev.clicks_all, 0)::int AS clicks_all,
                        (EXTRACT(EPOCH FROM (NOW() - mhist.last_touchpoint_at)) / 86400.0)::double precision AS days_since_last_touchpoint,
                        (EXTRACT(EPOCH FROM (NOW() - u.created_at)) / 86400.0)::double precision AS account_age_days,
                        (EXTRACT(EPOCH FROM (NOW() - uact.last_upload_at)) / 86400.0)::double precision AS days_since_last_upload,
                        (COALESCE(uu30.uploads_30d, 0) - COALESCE(uu_prev30.uploads_prev30d, 0))::double precision AS uploads_trend_30d,
                        (COALESCE(uu30.avg_views_30d, 0) - COALESCE(uu_prev30.avg_views_prev30d, 0))::double precision AS views_trend_30d,
                        CASE WHEN (
                            COALESCE((SELECT SUM(COALESCE(rt.amount,0)) FROM revenue_tracking rt WHERE rt.user_id = au.user_id AND rt.created_at >= NOW() - INTERVAL '30 days'), 0) > 0
                            OR EXISTS (SELECT 1 FROM marketing_events me WHERE me.user_id = au.user_id AND me.created_at >= NOW() - INTERVAL '30 days' AND me.event_type = 'converted')
                        ) THEN 1 ELSE 0 END AS converted_7d,
                        CASE WHEN (
                            COALESCE((SELECT SUM(COALESCE(rt.amount,0)) FROM revenue_tracking rt WHERE rt.user_id = au.user_id AND rt.created_at >= NOW() - INTERVAL '30 days'), 0) > 0
                            OR EXISTS (SELECT 1 FROM marketing_events me WHERE me.user_id = au.user_id AND me.created_at >= NOW() - INTERVAL '30 days' AND me.event_type IN ('converted','clicked','campaign_email_open'))
                        ) THEN 1 ELSE 0 END AS engaged_7d,
                        COALESCE((SELECT SUM(COALESCE(rt.amount,0))::double precision FROM revenue_tracking rt WHERE rt.user_id = au.user_id AND rt.created_at >= NOW() - INTERVAL '30 days'), 0.0) AS revenue_7d
                    FROM active_users au
                    JOIN users u ON u.id = au.user_id
                    LEFT JOIN wallets w ON w.user_id = au.user_id
                    LEFT JOIN uu30 ON uu30.user_id = au.user_id
                    LEFT JOIN uu_prev30 ON uu_prev30.user_id = au.user_id
                    LEFT JOIN pp30 ON pp30.user_id = au.user_id
                    LEFT JOIN uact ON uact.user_id = au.user_id
                    LEFT JOIN mhist ON mhist.user_id = au.user_id
                    LEFT JOIN mev ON mev.user_id = au.user_id
                )
                SELECT * FROM tp_rows
                UNION ALL
                SELECT * FROM snap_rows;

                CREATE OR REPLACE VIEW v_content_success_base AS
                SELECT
                    pci.upload_id,
                    pci.user_id,
                    LOWER(pci.platform)::varchar AS platform,
                    pci.published_at,
                    pci.duration_seconds::double precision AS duration_seconds,
                    GREATEST(0.0, EXTRACT(EPOCH FROM (COALESCE(pci.metrics_synced_at, NOW()) - pci.published_at)) / 86400.0)::double precision AS age_days,
                    COALESCE(urs.object_track_count, 0)::int AS object_track_count,
                    COALESCE(urs.person_segment_count, 0)::int AS person_segment_count,
                    COALESCE(urs.text_detection_count, 0)::int AS text_detection_count,
                    COALESCE(urs.logo_count, 0)::int AS logo_count,
                    COALESCE(urs.has_people, FALSE) AS has_people,
                    COALESCE(urs.coverage_seconds, 0)::double precision AS coverage_seconds,
                    COALESCE(urs.hydration_score, 0)::double precision AS hydration_score,
                    u.title,
                    u.caption,
                    COALESCE(array_length(u.hashtags, 1), 0)::int AS hashtag_count
                FROM platform_content_items pci
                LEFT JOIN upload_recognition_summary urs ON urs.upload_id = pci.upload_id
                LEFT JOIN uploads u ON u.id = pci.upload_id
                WHERE pci.upload_id IS NOT NULL;
            """),
        ]

        for version, sql in sorted(migrations, key=lambda item: item[0]):
            if version not in applied:
                try:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                    logger.info(f"Migration v{version} applied")
                except Exception as e:
                    logger.error(f"Migration v{version} failed: {e}")
                    raise

        await ensure_subscription_tier_constraint(conn)
