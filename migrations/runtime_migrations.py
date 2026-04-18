"""Versioned SQL migrations applied once at API startup.

Single source of truth: edit this file only. ``app.py`` lifespan calls
``run_migrations(db_pool)`` — do not duplicate migration lists elsewhere.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("uploadm8-api")

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
        ]

        for version, sql in migrations:
            if version not in applied:
                try:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                    logger.info(f"Migration v{version} applied")
                except Exception as e:
                    logger.error(f"Migration v{version} failed: {e}")
