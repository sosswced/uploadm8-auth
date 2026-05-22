-- =============================================================
-- UploadM8 Catalog Sync — DB single source of truth
-- =============================================================
-- Path: scripts/001_catalog_products.sql
-- Automatic apply: migrations/runtime_migrations.py v1066 (CATALOG_PRODUCTS_BOOTSTRAP_SQL).
-- Keep this file identical to v1066 for manual repair:
--   psql $DATABASE_URL -f scripts/001_catalog_products.sql
--
-- Seed rows match stages/entitlements.py TIER_CONFIG (subscriptions) and
-- TOPUP_PRODUCTS (top-ups). No orphaned 50/100 token SKUs unless added to Python too.
-- =============================================================

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


-- Subscriptions — align with TIER_CONFIG in stages/entitlements.py
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


-- PUT top-ups — keys and prices match TOPUP_PRODUCTS in stages/entitlements.py
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


-- AIC top-ups
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
