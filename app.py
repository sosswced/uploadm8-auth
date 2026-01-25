(7, """
    -- 1) Ensure user_id exists and can be primary identity
    ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS user_id UUID;

    -- 2) If legacy fb_user_id is NOT NULL, relax it (your app does not require it for all users)
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name='user_settings'
              AND column_name='fb_user_id'
              AND is_nullable='NO'
        ) THEN
            ALTER TABLE user_settings ALTER COLUMN fb_user_id DROP NOT NULL;
        END IF;
    END $$;

    -- 3) Backfill user_id when possible (if you have any mapping logic later, update this)
    -- (No-op if you can't infer; at least don't crash schema.)
    -- NOTE: leaving user_id NULL for legacy rows is acceptable short-term.

    -- 4) Create a unique constraint on user_id (idempotent)
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'user_settings'::regclass
              AND contype = 'u'
              AND conname = 'user_settings_user_id_uniq'
        ) THEN
            ALTER TABLE user_settings ADD CONSTRAINT user_settings_user_id_uniq UNIQUE (user_id);
        END IF;
    END $$;

    -- 5) Add FK to users (idempotent)
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'user_settings'::regclass
              AND contype = 'f'
              AND conname = 'user_settings_user_fk'
        ) THEN
            ALTER TABLE user_settings
            ADD CONSTRAINT user_settings_user_fk
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
        END IF;
    END $$;
"""),
