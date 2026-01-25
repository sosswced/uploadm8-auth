"""
UploadM8 Auth Service (Production Hardened)
===========================================
FastAPI backend for UploadM8 SaaS
- Auth (email/password) + refresh tokens
- Password reset via Mailgun
- R2 presigned URLs for direct browser uploads
- Multi-platform OAuth (TikTok, YouTube, Meta)
- Upload tracking & analytics
- Schema migrations (no more drift 500s)
- Observability + security headers + request IDs + JSON errors
- Locked /api/admin/db-info (schema + applied migrations)
- Baseline rate limiting (in-memory)
- Render bootstrap admin (one-time env var)
"""

import os
import json
import secrets
import hashlib
import base64
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import asyncpg
import jwt
import bcrypt
import boto3
from botocore.config import Config


# ============================================================
# Logging
# ============================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("uploadm8-auth")


# ============================================================
# Configuration from Environment
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")
BASE_URL = os.environ.get("BASE_URL", "https://auth.uploadm8.com")

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.uploadm8.com")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "uploadm8-app")

ACCESS_TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.environ.get("REFRESH_TOKEN_DAYS", "30"))

TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://app.uploadm8.com,https://uploadm8.com")

# Admin key (ops gate for admin endpoints)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

# One-time bootstrap (Render)
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()
BOOTSTRAP_ADMIN_GRANT_LIFETIME = os.environ.get("BOOTSTRAP_ADMIN_GRANT_LIFETIME", "true").lower() in ("1", "true", "yes")

# R2
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")

# Platform OAuth
META_APP_ID = os.environ.get("META_APP_ID", "")
META_APP_SECRET = os.environ.get("META_APP_SECRET", "")
META_API_VERSION = os.environ.get("META_API_VERSION", "v23.0")
TIKTOK_CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

# Mailgun
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "no-reply@auth.uploadm8.com")

# Stripe placeholders (optional)
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


# ============================================================
# Helpers
# ============================================================

def _split_origins(raw: str) -> List[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================
# Env Validation (fail fast)
# ============================================================

def parse_enc_keys() -> Dict[str, bytes]:
    """
    TOKEN_ENC_KEYS format:
      v1:BASE64_32_BYTES_KEY,v2:BASE64_32_BYTES_KEY
    Newest should be last. Old keys remain to decrypt.
    """
    if not TOKEN_ENC_KEYS:
        raise RuntimeError("TOKEN_ENC_KEYS is required")

    keys: Dict[str, bytes] = {}
    clean = TOKEN_ENC_KEYS.strip().strip('"').replace("\\n", "")
    parts = [p.strip() for p in clean.split(",") if p.strip()]

    for part in parts:
        if ":" not in part:
            continue
        kid, b64key = part.split(":", 1)
        raw = base64.b64decode(b64key.strip())
        if len(raw) != 32:
            raise RuntimeError(f"TOKEN_ENC_KEYS invalid: {kid} must decode to 32 bytes")
        keys[kid.strip()] = raw

    if not keys:
        raise RuntimeError("TOKEN_ENC_KEYS parsed empty/invalid; fix env var.")

    def _ver(k: str) -> int:
        try:
            return int(k.lstrip("v"))
        except Exception:
            return 0

    ordered = sorted(keys.keys(), key=_ver)
    return {k: keys[k] for k in ordered}

ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"

def init_enc_keys():
    global ENC_KEYS, CURRENT_KEY_ID
    ENC_KEYS = parse_enc_keys()
    CURRENT_KEY_ID = list(ENC_KEYS.keys())[-1]  # newest = encrypt key

def validate_env():
    missing = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not JWT_SECRET or JWT_SECRET == "change-me":
        missing.append("JWT_SECRET")
    if not TOKEN_ENC_KEYS:
        missing.append("TOKEN_ENC_KEYS")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    # Validate keys parse
    _ = parse_enc_keys()


# ============================================================
# Pydantic Models
# ============================================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=2)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8)

class UploadInit(BaseModel):
    filename: str
    file_size: int
    content_type: str
    platforms: List[str]
    title: str = ""
    caption: str = ""
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None

class SettingsUpdate(BaseModel):
    discord_webhook: Optional[str] = None
    telemetry_enabled: Optional[bool] = None
    hud_enabled: Optional[bool] = None
    hud_position: Optional[str] = None
    speeding_mph: Optional[int] = None
    euphoria_mph: Optional[int] = None

    # Optional legacy meta fields (for future master page use)
    fb_user_id: Optional[str] = None
    selected_page_id: Optional[str] = None
    selected_page_name: Optional[str] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class GrantEntitlement(BaseModel):
    email: EmailStr
    subscription_tier: str = Field(default="lifetime")  # lifetime / family / staff / pro / free
    upload_quota: Optional[int] = None  # None means "unlimited" from business logic perspective (see below)
    role: Optional[str] = None  # admin / user


# ============================================================
# Baseline Rate Limiting (in-memory)
# ============================================================

_rate_state: Dict[str, List[float]] = {}

RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
# You can override per-bucket in env later; this is baseline.
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "60"))

def rate_limit_key(request: Request, bucket: str) -> str:
    ip = request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else "unknown")
    return f"{bucket}:{ip}"

def check_rate_limit(key: str):
    now = time.time()
    arr = _rate_state.get(key, [])
    arr = [t for t in arr if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(arr) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Too many requests")
    arr.append(now)
    _rate_state[key] = arr


# ============================================================
# Password Hashing (bcrypt)
# ============================================================

def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    if len(pw) > 72:
        raise HTTPException(400, "Password too long (max 72 bytes)")
    if password.strip() != password:
        raise HTTPException(400, "Password cannot start/end with spaces")
    hashed = bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ============================================================
# Encryption Helpers
# ============================================================

def encrypt_blob(data: dict) -> dict:
    key = ENC_KEYS[CURRENT_KEY_ID]
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(data).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return {
        "kid": CURRENT_KEY_ID,
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "data": base64.b64encode(ciphertext).decode("utf-8"),
    }

def decrypt_blob(blob: dict) -> dict:
    kid = blob.get("kid", CURRENT_KEY_ID)
    if kid not in ENC_KEYS:
        raise ValueError(f"Unknown key ID: {kid}")
    key = ENC_KEYS[kid]
    aesgcm = AESGCM(key)
    nonce = base64.b64decode(blob["nonce"])
    ciphertext = base64.b64decode(blob["data"])
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return json.loads(plaintext)


# ============================================================
# Database + Migrations
# ============================================================

db_pool: Optional[asyncpg.Pool] = None

MIGRATIONS: List[Tuple[int, str]] = [
    (1, """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        );
    """),
    (2, """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            subscription_tier TEXT DEFAULT 'free',
            stripe_customer_id TEXT,
            upload_quota INTEGER DEFAULT 5,
            uploads_this_month INTEGER DEFAULT 0,
            quota_reset_date DATE DEFAULT CURRENT_DATE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_tier TEXT DEFAULT 'free';
        ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS upload_quota INTEGER DEFAULT 5;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS uploads_this_month INTEGER DEFAULT 0;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS quota_reset_date DATE DEFAULT CURRENT_DATE;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
        ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

        UPDATE users SET name = COALESCE(name, 'User') WHERE name IS NULL;
        ALTER TABLE users ALTER COLUMN name SET NOT NULL;
    """),
    (3, """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            discord_webhook TEXT,
            telemetry_enabled BOOLEAN DEFAULT true,
            hud_enabled BOOLEAN DEFAULT true,
            hud_position TEXT DEFAULT 'bottom_right',
            speeding_mph INTEGER DEFAULT 85,
            euphoria_mph INTEGER DEFAULT 100,

            -- legacy/meta fields (optional)
            fb_user_id TEXT,
            selected_page_id TEXT,
            selected_page_name TEXT,

            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """),
    (4, """
        CREATE TABLE IF NOT EXISTS password_resets (
            token_hash TEXT PRIMARY KEY,
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS platform_tokens (
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            platform TEXT NOT NULL,
            token_blob JSONB NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (user_id, platform)
        );

        CREATE TABLE IF NOT EXISTS uploads (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            r2_key TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_size BIGINT,
            platforms TEXT[] NOT NULL,
            title TEXT,
            caption TEXT,
            privacy TEXT DEFAULT 'public',
            status TEXT DEFAULT 'pending',
            scheduled_time TIMESTAMPTZ,
            trill_score REAL,
            platform_results JSONB DEFAULT '{}',
            error_message TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        );

        CREATE TABLE IF NOT EXISTS analytics_events (
            id BIGSERIAL PRIMARY KEY,
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            event_data JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user_id);
        CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
        CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics_events(user_id);
    """),
    (5, """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            revoked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
        CREATE INDEX IF NOT EXISTS idx_refresh_hash ON refresh_tokens(token_hash);
    """),
    # v6 exists in your prod already (per /db-info). Keep it idempotent.
    (6, """
        ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user';
        UPDATE users SET role='user' WHERE role IS NULL;
        CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
    """),
    # v7: normalize user_settings to stop drift and remove NOT NULL fb_user_id landmine.
    (7, """
        -- Ensure user_settings exists
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            discord_webhook TEXT,
            telemetry_enabled BOOLEAN DEFAULT true,
            hud_enabled BOOLEAN DEFAULT true,
            hud_position TEXT DEFAULT 'bottom_right',
            speeding_mph INTEGER DEFAULT 85,
            euphoria_mph INTEGER DEFAULT 100,
            fb_user_id TEXT,
            selected_page_id TEXT,
            selected_page_name TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Add missing columns (safe on legacy tables)
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS user_id UUID;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS discord_webhook TEXT;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS telemetry_enabled BOOLEAN DEFAULT true;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_enabled BOOLEAN DEFAULT true;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_position TEXT DEFAULT 'bottom_right';
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS speeding_mph INTEGER DEFAULT 85;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS euphoria_mph INTEGER DEFAULT 100;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS fb_user_id TEXT;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS selected_page_id TEXT;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS selected_page_name TEXT;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

        -- Drop NOT NULL on fb_user_id if legacy schema enforced it
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

        -- Ensure updated_at has a default
        DO $$
        DECLARE
            defexpr text;
        BEGIN
            SELECT column_default INTO defexpr
            FROM information_schema.columns
            WHERE table_name='user_settings' AND column_name='updated_at';
            IF defexpr IS NULL THEN
                ALTER TABLE user_settings ALTER COLUMN updated_at SET DEFAULT NOW();
            END IF;
        END $$;

        -- Ensure user_id is uniquely constrainable (needed for ON CONFLICT(user_id))
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

        -- Ensure FK exists (idempotent)
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
]

async def apply_migrations(conn: asyncpg.Connection):
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    applied = await conn.fetch("SELECT version FROM schema_migrations")
    applied_set = {r["version"] for r in applied}

    for version, sql in MIGRATIONS:
        if version in applied_set:
            continue
        logger.info(f"[MIGRATION] Applying v{version}")
        await conn.execute(sql)
        await conn.execute("INSERT INTO schema_migrations(version) VALUES($1)", version)
        logger.info(f"[MIGRATION] Applied v{version}")

async def bootstrap_admin_if_configured(conn: asyncpg.Connection):
    """
    Render-friendly one-time bootstrap.
    Set BOOTSTRAP_ADMIN_EMAIL once, deploy, confirm role, then REMOVE env var.
    """
    if not BOOTSTRAP_ADMIN_EMAIL:
        return

    # Promote role
    row = await conn.fetchrow(
        """
        UPDATE users
        SET role='admin', updated_at=NOW()
        WHERE lower(email)=lower($1)
        RETURNING id, email, role, subscription_tier, upload_quota
        """,
        BOOTSTRAP_ADMIN_EMAIL,
    )
    if not row:
        logger.warning(f"[BOOTSTRAP] BOOTSTRAP_ADMIN_EMAIL set but user not found: {BOOTSTRAP_ADMIN_EMAIL}")
        return

    # Optional: grant lifetime + unlimited (business semantics)
    if BOOTSTRAP_ADMIN_GRANT_LIFETIME:
        await conn.execute(
            """
            UPDATE users
            SET subscription_tier='lifetime',
                upload_quota=NULL,
                updated_at=NOW()
            WHERE lower(email)=lower($1)
            """,
            BOOTSTRAP_ADMIN_EMAIL,
        )

    logger.warning(f"[BOOTSTRAP] Promoted admin: email={row['email']} role=admin (REMOVE BOOTSTRAP_ADMIN_EMAIL AFTER THIS DEPLOY)")

async def init_db():
    global db_pool
    validate_env()
    init_enc_keys()

    db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
        command_timeout=30,
        max_inactive_connection_lifetime=300,
    )

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await apply_migrations(conn)
            await bootstrap_admin_if_configured(conn)

    logger.info("Database initialized and migrations applied")

async def close_db():
    global db_pool
    if db_pool:
        await db_pool.close()


# ============================================================
# JWT Helpers
# ============================================================

def create_access_jwt(user_id: str) -> str:
    now = _now_utc()
    payload = {
        "sub": user_id,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()),
        "jti": secrets.token_hex(16),
        "typ": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_access_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
        )
        if payload.get("typ") != "access":
            return None
        return payload.get("sub")
    except jwt.InvalidTokenError:
        return None

def hash_token(token: str) -> str:
    return _sha256_hex(token)

async def create_refresh_token(conn: asyncpg.Connection, user_id: str) -> str:
    token = secrets.token_urlsafe(48)
    th = hash_token(token)
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_DAYS)
    await conn.execute(
        """INSERT INTO refresh_tokens(user_id, token_hash, expires_at)
           VALUES($1, $2, $3)""",
        user_id, th, expires_at
    )
    return token

async def revoke_refresh_token(conn: asyncpg.Connection, refresh_token: str):
    th = hash_token(refresh_token)
    await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE token_hash = $1", th)

async def rotate_refresh_token(conn: asyncpg.Connection, refresh_token: str) -> Tuple[str, str]:
    th = hash_token(refresh_token)
    row = await conn.fetchrow(
        """SELECT user_id, expires_at, revoked_at
           FROM refresh_tokens WHERE token_hash = $1""",
        th
    )
    if not row:
        raise HTTPException(401, "Invalid refresh token")
    if row["revoked_at"] is not None:
        raise HTTPException(401, "Refresh token revoked")
    if row["expires_at"] < _now_utc():
        raise HTTPException(401, "Refresh token expired")

    await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE token_hash = $1", th)
    user_id = str(row["user_id"])
    new_refresh = await create_refresh_token(conn, user_id)
    new_access = create_access_jwt(user_id)
    return new_access, new_refresh

async def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(401, "Missing authorization header")

    token = authorization.replace("Bearer ", "")
    user_id = verify_access_jwt(token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            """SELECT id, email, name, subscription_tier, upload_quota,
                      uploads_this_month, quota_reset_date, role
               FROM users WHERE id = $1""",
            user_id,
        )
    if not user:
        raise HTTPException(401, "User not found")
    return dict(user)


# ============================================================
# R2 Storage
# ============================================================

def get_r2_client():
    if not R2_ACCOUNT_ID or not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
        return None

    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="auto",
    )

def generate_presigned_upload_url(key: str, content_type: str, expires: int = 3600) -> str:
    client = get_r2_client()
    if not client:
        raise HTTPException(500, "R2 storage not configured")

    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
    )


# ============================================================
# Email (Mailgun)
# ============================================================

async def send_email(to: str, subject: str, html: str):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.info(f"[MAIL] Not configured - would send to {to}: {subject}")
        return False

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={"from": MAIL_FROM, "to": to, "subject": subject, "html": html},
        )
        return resp.status_code == 200


# ============================================================
# Analytics Tracking
# ============================================================

async def track_event(user_id: str, event_type: str, event_data: dict = None):
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO analytics_events (user_id, event_type, event_data)
                   VALUES ($1, $2, $3)""",
                user_id,
                event_type,
                json.dumps(event_data or {}),
            )
    except Exception as e:
        logger.warning(f"[ANALYTICS] error: {e}")


# ============================================================
# Admin Auth Dependency (ops key)
# ============================================================

def require_admin_key(x_admin_key: str = Header(None)):
    if not ADMIN_API_KEY:
        raise HTTPException(500, "ADMIN_API_KEY not configured")
    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(403, "Forbidden")
    return True


# ============================================================
# FastAPI App + Middleware
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(title="UploadM8 API", version="1.2.0", lifespan=lifespan)

origins = _split_origins(ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

@app.middleware("http")
async def request_id_security_and_logging(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid

    start = time.time()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except HTTPException as e:
        status_code = e.status_code
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail, "request_id": rid},
            headers={"X-Request-ID": rid},
        )
    except Exception as e:
        logger.exception(f"[RID:{rid}] Unhandled exception: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "request_id": rid},
            headers={"X-Request-ID": rid},
        )
    finally:
        duration_ms = int((time.time() - start) * 1000)
        ip = request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else "unknown")
        logger.info(f"rid={rid} ip={ip} {request.method} {request.url.path} status={status_code} dur_ms={duration_ms}")

    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ============================================================
# Health
# ============================================================

@app.get("/")
async def root():
    return {"message": "UploadM8 Auth Service", "status": "running"}

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "1.2.0",
        "database": db_pool is not None,
        "allowed_origins": origins,
        "r2_configured": bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY),
        "mailgun_configured": bool(MAILGUN_API_KEY and MAILGUN_DOMAIN),
        "platforms": {
            "tiktok": bool(TIKTOK_CLIENT_KEY),
            "youtube": bool(GOOGLE_CLIENT_ID),
            "meta": bool(META_APP_ID),
        },
        "bootstrap_admin_email_set": bool(BOOTSTRAP_ADMIN_EMAIL),
    }


# ============================================================
# Admin: DB Info (locked by X-Admin-Key)
# ============================================================

@app.get("/api/admin/db-info")
async def admin_db_info(_: bool = Depends(require_admin_key)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        migrations = await conn.fetch("SELECT version, applied_at FROM schema_migrations ORDER BY version ASC")

        users_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name='users'
            ORDER BY ordinal_position
        """)
        settings_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name='user_settings'
            ORDER BY ordinal_position
        """)
        refresh_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name='refresh_tokens'
            ORDER BY ordinal_position
        """)

    return {
        "database_configured": True,
        "latest_schema_version": max([m["version"] for m in migrations]) if migrations else 0,
        "migrations": [{"version": m["version"], "applied_at": m["applied_at"].isoformat()} for m in migrations],
        "tables": {
            "users": [dict(c) for c in users_cols],
            "user_settings": [dict(c) for c in settings_cols],
            "refresh_tokens": [dict(c) for c in refresh_cols],
        },
    }

@app.get("/api/admin/user-role")
async def admin_user_role(email: str, _: bool = Depends(require_admin_key)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT email, role, subscription_tier, upload_quota FROM users WHERE lower(email)=lower($1)",
            email,
        )
    if not row:
        raise HTTPException(404, "User not found")
    return {
        "email": row["email"],
        "role": row["role"],
        "subscription_tier": row["subscription_tier"],
        "upload_quota": row["upload_quota"],
    }

@app.post("/api/admin/grant")
async def admin_grant_entitlement(payload: GrantEntitlement, _: bool = Depends(require_admin_key)):
    """
    Grants lifetime/family/staff tiers and/or admin role to a user by email.
    upload_quota:
      - If None: treated as "unlimited" by business logic (see uploads check).
      - If int: enforced monthly quota.
    """
    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE users
            SET subscription_tier = $2,
                upload_quota = $3,
                role = COALESCE($4, role),
                updated_at = NOW()
            WHERE lower(email)=lower($1)
            RETURNING id, email, role, subscription_tier, upload_quota
            """,
            payload.email.lower(),
            payload.subscription_tier,
            payload.upload_quota,
            payload.role,
        )
    if not row:
        raise HTTPException(404, "User not found")
    return {
        "status": "ok",
        "user": {
            "id": str(row["id"]),
            "email": row["email"],
            "role": row["role"],
            "subscription_tier": row["subscription_tier"],
            "upload_quota": row["upload_quota"],
        }
    }


# ============================================================
# Authentication
# ============================================================

@app.post("/api/auth/register")
async def register(request: Request, data: UserCreate, background_tasks: BackgroundTasks):
    check_rate_limit(rate_limit_key(request, "register"))

    password_hash = hash_password(data.password)

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            try:
                row = await conn.fetchrow(
                    """INSERT INTO users (email, password_hash, name)
                       VALUES ($1, $2, $3)
                       RETURNING id, email, name, subscription_tier, upload_quota, role""",
                    data.email.lower(),
                    password_hash,
                    data.name,
                )
            except asyncpg.UniqueViolationError:
                raise HTTPException(400, "Email already registered")

            user_id = str(row["id"])

            # After v7 migration, fb_user_id is nullable, user_id is unique, FK exists.
            # This will no longer silently fail due to legacy NOT NULL constraints.
            try:
                await conn.execute(
                    """
                    INSERT INTO user_settings (user_id, updated_at)
                    VALUES ($1, NOW())
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    user_id,
                )
            except Exception as e:
                # This should be extremely rare after v7; if it happens, surface it clearly.
                logger.warning(f"[SETTINGS] bootstrap deferred user_id={user_id} err={e}")

            access_token = create_access_jwt(user_id)
            refresh_token = await create_refresh_token(conn, user_id)

    background_tasks.add_task(track_event, user_id, "user_signup", {"email": data.email})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": row["email"],
            "name": row["name"],
            "subscription_tier": row["subscription_tier"],
            "upload_quota": row["upload_quota"],
            "role": row["role"],
        },
    }

@app.post("/api/auth/login")
async def login(request: Request, data: UserLogin, background_tasks: BackgroundTasks):
    check_rate_limit(rate_limit_key(request, "login"))

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, email, name, password_hash, subscription_tier,
                      upload_quota, uploads_this_month, role
               FROM users WHERE email = $1""",
            data.email.lower(),
        )

        if not row or not verify_password(data.password, row["password_hash"]):
            raise HTTPException(401, "Invalid email or password")

        user_id = str(row["id"])
        access_token = create_access_jwt(user_id)
        refresh_token = await create_refresh_token(conn, user_id)

    background_tasks.add_task(track_event, user_id, "user_login")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": row["email"],
            "name": row["name"],
            "subscription_tier": row["subscription_tier"],
            "upload_quota": row["upload_quota"],
            "uploads_this_month": row["uploads_this_month"],
            "role": row["role"],
        },
    }

@app.post("/api/auth/refresh")
async def refresh_token(data: RefreshRequest):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    async with db_pool.acquire() as conn:
        new_access, new_refresh = await rotate_refresh_token(conn, data.refresh_token)
    return {"access_token": new_access, "refresh_token": new_refresh, "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(data: RefreshRequest):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    async with db_pool.acquire() as conn:
        await revoke_refresh_token(conn, data.refresh_token)
    return {"status": "ok"}

@app.post("/api/auth/forgot-password")
async def forgot_password(request: Request, data: PasswordReset, background_tasks: BackgroundTasks):
    check_rate_limit(rate_limit_key(request, "forgot_password"))

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, name FROM users WHERE email = $1",
            data.email.lower(),
        )

        # Always return generic response (avoid user enumeration)
        if not user:
            return {"message": "If that email exists, we've sent a reset link"}

        token = secrets.token_urlsafe(32)
        token_hash = _sha256_hex(token)
        expires_at = _now_utc() + timedelta(hours=1)

        await conn.execute(
            """INSERT INTO password_resets (token_hash, user_id, expires_at)
               VALUES ($1, $2, $3)
               ON CONFLICT (token_hash) DO UPDATE SET expires_at = $3""",
            token_hash, str(user["id"]), expires_at,
        )

    reset_url = f"{BASE_URL}/reset-password?token={token}"
    html = f"""
    <h2>Reset Your UploadM8 Password</h2>
    <p>Hi {user['name']},</p>
    <p>Click the link below to reset your password:</p>
    <p><a href="{reset_url}" style="background:#3B82F6;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;display:inline-block;">Reset Password</a></p>
    <p>This link expires in 1 hour.</p>
    <p>If you didn't request this, you can ignore this email.</p>
    <p>- The UploadM8 Team</p>
    """
    background_tasks.add_task(send_email, user["email"], "Reset Your UploadM8 Password", html)
    return {"message": "If that email exists, we've sent a reset link"}

@app.post("/api/auth/reset-password")
async def reset_password(data: PasswordResetConfirm):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    token_hash = _sha256_hex(data.token)

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            reset = await conn.fetchrow(
                "SELECT user_id, expires_at FROM password_resets WHERE token_hash = $1",
                token_hash,
            )
            if not reset:
                raise HTTPException(400, "Invalid or expired reset token")
            if reset["expires_at"] < _now_utc():
                raise HTTPException(400, "Reset token has expired")

            new_hash = hash_password(data.new_password)
            await conn.execute(
                "UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2",
                new_hash, reset["user_id"],
            )
            await conn.execute("DELETE FROM password_resets WHERE token_hash = $1", token_hash)

    return {"message": "Password updated successfully"}

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "subscription_tier": user["subscription_tier"],
        "upload_quota": user["upload_quota"],
        "uploads_this_month": user["uploads_this_month"],
        "role": user.get("role"),
    }


# ============================================================
# Settings
# ============================================================

@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        settings = await conn.fetchrow(
            "SELECT * FROM user_settings WHERE user_id = $1",
            str(user["id"]),
        )

    if not settings:
        return {
            "telemetry_enabled": True,
            "hud_enabled": True,
            "hud_position": "bottom_right",
            "speeding_mph": 85,
            "euphoria_mph": 100,
            "discord_webhook_configured": False,
            "fb_user_id": None,
            "selected_page_id": None,
            "selected_page_name": None,
        }

    return {
        "telemetry_enabled": settings.get("telemetry_enabled", True),
        "hud_enabled": settings.get("hud_enabled", True),
        "hud_position": settings.get("hud_position", "bottom_right"),
        "speeding_mph": settings.get("speeding_mph", 85),
        "euphoria_mph": settings.get("euphoria_mph", 100),
        "discord_webhook_configured": bool(settings.get("discord_webhook")),
        "fb_user_id": settings.get("fb_user_id"),
        "selected_page_id": settings.get("selected_page_id"),
        "selected_page_name": settings.get("selected_page_name"),
    }

@app.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_settings (
                user_id, discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                speeding_mph, euphoria_mph, fb_user_id, selected_page_id, selected_page_name, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                discord_webhook = COALESCE($2, user_settings.discord_webhook),
                telemetry_enabled = COALESCE($3, user_settings.telemetry_enabled),
                hud_enabled = COALESCE($4, user_settings.hud_enabled),
                hud_position = COALESCE($5, user_settings.hud_position),
                speeding_mph = COALESCE($6, user_settings.speeding_mph),
                euphoria_mph = COALESCE($7, user_settings.euphoria_mph),
                fb_user_id = COALESCE($8, user_settings.fb_user_id),
                selected_page_id = COALESCE($9, user_settings.selected_page_id),
                selected_page_name = COALESCE($10, user_settings.selected_page_name),
                updated_at = NOW()
            """,
            str(user["id"]),
            data.discord_webhook,
            data.telemetry_enabled,
            data.hud_enabled,
            data.hud_position,
            data.speeding_mph,
            data.euphoria_mph,
            data.fb_user_id,
            data.selected_page_id,
            data.selected_page_name,
        )

    return {"status": "updated"}


# ============================================================
# Uploads (R2 Direct Upload)
# ============================================================

@app.post("/api/uploads/presign")
async def create_presigned_upload(
    data: UploadInit,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    # Business logic for "unlimited":
    # if upload_quota is NULL => unlimited.
    user_id = str(user["id"])
    quota = user.get("upload_quota")
    used = user.get("uploads_this_month", 0)
    if quota is not None and used >= quota:
        raise HTTPException(403, "Monthly upload quota exceeded. Please upgrade your plan.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = secrets.token_hex(4)
    r2_key = f"uploads/{user_id}/{timestamp}_{file_hash}_{data.filename}"

    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)

    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            """INSERT INTO uploads (user_id, r2_key, filename, file_size, platforms,
                   title, caption, privacy, status, scheduled_time)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending', $9)
               RETURNING id""",
            user_id,
            r2_key,
            data.filename,
            data.file_size,
            data.platforms,
            data.title,
            data.caption,
            data.privacy,
            data.scheduled_time,
        )

    upload_id = str(upload["id"])
    background_tasks.add_task(
        track_event,
        user_id,
        "upload_initiated",
        {"upload_id": upload_id, "platforms": data.platforms},
    )

    return {"upload_id": upload_id, "presigned_url": presigned_url, "r2_key": r2_key, "expires_in": 3600}

@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            upload = await conn.fetchrow(
                "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
                upload_id, user_id
            )
            if not upload:
                raise HTTPException(404, "Upload not found")

            await conn.execute("UPDATE uploads SET status = 'processing' WHERE id = $1", upload_id)
            await conn.execute(
                "UPDATE users SET uploads_this_month = uploads_this_month + 1, updated_at = NOW() WHERE id = $1",
                user_id
            )

    background_tasks.add_task(track_event, user_id, "upload_complete", {"upload_id": upload_id})
    return {"status": "processing", "upload_id": upload_id}

@app.get("/api/uploads")
async def list_uploads(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch(
                """SELECT id, filename, platforms, title, status, trill_score,
                          created_at, completed_at
                   FROM uploads WHERE user_id = $1 AND status = $2
                   ORDER BY created_at DESC LIMIT $3 OFFSET $4""",
                user_id, status, limit, offset
            )
        else:
            uploads = await conn.fetch(
                """SELECT id, filename, platforms, title, status, trill_score,
                          created_at, completed_at
                   FROM uploads WHERE user_id = $1
                   ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
                user_id, limit, offset
            )

    return {"uploads": [dict(u) for u in uploads], "limit": limit, "offset": offset}

@app.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: str, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, str(user["id"])
        )
    if not upload:
        raise HTTPException(404, "Upload not found")
    return dict(upload)


# ============================================================
# Platforms: Connected Tokens
# ============================================================

@app.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        tokens = await conn.fetch(
            "SELECT platform, updated_at FROM platform_tokens WHERE user_id = $1",
            user_id
        )

    connected = {row["platform"]: {"connected": True, "updated_at": row["updated_at"].isoformat()} for row in tokens}
    return {
        "tiktok": connected.get("tiktok", {"connected": False}),
        "youtube": connected.get("google", {"connected": False}),
        "facebook": connected.get("meta", {"connected": False}),
        "instagram": connected.get("meta", {"connected": False}),
    }


# ============================================================
# OAuth - TikTok
# ============================================================

@app.get("/oauth/tiktok/start")
async def tiktok_oauth_start(user: dict = Depends(get_current_user)):
    if not TIKTOK_CLIENT_KEY or not TIKTOK_CLIENT_SECRET:
        raise HTTPException(500, "TikTok OAuth not configured")

    state = create_access_jwt(str(user["id"]))
    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"

    code_verifier = secrets.token_urlsafe(43)
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).rstrip(b"=").decode()

    url = (
        f"https://www.tiktok.com/v2/auth/authorize?"
        f"client_key={TIKTOK_CLIENT_KEY}&"
        f"redirect_uri={redirect_uri}&"
        f"scope=user.info.basic,video.upload,video.publish&"
        f"state={state}|{code_verifier}&"
        f"response_type=code&"
        f"code_challenge={code_challenge}&"
        f"code_challenge_method=S256"
    )
    return RedirectResponse(url)

@app.get("/oauth/tiktok/callback")
async def tiktok_oauth_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
    background_tasks: BackgroundTasks = None,
):
    if error:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=tiktok_oauth_failed")
    if not state or "|" not in state:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=invalid_state")

    parts = state.split("|")
    user_id = verify_access_jwt(parts[0])
    code_verifier = parts[1] if len(parts) > 1 else ""
    if not user_id:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=invalid_state")

    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": TIKTOK_CLIENT_KEY,
                "client_secret": TIKTOK_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
        token_data = resp.json()

    if "error" in token_data:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=tiktok_token_failed")

    token_blob = encrypt_blob(token_data)

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'tiktok', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET
                   token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )

    if background_tasks:
        background_tasks.add_task(track_event, user_id, "platform_connected", {"platform": "tiktok"})
    return RedirectResponse(f"{BASE_URL}/dashboard?success=tiktok_connected")


# ============================================================
# OAuth - Google/YouTube
# ============================================================

@app.get("/oauth/google/start")
async def google_oauth_start(user: dict = Depends(get_current_user)):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(500, "Google OAuth not configured")

    state = create_access_jwt(str(user["id"]))
    redirect_uri = f"{BASE_URL}/oauth/google/callback"

    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"scope=https://www.googleapis.com/auth/youtube.upload&"
        f"state={state}&"
        f"response_type=code&"
        f"access_type=offline&"
        f"prompt=consent"
    )
    return RedirectResponse(url)

@app.get("/oauth/google/callback")
async def google_oauth_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
    background_tasks: BackgroundTasks = None,
):
    if error:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=google_oauth_failed")

    user_id = verify_access_jwt(state)
    if not user_id:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=invalid_state")

    redirect_uri = f"{BASE_URL}/oauth/google/callback"

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )
        token_data = resp.json()

    if "error" in token_data:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=google_token_failed")

    token_blob = encrypt_blob(token_data)

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'google', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET
                   token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )

    if background_tasks:
        background_tasks.add_task(track_event, user_id, "platform_connected", {"platform": "youtube"})
    return RedirectResponse(f"{BASE_URL}/dashboard?success=youtube_connected")


# ============================================================
# OAuth - Meta (Facebook/Instagram)
# ============================================================

@app.get("/oauth/meta/start")
async def meta_oauth_start(user: dict = Depends(get_current_user)):
    if not META_APP_ID or not META_APP_SECRET:
        raise HTTPException(500, "Meta OAuth not configured")

    state = create_access_jwt(str(user["id"]))
    redirect_uri = f"{BASE_URL}/oauth/meta/callback"

    url = (
        f"https://www.facebook.com/{META_API_VERSION}/dialog/oauth?"
        f"client_id={META_APP_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"scope=pages_show_list,pages_manage_posts,instagram_basic,instagram_content_publish&"
        f"state={state}&"
        f"response_type=code"
    )
    return RedirectResponse(url)

@app.get("/oauth/meta/callback")
async def meta_oauth_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
    background_tasks: BackgroundTasks = None,
):
    if error:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=meta_oauth_failed")

    user_id = verify_access_jwt(state)
    if not user_id:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=invalid_state")

    redirect_uri = f"{BASE_URL}/oauth/meta/callback"

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"https://graph.facebook.com/{META_API_VERSION}/oauth/access_token",
            params={
                "client_id": META_APP_ID,
                "client_secret": META_APP_SECRET,
                "redirect_uri": redirect_uri,
                "code": code,
            },
        )
        token_data = resp.json()

    if "error" in token_data:
        return RedirectResponse(f"{BASE_URL}/dashboard?error=meta_token_failed")

    token_blob = encrypt_blob(token_data)

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'meta', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET
                   token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )

    if background_tasks:
        background_tasks.add_task(track_event, user_id, "platform_connected", {"platform": "meta"})
    return RedirectResponse(f"{BASE_URL}/dashboard?success=meta_connected")


# ============================================================
# Analytics
# ============================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview(days: int = 30, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)

    async with db_pool.acquire() as conn:
        total_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE user_id = $1", user_id)
        period_uploads = await conn.fetchval(
            "SELECT COUNT(*) FROM uploads WHERE user_id = $1 AND created_at >= $2",
            user_id, since
        )
        successful = await conn.fetchval(
            """SELECT COUNT(*) FROM uploads
               WHERE user_id = $1 AND status = 'complete' AND created_at >= $2""",
            user_id, since
        )
        avg_trill = await conn.fetchval(
            """SELECT AVG(trill_score) FROM uploads
               WHERE user_id = $1 AND trill_score IS NOT NULL AND created_at >= $2""",
            user_id, since
        )

    success_rate = (successful / period_uploads * 100) if period_uploads else 0
    return {
        "total_uploads": total_uploads or 0,
        "period_uploads": period_uploads or 0,
        "success_rate": round(success_rate, 1),
        "avg_trill_score": round(avg_trill or 0, 1),
        "quota_used": user.get("uploads_this_month", 0),
        "quota_total": user.get("upload_quota"),
        "subscription_tier": user.get("subscription_tier"),
    }


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
