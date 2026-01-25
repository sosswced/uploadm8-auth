"""
UploadM8 Auth Service (Production Hardened) — Updated (Bootstrap + KPI + Admin foundations)
========================================================================================
FastAPI backend for UploadM8 SaaS
- Auth (email/password) + refresh tokens (rotation)
- Password reset via Mailgun
- R2 presigned URLs for direct browser uploads
- Multi-platform OAuth (TikTok, YouTube, Meta)
- Upload tracking & analytics + KPI time series
- Schema migrations (no more drift 500s) + legacy user_settings normalization (fb_user_id PK rebuild)
- Observability + security headers + request IDs + JSON errors
- Locked /api/admin/db-info (schema + applied migrations) via ADMIN_API_KEY
- Role-based admin (JWT) for future browser admin console
- Admin endpoints for lifetime/friends-and-family entitlements + KPI rollups (role-based)

Tell it like it is:
Note #1: ADMIN_API_KEY is fine for bootstrap/ops endpoints, not for a browser admin console.
         Your future admin/master page should use role-based JWT (users.role='admin') + CSRF protection on frontend.
Note #2: Your production DB already had a legacy user_settings primary key using fb_user_id.
         v7 migration below detects that and rebuilds user_settings into a clean user_id primary key table.
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

import httpx
import asyncpg
import jwt
import bcrypt
import boto3
from botocore.config import Config
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import stripe
import redis.asyncio as redis

from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from contextlib import asynccontextmanager

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

# ------------------------------------------------------------
# FRONTEND base URL (where dashboard lives)
# ------------------------------------------------------------
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.uploadm8.com")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "uploadm8-app")

ACCESS_TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.environ.get("REFRESH_TOKEN_DAYS", "30"))

TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")  # required for stable encryption

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://app.uploadm8.com,https://uploadm8.com",
)

# Admin (ops key for locked endpoints)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

# One-time bootstrap promotion (set in Render env vars, deploy once, then remove)
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()

# R2 Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")


# Redis (queue + distributed rate limiting)
REDIS_URL = os.environ.get("REDIS_URL", "")  # e.g. redis://:pass@host:6379/0
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")

# Stripe Billing
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_DEFAULT_LOOKUP_KEY = os.environ.get("STRIPE_DEFAULT_LOOKUP_KEY", "")  # e.g. "uploadm8_pro_monthly"
STRIPE_LOOKUP_KEYS = os.environ.get("STRIPE_LOOKUP_KEYS", STRIPE_DEFAULT_LOOKUP_KEY)  # comma-separated
STRIPE_SUCCESS_URL = os.environ.get("STRIPE_SUCCESS_URL", f"{FRONTEND_URL}/billing-success.html?session_id={{CHECKOUT_SESSION_ID}}")
STRIPE_CANCEL_URL = os.environ.get("STRIPE_CANCEL_URL", f"{FRONTEND_URL}/pricing.html")
STRIPE_PORTAL_RETURN_URL = os.environ.get("STRIPE_PORTAL_RETURN_URL", f"{FRONTEND_URL}/dashboard.html")
STRIPE_TRIAL_DAYS_DEFAULT = int(os.environ.get("STRIPE_TRIAL_DAYS_DEFAULT", "0"))
STRIPE_AUTOMATIC_TAX = os.environ.get("STRIPE_AUTOMATIC_TAX", "1")  # 1/0

# Admin notifications
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")


# ============================================================
# Token Vault Helpers (auto-refresh lifecycle)
# ============================================================

TOKEN_REFRESH_SKEW_SEC = int(os.environ.get("TOKEN_REFRESH_SKEW_SEC", "300"))  # refresh 5 min early

async def _get_platform_token_row(user_id: str, platform: str):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    async with db_pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT id, user_id, platform, token_blob, expires_at FROM platform_tokens WHERE user_id=$1 AND platform=$2",
            user_id, platform
        )

async def _save_platform_token(user_id: str, platform: str, token_data: dict, expires_at: Optional[datetime] = None):
    blob = encrypt_blob(token_data)
    exp = expires_at
    if exp is None and isinstance(token_data, dict):
        # Prefer explicit _expires_at (epoch seconds) if present
        ea = token_data.get("_expires_at")
        if ea:
            try:
                exp = datetime.fromtimestamp(float(ea), tz=timezone.utc)
            except Exception:
                exp = None
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob, expires_at, updated_at)
                 VALUES ($1,$2,$3,$4,NOW())
                 ON CONFLICT (user_id, platform) DO UPDATE SET
                   token_blob=EXCLUDED.token_blob,
                   expires_at=EXCLUDED.expires_at,
                   updated_at=NOW()""",
            user_id, platform, blob, exp
        )

async def refresh_google_token(token_data: dict) -> dict:
    rt = token_data.get("refresh_token")
    if not rt:
        raise HTTPException(401, "Google refresh_token missing; reconnect required")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": rt,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if r.status_code >= 400:
            raise HTTPException(401, f"Google refresh failed: {r.text[:200]}")
        newt = r.json()
    # merge
    token_data["access_token"] = newt.get("access_token", token_data.get("access_token"))
    token_data["expires_in"] = newt.get("expires_in", token_data.get("expires_in"))
    token_data["_obtained_at"] = time.time()
    token_data["_expires_at"] = time.time() + float(token_data.get("expires_in", 0) or 0) - 60
    return token_data

async def refresh_tiktok_token(token_data: dict) -> dict:
    rt = token_data.get("refresh_token")
    if not rt:
        raise HTTPException(401, "TikTok refresh_token missing; reconnect required")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": TIKTOK_CLIENT_KEY,
                "client_secret": TIKTOK_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": rt,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if r.status_code >= 400:
            raise HTTPException(401, f"TikTok refresh failed: {r.text[:200]}")
        newt = r.json()
    # TikTok wraps tokens under "data" sometimes. Normalize.
    data = newt.get("data") if isinstance(newt, dict) else None
    if isinstance(data, dict):
        newt = data
    token_data["access_token"] = newt.get("access_token", token_data.get("access_token"))
    token_data["refresh_token"] = newt.get("refresh_token", token_data.get("refresh_token"))
    token_data["expires_in"] = newt.get("expires_in", token_data.get("expires_in"))
    token_data["_obtained_at"] = time.time()
    token_data["_expires_at"] = time.time() + float(token_data.get("expires_in", 0) or 0) - 60
    return token_data

async def refresh_meta_token(token_data: dict) -> dict:
    at = token_data.get("access_token")
    if not at:
        raise HTTPException(401, "Meta access_token missing; reconnect required")
    # Exchange for a long-lived token
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            "https://graph.facebook.com/v19.0/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": META_APP_ID,
                "client_secret": META_APP_SECRET,
                "fb_exchange_token": at,
            },
        )
        if r.status_code >= 400:
            raise HTTPException(401, f"Meta token exchange failed: {r.text[:200]}")
        newt = r.json()
    token_data["access_token"] = newt.get("access_token", token_data.get("access_token"))
    token_data["expires_in"] = newt.get("expires_in", token_data.get("expires_in"))
    token_data["_obtained_at"] = time.time()
    token_data["_expires_at"] = time.time() + float(token_data.get("expires_in", 0) or 0) - 60
    return token_data

async def get_valid_platform_token(user_id: str, platform: str) -> dict:
    row = await _get_platform_token_row(user_id, platform)
    if not row:
        raise HTTPException(404, f"No token for platform: {platform}")
    token_data = decrypt_blob(row["token_blob"])
    expires_at = row["expires_at"]

    # Determine expiry from db or embedded metadata
    exp_epoch = None
    if isinstance(token_data, dict) and token_data.get("_expires_at"):
        try:
            exp_epoch = float(token_data["_expires_at"])
        except Exception:
            exp_epoch = None

    now_epoch = time.time()
    needs_refresh = False
    if exp_epoch is not None:
        needs_refresh = (exp_epoch - now_epoch) <= TOKEN_REFRESH_SKEW_SEC
    elif expires_at:
        needs_refresh = (expires_at - _now_utc()) <= timedelta(seconds=TOKEN_REFRESH_SKEW_SEC)

    if not needs_refresh:
        return token_data

    # Refresh based on platform
    if platform == "google":
        token_data = await refresh_google_token(token_data)
    elif platform == "tiktok":
        token_data = await refresh_tiktok_token(token_data)
    elif platform == "meta":
        token_data = await refresh_meta_token(token_data)
    else:
        return token_data

    await _save_platform_token(user_id, platform, token_data)
    return token_data

@app.get("/api/vault/token")
async def vault_token(platform: str = Query(...), user: dict = Depends(get_current_user)):
    """Returns a valid access_token for the platform. Auto-refreshes when close to expiry."""
    platform = platform.lower().strip()
    td = await get_valid_platform_token(str(user["id"]), platform)
    return {"platform": platform, "access_token": td.get("access_token"), "expires_in": td.get("expires_in"), "scope": td.get("scope")}

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

# Stripe (optional placeholder)
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
    fb_user_id: Optional[str] = None
    selected_page_id: Optional[str] = None
    selected_page_name: Optional[str] = None
    hud_speed_unit: Optional[str] = None  # mph | kmh
    hud_color: Optional[str] = None        # e.g. '#FFFFFF'
    hud_font_family: Optional[str] = None
    hud_font_size: Optional[int] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class AdminGrantEntitlement(BaseModel):
    email: EmailStr
    tier: str = Field(default="lifetime")  # lifetime | friends_family | pro | etc
    upload_quota: Optional[int] = None     # override quota
    note: Optional[str] = None

class AdminSetRole(BaseModel):
    email: EmailStr
    role: str = Field(default="admin")  # admin | user

# ============================================================
# Baseline Rate Limiting (in-memory)
# ============================================================

_rate_state: Dict[str, List[float]] = {}
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
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
# Distributed Rate Limiting (Redis-backed) — safe fallback
# ============================================================

async def check_rate_limit_any(key: str):
    """Redis-backed sliding window when Redis is configured; fallback to in-memory."""
    if redis_client is None:
        return check_rate_limit(key)

    now = int(time.time())
    window = RATE_LIMIT_WINDOW_SEC

    # Sliding window via sorted set
    zkey = f"ratelimit:{key}"
    try:
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(zkey, 0, now - window)
        pipe.zadd(zkey, {str(now): now})
        pipe.zcard(zkey)
        pipe.expire(zkey, window + 5)
        _, _, count, _ = await pipe.execute()
        if int(count) > RATE_LIMIT_MAX:
            raise HTTPException(429, "Too many requests")
    except HTTPException:
        raise
    except Exception as e:
        # Fail-open to preserve uptime; fallback to in-memory
        logger.warning(f"Redis rate limiter error (fallback to memory): {e}")
        return check_rate_limit(key)



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
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS user_id UUID;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS discord_webhook TEXT;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS telemetry_enabled BOOLEAN DEFAULT true;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_enabled BOOLEAN DEFAULT true;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_position TEXT DEFAULT 'bottom_right';
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS speeding_mph INTEGER DEFAULT 85;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS euphoria_mph INTEGER DEFAULT 100;
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

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
        CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type);
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
    (6, """
        ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user';
        UPDATE users SET role = COALESCE(role, 'user') WHERE role IS NULL;
    """),
    (7, """
        DO $$
        DECLARE
            pk_cols text[];
            has_fb_pk boolean := false;
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'user_settings'
            ) THEN
                CREATE TABLE user_settings (
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
                RETURN;
            END IF;

            SELECT array_agg(a.attname ORDER BY a.attnum)
            INTO pk_cols
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN unnest(c.conkey) WITH ORDINALITY AS ck(attnum, ord) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ck.attnum
            WHERE n.nspname='public' AND t.relname='user_settings' AND c.contype='p';

            IF pk_cols IS NOT NULL AND 'fb_user_id' = ANY(pk_cols) THEN
                has_fb_pk := true;
            END IF;

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

            IF has_fb_pk THEN
                CREATE TABLE IF NOT EXISTS user_settings_v2 (
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

                INSERT INTO user_settings_v2 (
                    user_id, discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                    speeding_mph, euphoria_mph, fb_user_id, selected_page_id, selected_page_name, updated_at
                )
                SELECT
                    us.user_id,
                    us.discord_webhook,
                    COALESCE(us.telemetry_enabled, true),
                    COALESCE(us.hud_enabled, true),
                    COALESCE(us.hud_position, 'bottom_right'),
                    COALESCE(us.speeding_mph, 85),
                    COALESCE(us.euphoria_mph, 100),
                    us.fb_user_id,
                    us.selected_page_id,
                    us.selected_page_name,
                    COALESCE(us.updated_at, NOW())
                FROM user_settings us
                WHERE us.user_id IS NOT NULL
                  AND EXISTS (SELECT 1 FROM users u WHERE u.id = us.user_id)
                ON CONFLICT (user_id) DO UPDATE SET
                    discord_webhook = EXCLUDED.discord_webhook,
                    telemetry_enabled = EXCLUDED.telemetry_enabled,
                    hud_enabled = EXCLUDED.hud_enabled,
                    hud_position = EXCLUDED.hud_position,
                    speeding_mph = EXCLUDED.speeding_mph,
                    euphoria_mph = EXCLUDED.euphoria_mph,
                    fb_user_id = EXCLUDED.fb_user_id,
                    selected_page_id = EXCLUDED.selected_page_id,
                    selected_page_name = EXCLUDED.selected_page_name,
                    updated_at = NOW();

                DROP TABLE user_settings;
                ALTER TABLE user_settings_v2 RENAME TO user_settings;
                RETURN;
            END IF;

            DO $inner$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'user_settings'::regclass
                      AND contype = 'u'
                      AND conname = 'user_settings_user_id_uniq'
                ) THEN
                    ALTER TABLE user_settings ADD CONSTRAINT user_settings_user_id_uniq UNIQUE (user_id);
                END IF;
            END
            $inner$;

            DO $inner2$
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
            END
            $inner2$;

        END $$;
    """),
    (8, """
        CREATE TABLE IF NOT EXISTS entitlements (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            tier TEXT NOT NULL,
            upload_quota_override INTEGER,
            is_lifetime BOOLEAN DEFAULT false,
            note TEXT,
            granted_by UUID,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS idx_entitlements_user ON entitlements(user_id);
        CREATE INDEX IF NOT EXISTS idx_entitlements_tier ON entitlements(tier);

        CREATE TABLE IF NOT EXISTS admin_audit (
            id BIGSERIAL PRIMARY KEY,
            actor_user_id UUID,
            actor_email TEXT,
            action TEXT NOT NULL,
            target_user_id UUID,
            target_email TEXT,
            meta JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_admin_audit_actor ON admin_audit(actor_user_id);
        CREATE INDEX IF NOT EXISTS idx_admin_audit_created ON admin_audit(created_at);
    """),
    (9, """
        -- Billing / processing / KPI support (additive)
        ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_status TEXT DEFAULT 'inactive';
        ALTER TABLE users ADD COLUMN IF NOT EXISTS current_period_end TIMESTAMPTZ;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS unlimited_uploads BOOLEAN DEFAULT false;

        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processed_r2_key TEXT;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMPTZ;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_finished_at TIMESTAMPTZ;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS error_code TEXT;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS error_detail TEXT;

        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_speed_unit TEXT DEFAULT 'mph';
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_color TEXT DEFAULT '#FFFFFF';
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_font_family TEXT DEFAULT 'Inter';
        ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS hud_font_size INTEGER DEFAULT 28;

        CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
        CREATE INDEX IF NOT EXISTS idx_uploads_created ON uploads(created_at);
        CREATE INDEX IF NOT EXISTS idx_uploads_user_status ON uploads(user_id, status);
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

async def bootstrap_promote_admin(conn: asyncpg.Connection):
    if not BOOTSTRAP_ADMIN_EMAIL:
        return

    row = await conn.fetchrow(
        "SELECT id, email, role FROM users WHERE lower(email)=lower($1)",
        BOOTSTRAP_ADMIN_EMAIL
    )
    if not row:
        logger.warning(f"[BOOTSTRAP] No user found for {BOOTSTRAP_ADMIN_EMAIL}. Register that email, then redeploy once with BOOTSTRAP_ADMIN_EMAIL.")
        return

    await conn.execute(
        "UPDATE users SET role='admin', updated_at=NOW() WHERE id=$1",
        row["id"]
    )
    logger.info(f"[BOOTSTRAP] Promoted {BOOTSTRAP_ADMIN_EMAIL} to admin")


# ============================================================
# Redis clients (queue + distributed rate limiting)
# ============================================================

redis_client: Optional[redis.Redis] = None

async def init_redis():
    """Initialize Redis if REDIS_URL is configured. Safe no-op otherwise."""
    global redis_client
    if not REDIS_URL:
        redis_client = None
        return
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        # Ping to validate connectivity early
        await redis_client.ping()
        logger.info("Redis initialized")
    except Exception as e:
        redis_client = None
        logger.error(f"Redis init failed (continuing without Redis): {e}")

async def close_redis():
    global redis_client
    try:
        if redis_client is not None:
            await redis_client.close()
    finally:
        redis_client = None

async def enqueue_job(job: dict):
    """Enqueue a job to Redis list. If Redis is not available, raise 503."""
    if redis_client is None:
        raise HTTPException(503, "Queue not available (Redis not configured)")
    payload = json.dumps(job, separators=(",", ":"))
    await redis_client.lpush(UPLOAD_JOB_QUEUE, payload)

async def init_db():
    global db_pool
    validate_env()
    init_enc_keys()
    await init_redis()

    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY

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
            await bootstrap_promote_admin(conn)

    logger.info("Database initialized and migrations applied")

async def close_db():
    global db_pool
    try:
        if db_pool:
            await db_pool.close()
    finally:
        await close_redis()

# ============================================================
# JWT Helpers (iss/aud/jti)
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
            """SELECT id, email, name, subscription_tier, role, upload_quota,
                      uploads_this_month, quota_reset_date
               FROM users WHERE id = $1""",
            user_id,
        )

    if not user:
        raise HTTPException(401, "User not found")

    return dict(user)

def require_admin_api_key(x_admin_key: str = Header(None)):
    if not ADMIN_API_KEY:
        raise HTTPException(500, "ADMIN_API_KEY not configured")
    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(403, "Forbidden")
    return True

async def require_admin_role(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin role required")
    return user

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

def generate_presigned_get_url(key: str, ttl_seconds: int = 3600) -> str:
    client = get_r2_client()
    if not client:
        raise HTTPException(500, "R2 storage not configured")
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key},
        ExpiresIn=ttl_seconds,
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

async def admin_audit_log(
    conn: asyncpg.Connection,
    actor_user: Optional[dict],
    action: str,
    target_user_id: Optional[str] = None,
    target_email: Optional[str] = None,
    meta: Optional[dict] = None,
):
    try:
        await conn.execute(
            """INSERT INTO admin_audit(actor_user_id, actor_email, action, target_user_id, target_email, meta)
               VALUES($1, $2, $3, $4, $5, $6)""",
            actor_user.get("id") if actor_user else None,
            actor_user.get("email") if actor_user else None,
            action,
            target_user_id,
            target_email,
            json.dumps(meta or {}),
        )
    except Exception as e:
        logger.warning(f"[ADMIN_AUDIT] failed: {e}")

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
# Health & Status
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
        "platforms": {"tiktok": bool(TIKTOK_CLIENT_KEY), "youtube": bool(GOOGLE_CLIENT_ID), "meta": bool(META_APP_ID)},
        "bootstrap_admin_email_set": bool(BOOTSTRAP_ADMIN_EMAIL),
        "frontend_url": FRONTEND_URL,
    }

# ============================================================
# Admin: DB Info (locked via ADMIN_API_KEY)
# ============================================================
@app.get("/health")
async def health_alias():
    return {"status": "ok"}



@app.get("/api/admin/db-info")
async def admin_db_info(_: bool = Depends(require_admin_api_key)):
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
        ent_cols = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name='entitlements'
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
        "migrations": [{"version": m["version"], "applied_at": m["applied_at"].isoformat()} for m in migrations],
        "tables": {
            "users": [dict(c) for c in users_cols],
            "user_settings": [dict(c) for c in settings_cols],
            "entitlements": [dict(c) for c in ent_cols],
            "refresh_tokens": [dict(c) for c in refresh_cols],
        },
    }

# ============================================================
# Admin: Role check endpoint (for frontend gating)
# ============================================================

@app.get("/api/admin/user-role")
async def admin_user_role(user: dict = Depends(get_current_user)):
    role = (user.get("role") or "user").lower()
    return {"role": role, "is_admin": role == "admin"}

# ============================================================
# Admin: KPI + entitlements (ROLE-based)
# ============================================================

@app.get("/api/admin/overview")
async def admin_overview(days: int = 30, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    since = _now_utc() - timedelta(days=days)
    since7 = _now_utc() - timedelta(days=7)
    since30 = _now_utc() - timedelta(days=30)

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        admins = await conn.fetchval("SELECT COUNT(*) FROM users WHERE role='admin'")
        lifetime = await conn.fetchval("SELECT COUNT(*) FROM entitlements WHERE tier='lifetime' OR is_lifetime=true")

        uploads_total = await conn.fetchval("SELECT COUNT(*) FROM uploads")
        uploads_period = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1", since)
        uploads_by_status = await conn.fetch(
            """SELECT status, COUNT(*)::int AS c
               FROM uploads WHERE created_at >= $1
               GROUP BY status ORDER BY c DESC""",
            since
        )

        active_7d = await conn.fetchval(
            """SELECT COUNT(DISTINCT user_id) FROM analytics_events
               WHERE created_at >= $1""",
            since7
        )
        active_30d = await conn.fetchval(
            """SELECT COUNT(DISTINCT user_id) FROM analytics_events
               WHERE created_at >= $1""",
            since30
        )

        platform_connected = await conn.fetch(
            """SELECT platform, COUNT(DISTINCT user_id)::int AS c
               FROM platform_tokens GROUP BY platform ORDER BY c DESC"""
        )

    return {
        "window_days": days,
        "users": {
            "total": int(total_users or 0),
            "new_in_window": int(new_users or 0),
            "admins": int(admins or 0),
            "lifetime_entitlements": int(lifetime or 0),
            "active_users_7d": int(active_7d or 0),
            "active_users_30d": int(active_30d or 0),
        },
        "uploads": {
            "total": int(uploads_total or 0),
            "in_window": int(uploads_period or 0),
            "by_status": {r["status"]: r["c"] for r in uploads_by_status},
        },
        "platforms_connected": {r["platform"]: r["c"] for r in platform_connected},
    }

@app.post("/api/admin/users/role")
async def admin_set_user_role(payload: AdminSetRole, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    role = payload.role.strip().lower()
    if role not in ("admin", "user"):
        raise HTTPException(400, "role must be 'admin' or 'user'")

    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT id, email, role FROM users WHERE lower(email)=lower($1)", payload.email)
        if not target:
            raise HTTPException(404, "User not found")

        await conn.execute("UPDATE users SET role=$1, updated_at=NOW() WHERE id=$2", role, target["id"])
        await admin_audit_log(
            conn,
            admin,
            action="set_role",
            target_user_id=str(target["id"]),
            target_email=target["email"],
            meta={"role": role},
        )

    return {"status": "ok", "email": payload.email.lower(), "role": role}

@app.post("/api/admin/entitlements/grant")
async def admin_grant_entitlement(payload: AdminGrantEntitlement, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    tier = payload.tier.strip().lower()
    if tier not in ("lifetime", "friends_family", "pro", "free"):
        raise HTTPException(400, "tier must be one of: lifetime, friends_family, pro, free")

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            target = await conn.fetchrow(
                "SELECT id, email, subscription_tier, upload_quota FROM users WHERE lower(email)=lower($1)",
                payload.email
            )
            if not target:
                raise HTTPException(404, "User not found")

            is_lifetime = tier == "lifetime"
            await conn.execute(
                """INSERT INTO entitlements(user_id, tier, upload_quota_override, is_lifetime, note, granted_by)
                   VALUES($1, $2, $3, $4, $5, $6)""",
                target["id"],
                tier,
                payload.upload_quota,
                is_lifetime,
                payload.note,
                admin["id"],
            )

            if payload.upload_quota is not None:
                await conn.execute(
                    "UPDATE users SET subscription_tier=$1, upload_quota=$2, updated_at=NOW() WHERE id=$3",
                    tier,
                    payload.upload_quota,
                    target["id"],
                )
            else:
                await conn.execute(
                    "UPDATE users SET subscription_tier=$1, updated_at=NOW() WHERE id=$2",
                    tier,
                    target["id"],
                )

            await admin_audit_log(
                conn,
                admin,
                action="grant_entitlement",
                target_user_id=str(target["id"]),
                target_email=target["email"],
                meta={"tier": tier, "upload_quota_override": payload.upload_quota, "note": payload.note},
            )

    return {"status": "ok", "email": payload.email.lower(), "tier": tier, "upload_quota_override": payload.upload_quota}

@app.get("/api/admin/users/search")
async def admin_search_users(q: str = "", limit: int = 25, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    limit = max(1, min(limit, 100))
    q = (q or "").strip().lower()

    async with db_pool.acquire() as conn:
        if q:
            rows = await conn.fetch(
                """SELECT id, email, name, role, subscription_tier, upload_quota, uploads_this_month, created_at
                   FROM users
                   WHERE lower(email) LIKE $1 OR lower(name) LIKE $1
                   ORDER BY created_at DESC
                   LIMIT $2""",
                f"%{q}%",
                limit,
            )
        else:
            rows = await conn.fetch(
                """SELECT id, email, name, role, subscription_tier, upload_quota, uploads_this_month, created_at
                   FROM users
                   ORDER BY created_at DESC
                   LIMIT $1""",
                limit,
            )

    return {"users": [dict(r) for r in rows]}

# ============================================================
# Authentication Routes
# ============================================================

@app.post("/api/auth/register")
async def register(request: Request, data: UserCreate, background_tasks: BackgroundTasks):
    await check_rate_limit_any(rate_limit_key(request, "register"))

    password_hash = hash_password(data.password)

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            try:
                row = await conn.fetchrow(
                    """INSERT INTO users (email, password_hash, name)
                       VALUES ($1, $2, $3)
                       RETURNING id, email, name, subscription_tier, role, upload_quota, uploads_this_month""",
                    data.email.lower(),
                    password_hash,
                    data.name,
                )
            except asyncpg.UniqueViolationError:
                raise HTTPException(400, "Email already registered")

            user_id = str(row["id"])

            await conn.execute(
                """INSERT INTO user_settings (user_id)
                   VALUES ($1)
                   ON CONFLICT (user_id) DO NOTHING""",
                user_id,
            )

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
            "role": row["role"],
            "upload_quota": row["upload_quota"],
            "uploads_this_month": row["uploads_this_month"],
        },
    }

@app.post("/api/auth/login")
async def login(request: Request, data: UserLogin, background_tasks: BackgroundTasks):
    await check_rate_limit_any(rate_limit_key(request, "login"))

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, email, name, password_hash, subscription_tier, role,
                      upload_quota, uploads_this_month, quota_reset_date
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
            "role": row["role"],
            "upload_quota": row["upload_quota"],
            "uploads_this_month": row["uploads_this_month"],
            "quota_reset_date": str(row["quota_reset_date"]) if row["quota_reset_date"] else None,
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
    await check_rate_limit_any(rate_limit_key(request, "forgot_password"))

    if not db_pool:
        raise HTTPException(500, "Database not available")

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, name FROM users WHERE email = $1",
            data.email.lower(),
        )

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

    # ------------------------------------------------------------
    # Redirect users to frontend reset page (not backend domain)
    # ------------------------------------------------------------
    reset_url = f"{FRONTEND_URL}/forgot-password.html?token={token}"

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
        "role": user.get("role", "user"),
        "upload_quota": user["upload_quota"],
        "uploads_this_month": user["uploads_this_month"],
        "quota_reset_date": str(user["quota_reset_date"]) if user.get("quota_reset_date") else None,
    }

# ============================================================
# Settings Routes
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
            "hud_speed_unit": "mph",
            "hud_color": "#FFFFFF",
            "hud_font_family": "Inter",
            "hud_font_size": 28,
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
        "hud_speed_unit": settings.get("hud_speed_unit", "mph"),
        "hud_color": settings.get("hud_color", "#FFFFFF"),
        "hud_font_family": settings.get("hud_font_family", "Inter"),
        "hud_font_size": settings.get("hud_font_size", 28),
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
            """INSERT INTO user_settings (user_id, discord_webhook, telemetry_enabled,
                   hud_enabled, hud_position, hud_speed_unit, hud_color, hud_font_family, hud_font_size,
                   speeding_mph, euphoria_mph, fb_user_id, selected_page_id, selected_page_name)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
               ON CONFLICT (user_id) DO UPDATE SET
                   discord_webhook = COALESCE($2, user_settings.discord_webhook),
                   telemetry_enabled = COALESCE($3, user_settings.telemetry_enabled),
                   hud_enabled = COALESCE($4, user_settings.hud_enabled),
                   hud_position = COALESCE($5, user_settings.hud_position),
                   hud_speed_unit = COALESCE($6, user_settings.hud_speed_unit),
                   hud_color = COALESCE($7, user_settings.hud_color),
                   hud_font_family = COALESCE($8, user_settings.hud_font_family),
                   hud_font_size = COALESCE($9, user_settings.hud_font_size),
                   speeding_mph = COALESCE($10, user_settings.speeding_mph),
                   euphoria_mph = COALESCE($11, user_settings.euphoria_mph),
                   fb_user_id = COALESCE($12, user_settings.fb_user_id),
                   selected_page_id = COALESCE($13, user_settings.selected_page_id),
                   selected_page_name = COALESCE($14, user_settings.selected_page_name),
                   updated_at = NOW()""",
            str(user["id"]),
            data.discord_webhook,
            data.telemetry_enabled,
            data.hud_enabled,
            data.hud_position,
            data.hud_speed_unit,
            data.hud_color,
            data.hud_font_family,
            data.hud_font_size,
            data.speeding_mph,
            data.euphoria_mph,
            data.fb_user_id,
            data.selected_page_id,
            data.selected_page_name,
        )

    return {"status": "updated"}


# ============================================================
# Billing (Stripe) — Checkout + Customer Portal + Webhooks
# ============================================================

class CheckoutRequest(BaseModel):
    lookup_key: Optional[str] = None
    trial_days: Optional[int] = None

class PortalRequest(BaseModel):
    return_url: Optional[str] = None

async def _discord_notify_admin(event: str, meta: dict):
    url = ADMIN_DISCORD_WEBHOOK_URL
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={"content": f"[BILLING] {event}", "embeds": [{"title": event, "fields": [{"name": k, "value": str(v)[:900], "inline": False} for k,v in meta.items()]}]})
    except Exception as e:
        logger.warning(f"Admin Discord notify failed: {e}")

def _tier_from_lookup_key(lookup_key: str) -> str:
    lk = (lookup_key or "").lower()
    if "enterprise" in lk:
        return "enterprise"
    if "pro" in lk:
        return "pro"
    if "starter" in lk:
        return "starter"
    return "free"

def _entitlements_for_tier(tier: str) -> dict:
    # Tell-it-like-it-is defaults. Adjust later via env-backed mapping once pricing is final.
    if tier == "enterprise":
        return {"tier": tier, "upload_quota": 10_000, "unlimited_uploads": True}
    if tier == "pro":
        return {"tier": tier, "upload_quota": 1_000, "unlimited_uploads": True}
    if tier == "starter":
        return {"tier": tier, "upload_quota": 100, "unlimited_uploads": False}
    return {"tier": "free", "upload_quota": 5, "unlimited_uploads": False}

@app.get("/api/billing/prices")
async def list_prices():
    if not STRIPE_SECRET_KEY:
        return {"configured": False, "prices": []}
    keys = [k.strip() for k in (STRIPE_LOOKUP_KEYS or "").split(",") if k.strip()]
    out = []
    try:
        # Stripe lookup_key resolves to Price objects
        for lk in keys:
            prices = stripe.Price.list(lookup_keys=[lk], expand=["data.product"], active=True, limit=5)
            for p in prices.data:
                out.append({
                    "id": p.id,
                    "lookup_key": lk,
                    "currency": p.currency,
                    "unit_amount": p.unit_amount,
                    "recurring": (p.recurring or {}),
                    "product": {"id": p.product.id if hasattr(p.product, "id") else p.product,
                                "name": p.product.name if hasattr(p.product, "name") else None},
                })
        return {"configured": True, "prices": out}
    except Exception as e:
        logger.error(f"Stripe list prices failed: {e}")
        raise HTTPException(500, "Stripe error")

@app.post("/api/billing/checkout")
async def create_checkout_session(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")

    lookup_key = (data.lookup_key or STRIPE_DEFAULT_LOOKUP_KEY or "").strip()
    if not lookup_key:
        raise HTTPException(400, "Missing lookup_key")

    trial_days = data.trial_days if data.trial_days is not None else STRIPE_TRIAL_DAYS_DEFAULT
    if trial_days is not None:
        trial_days = max(0, min(int(trial_days), 60))

    try:
        prices = stripe.Price.list(lookup_keys=[lookup_key], active=True, limit=1)
        if not prices.data:
            raise HTTPException(400, "Invalid lookup_key")
        price_id = prices.data[0].id

        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            cust = stripe.Customer.create(email=user["email"], metadata={"user_id": str(user["id"])})
            customer_id = cust.id
            if db_pool:
                async with db_pool.acquire() as conn:
                    await conn.execute("UPDATE users SET stripe_customer_id=$1, updated_at=NOW() WHERE id=$2", customer_id, str(user["id"]))

        params = {
            "mode": "subscription",
            "customer": customer_id,
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": STRIPE_SUCCESS_URL,
            "cancel_url": STRIPE_CANCEL_URL,
            "allow_promotion_codes": True,
            "client_reference_id": str(user["id"]),
            "subscription_data": {},
            "metadata": {"user_id": str(user["id"]), "lookup_key": lookup_key},
        }
        if STRIPE_AUTOMATIC_TAX == "1":
            params["automatic_tax"] = {"enabled": True}
        if trial_days and trial_days > 0:
            params["subscription_data"]["trial_period_days"] = trial_days

        session = stripe.checkout.Session.create(**params)

        await _discord_notify_admin("checkout_session_created", {"user": user["email"], "lookup_key": lookup_key, "session": session.id})
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO analytics_events(user_id,event_type,event_data) VALUES ($1,'billing_checkout_created',$2)",
                    str(user["id"]),
                    json.dumps({"lookup_key": lookup_key, "session_id": session.id}),
                )

        return {"url": session.url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stripe checkout failed: {e}")
        raise HTTPException(500, "Stripe checkout failed")

@app.post("/api/billing/portal")
async def create_portal_session(data: PortalRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")

    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(400, "No Stripe customer on file")

    return_url = (data.return_url or STRIPE_PORTAL_RETURN_URL or FRONTEND_URL).strip()
    try:
        sess = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
        return {"url": sess.url}
    except Exception as e:
        logger.error(f"Stripe portal failed: {e}")
        raise HTTPException(500, "Stripe portal failed")

async def _apply_subscription_entitlements(user_id: str, subscription: dict, lookup_key: Optional[str]):
    tier = _tier_from_lookup_key(lookup_key or "")
    ent = _entitlements_for_tier(tier)
    status = subscription.get("status") if isinstance(subscription, dict) else None
    sub_id = subscription.get("id") if isinstance(subscription, dict) else None
    current_period_end = subscription.get("current_period_end") if isinstance(subscription, dict) else None

    if db_pool:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE users SET
                       subscription_tier=$1,
                       upload_quota=$2,
                       unlimited_uploads=$3,
                       stripe_subscription_id=COALESCE($4, stripe_subscription_id),
                       subscription_status=COALESCE($5, subscription_status),
                       current_period_end=COALESCE(to_timestamp($6), current_period_end),
                       updated_at=NOW()
                   WHERE id=$7""",
                ent["tier"],
                ent["upload_quota"],
                ent["unlimited_uploads"],
                sub_id,
                status,
                current_period_end,
                user_id,
            )
            await conn.execute(
                "INSERT INTO analytics_events(user_id,event_type,event_data) VALUES ($1,'billing_entitlements_applied',$2)",
                user_id,
                json.dumps({"tier": ent["tier"], "status": status, "subscription_id": sub_id}),
            )

@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        # If webhook isn't configured, treat as 204 to avoid noisy retries in early dev
        return Response(status_code=204)

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.warning(f"Stripe webhook signature verification failed: {e}")
        return Response(status_code=400)

    etype = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}

    # Resolve user_id via metadata (preferred), else via customer lookup.
    user_id = (obj.get("metadata") or {}).get("user_id")
    lookup_key = (obj.get("metadata") or {}).get("lookup_key")

    try:
        if not user_id and obj.get("customer") and db_pool:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT id FROM users WHERE stripe_customer_id=$1", obj.get("customer"))
                if row:
                    user_id = str(row["id"])

        if user_id and db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO analytics_events(user_id,event_type,event_data) VALUES ($1,$2,$3)",
                    user_id,
                    f"stripe_{etype}",
                    json.dumps({"id": event.get("id"), "type": etype}),
                )

        # Source of truth: subscription.* events
        if etype in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
            await _apply_subscription_entitlements(user_id, obj, lookup_key)
            await _discord_notify_admin(etype, {"user_id": user_id, "status": obj.get("status"), "sub": obj.get("id")})

        if etype == "checkout.session.completed":
            await _discord_notify_admin(etype, {"user_id": user_id, "session": obj.get("id"), "customer": obj.get("customer")})

        return {"received": True}
    except Exception as e:
        logger.error(f"Stripe webhook handling failed: {e}")
        return Response(status_code=500)

# ============================================================
# Upload Routes (R2 Direct Upload)
# ============================================================

@app.post("/api/uploads/presign")
async def create_presigned_upload(
    data: UploadInit,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    if user["uploads_this_month"] >= user["upload_quota"]:
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
    background_tasks.add_task(track_event, user_id, "upload_initiated", {"upload_id": upload_id, "platforms": data.platforms})
    return {"upload_id": upload_id, "presigned_url": presigned_url, "r2_key": r2_key, "expires_in": 3600}

@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user_id)
            if not upload:
                raise HTTPException(404, "Upload not found")

            await conn.execute("UPDATE uploads SET status = 'queued' WHERE id = $1", upload_id)
            await conn.execute("UPDATE users SET uploads_this_month = uploads_this_month + 1, updated_at = NOW() WHERE id = $1", user_id)

        background_tasks.add_task(enqueue_job, {"type": "publish", "upload_id": upload_id, "user_id": user_id})
    background_tasks.add_task(track_event, user_id, "upload_complete", {"upload_id": upload_id})
    return {"status": "queued", "upload_id": upload_id}

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
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

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
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, str(user["id"]))
    if not upload:
        raise HTTPException(404, "Upload not found")
    return dict(upload)


@app.get("/api/uploads/{upload_id}/presign-get")
async def presign_get_for_upload(
    upload_id: str,
    ttl: int = 3600,
    user: dict = Depends(get_current_user),
):
    """Publicly readable URL for IG/Facebook ingest (pull_by_url), generated server-side."""
    if not db_pool:
        raise HTTPException(500, "Database not available")

    ttl = max(60, min(int(ttl), 24 * 3600))

    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT id, user_id, r2_key, processed_r2_key FROM uploads WHERE id=$1 AND user_id=$2",
            upload_id,
            str(user["id"]),
        )
    if not upload:
        raise HTTPException(404, "Upload not found")

    key = upload.get("processed_r2_key") or upload.get("r2_key")
    url = generate_presigned_get_url(key, ttl_seconds=ttl)
    return {"url": url, "expires_in": ttl, "r2_key": key}

# ============================================================
# Platform Connection Routes
# ============================================================

@app.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        tokens = await conn.fetch("SELECT platform, updated_at FROM platform_tokens WHERE user_id = $1", user_id)

    connected = {row["platform"]: {"connected": True, "updated_at": row["updated_at"].isoformat()} for row in tokens}
    return {
        "tiktok": connected.get("tiktok", {"connected": False}),
        "youtube": connected.get("google", {"connected": False}),
        "facebook": connected.get("meta", {"connected": False}),
        "instagram": connected.get("meta", {"connected": False}),
    }

# ============================================================
# OAuth Routes - TikTok
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
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=tiktok_oauth_failed")
    if not state or "|" not in state:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=invalid_state")

    parts = state.split("|")
    user_id = verify_access_jwt(parts[0])
    code_verifier = parts[1] if len(parts) > 1 else ""
    if not user_id:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=invalid_state")

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
        if isinstance(token_data, dict) and token_data.get('expires_in') is not None:
            token_data['_obtained_at'] = time.time()
            try:
                token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
            except Exception:
                pass

    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=tiktok_token_failed")

    token_blob = encrypt_blob(token_data)

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
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=tiktok_connected")

# ============================================================
# OAuth Routes - Google/YouTube
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
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=google_oauth_failed")

    user_id = verify_access_jwt(state)
    if not user_id:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=invalid_state")

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
        if isinstance(token_data, dict) and token_data.get('expires_in') is not None:
            token_data['_obtained_at'] = time.time()
            try:
                token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
            except Exception:
                pass

    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=google_token_failed")

    token_blob = encrypt_blob(token_data)

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
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=youtube_connected")

# ============================================================
# OAuth Routes - Meta (Facebook/Instagram)
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
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=meta_oauth_failed")

    user_id = verify_access_jwt(state)
    if not user_id:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=invalid_state")

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
        if isinstance(token_data, dict) and token_data.get('expires_in') is not None:
            token_data['_obtained_at'] = time.time()
            try:
                token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
            except Exception:
                pass

    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=meta_token_failed")

    token_blob = encrypt_blob(token_data)

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
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=meta_connected")

# ============================================================
# Analytics Routes (User KPI)
# ============================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview(days: int = 30, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")

    days = max(1, min(days, 365))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)

    async with db_pool.acquire() as conn:
        total_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE user_id = $1", user_id)
        period_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE user_id = $1 AND created_at >= $2", user_id, since)
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

        uploads_by_status = await conn.fetch(
            """SELECT status, COUNT(*)::int AS c
               FROM uploads WHERE user_id=$1 AND created_at >= $2
               GROUP BY status ORDER BY c DESC""",
            user_id, since
        )

    success_rate = (float(successful) / float(period_uploads) * 100.0) if period_uploads else 0.0
    return {
        "window_days": days,
        "total_uploads": int(total_uploads or 0),
        "period_uploads": int(period_uploads or 0),
        "success_rate": round(success_rate, 1),
        "avg_trill_score": round(float(avg_trill or 0), 1),
        "uploads_by_status": {r["status"]: r["c"] for r in uploads_by_status},
        "quota_used": int(user["uploads_this_month"]),
        "quota_total": int(user["upload_quota"]),
    }

@app.get("/api/analytics/timeseries")
async def analytics_timeseries(days: int = 30, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    days = max(1, min(days, 365))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)

    async with db_pool.acquire() as conn:
        uploads_daily = await conn.fetch(
            """
            SELECT date_trunc('day', created_at) AS day, COUNT(*)::int AS c
            FROM uploads
            WHERE user_id=$1 AND created_at >= $2
            GROUP BY day
            ORDER BY day ASC
            """,
            user_id, since
        )
        events_daily = await conn.fetch(
            """
            SELECT date_trunc('day', created_at) AS day, event_type, COUNT(*)::int AS c
            FROM analytics_events
            WHERE user_id=$1 AND created_at >= $2
            GROUP BY day, event_type
            ORDER BY day ASC
            """,
            user_id, since
        )

    return {
        "window_days": days,
        "uploads_daily": [{"day": r["day"].date().isoformat(), "count": r["c"]} for r in uploads_daily],
        "events_daily": [{"day": r["day"].date().isoformat(), "event_type": r["event_type"], "count": r["c"]} for r in events_daily],
    }


# ============================================================
# KPI (Commercial-grade) — Summary + Raw Extract + HTML UI
# ============================================================

@app.get("/api/kpi/summary")
async def kpi_summary(days: int = 30, platform: Optional[str] = None, status: Optional[str] = None, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    days = max(1, min(days, 365))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)

    filters = ["user_id=$1", "created_at >= $2"]
    args = [user_id, since]
    argn = 3
    if platform:
        filters.append(f"$%d = ANY(platforms)" % argn)
        args.append(platform)
        argn += 1
    if status:
        filters.append(f"status = $%d" % argn)
        args.append(status)
        argn += 1
    where = " AND ".join(filters)

    async with db_pool.acquire() as conn:
        totals = await conn.fetchrow(
            f"""SELECT
                    COUNT(*)::int AS total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)::int AS completed,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END)::int AS failed,
                    SUM(CASE WHEN status IN ('queued','processing') THEN 1 ELSE 0 END)::int AS backlog
                 FROM uploads
                 WHERE {where}""",
            *args
        )

        latency = await conn.fetchrow(
            f"""SELECT
                    AVG(EXTRACT(EPOCH FROM (COALESCE(processing_finished_at, completed_at, NOW()) - COALESCE(processing_started_at, created_at)))) AS avg_s,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (COALESCE(processing_finished_at, completed_at, NOW()) - COALESCE(processing_started_at, created_at)))) AS p95_s
                 FROM uploads
                 WHERE {where} AND (processing_started_at IS NOT NULL OR completed_at IS NOT NULL)""",
            *args
        )

        mix = await conn.fetch(
            f"""SELECT p AS platform, COUNT(*)::int AS c
                 FROM uploads, UNNEST(platforms) AS p
                 WHERE {where}
                 GROUP BY p
                 ORDER BY c DESC""",
            *args
        )

    total = int(totals["total"] or 0)
    completed = int(totals["completed"] or 0)
    failed = int(totals["failed"] or 0)
    backlog = int(totals["backlog"] or 0)

    success_rate = (completed / total) if total else 0.0
    failure_rate = (failed / total) if total else 0.0

    avg_s = float(latency["avg_s"]) if latency and latency["avg_s"] is not None else None
    p95_s = float(latency["p95_s"]) if latency and latency["p95_s"] is not None else None

    mix_list = [{"platform": r["platform"], "count": r["c"]} for r in mix]
    top_share = (mix_list[0]["count"] / total) if (mix_list and total) else 0.0

    return {
        "window_days": days,
        "filters": {"platform": platform, "status": status},
        "throughput": {
            "uploads_total": total,
            "uploads_per_day": round(total / days, 3) if days else 0,
            "uploads_per_month_equiv": round((total / days) * 30, 3) if days else 0,
        },
        "reliability": {"success_rate": success_rate, "failure_rate": failure_rate, "completed": completed, "failed": failed},
        "latency": {"avg_processing_s": avg_s, "p95_processing_s": p95_s},
        "backlog": {"queued_processing": backlog},
        "platform_mix": mix_list,
        "platform_concentration": {"top_platform_share": top_share},
        "finance": {"mrr": None, "churn": None, "arpa": None, "gross_margin": None, "cac_payback": None},
    }

@app.get("/api/kpi/raw")
async def kpi_raw(days: int = 30, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    days = max(1, min(days, 365))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, filename, title, status, platforms, r2_key, processed_r2_key,
                      created_at, scheduled_time, processing_started_at, processing_finished_at, completed_at,
                      error_code, error_detail
                 FROM uploads
                 WHERE user_id=$1 AND created_at >= $2
                 ORDER BY created_at DESC
                 LIMIT 5000""",
            user_id, since
        )

    def _row(r):
        return {
            "id": str(r["id"]),
            "filename": r["filename"],
            "title": r["title"],
            "status": r["status"],
            "platforms": r["platforms"],
            "r2_key": r["r2_key"],
            "processed_r2_key": r["processed_r2_key"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "scheduled_time": r["scheduled_time"].isoformat() if r["scheduled_time"] else None,
            "processing_started_at": r["processing_started_at"].isoformat() if r["processing_started_at"] else None,
            "processing_finished_at": r["processing_finished_at"].isoformat() if r["processing_finished_at"] else None,
            "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            "error_code": r["error_code"],
            "error_detail": r["error_detail"],
        }

    return {"window_days": days, "rows": [_row(r) for r in rows]}

@app.get("/kpi.html")
async def kpi_html():
    # Single-file KPI dashboard. Pulls data from /api/kpi/* endpoints.
    html = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>UploadM8 KPI</title>
  <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
  <script src='https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js'></script>
  <style>
    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:20px;}
    .row{display:flex; gap:12px; flex-wrap:wrap;}
    .card{border:1px solid #2223; border-radius:12px; padding:12px; min-width:220px;}
    .grid{display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap:12px;}
    .k{font-size:12px; opacity:.7}
    .v{font-size:24px; font-weight:700}
    .btn{padding:10px 12px; border:1px solid #2223; border-radius:10px; background:#fff; cursor:pointer}
    input,select{padding:10px; border-radius:10px; border:1px solid #2223}
    canvas{max-height:320px}
  </style>
</head>
<body>
  <h2>UploadM8 KPI Dashboard</h2>
  <div class='row'>
    <label>Days <input id='days' type='number' min='1' max='365' value='30'/></label>
    <label>Platform
      <select id='platform'>
        <option value=''>All</option>
        <option value='tiktok'>TikTok</option>
        <option value='instagram'>Instagram</option>
        <option value='facebook'>Facebook</option>
        <option value='youtube'>YouTube</option>
      </select>
    </label>
    <label>Status
      <select id='status'>
        <option value=''>All</option>
        <option value='queued'>Queued</option>
        <option value='processing'>Processing</option>
        <option value='completed'>Completed</option>
        <option value='failed'>Failed</option>
      </select>
    </label>
    <button class='btn' id='refresh'>Refresh</button>
    <button class='btn' id='export'>Export Excel</button>
  </div>

  <div class='grid' style='margin-top:14px'>
    <div class='card'><div class='k'>Uploads/day</div><div class='v' id='u_day'>—</div></div>
    <div class='card'><div class='k'>Success rate</div><div class='v' id='succ'>—</div></div>
    <div class='card'><div class='k'>Avg processing (s)</div><div class='v' id='avg'>—</div></div>
    <div class='card'><div class='k'>P95 processing (s)</div><div class='v' id='p95'>—</div></div>
    <div class='card'><div class='k'>Backlog</div><div class='v' id='backlog'>—</div></div>
    <div class='card'><div class='k'>Top platform share</div><div class='v' id='topshare'>—</div></div>
  </div>

  <div class='grid' style='margin-top:14px'>
    <div class='card'><div class='k'>Throughput trend</div><canvas id='trend'></canvas></div>
    <div class='card'><div class='k'>Platform mix</div><canvas id='mix'></canvas></div>
  </div>

<script>
async function api(path, opts){
  opts = opts || {};
  opts.headers = Object.assign({'Content-Type':'application/json'}, opts.headers||{});
  const r = await fetch(path, opts);
  if(!r.ok){ throw new Error(await r.text()); }
  return await r.json();
}
function pct(x){ return (x*100).toFixed(1)+'%'; }

let trendChart=null, mixChart=null, rawCache=null;

async function refresh(){
  const days = document.getElementById('days').value;
  const platform = document.getElementById('platform').value;
  const status = document.getElementById('status').value;

  const q = new URLSearchParams({days});
  if(platform) q.set('platform', platform);
  if(status) q.set('status', status);

  const sum = await api('/api/kpi/summary?'+q.toString());
  document.getElementById('u_day').textContent = sum.throughput.uploads_per_day;
  document.getElementById('succ').textContent = pct(sum.reliability.success_rate);
  document.getElementById('avg').textContent = sum.latency.avg_processing_s==null?'—':sum.latency.avg_processing_s.toFixed(1);
  document.getElementById('p95').textContent = sum.latency.p95_processing_s==null?'—':sum.latency.p95_processing_s.toFixed(1);
  document.getElementById('backlog').textContent = sum.backlog.queued_processing;
  document.getElementById('topshare').textContent = pct(sum.platform_concentration.top_platform_share);

  const ts = await api('/api/analytics/timeseries?days='+days);
  const labels = ts.uploads_daily.map(x=>x.day);
  const values = ts.uploads_daily.map(x=>x.count);
  if(trendChart) trendChart.destroy();
  trendChart = new Chart(document.getElementById('trend'), {type:'line', data:{labels, datasets:[{label:'Uploads/day', data:values}]} });

  const mixLabels = sum.platform_mix.map(x=>x.platform);
  const mixValues = sum.platform_mix.map(x=>x.count);
  if(mixChart) mixChart.destroy();
  mixChart = new Chart(document.getElementById('mix'), {type:'doughnut', data:{labels:mixLabels, datasets:[{label:'Mix', data:mixValues}]} });

  rawCache = await api('/api/kpi/raw?days='+days);
}

async function exportExcel(){
  if(!rawCache){ await refresh(); }
  const wb = XLSX.utils.book_new();

  // KPI sheet
  const days = document.getElementById('days').value;
  const platform = document.getElementById('platform').value;
  const status = document.getElementById('status').value;
  const sum = await api('/api/kpi/summary?'+new URLSearchParams({days,platform,status}).toString());

  const kpiRows = [
    ['Window (days)', sum.window_days],
    ['Uploads/day', sum.throughput.uploads_per_day],
    ['Uploads/month equiv', sum.throughput.uploads_per_month_equiv],
    ['Success rate', sum.reliability.success_rate],
    ['Failure rate', sum.reliability.failure_rate],
    ['Avg processing (s)', sum.latency.avg_processing_s],
    ['P95 processing (s)', sum.latency.p95_processing_s],
    ['Backlog', sum.backlog.queued_processing],
    ['Top platform share', sum.platform_concentration.top_platform_share],
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(kpiRows), 'KPI');

  // Raw extract
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rawCache.rows), 'Uploads');

  XLSX.writeFile(wb, 'uploadm8_kpi.xlsx');
}

document.getElementById('refresh').onclick = ()=>refresh().catch(e=>alert(e));
document.getElementById('export').onclick = ()=>exportExcel().catch(e=>alert(e));
refresh().catch(e=>alert(e));
</script>
</body></html>"""
    return Response(content=html, media_type="text/html")

@app.get("/pricing.html")
async def pricing_page():
    html = """<!doctype html>
<html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>UploadM8 Pricing</title>
<style>
  body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:20px;}
  .card{border:1px solid #2223; border-radius:12px; padding:14px; max-width:520px;}
  .btn{padding:10px 12px; border:1px solid #2223; border-radius:10px; background:#fff; cursor:pointer}
  select,input{padding:10px; border-radius:10px; border:1px solid #2223; width:100%;}
</style>
</head>
<body>
<h2>UploadM8 Pricing</h2>
<div class='card'>
  <div style='margin-bottom:10px'>Select plan</div>
  <select id='plan'></select>
  <div style='margin:12px 0'>
    <label>Trial days (optional)<input id='trial' type='number' min='0' max='60' value='0'/></label>
  </div>
  <button class='btn' id='checkout'>Checkout</button>
  <div id='msg' style='margin-top:10px; opacity:.75'></div>
</div>
<script>
async function api(path, opts){
  opts = opts || {};
  opts.headers = Object.assign({'Content-Type':'application/json'}, opts.headers||{});
  const r = await fetch(path, opts);
  if(!r.ok){ throw new Error(await r.text()); }
  return await r.json();
}
async function load(){
  const data = await api('/api/billing/prices');
  const sel = document.getElementById('plan');
  sel.innerHTML = '';
  if(!data.configured || !data.prices.length){
    document.getElementById('msg').textContent = 'Stripe not configured or no prices.';
    return;
  }
  data.prices.forEach(p=>{
    const amt = (p.unit_amount/100).toFixed(2);
    const interval = (p.recurring && p.recurring.interval) ? p.recurring.interval : '';
    const opt = document.createElement('option');
    opt.value = p.lookup_key;
    opt.textContent = (p.product && p.product.name ? p.product.name : p.lookup_key) + ' — $' + amt + '/' + interval;
    sel.appendChild(opt);
  });
}
document.getElementById('checkout').onclick = async ()=>{
  const lookup_key = document.getElementById('plan').value;
  const trial_days = parseInt(document.getElementById('trial').value || '0',10);
  const s = await api('/api/billing/checkout', {method:'POST', body: JSON.stringify({lookup_key, trial_days})});
  window.location = s.url;
};
load().catch(e=>document.getElementById('msg').textContent = e.toString());
</script>
</body></html>"""
    return Response(content=html, media_type="text/html")

@app.get("/billing-success.html")
async def billing_success_page(session_id: str = ""):
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Subscription Active</title></head>
<body style='font-family:system-ui; margin:20px'>
  <h2>Subscription confirmed</h2>
  <p>Session: <code>{session_id}</code></p>
  <p>You can now return to your dashboard.</p>
  <a href='{FRONTEND_URL}/dashboard.html'>Go to Dashboard</a>
</body></html>"""
    return Response(content=html, media_type="text/html")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
