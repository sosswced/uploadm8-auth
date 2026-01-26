"""
UploadM8 Auth Service (Commercial Production Build)
====================================================
FastAPI backend for UploadM8 SaaS - Multi-platform video upload service

Features:
- Auth (email/password) + refresh tokens (rotation)
- Password reset via Mailgun
- R2 presigned URLs for direct browser uploads (video + sidecar files)
- Multi-platform OAuth (TikTok, YouTube, Meta)
- Upload tracking & analytics + Commercial KPI dashboards
- Stripe billing (Checkout + Portal + Webhooks + trials + auto tax)
- Redis job queue for async video processing
- Distributed rate limiting (Redis-backed)
- Admin Discord webhook notifications
- Schema migrations + observability + security headers
"""

import os
import json
import secrets
import hashlib
import base64
import logging
import time
import uuid
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import httpx
import asyncpg
import jwt
import bcrypt
import boto3
from botocore.config import Config
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import stripe
import redis.asyncio as redis

from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, Request, Response, UploadFile, File, Form
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
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.uploadm8.com")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "uploadm8-app")

ACCESS_TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.environ.get("REFRESH_TOKEN_DAYS", "30"))
TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://app.uploadm8.com,https://uploadm8.com")

# Admin
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()

# R2 Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")

# Redis (queue + distributed rate limiting)
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")
TELEMETRY_JOB_QUEUE = os.environ.get("TELEMETRY_JOB_QUEUE", "uploadm8:telemetry")

# Stripe Billing
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_DEFAULT_LOOKUP_KEY = os.environ.get("STRIPE_DEFAULT_LOOKUP_KEY", "uploadm8_creator_monthly")
STRIPE_LOOKUP_KEYS = os.environ.get("STRIPE_LOOKUP_KEYS", "uploadm8_starter_monthly,uploadm8_solo_monthly,uploadm8_creator_monthly,uploadm8_growth_monthly,uploadm8_studio_monthly,uploadm8_agency_monthly")
STRIPE_SUCCESS_URL = os.environ.get("STRIPE_SUCCESS_URL", f"{FRONTEND_URL}/billing-success.html?session_id={{CHECKOUT_SESSION_ID}}")
STRIPE_CANCEL_URL = os.environ.get("STRIPE_CANCEL_URL", f"{FRONTEND_URL}/index.html#pricing")
STRIPE_PORTAL_RETURN_URL = os.environ.get("STRIPE_PORTAL_RETURN_URL", f"{FRONTEND_URL}/dashboard.html")
STRIPE_TRIAL_DAYS_DEFAULT = int(os.environ.get("STRIPE_TRIAL_DAYS_DEFAULT", "0"))
STRIPE_AUTOMATIC_TAX = os.environ.get("STRIPE_AUTOMATIC_TAX", "0")

# Admin Discord notifications
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")

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

# Token refresh skew
TOKEN_REFRESH_SKEW_SEC = int(os.environ.get("TOKEN_REFRESH_SKEW_SEC", "300"))

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
# Env Validation
# ============================================================

def parse_enc_keys() -> Dict[str, bytes]:
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
        raise RuntimeError("TOKEN_ENC_KEYS parsed empty/invalid")
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
    CURRENT_KEY_ID = list(ENC_KEYS.keys())[-1]

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
    schedule_mode: Optional[str] = "immediate"
    preferred_windows: Optional[str] = ""
    has_telemetry: bool = False
    telemetry_filename: Optional[str] = None

class UploadComplete(BaseModel):
    telemetry_key: Optional[str] = None

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
    hud_speed_unit: Optional[str] = None
    hud_color: Optional[str] = None
    hud_font_family: Optional[str] = None
    hud_font_size: Optional[int] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class AdminGrantEntitlement(BaseModel):
    email: EmailStr
    tier: str = Field(default="lifetime")
    upload_quota: Optional[int] = None
    note: Optional[str] = None

class AdminSetRole(BaseModel):
    email: EmailStr
    role: str = Field(default="admin")

class CheckoutRequest(BaseModel):
    lookup_key: Optional[str] = None
    trial_days: Optional[int] = None

class PortalRequest(BaseModel):
    return_url: Optional[str] = None

class WebhookTestRequest(BaseModel):
    discord_webhook: Optional[str] = None

# ============================================================
# Rate Limiting (Memory + Redis fallback)
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

async def check_rate_limit_any(key: str):
    """Redis-backed sliding window when configured; fallback to in-memory."""
    if redis_client is None:
        return check_rate_limit(key)
    
    now = int(time.time())
    window = RATE_LIMIT_WINDOW_SEC
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
        logger.warning(f"Redis rate limiter error (fallback to memory): {e}")
        return check_rate_limit(key)

# ============================================================
# Password Hashing
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
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
    }

def decrypt_blob(blob: Any) -> dict:
    if isinstance(blob, str):
        blob = json.loads(blob)
    kid = blob.get("kid", "v1")
    key = ENC_KEYS.get(kid)
    if not key:
        raise ValueError(f"Unknown key id: {kid}")
    nonce = base64.b64decode(blob["nonce"])
    ciphertext = base64.b64decode(blob["ciphertext"])
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))

# ============================================================
# JWT Helpers
# ============================================================

def create_access_jwt(user_id: str) -> str:
    now = _now_utc()
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_access_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        if payload.get("type") not in ("access", None):
            return None
        return payload.get("sub")
    except Exception:
        return None

async def create_refresh_token(conn, user_id: str) -> str:
    token_raw = secrets.token_urlsafe(64)
    token_hash = _sha256_hex(token_raw)
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_DAYS)
    await conn.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
           VALUES ($1, $2, $3)""",
        user_id, token_hash, expires_at
    )
    return token_raw

async def rotate_refresh_token(conn, old_token: str):
    old_hash = _sha256_hex(old_token)
    row = await conn.fetchrow(
        """SELECT id, user_id, expires_at, revoked FROM refresh_tokens WHERE token_hash=$1""",
        old_hash
    )
    if not row:
        raise HTTPException(401, "Invalid refresh token")
    if row["revoked"]:
        await conn.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE user_id=$1", row["user_id"])
        raise HTTPException(401, "Token reuse detected")
    if row["expires_at"] < _now_utc():
        raise HTTPException(401, "Refresh token expired")
    
    await conn.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE id=$1", row["id"])
    new_access = create_access_jwt(row["user_id"])
    new_refresh = await create_refresh_token(conn, row["user_id"])
    return new_access, new_refresh

async def revoke_refresh_token(conn, token: str):
    token_hash = _sha256_hex(token)
    await conn.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE token_hash=$1", token_hash)

# ============================================================
# R2 Presigned URLs
# ============================================================

def get_s3_client():
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def generate_presigned_upload_url(key: str, content_type: str, ttl: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type},
        ExpiresIn=ttl,
    )

def generate_presigned_get_url(key: str, ttl_seconds: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key},
        ExpiresIn=ttl_seconds,
    )

# ============================================================
# Discord Notifications
# ============================================================

async def _discord_notify_admin(event_type: str, data: dict):
    if not ADMIN_DISCORD_WEBHOOK_URL:
        return
    try:
        msg = f"**UploadM8 Event**: `{event_type}`\n```json\n{json.dumps(data, indent=2, default=str)[:1800]}\n```"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(ADMIN_DISCORD_WEBHOOK_URL, json={"content": msg})
    except Exception as e:
        logger.warning(f"Discord notify failed: {e}")

async def _discord_notify_user(webhook_url: str, message: str):
    if not webhook_url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(webhook_url, json={"content": message})
    except Exception as e:
        logger.warning(f"User Discord notify failed: {e}")

# ============================================================
# Redis Job Queue
# ============================================================

async def enqueue_job(job_data: dict, queue: str = None):
    """Enqueue a job to Redis for async processing."""
    if redis_client is None:
        logger.warning("Redis not configured; job not enqueued (will process inline if possible)")
        return False
    
    queue = queue or UPLOAD_JOB_QUEUE
    job_data["enqueued_at"] = _now_utc().isoformat()
    job_data["job_id"] = str(uuid.uuid4())
    
    try:
        await redis_client.lpush(queue, json.dumps(job_data))
        logger.info(f"Job enqueued: {job_data.get('job_id')} to {queue}")
        return True
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        return False

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
                user_id, event_type, json.dumps(event_data or {})
            )
    except Exception as e:
        logger.warning(f"Failed to track event: {e}")

# ============================================================
# Admin Audit Log
# ============================================================

async def admin_audit_log(conn, admin: dict, action: str, target_user_id: str = None, target_email: str = None, meta: dict = None):
    try:
        await conn.execute(
            """INSERT INTO admin_audit_log (admin_id, admin_email, action, target_user_id, target_email, meta)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            str(admin["id"]), admin.get("email"), action, target_user_id, target_email, json.dumps(meta or {})
        )
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")

# ============================================================
# Stripe Tier Mapping
# ============================================================

TIER_ENTITLEMENTS = {
    "starter": {"tier": "starter", "upload_quota": 10, "unlimited_uploads": False, "max_accounts": 1},
    "solo": {"tier": "solo", "upload_quota": 60, "unlimited_uploads": False, "max_accounts": 2},
    "creator": {"tier": "creator", "upload_quota": 200, "unlimited_uploads": False, "max_accounts": 4},
    "growth": {"tier": "growth", "upload_quota": 500, "unlimited_uploads": False, "max_accounts": 8},
    "studio": {"tier": "studio", "upload_quota": 1500, "unlimited_uploads": False, "max_accounts": 15},
    "agency": {"tier": "agency", "upload_quota": 5000, "unlimited_uploads": False, "max_accounts": 40},
    "lifetime": {"tier": "lifetime", "upload_quota": 999999, "unlimited_uploads": True, "max_accounts": 100},
    "friends_family": {"tier": "friends_family", "upload_quota": 999999, "unlimited_uploads": True, "max_accounts": 100},
    "free": {"tier": "free", "upload_quota": 10, "unlimited_uploads": False, "max_accounts": 1},
}

def _tier_from_lookup_key(lookup_key: str) -> str:
    lk = lookup_key.lower()
    for tier in TIER_ENTITLEMENTS.keys():
        if tier in lk:
            return tier
    return "free"

def _entitlements_for_tier(tier: str) -> dict:
    return TIER_ENTITLEMENTS.get(tier.lower(), TIER_ENTITLEMENTS["free"])

# ============================================================
# Global State
# ============================================================

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
origins = _split_origins(ALLOWED_ORIGINS)

# ============================================================
# Lifespan
# ============================================================

MIGRATIONS = [
    ("v1", """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL,
            subscription_tier VARCHAR(50) DEFAULT 'free',
            role VARCHAR(50) DEFAULT 'user',
            upload_quota INT DEFAULT 10,
            uploads_this_month INT DEFAULT 0,
            quota_reset_date DATE DEFAULT CURRENT_DATE,
            unlimited_uploads BOOLEAN DEFAULT FALSE,
            stripe_customer_id VARCHAR(255),
            stripe_subscription_id VARCHAR(255),
            subscription_status VARCHAR(50),
            current_period_end TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            token_hash VARCHAR(255) UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            revoked BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(50) PRIMARY KEY, applied_at TIMESTAMP DEFAULT NOW());
    """),
    ("v2", """
        CREATE TABLE IF NOT EXISTS platform_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            platform VARCHAR(50) NOT NULL,
            token_blob JSONB NOT NULL,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, platform)
        );
    """),
    ("v3", """
        CREATE TABLE IF NOT EXISTS uploads (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            r2_key VARCHAR(500) NOT NULL,
            processed_r2_key VARCHAR(500),
            telemetry_r2_key VARCHAR(500),
            filename VARCHAR(255) NOT NULL,
            file_size BIGINT,
            platforms TEXT[],
            title VARCHAR(500),
            caption TEXT,
            privacy VARCHAR(50) DEFAULT 'public',
            status VARCHAR(50) DEFAULT 'pending',
            trill_score INT,
            scheduled_time TIMESTAMP,
            schedule_mode VARCHAR(50) DEFAULT 'immediate',
            processing_started_at TIMESTAMP,
            processing_finished_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_code VARCHAR(100),
            error_detail TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user_id);
        CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
    """),
    ("v4", """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            discord_webhook VARCHAR(500),
            telemetry_enabled BOOLEAN DEFAULT TRUE,
            hud_enabled BOOLEAN DEFAULT TRUE,
            hud_position VARCHAR(50) DEFAULT 'bottom-left',
            speeding_mph INT DEFAULT 80,
            euphoria_mph INT DEFAULT 100,
            fb_user_id VARCHAR(255),
            selected_page_id VARCHAR(255),
            selected_page_name VARCHAR(255),
            hud_speed_unit VARCHAR(10) DEFAULT 'mph',
            hud_color VARCHAR(20) DEFAULT '#FFFFFF',
            hud_font_family VARCHAR(100) DEFAULT 'Plus Jakarta Sans',
            hud_font_size INT DEFAULT 18,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """),
    ("v5", """
        CREATE TABLE IF NOT EXISTS analytics_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            event_type VARCHAR(100) NOT NULL,
            event_data JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics_events(user_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type);
    """),
    ("v6", """
        CREATE TABLE IF NOT EXISTS entitlements (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            tier VARCHAR(50) NOT NULL,
            upload_quota_override INT,
            is_lifetime BOOLEAN DEFAULT FALSE,
            note TEXT,
            granted_by UUID,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            admin_id UUID,
            admin_email VARCHAR(255),
            action VARCHAR(100) NOT NULL,
            target_user_id UUID,
            target_email VARCHAR(255),
            meta JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """),
    ("v7", """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            token_hash VARCHAR(255) UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """),
    ("v8", """
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS telemetry_r2_key VARCHAR(500);
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS schedule_mode VARCHAR(50) DEFAULT 'immediate';
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMP;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_finished_at TIMESTAMP;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS error_code VARCHAR(100);
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS error_detail TEXT;
    """),
    ("v9", """
        CREATE TABLE IF NOT EXISTS job_queue (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_type VARCHAR(50) NOT NULL,
            payload JSONB NOT NULL,
            status VARCHAR(50) DEFAULT 'pending',
            attempts INT DEFAULT 0,
            max_attempts INT DEFAULT 3,
            last_error TEXT,
            scheduled_for TIMESTAMP DEFAULT NOW(),
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON job_queue(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_scheduled ON job_queue(scheduled_for);
    """),
]

async def run_migrations(conn):
    await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(50) PRIMARY KEY, applied_at TIMESTAMP DEFAULT NOW())")
    applied = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
    for version, sql in MIGRATIONS:
        if version not in applied:
            logger.info(f"Applying migration {version}")
            try:
                await conn.execute(sql)
                await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
            except Exception as e:
                logger.warning(f"Migration {version} partial: {e}")
                try:
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT DO NOTHING", version)
                except:
                    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client
    
    validate_env()
    init_enc_keys()
    
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
    
    # Database
    if DATABASE_URL:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=20)
        async with db_pool.acquire() as conn:
            await run_migrations(conn)
            # Bootstrap admin
            if BOOTSTRAP_ADMIN_EMAIL:
                await conn.execute(
                    "UPDATE users SET role='admin' WHERE lower(email)=$1 AND role='user'",
                    BOOTSTRAP_ADMIN_EMAIL
                )
        logger.info("Database connected")
    
    # Redis
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    
    yield
    
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="UploadM8 API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Auth Dependencies
# ============================================================

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization[7:]
    user_id = verify_access_jwt(token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, email, name, subscription_tier, role, upload_quota, 
                      uploads_this_month, unlimited_uploads, stripe_customer_id
               FROM users WHERE id = $1""",
            user_id
        )
    if not row:
        raise HTTPException(401, "User not found")
    return dict(row)

async def require_admin_api_key(x_api_key: str = Header(None)):
    if not ADMIN_API_KEY:
        raise HTTPException(403, "Admin API not configured")
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(403, "Invalid admin API key")
    return True

async def require_admin_role(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin role required")
    return user

# ============================================================
# Middleware
# ============================================================

@app.middleware("http")
async def request_middleware(request: Request, call_next):
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
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# ============================================================
# Health & Status
# ============================================================

@app.get("/")
async def root():
    return {"message": "UploadM8 Auth Service", "status": "running", "version": "2.0.0"}

@app.get("/health")
async def health_alias():
    return {"status": "ok"}

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "database": db_pool is not None,
        "redis": redis_client is not None,
        "stripe": bool(STRIPE_SECRET_KEY),
        "r2_configured": bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID),
        "platforms": {"tiktok": bool(TIKTOK_CLIENT_KEY), "youtube": bool(GOOGLE_CLIENT_ID), "meta": bool(META_APP_ID)},
        "frontend_url": FRONTEND_URL,
    }

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
                    data.email.lower(), password_hash, data.name
                )
            except asyncpg.UniqueViolationError:
                raise HTTPException(400, "Email already registered")
            
            user_id = str(row["id"])
            await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
            
            access_token = create_access_jwt(user_id)
            refresh_token = await create_refresh_token(conn, user_id)
    
    background_tasks.add_task(track_event, user_id, "user_signup", {"email": data.email})
    background_tasks.add_task(_discord_notify_admin, "user_signup", {"email": data.email, "name": data.name})
    
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
            data.email.lower()
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

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"user": user}

# ============================================================
# Password Reset
# ============================================================

@app.post("/api/auth/password-reset")
async def request_password_reset(request: Request, data: PasswordReset, background_tasks: BackgroundTasks):
    await check_rate_limit_any(rate_limit_key(request, "password_reset"))
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id FROM users WHERE email = $1", data.email.lower())
        if user:
            token = secrets.token_urlsafe(32)
            token_hash = _sha256_hex(token)
            expires_at = _now_utc() + timedelta(hours=1)
            await conn.execute(
                "INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)",
                user["id"], token_hash, expires_at
            )
            
            if MAILGUN_API_KEY and MAILGUN_DOMAIN:
                reset_link = f"{FRONTEND_URL}/forgot-password.html?token={token}"
                background_tasks.add_task(
                    send_email,
                    data.email,
                    "UploadM8 Password Reset",
                    f"Click here to reset your password: {reset_link}\n\nThis link expires in 1 hour."
                )
    
    return {"status": "ok", "message": "If the email exists, a reset link has been sent."}

@app.post("/api/auth/password-reset/confirm")
async def confirm_password_reset(data: PasswordResetConfirm):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    token_hash = _sha256_hex(data.token)
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, user_id, expires_at, used FROM password_reset_tokens WHERE token_hash = $1""",
            token_hash
        )
        
        if not row or row["used"] or row["expires_at"] < _now_utc():
            raise HTTPException(400, "Invalid or expired token")
        
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, row["user_id"])
        await conn.execute("UPDATE password_reset_tokens SET used = TRUE WHERE id = $1", row["id"])
        await conn.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = $1", row["user_id"])
    
    return {"status": "ok"}

async def send_email(to: str, subject: str, text: str):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.warning("Mailgun not configured")
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={"from": MAIL_FROM, "to": to, "subject": subject, "text": text}
            )
    except Exception as e:
        logger.error(f"Email send failed: {e}")

# ============================================================
# User Settings
# ============================================================

@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", str(user["id"]))
    
    if not row:
        return {"settings": {}}
    
    return {"settings": dict(row)}

@app.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    updates = {k: v for k, v in data.dict().items() if v is not None}
    
    if not updates:
        return {"status": "ok"}
    
    async with db_pool.acquire() as conn:
        await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
        
        set_parts = [f"{k} = ${i+2}" for i, k in enumerate(updates.keys())]
        set_parts.append("updated_at = NOW()")
        sql = f"UPDATE user_settings SET {', '.join(set_parts)} WHERE user_id = $1"
        await conn.execute(sql, user_id, *updates.values())
    
    return {"status": "ok"}

# ============================================================
# Stripe Billing
# ============================================================

@app.get("/api/billing/prices")
async def get_billing_prices():
    if not STRIPE_SECRET_KEY:
        return {"configured": False, "prices": []}
    
    try:
        lookup_keys = [k.strip() for k in STRIPE_LOOKUP_KEYS.split(",") if k.strip()]
        prices = stripe.Price.list(lookup_keys=lookup_keys, expand=["data.product"], active=True)
        return {"configured": True, "prices": prices.data}
    except Exception as e:
        logger.error(f"Stripe prices fetch failed: {e}")
        return {"configured": True, "prices": [], "error": str(e)}

@app.post("/api/billing/checkout")
async def create_checkout_session(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")
    
    lookup_key = data.lookup_key or STRIPE_DEFAULT_LOOKUP_KEY
    trial_days = data.trial_days if data.trial_days is not None else STRIPE_TRIAL_DAYS_DEFAULT
    
    try:
        prices = stripe.Price.list(lookup_keys=[lookup_key], active=True)
        if not prices.data:
            raise HTTPException(400, f"Price not found for lookup_key: {lookup_key}")
        price = prices.data[0]
        
        # Get or create Stripe customer
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user.get("name"), metadata={"user_id": str(user["id"])})
            customer_id = customer.id
            async with db_pool.acquire() as conn:
                await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, str(user["id"]))
        
        session_params = {
            "customer": customer_id,
            "mode": "subscription",
            "line_items": [{"price": price.id, "quantity": 1}],
            "success_url": STRIPE_SUCCESS_URL,
            "cancel_url": STRIPE_CANCEL_URL,
            "metadata": {"user_id": str(user["id"]), "lookup_key": lookup_key},
            "subscription_data": {"metadata": {"user_id": str(user["id"]), "lookup_key": lookup_key}},
        }
        
        if trial_days > 0:
            session_params["subscription_data"]["trial_period_days"] = trial_days
        
        if STRIPE_AUTOMATIC_TAX == "1":
            session_params["automatic_tax"] = {"enabled": True}
        
        session = stripe.checkout.Session.create(**session_params)
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
                ent["tier"], ent["upload_quota"], ent["unlimited_uploads"],
                sub_id, status, current_period_end, user_id
            )

@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        return Response(status_code=204)
    
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    
    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.warning(f"Stripe webhook verification failed: {e}")
        return Response(status_code=400)
    
    etype = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}
    
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
                    user_id, f"stripe_{etype}", json.dumps({"id": event.get("id"), "type": etype})
                )
        
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
async def create_presigned_upload(data: UploadInit, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    # Check quota (skip for unlimited users)
    if not user.get("unlimited_uploads"):
        if user["uploads_this_month"] >= user["upload_quota"]:
            raise HTTPException(403, "Monthly upload quota exceeded. Please upgrade your plan.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = secrets.token_hex(4)
    r2_key = f"uploads/{user_id}/{timestamp}_{file_hash}_{data.filename}"
    
    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    
    # Handle telemetry file if specified
    telemetry_presigned_url = None
    telemetry_r2_key = None
    if data.has_telemetry and data.telemetry_filename:
        telemetry_r2_key = f"uploads/{user_id}/{timestamp}_{file_hash}_{data.telemetry_filename}"
        telemetry_presigned_url = generate_presigned_upload_url(telemetry_r2_key, "application/octet-stream")
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            """INSERT INTO uploads (user_id, r2_key, telemetry_r2_key, filename, file_size, platforms,
                   title, caption, privacy, status, scheduled_time, schedule_mode)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending', $10, $11)
               RETURNING id""",
            user_id, r2_key, telemetry_r2_key, data.filename, data.file_size, data.platforms,
            data.title, data.caption, data.privacy, data.scheduled_time, data.schedule_mode
        )
    
    upload_id = str(upload["id"])
    background_tasks.add_task(track_event, user_id, "upload_initiated", {"upload_id": upload_id, "platforms": data.platforms, "has_telemetry": data.has_telemetry})
    
    result = {
        "upload_id": upload_id,
        "presigned_url": presigned_url,
        "r2_key": r2_key,
        "expires_in": 3600
    }
    
    if telemetry_presigned_url:
        result["telemetry_presigned_url"] = telemetry_presigned_url
        result["telemetry_r2_key"] = telemetry_r2_key
    
    return result

@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, background_tasks: BackgroundTasks, data: UploadComplete = None, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user_id)
            if not upload:
                raise HTTPException(404, "Upload not found")
            
            # Update telemetry key if provided
            if data and data.telemetry_key:
                await conn.execute("UPDATE uploads SET telemetry_r2_key = $1 WHERE id = $2", data.telemetry_key, upload_id)
            
            await conn.execute("UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1", upload_id)
            
            if not user.get("unlimited_uploads"):
                await conn.execute("UPDATE users SET uploads_this_month = uploads_this_month + 1, updated_at = NOW() WHERE id = $1", user_id)
    
    # Enqueue job for processing
    job_data = {
        "type": "process_upload",
        "upload_id": upload_id,
        "user_id": user_id,
        "has_telemetry": bool(upload.get("telemetry_r2_key"))
    }
    
    # Try Redis queue first, fallback to inline processing
    enqueued = await enqueue_job(job_data)
    if not enqueued:
        # Mark as processing even without worker (for demo purposes)
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE uploads SET status = 'processing', processing_started_at = NOW() WHERE id = $1", upload_id)
    
    background_tasks.add_task(track_event, user_id, "upload_complete", {"upload_id": upload_id})
    
    # Notify user via Discord webhook if configured
    async with db_pool.acquire() as conn:
        settings = await conn.fetchrow("SELECT discord_webhook FROM user_settings WHERE user_id = $1", user_id)
        if settings and settings.get("discord_webhook"):
            background_tasks.add_task(_discord_notify_user, settings["discord_webhook"], f"📤 Upload queued: {upload.get('title') or upload.get('filename')}")
    
    return {"status": "queued", "upload_id": upload_id}

@app.post("/api/uploads/{upload_id}/cancel")
async def cancel_upload(upload_id: str, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT status FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user_id)
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        if upload["status"] in ("completed", "failed"):
            raise HTTPException(400, "Cannot cancel completed or failed upload")
        
        await conn.execute("UPDATE uploads SET status = 'cancelled', updated_at = NOW() WHERE id = $1", upload_id)
    
    return {"status": "cancelled", "upload_id": upload_id}

@app.get("/api/uploads")
async def list_uploads(limit: int = 20, offset: int = 0, status: Optional[str] = None, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch(
                """SELECT id, filename, platforms, title, status, trill_score, scheduled_time,
                          created_at, completed_at
                   FROM uploads WHERE user_id = $1 AND status = $2
                   ORDER BY created_at DESC LIMIT $3 OFFSET $4""",
                user_id, status, limit, offset
            )
        else:
            uploads = await conn.fetch(
                """SELECT id, filename, platforms, title, status, trill_score, scheduled_time,
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
async def presign_get_for_upload(upload_id: str, ttl: int = 3600, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    ttl = max(60, min(int(ttl), 24 * 3600))
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT id, user_id, r2_key, processed_r2_key FROM uploads WHERE id=$1 AND user_id=$2",
            upload_id, str(user["id"])
        )
    if not upload:
        raise HTTPException(404, "Upload not found")
    
    key = upload.get("processed_r2_key") or upload.get("r2_key")
    url = generate_presigned_get_url(key, ttl_seconds=ttl)
    return {"url": url, "expires_in": ttl, "r2_key": key}

# ============================================================
# Telemetry File Upload (Sidecar .map files)
# ============================================================

@app.post("/api/uploads/{upload_id}/telemetry")
async def upload_telemetry_presign(upload_id: str, filename: str = "telemetry.map", user: dict = Depends(get_current_user)):
    """Get presigned URL for uploading telemetry/map file associated with a video upload."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT id, r2_key FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user_id)
    if not upload:
        raise HTTPException(404, "Upload not found")
    
    # Create telemetry key based on video key
    base_key = upload["r2_key"].rsplit("_", 1)[0]
    telemetry_r2_key = f"{base_key}_{filename}"
    
    presigned_url = generate_presigned_upload_url(telemetry_r2_key, "application/octet-stream")
    
    return {
        "telemetry_presigned_url": presigned_url,
        "telemetry_r2_key": telemetry_r2_key,
        "expires_in": 3600
    }

@app.post("/api/uploads/{upload_id}/telemetry/complete")
async def complete_telemetry_upload(upload_id: str, telemetry_key: str, user: dict = Depends(get_current_user)):
    """Mark telemetry file upload as complete and trigger processing."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT id, status FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user_id)
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        await conn.execute("UPDATE uploads SET telemetry_r2_key = $1, updated_at = NOW() WHERE id = $2", telemetry_key, upload_id)
    
    # Enqueue telemetry processing job
    job_data = {
        "type": "process_telemetry",
        "upload_id": upload_id,
        "user_id": user_id,
        "telemetry_key": telemetry_key
    }
    await enqueue_job(job_data, TELEMETRY_JOB_QUEUE)
    
    return {"status": "telemetry_queued", "upload_id": upload_id}

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
# OAuth Routes
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
async def tiktok_oauth_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None), background_tasks: BackgroundTasks = None):
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
            token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
    
    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=tiktok_token_failed")
    
    token_blob = encrypt_blob(token_data)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'tiktok', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )
    
    if background_tasks:
        background_tasks.add_task(track_event, user_id, "platform_connected", {"platform": "tiktok"})
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=tiktok_connected")

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
async def google_oauth_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None), background_tasks: BackgroundTasks = None):
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
            token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
    
    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=google_token_failed")
    
    token_blob = encrypt_blob(token_data)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'google', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )
    
    if background_tasks:
        background_tasks.add_task(track_event, user_id, "platform_connected", {"platform": "youtube"})
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=youtube_connected")

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
async def meta_oauth_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None), background_tasks: BackgroundTasks = None):
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
            token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
    
    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=meta_token_failed")
    
    token_blob = encrypt_blob(token_data)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'meta', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )
    
    if background_tasks:
        background_tasks.add_task(track_event, user_id, "platform_connected", {"platform": "meta"})
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=meta_connected")

# ============================================================
# Analytics Routes (User)
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
            "SELECT COUNT(*) FROM uploads WHERE user_id = $1 AND status = 'completed' AND created_at >= $2",
            user_id, since
        )
        avg_trill = await conn.fetchval(
            "SELECT AVG(trill_score) FROM uploads WHERE user_id = $1 AND trill_score IS NOT NULL AND created_at >= $2",
            user_id, since
        )
        trill_count = await conn.fetchval(
            "SELECT COUNT(*) FROM uploads WHERE user_id = $1 AND trill_score IS NOT NULL",
            user_id
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
        "trill_unlocked": int(trill_count or 0) > 0,
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
            """SELECT date_trunc('day', created_at) AS day, COUNT(*)::int AS c
               FROM uploads WHERE user_id=$1 AND created_at >= $2
               GROUP BY day ORDER BY day ASC""",
            user_id, since
        )
    
    return {
        "window_days": days,
        "uploads_daily": [{"day": r["day"].date().isoformat(), "count": r["c"]} for r in uploads_daily],
    }

# ============================================================
# Commercial KPI Endpoints
# ============================================================

@app.get("/api/kpi/summary")
async def kpi_summary(days: int = 30, platform: Optional[str] = None, status: Optional[str] = None, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    days = max(1, min(days, 365))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)
    
    filters = ["user_id=$1", "created_at >= $2"]
    args: List[Any] = [user_id, since]
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
                    SUM(CASE WHEN status IN ('queued','processing','pending') THEN 1 ELSE 0 END)::int AS backlog,
                    AVG(trill_score) AS avg_trill
                 FROM uploads WHERE {where}""",
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
                 GROUP BY p ORDER BY c DESC""",
            *args
        )
        
        # Additional metrics for commercial dashboard
        hourly_dist = await conn.fetch(
            f"""SELECT EXTRACT(HOUR FROM created_at)::int AS hour, COUNT(*)::int AS c
                 FROM uploads WHERE {where}
                 GROUP BY hour ORDER BY hour""",
            *args
        )
        
        error_breakdown = await conn.fetch(
            f"""SELECT COALESCE(error_code, 'unknown') AS code, COUNT(*)::int AS c
                 FROM uploads WHERE {where} AND status = 'failed'
                 GROUP BY code ORDER BY c DESC LIMIT 10""",
            *args
        )
    
    total = int(totals["total"] or 0)
    completed = int(totals["completed"] or 0)
    failed = int(totals["failed"] or 0)
    backlog = int(totals["backlog"] or 0)
    avg_trill = float(totals["avg_trill"]) if totals["avg_trill"] else None
    
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
        "reliability": {
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "completed": completed,
            "failed": failed
        },
        "latency": {
            "avg_processing_s": avg_s,
            "p95_processing_s": p95_s
        },
        "backlog": {"queued_processing": backlog},
        "platform_mix": mix_list,
        "platform_concentration": {"top_platform_share": top_share},
        "quality": {"avg_trill_score": avg_trill},
        "hourly_distribution": [{"hour": r["hour"], "count": r["c"]} for r in hourly_dist],
        "error_breakdown": [{"code": r["code"], "count": r["c"]} for r in error_breakdown],
        "finance": {"mrr": None, "churn": None, "arpa": None, "gross_margin": None, "cac_payback": None},
    }

@app.get("/api/kpi/raw")
async def kpi_raw(days: int = 30, limit: int = 5000, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    days = max(1, min(days, 365))
    limit = max(1, min(limit, 10000))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, filename, title, status, platforms, r2_key, processed_r2_key, trill_score,
                      created_at, scheduled_time, processing_started_at, processing_finished_at, completed_at,
                      error_code, error_detail
                 FROM uploads
                 WHERE user_id=$1 AND created_at >= $2
                 ORDER BY created_at DESC
                 LIMIT $3""",
            user_id, since, limit
        )
    
    def _row(r):
        processing_time = None
        if r["processing_started_at"] and (r["processing_finished_at"] or r["completed_at"]):
            end = r["processing_finished_at"] or r["completed_at"]
            processing_time = (end - r["processing_started_at"]).total_seconds()
        
        return {
            "id": str(r["id"]),
            "filename": r["filename"],
            "title": r["title"],
            "status": r["status"],
            "platforms": r["platforms"],
            "trill_score": r["trill_score"],
            "processing_time_s": processing_time,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "scheduled_time": r["scheduled_time"].isoformat() if r["scheduled_time"] else None,
            "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            "error_code": r["error_code"],
            "error_detail": r["error_detail"],
        }
    
    return {"window_days": days, "rows": [_row(r) for r in rows]}

# ============================================================
# Admin KPI (Global - Role-based)
# ============================================================

@app.get("/api/admin/kpi/global")
async def admin_global_kpi(days: int = 30, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    days = max(1, min(days, 365))
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        # User metrics
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        paying_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'starter')")
        
        # Upload metrics
        total_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1", since)
        completed_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status = 'completed' AND created_at >= $1", since)
        failed_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status = 'failed' AND created_at >= $1", since)
        
        # Platform breakdown
        platform_counts = await conn.fetch(
            """SELECT p AS platform, COUNT(*)::int AS c
               FROM uploads, UNNEST(platforms) AS p
               WHERE created_at >= $1
               GROUP BY p ORDER BY c DESC""",
            since
        )
        
        # Tier distribution
        tier_dist = await conn.fetch(
            "SELECT subscription_tier, COUNT(*)::int AS c FROM users GROUP BY subscription_tier ORDER BY c DESC"
        )
        
        # Revenue estimation (based on tier)
        mrr_estimate = await conn.fetchval(
            """SELECT SUM(CASE
                   WHEN subscription_tier = 'solo' THEN 9.99
                   WHEN subscription_tier = 'creator' THEN 19.99
                   WHEN subscription_tier = 'growth' THEN 29.99
                   WHEN subscription_tier = 'studio' THEN 49.99
                   WHEN subscription_tier = 'agency' THEN 99.99
                   ELSE 0
               END) FROM users WHERE subscription_status = 'active'"""
        )
    
    success_rate = (int(completed_uploads or 0) / int(total_uploads)) * 100 if total_uploads else 0
    
    return {
        "window_days": days,
        "users": {
            "total": int(total_users or 0),
            "new_in_window": int(new_users or 0),
            "paying": int(paying_users or 0),
            "conversion_rate": round((int(paying_users or 0) / int(total_users)) * 100, 2) if total_users else 0
        },
        "uploads": {
            "total_in_window": int(total_uploads or 0),
            "completed": int(completed_uploads or 0),
            "failed": int(failed_uploads or 0),
            "success_rate": round(success_rate, 1),
            "uploads_per_day": round(int(total_uploads or 0) / days, 1) if days else 0
        },
        "platforms": [{"platform": r["platform"], "count": r["c"]} for r in platform_counts],
        "tier_distribution": [{"tier": r["subscription_tier"], "count": r["c"]} for r in tier_dist],
        "finance": {
            "mrr_estimate": round(float(mrr_estimate or 0), 2),
            "arr_estimate": round(float(mrr_estimate or 0) * 12, 2)
        }
    }

@app.get("/api/admin/overview")
async def admin_overview(days: int = 30, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    since = _now_utc() - timedelta(days=days)
    
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
        
        platform_connected = await conn.fetch(
            "SELECT platform, COUNT(DISTINCT user_id)::int AS c FROM platform_tokens GROUP BY platform ORDER BY c DESC"
        )
    
    return {
        "window_days": days,
        "users": {
            "total": int(total_users or 0),
            "new_in_window": int(new_users or 0),
            "admins": int(admins or 0),
            "lifetime_entitlements": int(lifetime or 0),
        },
        "uploads": {
            "total": int(uploads_total or 0),
            "in_window": int(uploads_period or 0),
            "by_status": {r["status"]: r["c"] for r in uploads_by_status},
        },
        "platforms_connected": {r["platform"]: r["c"] for r in platform_connected},
    }

# ============================================================
# Notifications
# ============================================================

@app.post("/api/notifications/test")
async def test_notification(data: WebhookTestRequest, user: dict = Depends(get_current_user)):
    webhook_url = data.discord_webhook
    if not webhook_url:
        async with db_pool.acquire() as conn:
            settings = await conn.fetchrow("SELECT discord_webhook FROM user_settings WHERE user_id = $1", str(user["id"]))
            webhook_url = settings.get("discord_webhook") if settings else None
    
    if not webhook_url:
        raise HTTPException(400, "No webhook URL provided or configured")
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook_url, json={"content": "✅ **UploadM8 Test**: Webhook delivery confirmed!"})
            if resp.status_code >= 400:
                raise HTTPException(400, f"Webhook returned status {resp.status_code}")
        return {"status": "ok", "message": "Test notification sent"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Webhook test failed: {str(e)}")

# ============================================================
# Admin User Management
# ============================================================

@app.get("/api/admin/user-role")
async def admin_user_role(user: dict = Depends(get_current_user)):
    role = (user.get("role") or "user").lower()
    return {"role": role, "is_admin": role == "admin"}

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
        await admin_audit_log(conn, admin, action="set_role", target_user_id=str(target["id"]), target_email=target["email"], meta={"role": role})
    
    return {"status": "ok", "email": payload.email.lower(), "role": role}

@app.post("/api/admin/entitlements/grant")
async def admin_grant_entitlement(payload: AdminGrantEntitlement, admin: dict = Depends(require_admin_role)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    tier = payload.tier.strip().lower()
    if tier not in TIER_ENTITLEMENTS:
        raise HTTPException(400, f"tier must be one of: {', '.join(TIER_ENTITLEMENTS.keys())}")
    
    ent = _entitlements_for_tier(tier)
    
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            target = await conn.fetchrow("SELECT id, email FROM users WHERE lower(email)=lower($1)", payload.email)
            if not target:
                raise HTTPException(404, "User not found")
            
            is_lifetime = tier in ("lifetime", "friends_family")
            await conn.execute(
                """INSERT INTO entitlements(user_id, tier, upload_quota_override, is_lifetime, note, granted_by)
                   VALUES($1, $2, $3, $4, $5, $6)""",
                target["id"], tier, payload.upload_quota, is_lifetime, payload.note, admin["id"]
            )
            
            quota = payload.upload_quota if payload.upload_quota is not None else ent["upload_quota"]
            await conn.execute(
                """UPDATE users SET subscription_tier=$1, upload_quota=$2, unlimited_uploads=$3, updated_at=NOW() WHERE id=$4""",
                tier, quota, ent["unlimited_uploads"], target["id"]
            )
            
            await admin_audit_log(conn, admin, action="grant_entitlement", target_user_id=str(target["id"]), target_email=target["email"], meta={"tier": tier, "upload_quota_override": payload.upload_quota})
    
    await _discord_notify_admin("entitlement_granted", {"admin": admin["email"], "user": payload.email, "tier": tier})
    
    return {"status": "ok", "email": payload.email.lower(), "tier": tier}

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
                   FROM users WHERE lower(email) LIKE $1 OR lower(name) LIKE $1
                   ORDER BY created_at DESC LIMIT $2""",
                f"%{q}%", limit
            )
        else:
            rows = await conn.fetch(
                """SELECT id, email, name, role, subscription_tier, upload_quota, uploads_this_month, created_at
                   FROM users ORDER BY created_at DESC LIMIT $1""",
                limit
            )
    
    return {"users": [dict(r) for r in rows]}

@app.get("/api/admin/db-info")
async def admin_db_info(_: bool = Depends(require_admin_api_key)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        migrations = await conn.fetch("SELECT version, applied_at FROM schema_migrations ORDER BY version ASC")
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        )
    
    return {
        "database_configured": True,
        "migrations": [{"version": m["version"], "applied_at": m["applied_at"].isoformat()} for m in migrations],
        "tables": [t["table_name"] for t in tables],
    }

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
