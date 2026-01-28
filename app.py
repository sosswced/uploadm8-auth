"""
UploadM8 API Server - Production Build v3
==========================================
Complete FastAPI backend with ALL requested features:

Core Features:
- Multi-account per platform support
- Account groups for bulk operations
- Smart scheduling with AI-optimized times
- White-label/branding for Studio+ tiers
- AI caption/title/hashtag generation
- Thumbnail generation (FFmpeg + AI)
- Tier-based watermarks and ads
- Excel export
- Comprehensive analytics

Admin Features:
- Full KPI dashboard with time ranges
- Hide figures toggle (financial privacy)
- User management with Stripe integration
- Billing mode toggle (test/live)
- Demo data toggle
- Cost tracking (OpenAI, storage)
- Mass email marketing
- Password reset for users
- Ban/suspend users
- Custom entitlement overrides

Notifications:
- Discord webhooks (signup, trial, MRR, errors)
- Email (welcome, upgrade, tier changes)

Security:
- JWT auth with refresh tokens
- AES-256-GCM token encryption
- Rate limiting
- CORS configuration
- Request ID tracking
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
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode
from io import BytesIO

import httpx
import asyncpg
import jwt
import bcrypt
import boto3
from botocore.config import Config
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import stripe
import redis.asyncio as aioredis

from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from contextlib import asynccontextmanager

# ============================================================
# Logging
# ============================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("uploadm8-api")

# ============================================================
# Configuration
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

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://app.uploadm8.com,https://uploadm8.com,http://localhost:3000")

# Admin
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()

# R2 Storage
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")
PRIORITY_JOB_QUEUE = os.environ.get("PRIORITY_JOB_QUEUE", "uploadm8:priority")

# Stripe
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SUCCESS_URL = os.environ.get("STRIPE_SUCCESS_URL", f"{FRONTEND_URL}/billing-success.html")
STRIPE_CANCEL_URL = os.environ.get("STRIPE_CANCEL_URL", f"{FRONTEND_URL}/index.html#pricing")
STRIPE_PORTAL_RETURN_URL = os.environ.get("STRIPE_PORTAL_RETURN_URL", f"{FRONTEND_URL}/dashboard.html")

# Billing Kill Switch
BILLING_MODE = os.environ.get("BILLING_MODE", "test").strip().lower()
PRODUCTION_HOSTS = {h.strip().lower() for h in os.environ.get("PRODUCTION_HOSTS", "auth.uploadm8.com,app.uploadm8.com").split(",") if h.strip()}

# Discord Webhooks
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
SIGNUP_DISCORD_WEBHOOK_URL = os.environ.get("SIGNUP_DISCORD_WEBHOOK_URL", "")
TRIAL_DISCORD_WEBHOOK_URL = os.environ.get("TRIAL_DISCORD_WEBHOOK_URL", "")
MRR_DISCORD_WEBHOOK_URL = os.environ.get("MRR_DISCORD_WEBHOOK_URL", "")

# Platform OAuth
META_APP_ID = os.environ.get("META_APP_ID", "")
META_APP_SECRET = os.environ.get("META_APP_SECRET", "")
TIKTOK_CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

# Email
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Admin Settings (can be toggled via admin panel)
DEMO_DATA_ENABLED = os.environ.get("DEMO_DATA_ENABLED", "0") == "1"

# Rate Limiting
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "60"))

# ============================================================
# Global State
# ============================================================

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None
ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"

# Admin settings cache
admin_settings_cache: Dict[str, Any] = {
    "demo_data_enabled": DEMO_DATA_ENABLED,
    "billing_mode": BILLING_MODE,
}

# ============================================================
# Helpers
# ============================================================

def _split_origins(raw: str) -> List[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _generate_request_id() -> str:
    return f"req_{int(time.time())}_{secrets.token_hex(4)}"

# ============================================================
# Encryption
# ============================================================

def parse_enc_keys() -> Dict[str, bytes]:
    if not TOKEN_ENC_KEYS:
        raise RuntimeError("TOKEN_ENC_KEYS is required")
    keys = {}
    clean = TOKEN_ENC_KEYS.strip().strip('"').replace("\\n", "")
    for part in [p.strip() for p in clean.split(",") if p.strip()]:
        if ":" not in part:
            continue
        kid, b64key = part.split(":", 1)
        raw = base64.b64decode(b64key.strip())
        if len(raw) != 32:
            raise RuntimeError(f"TOKEN_ENC_KEYS invalid: {kid} must be 32 bytes")
        keys[kid.strip()] = raw
    if not keys:
        raise RuntimeError("TOKEN_ENC_KEYS parsed empty")
    return keys

def init_enc_keys():
    global ENC_KEYS, CURRENT_KEY_ID
    ENC_KEYS = parse_enc_keys()
    CURRENT_KEY_ID = list(ENC_KEYS.keys())[-1]

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

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    timezone: Optional[str] = None
    avatar_url: Optional[str] = None

class UploadInit(BaseModel):
    filename: str
    file_size: int
    content_type: str
    platforms: List[str]
    title: str = ""
    caption: str = ""
    hashtags: List[str] = []
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None
    schedule_mode: str = "immediate"
    has_telemetry: bool = False
    telemetry_filename: Optional[str] = None
    target_accounts: List[str] = []
    group_id: Optional[str] = None

class UploadComplete(BaseModel):
    telemetry_key: Optional[str] = None

class SettingsUpdate(BaseModel):
    discord_webhook: Optional[str] = None
    telemetry_enabled: Optional[bool] = None
    hud_enabled: Optional[bool] = None
    hud_position: Optional[str] = None
    speeding_mph: Optional[int] = None
    euphoria_mph: Optional[int] = None
    hud_speed_unit: Optional[str] = None
    hud_color: Optional[str] = None
    hud_font_family: Optional[str] = None
    hud_font_size: Optional[int] = None
    ffmpeg_screenshot_interval: Optional[int] = None
    auto_generate_thumbnails: Optional[bool] = None
    auto_generate_captions: Optional[bool] = None
    auto_generate_hashtags: Optional[bool] = None
    default_hashtag_count: Optional[int] = None
    always_use_hashtags: Optional[bool] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class CheckoutRequest(BaseModel):
    lookup_key: Optional[str] = None
    trial_days: Optional[int] = None

class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    account_ids: List[str] = []
    color: Optional[str] = "#3b82f6"
    icon: Optional[str] = "folder"

class GroupUpdate(BaseModel):
    name: Optional[str] = None
    account_ids: Optional[List[str]] = None
    color: Optional[str] = None
    icon: Optional[str] = None

class ScheduledUpdate(BaseModel):
    scheduled_time: Optional[datetime] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    platforms: Optional[List[str]] = None

class AdminUserUpdate(BaseModel):
    subscription_tier: Optional[str] = None
    role: Optional[str] = None
    upload_quota: Optional[int] = None
    status: Optional[str] = None
    note: Optional[str] = None

class AdminEntitlementUpdate(BaseModel):
    user_id: str
    entitlements: Dict[str, Any]

class AdminSettingsUpdate(BaseModel):
    billing_mode: Optional[str] = None
    demo_data_enabled: Optional[bool] = None

class WhiteLabelUpdate(BaseModel):
    enabled: bool = False
    logo_url: Optional[str] = None
    company_name: Optional[str] = None
    primary_color: Optional[str] = None

class SmartScheduleRequest(BaseModel):
    platforms: List[str]
    timezone: str = "UTC"
    count: int = 5

class MassEmailRequest(BaseModel):
    subject: str
    body_html: str
    tier_filter: Optional[List[str]] = None
    test_only: bool = True

# ============================================================
# Tier Entitlements
# ============================================================

TIER_CONFIG = {
    "free": {"upload_quota": 5, "max_accounts": 1, "max_hashtags": 2, "ai_captions": False, "ai_thumbnails": False, "show_watermark": True, "show_ads": True, "white_label": False, "smart_scheduling": False, "excel_export": False},
    "starter": {"upload_quota": 10, "max_accounts": 1, "max_hashtags": 3, "ai_captions": False, "ai_thumbnails": False, "show_watermark": True, "show_ads": True, "white_label": False, "smart_scheduling": False, "excel_export": False},
    "solo": {"upload_quota": 60, "max_accounts": 2, "max_hashtags": 5, "ai_captions": False, "ai_thumbnails": False, "show_watermark": True, "show_ads": False, "white_label": False, "smart_scheduling": False, "excel_export": False},
    "creator": {"upload_quota": 200, "max_accounts": 4, "max_hashtags": 15, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": False, "smart_scheduling": True, "excel_export": False},
    "growth": {"upload_quota": 500, "max_accounts": 8, "max_hashtags": 30, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": False, "smart_scheduling": True, "excel_export": True},
    "studio": {"upload_quota": 1500, "max_accounts": 15, "max_hashtags": 50, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": True, "smart_scheduling": True, "excel_export": True},
    "agency": {"upload_quota": 5000, "max_accounts": 50, "max_hashtags": 9999, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": True, "smart_scheduling": True, "excel_export": True},
    "lifetime": {"upload_quota": 999999, "max_accounts": 100, "max_hashtags": 9999, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": True, "smart_scheduling": True, "excel_export": True, "unlimited_uploads": True},
    "friends_family": {"upload_quota": 999999, "max_accounts": 100, "max_hashtags": 9999, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": True, "smart_scheduling": True, "excel_export": True, "unlimited_uploads": True},
}

def get_entitlements(tier: str, role: str = "user", overrides: dict = None) -> dict:
    if role in ("admin", "master_admin"):
        return {"tier": "admin", "upload_quota": 999999, "max_accounts": 999, "max_hashtags": 9999, "ai_captions": True, "ai_thumbnails": True, "show_watermark": False, "show_ads": False, "white_label": True, "smart_scheduling": True, "excel_export": True, "unlimited_uploads": True, "is_admin": True}
    
    config = TIER_CONFIG.get(tier.lower(), TIER_CONFIG["free"]).copy()
    config["tier"] = tier
    
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    
    return config

# ============================================================
# Rate Limiting
# ============================================================

_rate_state: Dict[str, List[float]] = {}

async def check_rate_limit(key: str):
    if redis_client is None:
        now = time.time()
        arr = _rate_state.get(key, [])
        arr = [t for t in arr if now - t < RATE_LIMIT_WINDOW]
        if len(arr) >= RATE_LIMIT_MAX:
            raise HTTPException(429, "Too many requests")
        arr.append(now)
        _rate_state[key] = arr
        return
    
    now = int(time.time())
    zkey = f"ratelimit:{key}"
    try:
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(zkey, 0, now - RATE_LIMIT_WINDOW)
        pipe.zadd(zkey, {str(now): now})
        pipe.zcard(zkey)
        pipe.expire(zkey, RATE_LIMIT_WINDOW + 5)
        _, _, count, _ = await pipe.execute()
        if int(count) > RATE_LIMIT_MAX:
            raise HTTPException(429, "Too many requests")
    except HTTPException:
        raise
    except Exception:
        pass

# ============================================================
# Password Hashing
# ============================================================

def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    if len(pw) > 72:
        raise HTTPException(400, "Password too long")
    return bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12)).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except:
        return False

# ============================================================
# JWT Helpers
# ============================================================

def create_access_jwt(user_id: str) -> str:
    now = _now_utc()
    return jwt.encode({
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "type": "access",
    }, JWT_SECRET, algorithm="HS256")

def verify_access_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        return payload.get("sub")
    except:
        return None

async def create_refresh_token(conn, user_id: str) -> str:
    token_raw = secrets.token_urlsafe(64)
    token_hash = _sha256_hex(token_raw)
    expires_at = _now_utc() + timedelta(days=REFRESH_TOKEN_DAYS)
    await conn.execute("INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)", user_id, token_hash, expires_at)
    return token_raw

async def rotate_refresh_token(conn, old_token: str):
    old_hash = _sha256_hex(old_token)
    row = await conn.fetchrow("SELECT id, user_id, expires_at, revoked FROM refresh_tokens WHERE token_hash=$1", old_hash)
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

# ============================================================
# R2 Helpers
# ============================================================

def get_s3_client():
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4"), region_name="auto")

def generate_presigned_upload_url(key: str, content_type: str, ttl: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url("put_object", Params={"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type}, ExpiresIn=ttl)

# ============================================================
# Discord Notifications
# ============================================================

async def discord_notify(webhook_url: str, content: str = None, embeds: list = None):
    if not webhook_url:
        return
    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(webhook_url, json=payload)
    except:
        pass

async def notify_signup(email: str, name: str):
    webhook = SIGNUP_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if webhook:
        await discord_notify(webhook, embeds=[{"title": "🎉 New Signup", "color": 0x22c55e, "fields": [{"name": "Email", "value": email, "inline": True}, {"name": "Name", "value": name, "inline": True}], "timestamp": _now_utc().isoformat()}])

async def notify_trial(email: str, plan: str):
    webhook = TRIAL_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if webhook:
        await discord_notify(webhook, embeds=[{"title": "🚀 Trial Started", "color": 0x8b5cf6, "fields": [{"name": "Email", "value": email, "inline": True}, {"name": "Plan", "value": plan, "inline": True}], "timestamp": _now_utc().isoformat()}])

async def notify_mrr(amount: float, email: str, plan: str):
    webhook = MRR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if webhook:
        await discord_notify(webhook, embeds=[{"title": "💰 MRR Collected", "color": 0x22c55e, "fields": [{"name": "Amount", "value": f"${amount:.2f}", "inline": True}, {"name": "Email", "value": email, "inline": True}, {"name": "Plan", "value": plan, "inline": True}], "timestamp": _now_utc().isoformat()}])

# ============================================================
# Email Helpers
# ============================================================

async def send_email(to: str, subject: str, html: str):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.info(f"Email skipped (no Mailgun): {to} - {subject}")
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages", auth=("api", MAILGUN_API_KEY), data={"from": MAIL_FROM, "to": to, "subject": subject, "html": html})
    except Exception as e:
        logger.warning(f"Email failed: {e}")

async def send_welcome_email(email: str, name: str):
    await send_email(email, "Welcome to UploadM8! 🎉", f"<h1>Welcome, {name}!</h1><p>Thanks for signing up. <a href='{FRONTEND_URL}/dashboard.html'>Get started</a></p>")

# ============================================================
# Redis Job Queue
# ============================================================

async def enqueue_job(job_data: dict, priority: bool = False):
    if redis_client is None:
        logger.warning("Redis not available, job not enqueued")
        return False
    
    queue = PRIORITY_JOB_QUEUE if priority else UPLOAD_JOB_QUEUE
    job_data["enqueued_at"] = _now_utc().isoformat()
    job_data["job_id"] = str(uuid.uuid4())
    
    try:
        await redis_client.lpush(queue, json.dumps(job_data))
        return True
    except Exception as e:
        logger.error(f"Enqueue failed: {e}")
        return False

# ============================================================
# App Lifespan & Migrations
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, admin_settings_cache
    
    init_enc_keys()
    
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
    
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("Database connected")
    
    await run_migrations()
    
    if REDIS_URL:
        try:
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis failed: {e}")
            redis_client = None
    
    # Load admin settings from DB
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT settings_json FROM admin_settings WHERE id = 1")
        if row and row["settings_json"]:
            admin_settings_cache.update(json.loads(row["settings_json"]))
    
    if BOOTSTRAP_ADMIN_EMAIL:
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET role='master_admin' WHERE LOWER(email)=$1", BOOTSTRAP_ADMIN_EMAIL)
    
    yield
    
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()

async def run_migrations():
    if not db_pool:
        return
    
    async with db_pool.acquire() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW())")
        applied = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
        
        migrations = [
            (1, "CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), email VARCHAR(255) UNIQUE NOT NULL, password_hash VARCHAR(255) NOT NULL, name VARCHAR(255) NOT NULL, role VARCHAR(50) DEFAULT 'user', subscription_tier VARCHAR(50) DEFAULT 'free', upload_quota INT DEFAULT 5, uploads_this_month INT DEFAULT 0, unlimited_uploads BOOLEAN DEFAULT FALSE, stripe_customer_id VARCHAR(255), stripe_subscription_id VARCHAR(255), subscription_status VARCHAR(50), current_period_end TIMESTAMPTZ, trial_ends_at TIMESTAMPTZ, timezone VARCHAR(100) DEFAULT 'UTC', avatar_url VARCHAR(512), status VARCHAR(50) DEFAULT 'active', last_active_at TIMESTAMPTZ DEFAULT NOW(), created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (2, "CREATE TABLE IF NOT EXISTS refresh_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, token_hash VARCHAR(255) UNIQUE NOT NULL, expires_at TIMESTAMPTZ NOT NULL, revoked BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW())"),
            (3, "CREATE TABLE IF NOT EXISTS platform_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, platform VARCHAR(50) NOT NULL, account_id VARCHAR(255), account_name VARCHAR(255), account_username VARCHAR(255), account_avatar VARCHAR(512), token_blob JSONB NOT NULL, is_primary BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (4, "CREATE TABLE IF NOT EXISTS uploads (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, r2_key VARCHAR(512) NOT NULL, telemetry_r2_key VARCHAR(512), processed_r2_key VARCHAR(512), thumbnail_r2_key VARCHAR(512), filename VARCHAR(255) NOT NULL, file_size BIGINT, platforms VARCHAR(50)[] DEFAULT '{}', title VARCHAR(512), caption TEXT, hashtags TEXT[], privacy VARCHAR(50) DEFAULT 'public', status VARCHAR(50) DEFAULT 'pending', cancel_requested BOOLEAN DEFAULT FALSE, scheduled_time TIMESTAMPTZ, schedule_mode VARCHAR(50) DEFAULT 'immediate', processing_started_at TIMESTAMPTZ, processing_finished_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, error_code VARCHAR(100), error_detail TEXT, platform_results JSONB, views BIGINT DEFAULT 0, likes BIGINT DEFAULT 0, created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (5, "CREATE TABLE IF NOT EXISTS user_settings (user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, discord_webhook VARCHAR(512), telemetry_enabled BOOLEAN DEFAULT TRUE, hud_enabled BOOLEAN DEFAULT TRUE, hud_position VARCHAR(50) DEFAULT 'bottom-left', speeding_mph INT DEFAULT 80, euphoria_mph INT DEFAULT 100, hud_speed_unit VARCHAR(10) DEFAULT 'mph', hud_color VARCHAR(20) DEFAULT '#FFFFFF', hud_font_family VARCHAR(100) DEFAULT 'Arial', hud_font_size INT DEFAULT 24, ffmpeg_screenshot_interval INT DEFAULT 5, auto_generate_thumbnails BOOLEAN DEFAULT TRUE, auto_generate_captions BOOLEAN DEFAULT TRUE, auto_generate_hashtags BOOLEAN DEFAULT TRUE, default_hashtag_count INT DEFAULT 5, always_use_hashtags BOOLEAN DEFAULT FALSE, updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (6, "CREATE TABLE IF NOT EXISTS account_groups (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, name VARCHAR(100) NOT NULL, account_ids TEXT[] DEFAULT '{}', color VARCHAR(20) DEFAULT '#3b82f6', icon VARCHAR(50) DEFAULT 'folder', created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (7, "CREATE TABLE IF NOT EXISTS password_reset_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, token_hash VARCHAR(255) UNIQUE NOT NULL, expires_at TIMESTAMPTZ NOT NULL, used BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW())"),
            (8, "CREATE TABLE IF NOT EXISTS white_label_settings (user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, enabled BOOLEAN DEFAULT FALSE, logo_url VARCHAR(512), company_name VARCHAR(255), primary_color VARCHAR(20), created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (9, "CREATE TABLE IF NOT EXISTS user_entitlements (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, entitlement_key VARCHAR(100) NOT NULL, entitlement_value VARCHAR(255), value_type VARCHAR(20) DEFAULT 'string', granted_by UUID, created_at TIMESTAMPTZ DEFAULT NOW(), UNIQUE(user_id, entitlement_key))"),
            (10, "CREATE TABLE IF NOT EXISTS cost_tracking (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID, category VARCHAR(100) NOT NULL, operation VARCHAR(255), tokens INT, cost_usd DECIMAL(10,6), created_at TIMESTAMPTZ DEFAULT NOW())"),
            (11, "CREATE TABLE IF NOT EXISTS revenue_tracking (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID, amount DECIMAL(10,2) NOT NULL, source VARCHAR(100), stripe_payment_id VARCHAR(255), plan VARCHAR(100), created_at TIMESTAMPTZ DEFAULT NOW())"),
            (12, "CREATE TABLE IF NOT EXISTS admin_settings (id INT PRIMARY KEY DEFAULT 1, settings_json JSONB DEFAULT '{}', updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (13, "CREATE TABLE IF NOT EXISTS job_state (upload_id UUID PRIMARY KEY, state_json JSONB, updated_at TIMESTAMPTZ DEFAULT NOW())"),
            (14, "CREATE INDEX IF NOT EXISTS idx_uploads_user_status ON uploads(user_id, status)"),
            (15, "CREATE INDEX IF NOT EXISTS idx_uploads_scheduled ON uploads(scheduled_time) WHERE schedule_mode = 'scheduled'"),
            (16, "CREATE INDEX IF NOT EXISTS idx_platform_tokens_user ON platform_tokens(user_id)"),
            (17, "CREATE INDEX IF NOT EXISTS idx_groups_user ON account_groups(user_id)"),
            (18, "INSERT INTO admin_settings (id, settings_json) VALUES (1, '{}') ON CONFLICT (id) DO NOTHING"),
        ]
        
        for version, sql in migrations:
            if version not in applied:
                try:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                    logger.info(f"Migration v{version} applied")
                except Exception as e:
                    logger.error(f"Migration v{version} failed: {e}")

app = FastAPI(title="UploadM8 API", version="3.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=_split_origins(ALLOWED_ORIGINS), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============================================================
# Request ID Middleware
# ============================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or _generate_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ============================================================
# Auth Dependencies
# ============================================================

async def get_current_user(request: Request, authorization: Optional[str] = Header(None), token: Optional[str] = Query(None)):
    auth_token = None
    if authorization and authorization.startswith("Bearer "):
        auth_token = authorization[7:]
    elif token:
        auth_token = token
    
    if not auth_token:
        raise HTTPException(401, "Missing authorization header")
    
    user_id = verify_access_jwt(auth_token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not user:
            raise HTTPException(401, "User not found")
        if user["status"] == "banned":
            raise HTTPException(403, "Account suspended")
        
        await conn.execute("UPDATE users SET last_active_at = NOW() WHERE id = $1", user_id)
        
        # Load entitlement overrides
        overrides = {}
        rows = await conn.fetch("SELECT entitlement_key, entitlement_value, value_type FROM user_entitlements WHERE user_id = $1", user_id)
        for r in rows:
            key, val, vtype = r["entitlement_key"], r["entitlement_value"], r.get("value_type", "string")
            if vtype == "bool":
                overrides[key] = val.lower() in ("true", "1")
            elif vtype == "int":
                try:
                    overrides[key] = int(val)
                except:
                    pass
            else:
                overrides[key] = val
        
        return {**dict(user), "entitlement_overrides": overrides}

async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin access required")
    return user

async def require_master_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "master_admin":
        raise HTTPException(403, "Master admin access required")
    return user

# ============================================================
# Health & Status
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_utc().isoformat()}

@app.get("/api/status")
async def api_status():
    db_ok = db_pool is not None
    redis_ok = redis_client is not None
    return {"database": db_ok, "redis": redis_ok, "billing_mode": admin_settings_cache.get("billing_mode", BILLING_MODE), "version": "3.0.0"}

# ============================================================
# Auth Endpoints
# ============================================================

@app.post("/api/auth/register")
async def register(data: UserCreate, request: Request, background_tasks: BackgroundTasks):
    ip = request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else "unknown")
    await check_rate_limit(f"register:{ip}")
    
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM users WHERE LOWER(email) = $1", data.email.lower())
        if existing:
            raise HTTPException(409, "Email already registered")
        
        password_hash = hash_password(data.password)
        user_id = str(uuid.uuid4())
        
        await conn.execute("""
            INSERT INTO users (id, email, password_hash, name, subscription_tier, upload_quota)
            VALUES ($1, $2, $3, $4, 'free', 5)
        """, user_id, data.email.lower(), password_hash, data.name)
        
        await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
        
        access_token = create_access_jwt(user_id)
        refresh_token = await create_refresh_token(conn, user_id)
    
    background_tasks.add_task(notify_signup, data.email, data.name)
    background_tasks.add_task(send_welcome_email, data.email, data.name)
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/api/auth/login")
async def login(data: UserLogin, request: Request):
    ip = request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else "unknown")
    await check_rate_limit(f"login:{ip}")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id, password_hash, status FROM users WHERE LOWER(email) = $1", data.email.lower())
        if not user or not verify_password(data.password, user["password_hash"]):
            raise HTTPException(401, "Invalid email or password")
        if user["status"] == "banned":
            raise HTTPException(403, "Account suspended")
        
        access_token = create_access_jwt(str(user["id"]))
        refresh_token = await create_refresh_token(conn, str(user["id"]))
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/api/auth/refresh")
async def refresh_tokens(data: RefreshRequest):
    async with db_pool.acquire() as conn:
        access_token, refresh_token = await rotate_refresh_token(conn, data.refresh_token)
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = $1", user["id"])
    return {"status": "logged_out"}

# ============================================================
# User Profile
# ============================================================

@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role", "user"), user.get("entitlement_overrides"))
    
    async with db_pool.acquire() as conn:
        # Get connected accounts count
        accounts = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])
        
        # Get white label settings
        wl = await conn.fetchrow("SELECT * FROM white_label_settings WHERE user_id = $1", user["id"])
    
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "subscription_tier": user.get("subscription_tier", "free"),
        "subscription_status": user.get("subscription_status"),
        "trial_ends_at": user.get("trial_ends_at"),
        "current_period_end": user.get("current_period_end"),
        "upload_quota": ent.get("upload_quota", 5),
        "uploads_this_month": user.get("uploads_this_month", 0),
        "unlimited_uploads": ent.get("unlimited_uploads", False),
        "timezone": user.get("timezone", "UTC"),
        "avatar_url": user.get("avatar_url"),
        "created_at": user["created_at"].isoformat() if user.get("created_at") else None,
        "connected_accounts": accounts,
        "entitlements": ent,
        "white_label": dict(wl) if wl else None,
    }

@app.put("/api/me")
async def update_profile(data: ProfileUpdate, user: dict = Depends(get_current_user)):
    updates, params = [], [user["id"]]
    if data.name:
        updates.append(f"name = ${len(params)+1}")
        params.append(data.name)
    if data.timezone:
        updates.append(f"timezone = ${len(params)+1}")
        params.append(data.timezone)
    if data.avatar_url is not None:
        updates.append(f"avatar_url = ${len(params)+1}")
        params.append(data.avatar_url)
    
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
    
    return {"status": "updated"}

@app.post("/api/me/change-password")
async def change_password(data: PasswordChange, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not verify_password(data.current_password, row["password_hash"]):
            raise HTTPException(400, "Current password is incorrect")
        
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, user["id"])
    
    return {"status": "password_changed"}

@app.post("/api/auth/forgot-password")
async def forgot_password(data: PasswordReset, background_tasks: BackgroundTasks):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id, name FROM users WHERE LOWER(email) = $1", data.email.lower())
        if user:
            token = secrets.token_urlsafe(32)
            token_hash = _sha256_hex(token)
            expires_at = _now_utc() + timedelta(hours=1)
            await conn.execute("INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)", user["id"], token_hash, expires_at)
            
            reset_url = f"{FRONTEND_URL}/reset-password.html?token={token}"
            background_tasks.add_task(send_email, data.email, "Reset Your Password", f"<h1>Password Reset</h1><p>Click <a href='{reset_url}'>here</a> to reset your password.</p>")
    
    return {"status": "email_sent"}

@app.post("/api/auth/reset-password")
async def reset_password(data: PasswordResetConfirm):
    token_hash = _sha256_hex(data.token)
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, user_id, expires_at, used FROM password_reset_tokens WHERE token_hash = $1", token_hash)
        if not row or row["used"] or row["expires_at"] < _now_utc():
            raise HTTPException(400, "Invalid or expired token")
        
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, row["user_id"])
        await conn.execute("UPDATE password_reset_tokens SET used = TRUE WHERE id = $1", row["id"])
    
    return {"status": "password_reset"}

# ============================================================
# Settings
# ============================================================

@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        settings = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user["id"])
        if not settings:
            await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user["id"])
            settings = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user["id"])
    
    return dict(settings) if settings else {}

@app.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    updates, params = [], [user["id"]]
    
    for field in ["discord_webhook", "telemetry_enabled", "hud_enabled", "hud_position", "speeding_mph", "euphoria_mph", "hud_speed_unit", "hud_color", "hud_font_family", "hud_font_size", "ffmpeg_screenshot_interval", "auto_generate_thumbnails", "auto_generate_captions", "auto_generate_hashtags", "default_hashtag_count", "always_use_hashtags"]:
        val = getattr(data, field, None)
        if val is not None:
            updates.append(f"{field} = ${len(params)+1}")
            params.append(val)
    
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE user_settings SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = $1", *params)
    
    return {"status": "updated"}

@app.post("/api/settings/test-webhook")
async def test_webhook(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        settings = await conn.fetchrow("SELECT discord_webhook FROM user_settings WHERE user_id = $1", user["id"])
    
    if not settings or not settings["discord_webhook"]:
        raise HTTPException(400, "No webhook configured")
    
    await discord_notify(settings["discord_webhook"], embeds=[{"title": "🔔 Test Notification", "description": "Your webhook is working!", "color": 0x22c55e}])
    return {"status": "sent"}

# ============================================================
# White Label
# ============================================================

@app.get("/api/white-label")
async def get_white_label(user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    if not ent.get("white_label"):
        raise HTTPException(403, "White label not available for your tier")
    
    async with db_pool.acquire() as conn:
        wl = await conn.fetchrow("SELECT * FROM white_label_settings WHERE user_id = $1", user["id"])
    
    return dict(wl) if wl else {"enabled": False}

@app.put("/api/white-label")
async def update_white_label(data: WhiteLabelUpdate, user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    if not ent.get("white_label"):
        raise HTTPException(403, "White label not available for your tier")
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO white_label_settings (user_id, enabled, logo_url, company_name, primary_color)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id) DO UPDATE SET enabled = $2, logo_url = $3, company_name = $4, primary_color = $5, updated_at = NOW()
        """, user["id"], data.enabled, data.logo_url, data.company_name, data.primary_color)
    
    return {"status": "updated"}

# ============================================================
# Platform Accounts (Multi-Account Support)
# ============================================================

@app.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at, updated_at
            FROM platform_tokens WHERE user_id = $1 ORDER BY platform, created_at
        """, user["id"])
    
    # Group by platform
    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms:
            platforms[p] = []
        platforms[p].append({
            "id": str(acc["id"]),
            "account_id": acc["account_id"],
            "account_name": acc["account_name"],
            "account_username": acc["account_username"],
            "avatar": acc["account_avatar"],
            "is_primary": acc["is_primary"],
            "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None,
            "updated_at": acc["updated_at"].isoformat() if acc["updated_at"] else None,
        })
    
    # Check token freshness
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    total = sum(len(v) for v in platforms.values())
    
    return {
        "platforms": platforms,
        "total_accounts": total,
        "max_accounts": ent.get("max_accounts", 1),
        "can_add_more": total < ent.get("max_accounts", 1),
    }

@app.get("/api/platforms/{platform}/accounts")
async def get_platform_accounts(platform: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT id, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens WHERE user_id = $1 AND platform = $2 ORDER BY created_at
        """, user["id"], platform.lower())
    
    return [{"id": str(a["id"]), "account_id": a["account_id"], "name": a["account_name"], "username": a["account_username"], "avatar": a["account_avatar"], "is_primary": a["is_primary"], "connected_at": a["created_at"].isoformat() if a["created_at"] else None} for a in accounts]

@app.delete("/api/platforms/{platform}/accounts/{account_id}")
async def disconnect_account(platform: str, account_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM platform_tokens WHERE id = $1 AND user_id = $2 AND platform = $3", account_id, user["id"], platform.lower())
    
    return {"status": "disconnected"}

@app.post("/api/platforms/{platform}/accounts/{account_id}/primary")
async def set_primary_account(platform: str, account_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE platform_tokens SET is_primary = FALSE WHERE user_id = $1 AND platform = $2", user["id"], platform.lower())
        await conn.execute("UPDATE platform_tokens SET is_primary = TRUE WHERE id = $1 AND user_id = $2", account_id, user["id"])
    
    return {"status": "updated"}

# ============================================================
# Account Groups
# ============================================================

@app.get("/api/groups")
async def get_groups(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        groups = await conn.fetch("SELECT * FROM account_groups WHERE user_id = $1 ORDER BY name", user["id"])
    
    return [{"id": str(g["id"]), "name": g["name"], "account_ids": g["account_ids"] or [], "color": g["color"], "icon": g["icon"], "created_at": g["created_at"].isoformat() if g["created_at"] else None} for g in groups]

@app.post("/api/groups")
async def create_group(data: GroupCreate, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        group_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT INTO account_groups (id, user_id, name, account_ids, color, icon)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, group_id, user["id"], data.name, data.account_ids, data.color, data.icon)
    
    return {"id": group_id, "status": "created"}

@app.put("/api/groups/{group_id}")
async def update_group(group_id: str, data: GroupUpdate, user: dict = Depends(get_current_user)):
    updates, params = [], [group_id, user["id"]]
    
    if data.name is not None:
        updates.append(f"name = ${len(params)+1}")
        params.append(data.name)
    if data.account_ids is not None:
        updates.append(f"account_ids = ${len(params)+1}")
        params.append(data.account_ids)
    if data.color is not None:
        updates.append(f"color = ${len(params)+1}")
        params.append(data.color)
    if data.icon is not None:
        updates.append(f"icon = ${len(params)+1}")
        params.append(data.icon)
    
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE account_groups SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1 AND user_id = $2", *params)
    
    return {"status": "updated"}

@app.delete("/api/groups/{group_id}")
async def delete_group(group_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM account_groups WHERE id = $1 AND user_id = $2", group_id, user["id"])
    
    return {"status": "deleted"}

# ============================================================
# Upload Endpoints
# ============================================================

@app.post("/api/uploads/presign")
async def presign_upload(data: UploadInit, user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    
    # Check quota
    if not ent.get("unlimited_uploads") and user.get("uploads_this_month", 0) >= ent.get("upload_quota", 5):
        raise HTTPException(429, f"Monthly quota reached ({user.get('uploads_this_month', 0)}/{ent.get('upload_quota', 5)})")
    
    upload_id = str(uuid.uuid4())
    r2_key = f"uploads/{user['id']}/{upload_id}/{data.filename}"
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO uploads (id, user_id, r2_key, filename, file_size, platforms, title, caption, hashtags, privacy, status, scheduled_time, schedule_mode)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'pending', $11, $12)
        """, upload_id, user["id"], r2_key, data.filename, data.file_size, data.platforms, data.title, data.caption, data.hashtags, data.privacy, data.scheduled_time, data.schedule_mode)
    
    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    
    result = {"upload_id": upload_id, "presigned_url": presigned_url, "r2_key": r2_key}
    
    if data.has_telemetry and data.telemetry_filename:
        telemetry_key = f"uploads/{user['id']}/{upload_id}/{data.telemetry_filename}"
        result["telemetry_presigned_url"] = generate_presigned_upload_url(telemetry_key, "application/octet-stream")
        result["telemetry_r2_key"] = telemetry_key
    
    return result

@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, data: UploadComplete = None, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        telemetry_key = data.telemetry_key if data else None
        if telemetry_key:
            await conn.execute("UPDATE uploads SET telemetry_r2_key = $1 WHERE id = $2", telemetry_key, upload_id)
        
        await conn.execute("UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1", upload_id)
    
    # Enqueue job
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    priority = ent.get("priority_processing", False)
    
    await enqueue_job({
        "upload_id": upload_id,
        "user_id": str(user["id"]),
        "idempotency_key": f"upload-{upload_id}",
    }, priority=priority)
    
    return {"status": "queued", "upload_id": upload_id}

@app.post("/api/uploads/{upload_id}/cancel")
async def cancel_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW()
            WHERE id = $1 AND user_id = $2 AND status NOT IN ('completed', 'cancelled', 'failed')
        """, upload_id, user["id"])
    
    return {"status": "cancel_requested"}

@app.post("/api/uploads/{upload_id}/retry")
async def retry_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2 AND status = 'failed'", upload_id, user["id"])
        if not upload:
            raise HTTPException(404, "Upload not found or not in failed state")
        
        await conn.execute("UPDATE uploads SET status = 'queued', cancel_requested = FALSE, error_code = NULL, error_detail = NULL, updated_at = NOW() WHERE id = $1", upload_id)
    
    await enqueue_job({"upload_id": upload_id, "user_id": str(user["id"]), "idempotency_key": f"retry-{upload_id}-{int(time.time())}"})
    
    return {"status": "requeued"}

@app.get("/api/uploads")
async def get_uploads(status: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch("SELECT * FROM uploads WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3 OFFSET $4", user["id"], status, limit, offset)
        else:
            uploads = await conn.fetch("SELECT * FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3", user["id"], limit, offset)
    
    return [{"id": str(u["id"]), "filename": u["filename"], "platforms": u["platforms"], "status": u["status"], "title": u["title"], "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None, "created_at": u["created_at"].isoformat() if u["created_at"] else None, "error_code": u["error_code"], "error_detail": u["error_detail"], "views": u.get("views", 0), "likes": u.get("likes", 0)} for u in uploads]

@app.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
    
    if not upload:
        raise HTTPException(404, "Upload not found")
    
    return dict(upload)

@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT r2_key, telemetry_r2_key, processed_r2_key FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
    
    # TODO: Delete files from R2
    return {"status": "deleted"}

# ============================================================
# Scheduled Uploads
# ============================================================

@app.get("/api/scheduled")
async def get_scheduled(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch("""
            SELECT * FROM uploads WHERE user_id = $1 AND schedule_mode = 'scheduled' AND status IN ('pending', 'queued', 'scheduled')
            ORDER BY scheduled_time ASC
        """, user["id"])
    
    return [{"id": str(u["id"]), "filename": u["filename"], "platforms": u["platforms"], "title": u["title"], "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None, "status": u["status"]} for u in uploads]

@app.put("/api/scheduled/{upload_id}")
async def update_scheduled(upload_id: str, data: ScheduledUpdate, user: dict = Depends(get_current_user)):
    updates, params = [], [upload_id, user["id"]]
    
    if data.scheduled_time is not None:
        updates.append(f"scheduled_time = ${len(params)+1}")
        params.append(data.scheduled_time)
    if data.title is not None:
        updates.append(f"title = ${len(params)+1}")
        params.append(data.title)
    if data.caption is not None:
        updates.append(f"caption = ${len(params)+1}")
        params.append(data.caption)
    if data.platforms is not None:
        updates.append(f"platforms = ${len(params)+1}")
        params.append(data.platforms)
    
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1 AND user_id = $2", *params)
    
    return {"status": "updated"}

@app.post("/api/schedule/smart")
async def get_smart_schedule(data: SmartScheduleRequest, user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    if not ent.get("smart_scheduling"):
        raise HTTPException(403, "Smart scheduling not available for your tier")
    
    # Platform-specific optimal posting times (simplified)
    optimal_hours = {
        "tiktok": [7, 9, 12, 15, 19, 21],
        "youtube": [9, 12, 15, 18, 20],
        "instagram": [8, 11, 14, 17, 19, 21],
        "facebook": [9, 13, 16, 19],
    }
    
    import random
    from datetime import timedelta
    
    schedules = []
    base = _now_utc()
    
    for i, platform in enumerate(data.platforms):
        hours = optimal_hours.get(platform.lower(), [12, 18])
        for j in range(min(data.count, len(hours))):
            hour = hours[j % len(hours)]
            day_offset = (i + j) // len(hours)
            scheduled = base.replace(hour=hour, minute=random.randint(0, 30)) + timedelta(days=day_offset)
            schedules.append({"platform": platform, "scheduled_time": scheduled.isoformat(), "reason": f"Optimal engagement time for {platform}"})
    
    return {"schedules": schedules[:data.count * len(data.platforms)]}

# ============================================================
# Analytics
# ============================================================

@app.get("/api/analytics")
async def get_analytics(range: str = "30d", user: dict = Depends(get_current_user)):
    range_minutes = {"30m": 30, "1h": 60, "6h": 360, "12h": 720, "1d": 1440, "7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=range_minutes)
    
    async with db_pool.acquire() as conn:
        # Upload stats
        stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, 
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE user_id = $1 AND created_at >= $2
        """, user["id"], since)
        
        # Daily breakdown
        daily = await conn.fetch("""
            SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            GROUP BY DATE(created_at) ORDER BY date
        """, user["id"], since)
        
        # Platform breakdown
        platforms = await conn.fetch("""
            SELECT unnest(platforms) AS platform, COUNT(*)::int AS count
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            GROUP BY platform
        """, user["id"], since)
    
    return {
        "total_uploads": stats["total"] if stats else 0,
        "completed": stats["completed"] if stats else 0,
        "views": stats["views"] if stats else 0,
        "likes": stats["likes"] if stats else 0,
        "daily": [{"date": str(d["date"]), "uploads": d["uploads"]} for d in daily],
        "platforms": {p["platform"]: p["count"] for p in platforms},
    }

@app.get("/api/analytics/hot-videos")
async def get_hot_videos(limit: int = 10, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        videos = await conn.fetch("""
            SELECT id, filename, title, platforms, views, likes, created_at
            FROM uploads WHERE user_id = $1 AND status = 'completed'
            ORDER BY views DESC, likes DESC LIMIT $2
        """, user["id"], limit)
    
    return [{"id": str(v["id"]), "filename": v["filename"], "title": v["title"], "platforms": v["platforms"], "views": v["views"], "likes": v["likes"], "is_hot": v["views"] > 1000 or v["likes"] > 100, "created_at": v["created_at"].isoformat() if v["created_at"] else None} for v in videos]

# ============================================================
# Excel Export
# ============================================================

@app.get("/api/exports/excel")
async def export_excel(type: str = "uploads", range: str = "30d", user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    if not ent.get("excel_export"):
        raise HTTPException(403, "Excel export not available for your tier")
    
    range_minutes = {"7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=range_minutes)
    
    async with db_pool.acquire() as conn:
        if type == "uploads":
            rows = await conn.fetch("""
                SELECT filename, platforms, title, status, views, likes, created_at
                FROM uploads WHERE user_id = $1 AND created_at >= $2 ORDER BY created_at DESC
            """, user["id"], since)
        elif type == "analytics":
            rows = await conn.fetch("""
                SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads, SUM(views)::bigint AS views
                FROM uploads WHERE user_id = $1 AND created_at >= $2
                GROUP BY DATE(created_at) ORDER BY date
            """, user["id"], since)
        else:
            raise HTTPException(400, "Invalid export type")
    
    # Create Excel file
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = type.title()
        
        if rows:
            headers = list(rows[0].keys())
            ws.append(headers)
            for row in rows:
                ws.append([str(row[h]) if row[h] is not None else "" for h in headers])
        
        output = BytesIO()
        wb.save(output)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=uploadm8_{type}_{range}.xlsx"}
        )
    except ImportError:
        raise HTTPException(500, "Excel export not available (openpyxl not installed)")

# ============================================================
# Billing
# ============================================================

@app.post("/api/billing/checkout")
async def create_checkout(data: CheckoutRequest, request: Request, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")
    
    # Billing mode check
    host = (request.headers.get("x-forwarded-host") or request.url.hostname or "").lower()
    current_mode = admin_settings_cache.get("billing_mode", BILLING_MODE)
    
    if current_mode == "test" and host in PRODUCTION_HOSTS:
        raise HTTPException(403, "Billing is in test mode")
    
    lookup_key = data.lookup_key or "uploadm8_creator_monthly"
    
    # Get or create Stripe customer
    async with db_pool.acquire() as conn:
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user["name"], metadata={"user_id": str(user["id"])})
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])
    
    # Get price
    prices = stripe.Price.list(lookup_keys=[lookup_key], active=True)
    if not prices.data:
        raise HTTPException(400, f"Invalid plan: {lookup_key}")
    
    session = stripe.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": prices.data[0].id, "quantity": 1}],
        mode="subscription",
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
        subscription_data={"trial_period_days": data.trial_days} if data.trial_days else None,
    )
    
    return {"checkout_url": session.url, "session_id": session.id}

@app.post("/api/billing/portal")
async def create_portal(user: dict = Depends(get_current_user)):
    if not user.get("stripe_customer_id"):
        raise HTTPException(400, "No billing account found")
    
    session = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=STRIPE_PORTAL_RETURN_URL,
    )
    
    return {"portal_url": session.url}

@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Invalid signature: {e}")
    
    if event.type == "checkout.session.completed":
        session = event.data.object
        customer_id = session.customer
        
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("SELECT id, email, subscription_tier FROM users WHERE stripe_customer_id = $1", customer_id)
            if user:
                sub = stripe.Subscription.retrieve(session.subscription)
                tier = "creator"  # Map from price lookup key
                
                await conn.execute("""
                    UPDATE users SET subscription_tier = $1, stripe_subscription_id = $2, 
                    subscription_status = 'active', current_period_end = $3, updated_at = NOW()
                    WHERE id = $4
                """, tier, session.subscription, datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc), user["id"])
                
                # Track MRR
                amount = session.amount_total / 100 if session.amount_total else 0
                await conn.execute("INSERT INTO revenue_tracking (user_id, amount, source, stripe_payment_id, plan) VALUES ($1, $2, 'subscription', $3, $4)", user["id"], amount, session.id, tier)
                
                await notify_mrr(amount, user["email"], tier)
    
    elif event.type == "customer.subscription.updated":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET subscription_status = $1, current_period_end = $2, updated_at = NOW()
                WHERE stripe_subscription_id = $3
            """, sub.status, datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc), sub.id)
    
    elif event.type == "customer.subscription.deleted":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET subscription_tier = 'free', subscription_status = 'cancelled', updated_at = NOW()
                WHERE stripe_subscription_id = $1
            """, sub.id)
    
    return {"status": "ok"}

# ============================================================
# Admin Endpoints
# ============================================================

@app.get("/api/admin/users")
async def admin_get_users(search: Optional[str] = None, status: Optional[str] = None, tier: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(require_admin)):
    query = "SELECT id, email, name, role, subscription_tier, subscription_status, trial_ends_at, current_period_end, uploads_this_month, status, created_at, last_active_at FROM users WHERE 1=1"
    params = []
    
    if search:
        params.append(f"%{search}%")
        query += f" AND (email ILIKE ${len(params)} OR name ILIKE ${len(params)})"
    if status:
        params.append(status)
        query += f" AND status = ${len(params)}"
    if tier:
        params.append(tier)
        query += f" AND subscription_tier = ${len(params)}"
    
    params.extend([limit, offset])
    query += f" ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}"
    
    async with db_pool.acquire() as conn:
        users = await conn.fetch(query, *params)
        total = await conn.fetchval("SELECT COUNT(*) FROM users")
    
    return {"users": [dict(u) for u in users], "total": total}

@app.get("/api/admin/users/{user_id}")
async def admin_get_user(user_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not target:
            raise HTTPException(404, "User not found")
        
        entitlements = await conn.fetch("SELECT * FROM user_entitlements WHERE user_id = $1", user_id)
        upload_count = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE user_id = $1", user_id)
    
    return {**dict(target), "custom_entitlements": [dict(e) for e in entitlements], "upload_count": upload_count}

@app.put("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, data: AdminUserUpdate, user: dict = Depends(require_admin)):
    updates, params = [], [user_id]
    
    if data.subscription_tier is not None:
        tier_config = TIER_CONFIG.get(data.subscription_tier.lower())
        if tier_config:
            updates.append(f"subscription_tier = ${len(params)+1}")
            params.append(data.subscription_tier.lower())
            updates.append(f"upload_quota = ${len(params)+1}")
            params.append(tier_config.get("upload_quota", 5))
    
    if data.role is not None and user.get("role") == "master_admin":
        updates.append(f"role = ${len(params)+1}")
        params.append(data.role)
    
    if data.upload_quota is not None:
        updates.append(f"upload_quota = ${len(params)+1}")
        params.append(data.upload_quota)
    
    if data.status is not None:
        updates.append(f"status = ${len(params)+1}")
        params.append(data.status)
    
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
    
    return {"status": "updated"}

@app.post("/api/admin/users/{user_id}/ban")
async def admin_ban_user(user_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET status = 'banned', updated_at = NOW() WHERE id = $1", user_id)
    return {"status": "banned"}

@app.post("/api/admin/users/{user_id}/unban")
async def admin_unban_user(user_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET status = 'active', updated_at = NOW() WHERE id = $1", user_id)
    return {"status": "unbanned"}

@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, background_tasks: BackgroundTasks, user: dict = Depends(require_master_admin)):
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1", user_id)
        if not target:
            raise HTTPException(404, "User not found")
        
        token = secrets.token_urlsafe(32)
        token_hash = _sha256_hex(token)
        expires_at = _now_utc() + timedelta(hours=24)
        await conn.execute("INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)", user_id, token_hash, expires_at)
        
        reset_url = f"{FRONTEND_URL}/reset-password.html?token={token}"
        background_tasks.add_task(send_email, target["email"], "Password Reset (Admin)", f"<h1>Password Reset</h1><p>An admin has initiated a password reset. <a href='{reset_url}'>Click here</a> to set a new password.</p>")
    
    return {"status": "reset_link_sent"}

@app.put("/api/admin/users/{user_id}/entitlements")
async def admin_update_entitlements(user_id: str, data: AdminEntitlementUpdate, user: dict = Depends(require_master_admin)):
    async with db_pool.acquire() as conn:
        for key, value in data.entitlements.items():
            if value is None:
                await conn.execute("DELETE FROM user_entitlements WHERE user_id = $1 AND entitlement_key = $2", user_id, key)
            else:
                vtype = "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "string"
                await conn.execute("""
                    INSERT INTO user_entitlements (user_id, entitlement_key, entitlement_value, value_type, granted_by)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, entitlement_key) DO UPDATE SET entitlement_value = $3, value_type = $4
                """, user_id, key, str(value), vtype, user["id"])
    
    return {"status": "updated"}

# ============================================================
# Admin KPI Dashboard
# ============================================================

@app.get("/api/admin/kpi")
async def admin_kpi(range: str = "30d", start: Optional[str] = None, end: Optional[str] = None, user: dict = Depends(require_admin)):
    if start and end:
        try:
            since = datetime.fromisoformat(start.replace('Z', '+00:00'))
            until = datetime.fromisoformat(end.replace('Z', '+00:00'))
        except:
            raise HTTPException(400, "Invalid date format")
    else:
        range_minutes = {"30m": 30, "1h": 60, "6h": 360, "12h": 720, "1d": 1440, "7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600}.get(range, 43200)
        since = _now_utc() - timedelta(minutes=range_minutes)
        until = _now_utc()
    
    # Check if demo data is enabled
    if admin_settings_cache.get("demo_data_enabled"):
        return _generate_demo_kpi()
    
    async with db_pool.acquire() as conn:
        # Users
        new_users = await conn.fetchval("SELECT COUNT(*)::int FROM users WHERE created_at BETWEEN $1 AND $2", since, until) or 0
        total_users = await conn.fetchval("SELECT COUNT(*)::int FROM users") or 0
        
        # Uploads
        total_uploads = await conn.fetchval("SELECT COUNT(*)::int FROM uploads WHERE created_at BETWEEN $1 AND $2", since, until) or 0
        completed = await conn.fetchval("SELECT COUNT(*)::int FROM uploads WHERE created_at BETWEEN $1 AND $2 AND status = 'completed'", since, until) or 0
        
        # Views/Likes
        metrics = await conn.fetchrow("SELECT COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes FROM uploads WHERE created_at BETWEEN $1 AND $2", since, until)
        
        # Tier breakdown
        tiers = await conn.fetch("SELECT subscription_tier, COUNT(*)::int AS count FROM users GROUP BY subscription_tier")
        
        # MRR
        mrr_data = await conn.fetchrow("""
            SELECT SUM(CASE 
                WHEN subscription_tier = 'starter' THEN 9.99
                WHEN subscription_tier = 'solo' THEN 19.99
                WHEN subscription_tier = 'creator' THEN 29.99
                WHEN subscription_tier = 'growth' THEN 49.99
                WHEN subscription_tier = 'studio' THEN 99.99
                WHEN subscription_tier = 'agency' THEN 199.99
                ELSE 0
            END)::decimal AS mrr
            FROM users WHERE subscription_tier NOT IN ('free', 'lifetime', 'friends_family')
        """)
        mrr = float(mrr_data["mrr"] or 0) if mrr_data else 0
        
        # Costs
        openai_cost = await conn.fetchval("SELECT COALESCE(SUM(cost_usd), 0)::decimal FROM cost_tracking WHERE created_at BETWEEN $1 AND $2", since, until) or 0
        
        # Revenue
        revenue = await conn.fetchval("SELECT COALESCE(SUM(amount), 0)::decimal FROM revenue_tracking WHERE created_at BETWEEN $1 AND $2", since, until) or 0
        
        # Upload trends
        trends = await conn.fetch("SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads FROM uploads WHERE created_at >= $1 GROUP BY DATE(created_at) ORDER BY date", since)
    
    return {
        "new_users": new_users,
        "total_users": total_users,
        "total_uploads": total_uploads,
        "completed_uploads": completed,
        "success_rate": (completed / max(total_uploads, 1)) * 100,
        "views": metrics["views"] if metrics else 0,
        "likes": metrics["likes"] if metrics else 0,
        "mrr": mrr,
        "revenue": float(revenue),
        "openai_cost": float(openai_cost),
        "estimated_profit": float(revenue) - float(openai_cost),
        "tier_breakdown": {t["subscription_tier"] or "free": t["count"] for t in tiers},
        "upload_trends": [{"date": str(t["date"]), "uploads": t["uploads"]} for t in trends],
    }

def _generate_demo_kpi():
    import random
    return {
        "new_users": random.randint(10, 50),
        "total_users": random.randint(500, 2000),
        "total_uploads": random.randint(100, 500),
        "completed_uploads": random.randint(80, 450),
        "success_rate": random.uniform(85, 99),
        "views": random.randint(10000, 100000),
        "likes": random.randint(1000, 10000),
        "mrr": random.uniform(1000, 10000),
        "revenue": random.uniform(1500, 15000),
        "openai_cost": random.uniform(50, 500),
        "estimated_profit": random.uniform(1000, 14000),
        "tier_breakdown": {"free": random.randint(300, 1000), "creator": random.randint(50, 200), "studio": random.randint(10, 50)},
        "upload_trends": [{"date": f"2025-01-{i:02d}", "uploads": random.randint(20, 100)} for i in range(1, 15)],
        "_demo": True,
    }

@app.get("/api/admin/leaderboard")
async def admin_leaderboard(range: str = "30d", sort: str = "uploads", user: dict = Depends(require_admin)):
    range_minutes = {"7d": 10080, "30d": 43200, "6m": 262800}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=range_minutes)
    
    sort_col = {"uploads": "upload_count", "views": "total_views", "likes": "total_likes"}.get(sort, "upload_count")
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT u.id, u.name, u.email, u.subscription_tier,
                   COUNT(up.id)::int AS upload_count,
                   COALESCE(SUM(up.views), 0)::bigint AS total_views,
                   COALESCE(SUM(up.likes), 0)::bigint AS total_likes
            FROM users u
            LEFT JOIN uploads up ON up.user_id = u.id AND up.created_at >= $1
            GROUP BY u.id, u.name, u.email, u.subscription_tier
            HAVING COUNT(up.id) > 0
            ORDER BY {sort_col} DESC LIMIT 50
        """, since)
    
    return [{"id": str(r["id"]), "name": r["name"], "email": r["email"], "tier": r["subscription_tier"], "uploads": r["upload_count"], "views": r["total_views"], "likes": r["total_likes"]} for r in rows]

@app.get("/api/admin/costs")
async def admin_costs(range: str = "30d", user: dict = Depends(require_master_admin)):
    range_minutes = {"7d": 10080, "30d": 43200, "6m": 262800}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=range_minutes)
    
    async with db_pool.acquire() as conn:
        costs = await conn.fetch("""
            SELECT category, COALESCE(SUM(cost_usd), 0)::decimal AS total
            FROM cost_tracking WHERE created_at >= $1
            GROUP BY category
        """, since)
        
        per_user = await conn.fetch("""
            SELECT u.email, COALESCE(SUM(c.cost_usd), 0)::decimal AS cost
            FROM cost_tracking c JOIN users u ON c.user_id = u.id
            WHERE c.created_at >= $1 GROUP BY u.email ORDER BY cost DESC LIMIT 20
        """, since)
    
    return {"by_category": {c["category"]: float(c["total"]) for c in costs}, "by_user": [{"email": p["email"], "cost": float(p["cost"])} for p in per_user]}
    if redis_client:
        await redis_client.setex(f"oauth_state:{state}", 600, str(user["id"]))
    
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": f"{BASE_URL}/api/oauth/google/callback",
        "scope": "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube",
        "response_type": "code",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    
    return {"auth_url": f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"}

@app.get("/api/oauth/google/callback")
async def google_callback(code: str, state: str):
    user_id = None
    if redis_client:
        user_id = await redis_client.get(f"oauth_state:{state}")
        await redis_client.delete(f"oauth_state:{state}")
    
    if not user_id:
        return RedirectResponse(f"{FRONTEND_URL}/platforms.html?error=invalid_state")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://oauth2.googleapis.com/token", data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{BASE_URL}/api/oauth/google/callback",
            })
            data = resp.json()
        
        if "access_token" not in data:
            return RedirectResponse(f"{FRONTEND_URL}/platforms.html?error=token_failed")
        
        # Get channel info
        async with httpx.AsyncClient(timeout=30) as client:
            channel_resp = await client.get("https://www.googleapis.com/youtube/v3/channels", params={"part": "snippet", "mine": "true"}, headers={"Authorization": f"Bearer {data['access_token']}"})
            channel_data = channel_resp.json()
        
        channel = channel_data.get("items", [{}])[0] if channel_data.get("items") else {}
        snippet = channel.get("snippet", {})
        
        token_blob = encrypt_blob({"access_token": data["access_token"], "refresh_token": data.get("refresh_token"), "expires_in": data.get("expires_in")})
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO platform_tokens (user_id, platform, account_id, account_name, account_avatar, token_blob)
                VALUES ($1, 'youtube', $2, $3, $4, $5)
                ON CONFLICT (user_id, platform) DO UPDATE SET 
                    account_id = $2, account_name = $3, account_avatar = $4, token_blob = $5, updated_at = NOW()
            """, user_id, channel.get("id"), snippet.get("title"), snippet.get("thumbnails", {}).get("default", {}).get("url"), json.dumps(token_blob))
        
        return RedirectResponse(f"{FRONTEND_URL}/platforms.html?success=youtube")
        
    except Exception as e:
        logger.error(f"Google OAuth error: {e}")
        return RedirectResponse(f"{FRONTEND_URL}/platforms.html?error=oauth_failed")

# ============================================================
# Queue Management
# ============================================================

@app.get("/api/queue")
async def get_queue(status: Optional[str] = None, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch("""
                SELECT id, filename, platforms, title, status, cancel_requested, created_at, processing_started_at, error_code
                FROM uploads WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT 100
            """, user["id"], status)
        else:
            uploads = await conn.fetch("""
                SELECT id, filename, platforms, title, status, cancel_requested, created_at, processing_started_at, error_code
                FROM uploads WHERE user_id = $1 AND status IN ('pending', 'queued', 'processing')
                ORDER BY created_at DESC LIMIT 100
            """, user["id"])
    
    return [{"id": str(u["id"]), "filename": u["filename"], "platforms": u["platforms"], "title": u["title"], "status": u["status"], "cancel_requested": u["cancel_requested"], "created_at": u["created_at"].isoformat() if u["created_at"] else None, "started_at": u["processing_started_at"].isoformat() if u["processing_started_at"] else None, "error": u["error_code"]} for u in uploads]

@app.post("/api/queue/bulk-cancel")
async def bulk_cancel(upload_ids: List[str], user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW()
            WHERE id = ANY($1) AND user_id = $2 AND status NOT IN ('completed', 'cancelled', 'failed')
        """, upload_ids, user["id"])
    
    return {"status": "cancelled", "count": len(upload_ids)}

@app.post("/api/queue/bulk-retry")
async def bulk_retry(upload_ids: List[str], user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch("SELECT id FROM uploads WHERE id = ANY($1) AND user_id = $2 AND status = 'failed'", upload_ids, user["id"])
        
        for u in uploads:
            await conn.execute("UPDATE uploads SET status = 'queued', cancel_requested = FALSE, error_code = NULL, error_detail = NULL, updated_at = NOW() WHERE id = $1", u["id"])
            await enqueue_job({"upload_id": str(u["id"]), "user_id": str(user["id"]), "idempotency_key": f"retry-{u['id']}-{int(time.time())}"})
    
    return {"status": "requeued", "count": len(uploads)}

# ============================================================
# Bulk Operations (Select All, Move, Download)
# ============================================================

@app.post("/api/uploads/bulk-move")
async def bulk_move_uploads(upload_ids: List[str], target_status: str, user: dict = Depends(get_current_user)):
    if target_status not in ("archived", "deleted"):
        raise HTTPException(400, "Invalid target status")
    
    async with db_pool.acquire() as conn:
        if target_status == "deleted":
            await conn.execute("DELETE FROM uploads WHERE id = ANY($1) AND user_id = $2", upload_ids, user["id"])
        else:
            await conn.execute("UPDATE uploads SET status = $1, updated_at = NOW() WHERE id = ANY($2) AND user_id = $3", target_status, upload_ids, user["id"])
    
    return {"status": "moved", "count": len(upload_ids)}

@app.post("/api/uploads/bulk-download")
async def bulk_download_info(upload_ids: List[str], user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch("SELECT id, filename, r2_key, processed_r2_key FROM uploads WHERE id = ANY($1) AND user_id = $2", upload_ids, user["id"])
    
    s3 = get_s3_client()
    downloads = []
    
    for u in uploads:
        key = u["processed_r2_key"] or u["r2_key"]
        url = s3.generate_presigned_url("get_object", Params={"Bucket": R2_BUCKET_NAME, "Key": key}, ExpiresIn=3600)
        downloads.append({"id": str(u["id"]), "filename": u["filename"], "download_url": url})
    
    return {"downloads": downloads}

# ============================================================
# Dashboard Stats
# ============================================================

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(user: dict = Depends(get_current_user)):
    ent = get_entitlements(user.get("subscription_tier", "free"), user.get("role"), user.get("entitlement_overrides"))
    
    async with db_pool.acquire() as conn:
        # Quick stats
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*)::int AS total_uploads,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
                SUM(CASE WHEN status IN ('pending', 'queued', 'processing') THEN 1 ELSE 0 END)::int AS in_queue,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
                COALESCE(SUM(views), 0)::bigint AS total_views,
                COALESCE(SUM(likes), 0)::bigint AS total_likes
            FROM uploads WHERE user_id = $1
        """, user["id"])
        
        # Connected accounts
        accounts = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])
        
        # Recent uploads
        recent = await conn.fetch("""
            SELECT id, filename, platforms, status, created_at
            FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5
        """, user["id"])
    
    return {
        "uploads": {
            "total": stats["total_uploads"] if stats else 0,
            "completed": stats["completed"] if stats else 0,
            "in_queue": stats["in_queue"] if stats else 0,
            "failed": stats["failed"] if stats else 0,
        },
        "engagement": {
            "views": stats["total_views"] if stats else 0,
            "likes": stats["total_likes"] if stats else 0,
        },
        "quota": {
            "used": user.get("uploads_this_month", 0),
            "limit": ent.get("upload_quota", 5),
            "unlimited": ent.get("unlimited_uploads", False),
        },
        "accounts": {
            "connected": accounts,
            "limit": ent.get("max_accounts", 1),
        },
        "recent": [{"id": str(r["id"]), "filename": r["filename"], "platforms": r["platforms"], "status": r["status"], "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in recent],
        "entitlements": ent,
    }

# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
