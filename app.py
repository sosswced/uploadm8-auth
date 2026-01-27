"""
UploadM8 Auth Service (Commercial Production Build v2)
======================================================
FastAPI backend for UploadM8 SaaS - Multi-platform video upload service

CHANGES IN THIS VERSION:
- Fixed OAuth: Accepts token from query parameter OR header
- Billing kill switch: BILLING_MODE=test|live
- FastAPI Limiter with Redis
- Admin restore endpoint
- Fixed upload cancellation
- Cost tracking KPIs for admin
- Redis health endpoint
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
from urllib.parse import urlencode

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

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")

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

# Billing Kill Switch
BILLING_MODE = os.environ.get("BILLING_MODE", "test").strip().lower()
BILLING_LIVE_ALLOWED = os.environ.get("BILLING_LIVE_ALLOWED", "0") == "1"
PRODUCTION_HOSTS = {h.strip().lower() for h in os.environ.get("PRODUCTION_HOSTS", "auth.uploadm8.com,app.uploadm8.com").split(",") if h.strip()}

# Admin Discord
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

# Rate limiting
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "60"))

# ============================================================
# Global State
# ============================================================

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None
ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"

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
# Encryption Keys
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
    hud_speed_unit: Optional[str] = None
    hud_color: Optional[str] = None
    hud_font_family: Optional[str] = None
    hud_font_size: Optional[int] = None
    selected_page_id: Optional[str] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class CheckoutRequest(BaseModel):
    lookup_key: Optional[str] = None
    trial_days: Optional[int] = None

class PortalRequest(BaseModel):
    return_url: Optional[str] = None

class AdminGrantEntitlement(BaseModel):
    email: EmailStr
    tier: str = "lifetime"
    upload_quota: Optional[int] = None
    note: Optional[str] = None

class AdminSetRole(BaseModel):
    email: EmailStr
    role: str = "admin"

class AdminRestoreRequest(BaseModel):
    secret_key: str
    email: EmailStr

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    timezone: Optional[str] = None
    avatar_url: Optional[str] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

class PreferencesUpdate(BaseModel):
    auto_captions: Optional[bool] = None
    auto_thumbnails: Optional[bool] = None
    notification_uploads: Optional[bool] = None
    notification_weekly: Optional[bool] = None
    discord_webhook: Optional[str] = None
    timezone: Optional[str] = None
    theme: Optional[str] = None

class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    account_ids: List[str] = []
    color: Optional[str] = None
    icon: Optional[str] = None

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

# ============================================================
# Rate Limiting (Redis-backed with memory fallback)
# ============================================================

_rate_state: Dict[str, List[float]] = {}

def rate_limit_key(request: Request, bucket: str) -> str:
    ip = request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else "unknown")
    return f"{bucket}:{ip}"

def check_rate_limit_memory(key: str):
    now = time.time()
    arr = _rate_state.get(key, [])
    arr = [t for t in arr if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(arr) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Too many requests")
    arr.append(now)
    _rate_state[key] = arr

async def check_rate_limit(key: str):
    """Redis-backed sliding window rate limiter."""
    if redis_client is None:
        return check_rate_limit_memory(key)
    
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
        logger.warning(f"Redis rate limiter error: {e}")
        return check_rate_limit_memory(key)

# ============================================================
# Password Hashing
# ============================================================

def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    if len(pw) > 72:
        raise HTTPException(400, "Password too long (max 72 bytes)")
    hashed = bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

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
        "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)",
        user_id, token_hash, expires_at
    )
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

# ============================================================
# Redis Job Queue
# ============================================================

async def enqueue_job(job_data: dict, queue: str = None):
    if redis_client is None:
        logger.warning("Redis not configured; job not enqueued")
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
# Tier Entitlements
# ============================================================

def _tier_from_lookup_key(lookup_key: str) -> str:
    mapping = {
        "uploadm8_starter_monthly": "starter",
        "uploadm8_solo_monthly": "solo",
        "uploadm8_creator_monthly": "creator",
        "uploadm8_growth_monthly": "growth",
        "uploadm8_studio_monthly": "studio",
        "uploadm8_agency_monthly": "agency",
    }
    return mapping.get(lookup_key.strip().lower(), "starter")

def _entitlements_for_tier(tier: str) -> dict:
    """
    Complete tier configuration with ALL advertised features.
    
    Features per tier:
    - upload_quota: Monthly upload limit
    - max_accounts: Number of connected platform accounts
    - can_generate_captions: Trill-based auto captions
    - can_burn_hud: Speed HUD overlay on video
    - can_use_ai_captions: AI-powered caption generation
    - team_seats: Number of team members allowed
    - priority_processing: Priority job queue
    - history_days: Upload history retention
    """
    tiers = {
        "starter": {
            "tier": "starter",
            "upload_quota": 10,
            "unlimited_uploads": False,
            "max_accounts": 1,
            "can_generate_captions": False,
            "can_burn_hud": False,
            "can_use_ai_captions": False,
            "team_seats": 1,
            "priority_processing": False,
            "history_days": 7,
            "can_schedule": False,
            "can_use_templates": False,
            "can_export": False,
            "can_use_webhooks": False,
            "support_tier": "basic",
        },
        "solo": {
            "tier": "solo",
            "upload_quota": 60,
            "unlimited_uploads": False,
            "max_accounts": 2,
            "can_generate_captions": False,
            "can_burn_hud": True,
            "can_use_ai_captions": False,
            "team_seats": 1,
            "priority_processing": False,
            "history_days": 14,
            "can_schedule": True,
            "can_use_templates": False,
            "can_export": False,
            "can_use_webhooks": False,
            "support_tier": "basic",
        },
        "creator": {
            "tier": "creator",
            "upload_quota": 200,
            "unlimited_uploads": False,
            "max_accounts": 4,
            "can_generate_captions": True,
            "can_burn_hud": True,
            "can_use_ai_captions": False,
            "team_seats": 1,
            "priority_processing": True,
            "history_days": 30,
            "can_schedule": True,
            "can_use_templates": True,
            "can_export": False,
            "can_use_webhooks": False,
            "support_tier": "standard",
        },
        "growth": {
            "tier": "growth",
            "upload_quota": 500,
            "unlimited_uploads": False,
            "max_accounts": 8,
            "can_generate_captions": True,
            "can_burn_hud": True,
            "can_use_ai_captions": True,
            "team_seats": 1,
            "priority_processing": True,
            "history_days": 30,
            "can_schedule": True,
            "can_use_templates": True,
            "can_export": True,
            "can_use_webhooks": True,
            "support_tier": "standard",
        },
        "studio": {
            "tier": "studio",
            "upload_quota": 1500,
            "unlimited_uploads": False,
            "max_accounts": 15,
            "can_generate_captions": True,
            "can_burn_hud": True,
            "can_use_ai_captions": True,
            "team_seats": 3,
            "priority_processing": True,
            "history_days": 90,
            "can_schedule": True,
            "can_use_templates": True,
            "can_export": True,
            "can_use_webhooks": True,
            "support_tier": "priority",
        },
        "agency": {
            "tier": "agency",
            "upload_quota": 5000,
            "unlimited_uploads": False,
            "max_accounts": 40,
            "can_generate_captions": True,
            "can_burn_hud": True,
            "can_use_ai_captions": True,
            "team_seats": 10,
            "priority_processing": True,
            "history_days": 365,
            "can_schedule": True,
            "can_use_templates": True,
            "can_export": True,
            "can_use_webhooks": True,
            "support_tier": "sla",
        },
        "lifetime": {
            "tier": "lifetime",
            "upload_quota": 999999,
            "unlimited_uploads": True,
            "max_accounts": 100,
            "can_generate_captions": True,
            "can_burn_hud": True,
            "can_use_ai_captions": True,
            "team_seats": 5,
            "priority_processing": True,
            "history_days": 365,
            "can_schedule": True,
            "can_use_templates": True,
            "can_export": True,
            "can_use_webhooks": True,
            "support_tier": "priority",
        },
        "friends_family": {
            "tier": "friends_family",
            "upload_quota": 999999,
            "unlimited_uploads": True,
            "max_accounts": 100,
            "can_generate_captions": True,
            "can_burn_hud": True,
            "can_use_ai_captions": True,
            "team_seats": 5,
            "priority_processing": True,
            "history_days": 365,
            "can_schedule": True,
            "can_use_templates": True,
            "can_export": True,
            "can_use_webhooks": True,
            "support_tier": "priority",
        },
    }
    return tiers.get(tier.lower(), tiers["starter"])


async def check_can_connect_platform(user_id: str, platform: str) -> tuple:
    """
    Check if user can connect another platform based on tier's max_accounts.
    Returns (can_connect: bool, message: str)
    """
    if not db_pool:
        return True, "ok"
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT subscription_tier FROM users WHERE id = $1", user_id)
        if not user:
            return False, "User not found"
        
        tier = user["subscription_tier"] or "starter"
        ent = _entitlements_for_tier(tier)
        max_accounts = ent.get("max_accounts", 1)
        
        # Count current connections (excluding the platform being reconnected)
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND platform != $2",
            user_id, platform
        )
        
        if count >= max_accounts:
            return False, f"Account limit reached ({count}/{max_accounts}). Upgrade to connect more."
        
        return True, "ok"

# ============================================================
# Billing Kill Switch
# ============================================================

def enforce_live_billing_guard(request: Request, user: dict):
    """Block live checkout unless all conditions are met."""
    host = (request.headers.get("x-forwarded-host") or request.url.hostname or "").lower()
    is_prod_host = (host in PRODUCTION_HOSTS) if PRODUCTION_HOSTS else False
    is_admin = user.get("role") == "admin"
    
    if BILLING_MODE != "live":
        raise HTTPException(403, "Billing is in TEST mode. Live checkout blocked.")
    if not is_prod_host:
        raise HTTPException(403, f"Live checkout blocked: host '{host}' is not approved.")
    if not BILLING_LIVE_ALLOWED:
        raise HTTPException(403, "Live checkout blocked: BILLING_LIVE_ALLOWED is disabled.")
    # Note: In test mode, allow all users. In live mode, consider admin-only initially.

# ============================================================
# App Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client
    
    validate_env()
    init_enc_keys()
    
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
    
    # Database
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("Database connected")
    
    # Run migrations
    await run_migrations()
    
    # Redis
    if REDIS_URL:
        try:
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    
    # Bootstrap admin
    if BOOTSTRAP_ADMIN_EMAIL and db_pool:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET role='admin' WHERE LOWER(email)=$1 AND role!='admin'",
                BOOTSTRAP_ADMIN_EMAIL
            )
    
    yield
    
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title="UploadM8 API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_split_origins(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Migrations
# ============================================================

async def run_migrations():
    """Run database migrations."""
    if not db_pool:
        return
    
    async with db_pool.acquire() as conn:
        # Create migrations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        applied = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
        
        migrations = [
            (1, """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'user',
                    subscription_tier VARCHAR(50) DEFAULT 'starter',
                    upload_quota INT DEFAULT 10,
                    uploads_this_month INT DEFAULT 0,
                    unlimited_uploads BOOLEAN DEFAULT FALSE,
                    stripe_customer_id VARCHAR(255),
                    stripe_subscription_id VARCHAR(255),
                    subscription_status VARCHAR(50),
                    current_period_end TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (2, """
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR(255) UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    revoked BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (3, """
                CREATE TABLE IF NOT EXISTS platform_tokens (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    platform VARCHAR(50) NOT NULL,
                    token_blob JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, platform)
                )
            """),
            (4, """
                CREATE TABLE IF NOT EXISTS uploads (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    r2_key VARCHAR(512) NOT NULL,
                    telemetry_r2_key VARCHAR(512),
                    processed_r2_key VARCHAR(512),
                    filename VARCHAR(255) NOT NULL,
                    file_size BIGINT,
                    platforms VARCHAR(50)[] DEFAULT '{}',
                    title VARCHAR(512),
                    caption TEXT,
                    privacy VARCHAR(50) DEFAULT 'public',
                    status VARCHAR(50) DEFAULT 'pending',
                    trill_score INT,
                    scheduled_time TIMESTAMPTZ,
                    schedule_mode VARCHAR(50) DEFAULT 'immediate',
                    processing_started_at TIMESTAMPTZ,
                    processing_finished_at TIMESTAMPTZ,
                    error_code VARCHAR(100),
                    error_detail TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """),
            (5, """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    discord_webhook VARCHAR(512),
                    telemetry_enabled BOOLEAN DEFAULT TRUE,
                    hud_enabled BOOLEAN DEFAULT TRUE,
                    hud_position VARCHAR(50) DEFAULT 'bottom-left',
                    speeding_mph INT DEFAULT 80,
                    euphoria_mph INT DEFAULT 100,
                    hud_speed_unit VARCHAR(10) DEFAULT 'mph',
                    hud_color VARCHAR(20) DEFAULT '#FFFFFF',
                    hud_font_family VARCHAR(100) DEFAULT 'Arial',
                    hud_font_size INT DEFAULT 24,
                    selected_page_id VARCHAR(255),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (6, """
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (7, """
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR(255) UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    used BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (8, "CREATE INDEX IF NOT EXISTS idx_uploads_user_status ON uploads(user_id, status)"),
            (9, "CREATE INDEX IF NOT EXISTS idx_uploads_created ON uploads(created_at DESC)"),
            (10, """
                CREATE TABLE IF NOT EXISTS account_groups (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    name VARCHAR(100) NOT NULL,
                    account_ids TEXT[] DEFAULT '{}',
                    color VARCHAR(20) DEFAULT '#3b82f6',
                    icon VARCHAR(50) DEFAULT 'folder',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (11, """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR(255) NOT NULL,
                    device_info VARCHAR(512),
                    ip_address VARCHAR(45),
                    last_active_at TIMESTAMPTZ DEFAULT NOW(),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (12, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS timezone VARCHAR(100) DEFAULT 'UTC'
            """),
            (13, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(512)
            """),
            (14, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'
            """),
            (15, """
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_active_at TIMESTAMPTZ DEFAULT NOW()
            """),
            (16, """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    auto_captions BOOLEAN DEFAULT FALSE,
                    auto_thumbnails BOOLEAN DEFAULT FALSE,
                    notification_uploads BOOLEAN DEFAULT TRUE,
                    notification_weekly BOOLEAN DEFAULT TRUE,
                    theme VARCHAR(20) DEFAULT 'dark',
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """),
            (17, """
                ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS account_name VARCHAR(255)
            """),
            (18, """
                ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS account_username VARCHAR(255)
            """),
            (19, """
                ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS account_avatar VARCHAR(512)
            """),
            (20, "CREATE INDEX IF NOT EXISTS idx_groups_user ON account_groups(user_id)"),
            (21, "CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)"),
            (22, "CREATE INDEX IF NOT EXISTS idx_uploads_scheduled ON uploads(scheduled_time) WHERE schedule_mode = 'scheduled'"),
        ]
        
        for version, sql in migrations:
            if version not in applied:
                try:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                    logger.info(f"Applied migration v{version}")
                except Exception as e:
                    logger.error(f"Migration v{version} failed: {e}")

# ============================================================
# Auth Dependencies
# ============================================================

async def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None)
) -> dict:
    """Get current user from JWT token (header or query parameter)."""
    # Try header first
    jwt_token = None
    if authorization and authorization.startswith("Bearer "):
        jwt_token = authorization[7:]
    # Fall back to query parameter (for OAuth redirects)
    if not jwt_token and token:
        jwt_token = token
    
    if not jwt_token:
        raise HTTPException(401, "Missing or invalid Authorization header")
    
    user_id = verify_access_jwt(jwt_token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    
    if not user:
        raise HTTPException(401, "User not found")
    
    return dict(user)

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user

# ============================================================
# Health Endpoints
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_utc().isoformat()}

@app.get("/api/redis/health")
async def redis_health():
    """Check Redis connection health."""
    if not redis_client:
        return {"ok": False, "error": "Redis not configured"}
    try:
        pong = await redis_client.ping()
        return {"ok": bool(pong)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ============================================================
# Auth Routes
# ============================================================

@app.post("/api/auth/register")
async def register(request: Request, data: UserCreate, background_tasks: BackgroundTasks):
    await check_rate_limit(rate_limit_key(request, "register"))
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    email = data.email.lower()
    pw_hash = hash_password(data.password)
    
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", email)
        if existing:
            raise HTTPException(409, "Email already registered")
        
        user = await conn.fetchrow(
            """INSERT INTO users (email, password_hash, name) VALUES ($1, $2, $3) RETURNING *""",
            email, pw_hash, data.name
        )
        
        access = create_access_jwt(str(user["id"]))
        refresh = await create_refresh_token(conn, str(user["id"]))
    
    return {
        "access_token": access,
        "refresh_token": refresh,
        "user": {"id": str(user["id"]), "email": user["email"], "name": user["name"]}
    }

@app.post("/api/auth/login")
async def login(request: Request, data: UserLogin):
    await check_rate_limit(rate_limit_key(request, "login"))
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", data.email.lower())
        if not user or not verify_password(data.password, user["password_hash"]):
            raise HTTPException(401, "Invalid credentials")
        
        access = create_access_jwt(str(user["id"]))
        refresh = await create_refresh_token(conn, str(user["id"]))
    
    return {
        "access_token": access,
        "refresh_token": refresh,
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "subscription_tier": user["subscription_tier"],
            "upload_quota": user["upload_quota"],
            "uploads_this_month": user["uploads_this_month"],
            "unlimited_uploads": user["unlimited_uploads"]
        }
    }

@app.post("/api/auth/refresh")
async def refresh_token(data: RefreshRequest):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        access, refresh = await rotate_refresh_token(conn, data.refresh_token)
    
    return {"access_token": access, "refresh_token": refresh}

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    # Get full entitlements for user's tier
    ent = _entitlements_for_tier(user.get("subscription_tier", "starter"))
    
    # Count connected platforms
    connected_platforms = 0
    if db_pool:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM platform_tokens WHERE user_id = $1",
                str(user["id"])
            )
            connected_platforms = row["count"] if row else 0
    
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "subscription_tier": user["subscription_tier"],
        "upload_quota": user["upload_quota"],
        "uploads_this_month": user["uploads_this_month"],
        "unlimited_uploads": user["unlimited_uploads"],
        "stripe_customer_id": user.get("stripe_customer_id"),
        # Full entitlements
        "entitlements": {
            "tier": ent["tier"],
            "upload_quota": ent["upload_quota"],
            "unlimited_uploads": ent["unlimited_uploads"],
            "max_accounts": ent["max_accounts"],
            "connected_accounts": connected_platforms,
            "can_generate_captions": ent["can_generate_captions"],
            "can_burn_hud": ent["can_burn_hud"],
            "can_use_ai_captions": ent["can_use_ai_captions"],
            "team_seats": ent["team_seats"],
            "priority_processing": ent["priority_processing"],
            "history_days": ent["history_days"],
            "can_schedule": ent["can_schedule"],
            "can_use_templates": ent["can_use_templates"],
            "can_export": ent["can_export"],
            "can_use_webhooks": ent["can_use_webhooks"],
            "support_tier": ent["support_tier"],
        }
    }

@app.post("/api/auth/logout")
async def logout(data: RefreshRequest):
    if db_pool:
        async with db_pool.acquire() as conn:
            token_hash = _sha256_hex(data.refresh_token)
            await conn.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE token_hash=$1", token_hash)
    return {"status": "ok"}

# ============================================================
# Settings Routes
# ============================================================

@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", str(user["id"]))
    
    return {"settings": dict(row) if row else {}}

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
# Billing Routes
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
async def create_checkout_session(request: Request, data: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")
    
    # Kill switch for live billing (optional - remove for testing)
    # enforce_live_billing_guard(request, user)
    
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
            customer = stripe.Customer.create(
                email=user["email"],
                name=user.get("name"),
                metadata={"user_id": str(user["id"])}
            )
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
        raise HTTPException(500, f"Stripe checkout failed: {str(e)}")

@app.post("/api/billing/portal")
async def create_portal_session(data: PortalRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured")
    
    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(400, "No Stripe customer on file. Please subscribe first.")
    
    return_url = (data.return_url or STRIPE_PORTAL_RETURN_URL or FRONTEND_URL).strip()
    
    try:
        sess = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
        return {"url": sess.url}
    except Exception as e:
        logger.error(f"Stripe portal failed: {e}")
        raise HTTPException(500, f"Stripe portal failed: {str(e)}")

async def _apply_subscription_entitlements(user_id: str, subscription: dict, lookup_key: Optional[str]):
    tier = _tier_from_lookup_key(lookup_key or "")
    ent = _entitlements_for_tier(tier)
    status = subscription.get("status") if isinstance(subscription, dict) else None
    sub_id = subscription.get("id") if isinstance(subscription, dict) else None
    
    if db_pool:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE users SET
                       subscription_tier=$1,
                       upload_quota=$2,
                       unlimited_uploads=$3,
                       stripe_subscription_id=COALESCE($4, stripe_subscription_id),
                       subscription_status=COALESCE($5, subscription_status),
                       updated_at=NOW()
                   WHERE id=$6""",
                ent["tier"], ent["upload_quota"], ent["unlimited_uploads"],
                sub_id, status, user_id
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
        
        if etype in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
            await _apply_subscription_entitlements(user_id, obj, lookup_key)
            await _discord_notify_admin(etype, {"user_id": user_id, "status": obj.get("status")})
        
        return {"received": True}
    except Exception as e:
        logger.error(f"Stripe webhook handling failed: {e}")
        return Response(status_code=500)

# ============================================================
# Upload Routes
# ============================================================

@app.post("/api/uploads/presign")
async def create_presigned_upload(data: UploadInit, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    # Check quota
    if not user.get("unlimited_uploads"):
        if user["uploads_this_month"] >= user["upload_quota"]:
            raise HTTPException(403, "Monthly upload quota exceeded. Please upgrade your plan.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = secrets.token_hex(4)
    r2_key = f"uploads/{user_id}/{timestamp}_{file_hash}_{data.filename}"
    
    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    
    # Handle telemetry file
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
            
            if data and data.telemetry_key:
                await conn.execute("UPDATE uploads SET telemetry_r2_key = $1 WHERE id = $2", data.telemetry_key, upload_id)
            
            await conn.execute("UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1", upload_id)
            
            if not user.get("unlimited_uploads"):
                await conn.execute("UPDATE users SET uploads_this_month = uploads_this_month + 1, updated_at = NOW() WHERE id = $1", user_id)
    
    # Enqueue job
    job_data = {
        "type": "process_upload",
        "upload_id": upload_id,
        "user_id": user_id,
        "has_telemetry": bool(upload.get("telemetry_r2_key"))
    }
    await enqueue_job(job_data)
    
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
        
        # Allow cancellation of pending, queued, and processing
        if upload["status"] in ("completed", "failed", "cancelled"):
            raise HTTPException(400, f"Cannot cancel upload with status: {upload['status']}")
        
        await conn.execute(
            "UPDATE uploads SET status = 'cancelled', updated_at = NOW(), error_code = 'user_cancelled' WHERE id = $1",
            upload_id
        )
        
        # Refund quota if was counted
        if upload["status"] == "queued":
            await conn.execute(
                "UPDATE users SET uploads_this_month = GREATEST(uploads_this_month - 1, 0), updated_at = NOW() WHERE id = $1",
                user_id
            )
    
    return {"status": "cancelled", "upload_id": upload_id}

@app.get("/api/uploads")
async def list_uploads(limit: int = 20, offset: int = 0, status: Optional[str] = None, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    limit = max(1, min(limit, 100))
    
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch(
                """SELECT id, filename, platforms, title, status, trill_score, scheduled_time, created_at, completed_at
                   FROM uploads WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3 OFFSET $4""",
                user_id, status, limit, offset
            )
        else:
            uploads = await conn.fetch(
                """SELECT id, filename, platforms, title, status, trill_score, scheduled_time, created_at, completed_at
                   FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT $3 OFFSET $4""",
                user_id, limit, offset
            )
    
    return {"uploads": [dict(u) for u in uploads]}

# ============================================================
# Platform Routes
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
# OAuth Routes (Fixed to accept token from query param)
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
async def tiktok_oauth_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None)):
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
        if isinstance(token_data, dict) and token_data.get('expires_in'):
            token_data['_obtained_at'] = time.time()
            token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
    
    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=tiktok_token_failed")
    
    # Check max_accounts limit
    can_connect, msg = await check_can_connect_platform(user_id, "tiktok")
    if not can_connect:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=account_limit_reached")
    
    token_blob = encrypt_blob(token_data)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'tiktok', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )
    
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
async def google_oauth_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None)):
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
        if isinstance(token_data, dict) and token_data.get('expires_in'):
            token_data['_obtained_at'] = time.time()
            token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
    
    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=google_token_failed")
    
    # Check max_accounts limit
    can_connect, msg = await check_can_connect_platform(user_id, "google")
    if not can_connect:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=account_limit_reached")
    
    token_blob = encrypt_blob(token_data)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'google', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )
    
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
async def meta_oauth_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None)):
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
        if isinstance(token_data, dict) and token_data.get('expires_in'):
            token_data['_obtained_at'] = time.time()
            token_data['_expires_at'] = time.time() + float(token_data.get('expires_in') or 0) - 60
    
    if "error" in token_data:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=meta_token_failed")
    
    # Check max_accounts limit
    can_connect, msg = await check_can_connect_platform(user_id, "meta")
    if not can_connect:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?error=account_limit_reached")
    
    token_blob = encrypt_blob(token_data)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO platform_tokens (user_id, platform, token_blob)
               VALUES ($1, 'meta', $2)
               ON CONFLICT (user_id, platform) DO UPDATE SET token_blob = $2, updated_at = NOW()""",
            user_id, json.dumps(token_blob)
        )
    
    return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?success=meta_connected")

# ============================================================
# Admin Routes
# ============================================================

@app.get("/api/admin/overview")
async def admin_overview(user: dict = Depends(require_admin)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        total_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads")
        active_users_30d = await conn.fetchval(
            "SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at > NOW() - INTERVAL '30 days'"
        )
        uploads_30d = await conn.fetchval(
            "SELECT COUNT(*) FROM uploads WHERE created_at > NOW() - INTERVAL '30 days'"
        )
        tier_breakdown = await conn.fetch(
            "SELECT subscription_tier, COUNT(*)::int as count FROM users GROUP BY subscription_tier ORDER BY count DESC"
        )
    
    return {
        "total_users": total_users,
        "total_uploads": total_uploads,
        "active_users_30d": active_users_30d,
        "uploads_30d": uploads_30d,
        "tier_breakdown": {r["subscription_tier"]: r["count"] for r in tier_breakdown},
    }

@app.get("/api/admin/users/search")
async def admin_search_users(q: str = "", limit: int = 20, user: dict = Depends(require_admin)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        if q:
            users = await conn.fetch(
                """SELECT id, email, name, role, subscription_tier, upload_quota, uploads_this_month, created_at
                   FROM users WHERE email ILIKE $1 OR name ILIKE $1 ORDER BY created_at DESC LIMIT $2""",
                f"%{q}%", limit
            )
        else:
            users = await conn.fetch(
                """SELECT id, email, name, role, subscription_tier, upload_quota, uploads_this_month, created_at
                   FROM users ORDER BY created_at DESC LIMIT $1""",
                limit
            )
    
    return {"users": [dict(u) for u in users]}

@app.post("/api/admin/users/role")
async def admin_set_role(data: AdminSetRole, user: dict = Depends(require_admin)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET role = $1, updated_at = NOW() WHERE LOWER(email) = $2",
            data.role, data.email.lower()
        )
    
    await _discord_notify_admin("admin_role_change", {"email": data.email, "role": data.role, "by": user["email"]})
    return {"status": "ok"}

@app.post("/api/admin/entitlements/grant")
async def admin_grant_entitlement(data: AdminGrantEntitlement, user: dict = Depends(require_admin)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    ent = _entitlements_for_tier(data.tier)
    quota = data.upload_quota or ent["upload_quota"]
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE users SET
                   subscription_tier = $1,
                   upload_quota = $2,
                   unlimited_uploads = $3,
                   updated_at = NOW()
               WHERE LOWER(email) = $4""",
            ent["tier"], quota, ent["unlimited_uploads"], data.email.lower()
        )
    
    await _discord_notify_admin("entitlement_granted", {"email": data.email, "tier": data.tier, "by": user["email"]})
    return {"status": "ok"}

@app.post("/api/admin/restore")
async def admin_restore(data: AdminRestoreRequest):
    """Emergency admin restore endpoint."""
    # This requires a secret key set in environment
    expected_key = os.environ.get("ADMIN_RESTORE_KEY", "")
    if not expected_key or data.secret_key != expected_key:
        raise HTTPException(403, "Invalid restore key")
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET role = 'admin', updated_at = NOW() WHERE LOWER(email) = $1",
            data.email.lower()
        )
    
    logger.warning(f"Admin restored via secret key: {data.email}")
    await _discord_notify_admin("admin_restored", {"email": data.email})
    return {"status": "ok", "message": f"Admin role restored for {data.email}"}

# ============================================================
# KPI Routes
# ============================================================

@app.get("/api/kpi/summary")
async def kpi_summary(days: int = 30, user: dict = Depends(get_current_user)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    days = max(1, min(days, 365))
    user_id = str(user["id"])
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        totals = await conn.fetchrow(
            """SELECT
                    COUNT(*)::int AS total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)::int AS completed,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END)::int AS failed,
                    SUM(CASE WHEN status IN ('queued','processing','pending') THEN 1 ELSE 0 END)::int AS backlog,
                    AVG(trill_score) AS avg_trill
               FROM uploads WHERE user_id = $1 AND created_at >= $2""",
            user_id, since
        )
        
        platform_mix = await conn.fetch(
            """SELECT p AS platform, COUNT(*)::int AS c
               FROM uploads, UNNEST(platforms) AS p
               WHERE user_id = $1 AND created_at >= $2
               GROUP BY p ORDER BY c DESC""",
            user_id, since
        )
    
    return {
        "window_days": days,
        "total": totals["total"] or 0,
        "completed": totals["completed"] or 0,
        "failed": totals["failed"] or 0,
        "backlog": totals["backlog"] or 0,
        "success_rate": round((totals["completed"] / totals["total"] * 100) if totals["total"] else 0, 1),
        "avg_trill_score": round(float(totals["avg_trill"] or 0), 1),
        "platform_mix": {r["platform"]: r["c"] for r in platform_mix},
    }

@app.get("/api/admin/kpi/global")
async def admin_kpi_global(days: int = 30, user: dict = Depends(require_admin)):
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    days = max(1, min(days, 365))
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        totals = await conn.fetchrow(
            """SELECT
                    COUNT(*)::int AS total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)::int AS completed,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END)::int AS failed,
                    COUNT(DISTINCT user_id)::int AS unique_users
               FROM uploads WHERE created_at >= $1""",
            since
        )
        
        tier_uploads = await conn.fetch(
            """SELECT u.subscription_tier, COUNT(up.id)::int as upload_count
               FROM users u LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1
               GROUP BY u.subscription_tier""",
            since
        )
    
    return {
        "window_days": days,
        "total_uploads": totals["total"] or 0,
        "completed_uploads": totals["completed"] or 0,
        "failed_uploads": totals["failed"] or 0,
        "unique_users": totals["unique_users"] or 0,
        "uploads_by_tier": {r["subscription_tier"]: r["upload_count"] for r in tier_uploads},
    }

# ============================================================
# Notification Test
# ============================================================

@app.post("/api/notifications/test")
async def test_notification(data: dict, user: dict = Depends(get_current_user)):
    webhook_url = data.get("discord_webhook")
    if not webhook_url:
        raise HTTPException(400, "discord_webhook required")
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                webhook_url,
                json={"content": "✅ UploadM8 webhook test: delivery confirmed."}
            )
            if resp.status_code not in (200, 204):
                raise HTTPException(400, f"Webhook returned {resp.status_code}")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Webhook test failed: {e}")


# ============================================================
# Profile & User Management
# ============================================================

@app.get("/api/me")
async def get_profile(user: dict = Depends(get_current_user)):
    """Get current user profile with entitlements."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    entitlements = _entitlements_for_tier(user.get("subscription_tier", "starter"))
    
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow("SELECT * FROM user_preferences WHERE user_id = $1", user_id)
        settings = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user_id)
        
        # Count connected accounts
        accounts = await conn.fetch("SELECT platform FROM platform_tokens WHERE user_id = $1", user_id)
    
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "role": user.get("role", "user"),
        "avatar_url": user.get("avatar_url"),
        "timezone": user.get("timezone", "UTC"),
        "created_at": user["created_at"].isoformat() if user.get("created_at") else None,
        "subscription": {
            "tier": user.get("subscription_tier", "starter"),
            "status": user.get("subscription_status", "active"),
            "uploads_used": user.get("uploads_this_month", 0),
            "uploads_limit": entitlements.get("upload_quota", 10),
            "unlimited": entitlements.get("unlimited_uploads", False),
            "accounts_used": len(accounts),
            "accounts_limit": entitlements.get("max_accounts", 1),
            "current_period_end": user.get("current_period_end").isoformat() if user.get("current_period_end") else None,
        },
        "entitlements": entitlements,
        "preferences": dict(prefs) if prefs else {
            "auto_captions": False,
            "auto_thumbnails": False,
            "notification_uploads": True,
            "notification_weekly": True,
            "theme": "dark",
        },
        "settings": {
            "discord_webhook": settings.get("discord_webhook") if settings else None,
            "hud_enabled": settings.get("hud_enabled", True) if settings else True,
            "hud_position": settings.get("hud_position", "bottom-left") if settings else "bottom-left",
        } if settings else {},
    }

@app.patch("/api/me")
async def update_profile(data: ProfileUpdate, user: dict = Depends(get_current_user)):
    """Update user profile."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    updates = []
    values = []
    idx = 1
    
    if data.name is not None:
        updates.append(f"name = ${idx}")
        values.append(data.name)
        idx += 1
    if data.timezone is not None:
        updates.append(f"timezone = ${idx}")
        values.append(data.timezone)
        idx += 1
    if data.avatar_url is not None:
        updates.append(f"avatar_url = ${idx}")
        values.append(data.avatar_url)
        idx += 1
    
    if not updates:
        return {"status": "ok", "message": "No changes"}
    
    updates.append("updated_at = NOW()")
    values.append(user_id)
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ${idx}",
            *values
        )
    
    return {"status": "ok"}

@app.post("/api/me/avatar")
async def upload_avatar(user: dict = Depends(get_current_user)):
    """Generate presigned URL for avatar upload."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    avatar_key = f"avatars/{user_id}/{secrets.token_urlsafe(8)}.jpg"
    
    upload_url = generate_presigned_upload_url(avatar_key, "image/jpeg", ttl=300)
    
    # Public URL for the avatar
    public_url = f"https://{R2_BUCKET_NAME}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{avatar_key}"
    
    return {
        "upload_url": upload_url,
        "avatar_key": avatar_key,
        "public_url": public_url,
    }

@app.delete("/api/me/avatar")
async def delete_avatar(user: dict = Depends(get_current_user)):
    """Remove avatar."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET avatar_url = NULL, updated_at = NOW() WHERE id = $1", user_id)
    
    return {"status": "ok"}

@app.put("/api/me/preferences")
async def update_preferences(data: PreferencesUpdate, user: dict = Depends(get_current_user)):
    """Update user preferences."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        # Upsert preferences
        await conn.execute("""
            INSERT INTO user_preferences (user_id, auto_captions, auto_thumbnails, 
                notification_uploads, notification_weekly, theme, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                auto_captions = COALESCE($2, user_preferences.auto_captions),
                auto_thumbnails = COALESCE($3, user_preferences.auto_thumbnails),
                notification_uploads = COALESCE($4, user_preferences.notification_uploads),
                notification_weekly = COALESCE($5, user_preferences.notification_weekly),
                theme = COALESCE($6, user_preferences.theme),
                updated_at = NOW()
        """, user_id, data.auto_captions, data.auto_thumbnails, 
            data.notification_uploads, data.notification_weekly, data.theme)
        
        # Update discord webhook in settings if provided
        if data.discord_webhook is not None:
            await conn.execute("""
                INSERT INTO user_settings (user_id, discord_webhook, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    discord_webhook = $2, updated_at = NOW()
            """, user_id, data.discord_webhook)
        
        # Update timezone on user if provided
        if data.timezone is not None:
            await conn.execute("UPDATE users SET timezone = $1, updated_at = NOW() WHERE id = $2",
                             data.timezone, user_id)
    
    return {"status": "ok"}

@app.put("/api/me/password")
async def change_password(data: PasswordChange, user: dict = Depends(get_current_user)):
    """Change password."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user_id)
        if not row or not verify_password(data.current_password, row["password_hash"]):
            raise HTTPException(400, "Current password is incorrect")
        
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2",
                          new_hash, user_id)
        
        # Revoke all refresh tokens except current
        await conn.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = $1", user_id)
    
    return {"status": "ok"}

@app.delete("/api/me")
async def delete_account(user: dict = Depends(get_current_user)):
    """Delete user account."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    user_email = user["email"]
    
    # Cancel Stripe subscription if exists
    if user.get("stripe_subscription_id") and stripe.api_key:
        try:
            stripe.Subscription.delete(user["stripe_subscription_id"])
        except Exception as e:
            logger.warning(f"Failed to cancel Stripe subscription: {e}")
    
    async with db_pool.acquire() as conn:
        # Delete user (cascades to related tables)
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)
    
    await _discord_notify_admin("user_deleted", {"email": user_email})
    return {"status": "ok"}

# ============================================================
# Session Management
# ============================================================

@app.get("/api/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """List active sessions."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        sessions = await conn.fetch("""
            SELECT id, device_info, ip_address, last_active_at, created_at
            FROM user_sessions WHERE user_id = $1
            ORDER BY last_active_at DESC
        """, user_id)
    
    return {
        "sessions": [
            {
                "id": str(s["id"]),
                "device": s["device_info"] or "Unknown device",
                "ip": s["ip_address"],
                "last_active": s["last_active_at"].isoformat() if s["last_active_at"] else None,
                "created_at": s["created_at"].isoformat() if s["created_at"] else None,
            }
            for s in sessions
        ]
    }

@app.post("/api/auth/logout-all")
async def logout_all_sessions(user: dict = Depends(get_current_user)):
    """Logout all sessions."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM user_sessions WHERE user_id = $1", user_id)
    
    return {"status": "ok"}

@app.delete("/api/sessions/{session_id}")
async def revoke_session(session_id: str, user: dict = Depends(get_current_user)):
    """Revoke a specific session."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_sessions WHERE id = $1 AND user_id = $2",
            session_id, user_id
        )
    
    return {"status": "ok"}

# ============================================================
# Password Reset
# ============================================================

@app.post("/api/auth/forgot-password")
async def forgot_password(data: PasswordReset, request: Request):
    """Send password reset email."""
    await check_rate_limit(rate_limit_key(request, "forgot"))
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id, name FROM users WHERE LOWER(email) = $1", data.email.lower())
        
        if user:
            # Generate reset token
            token_raw = secrets.token_urlsafe(32)
            token_hash = _sha256_hex(token_raw)
            expires_at = _now_utc() + timedelta(hours=1)
            
            await conn.execute("""
                INSERT INTO password_reset_tokens (user_id, token_hash, expires_at)
                VALUES ($1, $2, $3)
            """, user["id"], token_hash, expires_at)
            
            # Send email (if Mailgun configured)
            reset_url = f"{FRONTEND_URL}/reset-password.html?token={token_raw}"
            if MAILGUN_API_KEY and MAILGUN_DOMAIN:
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                            auth=("api", MAILGUN_API_KEY),
                            data={
                                "from": MAIL_FROM,
                                "to": data.email,
                                "subject": "Reset your UploadM8 password",
                                "html": f"""
                                    <h2>Password Reset</h2>
                                    <p>Hi {user['name']},</p>
                                    <p>Click the link below to reset your password:</p>
                                    <p><a href="{reset_url}">Reset Password</a></p>
                                    <p>This link expires in 1 hour.</p>
                                    <p>If you didn't request this, you can ignore this email.</p>
                                """,
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to send reset email: {e}")
    
    # Always return success (don't reveal if email exists)
    return {"status": "ok", "message": "If that email exists, you'll receive reset instructions."}

@app.post("/api/auth/reset-password")
async def reset_password(data: PasswordResetConfirm, request: Request):
    """Reset password with token."""
    await check_rate_limit(rate_limit_key(request, "reset"))
    
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    token_hash = _sha256_hex(data.token)
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, user_id, expires_at, used FROM password_reset_tokens
            WHERE token_hash = $1
        """, token_hash)
        
        if not row:
            raise HTTPException(400, "Invalid or expired reset token")
        if row["used"]:
            raise HTTPException(400, "This reset link has already been used")
        if row["expires_at"] < _now_utc():
            raise HTTPException(400, "This reset link has expired")
        
        # Update password
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2",
                          new_hash, row["user_id"])
        
        # Mark token as used
        await conn.execute("UPDATE password_reset_tokens SET used = TRUE WHERE id = $1", row["id"])
        
        # Revoke all refresh tokens
        await conn.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = $1", row["user_id"])
    
    return {"status": "ok"}

# ============================================================
# Upload Management (Extended)
# ============================================================

@app.get("/api/uploads/{upload_id}")
async def get_upload_detail(upload_id: str, user: dict = Depends(get_current_user)):
    """Get single upload details."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("""
            SELECT * FROM uploads WHERE id = $1 AND user_id = $2
        """, upload_id, user_id)
    
    if not upload:
        raise HTTPException(404, "Upload not found")
    
    return {
        "id": str(upload["id"]),
        "filename": upload["filename"],
        "file_size": upload["file_size"],
        "platforms": upload["platforms"],
        "title": upload["title"],
        "caption": upload["caption"],
        "privacy": upload["privacy"],
        "status": upload["status"],
        "trill_score": upload["trill_score"],
        "scheduled_time": upload["scheduled_time"].isoformat() if upload["scheduled_time"] else None,
        "schedule_mode": upload["schedule_mode"],
        "error_code": upload["error_code"],
        "error_detail": upload["error_detail"],
        "created_at": upload["created_at"].isoformat() if upload["created_at"] else None,
        "completed_at": upload["completed_at"].isoformat() if upload["completed_at"] else None,
    }

@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Delete an upload."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        # Get upload to delete R2 files
        upload = await conn.fetchrow(
            "SELECT r2_key, telemetry_r2_key, processed_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user_id
        )
        
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        # Delete from database
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        
        # Delete from R2 (background)
        try:
            s3 = get_s3_client()
            for key in [upload["r2_key"], upload["telemetry_r2_key"], upload["processed_r2_key"]]:
                if key:
                    try:
                        s3.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to delete R2 objects: {e}")
    
    return {"status": "ok"}

@app.post("/api/uploads/{upload_id}/retry")
async def retry_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Retry a failed upload."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("""
            SELECT * FROM uploads WHERE id = $1 AND user_id = $2 AND status = 'failed'
        """, upload_id, user_id)
        
        if not upload:
            raise HTTPException(404, "Failed upload not found")
        
        # Reset status to queued
        await conn.execute("""
            UPDATE uploads SET status = 'queued', error_code = NULL, error_detail = NULL, 
                updated_at = NOW() WHERE id = $1
        """, upload_id)
        
        # Re-enqueue job
        job_data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "r2_key": upload["r2_key"],
            "platforms": upload["platforms"],
            "title": upload["title"],
            "caption": upload["caption"],
            "privacy": upload["privacy"],
        }
        await enqueue_job(job_data)
    
    return {"status": "ok", "upload_id": upload_id}

# ============================================================
# Scheduled Uploads
# ============================================================

@app.get("/api/scheduled")
async def list_scheduled(
    start: Optional[str] = None,
    end: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """List scheduled uploads."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    query = """
        SELECT id, filename, title, caption, platforms, scheduled_time, status, created_at
        FROM uploads 
        WHERE user_id = $1 AND schedule_mode = 'scheduled'
    """
    params = [user_id]
    
    if start:
        try:
            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            query += f" AND scheduled_time >= ${len(params) + 1}"
            params.append(start_dt)
        except:
            pass
    
    if end:
        try:
            end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
            query += f" AND scheduled_time <= ${len(params) + 1}"
            params.append(end_dt)
        except:
            pass
    
    query += " ORDER BY scheduled_time ASC"
    
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch(query, *params)
    
    return {
        "scheduled": [
            {
                "id": str(u["id"]),
                "filename": u["filename"],
                "title": u["title"],
                "caption": u["caption"],
                "platforms": u["platforms"],
                "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None,
                "status": u["status"],
                "created_at": u["created_at"].isoformat() if u["created_at"] else None,
            }
            for u in uploads
        ]
    }

@app.put("/api/scheduled/{upload_id}")
async def update_scheduled(upload_id: str, data: ScheduledUpdate, user: dict = Depends(get_current_user)):
    """Update a scheduled upload."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    updates = []
    values = []
    idx = 1
    
    if data.scheduled_time is not None:
        updates.append(f"scheduled_time = ${idx}")
        values.append(data.scheduled_time)
        idx += 1
    if data.title is not None:
        updates.append(f"title = ${idx}")
        values.append(data.title)
        idx += 1
    if data.caption is not None:
        updates.append(f"caption = ${idx}")
        values.append(data.caption)
        idx += 1
    if data.platforms is not None:
        updates.append(f"platforms = ${idx}")
        values.append(data.platforms)
        idx += 1
    
    if not updates:
        return {"status": "ok", "message": "No changes"}
    
    updates.append("updated_at = NOW()")
    values.extend([upload_id, user_id])
    
    async with db_pool.acquire() as conn:
        result = await conn.execute(f"""
            UPDATE uploads SET {', '.join(updates)}
            WHERE id = ${idx} AND user_id = ${idx + 1} AND schedule_mode = 'scheduled'
        """, *values)
    
    return {"status": "ok"}

@app.delete("/api/scheduled/{upload_id}")
async def cancel_scheduled(upload_id: str, user: dict = Depends(get_current_user)):
    """Cancel a scheduled upload."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE uploads SET status = 'cancelled', schedule_mode = 'cancelled', updated_at = NOW()
            WHERE id = $1 AND user_id = $2 AND schedule_mode = 'scheduled' AND status IN ('pending', 'queued')
        """, upload_id, user_id)
    
    return {"status": "ok"}

# ============================================================
# Platform Accounts
# ============================================================

@app.get("/api/accounts")
async def list_accounts(user: dict = Depends(get_current_user)):
    """List connected platform accounts."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT id, platform, account_name, account_username, account_avatar, created_at, updated_at
            FROM platform_tokens WHERE user_id = $1
        """, user_id)
    
    return {
        "accounts": [
            {
                "id": str(a["id"]),
                "platform": a["platform"],
                "name": a["account_name"] or a["platform"].title(),
                "username": a["account_username"],
                "avatar": a["account_avatar"],
                "connected_at": a["created_at"].isoformat() if a["created_at"] else None,
                "last_refresh": a["updated_at"].isoformat() if a["updated_at"] else None,
            }
            for a in accounts
        ]
    }

@app.delete("/api/accounts/{account_id}")
async def disconnect_account(account_id: str, user: dict = Depends(get_current_user)):
    """Disconnect a platform account."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM platform_tokens WHERE id = $1 AND user_id = $2",
            account_id, user_id
        )
    
    return {"status": "ok"}

# ============================================================
# Account Groups
# ============================================================

@app.get("/api/groups")
async def list_groups(user: dict = Depends(get_current_user)):
    """List account groups."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        groups = await conn.fetch("""
            SELECT id, name, account_ids, color, icon, created_at
            FROM account_groups WHERE user_id = $1
            ORDER BY created_at ASC
        """, user_id)
    
    return {
        "groups": [
            {
                "id": str(g["id"]),
                "name": g["name"],
                "account_ids": g["account_ids"] or [],
                "color": g["color"],
                "icon": g["icon"],
                "created_at": g["created_at"].isoformat() if g["created_at"] else None,
            }
            for g in groups
        ]
    }

@app.post("/api/groups")
async def create_group(data: GroupCreate, user: dict = Depends(get_current_user)):
    """Create an account group."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        # Check group limit (max 20 groups)
        count = await conn.fetchval("SELECT COUNT(*) FROM account_groups WHERE user_id = $1", user_id)
        if count >= 20:
            raise HTTPException(400, "Maximum 20 groups allowed")
        
        group_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT INTO account_groups (id, user_id, name, account_ids, color, icon)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, group_id, user_id, data.name, data.account_ids, data.color or "#3b82f6", data.icon or "folder")
    
    return {"status": "ok", "id": group_id}

@app.put("/api/groups/{group_id}")
async def update_group(group_id: str, data: GroupUpdate, user: dict = Depends(get_current_user)):
    """Update an account group."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    updates = []
    values = []
    idx = 1
    
    if data.name is not None:
        updates.append(f"name = ${idx}")
        values.append(data.name)
        idx += 1
    if data.account_ids is not None:
        updates.append(f"account_ids = ${idx}")
        values.append(data.account_ids)
        idx += 1
    if data.color is not None:
        updates.append(f"color = ${idx}")
        values.append(data.color)
        idx += 1
    if data.icon is not None:
        updates.append(f"icon = ${idx}")
        values.append(data.icon)
        idx += 1
    
    if not updates:
        return {"status": "ok", "message": "No changes"}
    
    updates.append("updated_at = NOW()")
    values.extend([group_id, user_id])
    
    async with db_pool.acquire() as conn:
        await conn.execute(f"""
            UPDATE account_groups SET {', '.join(updates)}
            WHERE id = ${idx} AND user_id = ${idx + 1}
        """, *values)
    
    return {"status": "ok"}

@app.delete("/api/groups/{group_id}")
async def delete_group(group_id: str, user: dict = Depends(get_current_user)):
    """Delete an account group."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM account_groups WHERE id = $1 AND user_id = $2", group_id, user_id)
    
    return {"status": "ok"}

# ============================================================
# Analytics
# ============================================================

@app.get("/api/analytics")
async def get_analytics(days: int = 30, user: dict = Depends(get_current_user)):
    """Get analytics data."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    days = max(1, min(days, 365))
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        # Overall stats
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*)::int AS total_uploads,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
                AVG(CASE WHEN trill_score IS NOT NULL THEN trill_score ELSE NULL END) AS avg_trill
            FROM uploads WHERE user_id = $1 AND created_at >= $2
        """, user_id, since)
        
        # Daily breakdown
        daily = await conn.fetch("""
            SELECT DATE(created_at) AS date, COUNT(*)::int AS count,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """, user_id, since)
        
        # Platform breakdown
        platforms = await conn.fetch("""
            SELECT p AS platform, COUNT(*)::int AS count
            FROM uploads, UNNEST(platforms) AS p
            WHERE user_id = $1 AND created_at >= $2
            GROUP BY p ORDER BY count DESC
        """, user_id, since)
        
        # Status breakdown
        status_breakdown = await conn.fetch("""
            SELECT status, COUNT(*)::int AS count
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            GROUP BY status
        """, user_id, since)
    
    return {
        "period_days": days,
        "total_uploads": stats["total_uploads"] or 0,
        "completed": stats["completed"] or 0,
        "failed": stats["failed"] or 0,
        "success_rate": round((stats["completed"] / stats["total_uploads"] * 100) if stats["total_uploads"] else 0, 1),
        "avg_trill_score": round(float(stats["avg_trill"] or 0), 1),
        "daily": [
            {"date": str(d["date"]), "total": d["count"], "completed": d["completed"]}
            for d in daily
        ],
        "by_platform": {p["platform"]: p["count"] for p in platforms},
        "by_status": {s["status"]: s["count"] for s in status_breakdown},
    }

@app.get("/api/analytics/export")
async def export_analytics(days: int = 30, user: dict = Depends(get_current_user)):
    """Export analytics data as CSV."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    user_id = str(user["id"])
    tier = user.get("subscription_tier", "starter")
    entitlements = _entitlements_for_tier(tier)
    
    if not entitlements.get("can_export"):
        raise HTTPException(403, "Export requires Pro plan or higher")
    
    days = max(1, min(days, 365))
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch("""
            SELECT id, filename, title, platforms, status, trill_score, 
                created_at, completed_at
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            ORDER BY created_at DESC
        """, user_id, since)
    
    # Generate CSV
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Filename", "Title", "Platforms", "Status", "Trill Score", "Created", "Completed"])
    
    for u in uploads:
        writer.writerow([
            str(u["id"]),
            u["filename"],
            u["title"],
            ",".join(u["platforms"] or []),
            u["status"],
            u["trill_score"] or "",
            u["created_at"].isoformat() if u["created_at"] else "",
            u["completed_at"].isoformat() if u["completed_at"] else "",
        ])
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=uploadm8-analytics-{days}d.csv"}
    )

# ============================================================
# Billing
# ============================================================

@app.get("/api/billing")
async def get_billing(user: dict = Depends(get_current_user)):
    """Get billing information."""
    tier = user.get("subscription_tier", "starter")
    entitlements = _entitlements_for_tier(tier)
    
    billing_info = {
        "tier": tier,
        "status": user.get("subscription_status", "active"),
        "uploads_used": user.get("uploads_this_month", 0),
        "uploads_limit": entitlements.get("upload_quota", 10),
        "unlimited": entitlements.get("unlimited_uploads", False),
        "current_period_end": user.get("current_period_end").isoformat() if user.get("current_period_end") else None,
        "stripe_customer_id": user.get("stripe_customer_id"),
        "has_payment_method": False,
    }
    
    # Check Stripe for payment method
    if user.get("stripe_customer_id") and stripe.api_key:
        try:
            customer = stripe.Customer.retrieve(user["stripe_customer_id"])
            billing_info["has_payment_method"] = bool(customer.get("default_source") or customer.get("invoice_settings", {}).get("default_payment_method"))
        except Exception:
            pass
    
    return billing_info

# ============================================================
# Admin - Extended
# ============================================================

@app.get("/api/admin/stats")
async def admin_stats(user: dict = Depends(require_admin)):
    """Get admin overview stats."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        # User counts
        users = await conn.fetchrow("""
            SELECT 
                COUNT(*)::int AS total,
                SUM(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END)::int AS new_7d,
                SUM(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END)::int AS new_30d,
                SUM(CASE WHEN subscription_tier NOT IN ('starter', 'free') THEN 1 ELSE 0 END)::int AS paid
            FROM users
        """)
        
        # Upload counts
        uploads = await conn.fetchrow("""
            SELECT 
                COUNT(*)::int AS total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
                SUM(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 ELSE 0 END)::int AS last_24h
            FROM uploads
        """)
        
        # MRR calculation (approximate)
        mrr = await conn.fetchrow("""
            SELECT 
                SUM(CASE WHEN subscription_tier = 'starter' AND subscription_status = 'active' THEN 9 ELSE 0 END) +
                SUM(CASE WHEN subscription_tier = 'solo' AND subscription_status = 'active' THEN 19 ELSE 0 END) +
                SUM(CASE WHEN subscription_tier = 'creator' AND subscription_status = 'active' THEN 29 ELSE 0 END) +
                SUM(CASE WHEN subscription_tier = 'growth' AND subscription_status = 'active' THEN 49 ELSE 0 END) +
                SUM(CASE WHEN subscription_tier = 'studio' AND subscription_status = 'active' THEN 79 ELSE 0 END) +
                SUM(CASE WHEN subscription_tier = 'agency' AND subscription_status = 'active' THEN 99 ELSE 0 END)
                AS estimated_mrr
            FROM users
        """)
    
    return {
        "users": {
            "total": users["total"] or 0,
            "new_7d": users["new_7d"] or 0,
            "new_30d": users["new_30d"] or 0,
            "paid": users["paid"] or 0,
        },
        "uploads": {
            "total": uploads["total"] or 0,
            "completed": uploads["completed"] or 0,
            "failed": uploads["failed"] or 0,
            "last_24h": uploads["last_24h"] or 0,
        },
        "revenue": {
            "estimated_mrr": float(mrr["estimated_mrr"] or 0),
        },
    }

@app.get("/api/admin/users")
async def admin_list_users(
    search: Optional[str] = None,
    tier: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    user: dict = Depends(require_admin)
):
    """List users with filtering."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    page = max(1, page)
    limit = max(1, min(limit, 100))
    offset = (page - 1) * limit
    
    query = "SELECT * FROM users WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM users WHERE 1=1"
    params = []
    idx = 1
    
    if search:
        query += f" AND (LOWER(email) LIKE ${idx} OR LOWER(name) LIKE ${idx})"
        count_query += f" AND (LOWER(email) LIKE ${idx} OR LOWER(name) LIKE ${idx})"
        params.append(f"%{search.lower()}%")
        idx += 1
    
    if tier:
        query += f" AND subscription_tier = ${idx}"
        count_query += f" AND subscription_tier = ${idx}"
        params.append(tier)
        idx += 1
    
    if status:
        query += f" AND status = ${idx}"
        count_query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    
    query += f" ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx + 1}"
    params.extend([limit, offset])
    
    async with db_pool.acquire() as conn:
        users_list = await conn.fetch(query, *params[:-2] if len(params) > 2 else [], limit, offset)
        total = await conn.fetchval(count_query, *params[:-2] if len(params) > 2 else [])
        
        # Get upload counts
        user_ids = [str(u["id"]) for u in users_list]
        if user_ids:
            upload_counts = await conn.fetch("""
                SELECT user_id, COUNT(*)::int AS count, 
                    SUM(CASE WHEN created_at >= DATE_TRUNC('month', NOW()) THEN 1 ELSE 0 END)::int AS this_month
                FROM uploads WHERE user_id = ANY($1::uuid[])
                GROUP BY user_id
            """, user_ids)
            upload_map = {str(u["user_id"]): u for u in upload_counts}
        else:
            upload_map = {}
    
    return {
        "users": [
            {
                "id": str(u["id"]),
                "email": u["email"],
                "name": u["name"],
                "role": u.get("role", "user"),
                "tier": u.get("subscription_tier", "starter"),
                "status": u.get("status", "active"),
                "uploads_this_month": upload_map.get(str(u["id"]), {}).get("this_month", 0),
                "total_uploads": upload_map.get(str(u["id"]), {}).get("count", 0),
                "stripe_customer_id": u.get("stripe_customer_id"),
                "created_at": u["created_at"].isoformat() if u.get("created_at") else None,
            }
            for u in users_list
        ],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total or 0,
            "pages": (total or 0) // limit + (1 if (total or 0) % limit else 0),
        },
    }

@app.get("/api/admin/users/{target_user_id}")
async def admin_get_user(target_user_id: str, user: dict = Depends(require_admin)):
    """Get detailed user info."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT * FROM users WHERE id = $1", target_user_id)
        if not target:
            raise HTTPException(404, "User not found")
        
        # Get connected accounts
        accounts = await conn.fetch(
            "SELECT platform, account_name, account_username FROM platform_tokens WHERE user_id = $1",
            target_user_id
        )
        
        # Get upload stats
        stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total,
                SUM(CASE WHEN created_at >= DATE_TRUNC('month', NOW()) THEN 1 ELSE 0 END)::int AS this_month
            FROM uploads WHERE user_id = $1
        """, target_user_id)
    
    return {
        "id": str(target["id"]),
        "email": target["email"],
        "name": target["name"],
        "role": target.get("role", "user"),
        "tier": target.get("subscription_tier", "starter"),
        "status": target.get("status", "active"),
        "timezone": target.get("timezone", "UTC"),
        "avatar_url": target.get("avatar_url"),
        "stripe_customer_id": target.get("stripe_customer_id"),
        "stripe_subscription_id": target.get("stripe_subscription_id"),
        "subscription_status": target.get("subscription_status"),
        "uploads_this_month": stats["this_month"] or 0,
        "total_uploads": stats["total"] or 0,
        "upload_quota": target.get("upload_quota", 10),
        "connected_accounts": [
            {"platform": a["platform"], "name": a["account_name"], "username": a["account_username"]}
            for a in accounts
        ],
        "created_at": target["created_at"].isoformat() if target.get("created_at") else None,
        "updated_at": target["updated_at"].isoformat() if target.get("updated_at") else None,
    }

@app.patch("/api/admin/users/{target_user_id}")
async def admin_update_user(target_user_id: str, data: AdminUserUpdate, user: dict = Depends(require_admin)):
    """Update user (tier, status, quota)."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    updates = []
    values = []
    idx = 1
    
    if data.subscription_tier is not None:
        updates.append(f"subscription_tier = ${idx}")
        values.append(data.subscription_tier)
        # Update quota based on tier
        entitlements = _entitlements_for_tier(data.subscription_tier)
        updates.append(f"upload_quota = ${idx + 1}")
        values.append(entitlements.get("upload_quota", 10))
        updates.append(f"unlimited_uploads = ${idx + 2}")
        values.append(entitlements.get("unlimited_uploads", False))
        idx += 3
    
    if data.role is not None:
        updates.append(f"role = ${idx}")
        values.append(data.role)
        idx += 1
    
    if data.upload_quota is not None:
        updates.append(f"upload_quota = ${idx}")
        values.append(data.upload_quota)
        idx += 1
    
    if data.status is not None:
        updates.append(f"status = ${idx}")
        values.append(data.status)
        idx += 1
    
    if not updates:
        return {"status": "ok", "message": "No changes"}
    
    updates.append("updated_at = NOW()")
    values.append(target_user_id)
    
    async with db_pool.acquire() as conn:
        await conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ${idx}", *values)
    
    await _discord_notify_admin("admin_user_updated", {
        "target_id": target_user_id,
        "admin_email": user["email"],
        "changes": data.dict(exclude_none=True),
    })
    
    return {"status": "ok"}

@app.post("/api/admin/users/{target_user_id}/impersonate")
async def admin_impersonate(target_user_id: str, user: dict = Depends(require_admin)):
    """Generate impersonation token for a user."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT id, email FROM users WHERE id = $1", target_user_id)
        if not target:
            raise HTTPException(404, "User not found")
    
    # Create short-lived access token (1 hour)
    impersonate_token = create_access_jwt(target_user_id)
    
    await _discord_notify_admin("admin_impersonation", {
        "admin_email": user["email"],
        "target_email": target["email"],
    })
    
    return {
        "token": impersonate_token,
        "expires_in": ACCESS_TOKEN_MINUTES * 60,
    }

@app.get("/api/admin/kpis")
async def admin_kpis(range: str = "30d", user: dict = Depends(require_admin)):
    """Get KPI data for admin dashboard."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    # Parse range
    days_map = {"7d": 7, "30d": 30, "90d": 90, "12m": 365, "all": 3650}
    days = days_map.get(range, 30)
    since = _now_utc() - timedelta(days=days)
    
    async with db_pool.acquire() as conn:
        # User metrics
        user_metrics = await conn.fetchrow("""
            SELECT 
                COUNT(*)::int AS total_users,
                SUM(CASE WHEN last_active_at >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END)::int AS active_30d,
                SUM(CASE WHEN subscription_tier NOT IN ('starter', 'free') THEN 1 ELSE 0 END)::int AS paid,
                SUM(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END)::int AS new_7d
            FROM users
        """)
        
        # Monthly cohort data (simplified)
        cohorts = await conn.fetch("""
            SELECT DATE_TRUNC('month', created_at) AS cohort_month,
                COUNT(*)::int AS users
            FROM users
            WHERE created_at >= NOW() - INTERVAL '6 months'
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY cohort_month ASC
        """)
        
        # Upload trends
        upload_trends = await conn.fetch("""
            SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads
            FROM uploads WHERE created_at >= $1
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """, since)
    
    return {
        "users": {
            "total": user_metrics["total_users"] or 0,
            "active_30d": user_metrics["active_30d"] or 0,
            "paid": user_metrics["paid"] or 0,
            "new_7d": user_metrics["new_7d"] or 0,
        },
        "cohorts": [
            {"month": c["cohort_month"].strftime("%Y-%m"), "users": c["users"]}
            for c in cohorts
        ],
        "upload_trends": [
            {"date": str(t["date"]), "uploads": t["uploads"]}
            for t in upload_trends
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
