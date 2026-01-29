"""
UploadM8 API Server - Production Build v4
==========================================
Complete implementation with:
- PUT/AIC wallet system with ledger
- Announcements system
- Full KPI dashboards
- Weekly cost reports
- All Stripe integrations
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
from decimal import Decimal
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

from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from contextlib import asynccontextmanager

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("uploadm8-api")

# ============================================================
# Configuration
# ============================================================
DATABASE_URL = os.environ.get("DATABASE_URL")
BASE_URL = os.environ.get("BASE_URL", "https://auth.uploadm8.com")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.uploadm8.com")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "uploadm8-app")
ACCESS_TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.environ.get("REFRESH_TOKEN_DAYS", "30"))
TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://app.uploadm8.com,https://uploadm8.com,http://localhost:3000")
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()

# R2/S3
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

BILLING_MODE = os.environ.get("BILLING_MODE", "test").strip().lower()

# Discord Webhooks
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
SIGNUP_DISCORD_WEBHOOK_URL = os.environ.get("SIGNUP_DISCORD_WEBHOOK_URL", "")
MRR_DISCORD_WEBHOOK_URL = os.environ.get("MRR_DISCORD_WEBHOOK_URL", "")
COMMUNITY_DISCORD_WEBHOOK_URL = os.environ.get("COMMUNITY_DISCORD_WEBHOOK_URL", "")

# Email
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")

# Cost modeling
COST_PER_OPENAI_TOKEN = float(os.environ.get("COST_PER_OPENAI_TOKEN", "0.00001"))
COST_PER_GB_MONTH = float(os.environ.get("COST_PER_GB_MONTH", "0.015"))
COST_PER_COMPUTE_SECOND = float(os.environ.get("COST_PER_COMPUTE_SECOND", "0.0001"))

# Global State
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None
ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"
admin_settings_cache: Dict[str, Any] = {"demo_data_enabled": False, "billing_mode": BILLING_MODE}

# ============================================================
# Plan Configuration (PUT/AIC based)
# ============================================================
PLAN_CONFIG = {
    "free": {"name": "Free", "price": 0, "put_daily": 1, "put_monthly": 30, "aic_monthly": 0, "max_accounts": 1, "watermark": True, "ads": True, "ai": False, "scheduling": False, "webhooks": False, "white_label": False, "excel": False, "priority": False, "flex": False},
    "launch": {"name": "Launch", "price": 29, "put_daily": 5, "put_monthly": 600, "aic_monthly": 30, "max_accounts": 4, "watermark": True, "ads": False, "ai": True, "scheduling": False, "webhooks": False, "white_label": False, "excel": False, "priority": False, "flex": False},
    "creator_pro": {"name": "Creator Pro", "price": 59, "put_daily": 10, "put_monthly": 1200, "aic_monthly": 300, "max_accounts": 8, "watermark": False, "ads": False, "ai": True, "scheduling": True, "webhooks": True, "white_label": False, "excel": False, "priority": False, "flex": False},
    "studio": {"name": "Studio", "price": 99, "put_daily": 25, "put_monthly": 3000, "aic_monthly": 1000, "max_accounts": 20, "watermark": False, "ads": False, "ai": True, "scheduling": True, "webhooks": True, "white_label": True, "excel": True, "priority": False, "flex": False},
    "agency": {"name": "Agency", "price": 199, "put_daily": 75, "put_monthly": 9000, "aic_monthly": 3000, "max_accounts": 60, "watermark": False, "ads": False, "ai": True, "scheduling": True, "webhooks": True, "white_label": True, "excel": True, "priority": True, "flex": False},
    "master_admin": {"name": "Admin", "price": 0, "put_daily": 9999, "put_monthly": 999999, "aic_monthly": 999999, "max_accounts": 999, "watermark": False, "ads": False, "ai": True, "scheduling": True, "webhooks": True, "white_label": True, "excel": True, "priority": True, "flex": True, "internal": True},
    "friends_family": {"name": "Friends", "price": 0, "put_daily": 100, "put_monthly": 12000, "aic_monthly": 5000, "max_accounts": 80, "watermark": False, "ads": False, "ai": True, "scheduling": True, "webhooks": True, "white_label": True, "excel": True, "priority": True, "flex": True, "internal": True},
    "lifetime": {"name": "Lifetime", "price": 0, "put_daily": 100, "put_monthly": 12000, "aic_monthly": 5000, "max_accounts": 80, "watermark": False, "ads": False, "ai": True, "scheduling": True, "webhooks": True, "white_label": True, "excel": True, "priority": True, "flex": True, "internal": True},
}

STRIPE_LOOKUP_TO_TIER = {"uploadm8_launch_monthly": "launch", "uploadm8_creatorpro_monthly": "creator_pro", "uploadm8_studio_monthly": "studio", "uploadm8_agency_monthly": "agency"}
TOPUP_PRODUCTS = {
    "uploadm8_put_100": {"wallet": "put", "amount": 100, "price": 15},
    "uploadm8_put_500": {"wallet": "put", "amount": 500, "price": 59},
    "uploadm8_put_2000": {"wallet": "put", "amount": 2000, "price": 199},
    "uploadm8_aic_50": {"wallet": "aic", "amount": 50, "price": 12},
    "uploadm8_aic_250": {"wallet": "aic", "amount": 250, "price": 49},
    "uploadm8_aic_1000": {"wallet": "aic", "amount": 1000, "price": 149},
}

def get_plan(tier: str) -> dict:
    return PLAN_CONFIG.get(tier.lower(), PLAN_CONFIG["free"])

# ============================================================
# Helpers
# ============================================================
def _now_utc(): return datetime.now(timezone.utc)
def _sha256_hex(s: str): return hashlib.sha256(s.encode()).hexdigest()
def _req_id(): return f"req_{int(time.time())}_{secrets.token_hex(4)}"

def parse_enc_keys():
    if not TOKEN_ENC_KEYS:
        return {"v1": secrets.token_bytes(32)}
    keys = {}
    for part in TOKEN_ENC_KEYS.replace("\\n", "").split(","):
        if ":" in part:
            kid, b64 = part.split(":", 1)
            keys[kid.strip()] = base64.b64decode(b64.strip())
    return keys if keys else {"v1": secrets.token_bytes(32)}

def init_enc_keys():
    global ENC_KEYS, CURRENT_KEY_ID
    ENC_KEYS = parse_enc_keys()
    CURRENT_KEY_ID = list(ENC_KEYS.keys())[-1]

def encrypt_blob(data: dict) -> dict:
    key = ENC_KEYS[CURRENT_KEY_ID]
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    ct = aesgcm.encrypt(nonce, json.dumps(data).encode(), None)
    return {"kid": CURRENT_KEY_ID, "nonce": base64.b64encode(nonce).decode(), "ciphertext": base64.b64encode(ct).decode()}

def decrypt_blob(blob):
    if isinstance(blob, str): blob = json.loads(blob)
    key = ENC_KEYS.get(blob.get("kid", "v1"))
    if not key: raise ValueError("Unknown key")
    aesgcm = AESGCM(key)
    return json.loads(aesgcm.decrypt(base64.b64decode(blob["nonce"]), base64.b64decode(blob["ciphertext"]), None))

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()

def verify_password(pw: str, hashed: str) -> bool:
    try: return bcrypt.checkpw(pw.encode(), hashed.encode())
    except: return False

def create_access_jwt(user_id: str) -> str:
    now = _now_utc()
    return jwt.encode({"sub": user_id, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()), "iss": JWT_ISSUER, "aud": JWT_AUDIENCE}, JWT_SECRET, algorithm="HS256")

def verify_access_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except (jwt.InvalidAudienceError, jwt.InvalidIssuerError):
        # Fallback: try without strict aud/iss validation
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False, "verify_iss": False})
            return payload.get("sub")
        except:
            return None
    except Exception as e:
        logger.warning(f"JWT verification failed: {e}")
        return None
    except: return None

async def create_refresh_token(conn, user_id: str) -> str:
    token = secrets.token_urlsafe(64)
    await conn.execute("INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)", user_id, _sha256_hex(token), _now_utc() + timedelta(days=REFRESH_TOKEN_DAYS))
    return token

async def rotate_refresh_token(conn, old_token: str):
    h = _sha256_hex(old_token)
    row = await conn.fetchrow("SELECT id, user_id, expires_at, revoked FROM refresh_tokens WHERE token_hash=$1", h)
    if not row: raise HTTPException(401, "Invalid")
    if row["revoked"]:
        await conn.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE user_id=$1", row["user_id"])
        raise HTTPException(401, "Reuse detected")
    if row["expires_at"] < _now_utc(): raise HTTPException(401, "Expired")
    await conn.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE id=$1", row["id"])
    return create_access_jwt(row["user_id"]), await create_refresh_token(conn, row["user_id"])

# ============================================================
# Discord & Email Notifications
# ============================================================
async def discord_notify(webhook_url: str, content: str = None, embeds: list = None):
    if not webhook_url: return
    payload = {}
    if content: payload["content"] = content
    if embeds: payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(webhook_url, json=payload)
    except: pass

async def notify_signup(email: str, name: str):
    wh = SIGNUP_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        await discord_notify(wh, embeds=[{"title": "🎉 New Signup", "color": 0x22c55e, "fields": [{"name": "Email", "value": email}, {"name": "Name", "value": name}]}])

async def notify_mrr(amount: float, email: str, plan: str, event_type: str = "charge"):
    wh = MRR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        await discord_notify(wh, embeds=[{"title": f"💰 {event_type.title()}", "color": 0x22c55e, "fields": [{"name": "Amount", "value": f"${amount:.2f}"}, {"name": "Email", "value": email}, {"name": "Plan", "value": plan}]}])

async def notify_topup(amount: float, email: str, wallet: str, tokens: int):
    wh = MRR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        await discord_notify(wh, embeds=[{"title": "💳 Top-up Purchase", "color": 0x8b5cf6, "fields": [{"name": "Amount", "value": f"${amount:.2f}"}, {"name": "Wallet", "value": wallet.upper()}, {"name": "Tokens", "value": str(tokens)}, {"name": "Email", "value": email}]}])

async def notify_weekly_costs(openai_cost: float, storage_cost: float, compute_cost: float, revenue: float):
    wh = ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        margin = revenue - (openai_cost + storage_cost + compute_cost)
        await discord_notify(wh, embeds=[{"title": "📊 Weekly Cost Report", "color": 0x3b82f6, "fields": [
            {"name": "OpenAI", "value": f"${openai_cost:.2f}", "inline": True},
            {"name": "Storage", "value": f"${storage_cost:.2f}", "inline": True},
            {"name": "Compute", "value": f"${compute_cost:.2f}", "inline": True},
            {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
            {"name": "Est. Margin", "value": f"${margin:.2f}", "inline": True},
        ]}])

async def send_email(to: str, subject: str, html: str):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN: return
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages", auth=("api", MAILGUN_API_KEY), data={"from": MAIL_FROM, "to": to, "subject": subject, "html": html})
    except Exception as e:
        logger.warning(f"Email failed: {e}")

# ============================================================
# Wallet & Ledger Functions
# ============================================================
async def get_wallet(conn, user_id: str) -> dict:
    row = await conn.fetchrow("SELECT * FROM wallets WHERE user_id = $1", user_id)
    if not row:
        await conn.execute("INSERT INTO wallets (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
        row = await conn.fetchrow("SELECT * FROM wallets WHERE user_id = $1", user_id)
    return dict(row) if row else {"put_balance": 0, "aic_balance": 0, "put_reserved": 0, "aic_reserved": 0}

async def ledger_entry(conn, user_id: str, token_type: str, delta: int, reason: str, upload_id: str = None, stripe_event_id: str = None, platform: str = None, meta: dict = None):
    await conn.execute("""
        INSERT INTO token_ledger (user_id, token_type, platform, delta, reason, upload_id, stripe_event_id, meta)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """, user_id, token_type, platform, delta, reason, upload_id, stripe_event_id, json.dumps(meta) if meta else None)

async def reserve_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str) -> bool:
    wallet = await get_wallet(conn, user_id)
    available_put = wallet["put_balance"] - wallet["put_reserved"]
    available_aic = wallet["aic_balance"] - wallet["aic_reserved"]
    if available_put < put_count or available_aic < aic_count:
        return False
    await conn.execute("UPDATE wallets SET put_reserved = put_reserved + $1, aic_reserved = aic_reserved + $2 WHERE user_id = $3", put_count, aic_count, user_id)
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "reserve", upload_id)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "reserve", upload_id)
    return True

async def spend_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str, platforms: list = None):
    await conn.execute("UPDATE wallets SET put_balance = put_balance - $1, aic_balance = aic_balance - $2, put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3", put_count, aic_count, user_id)
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "spend", upload_id, platform=",".join(platforms) if platforms else None)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "spend", upload_id)

async def refund_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str):
    await conn.execute("UPDATE wallets SET put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3", put_count, aic_count, user_id)
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", put_count, "refund", upload_id)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", aic_count, "refund", upload_id)

async def credit_wallet(conn, user_id: str, wallet_type: str, amount: int, reason: str, stripe_event_id: str = None):
    if wallet_type == "put":
        await conn.execute("UPDATE wallets SET put_balance = put_balance + $1 WHERE user_id = $2", amount, user_id)
    else:
        await conn.execute("UPDATE wallets SET aic_balance = aic_balance + $1 WHERE user_id = $2", amount, user_id)
    await ledger_entry(conn, user_id, wallet_type, amount, reason, stripe_event_id=stripe_event_id)

async def transfer_tokens(conn, user_id: str, from_platform: str, to_platform: str, amount: int, burn_pct: float = 0.02) -> bool:
    # Check flex enabled
    user = await conn.fetchrow("SELECT subscription_tier, flex_enabled FROM users WHERE id = $1", user_id)
    if not user or not user.get("flex_enabled"):
        return False
    wallet = await get_wallet(conn, user_id)
    if wallet["put_balance"] - wallet["put_reserved"] < amount:
        return False
    burn = int(amount * burn_pct)
    net = amount - burn
    await ledger_entry(conn, user_id, "put", -amount, "transfer_out", platform=from_platform)
    await ledger_entry(conn, user_id, "put", net, "transfer_in", platform=to_platform)
    if burn > 0:
        await ledger_entry(conn, user_id, "put", -burn, "transfer_burn")
        await conn.execute("UPDATE wallets SET put_balance = put_balance - $1 WHERE user_id = $2", burn, user_id)
    return True

async def daily_refill(conn, user_id: str, tier: str):
    plan = get_plan(tier)
    daily = plan.get("put_daily", 1) * 4  # 4 platforms
    wallet = await get_wallet(conn, user_id)
    last_refill = wallet.get("last_refill_date")
    today = _now_utc().date()
    if last_refill and last_refill >= today:
        return
    monthly_cap = plan.get("put_monthly", 30)
    current = wallet["put_balance"]
    if current < monthly_cap:
        add = min(daily, monthly_cap - current)
        await conn.execute("UPDATE wallets SET put_balance = put_balance + $1, last_refill_date = $2 WHERE user_id = $3", add, today, user_id)
        await ledger_entry(conn, user_id, "put", add, "daily_refill")

# ============================================================
# R2 Storage
# ============================================================
def get_s3_client():
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4"), region_name="auto")

def generate_presigned_upload_url(key: str, content_type: str, ttl: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url("put_object", Params={"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type}, ExpiresIn=ttl)

# ============================================================
# Redis Queue
# ============================================================
async def enqueue_job(job_data: dict, priority: bool = False):
    if not redis_client: return False
    queue = PRIORITY_JOB_QUEUE if priority else UPLOAD_JOB_QUEUE
    job_data["enqueued_at"] = _now_utc().isoformat()
    job_data["job_id"] = str(uuid.uuid4())
    try:
        await redis_client.lpush(queue, json.dumps(job_data))
        return True
    except: return False


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

class RefreshRequest(BaseModel):
    refresh_token: str

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
    use_ai: bool = False

class SettingsUpdate(BaseModel):
    discord_webhook: Optional[str] = None
    telemetry_enabled: Optional[bool] = None
    hud_enabled: Optional[bool] = None
    hud_position: Optional[str] = None
    speeding_mph: Optional[int] = None
    euphoria_mph: Optional[int] = None

class CheckoutRequest(BaseModel):
    lookup_key: str
    kind: str = "subscription"  # subscription | topup | addon

class TransferRequest(BaseModel):
    from_platform: str
    to_platform: str
    amount: int

class AnnouncementRequest(BaseModel):
    title: str
    body: str
    send_email: bool = True
    send_discord_community: bool = True
    send_user_webhooks: bool = False
    target: str = "all"  # all | paid | trial | free | specific_tiers
    target_tiers: List[str] = []

class AdminUserUpdate(BaseModel):
    subscription_tier: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    flex_enabled: Optional[bool] = None

# ============================================================
# App Lifespan & Migrations
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, admin_settings_cache
    init_enc_keys()
    if STRIPE_SECRET_KEY: stripe.api_key = STRIPE_SECRET_KEY
    
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
    
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT settings_json FROM admin_settings WHERE id = 1")
            if row and row["settings_json"]:
                admin_settings_cache.update(json.loads(row["settings_json"]))
    except: pass
    
    if BOOTSTRAP_ADMIN_EMAIL:
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET role='master_admin', subscription_tier='master_admin' WHERE LOWER(email)=$1", BOOTSTRAP_ADMIN_EMAIL)
    
    yield
    if db_pool: await db_pool.close()
    if redis_client: await redis_client.close()

async def run_migrations():
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
            (2, "CREATE TABLE IF NOT EXISTS refresh_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, token_hash VARCHAR(255) UNIQUE NOT NULL, expires_at TIMESTAMPTZ NOT NULL, revoked BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW())"),
            (3, "CREATE TABLE IF NOT EXISTS platform_tokens (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(id) ON DELETE CASCADE, platform VARCHAR(50) NOT NULL, account_id VARCHAR(255), account_name VARCHAR(255), account_username VARCHAR(255), account_avatar VARCHAR(512), token_blob JSONB NOT NULL, is_primary BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())"),
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
        ]
        
        for version, sql in migrations:
            if version not in applied:
                try:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                    logger.info(f"Migration v{version} applied")
                except Exception as e:
                    logger.error(f"Migration v{version} failed: {e}")

app = FastAPI(title="UploadM8 API", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS.split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or _req_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# ============================================================
# Auth Dependencies
# ============================================================
async def get_current_user(request: Request, authorization: Optional[str] = Header(None), token: Optional[str] = Query(None)):
    auth_token = authorization[7:] if authorization and authorization.startswith("Bearer ") else token
    if not auth_token: raise HTTPException(401, "Missing authorization")
    user_id = verify_access_jwt(auth_token)
    if not user_id: raise HTTPException(401, "Invalid token")
    
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not user: raise HTTPException(401, "User not found")
        if user["status"] == "banned": raise HTTPException(403, "Account suspended")
        await conn.execute("UPDATE users SET last_active_at = NOW() WHERE id = $1", user_id)
        # Daily token refill
        await daily_refill(conn, user_id, user["subscription_tier"])
        wallet = await get_wallet(conn, user_id)
        return {**dict(user), "wallet": wallet}

async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "master_admin"): raise HTTPException(403, "Admin required")
    return user

async def require_master_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "master_admin": raise HTTPException(403, "Master admin required")
    return user

# ============================================================
# Health
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_utc().isoformat()}

# ============================================================
# Auth Endpoints
# ============================================================
@app.post("/api/auth/register")
async def register(data: UserCreate, background_tasks: BackgroundTasks):
    async with db_pool.acquire() as conn:
        if await conn.fetchrow("SELECT id FROM users WHERE LOWER(email) = $1", data.email.lower()):
            raise HTTPException(409, "Email already registered")
        user_id = str(uuid.uuid4())
        await conn.execute("INSERT INTO users (id, email, password_hash, name) VALUES ($1, $2, $3, $4)", user_id, data.email.lower(), hash_password(data.password), data.name)
        await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1)", user_id)
        await conn.execute("INSERT INTO wallets (user_id, put_balance, aic_balance) VALUES ($1, 30, 0)", user_id)
        await ledger_entry(conn, user_id, "put", 30, "signup_bonus")
        access = create_access_jwt(user_id)
        refresh = await create_refresh_token(conn, user_id)
    background_tasks.add_task(notify_signup, data.email, data.name)
    background_tasks.add_task(send_email, data.email, "Welcome to UploadM8!", f"<h1>Welcome, {data.name}!</h1><p>Start uploading at <a href='{FRONTEND_URL}'>uploadm8.com</a></p>")
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}

@app.post("/api/auth/login")
async def login(data: UserLogin):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id, password_hash, status FROM users WHERE LOWER(email) = $1", data.email.lower())
        if not user or not verify_password(data.password, user["password_hash"]): raise HTTPException(401, "Invalid credentials")
        if user["status"] == "banned": raise HTTPException(403, "Account suspended")
        return {"access_token": create_access_jwt(str(user["id"])), "refresh_token": await create_refresh_token(conn, str(user["id"])), "token_type": "bearer"}

@app.post("/api/auth/refresh")
async def refresh(data: RefreshRequest):
    async with db_pool.acquire() as conn:
        access, refresh = await rotate_refresh_token(conn, data.refresh_token)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = $1", user["id"])
    return {"status": "logged_out"}

# ============================================================
# User Profile & Wallet
# ============================================================
@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})
    put_available = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
    aic_available = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)
    put_pct = (put_available / max(plan.get("put_monthly", 30), 1)) * 100
    
    # Compute banners
    banners = []
    if put_pct <= 0:
        banners.append({"type": "blocking", "message": "You're out of PUT tokens!", "cta": "top-up"})
    elif put_pct <= 10:
        banners.append({"type": "urgent", "message": f"Only {put_available} PUT left!", "cta": "top-up"})
    elif put_pct <= 30:
        banners.append({"type": "warning", "message": f"{put_available} PUT remaining this period", "cta": "upgrade"})
    
    if plan.get("ai") and aic_available <= 5:
        banners.append({"type": "warning", "message": "Low AI credits!", "cta": "buy-aic"})
    
    return {
        "id": str(user["id"]), "email": user["email"], "name": user["name"], "role": user["role"],
        "subscription_tier": user.get("subscription_tier", "free"), "subscription_status": user.get("subscription_status"),
        "plan": plan, "wallet": {"put_balance": wallet.get("put_balance", 0), "put_reserved": wallet.get("put_reserved", 0),
            "put_available": put_available, "aic_balance": wallet.get("aic_balance", 0), "aic_reserved": wallet.get("aic_reserved", 0),
            "aic_available": aic_available},
        "banners": banners, "flex_enabled": user.get("flex_enabled", False),
        "created_at": user["created_at"].isoformat() if user.get("created_at") else None,
    }

@app.get("/api/wallet")
async def get_wallet_endpoint(user: dict = Depends(get_current_user)):
    wallet = user.get("wallet", {})
    plan = get_plan(user.get("subscription_tier", "free"))
    async with db_pool.acquire() as conn:
        ledger = await conn.fetch("SELECT * FROM token_ledger WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50", user["id"])
    return {
        "wallet": wallet, "plan_limits": {"put_daily": plan.get("put_daily", 1), "put_monthly": plan.get("put_monthly", 30), "aic_monthly": plan.get("aic_monthly", 0)},
        "ledger": [dict(l) for l in ledger],
    }

@app.post("/api/wallet/topup")
async def wallet_topup(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    product = TOPUP_PRODUCTS.get(data.lookup_key)
    if not product: raise HTTPException(400, "Invalid product")
    
    async with db_pool.acquire() as conn:
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user["name"])
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])
    
    prices = stripe.Price.list(lookup_keys=[data.lookup_key], active=True)
    if not prices.data: raise HTTPException(400, "Price not found")
    
    session = stripe.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": prices.data[0].id, "quantity": 1}],
        mode="payment",
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
        metadata={"user_id": str(user["id"]), "wallet": product["wallet"], "amount": product["amount"]},
    )
    return {"checkout_url": session.url}

@app.post("/api/wallet/transfer")
async def wallet_transfer(data: TransferRequest, user: dict = Depends(get_current_user)):
    if not user.get("flex_enabled"):
        raise HTTPException(403, "Flex add-on required for transfers")
    async with db_pool.acquire() as conn:
        success = await transfer_tokens(conn, user["id"], data.from_platform, data.to_platform, data.amount)
    if not success: raise HTTPException(400, "Transfer failed - insufficient balance")
    return {"status": "transferred", "amount": data.amount, "burn": int(data.amount * 0.02)}

# ============================================================
# Settings
# ============================================================
@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        settings = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user["id"])
    return dict(settings) if settings else {}

@app.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    updates, params = [], [user["id"]]
    for field in ["discord_webhook", "telemetry_enabled", "hud_enabled", "hud_position", "speeding_mph", "euphoria_mph"]:
        val = getattr(data, field, None)
        if val is not None:
            updates.append(f"{field} = ${len(params)+1}")
            params.append(val)
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE user_settings SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = $1", *params)
    return {"status": "updated"}

# ============================================================
# Uploads
# ============================================================
@app.post("/api/uploads/presign")
async def presign_upload(data: UploadInit, user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})
    
    # Calculate PUT cost (1 per platform, 2x for large files)
    put_cost = len(data.platforms)
    if data.file_size > 100 * 1024 * 1024:  # >100MB
        put_cost *= 2
    
    # Calculate AIC cost if using AI
    aic_cost = 0
    if data.use_ai and plan.get("ai"):
        aic_cost = 1  # 1 AIC per AI generation
    
    # Check balance
    put_avail = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
    aic_avail = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)
    if put_avail < put_cost:
        raise HTTPException(429, f"Insufficient PUT tokens ({put_avail} available, {put_cost} needed)")
    if aic_cost > 0 and aic_avail < aic_cost:
        raise HTTPException(429, f"Insufficient AIC credits ({aic_avail} available, {aic_cost} needed)")
    
    upload_id = str(uuid.uuid4())
    r2_key = f"uploads/{user['id']}/{upload_id}/{data.filename}"
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO uploads (id, user_id, r2_key, filename, file_size, platforms, title, caption, hashtags, privacy, status, scheduled_time, schedule_mode, put_reserved, aic_reserved)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'pending', $11, $12, $13, $14)
        """, upload_id, user["id"], r2_key, data.filename, data.file_size, data.platforms, data.title, data.caption, data.hashtags, data.privacy, data.scheduled_time, data.schedule_mode, put_cost, aic_cost)
        
        # Reserve tokens
        await reserve_tokens(conn, user["id"], put_cost, aic_cost, upload_id)
    
    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    result = {"upload_id": upload_id, "presigned_url": presigned_url, "r2_key": r2_key, "put_cost": put_cost, "aic_cost": aic_cost}
    
    if data.has_telemetry:
        telem_key = f"uploads/{user['id']}/{upload_id}/telemetry.map"
        result["telemetry_presigned_url"] = generate_presigned_upload_url(telem_key, "application/octet-stream")
        result["telemetry_r2_key"] = telem_key
    
    return result

@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        await conn.execute("UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1", upload_id)
    
    plan = get_plan(user.get("subscription_tier", "free"))
    await enqueue_job({"upload_id": upload_id, "user_id": str(user["id"])}, priority=plan.get("priority", False))
    return {"status": "queued", "upload_id": upload_id}

@app.post("/api/uploads/{upload_id}/cancel")
async def cancel_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT put_reserved, aic_reserved, status FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("completed", "cancelled", "failed"):
            raise HTTPException(400, "Cannot cancel this upload")
        
        await conn.execute("UPDATE uploads SET cancel_requested = TRUE, status = 'cancelled', updated_at = NOW() WHERE id = $1", upload_id)
        # Refund reserved tokens
        await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
    return {"status": "cancelled"}

@app.get("/api/uploads")
async def get_uploads(status: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch("SELECT * FROM uploads WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3 OFFSET $4", user["id"], status, limit, offset)
        else:
            uploads = await conn.fetch("SELECT * FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3", user["id"], limit, offset)
    return [{"id": str(u["id"]), "filename": u["filename"], "platforms": u["platforms"], "status": u["status"], "title": u["title"], "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None, "created_at": u["created_at"].isoformat() if u["created_at"] else None, "put_cost": u["put_reserved"], "aic_cost": u["aic_reserved"]} for u in uploads]

@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT put_reserved, aic_reserved, status FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("pending", "queued"):
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
    return {"status": "deleted"}

# ============================================================
# Platforms
# ============================================================
@app.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("SELECT id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at FROM platform_tokens WHERE user_id = $1 ORDER BY platform, created_at", user["id"])
    
    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms: platforms[p] = []
        platforms[p].append({"id": str(acc["id"]), "account_id": acc["account_id"], "name": acc["account_name"], "username": acc["account_username"], "avatar": acc["account_avatar"], "is_primary": acc["is_primary"]})
    
    plan = get_plan(user.get("subscription_tier", "free"))
    total = sum(len(v) for v in platforms.values())
    return {"platforms": platforms, "total_accounts": total, "max_accounts": plan.get("max_accounts", 1), "can_add_more": total < plan.get("max_accounts", 1)}

@app.delete("/api/platforms/{platform}/accounts/{account_id}")
async def disconnect_account(platform: str, account_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM platform_tokens WHERE id = $1 AND user_id = $2", account_id, user["id"])
    return {"status": "disconnected"}

# ============================================================
# Billing
# ============================================================
@app.post("/api/billing/checkout")
async def create_checkout(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY: raise HTTPException(503, "Billing not configured")
    
    async with db_pool.acquire() as conn:
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user["name"])
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])
    
    prices = stripe.Price.list(lookup_keys=[data.lookup_key], active=True)
    if not prices.data: raise HTTPException(400, f"Invalid plan: {data.lookup_key}")
    
    if data.kind == "subscription":
        session = stripe.checkout.Session.create(
            customer=customer_id,
            line_items=[{"price": prices.data[0].id, "quantity": 1}],
            mode="subscription",
            success_url=STRIPE_SUCCESS_URL,
            cancel_url=STRIPE_CANCEL_URL,
            metadata={"user_id": str(user["id"])},
        )
    else:  # topup
        product = TOPUP_PRODUCTS.get(data.lookup_key, {})
        session = stripe.checkout.Session.create(
            customer=customer_id,
            line_items=[{"price": prices.data[0].id, "quantity": 1}],
            mode="payment",
            success_url=STRIPE_SUCCESS_URL,
            cancel_url=STRIPE_CANCEL_URL,
            metadata={"user_id": str(user["id"]), "wallet": product.get("wallet", "put"), "amount": product.get("amount", 0)},
        )
    return {"checkout_url": session.url, "session_id": session.id}

@app.post("/api/billing/portal")
async def create_portal(user: dict = Depends(get_current_user)):
    if not user.get("stripe_customer_id"): raise HTTPException(400, "No billing account")
    session = stripe.billing_portal.Session.create(customer=user["stripe_customer_id"], return_url=f"{FRONTEND_URL}/settings.html")
    return {"portal_url": session.url}

@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Invalid signature: {e}")
    
    if event.type == "checkout.session.completed":
        session = event.data.object
        user_id = session.metadata.get("user_id")
        
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
            
            if session.mode == "subscription":
                sub = stripe.Subscription.retrieve(session.subscription)
                lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
                tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, "launch")
                plan = get_plan(tier)
                
                await conn.execute("""
                    UPDATE users SET subscription_tier = $1, stripe_subscription_id = $2, subscription_status = 'active',
                    current_period_end = $3, updated_at = NOW() WHERE id = $4
                """, tier, session.subscription, datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc), user_id)
                
                # Credit monthly AIC
                if plan.get("aic_monthly", 0) > 0:
                    await credit_wallet(conn, user_id, "aic", plan["aic_monthly"], "subscription_credit", session.id)
                
                # Track revenue
                amount = (session.amount_total or 0) / 100
                await conn.execute("INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) VALUES ($1, $2, 'subscription', $3, $4)", user_id, amount, session.id, tier)
                
                background_tasks.add_task(notify_mrr, amount, user["email"] if user else "", tier, "subscription")
            
            elif session.mode == "payment":
                # Top-up
                wallet = session.metadata.get("wallet", "put")
                amount_tokens = int(session.metadata.get("amount", 0))
                if amount_tokens > 0:
                    await credit_wallet(conn, user_id, wallet, amount_tokens, "topup", session.id)
                    amount = (session.amount_total or 0) / 100
                    await conn.execute("INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) VALUES ($1, $2, 'topup', $3, $4)", user_id, amount, session.id, f"{wallet}_{amount_tokens}")
                    background_tasks.add_task(notify_topup, amount, user["email"] if user else "", wallet, amount_tokens)
    
    elif event.type == "customer.subscription.updated":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET subscription_status = $1, current_period_end = $2 WHERE stripe_subscription_id = $3",
                sub.status, datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc), sub.id)
    
    elif event.type == "customer.subscription.deleted":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET subscription_tier = 'free', subscription_status = 'cancelled' WHERE stripe_subscription_id = $1", sub.id)
    
    return {"status": "ok"}

# ============================================================
# Analytics
# ============================================================
@app.get("/api/analytics")
async def get_analytics(range: str = "30d", user: dict = Depends(get_current_user)):
    minutes = {"30m": 30, "1h": 60, "6h": 360, "12h": 720, "1d": 1440, "7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
            COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes,
            COALESCE(SUM(put_spent), 0)::int AS put_used, COALESCE(SUM(aic_spent), 0)::int AS aic_used
            FROM uploads WHERE user_id = $1 AND created_at >= $2
        """, user["id"], since)
        
        daily = await conn.fetch("SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads FROM uploads WHERE user_id = $1 AND created_at >= $2 GROUP BY DATE(created_at) ORDER BY date", user["id"], since)
        platforms = await conn.fetch("SELECT unnest(platforms) AS platform, COUNT(*)::int AS count FROM uploads WHERE user_id = $1 AND created_at >= $2 GROUP BY platform", user["id"], since)
    
    return {"total_uploads": stats["total"] if stats else 0, "completed": stats["completed"] if stats else 0, "views": stats["views"] if stats else 0, "likes": stats["likes"] if stats else 0, "put_used": stats["put_used"] if stats else 0, "aic_used": stats["aic_used"] if stats else 0, "daily": [{"date": str(d["date"]), "uploads": d["uploads"]} for d in daily], "platforms": {p["platform"]: p["count"] for p in platforms}}

@app.get("/api/exports/excel")
async def export_excel(type: str = "uploads", range: str = "30d", user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    if not plan.get("excel"): raise HTTPException(403, "Excel export requires Studio+ plan")
    
    minutes = {"7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT filename, platforms, title, status, views, likes, put_spent, aic_spent, created_at FROM uploads WHERE user_id = $1 AND created_at >= $2 ORDER BY created_at DESC", user["id"], since)
    
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Uploads"
        headers = ["Filename", "Platforms", "Title", "Status", "Views", "Likes", "PUT", "AIC", "Created"]
        ws.append(headers)
        for r in rows:
            ws.append([r["filename"], ",".join(r["platforms"] or []), r["title"], r["status"], r["views"], r["likes"], r["put_spent"], r["aic_spent"], str(r["created_at"])])
        output = BytesIO()
        wb.save(output)
        output.seek(0)
        return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=uploadm8_exports.xlsx"})
    except ImportError:
        raise HTTPException(500, "Excel export not available")

# ============================================================
# Admin Endpoints
# ============================================================
@app.get("/api/admin/users")
async def admin_get_users(search: Optional[str] = None, tier: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(require_admin)):
    query = "SELECT id, email, name, role, subscription_tier, subscription_status, status, created_at, last_active_at FROM users WHERE 1=1"
    params = []
    if search:
        params.append(f"%{search}%")
        query += f" AND (email ILIKE ${len(params)} OR name ILIKE ${len(params)})"
    if tier:
        params.append(tier)
        query += f" AND subscription_tier = ${len(params)}"
    params.extend([limit, offset])
    query += f" ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}"
    
    async with db_pool.acquire() as conn:
        users = await conn.fetch(query, *params)
        total = await conn.fetchval("SELECT COUNT(*) FROM users")
    return {"users": [dict(u) for u in users], "total": total}

@app.put("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, data: AdminUserUpdate, user: dict = Depends(require_admin)):
    updates, params = [], [user_id]
    if data.subscription_tier:
        updates.append(f"subscription_tier = ${len(params)+1}")
        params.append(data.subscription_tier)
    if data.role and user.get("role") == "master_admin":
        updates.append(f"role = ${len(params)+1}")
        params.append(data.role)
    if data.status:
        updates.append(f"status = ${len(params)+1}")
        params.append(data.status)
    if data.flex_enabled is not None:
        updates.append(f"flex_enabled = ${len(params)+1}")
        params.append(data.flex_enabled)
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
    return {"status": "updated"}

@app.post("/api/admin/users/{user_id}/ban")
async def admin_ban_user(user_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET status = 'banned' WHERE id = $1", user_id)
    return {"status": "banned"}

@app.post("/api/admin/users/{user_id}/unban")
async def admin_unban_user(user_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET status = 'active' WHERE id = $1", user_id)
    return {"status": "unbanned"}

@app.post("/api/admin/users/assign-tier")
async def admin_assign_tier(user_id: str = Query(...), tier: str = Query(...), user: dict = Depends(require_master_admin)):
    if tier not in PLAN_CONFIG:
        raise HTTPException(400, "Invalid tier")
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET subscription_tier = $1 WHERE id = $2", tier, user_id)
    return {"status": "assigned", "tier": tier}

# ============================================================
# Announcements
# ============================================================
@app.post("/api/admin/announcements/send")
async def send_announcement(data: AnnouncementRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        query = "SELECT id, email, name FROM users WHERE status = 'active'"
        params = []
        if data.target == "paid":
            query += " AND subscription_tier NOT IN ('free')"
        elif data.target == "free":
            query += " AND subscription_tier = 'free'"
        elif data.target == "specific_tiers" and data.target_tiers:
            params.append(data.target_tiers)
            query += f" AND subscription_tier = ANY(${len(params)})"
        
        users = await conn.fetch(query, *params) if params else await conn.fetch(query)
        
        ann_id = str(uuid.uuid4())
        channels = {"email": data.send_email, "discord_community": data.send_discord_community, "user_webhooks": data.send_user_webhooks}
        await conn.execute("INSERT INTO announcements (id, title, body, channels, target, target_tiers, created_by) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            ann_id, data.title, data.body, json.dumps(channels), data.target, data.target_tiers, user["id"])
        
        email_count, discord_count, webhook_count = 0, 0, 0
        
        if data.send_discord_community and COMMUNITY_DISCORD_WEBHOOK_URL:
            await discord_notify(COMMUNITY_DISCORD_WEBHOOK_URL, embeds=[{"title": f"📢 {data.title}", "description": data.body, "color": 0xf97316}])
            discord_count = 1
        
        if data.send_email:
            for u in users:
                background_tasks.add_task(send_email, u["email"], data.title, f"<h1>{data.title}</h1><p>{data.body}</p>")
                email_count += 1
        
        if data.send_user_webhooks:
            settings = await conn.fetch("SELECT user_id, discord_webhook FROM user_settings WHERE discord_webhook IS NOT NULL AND discord_webhook != ''")
            for s in settings:
                background_tasks.add_task(discord_notify, s["discord_webhook"], embeds=[{"title": f"📢 {data.title}", "description": data.body, "color": 0xf97316}])
                webhook_count += 1
        
        await conn.execute("UPDATE announcements SET email_sent = $1, discord_sent = $2, webhook_sent = $3 WHERE id = $4", email_count, discord_count, webhook_count, ann_id)
    
    return {"status": "sent", "announcement_id": ann_id, "email_count": email_count, "discord_count": discord_count, "webhook_count": webhook_count}

@app.get("/api/admin/announcements")
async def get_announcements(limit: int = 20, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        anns = await conn.fetch("SELECT * FROM announcements ORDER BY created_at DESC LIMIT $1", limit)
    return [dict(a) for a in anns]

# ============================================================
# KPI Endpoints
# ============================================================
@app.get("/api/admin/kpi/overview")
async def kpi_overview(range: str = "30d", user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family')")
        
        upload_stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE created_at >= $1
        """, since)
        
        revenue = await conn.fetchrow("SELECT COALESCE(SUM(amount), 0)::decimal AS total, COALESCE(SUM(CASE WHEN source = 'subscription' THEN amount ELSE 0 END), 0)::decimal AS subscriptions, COALESCE(SUM(CASE WHEN source = 'topup' THEN amount ELSE 0 END), 0)::decimal AS topups FROM revenue_tracking WHERE created_at >= $1", since)
        
        mrr_data = await conn.fetch("SELECT subscription_tier, COUNT(*) AS count FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active' GROUP BY subscription_tier")
        mrr = sum(get_plan(r["subscription_tier"]).get("price", 0) * r["count"] for r in mrr_data)
        
        tiers = await conn.fetch("SELECT subscription_tier, COUNT(*)::int AS count FROM users GROUP BY subscription_tier")
    
    return {
        "users": {"new": new_users, "total": total_users, "paid": paid_users},
        "uploads": {"total": upload_stats["total"] if upload_stats else 0, "completed": upload_stats["completed"] if upload_stats else 0, "failed": upload_stats["failed"] if upload_stats else 0, "success_rate": ((upload_stats["completed"] or 0) / max(upload_stats["total"] or 1, 1)) * 100},
        "engagement": {"views": upload_stats["views"] if upload_stats else 0, "likes": upload_stats["likes"] if upload_stats else 0},
        "revenue": {"total": float(revenue["total"]) if revenue else 0, "subscriptions": float(revenue["subscriptions"]) if revenue else 0, "topups": float(revenue["topups"]) if revenue else 0, "mrr": mrr},
        "tiers": {t["subscription_tier"]: t["count"] for t in tiers},
    }

@app.get("/api/admin/kpi/margins")
async def kpi_margins(range: str = "30d", user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200, "6m": 262800}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        costs = await conn.fetchrow("""
            SELECT COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
            COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
            COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        
        revenue = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking WHERE created_at >= $1", since)
        
        tier_data = await conn.fetch("""
            SELECT u.subscription_tier, COUNT(up.id)::int AS uploads, COALESCE(SUM(up.cost_attributed), 0)::decimal AS cost
            FROM users u LEFT JOIN uploads up ON up.user_id = u.id AND up.created_at >= $1
            GROUP BY u.subscription_tier
        """, since)
        
        platform_data = await conn.fetch("""
            SELECT unnest(platforms) AS platform, COUNT(*)::int AS uploads, COALESCE(SUM(cost_attributed), 0)::decimal AS cost
            FROM uploads WHERE created_at >= $1 GROUP BY platform
        """, since)
    
    total_cost = float(costs["openai"] or 0) + float(costs["storage"] or 0) + float(costs["compute"] or 0)
    gross_margin = float(revenue or 0) - total_cost
    
    return {
        "costs": {"openai": float(costs["openai"] or 0), "storage": float(costs["storage"] or 0), "compute": float(costs["compute"] or 0), "total": total_cost},
        "revenue": float(revenue or 0),
        "gross_margin": gross_margin,
        "margin_pct": (gross_margin / max(float(revenue or 1), 1)) * 100,
        "by_tier": {t["subscription_tier"]: {"uploads": t["uploads"], "cost": float(t["cost"])} for t in tier_data},
        "by_platform": {p["platform"]: {"uploads": p["uploads"], "cost": float(p["cost"])} for p in platform_data},
    }

@app.get("/api/admin/kpi/burn")
async def kpi_burn(range: str = "30d", user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        token_stats = await conn.fetchrow("""
            SELECT COALESCE(SUM(CASE WHEN token_type = 'put' AND delta < 0 THEN ABS(delta) ELSE 0 END), 0)::int AS put_spent,
            COALESCE(SUM(CASE WHEN token_type = 'aic' AND delta < 0 THEN ABS(delta) ELSE 0 END), 0)::int AS aic_spent,
            COALESCE(SUM(CASE WHEN reason = 'topup' THEN delta ELSE 0 END), 0)::int AS tokens_purchased
            FROM token_ledger WHERE created_at >= $1
        """, since)
        
        quota_data = await conn.fetch("""
            SELECT u.id, u.subscription_tier, w.put_balance, w.put_reserved
            FROM users u JOIN wallets w ON w.user_id = u.id WHERE u.status = 'active'
        """)
        
        hitting_quota = sum(1 for u in quota_data if (u["put_balance"] - u["put_reserved"]) <= get_plan(u["subscription_tier"]).get("put_daily", 1))
        total_active = len(quota_data)
    
    return {
        "put_spent": token_stats["put_spent"] if token_stats else 0,
        "aic_spent": token_stats["aic_spent"] if token_stats else 0,
        "tokens_purchased": token_stats["tokens_purchased"] if token_stats else 0,
        "users_hitting_quota": hitting_quota,
        "total_active_users": total_active,
        "quota_hit_pct": (hitting_quota / max(total_active, 1)) * 100,
    }

@app.get("/api/admin/kpi/funnels")
async def kpi_funnels(user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        signups_24h = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '24 hours'")
        first_uploads_24h = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE user_id IN (SELECT id FROM users WHERE created_at >= NOW() - INTERVAL '24 hours')")
        
        total_free = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier = 'free'")
        converted = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family') AND subscription_status = 'active'")
        
        ai_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE aic_spent > 0")
        total_uploaders = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads")
        
        topup_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM token_ledger WHERE reason = 'topup'")
        flex_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE flex_enabled = TRUE")
        churned = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_status = 'cancelled' AND updated_at >= NOW() - INTERVAL '30 days'")
    
    return {
        "signup_to_upload_24h": {"signups": signups_24h, "uploads": first_uploads_24h, "rate": (first_uploads_24h / max(signups_24h, 1)) * 100},
        "free_to_paid": {"free": total_free, "paid": converted, "rate": (converted / max(total_free + converted, 1)) * 100},
        "ai_usage": {"users": ai_users, "total": total_uploaders, "rate": (ai_users / max(total_uploaders, 1)) * 100},
        "topup_adoption": {"users": topup_users, "total": total_uploaders, "rate": (topup_users / max(total_uploaders, 1)) * 100},
        "flex_adoption": {"users": flex_users},
        "churn_30d": churned,
    }

@app.get("/api/admin/settings")
async def get_admin_settings(user: dict = Depends(require_master_admin)):
    return admin_settings_cache

@app.put("/api/admin/settings")
async def update_admin_settings(settings: dict, user: dict = Depends(require_master_admin)):
    global admin_settings_cache
    admin_settings_cache.update(settings)
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1", json.dumps(admin_settings_cache))
    return {"status": "updated", "settings": admin_settings_cache}

@app.post("/api/admin/weekly-report")
async def trigger_weekly_report(user: dict = Depends(require_master_admin)):
    since = _now_utc() - timedelta(days=7)
    async with db_pool.acquire() as conn:
        costs = await conn.fetchrow("""
            SELECT COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
            COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
            COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        revenue = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking WHERE created_at >= $1", since)
    
    await notify_weekly_costs(float(costs["openai"] or 0), float(costs["storage"] or 0), float(costs["compute"] or 0), float(revenue or 0))
    return {"status": "sent"}

# ============================================================
# Dashboard Stats
# ============================================================
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})
    
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::int AS completed,
            SUM(CASE WHEN status IN ('pending', 'queued', 'processing') THEN 1 ELSE 0 END)::int AS in_queue,
            COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE user_id = $1
        """, user["id"])
        accounts = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])
        recent = await conn.fetch("SELECT id, filename, platforms, status, created_at FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5", user["id"])
    
    put_avail = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
    aic_avail = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)
    
    return {
        "uploads": {"total": stats["total"] if stats else 0, "completed": stats["completed"] if stats else 0, "in_queue": stats["in_queue"] if stats else 0},
        "engagement": {"views": stats["views"] if stats else 0, "likes": stats["likes"] if stats else 0},
        "wallet": {"put_available": put_avail, "put_total": wallet.get("put_balance", 0), "aic_available": aic_avail, "aic_total": wallet.get("aic_balance", 0)},
        "accounts": {"connected": accounts, "limit": plan.get("max_accounts", 1)},
        "recent": [{"id": str(r["id"]), "filename": r["filename"], "platforms": r["platforms"], "status": r["status"]} for r in recent],
        "plan": plan,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
