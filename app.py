"""
UploadM8 API Server - Production Build v4
# ====================
Complete implementation with:
- PUT/AIC wallet system with ledger
- Announcements system
- Full KPI dashboards
- Weekly cost reports
- All Stripe integrations
"""

import os
import pathlib
import csv
import io
import json


# ---------------------------------------------------------------------------
# DB JSON CODECS (architectural cleanliness)
# Forces asyncpg to decode json/jsonb into Python objects.
# Keep _safe_json as a belt-and-suspenders fallback until schema is fully normalized.
# ---------------------------------------------------------------------------
async def _init_asyncpg_codecs(conn):
    try:
        await conn.set_type_codec(
            'json',
            encoder=lambda v: json.dumps(v, separators=(',', ':'), ensure_ascii=False),
            decoder=json.loads,
            schema='pg_catalog',
        )
    except Exception:
        pass
    try:
        await conn.set_type_codec(
            'jsonb',
            encoder=lambda v: json.dumps(v, separators=(',', ':'), ensure_ascii=False),
            decoder=json.loads,
            schema='pg_catalog',
        )
    except Exception:
        pass

_UPLOADS_COLS = None

async def _load_uploads_columns(pool):
    """Cache uploads column set to avoid UndefinedColumnError when schema drifts."""
    global _UPLOADS_COLS
    if _UPLOADS_COLS is not None:
        return _UPLOADS_COLS
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name
                 FROM information_schema.columns
                 WHERE table_schema='public' AND table_name='uploads'"""
        )
        _UPLOADS_COLS = {r['column_name'] for r in rows}
    return _UPLOADS_COLS

def _pick_cols(wanted, available):
    return [c for c in wanted if c in available]


def _safe_json(v, default):
    """Parse JSON stored as text OR already-parsed objects. Defensive until schema is fully jsonb."""
    if v is None:
        return default
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return default
import secrets
import hashlib
import base64
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional
from decimal import Decimal
from urllib.parse import urlencode, quote, urlsplit, urlunsplit, parse_qsl

# Sensitive query-param keys that must never appear in logs
_SENSITIVE_KEYS = {"access_token", "client_secret", "code", "refresh_token", "fb_exchange_token"}

def redact_url(url: str) -> str:
    """Strip sensitive query params from a URL before logging it."""
    try:
        parts = urlsplit(url)
        q = parse_qsl(parts.query, keep_blank_values=True)
        redacted = [(k, "***" if k in _SENSITIVE_KEYS else v) for k, v in q]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted), parts.fragment))
    except Exception:
        return "<url-redact-error>"
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

from fastapi import FastAPI, HTTPException, Depends, Query, Header, BackgroundTasks, Request, UploadFile, File, Body
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from contextlib import asynccontextmanager

# ============================================================
# USER PREFERENCES SYSTEM
# ============================================================

class PlatformHashtags(BaseModel):
    tiktok: List[str] = Field(default_factory=list)
    youtube: List[str] = Field(default_factory=list)
    instagram: List[str] = Field(default_factory=list)
    facebook: List[str] = Field(default_factory=list)

class UserPreferencesUpdate(BaseModel):
    # Accept both snake_case (backend) and camelCase (frontend) keys.
    auto_captions: bool = Field(False, alias="autoCaptions")
    auto_thumbnails: bool = Field(False, alias="autoThumbnails")
    thumbnail_interval: int = Field(5, ge=1, le=60, alias="thumbnailInterval")

    default_privacy: Literal["public", "private", "unlisted"] = Field("public", alias="defaultPrivacy")

    ai_hashtags_enabled: bool = Field(False, alias="aiHashtagsEnabled")
    ai_hashtag_count: int = Field(5, ge=1, le=30, alias="aiHashtagCount")
    ai_hashtag_style: Literal["lowercase", "capitalized", "camelcase", "mixed"] = Field("mixed", alias="aiHashtagStyle")
    hashtag_position: Literal["start", "end", "caption"] = Field("end", alias="hashtagPosition")

    max_hashtags: int = Field(15, ge=1, le=50, alias="maxHashtags")
    always_hashtags: List[str] = Field(default_factory=list, alias="alwaysHashtags")
    blocked_hashtags: List[str] = Field(default_factory=list, alias="blockedHashtags")
    platform_hashtags: PlatformHashtags = Field(default_factory=PlatformHashtags, alias="platformHashtags")
    email_notifications: bool = Field(True, alias="emailNotifications")
    discord_webhook: Optional[str] = Field(None, alias="discordWebhook")

    class Config:
        populate_by_name = True
        extra = "ignore"



LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("uploadm8-api")

# httpx logs every outbound request URL at INFO level — including access_token,
# client_secret, and OAuth codes in query params. Suppress to WARNING so those
# URLs never appear in logs or log aggregators.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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


def _normalize_r2_key(key: str) -> str:
    """Normalize object keys to prevent bucket/bucket/... poisoning and signature mismatches."""
    if not key:
        return ""
    k = str(key).lstrip("/")
    bucket = (R2_BUCKET_NAME or "").strip()
    if bucket:
        prefix = bucket + "/"
        # Strip duplicated bucket prefixes (e.g., bucket/bucket/key or bucket/key)
        while k.startswith(prefix):
            k = k[len(prefix):]
    # Collapse accidental double slashes
    while "//" in k:
        k = k.replace("//", "/")
    return k

def generate_presigned_download_url(key: str, ttl: int = 3600) -> str:
    """Generate a short-lived signed GET URL for a private R2 object."""
    k = _normalize_r2_key(key)
    if not k:
        return ""
    if not R2_BUCKET_NAME:
        raise RuntimeError("Missing R2_BUCKET_NAME env var")
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": k},
        ExpiresIn=int(ttl),
    )
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "")

# ── 4-Lane Queue Architecture ─────────────────────────────────
# process lanes: FFmpeg-heavy jobs (WORKER_CONCURRENCY=3 slots)
PROCESS_PRIORITY_QUEUE = os.environ.get("PROCESS_PRIORITY_QUEUE", "uploadm8:process:priority")
PROCESS_NORMAL_QUEUE   = os.environ.get("PROCESS_NORMAL_QUEUE",   "uploadm8:process:normal")
# publish lanes: API-light jobs (PUBLISH_CONCURRENCY=5 slots)
PUBLISH_PRIORITY_QUEUE = os.environ.get("PUBLISH_PRIORITY_QUEUE", "uploadm8:publish:priority")
PUBLISH_NORMAL_QUEUE   = os.environ.get("PUBLISH_NORMAL_QUEUE",   "uploadm8:publish:normal")
# Legacy env vars kept for backward compat
UPLOAD_JOB_QUEUE   = os.environ.get("UPLOAD_JOB_QUEUE",   "uploadm8:jobs")
PRIORITY_JOB_QUEUE = os.environ.get("PRIORITY_JOB_QUEUE", "uploadm8:priority")

# Stripe
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SUCCESS_URL = os.environ.get(
    "STRIPE_SUCCESS_URL",
    f"{FRONTEND_URL}/billing/success.html?session_id={{CHECKOUT_SESSION_ID}}"
)
STRIPE_CANCEL_URL = os.environ.get("STRIPE_CANCEL_URL", f"{FRONTEND_URL}/index.html#pricing")

BILLING_MODE = os.environ.get("BILLING_MODE", "test").strip().lower()

# Discord Webhooks
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
SIGNUP_DISCORD_WEBHOOK_URL = os.environ.get("SIGNUP_DISCORD_WEBHOOK_URL", "")
MRR_DISCORD_WEBHOOK_URL = os.environ.get("MRR_DISCORD_WEBHOOK_URL", "")
COMMUNITY_DISCORD_WEBHOOK_URL = (
    os.getenv("DISCORD_COMMUNITY_WEBHOOK_URL", "").strip()
    or os.getenv("DISCORD_COMMUNITY_WEBHOOK", "").strip()
    or os.getenv("COMMUNITY_DISCORD_WEBHOOK_URL", "").strip()
)
# Email
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")

# Cost modeling
COST_PER_OPENAI_TOKEN = float(os.environ.get("COST_PER_OPENAI_TOKEN", "0.00001"))
COST_PER_GB_MONTH = float(os.environ.get("COST_PER_GB_MONTH", "0.015"))
COST_PER_COMPUTE_SECOND = float(os.environ.get("COST_PER_COMPUTE_SECOND", "0.0001"))

# Trill Telemetry Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GAZETTEER_PLACES_PATH = os.environ.get("GAZETTEER_PLACES_PATH", "")
PADUS_PATH = os.environ.get("PADUS_PATH", "")
PADUS_LAYER = os.environ.get("PADUS_LAYER", "")

# ── External Provider Cost Sync ──────────────────────────────────────────────
# Upstash: email + API key from console.upstash.com → Account → Management API
UPSTASH_EMAIL     = os.environ.get("UPSTASH_EMAIL", "")
UPSTASH_API_KEY   = os.environ.get("UPSTASH_API_KEY", "")
UPSTASH_DB_ID     = os.environ.get("UPSTASH_DB_ID", "")   # from console URL or list endpoint

# Cloudflare: API token with R2 Read + Account Analytics Read permissions
CF_API_TOKEN      = os.environ.get("CF_API_TOKEN", "")    # separate from R2 access key

# Render: no billing API exists — enter monthly total as an env var.
# Update this whenever your service config changes (new services, tier upgrades).
RENDER_MONTHLY_COST = float(os.environ.get("RENDER_MONTHLY_COST", "0"))

# Mailgun: uses existing MAILGUN_API_KEY + MAILGUN_DOMAIN already defined above
# Stripe:  uses existing STRIPE_SECRET_KEY already defined above

# Trill system prompt for AI content generation
TRILL_SYSTEM_PROMPT = """You are an expert social media content creator specializing in driving/automotive content.
Create viral, high-engagement content that balances excitement with platform safety guidelines.

STYLE GUIDE:
- High-energy, aspirational tone
- Use curiosity gaps and FOMO triggers
- Platform-native language (not corporate)
- Mystery hooks that make people watch
- Avoid: Illegal references, exact speeds, clickbait lies

EMOJI USAGE:
- Titles: 1-2 emojis max (fire 🔥, lightning ⚡, eyes 👀)
- Captions: 2-3 emojis strategically placed
- Never excessive or spammy

HASHTAG STRATEGY:
- Mix viral mega-tags (#fyp, #viral, #trending)
- Niche community tags (#spiriteddrive, #roadtrip)
- Location-based tags (#Utah, #Moab)
- Motion tags when relevant (#curvyroads, #switchbacks)
- Protected lands tags ONLY when verified
"""

# Global State
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[aioredis.Redis] = None
ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"
admin_settings_cache: Dict[str, Any] = {"demo_data_enabled": False, "billing_mode": BILLING_MODE}

# ============================================================
# Plan Configuration (PUT/AIC based)
# ============================================================
# ── Entitlements: single source of truth ──────────────────────
# PLAN_CONFIG removed. All tier data lives in entitlements.py.
# Import everything we need from there.
from stages.entitlements import (
    TIER_CONFIG,
    STRIPE_LOOKUP_TO_TIER,
    TOPUP_PRODUCTS,
    PRIORITY_QUEUE_CLASSES,
    get_entitlements_for_tier,
    get_entitlements_from_user,
    entitlements_to_dict,
    get_tier_from_lookup_key,
    get_priority_lane,
    check_queue_depth,
    compute_upload_cost,   # canonical PUT/AIC formula
)

def get_plan(tier: str) -> dict:
    """
    Backward-compat shim — returns entitlements_to_dict() so existing callers
    that do plan.get("ai") / plan.get("watermark") keep working unchanged.
    """
    return entitlements_to_dict(get_entitlements_for_tier(tier))

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
    row = await conn.fetchrow("SELECT id, user_id, expires_at, revoked_at FROM refresh_tokens WHERE token_hash=$1", h)
    if not row: raise HTTPException(401, "Invalid")
    if row["revoked_at"]:
        await conn.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL", row["user_id"])
        raise HTTPException(401, "Reuse detected")
    if row["expires_at"] < _now_utc(): raise HTTPException(401, "Expired")
    await conn.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE id=$1", row["id"])
    return create_access_jwt(str(row["user_id"])), await create_refresh_token(conn, row["user_id"])

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
    ent = get_entitlements_for_tier(tier)
    daily = ent.put_daily * 4  # 4 platforms
    wallet = await get_wallet(conn, user_id)
    last_refill = wallet.get("last_refill_date")
    today = _now_utc().date()
    if last_refill and last_refill >= today:
        return
    monthly_cap = ent.put_monthly
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



# --- R2 helpers (single source of truth: users.avatar_r2_key) ---
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME") or os.getenv("R2_BUCKET") or os.getenv("R2_BUCKET_NAME".lower())

def r2_presign_get_url(r2_key: str, expires_in: int = 3600) -> str:

    """Generate a short-lived signed URL for a private R2 object."""
    return generate_presigned_download_url(r2_key, ttl=int(expires_in))

def generate_presigned_upload_url(key: str, content_type: str, ttl: int = 3600) -> str:
    key = _normalize_r2_key(key)
    s3 = get_s3_client()
    return s3.generate_presigned_url("put_object", Params={"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type}, ExpiresIn=ttl)

# ============================================================
# Redis Queue — 4-Lane Architecture
# ============================================================
async def enqueue_job(
    job_data: dict,
    lane: str = "process",
    priority_class: str = "p4",
) -> bool:
    """
    Push a job to the correct Redis lane based on job type and tier priority.

    Args:
        job_data:       Job payload dict. upload_id required.
        lane:           "process" (FFmpeg-heavy) or "publish" (API-light).
        priority_class: Tier priority class p0-p4. p0/p1/p2 → priority queue.
                        p3/p4 → normal queue.

    Queue routing:
        process + priority_class in {p0,p1,p2} → PROCESS_PRIORITY_QUEUE
        process + priority_class in {p3,p4}    → PROCESS_NORMAL_QUEUE
        publish + priority_class in {p0,p1,p2} → PUBLISH_PRIORITY_QUEUE
        publish + priority_class in {p3,p4}    → PUBLISH_NORMAL_QUEUE
    """
    if not redis_client:
        logger.warning("enqueue_job called but redis_client is None")
        return False

    is_priority = priority_class in PRIORITY_QUEUE_CLASSES

    if lane == "publish":
        queue = PUBLISH_PRIORITY_QUEUE if is_priority else PUBLISH_NORMAL_QUEUE
    else:
        queue = PROCESS_PRIORITY_QUEUE if is_priority else PROCESS_NORMAL_QUEUE

    job_data["enqueued_at"]    = _now_utc().isoformat()
    job_data["job_id"]         = str(uuid.uuid4())
    job_data["lane"]           = lane
    job_data["priority_class"] = priority_class

    try:
        await redis_client.lpush(queue, json.dumps(job_data))
        logger.debug(
            f"[{job_data.get('upload_id', '?')}] Enqueued → {queue} "            f"(lane={lane} priority_class={priority_class})"
        )
        return True
    except Exception as e:
        logger.error(f"enqueue_job failed: {e}")
        return False


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


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=16)
    new_password: str = Field(min_length=8)


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
    schedule_mode: str = "immediate"  # immediate | scheduled | smart
    has_telemetry: bool = False
    use_ai: bool = False
    smart_schedule_days: int = 7  # How many days to spread uploads across

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

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

class ProfileUpdateSettings(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: Optional[str] = None

class PreferencesUpdate(BaseModel):
    emailNotifs: Optional[bool] = None
    uploadCompleteNotifs: Optional[bool] = None
    marketingEmails: Optional[bool] = None
    theme: Optional[str] = None
    accentColor: Optional[str] = None
    defaultPrivacy: Optional[str] = None
    autoPublish: Optional[bool] = None
    alwaysHashtags: Optional[List[str]] = None
    blockedHashtags: Optional[List[str]] = None
    tiktokHashtags: Optional[str] = None
    youtubeHashtags: Optional[str] = None
    instagramHashtags: Optional[str] = None
    facebookHashtags: Optional[str] = None
    hashtagPosition: Optional[str] = None
    maxHashtags: Optional[int] = None
    aiHashtagsEnabled: Optional[bool] = None
    aiHashtagCount: Optional[int] = None
    aiHashtagStyle: Optional[str] = None
    platformHashtags: Optional[dict] = None

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


class AdminWalletAdjust(BaseModel):
    wallet: str  = Field(..., pattern="^(put|aic)$")
    mode:   str  = Field(..., pattern="^(add|subtract|set)$")
    amount: int  = Field(..., ge=0, le=999999)
    reason: str  = Field(..., min_length=3, max_length=200)



class AdminUpdateEmailIn(BaseModel):
    email: EmailStr

class AdminResetPasswordIn(BaseModel):
    temp_password: str = Field(min_length=8, max_length=128)

class ScheduledUploadUpdate(BaseModel):
    title: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    timezone: Optional[str] = None
    platforms: Optional[List[str]] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None

class ColorPreferencesUpdate(BaseModel):
    tiktok_color: Optional[str] = None
    youtube_color: Optional[str] = None
    instagram_color: Optional[str] = None
    facebook_color: Optional[str] = None
    accent_color: Optional[str] = None

# ============================================================
# Smart Upload Scheduling
# ============================================================
# Platform-specific optimal posting times (in UTC)
# Based on general social media engagement research
PLATFORM_OPTIMAL_TIMES = {
    "tiktok": [
        {"hour": 7, "minute": 0, "weight": 0.8},   # 7 AM - morning scroll
        {"hour": 12, "minute": 0, "weight": 0.9},  # 12 PM - lunch break
        {"hour": 15, "minute": 0, "weight": 0.7},  # 3 PM - afternoon break
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - evening prime time
        {"hour": 21, "minute": 0, "weight": 0.95}, # 9 PM - night engagement
        {"hour": 23, "minute": 0, "weight": 0.6},  # 11 PM - late night
    ],
    "youtube": [
        {"hour": 12, "minute": 0, "weight": 0.7},  # 12 PM - lunch views
        {"hour": 14, "minute": 0, "weight": 0.8},  # 2 PM - afternoon
        {"hour": 17, "minute": 0, "weight": 0.9},  # 5 PM - after work/school
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 21, "minute": 0, "weight": 0.95}, # 9 PM - evening viewing
    ],
    "instagram": [
        {"hour": 6, "minute": 0, "weight": 0.7},   # 6 AM - early morning
        {"hour": 11, "minute": 0, "weight": 0.85}, # 11 AM - mid-morning
        {"hour": 13, "minute": 0, "weight": 0.9},  # 1 PM - lunch
        {"hour": 17, "minute": 0, "weight": 0.8},  # 5 PM - commute
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 21, "minute": 0, "weight": 0.9},  # 9 PM - evening
    ],
    "facebook": [
        {"hour": 9, "minute": 0, "weight": 0.8},   # 9 AM - morning check
        {"hour": 11, "minute": 0, "weight": 0.7},  # 11 AM - mid-morning
        {"hour": 13, "minute": 0, "weight": 0.9},  # 1 PM - lunch break
        {"hour": 16, "minute": 0, "weight": 0.85}, # 4 PM - afternoon
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 20, "minute": 0, "weight": 0.9},  # 8 PM - evening
    ],
}

# Best days for each platform (0=Monday, 6=Sunday)
PLATFORM_OPTIMAL_DAYS = {
    "tiktok": [1, 2, 3, 4],      # Tue, Wed, Thu, Fri - highest engagement
    "youtube": [3, 4, 5],        # Thu, Fri, Sat - weekend viewing prep
    "instagram": [0, 1, 2, 4],   # Mon, Tue, Wed, Fri - weekday engagement
    "facebook": [0, 1, 2, 3],    # Mon, Tue, Wed, Thu - business days
}

import random
def calculate_smart_schedule(platforms: List[str], num_days: int = 7, user_timezone: str = "UTC") -> Dict[str, datetime]:
    """
    Calculate optimal upload times for each platform.
    Ensures uploads are spread across different days.
    Returns a dict mapping platform -> scheduled datetime
    """
    now = _now_utc()
    schedule = {}
    used_days = set()
    
    # Sort platforms to ensure consistent ordering
    platforms_sorted = sorted(platforms)
    
    for platform in platforms_sorted:
        optimal_times = PLATFORM_OPTIMAL_TIMES.get(platform, PLATFORM_OPTIMAL_TIMES["tiktok"])
        optimal_days = PLATFORM_OPTIMAL_DAYS.get(platform, [0, 1, 2, 3, 4])
        
        # Find an available day that hasn't been used
        available_days = []
        for day_offset in range(1, num_days + 1):
            target_date = now + timedelta(days=day_offset)
            weekday = target_date.weekday()
            
            # Prefer optimal days for this platform, but allow any day if needed
            if day_offset not in used_days:
                priority = 2 if weekday in optimal_days else 1
                available_days.append((day_offset, priority, weekday))
        
        if not available_days:
            # All days used, pick a random future day
            day_offset = random.randint(1, num_days)
        else:
            # Sort by priority (optimal days first), then randomize within priority
            available_days.sort(key=lambda x: (-x[1], random.random()))
            day_offset = available_days[0][0]
        
        used_days.add(day_offset)
        
        # Pick an optimal time slot with weighted randomization
        weights = [t["weight"] for t in optimal_times]
        total_weight = sum(weights)
        rand_val = random.uniform(0, total_weight)
        
        cumulative = 0
        selected_time = optimal_times[0]
        for t in optimal_times:
            cumulative += t["weight"]
            if rand_val <= cumulative:
                selected_time = t
                break
        
        # Add randomization to the time (±30 minutes)
        minute_offset = random.randint(-30, 30)
        
        # Calculate the final datetime
        target_date = now + timedelta(days=day_offset)
        scheduled_dt = target_date.replace(
            hour=selected_time["hour"],
            minute=max(0, min(59, selected_time["minute"] + minute_offset)),
            second=0,
            microsecond=0
        )
        
        # Make sure it's in the future
        if scheduled_dt <= now:
            scheduled_dt += timedelta(days=1)
        
        schedule[platform] = scheduled_dt
    
    return schedule

async def get_existing_scheduled_days(conn, user_id: str, num_days: int = 7) -> set:
    """Get days that already have scheduled uploads for this user"""
    now = _now_utc()
    end_date = now + timedelta(days=num_days)
    
    existing = await conn.fetch("""
        SELECT DISTINCT DATE(scheduled_time) as sched_date 
        FROM uploads 
        WHERE user_id = $1 
        AND scheduled_time >= $2 
        AND scheduled_time <= $3 
        AND status IN ('pending', 'queued', 'scheduled')
    """, user_id, now, end_date)
    
    used_days = set()
    for row in existing:
        if row["sched_date"]:
            day_diff = (row["sched_date"] - now.date()).days
            if day_diff > 0:
                used_days.add(day_diff)
    
    return used_days
# App Lifespan & Migrations
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, admin_settings_cache
    init_enc_keys()
    if STRIPE_SECRET_KEY: stripe.api_key = STRIPE_SECRET_KEY
    
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, init=_init_asyncpg_codecs)
    await _load_uploads_columns(db_pool)
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
    
    # Seed trill places for geo-targeting
    try:
        async with db_pool.acquire() as conn:
            await seed_trill_places(conn)
            logger.info("Trill places seeded")
    except Exception as e:
        logger.warning(f"Trill places seeding failed: {e}")
    
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

# ── Self-serve deletion audit trail ──────────────────────────────────────
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
        -- raw envelope fields from TikTok
        client_key      TEXT,
        event           TEXT NOT NULL,
        create_time     BIGINT,
        user_openid     TEXT,
        content         JSONB,
        raw_body        TEXT,
        -- our processing result
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

# ── Per-upload engagement metrics (comments + shares) ──────────────────────
(604, """
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS comments BIGINT DEFAULT 0;
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS shares   BIGINT DEFAULT 0;
"""),

# ── Analytics auto-sync tracking ─────────────────────────────────────────
(605, """
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS analytics_synced_at TIMESTAMPTZ;
    CREATE INDEX IF NOT EXISTS idx_uploads_analytics_sync
        ON uploads(status, analytics_synced_at)
        WHERE status IN ('completed', 'succeeded', 'partial');
"""),

# ── Comprehensive audit system — corporate-grade event logging ──────────────
(700, """
    -- Upgrade admin_audit_log with full corporate columns
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

    -- system_event_log: user-triggered events (uploads, platform connects, button clicks)
    -- admin_audit_log tracks admin actions; this tracks all user actions
    CREATE TABLE IF NOT EXISTS system_event_log (
        id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         UUID        REFERENCES users(id) ON DELETE SET NULL,
        event_category  VARCHAR(50) NOT NULL,   -- UPLOAD, PLATFORM, AUTH, UI_ACTION, SYSTEM
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

# ── CORS-safe exception handler ──────────────────────────────────────────────
# FastAPI's CORSMiddleware does NOT add Access-Control-Allow-Origin to 500
# responses when an unhandled exception propagates — the browser then reports
# a CORS error instead of the real HTTP 500.  This handler catches every
# unhandled exception and returns a proper JSON 500 with CORS headers so the
# browser (and developer tools) always see the real error.
@app.exception_handler(Exception)
async def _cors_safe_500_handler(request: Request, exc: Exception) -> JSONResponse:
    origin = request.headers.get("origin", "")
    allowed = ALLOWED_ORIGINS.split(",")
    # Only reflect the origin if it's in our allow-list
    cors_origin = origin if origin in allowed else (allowed[0] if allowed else "*")

    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
        headers={
            "Access-Control-Allow-Origin":      cors_origin,
            "Access-Control-Allow-Credentials": "true",
        },
    )

# ============================================================
# SECURITY + RATE LIMITING (in-memory MVP)
# Replace with Redis later (same interface)
# ============================================================
_RATE_BUCKETS: Dict[str, Dict[str, Any]] = {}

def _rl_now() -> float:
    return time.time()

def rate_limit(key: str, limit: int, window_sec: int) -> bool:
    """Count requests within a sliding window. Returns True if allowed."""
    bucket = _RATE_BUCKETS.get(key)
    t = _rl_now()
    if not bucket or t > bucket["reset_at"]:
        _RATE_BUCKETS[key] = {"count": 1, "reset_at": t + window_sec}
        return True
    if bucket["count"] >= limit:
        return False
    bucket["count"] += 1
    return True

def client_ip(req: Request) -> str:
    # If behind proxy/CDN, consider X-Forwarded-For parsing.
    return (req.client.host if req.client else "unknown")

def _json_429(detail: str) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": detail})

def install_rate_limit_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def rl_middleware(request: Request, call_next):
        ip = client_ip(request)
        path = request.url.path

        # Global guardrail
        if not rate_limit(f"ip:{ip}:global", limit=300, window_sec=60):
            return _json_429("Rate limit exceeded (global)")

        # Sensitive surfaces
        if path.startswith("/api/auth/"):
            if not rate_limit(f"ip:{ip}:auth", limit=30, window_sec=60):
                return _json_429("Rate limit exceeded (auth)")
        if path.startswith("/api/admin/"):
            if not rate_limit(f"ip:{ip}:admin", limit=60, window_sec=60):
                return _json_429("Rate limit exceeded (admin)")

        return await call_next(request)

# Activate in-memory rate limiting
install_rate_limit_middleware(app)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or _req_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline' https:; script-src 'self' 'unsafe-inline' https:; connect-src 'self' https:;"
    return resp

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


async def log_admin_audit(conn, *, user_id: str, admin: dict, action: str, details: dict = None,
                          request=None, event_category: str = "ADMIN", resource_type: str = None,
                          resource_id: str = None, severity: str = "INFO", outcome: str = "SUCCESS"):
    """
    Write a tamper-evident audit record to admin_audit_log.
    Corporate-grade: captures category, resource, actor, IP, user-agent, severity.
    Safe to call inside an existing connection/transaction. NEVER raises.
    """
    try:
        ip_address = None
        user_agent = None
        if request is not None:
            forwarded = request.headers.get("x-forwarded-for")
            ip_address = forwarded.split(",")[0].strip() if forwarded else (
                request.client.host if request.client else None
            )
            user_agent = request.headers.get("user-agent", "")[:512]

        await conn.execute(
            """
            INSERT INTO admin_audit_log
                (user_id, admin_id, admin_email, action, details, ip_address,
                 event_category, actor_user_id, resource_type, resource_id,
                 user_agent, severity, outcome)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb, $6, $7, $8::uuid, $9, $10, $11, $12, $13)
            """,
            str(user_id),
            str(admin.get("id", "")),
            admin.get("email", ""),
            action,
            json.dumps(details or {}),
            ip_address,
            event_category,
            str(admin.get("id", "")),
            resource_type,
            str(resource_id) if resource_id else None,
            user_agent,
            severity,
            outcome,
        )
    except Exception as e:
        logger.error(f"[audit] Failed to write admin audit log: {e}")
        # NEVER raise — audit failure must never break the primary operation


async def log_system_event(conn=None, *, user_id: str = None, action: str, event_category: str = "SYSTEM",
                            resource_type: str = None, resource_id: str = None, details: dict = None,
                            request=None, severity: str = "INFO", outcome: str = "SUCCESS"):
    """
    Write a system/user-action event to system_event_log.
    Used for uploads, platform connects, UI button clicks, auth events.
    Accepts an existing conn or acquires its own. NEVER raises.
    """
    async def _write(c):
        try:
            ip_address = None
            user_agent = None
            session_id = None
            if request is not None:
                forwarded = request.headers.get("x-forwarded-for")
                ip_address = forwarded.split(",")[0].strip() if forwarded else (
                    request.client.host if request.client else None
                )
                user_agent = request.headers.get("user-agent", "")[:512]
                session_id = request.headers.get("x-session-id", "")[:128] or None

            await c.execute(
                """
                INSERT INTO system_event_log
                    (user_id, event_category, action, resource_type, resource_id,
                     details, ip_address, user_agent, session_id, severity, outcome)
                VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
                """,
                str(user_id) if user_id else None,
                event_category,
                action,
                resource_type,
                str(resource_id) if resource_id else None,
                json.dumps(details or {}),
                ip_address,
                user_agent,
                session_id,
                severity,
                outcome,
            )
        except Exception as e:
            logger.error(f"[audit] Failed to write system event log: {e}")

    try:
        if conn is not None:
            await _write(conn)
        else:
            async with db_pool.acquire() as c:
                await _write(c)
    except Exception as e:
        logger.error(f"[audit] System event log pool error: {e}")


async def _purge_old_audit_logs():
    """Purge audit records older than 6 months (rolling window). Run periodically."""
    try:
        async with db_pool.acquire() as conn:
            deleted_admin = await conn.fetchval(
                "DELETE FROM admin_audit_log WHERE created_at < NOW() - INTERVAL '6 months' RETURNING id"
            )
            deleted_sys = await conn.fetchval(
                "DELETE FROM system_event_log WHERE created_at < NOW() - INTERVAL '6 months' RETURNING id"
            )
            logger.info(f"[audit] Purged old logs: admin_audit={deleted_admin or 0}, system_event={deleted_sys or 0}")
    except Exception as e:
        logger.error(f"[audit] Purge failed: {e}")


# ============================================================
# TRILL TELEMETRY SYSTEM
# ============================================================

def generate_trill_content(trill_metadata: dict, user_prefs: dict = None) -> dict:
    """
    Use OpenAI to generate viral titles, captions, and hashtags based on trill metrics.
    
    Args:
        trill_metadata: Output from telemetry_trill.analyze_video()
        user_prefs: User preferences for generation style
        
    Returns:
        {
            "title": "Generated title",
            "caption": "Generated caption", 
            "hashtags": ["tag1", "tag2", ...],
            "tokens_used": 150,
            "model": "gpt-4o-mini"
        }
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    import openai
    openai.api_key = OPENAI_API_KEY
    
    # Extract key metrics
    score = trill_metadata.get("trill_score", 0)
    bucket = trill_metadata.get("speed_bucket", "CRUISE MODE")
    place = trill_metadata.get("place_name", "")
    state = trill_metadata.get("state", "")
    protected_name = trill_metadata.get("protected_name")
    near_protected = trill_metadata.get("near_protected", False)
    elev_gain = trill_metadata.get("elev_gain_m", 0)
    curv_score = trill_metadata.get("curv_score", 0)
    dyn_score = trill_metadata.get("dyn_score", 0)
    turny = trill_metadata.get("turny", False)
    spirited = trill_metadata.get("spirited", False)
    
    # Build context
    location = f"near {place}, {state}" if place and state else state if state else "the open road"
    scene = f"{protected_name} (verified protected lands)" if near_protected and protected_name else "public lands" if near_protected else "backroads"
    
    # User preferences
    prefs = user_prefs or {}
    model = prefs.get("trill_openai_model", "gpt-4o-mini")
    
    # Build prompt
    user_prompt = f"""Generate viral social media content for a driving video with these metrics:

TRILL SCORE: {score}/100 (higher = more thrilling)
SPEED BUCKET: {bucket}
LOCATION: {location}
SCENE: {scene}
ELEVATION GAIN: {elev_gain}m
CURVATURE: {curv_score}/10 (higher = more twisty/switchbacks)
DYNAMICS: {dyn_score}/10 (higher = more spirited cornering)
MOTION FLAGS: {"Turny roads" if turny else ""} {"Spirited driving" if spirited else ""}

GENERATE:
1. TITLE (max 80 chars)
   - Create mystery/curiosity gap
   - Use 1-2 emojis strategically
   - Make it stop-the-scroll worthy
   - Examples: "This road changed my perspective 🔥" or "POV: You find the perfect line ⚡"
   
2. CAPTION (max 200 chars)
   - First-person, conversational
   - Create FOMO/aspiration
   - Ask a question or prompt engagement
   - 2-3 emojis max
   
3. HASHTAGS (exactly 15 tags)
   - 3-4 mega viral: #fyp #foryou #viral #trending
   - 4-5 niche community: #roadtrip #driving #explore
   - 3-4 location: #{state} #{place} (if available)
   - 2-3 motion: #curvyroads #spiriteddrive (if applicable)
   - 1-2 protected lands: #publiclands #nationalpark (ONLY if verified: {near_protected})

RETURN ONLY THIS JSON (no markdown, no backticks):
{{
  "title": "your title here",
  "caption": "your caption here",
  "hashtags": ["tag1", "tag2", ...]
}}
"""
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRILL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.85,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Parse response
        result = json.loads(content)
        
        # Validate and clean hashtags
        hashtags = result.get("hashtags", [])
        hashtags = [h.lower().replace("#", "").replace(" ", "") for h in hashtags]
        hashtags = [h for h in hashtags if h and len(h) <= 30][:15]
        
        return {
            "title": result.get("title", "")[:100],
            "caption": result.get("caption", "")[:250],
            "hashtags": hashtags,
            "tokens_used": tokens_used,
            "model": model,
            "trill_score": score,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        return {
            "title": trill_metadata.get("title", ""),
            "caption": trill_metadata.get("caption", ""),
            "hashtags": trill_metadata.get("hashtags", []),
            "tokens_used": 0,
            "model": "fallback",
            "error": str(e)
        }

async def seed_trill_places(conn):
    """Seed database with popular driving locations"""
    places = [
        {"name": "Moab", "state": "UT", "lat": 38.5733, "lon": -109.5498, "popularity": 95, "protected_name": "Arches National Park"},
        {"name": "Zion", "state": "UT", "lat": 37.2982, "lon": -113.0263, "popularity": 90, "protected_name": "Zion National Park"},
        {"name": "Big Sur", "state": "CA", "lat": 36.2704, "lon": -121.8081, "popularity": 92, "protected_name": "Los Padres National Forest"},
        {"name": "Malibu", "state": "CA", "lat": 34.0259, "lon": -118.7798, "popularity": 85},
        {"name": "Yosemite", "state": "CA", "lat": 37.8651, "lon": -119.5383, "popularity": 88, "protected_name": "Yosemite National Park"},
        {"name": "Rocky Mountain NP", "state": "CO", "lat": 40.3428, "lon": -105.6836, "popularity": 87, "protected_name": "Rocky Mountain National Park"},
        {"name": "Sedona", "state": "AZ", "lat": 34.8697, "lon": -111.7610, "popularity": 88, "protected_name": "Coconino National Forest"},
        {"name": "Grand Canyon", "state": "AZ", "lat": 36.1069, "lon": -112.1129, "popularity": 95, "protected_name": "Grand Canyon National Park"},
        {"name": "Glacier National Park", "state": "MT", "lat": 48.7596, "lon": -113.7870, "popularity": 85, "protected_name": "Glacier National Park"},
        {"name": "Yellowstone", "state": "WY", "lat": 44.4280, "lon": -110.5885, "popularity": 92, "protected_name": "Yellowstone National Park"},
        {"name": "Blue Ridge Parkway", "state": "NC", "lat": 35.5951, "lon": -82.5515, "popularity": 82, "protected_name": "Blue Ridge Parkway"},
        {"name": "Tail of the Dragon", "state": "NC", "lat": 35.5159, "lon": -83.9293, "popularity": 90},
    ]
    
    for p in places:
        hashtags = [p["name"].lower().replace(" ", ""), f"{p['state'].lower()}roadtrip"]
        if p.get("protected_name"):
            hashtags.extend(["publiclands", "nationalpark"])
        
        try:
            await conn.execute("""
                INSERT INTO trill_places (name, state, lat, lon, popularity_score, is_protected, protected_name, hashtags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (name, state) DO UPDATE SET
                    popularity_score = EXCLUDED.popularity_score,
                    is_protected = EXCLUDED.is_protected,
                    protected_name = EXCLUDED.protected_name,
                    hashtags = EXCLUDED.hashtags,
                    updated_at = NOW()
            """, p["name"], p["state"], p["lat"], p["lon"], p.get("popularity", 50), 
                 bool(p.get("protected_name")), p.get("protected_name"), hashtags)
        except Exception as e:
            logger.error(f"Failed to seed trill place {p['name']}: {e}")

async def get_nearby_trill_place(conn, lat: float, lon: float, max_distance_km: float = 50) -> Optional[dict]:
    """Find nearest popular trill place for geo-targeting"""
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    
    places = await conn.fetch("SELECT * FROM trill_places")
    
    best = None
    min_dist = float('inf')
    
    for p in places:
        dist = haversine(lat, lon, float(p["lat"]), float(p["lon"]))
        if dist < min_dist and dist <= max_distance_km:
            min_dist = dist
            best = {**dict(p), "distance_km": dist}
    
    return best

async def process_telemetry(conn, upload_id: str, user_id: str, video_path: str, map_path: str, user_prefs: dict) -> dict:
    """
    Process telemetry data and generate content.
    
    Returns:
        {
            "trill_metadata": {...},
            "ai_content": {...},
            "hud_path": "/path/to/hud.mp4" or None
        }
    """
    # Import trill module dynamically
    try:
        import telemetry_trill as tt
    except ImportError:
        raise HTTPException(503, "Telemetry processing not available - telemetry_trill.py not found")
    
    try:
        # Run trill analysis
        result = tt.safe_analyze_video(
            video_path,
            map_path,
            gaz_places_path=GAZETTEER_PLACES_PATH if os.path.exists(GAZETTEER_PLACES_PATH) else None,
            padus_path=PADUS_PATH if os.path.exists(PADUS_PATH) else None,
            padus_layer=PADUS_LAYER,
            hud_enabled=user_prefs.get("trill_hud_enabled", False)
        )
        
        if not result.get("ok"):
            raise Exception(result.get("error", "Analysis failed"))
        
        trill_data = result["data"]
        
        # Check if score meets minimum threshold
        trill_score = trill_data.get("trill_score", 0)
        min_score = user_prefs.get("trill_min_score", 60)
        
        # Enrich with nearby trill place if available
        mid_lat = trill_data.get("place_lat")
        mid_lon = trill_data.get("place_lon")
        if mid_lat and mid_lon:
            trill_place = await get_nearby_trill_place(conn, mid_lat, mid_lon)
            if trill_place:
                trill_data["trill_place"] = trill_place["name"]
                trill_data["trill_place_hashtags"] = trill_place.get("hashtags", [])
        
        # Generate AI content if enabled and score is high enough
        ai_content = None
        if user_prefs.get("trill_ai_enhance", True) and trill_score >= min_score:
            try:
                ai_content = generate_trill_content(trill_data, user_prefs)
            except Exception as e:
                logger.error(f"AI generation failed: {e}")
                ai_content = {
                    "title": trill_data.get("title"),
                    "caption": trill_data.get("caption"),
                    "hashtags": trill_data.get("hashtags", []),
                    "model": "trill_fallback"
                }
        
        # Generate HUD if enabled
        hud_path = None
        if user_prefs.get("trill_hud_enabled", False):
            try:
                hud_path = tt.ensure_hud_mp4(video_path, map_path)
            except Exception as e:
                logger.error(f"HUD generation failed: {e}")
        
        # Store in database
        await conn.execute("""
            UPDATE uploads SET 
                trill_score = $1,
                speed_bucket = $2,
                trill_metadata = $3,
                ai_generated_title = $4,
                ai_generated_caption = $5,
                ai_generated_hashtags = $6,
                updated_at = NOW()
            WHERE id = $7
        """, 
            trill_score,
            trill_data.get("speed_bucket"),
            json.dumps(trill_data),
            ai_content.get("title") if ai_content else None,
            ai_content.get("caption") if ai_content else None,
            ai_content.get("hashtags") if ai_content else None,
            upload_id
        )
        
        # Track OpenAI costs if used
        if ai_content and ai_content.get("tokens_used"):
            cost = ai_content["tokens_used"] * COST_PER_OPENAI_TOKEN
            await conn.execute("""
                INSERT INTO cost_tracking (user_id, category, operation, tokens, cost_usd)
                VALUES ($1, 'openai', 'trill_generation', $2, $3)
            """, user_id, ai_content["tokens_used"], cost)
        
        return {
            "trill_metadata": trill_data,
            "ai_content": ai_content,
            "hud_path": hud_path
        }
        
    except Exception as e:
        logger.error(f"Telemetry processing failed: {e}")
        raise HTTPException(500, f"Telemetry processing failed: {str(e)}")

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
        await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL", user["id"])
    return {"status": "logged_out"}

@app.post("/api/auth/logout-all")
async def logout_all(user: dict = Depends(get_current_user)):
    """Revoke all refresh tokens for current user (log out all devices)."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )
    return {"status": "logged_out_all"}



@app.post("/api/auth/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, background: BackgroundTasks):
    """Initiate password reset. Always returns OK to prevent account enumeration."""
    email = payload.email.lower()
    async with db_pool.acquire() as conn:
        user_row = await conn.fetchrow("SELECT id, email, status FROM users WHERE LOWER(email)=$1", email)
        if user_row and user_row["status"] != "disabled":
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

            # Invalidate prior unused tokens for this user
            await conn.execute(
                "UPDATE password_resets SET used_at = NOW() WHERE user_id=$1 AND used_at IS NULL",
                user_row["id"],
            )
            await conn.execute(
                "INSERT INTO password_resets (user_id, token_hash, expires_at) VALUES ($1,$2,$3)",
                user_row["id"], token_hash, expires_at
            )

            reset_link = f"{FRONTEND_URL.rstrip('/')}/reset-password?token={quote(token)}"
            html = f"""
                <p>You requested a password reset for UploadM8.</p>
                <p><a href="{reset_link}">Reset your password</a></p>
                <p>This link expires in 60 minutes. If you did not request this, ignore this email.</p>
            """
            background.add_task(send_email, user_row["email"], "Reset your UploadM8 password", html)

    return {"ok": True}

@app.post("/api/auth/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    async with db_pool.acquire() as conn:
        pr = await conn.fetchrow(
            """
            SELECT id, user_id, expires_at, used_at
            FROM password_resets
            WHERE token_hash=$1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            token_hash
        )
        if not pr or pr["used_at"] is not None:
            raise HTTPException(status_code=400, detail="Invalid or used reset token")
        if pr["expires_at"] < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Reset token expired")

        new_hash = hash_password(payload.new_password)

        await conn.execute(
            "UPDATE users SET password_hash=$1, updated_at=NOW() WHERE id=$2",
            new_hash, pr["user_id"]
        )
        await conn.execute("UPDATE password_resets SET used_at=NOW() WHERE id=$1", pr["id"])

        # Force logout across devices/sessions
        await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id=$1 AND revoked_at IS NULL", pr["user_id"])

    return {"ok": True}

# ============================================================
# User Profile & Wallet
# ============================================================
@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})
    role = user.get("role", "user")

    # Avatar: single source of truth = users.avatar_r2_key (private bucket -> signed URL)
    avatar_r2_key = user.get("avatar_r2_key")
    avatar_signed_url = None
    if avatar_r2_key:
        try:
            avatar_signed_url = generate_presigned_download_url(avatar_r2_key)
        except Exception as e:
            logger.warning(f"Failed to presign avatar for user {user.get('id')}: {e}")

    # Stabilization window: return both snake_case + camelCase keys
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name"),
        "role": role,
        "timezone": user.get("timezone") or "America/Chicago",

        # Avatar outputs (private, signed)
        "avatar_r2_key": avatar_r2_key,
        "avatar_url": avatar_signed_url,
        "avatarUrl": avatar_signed_url,
        "avatar_signed_url": avatar_signed_url,
        "avatarSignedUrl": avatar_signed_url,

        "subscription_tier":      user.get("subscription_tier", "free"),
        "subscription_status":     user.get("subscription_status"),
        "current_period_end":      user.get("current_period_end").isoformat() if user.get("current_period_end") else None,
        "trial_end":               user.get("trial_end").isoformat() if user.get("trial_end") else None,
        "stripe_subscription_id":  user.get("stripe_subscription_id"),
        "billing_mode":            BILLING_MODE,   # "test" | "live"
        "wallet": {
            "put_balance":  float(wallet.get("put_balance", 0.0) or 0.0),
            "aic_balance":  float(wallet.get("aic_balance", 0.0) or 0.0),
            "put_reserved": float(wallet.get("put_reserved", 0.0) or 0.0),
            "aic_reserved": float(wallet.get("aic_reserved", 0.0) or 0.0),
            "updated_at":   wallet.get("updated_at"),
        },
        "plan": plan,
        "features": {
            "uploads":     plan.get("uploads", False),
            "scheduler":   plan.get("scheduler", False),
            "analytics":   plan.get("analytics", False),
            "watermark":   plan.get("watermark", False),
            "white_label": plan.get("white_label", False),
            "support":     plan.get("support", False),
        },
        # Full entitlements dict — used by hasEntitlement() in app.js
        "entitlements": entitlements_to_dict(
            get_entitlements_from_user(dict(user))
        ),
    }


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    timezone: Optional[str] = None

@app.put("/api/me")
async def update_me(data: ProfileUpdate, user: dict = Depends(get_current_user)):
    """Update user profile"""
    updates, params = [], [user["id"]]
    if data.name:
        updates.append(f"name = ${len(params)+1}")
        params.append(data.name)
    if data.timezone:
        updates.append(f"timezone = ${len(params)+1}")
        params.append(data.timezone)
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
    return {"status": "updated"}

@app.post("/api/auth/change-password")
async def change_password(data: PasswordChange, user: dict = Depends(get_current_user)):
    """Change user password"""
    async with db_pool.acquire() as conn:
        # Verify current password
        user_row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")
        
        # Update to new password
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, user["id"])
        
        # Optionally invalidate other sessions (refresh tokens)
        await conn.execute("DELETE FROM refresh_tokens WHERE user_id = $1", user["id"])
    
    logger.info(f"Password changed for user {user['id']}")
    return {"status": "password_changed"}

# ============================================================
# Settings Endpoints
# ============================================================
@app.put("/api/settings/profile")
async def update_profile_settings(data: ProfileUpdateSettings, user: dict = Depends(get_current_user)):
    """Update user profile (first name, last name)"""
    updates, params = [], [user["id"]]
    
    if data.first_name is not None:
        updates.append(f"first_name = ${len(params)+1}")
        params.append(data.first_name.strip())
    
    if data.last_name is not None:
        updates.append(f"last_name = ${len(params)+1}")
        params.append(data.last_name.strip())
    
    # Also update the combined name field for backwards compatibility
    if data.first_name is not None or data.last_name is not None:
        first = data.first_name.strip() if data.first_name else user.get("first_name", "")
        last = data.last_name.strip() if data.last_name else user.get("last_name", "")
        full_name = f"{first} {last}".strip() or user.get("name", "User")
        updates.append(f"name = ${len(params)+1}")
        params.append(full_name)
    
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(
                f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", 
                *params
            )
        logger.info(f"Profile updated for user {user['id']}")
        return {"status": "success", "message": "Profile updated successfully"}
    
    return {"status": "success", "message": "No changes made"}

@app.put("/api/settings/preferences/legacy")
async def update_preferences_legacy(data: PreferencesUpdate, user: dict = Depends(get_current_user)):
    """Update user preferences (notifications, theme, hashtags, etc.)"""
    async with db_pool.acquire() as conn:
        # Ensure user_settings row exists
        await conn.execute(
            "INSERT INTO user_settings (user_id, preferences_json) VALUES ($1, '{}') ON CONFLICT (user_id) DO NOTHING",
            user["id"]
        )
        
        # Get current preferences
        current_prefs = await conn.fetchval(
            "SELECT preferences_json FROM user_settings WHERE user_id = $1",
            user["id"]
        )
        
        # Parse current preferences
        prefs = current_prefs if current_prefs else {}
        if isinstance(prefs, str):
            prefs = json.loads(prefs)
        
        # Update with new values (only update fields that are provided)
        if data.emailNotifs is not None:
            prefs["emailNotifs"] = data.emailNotifs
        if data.uploadCompleteNotifs is not None:
            prefs["uploadCompleteNotifs"] = data.uploadCompleteNotifs
        if data.marketingEmails is not None:
            prefs["marketingEmails"] = data.marketingEmails
        if data.theme is not None:
            prefs["theme"] = data.theme
        if data.accentColor is not None:
            prefs["accentColor"] = data.accentColor
        if data.defaultPrivacy is not None:
            prefs["defaultPrivacy"] = data.defaultPrivacy
        if data.autoPublish is not None:
            prefs["autoPublish"] = data.autoPublish
        if data.alwaysHashtags is not None:
            prefs["alwaysHashtags"] = data.alwaysHashtags
        if data.blockedHashtags is not None:
            prefs["blockedHashtags"] = data.blockedHashtags
        if data.tiktokHashtags is not None:
            prefs["tiktokHashtags"] = data.tiktokHashtags
        if data.youtubeHashtags is not None:
            prefs["youtubeHashtags"] = data.youtubeHashtags
        if data.instagramHashtags is not None:
            prefs["instagramHashtags"] = data.instagramHashtags
        if data.facebookHashtags is not None:
            prefs["facebookHashtags"] = data.facebookHashtags
        if data.hashtagPosition is not None:
            prefs["hashtagPosition"] = data.hashtagPosition
        if data.maxHashtags is not None:
            prefs["maxHashtags"] = data.maxHashtags
        if data.aiHashtagsEnabled is not None:
            prefs["aiHashtagsEnabled"] = data.aiHashtagsEnabled
        if data.aiHashtagCount is not None:
            prefs["aiHashtagCount"] = data.aiHashtagCount
        if data.aiHashtagStyle is not None:
            prefs["aiHashtagStyle"] = data.aiHashtagStyle
        if data.platformHashtags is not None:
            prefs["platformHashtags"] = data.platformHashtags
        
        # Save back to database
        await conn.execute(
            "UPDATE user_settings SET preferences_json = $1, updated_at = NOW() WHERE user_id = $2",
            json.dumps(prefs),
            user["id"]
        )
    
    logger.info(f"Preferences updated for user {user['id']}")
    return {"status": "success", "message": "Preferences saved successfully", "preferences": prefs}

# (removed) obsolete /api/settings/preferences handler (used user_settings.preferences_json)

@app.put("/api/settings/password")
async def update_password_settings(data: PasswordChange, user: dict = Depends(get_current_user)):
    """Change user password (settings endpoint version)"""
    async with db_pool.acquire() as conn:
        # Verify current password
        user_row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")
        
        # Update to new password
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, user["id"])
        
        # Optionally invalidate other sessions (refresh tokens)
        await conn.execute("DELETE FROM refresh_tokens WHERE user_id = $1", user["id"])
    
    logger.info(f"Password changed via settings for user {user['id']}")
    return {"status": "success", "message": "Password changed successfully"}

@app.post("/api/settings/avatar")
async def upload_avatar(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Must be JPEG, PNG, GIF, or WebP")

        # Read file content
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")

        # Create unique filename
        ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "png"
        r2_key = f"avatars/{user['id']}/{uuid.uuid4()}.{ext}"

        # Upload to private R2 bucket
        s3 = get_s3_client()
        if not R2_BUCKET_NAME:
            raise HTTPException(status_code=500, detail="Missing R2_BUCKET_NAME")
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=r2_key,
            Body=content,
            ContentType=file.content_type,
        )

        # Store single source of truth in DB
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET avatar_r2_key = $1, updated_at = NOW() WHERE id = $2",
                r2_key,
                user["id"],
            )

        signed_url = r2_presign_get_url(r2_key)

        logger.info(f"Avatar uploaded for user {user['id']}: {r2_key}")
        return {"success": True, "r2_key": r2_key, "avatar_url": signed_url, "avatarUrl": signed_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Avatar upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar")



# ──────────────────────────────────────────────────────────────────────────────
# Platform OAuth token revocation helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _revoke_tiktok_token(access_token: str) -> bool:
    """Revoke a TikTok access token at the provider."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/oauth/revoke/",
                data={"client_key": TIKTOK_CLIENT_KEY, "client_secret": TIKTOK_CLIENT_SECRET, "token": access_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"TikTok token revoke failed: {e}")
        return False


async def _revoke_google_token(access_token: str) -> bool:
    """Revoke a Google/YouTube access token at the provider."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": access_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"Google token revoke failed: {e}")
        return False


async def _revoke_meta_token(access_token: str) -> bool:
    """Revoke a Facebook/Instagram access token at the provider."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(
                "https://graph.facebook.com/me/permissions",
                params={"access_token": access_token},
            )
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"Meta token revoke failed: {e}")
        return False


async def _revoke_platform_token(platform: str, token_blob: dict) -> tuple[bool, str]:
    """
    Attempt to revoke `token_blob` at the platform.
    Returns (success: bool, error_msg: str).
    """
    try:
        tok = token_blob if isinstance(token_blob, dict) else {}
        if isinstance(tok, str):
            try:
                tok = json.loads(tok)
            except Exception:
                tok = {}
        # Encrypted blobs: decrypt first
        if "kid" in tok and "ciphertext" in tok:
            try:
                tok = decrypt_blob(tok)
            except Exception:
                return False, "blob-decrypt-failed"
        access_token = tok.get("access_token", "")
        if not access_token:
            return False, "no-access-token"

        if platform == "youtube":
            ok = await _revoke_google_token(access_token)
        elif platform in ("facebook", "instagram"):
            ok = await _revoke_meta_token(access_token)
        elif platform == "tiktok":
            ok = await _revoke_tiktok_token(access_token)
        else:
            return False, f"unsupported-platform:{platform}"

        return ok, ("" if ok else "provider-rejected")
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# R2 bulk-delete helper
# ──────────────────────────────────────────────────────────────────────────────

async def _delete_r2_objects(keys: list[str]) -> int:
    """
    Delete a list of R2 object keys.  Runs in a thread-pool executor so it
    doesn't block the event loop.  Returns the number of objects deleted.
    """
    if not keys or not R2_BUCKET_NAME:
        return 0
    import asyncio
    loop = asyncio.get_event_loop()

    def _bulk_delete(chunk):
        s3 = get_s3_client()
        objects = [{"Key": _normalize_r2_key(k)} for k in chunk if k]
        if not objects:
            return 0
        resp = s3.delete_objects(Bucket=R2_BUCKET_NAME, Delete={"Objects": objects, "Quiet": True})
        errors = resp.get("Errors", [])
        if errors:
            logger.warning(f"R2 delete_objects errors: {errors}")
        return len(objects) - len(errors)

    deleted = 0
    # S3 delete_objects accepts up to 1 000 keys per call
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        deleted += await loop.run_in_executor(None, _bulk_delete, chunk)
    return deleted


# ──────────────────────────────────────────────────────────────────────────────
# Self-serve account deletion  DELETE /api/me
# ──────────────────────────────────────────────────────────────────────────────

@app.delete("/api/me")
async def delete_account(request: Request, user: dict = Depends(get_current_user)):
    """
    Self-serve account deletion.  Proves it deletes:
      • all platform OAuth tokens (revoked at provider first)
      • all R2 upload artifacts (video, thumbnail, processed, telemetry)
      • all personal data rows (users, uploads, settings, wallets, ledger …)
      • Stripe subscription (cancelled immediately if active)
    Writes a tamper-evident audit record BEFORE touching any data so even a
    mid-flight crash leaves an evidence trail.
    """
    user_id = str(user["id"])
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)

    if user.get("role") == "master_admin":
        raise HTTPException(403, "Master admin accounts cannot be deleted via this endpoint.")

    async with db_pool.acquire() as conn:
        # ── 1. Write deletion REQUEST record immediately ──────────────────
        deletion_log_id = await conn.fetchval(
            """
            INSERT INTO account_deletion_log
                (user_id, user_email, user_name, initiated_by, ip_address)
            VALUES ($1, $2, $3, 'self', $4)
            RETURNING id
            """,
            user_id,
            user.get("email", ""),
            user.get("name", ""),
            ip_addr,
        )

        # ── 2. Collect all R2 keys for this user ─────────────────────────
        r2_rows = await conn.fetch(
            """
            SELECT r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key
            FROM uploads WHERE user_id = $1
            """,
            user["id"],
        )
        r2_keys = []
        for row in r2_rows:
            for col in ("r2_key", "telemetry_r2_key", "processed_r2_key", "thumbnail_r2_key"):
                v = row[col] if col in row.keys() else None
                if v:
                    r2_keys.append(v)
        # Also pick up avatar if stored in R2
        avatar_key = user.get("avatar_r2_key") or ""
        if avatar_key:
            r2_keys.append(avatar_key)

        # ── 3. Collect & revoke platform tokens ──────────────────────────
        token_rows = await conn.fetch(
            "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE user_id = $1",
            user["id"],
        )
        tokens_revoked = 0
        for trow in token_rows:
            ok, err = await _revoke_platform_token(trow["platform"], trow["token_blob"])
            if ok:
                tokens_revoked += 1
            # Log each disconnect individually
            await conn.execute(
                """
                INSERT INTO platform_disconnect_log
                    (user_id, platform, account_id, account_name,
                     revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
                VALUES ($1,$2,$3,$4,$5,$6,'account_deletion',$7)
                """,
                user_id,
                trow["platform"],
                trow["account_id"],
                trow["account_name"],
                ok,
                err or None,
                ip_addr,
            )

        # ── 4. Cancel Stripe subscription ────────────────────────────────
        stripe_cancelled = False
        stripe_sub_id = user.get("stripe_subscription_id")
        if stripe_sub_id and STRIPE_SECRET_KEY:
            try:
                stripe.Subscription.cancel(stripe_sub_id)
                stripe_cancelled = True
            except Exception as e:
                logger.warning(f"Stripe cancel failed for {user_id}: {e}")

        # ── 5. Count rows for the audit manifest ─────────────────────────
        rows_deleted = {}
        for tbl in ("uploads", "platform_tokens", "token_ledger", "wallets",
                    "user_settings", "user_preferences", "refresh_tokens",
                    "user_color_preferences", "account_groups",
                    "white_label_settings"):
            try:
                n = await conn.fetchval(f"SELECT COUNT(*) FROM {tbl} WHERE user_id = $1", user["id"])
                rows_deleted[tbl] = int(n)
            except Exception:
                pass
        rows_deleted["users"] = 1

        # ── 6. Delete DB rows (cascade-safe ordering) ────────────────────
        await conn.execute("DELETE FROM refresh_tokens          WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM token_ledger             WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM wallets                  WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM user_settings            WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM user_preferences         WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM platform_tokens          WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM user_color_preferences   WHERE user_id = $1", user["id"])
        await conn.execute("DELETE FROM account_groups           WHERE user_id = $1", user["id"])
        try:
            await conn.execute("DELETE FROM white_label_settings WHERE user_id = $1", user["id"])
        except Exception:
            pass
        await conn.execute("DELETE FROM uploads                  WHERE user_id = $1", user["id"])
        # Pseudonymize support_messages (keep subject/message for audit, strip PII)
        try:
            await conn.execute(
                "UPDATE support_messages SET name = '[deleted]', email = '[deleted]' WHERE user_id = $1",
                user["id"],
            )
        except Exception:
            pass
        await conn.execute("DELETE FROM users WHERE id = $1", user["id"])

        # ── 7. Delete R2 objects (non-blocking; best-effort) ─────────────
        r2_deleted = await _delete_r2_objects(r2_keys)

        # ── 8. Mark audit record as COMPLETED ────────────────────────────
        await conn.execute(
            """
            UPDATE account_deletion_log
            SET completed_at   = NOW(),
                r2_keys_deleted  = $2,
                tokens_revoked   = $3,
                stripe_cancelled = $4,
                rows_deleted     = $5
            WHERE id = $1
            """,
            deletion_log_id,
            r2_deleted,
            tokens_revoked,
            stripe_cancelled,
            json.dumps(rows_deleted),
        )

    logger.info(
        f"[DELETION COMPLETE] user={user_id} "
        f"r2={r2_deleted} tokens_revoked={tokens_revoked} "
        f"stripe_cancelled={stripe_cancelled}"
    )
    return {
        "status": "account_deleted",
        "summary": {
            "r2_objects_deleted": r2_deleted,
            "platform_tokens_revoked": tokens_revoked,
            "stripe_subscription_cancelled": stripe_cancelled,
            "rows_deleted": rows_deleted,
        },
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
    """Get user settings including Trill preferences"""
    async with db_pool.acquire() as conn:
        settings = await conn.fetchrow("""
            SELECT 
                discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                speeding_mph, euphoria_mph, hud_speed_unit, hud_color,
                hud_font_family, hud_font_size, ffmpeg_screenshot_interval,
                auto_generate_thumbnails, auto_generate_captions,
                auto_generate_hashtags, default_hashtag_count, always_use_hashtags
            FROM user_settings 
            WHERE user_id = $1
        """, user["id"])

        if not settings:
            # Return defaults
            return {
                "discord_webhook": None,
                "telemetry_enabled": True,
                "hud_enabled": True,
                "hud_position": "bottom-left",
                "speeding_mph": 80,        # Default Trill threshold
                "euphoria_mph": 100,       # Default Trill threshold
                "hud_speed_unit": "mph",
                "hud_color": "#FFFFFF",
                "hud_font_family": "Arial",
                "hud_font_size": 24,
                "ffmpeg_screenshot_interval": 5,
                "auto_generate_thumbnails": True,
                "auto_generate_captions": True,
                "auto_generate_hashtags": True,
                "default_hashtag_count": 5,
                "always_use_hashtags": False
            }

    return dict(settings)

@app.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    """Update user settings including Trill thresholds"""
    updates, params = [], [user["id"]]

    # All possible settings fields
    fields = [
        "discord_webhook", "telemetry_enabled", "hud_enabled", 
        "hud_position", "speeding_mph", "euphoria_mph",
        "hud_speed_unit", "hud_color", "hud_font_family", "hud_font_size",
        "ffmpeg_screenshot_interval", "auto_generate_thumbnails",
        "auto_generate_captions", "auto_generate_hashtags",
        "default_hashtag_count", "always_use_hashtags"
    ]

    for field in fields:
        val = getattr(data, field, None)
        if val is not None:
            updates.append(f"{field} = ${len(params)+1}")
            params.append(val)

    if updates:
        async with db_pool.acquire() as conn:
            # Create user_settings row if doesn't exist
            await conn.execute("""
                INSERT INTO user_settings (user_id) 
                VALUES ($1) 
                ON CONFLICT (user_id) DO NOTHING
            """, user["id"])

            # Update settings
            await conn.execute(
                f"UPDATE user_settings SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = $1",
                *params
            )

            logger.info(f"Updated settings for user {user['id']}: {updates}")

    return {"status": "updated"}

@app.get("/api/me/preferences")
async def get_preferences(user: dict = Depends(get_current_user)):
    """Get user preferences including hashtag settings"""
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
    if prefs and prefs["preferences"]:
        return json.loads(prefs["preferences"]) if isinstance(prefs["preferences"], str) else prefs["preferences"]
    return {}

@app.put("/api/me/preferences")
async def update_preferences(request: Request, user: dict = Depends(get_current_user)):
    """Update user preferences including hashtag settings"""
    prefs = await request.json()
    
    # Validate and sanitize hashtag data
    if "alwaysHashtags" in prefs:
        prefs["alwaysHashtags"] = [str(h).lower().strip()[:50] for h in prefs["alwaysHashtags"][:100]]
    if "blockedHashtags" in prefs:
        prefs["blockedHashtags"] = [str(h).lower().strip()[:50] for h in prefs["blockedHashtags"][:100]]
    if "platformHashtags" in prefs:
        for platform in ["tiktok", "youtube", "instagram", "facebook"]:
            if platform in prefs["platformHashtags"]:
                prefs["platformHashtags"][platform] = [str(h).lower().strip()[:50] for h in prefs["platformHashtags"][platform][:50]]
    
    # Validate numeric hashtag settings
    if "maxHashtags" in prefs:
        prefs["maxHashtags"] = max(1, min(50, int(prefs["maxHashtags"])))
    if "aiHashtagCount" in prefs:
        prefs["aiHashtagCount"] = max(1, min(30, int(prefs["aiHashtagCount"])))
    
    # Validate hashtag position
    if "hashtagPosition" in prefs and prefs["hashtagPosition"] not in ["start", "end", "caption", "comment"]:
        prefs["hashtagPosition"] = "end"
    
    # Validate AI hashtag style
    if "aiHashtagStyle" in prefs and prefs["aiHashtagStyle"] not in ["lowercase", "capitalized", "camelcase", "mixed"]:
        prefs["aiHashtagStyle"] = "mixed"
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET preferences = $1, updated_at = NOW() WHERE id = $2",
            json.dumps(prefs), user["id"]
        )
    return {"status": "updated"}

# ============================================================
# Uploads
# ============================================================
@app.post("/api/uploads/presign")
async def presign_upload(data: UploadInit, request: Request, user: dict = Depends(get_current_user)):
    """Create upload with user preferences applied"""
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})

    # Normalize optional fields coming from the client
    if getattr(data, "hashtags", None) is None:
        data.hashtags = []
    if getattr(data, "platforms", None) is None:
        data.platforms = []

    async with db_pool.acquire() as conn:
        # Fetch user preferences to apply defaults
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])

        # Apply preference defaults if user didn't specify
        if not getattr(data, "privacy", None):
            data.privacy = user_prefs["default_privacy"]

        # --- Hashtag assembly ---
        # Step 1: Always merge always_hashtags + platform-specific hashtags.
        #         These are manual user settings — work on every plan, no AI flag needed.
        combined = list(data.hashtags) + list(user_prefs.get("always_hashtags", []) or [])

        platform_hashtags = user_prefs.get("platform_hashtags") or {}
        for platform in data.platforms:
            if isinstance(platform_hashtags, dict):
                tags = platform_hashtags.get(platform) or platform_hashtags.get(platform.lower()) or []
                if tags:
                    combined.extend(tags)

        # Step 2: AI-generated hashtag injection — gated on plan + user toggle.
        #         Actual AI generation happens later in caption_stage.
        if user_prefs.get("ai_hashtags_enabled") and plan.get("ai"):
            pass  # ai_hashtags merged into ctx later by caption_stage

        # Step 3: Deduplicate, strip blocked, enforce limit — always runs.
        blocked = set(user_prefs.get("blocked_hashtags", []) or [])
        combined = [h for h in combined if h and h not in blocked]
        data.hashtags = list(dict.fromkeys(combined))[: int(user_prefs.get("max_hashtags", 30))]

        # ── Compute PUT/AIC cost — canonical formula from entitlements ──
        ent_cost = get_entitlements_for_tier(user.get("subscription_tier", "free"))
        use_ai  = bool(getattr(data, "use_ai", False)) and ent_cost.can_ai
        use_hud = bool(user_prefs.get("hud_enabled", False)) and ent_cost.can_burn_hud

        put_cost, aic_cost = compute_upload_cost(
            entitlements=ent_cost,
            num_platforms=len(data.platforms),
            use_ai=use_ai,
            use_hud=use_hud,
            num_thumbnails=ent_cost.max_thumbnails,
        )

        # Check balance (available = balance - reserved holds)
        put_avail = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
        aic_avail = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)

        if put_avail < put_cost:
            raise HTTPException(429, {
                "code": "insufficient_put",
                "message": f"Insufficient PUT tokens ({put_avail} available, {put_cost} needed).",
                "topup_url": "/settings.html#billing",
            })
        if aic_cost > 0 and aic_avail < aic_cost:
            raise HTTPException(429, {
                "code": "insufficient_aic",
                "message": f"Insufficient AIC credits ({aic_avail} available, {aic_cost} needed).",
                "topup_url": "/settings.html#billing",
            })

        # ── Queue depth guard ─────────────────────────────────────────
        pending_count = await conn.fetchval(
            """SELECT COUNT(*) FROM uploads
               WHERE user_id = $1
               AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
            user["id"]
        )
        ent_check = get_entitlements_for_tier(user.get("subscription_tier", "free"))
        if pending_count >= ent_check.queue_depth:
            raise HTTPException(
                429,
                f"Queue limit reached ({pending_count}/{ent_check.queue_depth} uploads pending). "
                "Wait for existing uploads to complete or upgrade your plan."
            )

        upload_id = str(uuid.uuid4())
        r2_key = f"uploads/{user['id']}/{upload_id}/{data.filename}"

        # Smart scheduling logic
        smart_schedule = None
        if getattr(data, "schedule_mode", None) == "smart":
            smart_schedule = calculate_smart_schedule(
                data.platforms,
                num_days=getattr(data, "smart_schedule_days", 7)
            )
            existing_days = await get_existing_scheduled_days(conn, user["id"], getattr(data, "smart_schedule_days", 7))
            if existing_days:
                smart_schedule = calculate_smart_schedule(
                    data.platforms,
                    num_days=getattr(data, "smart_schedule_days", 7)
                )

        scheduled_time = getattr(data, "scheduled_time", None)
        schedule_metadata = None

        if getattr(data, "schedule_mode", None) == "smart" and smart_schedule:
            schedule_metadata = {p: dt.isoformat() for p, dt in smart_schedule.items()}
            scheduled_time = min(smart_schedule.values())

        # Store upload with preferences metadata
        await conn.execute("""
            INSERT INTO uploads (
                id, user_id, r2_key, filename, file_size, platforms,
                title, caption, hashtags, privacy, status, scheduled_time,
                schedule_mode, put_reserved, aic_reserved, schedule_metadata,
                user_preferences
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'pending', $11, $12, $13, $14, $15, $16)
        """,
            upload_id, user["id"], r2_key, data.filename, data.file_size,
            data.platforms, data.title, data.caption, data.hashtags,
            data.privacy, scheduled_time, data.schedule_mode, put_cost,
            aic_cost, json.dumps(schedule_metadata) if schedule_metadata else None,
            json.dumps(user_prefs)
        )

        # Reserve tokens
        await reserve_tokens(conn, user["id"], put_cost, aic_cost, upload_id)

    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    result = {
        "upload_id": upload_id,
        "presigned_url": presigned_url,
        "r2_key": r2_key,
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "schedule_mode": data.schedule_mode,
        "preferences_applied": {
            "auto_captions": bool(user_prefs.get("auto_captions")),
            "auto_thumbnails": bool(user_prefs.get("auto_thumbnails")),
            "ai_hashtags": bool(user_prefs.get("ai_hashtags_enabled"))
        }
    }

    if smart_schedule:
        result["smart_schedule"] = {p: dt.isoformat() for p, dt in smart_schedule.items()}

    if getattr(data, "has_telemetry", False):
        telem_key = f"uploads/{user['id']}/{upload_id}/telemetry.map"
        result["telemetry_presigned_url"] = generate_presigned_upload_url(telem_key, "application/octet-stream")
        result["telemetry_r2_key"] = telem_key

    # Fire-and-forget audit — does not affect upload flow
    async with db_pool.acquire() as _ac:
        await log_system_event(_ac, user_id=str(user["id"]), action="UPLOAD_INITIATED",
                               event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                               details={"filename": data.filename, "platforms": list(data.platforms or []),
                                        "schedule_mode": data.schedule_mode, "put_cost": put_cost,
                                        "aic_cost": aic_cost, "file_size": data.file_size},
                               request=request)

    return result


@app.post("/api/uploads/smart-schedule/preview")
async def preview_smart_schedule(platforms: List[str] = Query(...), days: int = Query(7), user: dict = Depends(get_current_user)):
    """Preview what the smart schedule would look like for given platforms"""
    if not platforms:
        raise HTTPException(400, "At least one platform required")
    
    schedule = calculate_smart_schedule(platforms, num_days=days)
    
    return {
        "schedule": {p: dt.isoformat() for p, dt in schedule.items()},
        "explanation": {
            p: {
                "date": dt.strftime("%A, %B %d"),
                "time": dt.strftime("%I:%M %p"),
                "reason": f"Optimal posting time for {p.title()}"
            }
            for p, dt in schedule.items()
        }
    }

@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    """
    Complete upload and either enqueue immediately (immediate mode) or stage
    for deferred processing (scheduled / smart mode).

    IMMEDIATE  → status=queued, pushed to Redis → worker fires NOW
    SCHEDULED  → status=staged, NOT pushed to Redis → scheduler fires at scheduled_time - processing_window
    SMART      → status=staged, NOT pushed to Redis → scheduler fires at first scheduled_time - processing_window
    """
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        # Fetch preferences (fresher than what presign stored)
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])

        schedule_mode = upload.get("schedule_mode") or "immediate"

        # ── Determine status and whether to enqueue ──────────────────
        if schedule_mode in ("scheduled", "smart"):
            # DO NOT touch the Redis queue — scheduler loop handles this.
            # Mark as 'staged' so the scheduler knows files are ready.
            new_status = "staged"
            await conn.execute(
                "UPDATE uploads SET status = 'staged', updated_at = NOW() WHERE id = $1",
                upload_id
            )
        else:
            # Immediate publish — enqueue to Redis right now.
            new_status = "queued"
            await conn.execute(
                "UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1",
                upload_id
            )

    # Resolve full entitlements — drives queue routing, AI depth, priority class
    ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))

    if schedule_mode not in ("scheduled", "smart"):
        job_data = {
            "upload_id": upload_id,
            "user_id": str(user["id"]),
            "preferences": user_prefs,
            "plan_features": {
                "ai":           ent.can_ai,
                "priority":     ent.can_priority,
                "watermark":    ent.can_watermark,
                "ai_depth":     ent.ai_depth,
                "caption_frames": ent.max_caption_frames,
            },
            "priority_class": ent.priority_class,
        }
        await enqueue_job(job_data, lane="process", priority_class=ent.priority_class)

    # Compute scheduled_time display for smart schedules
    schedule_metadata = upload.get("schedule_metadata")
    smart_schedule_display = None
    if schedule_mode == "smart" and schedule_metadata:
        try:
            sm = schedule_metadata if isinstance(schedule_metadata, dict) else json.loads(schedule_metadata)
            smart_schedule_display = {p: v for p, v in sm.items()}
        except Exception:
            pass

    # Audit: upload submitted to pipeline
    await log_system_event(user_id=str(user["id"]), action="UPLOAD_SUBMITTED",
                           event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                           details={"schedule_mode": schedule_mode, "new_status": new_status,
                                    "platforms": list(upload.get("platforms") or [])},
                           request=request)

    return {
        "status": new_status,
        "upload_id": upload_id,
        "schedule_mode": schedule_mode,
        "scheduled_time": upload["scheduled_time"].isoformat() if upload.get("scheduled_time") else None,
        "smart_schedule": smart_schedule_display,
        "processing_features": {
            # `plan` is not defined in this scope — use ent (resolved above)
            "auto_captions":  bool(user_prefs.get("auto_captions"))        if ent.can_ai else False,
            "auto_thumbnails": bool(user_prefs.get("auto_thumbnails"))     if ent.can_ai else False,
            "ai_hashtags":    bool(user_prefs.get("ai_hashtags_enabled"))  if ent.can_ai else False,
        }
    }


@app.post("/api/uploads/{upload_id}/reprepare")
async def reprepare_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Generate a fresh presigned R2 URL for an upload stuck in pending state.
    Used when the browser refreshed mid-transfer before /complete was called.
    The DB record exists — we just issue new PUT URLs so the client can retry.
    """
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT id, r2_key, filename, status, telemetry_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        if upload["status"] not in ("pending",):
            raise HTTPException(400, f"Upload is not resumable (status: {upload['status']}). Use /retry for failed uploads.")

    r2_key = upload["r2_key"]
    filename = upload["filename"] or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    ct_map = {"mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo", "webm": "video/webm"}
    content_type = ct_map.get(ext, "video/mp4")

    result = {
        "upload_id": upload_id,
        "presigned_url": generate_presigned_upload_url(r2_key, content_type),
        "r2_key": r2_key,
        "filename": filename,
        "status": upload["status"],
    }
    if upload["telemetry_r2_key"]:
        result["telemetry_presigned_url"] = generate_presigned_upload_url(
            upload["telemetry_r2_key"], "application/octet-stream"
        )
        result["telemetry_r2_key"] = upload["telemetry_r2_key"]

    return result



@app.post("/api/uploads/{upload_id}/cancel")
async def cancel_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT put_reserved, aic_reserved, status FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("completed", "succeeded", "cancelled", "failed"):
            raise HTTPException(400, "Cannot cancel this upload")

        current_status = upload["status"]

        if current_status == "processing":
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCEL_REQUESTED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request)
            return {"status": "cancel_requested", "message": "Cancel signal sent — job will stop at next checkpoint"}
        else:
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, status = 'cancelled', updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCELLED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request, severity="WARNING")
            return {"status": "cancelled"}


@app.get("/api/uploads")
async def get_uploads(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    trill_only: bool = False,
    meta: bool = False,
    user: dict = Depends(get_current_user),
):
    """
    Upload queue list for current user.

    Contract (frontend-safe):
      - thumbnail_url: presigned R2 URL (if thumbnail_r2_key exists)
      - platform_results: always a list (even if DB stored dict)
      - hashtags: always list[str]
      - title/caption: falls back to AI-generated values if base fields are empty
      - ai_title/ai_caption/ai_hashtags: explicit AI fields (may be empty)
      - views/likes: totals from uploads table (platform-specific lives in platform_results if present)
    """
    cols = await _load_uploads_columns(db_pool)

    wanted = [
        "id","filename","platforms","status","privacy",
        "title","caption","hashtags",
        "scheduled_time","created_at","completed_at",
        "put_reserved","aic_reserved",
        "error_code","error_detail",
        "thumbnail_r2_key","platform_results","file_size",
        "processing_started_at","processing_finished_at",
        "processing_stage","processing_progress",
        "views","likes","comments","shares",
        # Schedule fields — required by dashboard to distinguish smart vs manual
        "schedule_mode","schedule_metadata",
        # Direct video URL (populated by worker after successful upload)
        "video_url",
        # AI fields (older/newer schema variants)
        "ai_title","ai_caption",
        "ai_generated_title","ai_generated_caption","ai_generated_hashtags",
    ]
    select_cols = _pick_cols(wanted, cols) or ["id","filename","platforms","status","created_at"]
    select_sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE user_id = $1"
    params = [user["id"]]

    if status:
        params.append(status)
        select_sql += f" AND status = ${len(params)}"
    if trill_only and "trill_score" in cols:
        select_sql += " AND trill_score IS NOT NULL"

    params.extend([limit, offset])
    select_sql += f" ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}"

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(select_sql, *params)

        total = None
        if meta:
            # total count uses same filters
            count_sql = "SELECT COUNT(*) FROM uploads WHERE user_id = $1"
            count_params = [user["id"]]
            if status:
                count_params.append(status)
                count_sql += f" AND status = ${len(count_params)}"
            if trill_only and "trill_score" in cols:
                count_sql += " AND trill_score IS NOT NULL"
            total = await conn.fetchval(count_sql, *count_params)

    def _normalize_platform_results(raw):
        """
        DB may store:
          - list: [{platform:..., status:..., url:...}, ...]
          - dict: {"tiktok": {...}, "youtube": {...}}
        Always return list[dict].

        Field aliasing (frontend-safe):
          platform_video_id → video_id   (canonical worker field)
          platform_url      → url        (canonical worker field)
        Both original fields are kept so nothing breaks.
        """
        pr = _safe_json(raw, [])
        if isinstance(pr, list):
            items = [x for x in pr if isinstance(x, dict)]
        elif isinstance(pr, dict):
            items = []
            for k, v in pr.items():
                if isinstance(v, dict):
                    items.append({"platform": k, **v})
                else:
                    items.append({"platform": k, "value": v})
        else:
            return []

        # Alias platform_video_id → video_id and platform_url → url
        # so buildPlatformUrl() in the frontend finds them with its
        # existing entry.video_id / entry.url lookups.
        out = []
        for item in items:
            d = dict(item)
            if d.get("platform_video_id") and not d.get("video_id"):
                d["video_id"] = d["platform_video_id"]
            if d.get("platform_url") and not d.get("url"):
                d["url"] = d["platform_url"]
            out.append(d)
        return out

    def _normalize_hashtags(raw):
        tags = _safe_json(raw, [])
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        if isinstance(tags, str) and tags.strip():
            # If it's a single tag string, wrap it.
            return [tags.strip()]
        return []

    s3 = None
    out = []
    for r in rows:
        d = dict(r)

        # AI fields (explicit)
        ai_title = (d.get("ai_title") or d.get("ai_generated_title") or "") or ""
        ai_caption = (d.get("ai_caption") or d.get("ai_generated_caption") or "") or ""
        ai_hashtags = _normalize_hashtags(d.get("ai_generated_hashtags"))

        # Base fields with fallback to AI if empty
        title = (d.get("title") or "").strip() or ai_title
        caption = (d.get("caption") or "").strip() or ai_caption

        hashtags = _normalize_hashtags(d.get("hashtags"))
        platform_results = _normalize_platform_results(d.get("platform_results"))

        thumbnail_url = None
        thumb_key = d.get("thumbnail_r2_key")
        if thumb_key:
            try:
                s3 = s3 or get_s3_client()
                thumbnail_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(thumb_key)},
                    ExpiresIn=3600,
                )
            except Exception:
                thumbnail_url = None

        item = {
            "id": str(d.get("id")),
            "filename": d.get("filename"),
            "platforms": list(d.get("platforms") or []),
            "status": d.get("status"),
            "privacy": d.get("privacy", "public"),
            "title": title,
            "caption": caption,
            "hashtags": hashtags,

            "ai_title": ai_title,
            "ai_caption": ai_caption,
            "ai_hashtags": ai_hashtags,

            "scheduled_time": d.get("scheduled_time").isoformat() if d.get("scheduled_time") else None,
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
            "completed_at": d.get("completed_at").isoformat() if d.get("completed_at") else None,

            "put_cost": int(d.get("put_reserved") or 0),
            "aic_cost": int(d.get("aic_reserved") or 0),

            "error_code": d.get("error_code"),
            "error": d.get("error_detail") or d.get("error_code") or None,

            "thumbnail_url": thumbnail_url,
            "platform_results": platform_results,

            "file_size": d.get("file_size"),
            "views":    int(d.get("views")    or 0),
            "likes":    int(d.get("likes")    or 0),
            "comments": int(d.get("comments") or 0),
            "shares":   int(d.get("shares")   or 0),

            "progress": int(d.get("processing_progress") or 0),
            "current_stage": d.get("processing_stage"),

            # Schedule fields — lets frontend distinguish smart vs manual scheduled
            "schedule_mode":     d.get("schedule_mode") or "immediate",
            "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),

            # Direct video URL if stored by worker
            "video_url": d.get("video_url"),
        }
        out.append(item)

    if not meta:
        return out

    return {"uploads": out, "total": int(total or 0), "limit": limit, "offset": offset}


async def partial_refund_tokens(
    conn,
    user_id: str,
    upload_id: str,
    succeeded_platforms: list,
    failed_platforms: list,
    original_put_cost: int,
    original_aic_cost: int,
):
    """
    Pro-rate a refund for platforms that failed in a partial upload.

    PUT cost formula (mirrors entitlements.py compute_put_cost):
        base = 10  (covers the first platform)
        +2 per *extra* platform beyond the first
        (HUD / priority / thumbnail costs are per-job, not per-platform — kept)

    For a partial failure we refund:
        2 tokens × number_of_failed_platforms
        (the base-10 is always kept because work was done)

    AIC is per-job, not per-platform — no partial AIC refund.

    Only runs when there is at least one success AND at least one failure.
    On full failure the worker calls release/unreserve instead.
    """
    n_failed = len(failed_platforms or [])
    if n_failed == 0 or not (succeeded_platforms or []):
        return  # nothing to refund

    put_refund = n_failed * 2
    put_refund = min(put_refund, max(0, int(original_put_cost or 0) - 10))  # never refund the base-10

    if put_refund <= 0:
        return

    await conn.execute(
        """
        UPDATE wallets
        SET
            put_balance  = put_balance  + $1,
            put_reserved = GREATEST(0, put_reserved - $1),
            updated_at   = NOW()
        WHERE user_id = $2
        """,
        put_refund,
        user_id,
    )

    await conn.execute(
        """
        INSERT INTO token_ledger
            (user_id, token_type, delta, reason, upload_id, ref_type, metadata)
        VALUES
            ($1, 'put', $2, 'partial_platform_refund', $3, 'upload',
             jsonb_build_object(
                 'failed_platforms',    $4::text,
                 'succeeded_platforms', $5::text,
                 'original_put_cost',   $6
             ))
        """,
        user_id,
        put_refund,
        upload_id,
        ",".join([str(x) for x in (failed_platforms or [])]),
        ",".join([str(x) for x in (succeeded_platforms or [])]),
        int(original_put_cost or 0),
    )

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
    ent = get_entitlements_for_tier(tier)
    daily = ent.put_daily * 4  # 4 platforms
    wallet = await get_wallet(conn, user_id)
    last_refill = wallet.get("last_refill_date")
    today = _now_utc().date()
    if last_refill and last_refill >= today:
        return
    monthly_cap = ent.put_monthly
    current = wallet["put_balance"]
    if current < monthly_cap:
        add = min(daily, monthly_cap - current)
        await conn.execute("UPDATE wallets SET put_balance = put_balance + $1, last_refill_date = $2 WHERE user_id = $3", add, today, user_id)
        await ledger_entry(conn, user_id, "put", add, "daily_refill")



@app.get("/api/scheduled")
async def get_scheduled(user: dict = Depends(get_current_user)):
    """Get scheduled uploads (scheduled + smart modes, all pending statuses)"""
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch("""
            SELECT * FROM uploads
            WHERE user_id = $1
              AND schedule_mode IN ('scheduled', 'smart')
              AND status IN ('pending', 'queued', 'scheduled', 'staged', 'ready_to_publish')
            ORDER BY scheduled_time ASC
        """, user["id"])
    return [{"id": str(u["id"]), "filename": u["filename"], "platforms": u["platforms"], "status": u["status"], "title": u["title"], "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None, "created_at": u["created_at"].isoformat() if u["created_at"] else None, "schedule_mode": u["schedule_mode"]} for u in uploads]


@app.post("/api/uploads/{upload_id}/generate-thumbnail")
async def generate_thumbnail_for_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Backfill / regenerate the thumbnail for an existing upload.

    Workflow:
      1. Fetch the video from R2 to a temp file
      2. Run FFmpeg to extract a frame at 30% into the video
      3. Upload the JPEG to R2 at thumbnails/{user_id}/{upload_id}/thumbnail.jpg
      4. Update thumbnail_r2_key in the uploads row
      5. Return a fresh presigned URL

    This fixes the gap where uploads processed before the worker fix
    have thumbnail_r2_key = NULL in the database.
    """
    import tempfile, subprocess
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, r2_key, thumbnail_r2_key, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    # If thumbnail already exists, just return the presigned URL
    if row.get("thumbnail_r2_key"):
        try:
            s3 = get_s3_client()
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(row["thumbnail_r2_key"])},
                ExpiresIn=3600,
            )
            return {"thumbnail_url": url, "r2_key": row["thumbnail_r2_key"], "generated": False}
        except Exception:
            pass  # fall through and regenerate

    r2_key = row.get("r2_key")
    if not r2_key:
        raise HTTPException(400, "No video file key found for this upload")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        video_path = tmp_path / "video.mp4"
        thumb_path = tmp_path / "thumbnail.jpg"

        # 1. Download video from R2
        try:
            s3 = get_s3_client()
            s3.download_file(R2_BUCKET_NAME, _normalize_r2_key(r2_key), str(video_path))
        except Exception as e:
            raise HTTPException(500, f"Could not download video from storage: {e}")

        # 2. Get duration then extract frame at 30%
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            duration = 10.0
            if probe.returncode == 0:
                import json as _json
                for stream in _json.loads(probe.stdout).get("streams", []):
                    if stream.get("codec_type") == "video":
                        duration = float(stream.get("duration", 10) or 10)
                        break
            offset = max(0.5, duration * 0.30)
        except Exception:
            offset = 5.0

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{offset:.3f}",
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-q:v", "2",
                    "-vf", "scale=1080:-2",
                    str(thumb_path),
                ],
                capture_output=True, timeout=60
            )
            if result.returncode != 0 or not thumb_path.exists():
                # Fallback: try at 1 second
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", "1", "-i", str(video_path),
                     "-vframes", "1", "-q:v", "2", "-vf", "scale=1080:-2", str(thumb_path)],
                    capture_output=True, timeout=30
                )
        except Exception as e:
            raise HTTPException(500, f"FFmpeg thumbnail extraction failed: {e}")

        if not thumb_path.exists():
            raise HTTPException(500, "Thumbnail extraction produced no output")

        # 3. Upload to R2
        thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/thumbnail.jpg"
        try:
            s3.upload_file(
                str(thumb_path), R2_BUCKET_NAME, thumb_r2_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to upload thumbnail to storage: {e}")

        # 4. Update DB
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE uploads SET thumbnail_r2_key = $1, updated_at = NOW() WHERE id = $2",
                thumb_r2_key, upload_id
            )

        # 5. Return presigned URL
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": thumb_r2_key},
                ExpiresIn=3600,
            )
        except Exception:
            url = None

        return {
            "thumbnail_url": url,
            "r2_key": thumb_r2_key,
            "generated": True,
            "offset_seconds": offset,
        }


@app.post("/api/uploads/{upload_id}/sync-analytics")
async def sync_upload_analytics(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Fetch latest engagement stats for a single completed upload from platform APIs.
    Uses the video IDs stored in platform_results to query per-video metrics.
    Updates the uploads table (views, likes, comments, shares) and returns fresh data.
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platforms, platform_results, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if row["status"] not in ("completed", "succeeded", "partial"):
        return {"synced": False, "reason": "not_completed", "views": 0, "likes": 0, "comments": 0, "shares": 0}

    # Parse platform_results to get per-platform video IDs
    raw_pr = _safe_json(row["platform_results"], [])
    pr_list = []
    if isinstance(raw_pr, list):
        pr_list = [x for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, dict):
        pr_list = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw_pr.items()]
    # Alias canonical worker fields so lookups below find them
    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]
        if pr.get("platform_url") and not pr.get("url"):
            pr["url"] = pr["platform_url"]

    # Get tokens for all connected platforms
    async with db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )

    token_map = {}
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if dec:
                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                    dec["ig_user_id"] = str(tr["account_id"])
                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                    dec["page_id"] = str(tr["account_id"])
                token_map[tr["platform"]] = dec
        except Exception:
            pass

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats = {}

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            ok   = pr.get("success") == True or str(pr.get("status","")).lower() in ("published","succeeded","success")
            if not ok:
                continue

            tok = token_map.get(plat, {})
            access_token = tok.get("access_token", "")
            if not access_token:
                continue

            # platform_video_id is the canonical field written by db.py/mark_processing_completed
            # video_id / media_id / share_id etc. are legacy / webhook-written variants
            video_id = (
                pr.get("platform_video_id")  # canonical (worker pipeline)
                or pr.get("video_id") or pr.get("videoId") or pr.get("id")
                or pr.get("media_id") or pr.get("post_id") or pr.get("share_id")
            )

            try:
                if plat == "tiktok" and video_id:
                    resp = await client.post(
                        "https://open.tiktokapis.com/v2/video/query/",
                        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                        json={"filters": {"video_ids": [str(video_id)]},
                              "fields": ["id","view_count","like_count","comment_count","share_count"]},
                    )
                    if resp.status_code == 200:
                        vids = resp.json().get("data", {}).get("videos", []) or []
                        if vids:
                            v = vids[0]
                            s = {"views": int(v.get("view_count") or 0), "likes": int(v.get("like_count") or 0),
                                 "comments": int(v.get("comment_count") or 0), "shares": int(v.get("share_count") or 0)}
                            platform_stats["tiktok"] = s
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]; total_shares  += s["shares"]

                elif plat == "youtube" and video_id:
                    resp = await client.get(
                        "https://www.googleapis.com/youtube/v3/videos",
                        params={"part": "statistics", "id": str(video_id)},
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    if resp.status_code == 200:
                        items = resp.json().get("items", []) or []
                        if items:
                            st = items[0].get("statistics", {})
                            s = {"views": int(st.get("viewCount") or 0), "likes": int(st.get("likeCount") or 0),
                                 "comments": int(st.get("commentCount") or 0), "shares": 0}
                            platform_stats["youtube"] = s
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]

                elif plat == "instagram" and video_id:
                    shortcode = pr.get("shortcode") or video_id
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{shortcode}/insights",
                        params={"access_token": access_token,
                                "metric": "plays,likes,comments,saved,shares,reach"},
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        for m in resp.json().get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [])
                            val  = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                            if name == "plays":       s["views"]    += val
                            elif name == "likes":     s["likes"]    += val
                            elif name == "comments":  s["comments"] += val
                            elif name == "shares":    s["shares"]   += val
                        platform_stats["instagram"] = s
                        total_views    += s["views"];    total_likes   += s["likes"]
                        total_comments += s["comments"]; total_shares  += s["shares"]

                elif plat == "facebook" and video_id:
                    page_id = tok.get("page_id", "")
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{video_id}",
                        params={"access_token": access_token,
                                "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)"},
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        for m in resp.json().get("insights", {}).get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [{}])
                            val  = vals[-1].get("value", 0) if vals else 0
                            if isinstance(val, dict): val = sum(val.values())
                            val = int(val or 0)
                            if name == "total_video_views":                      s["views"]    += val
                            elif name == "total_video_reactions_by_type_total":  s["likes"]    += val
                            elif name == "total_video_comments":                  s["comments"] += val
                            elif name == "total_video_shares":                    s["shares"]   += val
                        platform_stats["facebook"] = s
                        total_views    += s["views"];    total_likes   += s["likes"]
                        total_comments += s["comments"]; total_shares  += s["shares"]

            except Exception as e:
                logger.warning(f"sync-analytics error for {plat}/{video_id}: {e}")
                continue

    # Persist to DB
    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4, updated_at=NOW()
               WHERE id=$5 AND user_id=$6""",
            total_views, total_likes, total_comments, total_shares,
            upload_id, user["id"],
        )

    return {
        "synced": True,
        "views": total_views, "likes": total_likes,
        "comments": total_comments, "shares": total_shares,
        "platform_stats": platform_stats,
    }


@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT put_reserved, aic_reserved, status, title, platforms FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("pending", "queued"):
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_DELETED",
                               event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                               details={"title": upload["title"], "status_at_delete": upload["status"],
                                        "platforms": list(upload["platforms"] or [])},
                               request=request, severity="WARNING")
    return {"status": "deleted"}

# ============================================================
# Scheduled Uploads Management
# ============================================================
@app.get("/api/scheduled/stats")
async def get_scheduled_stats(user: dict = Depends(get_current_user)):
    """Get scheduled upload statistics for the current user"""
    async with db_pool.acquire() as conn:
        now = _now_utc()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        week_end = now + timedelta(days=7)
        
        # Count pending uploads
        pending_count = await conn.fetchval("""
            SELECT COUNT(*) FROM uploads 
            WHERE user_id = $1 
            AND scheduled_time IS NOT NULL 
            AND scheduled_time > $2
            AND status IN ('pending', 'scheduled', 'queued')
        """, user["id"], now)
        
        # Count uploads today
        today_count = await conn.fetchval("""
            SELECT COUNT(*) FROM uploads 
            WHERE user_id = $1 
            AND scheduled_time >= $2 
            AND scheduled_time < $3
            AND status IN ('pending', 'scheduled', 'queued')
        """, user["id"], today_start, today_end)
        
        # Count uploads this week
        week_count = await conn.fetchval("""
            SELECT COUNT(*) FROM uploads 
            WHERE user_id = $1 
            AND scheduled_time >= $2 
            AND scheduled_time < $3
            AND status IN ('pending', 'scheduled', 'queued')
        """, user["id"], now, week_end)
        
    return {
        "pending": pending_count or 0,
        "today": today_count or 0,
        "week": week_count or 0
    }

@app.get("/api/scheduled/list")
async def get_scheduled_list(user: dict = Depends(get_current_user)):
    """Get list of all scheduled uploads for the current user"""
    async with db_pool.acquire() as conn:
        now = _now_utc()
        
        uploads = await conn.fetch("""
            SELECT 
                id, title, scheduled_time, platforms, 
                thumbnail_r2_key, caption, status, 
                created_at, timezone, schedule_mode, schedule_metadata
            FROM uploads 
            WHERE user_id = $1 
            AND scheduled_time IS NOT NULL 
            AND scheduled_time > $2
            AND schedule_mode IN ('scheduled', 'smart')
            AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
            ORDER BY scheduled_time ASC
        """, user["id"], now)
        
    result = []
    for upload in uploads:
        thumbnail_url = None
        if upload["thumbnail_r2_key"]:
            # Generate presigned URL for thumbnail
            try:
                s3 = get_s3_client()
                thumbnail_url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': R2_BUCKET_NAME, 'Key': upload["thumbnail_r2_key"]},
                    ExpiresIn=3600
                )
            except:
                pass
        
        # Parse smart schedule metadata for per-platform times
        smart_schedule = None
        try:
            sm = upload["schedule_metadata"]
            if sm and upload["schedule_mode"] == "smart":
                smart_schedule = sm if isinstance(sm, dict) else json.loads(sm)
        except Exception:
            pass

        result.append({
            "id": str(upload["id"]),
            "title": upload["title"] or "Untitled",
            "scheduled_time": upload["scheduled_time"].isoformat() if upload["scheduled_time"] else None,
            "timezone": upload["timezone"] or "UTC",
            "platforms": list(upload["platforms"]) if upload["platforms"] else [],
            "thumbnail": thumbnail_url,
            "caption": upload["caption"],
            "status": upload["status"],
            "schedule_mode": upload["schedule_mode"] or "scheduled",
            "smart_schedule": smart_schedule,
            "created_at": upload["created_at"].isoformat() if upload["created_at"] else None
        })
    
    return result

@app.get("/api/scheduled/{upload_id}")
async def get_scheduled_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Get details of a specific scheduled upload"""
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("""
            SELECT 
                id, title, scheduled_time, platforms, timezone,
                thumbnail_r2_key, caption, hashtags, privacy,
                status, created_at
            FROM uploads 
            WHERE id = $1 AND user_id = $2
        """, upload_id, user["id"])
        
        if not upload:
            raise HTTPException(404, "Scheduled upload not found")
        
        thumbnail_url = None
        if upload["thumbnail_r2_key"]:
            try:
                s3 = get_s3_client()
                thumbnail_url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': R2_BUCKET_NAME, 'Key': upload["thumbnail_r2_key"]},
                    ExpiresIn=3600
                )
            except:
                pass
        
    return {
        "id": str(upload["id"]),
        "title": upload["title"] or "Untitled",
        "scheduled_time": upload["scheduled_time"].isoformat() if upload["scheduled_time"] else None,
        "timezone": upload["timezone"] or "UTC",
        "platforms": list(upload["platforms"]) if upload["platforms"] else [],
        "thumbnail": thumbnail_url,
        "caption": upload["caption"],
        "hashtags": list(upload["hashtags"]) if upload["hashtags"] else [],
        "privacy": upload["privacy"],
        "status": upload["status"],
        "created_at": upload["created_at"].isoformat() if upload["created_at"] else None
    }

@app.patch("/api/scheduled/{upload_id}")
async def update_scheduled_upload(
    upload_id: str, 
    update_data: ScheduledUploadUpdate, 
    user: dict = Depends(get_current_user)
):
    """Update a scheduled upload's details"""
    async with db_pool.acquire() as conn:
        # Verify ownership
        upload = await conn.fetchrow(
            "SELECT id, status FROM uploads WHERE id = $1 AND user_id = $2", 
            upload_id, user["id"]
        )
        
        if not upload:
            raise HTTPException(404, "Scheduled upload not found")
        
        if upload["status"] not in ['pending', 'scheduled', 'queued']:
            raise HTTPException(400, "Cannot edit upload that is already processing or completed")
        
        # Build update query dynamically
        updates = []
        params = [upload_id, user["id"]]
        param_count = 2
        
        if update_data.title is not None:
            param_count += 1
            updates.append(f"title = ${param_count}")
            params.append(update_data.title)
        
        if update_data.scheduled_time is not None:
            param_count += 1
            updates.append(f"scheduled_time = ${param_count}")
            params.append(update_data.scheduled_time)
        
        if update_data.timezone is not None:
            param_count += 1
            updates.append(f"timezone = ${param_count}")
            params.append(update_data.timezone)
        
        if update_data.caption is not None:
            param_count += 1
            updates.append(f"caption = ${param_count}")
            params.append(update_data.caption)
        
        if update_data.hashtags is not None:
            param_count += 1
            updates.append(f"hashtags = ${param_count}")
            params.append(update_data.hashtags)
        
        if update_data.platforms is not None:
            param_count += 1
            updates.append(f"platforms = ${param_count}")
            params.append(update_data.platforms)
        
        if not updates:
            raise HTTPException(400, "No updates provided")
        
        # Always update updated_at
        param_count += 1
        updates.append(f"updated_at = ${param_count}")
        params.append(_now_utc())
        
        query = f"""
            UPDATE uploads 
            SET {', '.join(updates)}
            WHERE id = $1 AND user_id = $2
        """
        
        await conn.execute(query, *params)
    
    return {"status": "updated", "id": upload_id}

@app.delete("/api/scheduled/{upload_id}")
async def cancel_scheduled_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Cancel/delete a scheduled upload"""
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("""
            SELECT id, put_reserved, aic_reserved, status 
            FROM uploads 
            WHERE id = $1 AND user_id = $2
        """, upload_id, user["id"])
        
        if not upload:
            raise HTTPException(404, "Scheduled upload not found")
        
        if upload["status"] not in ['pending', 'scheduled', 'queued']:
            raise HTTPException(400, "Cannot cancel upload that is already processing or completed")
        
        # Refund reserved tokens if any
        if upload["put_reserved"] > 0 or upload["aic_reserved"] > 0:
            await refund_tokens(
                conn, 
                user["id"], 
                upload["put_reserved"], 
                upload["aic_reserved"], 
                upload_id
            )
        
        # Delete the upload
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
    
    return {"status": "cancelled", "id": upload_id}

@app.post("/api/uploads/{upload_id}/retry")
async def retry_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Reset a failed/cancelled upload and re-queue it for processing."""
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        # Only allow retry for terminal states
        if upload["status"] not in ("failed", "cancelled"):
            raise HTTPException(400, "Only failed or cancelled uploads can be retried")

        # Reset processing state (keep engagement + cost fields intact)
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'pending',
                error_code = NULL,
                error_detail = NULL,
                processing_started_at = NULL,
                processing_finished_at = NULL,
                completed_at = NULL,
                cancel_requested = FALSE,
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            """,
            upload_id, user["id"]
        )

        # Pull latest preferences (and respect plan entitlements)
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])
        plan = get_plan(user.get("subscription_tier", "free"))

    job_data = {
        "job_id": str(uuid.uuid4()),
        "upload_id": upload_id,
        "user_id": str(user["id"]),
        "preferences": user_prefs,
        "plan_features": {
            "ai": plan.get("ai", False),
            "priority": plan.get("priority", False),
            "watermark": plan.get("watermark", True),
        },
        "action": "retry",
    }

    await enqueue_job(job_data, priority=plan.get("priority", False))
    return {"status": "requeued", "upload_id": upload_id}


# ============================================================
# User Color Preferences
# ============================================================
@app.get("/api/colors")
async def get_color_preferences(user: dict = Depends(get_current_user)):
    """Get user's custom color preferences for platforms"""
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow("""
            SELECT 
                tiktok_color, youtube_color, instagram_color, 
                facebook_color, accent_color
            FROM user_color_preferences 
            WHERE user_id = $1
        """, user["id"])
        
        if not prefs:
            # Return defaults
            return {
                "tiktok_color": "#000000",
                "youtube_color": "#FF0000",
                "instagram_color": "#E4405F",
                "facebook_color": "#1877F2",
                "accent_color": "#F97316"
            }
        
    return {
        "tiktok_color": prefs["tiktok_color"],
        "youtube_color": prefs["youtube_color"],
        "instagram_color": prefs["instagram_color"],
        "facebook_color": prefs["facebook_color"],
        "accent_color": prefs["accent_color"]
    }

@app.put("/api/colors")
async def update_color_preferences(
    colors: ColorPreferencesUpdate, 
    user: dict = Depends(get_current_user)
):
    """Update user's custom color preferences"""
    async with db_pool.acquire() as conn:
        # Check if preferences exist
        exists = await conn.fetchval(
            "SELECT 1 FROM user_color_preferences WHERE user_id = $1", 
            user["id"]
        )
        
        if not exists:
            # Create default preferences
            await conn.execute("""
                INSERT INTO user_color_preferences (user_id) 
                VALUES ($1)
            """, user["id"])
        
        # Build update query
        updates = []
        params = [user["id"]]
        param_count = 1
        
        if colors.tiktok_color is not None:
            param_count += 1
            updates.append(f"tiktok_color = ${param_count}")
            params.append(colors.tiktok_color)
        
        if colors.youtube_color is not None:
            param_count += 1
            updates.append(f"youtube_color = ${param_count}")
            params.append(colors.youtube_color)
        
        if colors.instagram_color is not None:
            param_count += 1
            updates.append(f"instagram_color = ${param_count}")
            params.append(colors.instagram_color)
        
        if colors.facebook_color is not None:
            param_count += 1
            updates.append(f"facebook_color = ${param_count}")
            params.append(colors.facebook_color)
        
        if colors.accent_color is not None:
            param_count += 1
            updates.append(f"accent_color = ${param_count}")
            params.append(colors.accent_color)
        
        if not updates:
            raise HTTPException(400, "No color updates provided")
        
        # Always update updated_at
        param_count += 1
        updates.append(f"updated_at = ${param_count}")
        params.append(_now_utc())
        
        query = f"""
            UPDATE user_color_preferences 
            SET {', '.join(updates)}
            WHERE user_id = $1
        """
        
        await conn.execute(query, *params)
    
    return {"status": "updated"}

# ============================================================
# Account Groups
# ============================================================

@app.get("/api/settings/preferences")
async def get_user_preferences(user: dict = Depends(get_current_user)):
    """GET user content preferences - used by settings page AND upload workflow"""
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow(
            "SELECT * FROM user_preferences WHERE user_id = $1",
            user["id"]
        )

        if not prefs:
            await conn.execute("INSERT INTO user_preferences (user_id) VALUES ($1)", user["id"])
            prefs = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1",
                user["id"]
            )

        d = dict(prefs) if prefs else {}
        
        # Ensure arrays are properly formatted as lists
        always_tags = d.get("always_hashtags")
        blocked_tags = d.get("blocked_hashtags")
        platform_tags = d.get("platform_hashtags")
        
        # DEBUG: Log what we loaded from database
        logger.info(f"Loading preferences for user {user['id']}")
        logger.info(f"always_hashtags from DB: {always_tags} (type: {type(always_tags)})")
        logger.info(f"blocked_hashtags from DB: {blocked_tags} (type: {type(blocked_tags)})")
        
        # Parse JSON strings if needed (JSONB might come back as strings)
        if isinstance(always_tags, str):
            try:
                always_tags = json.loads(always_tags)
            except:
                always_tags = []
        if isinstance(blocked_tags, str):
            try:
                blocked_tags = json.loads(blocked_tags)
            except:
                blocked_tags = []
        if isinstance(platform_tags, str):
            try:
                platform_tags = json.loads(platform_tags)
            except:
                platform_tags = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}

        return {
            "autoCaptions": d.get("auto_captions", False),
            "autoThumbnails": d.get("auto_thumbnails", False),
            "thumbnailInterval": str(d.get("thumbnail_interval", 5)),
            "defaultPrivacy": d.get("default_privacy", "public"),
            "aiHashtagsEnabled": d.get("ai_hashtags_enabled", False),
            "aiHashtagCount": str(d.get("ai_hashtag_count", 5)),
            "aiHashtagStyle": d.get("ai_hashtag_style", "mixed"),
            "hashtagPosition": d.get("hashtag_position", "end"),
            "maxHashtags": str(d.get("max_hashtags", 15)),
            "alwaysHashtags": always_tags or [],
            "blockedHashtags": blocked_tags or [],
            "platformHashtags": platform_tags or {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            "emailNotifications": d.get("email_notifications", True),
            "discordWebhook": d.get("discord_webhook"),
            "trillEnabled": bool(d.get("trill_enabled", False)),
            "trillMinScore": int(d.get("trill_min_score", 0) or 0),
            "trillHudEnabled": bool(d.get("trill_hud_enabled", False)),
            "trillAiEnhance": bool(d.get("trill_ai_enhance", False)),
            "trillOpenaiModel": d.get("trill_openai_model", "gpt-4o-mini"),
        }

@app.post("/api/settings/preferences")
async def save_user_preferences(
    payload: dict = Body(...),
    user: dict = Depends(get_current_user),
):
    """
    SAVE user content preferences.

    Contract:
    - Frontend sends camelCase keys.
    - DB stores snake_case columns in user_preferences.
    - JSON columns are stored as jsonb (always_hashtags, blocked_hashtags, platform_hashtags).
    """

    CAMEL_TO_SNAKE = {
        "autoCaptions": "auto_captions",
        "autoThumbnails": "auto_thumbnails",
        "thumbnailInterval": "thumbnail_interval",
        "defaultPrivacy": "default_privacy",
        "aiHashtagsEnabled": "ai_hashtags_enabled",
        "aiHashtagCount": "ai_hashtag_count",
        "aiHashtagStyle": "ai_hashtag_style",
        "hashtagPosition": "hashtag_position",
        "maxHashtags": "max_hashtags",
        "alwaysHashtags": "always_hashtags",
        "blockedHashtags": "blocked_hashtags",
        "platformHashtags": "platform_hashtags",
        "emailNotifications": "email_notifications",
        "discordWebhook": "discord_webhook",
        "trillEnabled": "trill_enabled",
        "trillMinScore": "trill_min_score",
        "trillHudEnabled": "trill_hud_enabled",
        "trillAiEnhance": "trill_ai_enhance",
        "trillOpenaiModel": "trill_openai_model",
    }

    def normalize_prefs_payload(p: dict) -> dict:
        out: dict = {}
        for k, v in (p or {}).items():
            out[CAMEL_TO_SNAKE.get(k, k)] = v
        return out

    p = normalize_prefs_payload(payload)

    # defaults / coercions (frontend may send strings from text inputs)
    def _coerce_hashtag_list(v):
        """Normalize hashtag input to flat list of strings"""
        if v is None:
            return []
        if isinstance(v, list):
            # Simple flatten - just convert each item to string and filter
            result = []
            for item in v:
                if isinstance(item, str) and item and not item.startswith('[') and not item.startswith('"'):
                    # Only add if it's a simple string, not JSON garbage
                    clean = item.strip().lower().replace('#', '')
                    if clean and len(clean) < 50:  # Reasonable hashtag length
                        result.append(clean)
            return result
        if isinstance(v, str):
            # Simple comma-separated string
            s = v.strip()
            if not s or s.startswith('[') or s.startswith('"'):
                # Looks like JSON garbage, ignore it
                return []
            # Normal comma-separated
            parts = [p.strip().lower().replace('#', '') for p in s.split(',')]
            return [p for p in parts if p and len(p) < 50]
        return []

    def _coerce_platform_map(v):
        default_map = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
        if v is None:
            return default_map
        if isinstance(v, dict):
            # Ensure each platform value is a list of strings
            out = {}
            for k, val in v.items():
                out[str(k)] = _coerce_hashtag_list(val)
            # Ensure all expected keys exist
            for k in default_map.keys():
                out.setdefault(k, [])
            return out
        if isinstance(v, str):
            # Try JSON string first, else treat as a global hashtag list applied to all
            s = v.strip()
            if not s:
                return default_map
            try:
                obj = json.loads(s)
                return _coerce_platform_map(obj)
            except Exception:
                lst = _coerce_hashtag_list(s)
                return {k: lst[:] for k in default_map.keys()}
        return default_map

    always = _coerce_hashtag_list(p.get("always_hashtags") or p.get("alwaysHashtags"))
    blocked = _coerce_hashtag_list(p.get("blocked_hashtags") or p.get("blockedHashtags"))
    platform = _coerce_platform_map(p.get("platform_hashtags") or p.get("platformHashtags"))
    
    # DEBUG: Log what we're about to save
    logger.info(f"Saving preferences for user {user['id']}")
    logger.info(f"always_hashtags: {always} (type: {type(always)})")
    logger.info(f"blocked_hashtags: {blocked} (type: {type(blocked)})")
    logger.info(f"platform_hashtags: {platform} (type: {type(platform)})")

    # core scalar coercions
    auto_captions = bool(p.get("auto_captions", False))
    auto_thumbnails = bool(p.get("auto_thumbnails", False))

    try:
        thumbnail_interval = int(p.get("thumbnail_interval", 5))
    except Exception:
        thumbnail_interval = 5

    default_privacy = str(p.get("default_privacy", "public") or "public").lower()
    if default_privacy not in ("public", "unlisted", "private"):
        default_privacy = "public"

    ai_hashtags_enabled = bool(p.get("ai_hashtags_enabled", False))

    try:
        ai_hashtag_count = int(p.get("ai_hashtag_count", 5))
    except Exception:
        ai_hashtag_count = 5

    ai_hashtag_style = str(p.get("ai_hashtag_style", "mixed") or "mixed").lower()
    if ai_hashtag_style not in ("trending", "niche", "mixed"):
        ai_hashtag_style = "mixed"

    hashtag_position = str(p.get("hashtag_position", "end") or "end").lower()
    if hashtag_position not in ("start", "end"):
        hashtag_position = "end"

    try:
        max_hashtags = int(p.get("max_hashtags", 15))
    except Exception:
        max_hashtags = 15

    email_notifications = bool(p.get("email_notifications", True))
    discord_webhook = p.get("discord_webhook")

    async with db_pool.acquire() as conn:
        # ensure row exists
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            user["id"],
        )

        await conn.execute(
            """
            UPDATE user_preferences SET
                auto_captions = $1,
                auto_thumbnails = $2,
                thumbnail_interval = $3,
                default_privacy = $4,
                ai_hashtags_enabled = $5,
                ai_hashtag_count = $6,
                ai_hashtag_style = $7,
                hashtag_position = $8,
                max_hashtags = $9,
                always_hashtags = $10::jsonb,
                blocked_hashtags = $11::jsonb,
                platform_hashtags = $12::jsonb,
                email_notifications = $13,
                discord_webhook = $14,
                updated_at = NOW()
            WHERE user_id = $15
            """,
            auto_captions,
            auto_thumbnails,
            thumbnail_interval,
            default_privacy,
            ai_hashtags_enabled,
            ai_hashtag_count,
            ai_hashtag_style,
            hashtag_position,
            max_hashtags,
            json.dumps(always),  # Always use json.dumps for JSONB
            json.dumps(blocked),  # Always use json.dumps for JSONB
            json.dumps(platform),  # Always use json.dumps for JSONB
            email_notifications,
            discord_webhook,
            user["id"],
        )

        # immediate read-after-write to validate persistence (helps front-end debugging)
        row = await conn.fetchrow(
            "SELECT updated_at FROM user_preferences WHERE user_id = $1",
            user["id"],
        )

    return {"ok": True, "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None)}



@app.put("/api/settings/preferences")
async def save_user_preferences_put(
    prefs: UserPreferencesUpdate,
    user: dict = Depends(get_current_user)
):
    """Backward-compatible alias for clients that still call PUT"""
    return await save_user_preferences(prefs, user)

async def get_user_prefs_for_upload(conn, user_id: int) -> dict:
    """Helper to fetch user preferences for upload processing"""
    # Read from user_preferences table (primary source)
    prefs_row = await conn.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1",
        user_id
    )
    
    if prefs_row:
        # User has preferences in the table
        return {
            "auto_captions": prefs_row["auto_captions"],
            "auto_thumbnails": prefs_row["auto_thumbnails"],
            "thumbnail_interval": prefs_row["thumbnail_interval"],
            "default_privacy": prefs_row["default_privacy"],
            "ai_hashtags_enabled": prefs_row["ai_hashtags_enabled"],
            "ai_hashtag_count": prefs_row["ai_hashtag_count"],
            "ai_hashtag_style": prefs_row["ai_hashtag_style"],
            "hashtag_position": prefs_row["hashtag_position"],
            "max_hashtags": prefs_row["max_hashtags"],
            "always_hashtags": prefs_row["always_hashtags"] or [],
            "blocked_hashtags": prefs_row["blocked_hashtags"] or [],
            "platform_hashtags": prefs_row["platform_hashtags"] or {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            "email_notifications": prefs_row["email_notifications"],
            "discord_webhook": prefs_row["discord_webhook"]
        }
    
    # Fallback: Try legacy JSONB locations
    # Try user_settings.preferences_json first
    prefs_row = await conn.fetchrow(
        "SELECT preferences_json FROM user_settings WHERE user_id = $1",
        user_id
    )
    
    if prefs_row and prefs_row["preferences_json"]:
        prefs_data = prefs_row["preferences_json"]
        if isinstance(prefs_data, str):
            prefs = json.loads(prefs_data)
        else:
            prefs = prefs_data
    else:
        # Try users.preferences (oldest fallback)
        prefs_row = await conn.fetchrow(
            "SELECT preferences FROM users WHERE id = $1",
            user_id
        )
        if prefs_row and prefs_row["preferences"]:
            prefs_data = prefs_row["preferences"]
            if isinstance(prefs_data, str):
                prefs = json.loads(prefs_data)
            else:
                prefs = prefs_data
        else:
            prefs = {}
    
    # Return preferences with defaults (convert camelCase to snake_case for internal use)
    return {
        "auto_captions": prefs.get("autoCaptions", False),
        "auto_thumbnails": prefs.get("autoThumbnails", False),
        "thumbnail_interval": prefs.get("thumbnailInterval", 5),
        "default_privacy": prefs.get("defaultPrivacy", "public"),
        "ai_hashtags_enabled": prefs.get("aiHashtagsEnabled", False),
        "ai_hashtag_count": prefs.get("aiHashtagCount", 5),
        "ai_hashtag_style": prefs.get("aiHashtagStyle", "mixed"),
        "hashtag_position": prefs.get("hashtagPosition", "end"),
        "max_hashtags": prefs.get("maxHashtags", 30),
        "always_hashtags": prefs.get("alwaysHashtags", []),
        "blocked_hashtags": prefs.get("blockedHashtags", []),
        "platform_hashtags": prefs.get("platformHashtags", {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}),
        "email_notifications": prefs.get("emailNotifications", True),
        "discord_webhook": prefs.get("discordWebhook", None)
    }



class AccountGroupIn(BaseModel):
    name: str
    color: str | None = None
    account_ids: list[str] | None = None

class AccountGroupUpdate(BaseModel):
    name: str | None = None
    color: str | None = None
    account_ids: list[str] | None = None


@app.get("/api/groups")
async def get_groups(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        groups = await conn.fetch(
            """
            SELECT id, user_id, name, description, account_ids, color, created_at, updated_at
            FROM account_groups
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user["id"],
        )
    return [
        {
            "id": str(g["id"]),
            "name": g["name"],
            "description": g["description"],
            "account_ids": g["account_ids"] or [],
            "members": g["account_ids"] or [],
            "color": g["color"],
            "created_at": g["created_at"].isoformat() if g["created_at"] else None,
            "updated_at": g["updated_at"].isoformat() if g["updated_at"] else None,
        }
        for g in groups
    ]


@app.get("/api/groups/{group_id}")
async def get_group(group_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        g = await conn.fetchrow(
            """
            SELECT id, user_id, name, description, account_ids, color, created_at, updated_at
            FROM account_groups
            WHERE id = $1 AND user_id = $2
            """,
            group_id,
            user["id"],
        )
    if not g:
        raise HTTPException(404, "Group not found")
    return {
        "id": str(g["id"]),
        "name": g["name"],
        "description": g["description"],
        "account_ids": g["account_ids"] or [],
        "members": g["account_ids"] or [],
        "color": g["color"],
        "created_at": g["created_at"].isoformat() if g["created_at"] else None,
        "updated_at": g["updated_at"].isoformat() if g["updated_at"] else None,
    }


class GroupUpsert(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    account_ids: Optional[List[str]] = None  # platform_tokens.id values as strings
    members: Optional[List[str]] = None      # frontend alias for account_ids


@app.post("/api/groups")
async def create_group(payload: GroupUpsert, user: dict = Depends(get_current_user)):
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    color = payload.color or "#3b82f6"
    account_ids = payload.account_ids if payload.account_ids is not None else (payload.members or [])
    group_id = str(uuid.uuid4())

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO account_groups (id, user_id, name, description, account_ids, color, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
            """,
            group_id,
            user["id"],
            name,
            payload.description,
            account_ids,
            color,
        )

    return {"id": group_id, "name": name, "description": payload.description, "color": color, "account_ids": account_ids, "members": account_ids}


@app.put("/api/groups/{group_id}")
async def update_group(group_id: str, payload: GroupUpsert, user: dict = Depends(get_current_user)):
    updates = []
    params = [group_id, user["id"]]

    if payload.name is not None:
        updates.append(f"name = ${len(params)+1}")
        params.append(payload.name.strip())

    if payload.description is not None:
        updates.append(f"description = ${len(params)+1}")
        params.append(payload.description)

    if payload.color is not None:
        updates.append(f"color = ${len(params)+1}")
        params.append(payload.color)

    # accept either account_ids or members
    if payload.account_ids is not None or payload.members is not None:
        ids = payload.account_ids if payload.account_ids is not None else (payload.members or [])
        updates.append(f"account_ids = ${len(params)+1}")
        params.append(ids)

    updates.append("updated_at = NOW()")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM account_groups WHERE id = $1 AND user_id = $2",
            group_id,
            user["id"],
        )
        if not row:
            raise HTTPException(404, "Group not found")

        if updates:
            await conn.execute(
                f"UPDATE account_groups SET {', '.join(updates)} WHERE id = $1 AND user_id = $2",
                *params,
            )

    return {"status": "updated"}


@app.delete("/api/groups/{group_id}")
async def delete_group(group_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM account_groups WHERE id = $1 AND user_id = $2", group_id, user["id"])
    return {"status": "deleted"}

# ============================================================
# Platforms
# ============================================================
@app.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])
    
    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms: platforms[p] = []
        platforms[p].append({"id": str(acc["id"]), "account_id": acc["account_id"], "name": acc["account_name"], "username": acc["account_username"], "avatar": acc["account_avatar"], "is_primary": acc["is_primary"], "status": "active", "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None})
    
    plan = get_plan(user.get("subscription_tier", "free"))
    total = sum(len(v) for v in platforms.values())
    return {"platforms": platforms, "total_accounts": total, "max_accounts": plan.get("max_accounts", 1), "can_add_more": total < plan.get("max_accounts", 1)}

# Alias endpoint for frontend compatibility
@app.get("/api/platform-accounts")
async def get_platform_accounts(user: dict = Depends(get_current_user)):
    """Returns flat list of accounts for frontend compatibility"""
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])
    
    result = []
    for acc in accounts:
        result.append({
            "id": str(acc["id"]),
            "platform": acc["platform"],
            "account_id": acc["account_id"],
            "account_name": acc["account_name"],
            "account_username": acc["account_username"],
            "account_avatar_url": acc["account_avatar"],
            "is_primary": acc["is_primary"],
            "status": "active",
            "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None,
        })
    return {"accounts": result}

@app.get("/api/accounts")
async def get_accounts_simple(user: dict = Depends(get_current_user)):
    """Simple accounts list for dashboard"""
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("SELECT DISTINCT ON (platform, account_id) id, platform, account_name, account_username, account_avatar FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL AND account_id IS NOT NULL AND account_id <> '' AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '') ORDER BY platform, account_id, created_at DESC", user["id"])
    return [{"id": str(a["id"]), "platform": a["platform"], "name": a["account_name"], "username": a["account_username"], "avatar": a["account_avatar"], "status": "active"} for a in accounts]

@app.delete("/api/platforms/{platform}/accounts/{account_id}")
async def disconnect_account(
    platform: str,
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Disconnect a linked platform account.
    1. Revokes the access token at the provider.
    2. Hard-deletes the platform_tokens row.
    3. Writes a platform_disconnect_log record.
    """
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE id = $1 AND user_id = $2",
            account_id,
            user["id"],
        )
        if not row:
            raise HTTPException(404, "Account not found")

        # Revoke token at provider
        ok, err = await _revoke_platform_token(row["platform"], row["token_blob"])

        # Hard-delete (mark revoked_at first to satisfy the partial unique index,
        # then delete so no stale token lingers)
        await conn.execute(
            "UPDATE platform_tokens SET revoked_at = NOW() WHERE id = $1", row["id"]
        )
        await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])

        # Audit log — platform_disconnect_log (existing) + system_event_log (new)
        await conn.execute(
            """
            INSERT INTO platform_disconnect_log
                (user_id, platform, account_id, account_name,
                 revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
            VALUES ($1,$2,$3,$4,$5,$6,'self',$7)
            """,
            str(user["id"]),
            row["platform"],
            row["account_id"],
            row["account_name"],
            ok,
            err or None,
            ip_addr,
        )
        await log_system_event(conn, user_id=str(user["id"]), action="PLATFORM_DISCONNECTED",
                               event_category="PLATFORM", resource_type="platform",
                               resource_id=f"{row['platform']}:{row['account_id']}",
                               details={"platform": row["platform"], "account_name": row["account_name"],
                                        "provider_revoked": ok, "provider_error": err},
                               severity="WARNING")

    return {"status": "disconnected", "provider_revoked": ok}


@app.delete("/api/platform-accounts/{account_id}")
async def disconnect_account_by_id(
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Alias for the disconnect endpoint (used by older frontend code)."""
    return await disconnect_account(
        platform="",      # platform is looked up from the DB row
        account_id=account_id,
        request=request,
        user=user,
    )

# ============================================================
# OAuth Platform Connections
# ============================================================
OAUTH_CONFIG = {
    "tiktok": {
        "auth_url": "https://www.tiktok.com/v2/auth/authorize/",
        "token_url": "https://open.tiktokapis.com/v2/oauth/token/",
        # Added:
        # - video.list (required for /v2/video/list/ stats reads)
        # - user.info.stats (required to fetch follower_count, etc via /v2/user/info/)
        "scope": "user.info.basic,user.info.stats,video.publish,video.upload,video.list",
    },
    "youtube": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        # Added yt-analytics.readonly because _fetch_youtube_metrics() calls youtubeanalytics.googleapis.com/v2/reports
        "scope": (
            "https://www.googleapis.com/auth/youtube.upload "
            "https://www.googleapis.com/auth/youtube.readonly "
            "https://www.googleapis.com/auth/yt-analytics.readonly"
        ),
    },
    "instagram": {
        # Instagram Graph API uses Facebook OAuth (for publishing Reels)
        "auth_url": "https://www.facebook.com/v18.0/dialog/oauth",
        "token_url": "https://graph.facebook.com/v18.0/oauth/access_token",
        # Added instagram_manage_insights for full Insights API access
        "scope": (
            "instagram_basic,"
            "instagram_content_publish,"
            "instagram_manage_insights,"
            "pages_show_list,"
            "pages_read_engagement,"
            "business_management"
        ),
    },
    "facebook": {
        "auth_url": "https://www.facebook.com/v18.0/dialog/oauth",
        "token_url": "https://graph.facebook.com/v18.0/oauth/access_token",
        # Added read_insights + pages_read_user_content to harden page/video insights + video listing
        "scope": (
            "pages_manage_posts,"
            "pages_read_engagement,"
            "pages_read_user_content,"
            "pages_show_list,"
            "publish_video,"
            "read_insights"
        ),
    },
}


# OAuth state storage (in production, use Redis)
oauth_states: Dict[str, dict] = {}

# OAuth credentials from environment
# TikTok
# OAuth credentials from environment
# TikTok
TIKTOK_CLIENT_KEY    = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
# Separate secret used to verify TikTok webhook payloads (HMAC-SHA256).
# Set this to the value shown in TikTok Developer Portal → your app →
# Webhooks → "Client Secret". Falls back to TIKTOK_CLIENT_SECRET if
# you haven't configured a separate one yet (the common starting point).
TIKTOK_WEBHOOK_SECRET = (
    os.environ.get("TIKTOK_WEBHOOK_SECRET", "")
    or TIKTOK_CLIENT_SECRET
)
# YouTube/Google
YOUTUBE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "") or os.environ.get("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "") or os.environ.get("YOUTUBE_CLIENT_SECRET", "")
# Meta (Facebook & Instagram share credentials)
# Meta (Facebook & Instagram have SEPARATE credentials)
META_APP_ID = os.environ.get("META_APP_ID", "")
META_APP_SECRET = os.environ.get("META_APP_SECRET", "")
# Instagram Graph API authenticates via Facebook OAuth (uses META credentials)
INSTAGRAM_CLIENT_ID = META_APP_ID
INSTAGRAM_CLIENT_SECRET = META_APP_SECRET
# Facebook uses the main Meta App ID
FACEBOOK_CLIENT_ID = os.environ.get("FACEBOOK_CLIENT_ID", "") or META_APP_ID
FACEBOOK_CLIENT_SECRET = os.environ.get("FACEBOOK_CLIENT_SECRET", "") or META_APP_SECRET

def get_oauth_redirect_uri(platform: str) -> str:
    return f"{BASE_URL}/api/oauth/{platform}/callback"

@app.get("/api/oauth/{platform}/start")
async def oauth_start(platform: str, user: dict = Depends(get_current_user)):
    """Start OAuth flow for a platform"""
    if platform not in OAUTH_CONFIG:
        raise HTTPException(400, f"Unsupported platform: {platform}")
    
    # Check account limits
    plan = get_plan(user.get("subscription_tier", "free"))
    async with db_pool.acquire() as conn:
        current_count = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])
    
    if current_count >= plan.get("max_accounts", 1):
        raise HTTPException(403, f"Account limit reached ({plan.get('max_accounts', 1)}). Upgrade to add more.")
    
    config = OAUTH_CONFIG[platform]
    state = secrets.token_urlsafe(32)
    
    # Store state with user info
    oauth_states[state] = {
        "user_id": str(user["id"]),
        "platform": platform,
        "created_at": _now_utc().isoformat()
    }
    
    redirect_uri = get_oauth_redirect_uri(platform)
    
    if platform == "tiktok":
        params = {
            "client_key": TIKTOK_CLIENT_KEY,
            "scope": config["scope"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
        }
    elif platform == "youtube":
        params = {
            "client_id": YOUTUBE_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": config["scope"],
            "state": state,
            "access_type": "offline",
            "prompt": "select_account consent",  # Forces account picker + fresh consent
        }
    elif platform == "instagram":
        params = {
            "client_id": INSTAGRAM_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": config["scope"],
            "response_type": "code",
            "state": state,
            "auth_type": "rerequest",  # Force re-authentication
        }
    elif platform == "facebook":
        params = {
            "client_id": FACEBOOK_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": config["scope"],
            "response_type": "code",
            "state": state,
            "auth_type": "rerequest",  # Force re-authentication
        }
    
    auth_url = f"{config['auth_url']}?{urlencode(params)}"
    return {"auth_url": auth_url, "state": state}

@app.get("/api/oauth/{platform}/callback")
async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):
    """Handle OAuth callback - returns HTML that communicates with parent window"""
    
    def popup_response(success: bool, platform: str, error_msg: str = None):
        """Generate HTML that posts message to parent window and closes popup"""
        if success:
            message = f'{{"type": "oauth_success", "platform": "{platform}"}}'
        else:
            safe_error = (error_msg or "Unknown error").replace('"', '\\"').replace('\n', ' ')[:200]
            message = f'{{"type": "oauth_error", "platform": "{platform}", "error": "{safe_error}"}}'
        
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Connecting...</title></head>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #1a1a2e; color: white;">
            <div style="text-align: center;">
                <p>{"✓ Connected successfully!" if success else "✗ Connection failed"}</p>
                <p style="color: #888; font-size: 14px;">This window will close automatically...</p>
            </div>
            <script>
                if (window.opener) {{
                    window.opener.postMessage({message}, '{FRONTEND_URL}');
                }}
                setTimeout(() => window.close(), 1500);
            </script>
        </body>
        </html>
        """)
    
    if error:
        return popup_response(False, platform, error)
    
    if not state or state not in oauth_states:
        return popup_response(False, platform, "Invalid or expired session. Please try again.")
    
    state_data = oauth_states.pop(state)
    user_id = state_data["user_id"]
    
    if not code:
        return popup_response(False, platform, "No authorization code received")
    
    config = OAUTH_CONFIG[platform]
    redirect_uri = get_oauth_redirect_uri(platform)
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Exchange code for tokens based on platform
            if platform == "tiktok":
                token_response = await client.post(config["token_url"], data={
                    "client_key": TIKTOK_CLIENT_KEY,
                    "client_secret": TIKTOK_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                access_token = token_data.get("access_token")
                account_id = token_data.get("open_id", secrets.token_hex(8))
                account_name = "TikTok User"
                account_username = ""
                account_avatar = ""
                
            elif platform == "youtube":
                token_response = await client.post(config["token_url"], data={
                    "client_id": YOUTUBE_CLIENT_ID,
                    "client_secret": YOUTUBE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                access_token = token_data.get("access_token")
                
                # Get channel info
                user_response = await client.get(
                    "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                channels = user_response.json().get("items", [])
                if channels:
                    channel = channels[0]
                    snippet = channel.get("snippet", {})
                    account_id = channel.get("id")
                    account_name = snippet.get("title", "YouTube Channel")
                    account_username = snippet.get("customUrl", "")
                    account_avatar = snippet.get("thumbnails", {}).get("default", {}).get("url", "")
                else:
                    account_id = secrets.token_hex(8)
                    account_name = "YouTube Channel"
                    account_username = ""
                    account_avatar = ""
                    
            elif platform == "instagram":
                # Instagram Graph API: authenticate via Facebook, get Instagram Business Account
                token_response = await client.get(config["token_url"], params={
                    "client_id": INSTAGRAM_CLIENT_ID,
                    "client_secret": INSTAGRAM_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                user_access_token = token_data.get("access_token")
                
                if not user_access_token:
                    raise Exception(f"No access token: {token_data}")
                
                # Get Facebook Pages the user manages
                pages_response = await client.get(
                    f"https://graph.facebook.com/v18.0/me/accounts?access_token={user_access_token}"
                )
                pages_data = pages_response.json()
                pages = pages_data.get("data", [])
                
                if not pages:
                    raise Exception("No Facebook Pages found. You need a Facebook Page connected to an Instagram Business account.")
                
                # Find Instagram Business Account connected to any page
                instagram_account = None
                page_access_token = None
                
                for page in pages:
                    page_id = page.get("id")
                    page_token = page.get("access_token")
                    
                    # Check if this page has an Instagram Business Account
                    ig_response = await client.get(
                        f"https://graph.facebook.com/v18.0/{page_id}?fields=instagram_business_account&access_token={page_token}"
                    )
                    ig_data = ig_response.json()
                    
                    if "instagram_business_account" in ig_data:
                        ig_account_id = ig_data["instagram_business_account"]["id"]
                        
                        # Get Instagram account details
                        ig_details_response = await client.get(
                            f"https://graph.facebook.com/v18.0/{ig_account_id}?fields=id,username,name,profile_picture_url&access_token={page_token}"
                        )
                        ig_details = ig_details_response.json()
                        
                        instagram_account = ig_details
                        page_access_token = page_token
                        break
                
                if not instagram_account:
                    raise Exception("No Instagram Business account found connected to your Facebook Pages. Connect your Instagram Business/Creator account to a Facebook Page first.")
                
                account_id = instagram_account.get("id")
                account_name = instagram_account.get("name") or instagram_account.get("username", "Instagram Account")
                account_username = instagram_account.get("username", "")
                account_avatar = instagram_account.get("profile_picture_url", "")
                access_token = page_access_token  # Use Page token for API calls
                
            elif platform == "facebook":
                token_response = await client.get(config["token_url"], params={
                    "client_id": FACEBOOK_CLIENT_ID,
                    "client_secret": FACEBOOK_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                user_access_token = token_data.get("access_token")

                if not user_access_token:
                    raise Exception(f"No access token returned: {token_data}")

                # Facebook Reels require a Page token, not a user token.
                # Fetch the user's Pages and use the first one.
                pages_response = await client.get(
                    "https://graph.facebook.com/v18.0/me/accounts",
                    params={"access_token": user_access_token, "fields": "id,name,access_token,picture"},
                )
                pages_data = pages_response.json()
                pages = pages_data.get("data", [])

                if not pages:
                    raise Exception(
                        "No Facebook Pages found. You need a Facebook Page to publish Reels. "
                        "Create a Page at facebook.com/pages/create and try again."
                    )

                # Use the first Page
                page = pages[0]
                account_id    = page["id"]                                         # Page ID
                account_name  = page.get("name", "Facebook Page")
                account_username = ""
                account_avatar   = page.get("picture", {}).get("data", {}).get("url", "")
                access_token     = page["access_token"]                            # Page token
            
            # Refuse to persist "ghost" connections with no identity
            if not account_id or (isinstance(account_id, str) and account_id.strip() == ""):
                raise Exception("Provider did not return account_id; refusing to store token (prevents phantom accounts).")

            # Store the token — include platform-specific IDs in the blob so
            # publish_stage can read them without needing a separate DB lookup.
            blob_payload = {
                "access_token": access_token,
                "refresh_token": token_data.get("refresh_token"),
                "expires_at": token_data.get("expires_in"),
            }
            if platform == "instagram" and account_id:
                blob_payload["ig_user_id"] = str(account_id)
            if platform == "facebook" and account_id:
                blob_payload["page_id"] = str(account_id)
            token_blob = encrypt_blob(blob_payload)
            
            async with db_pool.acquire() as conn:
                # Check if account already connected
                existing = await conn.fetchrow(
                    "SELECT id FROM platform_tokens WHERE user_id = $1 AND platform = $2 AND account_id = $3",
                    user_id, platform, account_id
                )
                
                if existing:
                    await conn.execute("""
                        UPDATE platform_tokens SET token_blob = $1, account_name = $2, account_username = $3, 
                        account_avatar = $4, updated_at = NOW() WHERE id = $5
                    """, json.dumps(token_blob), account_name, account_username, account_avatar, existing["id"])
                    connect_action = "PLATFORM_RECONNECTED"
                else:
                    await conn.execute("""
                        INSERT INTO platform_tokens (user_id, platform, account_id, account_name, account_username, account_avatar, token_blob)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, user_id, platform, account_id, account_name, account_username, account_avatar, json.dumps(token_blob))
                    connect_action = "PLATFORM_CONNECTED"

                await log_system_event(conn, user_id=str(user_id), action=connect_action,
                                       event_category="PLATFORM", resource_type="platform",
                                       resource_id=f"{platform}:{account_id}",
                                       details={"platform": platform, "account_name": account_name,
                                                "account_username": account_username})
            
            return popup_response(True, platform)
    
    except Exception as e:
        # Sanitize error before logging — exception message may contain tokens or secrets
        err_type = type(e).__name__
        err_safe = str(e)
        # Strip anything that looks like a token (long alphanumeric strings)
        import re as _re
        err_safe = _re.sub(r'[A-Za-z0-9_-]{40,}', '***', err_safe)
        logger.error(f"OAuth callback error for {platform} ({err_type}): {err_safe}")
        # Return a stable user-facing message — never echo raw exception text to the browser
        user_msg = str(e) if len(str(e)) < 200 and "token" not in str(e).lower() else "Connection failed. Please try again."
        return popup_response(False, platform, user_msg)

# ============================================================
# Billing
# ============================================================
@app.post("/api/billing/checkout")
async def create_checkout(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")

    async with db_pool.acquire() as conn:
        # ── Double-subscribe guard ────────────────────────────────────
        if data.kind == "subscription":
            existing_sub_id  = user.get("stripe_subscription_id")
            existing_status  = user.get("subscription_status")
            if existing_sub_id and existing_status in ("active", "trialing"):
                # User already has an active sub — send straight to billing portal
                # so they can upgrade/downgrade there rather than creating a duplicate
                try:
                    portal = stripe.billing_portal.Session.create(
                        customer=user["stripe_customer_id"],
                        return_url=f"{FRONTEND_URL}/settings.html#billing",
                    )
                    return {"checkout_url": portal.url, "session_id": None, "portal_redirect": True}
                except Exception:
                    pass  # Fall through to new checkout if portal fails

        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user.get("name") or user["email"])
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])

    prices = stripe.Price.list(lookup_keys=[data.lookup_key], active=True)
    if not prices.data:
        raise HTTPException(400, f"Price not found for lookup_key: {data.lookup_key}. Run stripe_setup.py to create prices.")

    if data.kind == "subscription":
        # Resolve trial days from entitlements (7 days for all paid tiers)
        tier = STRIPE_LOOKUP_TO_TIER.get(data.lookup_key, "free")
        ent  = get_entitlements_for_tier(tier)
        trial_days = ent.trial_days  # 0 for free/internal, 7 for paid

        session_params = dict(
            customer              = customer_id,
            line_items            = [{"price": prices.data[0].id, "quantity": 1}],
            mode                  = "subscription",
            success_url           = STRIPE_SUCCESS_URL,
            cancel_url            = STRIPE_CANCEL_URL,
            allow_promotion_codes = True,
            metadata              = {"user_id": str(user["id"]), "tier": tier},
        )
        if trial_days > 0:
            session_params["subscription_data"] = {
                "trial_period_days": trial_days,
                "metadata": {"user_id": str(user["id"]), "tier": tier},
            }

        session = stripe.checkout.Session.create(**session_params)

    else:  # topup / one-time payment
        product = TOPUP_PRODUCTS.get(data.lookup_key, {})
        if not product:
            raise HTTPException(400, f"Unknown topup product: {data.lookup_key}")

        session = stripe.checkout.Session.create(
            customer    = customer_id,
            line_items  = [{"price": prices.data[0].id, "quantity": 1}],
            mode        = "payment",
            success_url = STRIPE_SUCCESS_URL,
            cancel_url  = STRIPE_CANCEL_URL,
            metadata    = {
                "user_id": str(user["id"]),
                "wallet":  product.get("wallet", "put"),
                "amount":  str(product.get("amount", 0)),
            },
        )

    return {"checkout_url": session.url, "session_id": session.id}

@app.post("/api/billing/portal")
async def create_portal(user: dict = Depends(get_current_user)):
    if not user.get("stripe_customer_id"): raise HTTPException(400, "No billing account")
    session = stripe.billing_portal.Session.create(customer=user["stripe_customer_id"], return_url=f"{FRONTEND_URL}/settings.html")
    return {"portal_url": session.url}

@app.get("/api/billing/session")
async def get_billing_session(
    session_id: str = Query(..., description="Stripe checkout session ID (cs_test_* or cs_live_*)"),
    user: dict = Depends(get_current_user),
):
    """
    Read a Stripe Checkout Session directly from the Stripe API.
    Used by billing/success.html to render the confirmation screen.
    Works for both test (cs_test_*) and live (cs_live_*) sessions.
    The session_id prefix reveals the mode — no separate flag needed.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")

    try:
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["subscription", "subscription.items.data.price", "line_items"],
        )
    except stripe.error.InvalidRequestError as e:
        raise HTTPException(404, f"Stripe session not found: {e}")
    except stripe.error.AuthenticationError:
        raise HTTPException(503, "Stripe authentication failed — check STRIPE_SECRET_KEY")
    except Exception as e:
        raise HTTPException(502, f"Stripe API error: {e}")

    # Security: session must belong to this user (metadata.user_id match)
    meta_user_id = (sess.get("metadata") or {}).get("user_id")
    if meta_user_id and str(meta_user_id) != str(user.get("id")):
        raise HTTPException(403, "This session does not belong to your account")

    mode           = sess.get("mode")            # "subscription" | "payment"
    payment_status = sess.get("payment_status")  # "paid" | "unpaid" | "no_payment_required"
    amount_total   = (sess.get("amount_total") or 0) / 100
    currency       = (sess.get("currency") or "usd").upper()

    # ── Subscription fields ──────────────────────────────────────────
    tier                  = None
    lookup_key            = None
    plan_name             = None
    sub_status            = None
    trial_end_ts          = None
    current_period_end_ts = None

    if mode == "subscription":
        sub = sess.get("subscription")
        if isinstance(sub, dict):
            sub_status            = sub.get("status")        # active | trialing | past_due
            trial_end_ts          = sub.get("trial_end")     # unix ts or None
            current_period_end_ts = sub.get("current_period_end")
            try:
                price      = sub["items"]["data"][0]["price"]
                lookup_key = price.get("lookup_key", "")
                tier       = STRIPE_LOOKUP_TO_TIER.get(lookup_key)
                if not tier:
                    # Fallback: try to match by product name
                    prod_id = price.get("product")
                    if prod_id:
                        prod = stripe.Product.retrieve(prod_id)
                        plan_name = prod.get("name", "")
            except Exception:
                pass

    # ── Topup fields ─────────────────────────────────────────────────
    topup_wallet = None
    topup_amount = None
    if mode == "payment":
        meta         = sess.get("metadata") or {}
        topup_wallet = meta.get("wallet")        # "put" | "aic"
        topup_amount = meta.get("amount")        # token count as string

    # Resolve display name
    if tier:
        cfg       = TIER_CONFIG.get(tier, {})
        plan_name = cfg.get("name", tier.replace("_", " ").title())
    elif not plan_name:
        plan_name = "Your Plan"

    return {
        "session_id":          session_id,
        "mode":                mode,
        "payment_status":      payment_status,
        "amount_total":        amount_total,
        "currency":            currency,
        "tier":                tier,
        "plan_name":           plan_name,
        "lookup_key":          lookup_key,
        "subscription_status": sub_status,
        "trial_end":           trial_end_ts,            # unix timestamp or None
        "current_period_end":  current_period_end_ts,   # unix timestamp or None
        "topup_wallet":        topup_wallet,
        "topup_amount":        int(topup_amount) if topup_amount else None,
        "is_test_mode":        session_id.startswith("cs_test_"),
        "billing_mode":        BILLING_MODE,
    }


@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Stripe billing webhook — idempotent, signature-verified."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Invalid signature: {e}")

    etype = event.type

    # ── checkout.session.completed ──────────────────────────────────────
    if etype == "checkout.session.completed":
        session = event.data.object
        user_id = session.metadata.get("user_id")
        if not user_id:
            return {"status": "no_user_id"}

        async with db_pool.acquire() as conn:
            user_row = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
            email = user_row["email"] if user_row else ""

            if session.mode == "subscription":
                sub = stripe.Subscription.retrieve(session.subscription)
                lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
                tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, "free")
                ent  = get_entitlements_for_tier(tier)
                status = sub.status

                period_start = datetime.fromtimestamp(sub.current_period_start, tz=timezone.utc) if sub.get("current_period_start") else _now_utc()
                period_end   = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
                trial_end    = datetime.fromtimestamp(sub.trial_end, tz=timezone.utc) if sub.get("trial_end") else None

                await conn.execute("""
                    UPDATE users SET
                        subscription_tier      = $1,
                        stripe_subscription_id = $2,
                        subscription_status    = $3,
                        current_period_end     = $4,
                        trial_end              = $5,
                        updated_at             = NOW()
                    WHERE id = $6
                """, tier, session.subscription, status, period_end, trial_end, user_id)

                # Seed wallet for first period / trial — deduped by invoice id
                refill_ref = sub.get("latest_invoice") or session.id
                await _do_monthly_refill(conn, user_id, tier, ent, refill_ref, period_start, period_end)

                amount = (session.amount_total or 0) / 100
                await conn.execute(
                    "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                    "VALUES ($1,$2,'subscription',$3,$4) ON CONFLICT DO NOTHING",
                    user_id, amount, session.id, tier
                )
                background_tasks.add_task(notify_mrr, amount, email, tier, status)

            elif session.mode == "payment":
                wallet_type   = session.metadata.get("wallet", "put")
                amount_tokens = int(session.metadata.get("amount", 0))
                if amount_tokens > 0:
                    await credit_wallet(conn, user_id, wallet_type, amount_tokens, "topup_purchase", session.id)
                    amount = (session.amount_total or 0) / 100
                    await conn.execute(
                        "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                        "VALUES ($1,$2,'topup',$3,$4) ON CONFLICT DO NOTHING",
                        user_id, amount, session.id, f"{wallet_type}_{amount_tokens}"
                    )
                    background_tasks.add_task(notify_topup, amount, email, wallet_type, amount_tokens)

    # ── invoice.paid — monthly wallet refill on every renewal ──────────
    elif etype == "invoice.paid":
        invoice = event.data.object
        sub_id  = invoice.get("subscription")
        if not sub_id:
            return {"status": "no_subscription"}

        async with db_pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT id, email, subscription_tier FROM users WHERE stripe_subscription_id = $1", sub_id
            )
            if not user_row:
                logger.warning(f"invoice.paid: no user for subscription {sub_id}")
                return {"status": "user_not_found"}

            user_id = str(user_row["id"])
            email   = user_row["email"]
            sub = stripe.Subscription.retrieve(sub_id)
            lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
            tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, user_row["subscription_tier"] or "free")
            ent  = get_entitlements_for_tier(tier)

            period_start = datetime.fromtimestamp(invoice.period_start, tz=timezone.utc)
            period_end   = datetime.fromtimestamp(invoice.period_end, tz=timezone.utc)
            invoice_id   = invoice.id

            await conn.execute("""
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = 'active',
                    current_period_end  = $2,
                    updated_at          = NOW()
                WHERE id = $3
            """, tier, period_end, user_id)

            # Monthly wallet refill — deduped by invoice_id
            await _do_monthly_refill(conn, user_id, tier, ent, invoice_id, period_start, period_end)

            amount = (invoice.amount_paid or 0) / 100
            await conn.execute(
                "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                "VALUES ($1,$2,'renewal',$3,$4) ON CONFLICT DO NOTHING",
                user_id, amount, invoice_id, tier
            )
            background_tasks.add_task(notify_mrr, amount, email, tier, "renewal")

    # ── subscription.updated — status changes, upgrades, downgrades ────
    elif etype == "customer.subscription.updated":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            lookup_key = sub["items"]["data"][0]["price"].get("lookup_key", "")
            tier = STRIPE_LOOKUP_TO_TIER.get(lookup_key, "free")
            period_end = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
            await conn.execute("""
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = $2,
                    current_period_end  = $3,
                    updated_at          = NOW()
                WHERE stripe_subscription_id = $4
            """, tier, sub.status, period_end, sub.id)

    # ── subscription.deleted — downgrade to free, keep balance ─────────
    elif etype == "customer.subscription.deleted":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET
                    subscription_tier   = 'free',
                    subscription_status = 'cancelled',
                    updated_at          = NOW()
                WHERE stripe_subscription_id = $1
            """, sub.id)

    return {"status": "ok"}


async def _do_monthly_refill(conn, user_id, tier, ent, invoice_id, period_start, period_end):
    """Credit monthly PUT+AIC. Deduped by invoice_id — safe to call on webhook retry."""
    # Check dedup table
    try:
        existing = await conn.fetchrow(
            "SELECT invoice_id FROM stripe_invoice_log WHERE invoice_id = $1", invoice_id
        )
        if existing:
            logger.info(f"Monthly refill already processed for {invoice_id}, skipping.")
            return False
    except Exception:
        pass  # Table may not exist yet — proceed, credit_wallet is idempotent enough

    put_amount = ent.put_monthly
    aic_amount = ent.aic_monthly

    if put_amount > 0:
        await conn.execute(
            "UPDATE wallets SET put_balance = put_balance + $1, updated_at = NOW() WHERE user_id = $2",
            put_amount, user_id
        )
        await ledger_entry(conn, user_id, "put", put_amount, "monthly_refill",
                           stripe_event_id=invoice_id)

    if aic_amount > 0:
        await conn.execute(
            "UPDATE wallets SET aic_balance = aic_balance + $1, updated_at = NOW() WHERE user_id = $2",
            aic_amount, user_id
        )
        await ledger_entry(conn, user_id, "aic", aic_amount, "monthly_refill",
                           stripe_event_id=invoice_id)

    try:
        await conn.execute("""
            INSERT INTO stripe_invoice_log
                (invoice_id, user_id, tier_slug, put_credited, aic_credited, period_start, period_end)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (invoice_id) DO NOTHING
        """, invoice_id, user_id, tier, put_amount, aic_amount, period_start, period_end)
    except Exception:
        pass  # Non-critical — dedup log insert failure doesn't break billing

    logger.info(f"Monthly refill: user={user_id} tier={tier} +{put_amount} PUT +{aic_amount} AIC invoice={invoice_id}")
    return True


# ============================================================
# TikTok Webhooks
# Endpoint: https://auth.uploadm8.com/api/webhooks/tiktok
#
# Register this URL in TikTok Developer Portal →
#   your app → Settings → Webhooks → Callback URL
#
# Signature spec (TikTok-Signature header):
#   t=<unix_ts>,s=<hex>
#   signed_payload  = f"{timestamp}.{raw_body}"
#   key             = TIKTOK_WEBHOOK_SECRET (= TIKTOK_CLIENT_SECRET by default)
#   algo            = HMAC-SHA256
# ============================================================

import hmac as _hmac

TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC = 300   # reject events older than 5 minutes


def _verify_tiktok_signature(raw_body: bytes, header: str, secret: str) -> tuple[bool, str]:
    """
    Parse and verify the Tiktok-Signature header.

    Returns (ok: bool, reason: str).
    ok=True  → signature is valid and timestamp is fresh.
    ok=False → verification failed (reason explains why).

    If TIKTOK_WEBHOOK_SECRET is empty the check is skipped and we return
    (True, "sig-check-skipped-no-secret") so the developer can still receive
    events during initial setup without crashing.
    """
    if not secret:
        return True, "sig-check-skipped-no-secret"

    if not header:
        return False, "missing-Tiktok-Signature-header"

    # Parse  "t=1633174587,s=18494715036ac441..."
    parts: dict[str, str] = {}
    for segment in header.split(","):
        if "=" in segment:
            k, _, v = segment.partition("=")
            parts[k.strip()] = v.strip()

    ts_str = parts.get("t", "")
    sig_received = parts.get("s", "")

    if not ts_str or not sig_received:
        return False, f"malformed-header:{header[:80]}"

    try:
        ts = int(ts_str)
    except ValueError:
        return False, f"non-numeric-timestamp:{ts_str}"

    # Replay-attack protection
    age = abs(int(time.time()) - ts)
    if age > TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC:
        return False, f"timestamp-too-old:{age}s"

    # Compute expected signature
    signed_payload = f"{ts_str}.".encode() + raw_body
    expected = _hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()

    if not _hmac.compare_digest(expected, sig_received):
        return False, "signature-mismatch"

    return True, "ok"


async def _handle_tiktok_event(event_type: str, payload: dict, user_openid: str) -> str:
    """
    Process a verified TikTok webhook event in the background.
    Returns a short string describing what was done (stored in handling_notes).
    """
    notes = f"event={event_type}"

    try:
        async with db_pool.acquire() as conn:

            # ── video.publish.completed ────────────────────────────────────
            if event_type == "video.publish.completed":
                # content may contain share_id or video_id — store it in
                # platform_results and mark the upload completed if we can
                # match it by TikTok open_id.
                content = payload.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"raw": content}

                share_id = content.get("share_id", "")
                video_id = content.get("video_id", share_id)

                # Find the most recent tiktok upload for this open_id that is
                # still in a processing/queued state so we can mark it done.
                upload = await conn.fetchrow(
                    """
                    SELECT u.id, u.user_id, u.platform_results
                    FROM uploads u
                    JOIN platform_tokens pt
                        ON u.user_id = pt.user_id
                       AND pt.platform = 'tiktok'
                       AND pt.account_id = $1
                       AND pt.revoked_at IS NULL
                    WHERE u.status NOT IN ('completed', 'succeeded', 'failed', 'cancelled')
                    ORDER BY u.created_at DESC
                    LIMIT 1
                    """,
                    user_openid,
                )

                if upload:
                    existing = _safe_json(upload["platform_results"], {})
                    existing["tiktok"] = {
                        "status": "published",
                        "video_id": video_id,
                        "share_id": share_id,
                        "published_at": _now_utc().isoformat(),
                    }
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status           = 'completed',
                            completed_at     = NOW(),
                            platform_results = $1,
                            updated_at       = NOW()
                        WHERE id = $2
                        """,
                        json.dumps(existing),
                        upload["id"],
                    )
                    notes += f" upload={upload['id']} marked=completed video_id={video_id}"
                else:
                    notes += f" no-matching-upload-found openid={user_openid}"

            # ── video.upload.failed ────────────────────────────────────────
            elif event_type == "video.upload.failed":
                content = payload.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"raw": content}

                share_id = content.get("share_id", "")

                upload = await conn.fetchrow(
                    """
                    SELECT u.id, u.user_id, u.platform_results,
                           u.put_reserved, u.aic_reserved
                    FROM uploads u
                    JOIN platform_tokens pt
                        ON u.user_id = pt.user_id
                       AND pt.platform = 'tiktok'
                       AND pt.account_id = $1
                       AND pt.revoked_at IS NULL
                    WHERE u.status NOT IN ('completed', 'succeeded', 'failed', 'cancelled')
                    ORDER BY u.created_at DESC
                    LIMIT 1
                    """,
                    user_openid,
                )

                if upload:
                    existing = _safe_json(upload["platform_results"], {})
                    existing["tiktok"] = {
                        "status": "failed",
                        "share_id": share_id,
                        "failed_at": _now_utc().isoformat(),
                    }
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status           = 'failed',
                            error_code       = 'tiktok_upload_failed',
                            error_detail     = 'TikTok reported upload failure via webhook',
                            platform_results = $1,
                            updated_at       = NOW()
                        WHERE id = $2
                        """,
                        json.dumps(existing),
                        upload["id"],
                    )
                    # Refund reserved tokens so the user isn't charged
                    if upload["put_reserved"] or upload["aic_reserved"]:
                        await refund_tokens(
                            conn,
                            str(upload["user_id"]),
                            upload["put_reserved"] or 0,
                            upload["aic_reserved"] or 0,
                            str(upload["id"]),
                        )
                    notes += f" upload={upload['id']} marked=failed tokens-refunded"
                else:
                    notes += f" no-matching-upload-found openid={user_openid}"

            # ── authorization.removed ──────────────────────────────────────
            elif event_type == "authorization.removed":
                # TikTok has already revoked the token on their side; we just
                # need to purge the platform_tokens row and log the disconnect.
                rows = await conn.fetch(
                    """
                    UPDATE platform_tokens
                    SET revoked_at = NOW()
                    WHERE platform = 'tiktok'
                      AND account_id = $1
                      AND revoked_at IS NULL
                    RETURNING id, user_id, account_id, account_name
                    """,
                    user_openid,
                )
                for row in rows:
                    # Hard-delete after marking revoked
                    await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])
                    await conn.execute(
                        """
                        INSERT INTO platform_disconnect_log
                            (user_id, platform, account_id, account_name,
                             revoked_at_provider, provider_revoke_error,
                             initiated_by)
                        VALUES ($1, 'tiktok', $2, $3, TRUE, NULL, 'tiktok_webhook')
                        """,
                        str(row["user_id"]),
                        row["account_id"],
                        row["account_name"],
                    )
                    notes += f" purged_token={row['id']}"

                if not rows:
                    notes += f" no-active-token-found openid={user_openid}"

            else:
                notes += " unhandled-event-type"

    except Exception as exc:
        notes += f" ERROR:{exc}"
        logger.error(f"[tiktok-webhook] background handler error: {exc}", exc_info=True)

    return notes


# ── GET /api/webhooks/tiktok  (URL challenge / health check) ──────────────────
@app.get("/api/webhooks/tiktok")
async def tiktok_webhook_challenge(challenge: Optional[str] = Query(None)):
    """
    TikTok calls this as a GET with ?challenge=<token> to verify the endpoint
    before activating the webhook subscription.  We must echo the challenge
    back as plain text with a 200 status.

    Also returns 200 (no body) if called with no challenge — useful for
    uptime checks.
    """
    if challenge:
        # TikTok expects the exact challenge value as the plain-text response body
        return JSONResponse(content=challenge, media_type="text/plain")
    return JSONResponse(content={"status": "tiktok-webhook-endpoint-ok"})


# ── POST /api/webhooks/tiktok  (live event receiver) ─────────────────────────
@app.post("/api/webhooks/tiktok")
async def tiktok_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives event notifications from TikTok Open Platform.

    Security
    --------
    1.  The raw body is read BEFORE any JSON parsing so the HMAC is computed
        over exactly the bytes TikTok signed.
    2.  The Tiktok-Signature header is verified with a constant-time compare.
    3.  Replay attacks are blocked — events older than 5 minutes are rejected.
    4.  We return 200 OK immediately and process the event in a BackgroundTask
        so TikTok never times out waiting for us.
    5.  Every event (including rejected ones) is logged to tiktok_webhook_events
        for auditability.

    TikTok docs: https://developers.tiktok.com/doc/webhooks-verification
    """
    # ── 1. Read raw body ───────────────────────────────────────────────────
    raw_body = await request.body()

    # ── 2. Verify signature ────────────────────────────────────────────────
    sig_header = request.headers.get("Tiktok-Signature", "")
    sig_ok, sig_reason = _verify_tiktok_signature(raw_body, sig_header, TIKTOK_WEBHOOK_SECRET)

    if not sig_ok:
        logger.warning(f"[tiktok-webhook] signature rejected: {sig_reason} | header={sig_header[:80]}")
        # Still 200 so TikTok doesn't keep retrying a bad delivery;
        # we just won't process the event.
        return JSONResponse(
            status_code=200,
            content={"status": "rejected", "reason": sig_reason},
        )

    # ── 3. Parse JSON payload ──────────────────────────────────────────────
    try:
        payload = json.loads(raw_body)
    except Exception:
        logger.warning("[tiktok-webhook] non-JSON body")
        return JSONResponse(status_code=200, content={"status": "ok"})

    event_type   = payload.get("event", "unknown")
    user_openid  = payload.get("user_openid", "")
    create_time  = payload.get("create_time")
    client_key   = payload.get("client_key", "")

    # content field is sometimes a JSON string, sometimes an object
    content_raw = payload.get("content", {})
    if isinstance(content_raw, str):
        try:
            content_parsed = json.loads(content_raw)
        except Exception:
            content_parsed = {"raw": content_raw}
    else:
        content_parsed = content_raw

    logger.info(
        f"[tiktok-webhook] event={event_type} openid={user_openid} "
        f"sig_ok={sig_ok} sig_reason={sig_reason}"
    )

    # ── 4. Log the raw event immediately ──────────────────────────────────
    async def _persist_event(notes: str):
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO tiktok_webhook_events
                        (client_key, event, create_time, user_openid,
                         content, raw_body, sig_verified, handling_notes)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    """,
                    client_key,
                    event_type,
                    create_time,
                    user_openid,
                    json.dumps(content_parsed),
                    raw_body.decode(errors="replace"),
                    sig_ok,
                    notes,
                )
        except Exception as exc:
            logger.error(f"[tiktok-webhook] failed to persist event log: {exc}")

    # ── 5. Dispatch handler in background ─────────────────────────────────
    async def _process_and_log():
        notes = await _handle_tiktok_event(event_type, payload, user_openid)
        await _persist_event(notes)

    background_tasks.add_task(_process_and_log)

    # ── 6. Return 200 immediately ──────────────────────────────────────────
    return JSONResponse(status_code=200, content={"status": "ok"})


# ── Admin: inspect TikTok webhook event log ───────────────────────────────────
@app.get("/api/admin/tiktok-webhook-log")
async def admin_tiktok_webhook_log(
    limit: int = Query(50, le=500),
    offset: int = Query(0),
    event: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Read the tiktok_webhook_events log (admin only)."""
    if user.get("role") not in ("admin", "master_admin"):
        raise HTTPException(403, "Admin only")

    async with db_pool.acquire() as conn:
        where = "WHERE event = $3" if event else ""
        params = [limit, offset] + ([event] if event else [])
        rows = await conn.fetch(
            f"""
            SELECT id, client_key, event, create_time, user_openid,
                   content, processed_at, sig_verified, handling_notes
            FROM tiktok_webhook_events
            {where}
            ORDER BY processed_at DESC
            LIMIT $1 OFFSET $2
            """,
            *params,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM tiktok_webhook_events" + (" WHERE event=$1" if event else ""),
            *([event] if event else []),
        )

    return {
        "total": total,
        "records": [
            {
                **dict(r),
                "id": str(r["id"]),
                "processed_at": r["processed_at"].isoformat() if r["processed_at"] else None,
                "content": _safe_json(r["content"], {}),
            }
            for r in rows
        ],
    }


# ============================================================
# Facebook / Instagram Webhooks
# ============================================================
# Meta calls GET to verify the endpoint (hub challenge), then POST for events.
#
# Setup in Meta Developer Console:
#   Callback URL : https://auth.uploadm8.com/api/webhooks/facebook
#   Verify Token : value of FACEBOOK_WEBHOOK_VERIFY_TOKEN env var
#
# Required env var:
#   FACEBOOK_WEBHOOK_VERIFY_TOKEN  — any secret string you set in the console
#
# Meta docs: https://developers.facebook.com/docs/graph-api/webhooks/getting-started
# ============================================================

FACEBOOK_WEBHOOK_VERIFY_TOKEN = os.environ.get("FACEBOOK_WEBHOOK_VERIFY_TOKEN", "")


@app.get("/api/webhooks/facebook")
async def facebook_webhook_challenge(
    hub_mode: Optional[str]      = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str]  = Query(None, alias="hub.challenge"),
):
    """
    Meta calls this GET endpoint when you click 'Verify and save' in the
    developer console.  We must:
      1. Confirm hub.mode == 'subscribe'
      2. Confirm hub.verify_token matches our secret
      3. Return hub.challenge as plain text with HTTP 200
    """
    if hub_mode == "subscribe" and hub_verify_token:
        expected = FACEBOOK_WEBHOOK_VERIFY_TOKEN
        if expected and hub_verify_token == expected:
            logger.info("[facebook-webhook] challenge verified OK")
            return PlainTextResponse(hub_challenge or "")
        else:
            logger.warning(
                f"[facebook-webhook] verify token mismatch — "
                f"got={hub_verify_token[:20]}… expected={'(not set)' if not expected else '***'}"
            )
            raise HTTPException(403, "Verify token mismatch")

    # Health check (no params)
    return JSONResponse({"status": "facebook-webhook-endpoint-ok"})


@app.post("/api/webhooks/facebook")
async def facebook_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives real-time event notifications from Meta (Facebook & Instagram).

    Security
    --------
    Signature is verified using X-Hub-Signature-256: sha256=<hmac>
    computed with META_APP_SECRET as the key over the raw request body.
    Events that fail signature verification are rejected with 403.
    We return 200 immediately and process in a BackgroundTask.

    Subscribed fields to configure in Meta console (under 'Webhook Fields'):
      - For Page events: feed, video_feeds
      - For Instagram: mentions, story_insights, comments, live_comments
    """
    raw_body = await request.body()

    # ── Verify signature ────────────────────────────────────────────────────
    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if META_APP_SECRET and sig_header:
        import hmac as _hmac
        expected_sig = "sha256=" + _hmac.new(
            META_APP_SECRET.encode(), raw_body, hashlib.sha256
        ).hexdigest()
        if not _hmac.compare_digest(expected_sig, sig_header):
            logger.warning(f"[facebook-webhook] signature mismatch — header={sig_header[:60]}")
            raise HTTPException(403, "Invalid signature")
    elif META_APP_SECRET and not sig_header:
        # Signature header missing entirely — reject in production
        logger.warning("[facebook-webhook] missing X-Hub-Signature-256 header")
        raise HTTPException(400, "Missing signature")

    # ── Parse payload ───────────────────────────────────────────────────────
    try:
        payload = json.loads(raw_body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    object_type = payload.get("object", "")
    entries     = payload.get("entry", []) or []

    logger.info(
        f"[facebook-webhook] received object={object_type} entries={len(entries)}"
    )

    async def _process_fb_events():
        for entry in entries:
            entry_id   = entry.get("id", "")
            changes    = entry.get("changes", []) or []
            messaging  = entry.get("messaging", []) or []

            for change in changes:
                field  = change.get("field", "")
                value  = change.get("value", {}) or {}

                # ── Video published (Page feed) ─────────────────────────
                if object_type == "page" and field in ("feed", "video_feeds"):
                    item = value.get("item", "")
                    verb = value.get("verb", "")
                    video_id = value.get("video_id") or value.get("post_id", "")
                    logger.info(
                        f"[facebook-webhook] page={entry_id} field={field} "
                        f"item={item} verb={verb} video_id={video_id}"
                    )

                # ── Instagram media / comments ──────────────────────────
                elif object_type == "instagram":
                    media_id = (
                        value.get("media_id")
                        or value.get("id")
                        or value.get("item_id", "")
                    )
                    logger.info(
                        f"[facebook-webhook] instagram field={field} "
                        f"media_id={media_id} value_keys={list(value.keys())}"
                    )

            # ── (Optional) Messenger events — ignored for now ───────────
            if messaging:
                logger.debug(
                    f"[facebook-webhook] messaging events={len(messaging)} (not handled)"
                )

    background_tasks.add_task(_process_fb_events)

    # Meta requires a fast 200 response
    return JSONResponse({"status": "ok"})


# ============================================================
# Analytics
# ============================================================
@app.get("/api/analytics")
async def get_analytics(range: str = "30d", user: dict = Depends(get_current_user)):
    # 'all' maps to 1y (largest supported window). 'partial' = at least one platform succeeded.
    minutes = {"30m": 30, "1h": 60, "6h": 360, "12h": 720, "1d": 1440,
               "7d": 10080, "30d": 43200, "6m": 262800, "1y": 525600, "all": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)

    async with db_pool.acquire() as conn:
        try:
            stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN ('completed','succeeded','partial') THEN 1 ELSE 0 END)::int AS completed,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes,
                   COALESCE(SUM(put_spent), 0)::int AS put_used,
                   COALESCE(SUM(aic_spent), 0)::int AS aic_used
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            """, user["id"], since)
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise
            stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN ('completed','succeeded','partial') THEN 1 ELSE 0 END)::int AS completed,
                   0::bigint AS views, 0::bigint AS likes,
                   0::int AS put_used, 0::int AS aic_used
            FROM uploads WHERE user_id = $1 AND created_at >= $2
            """, user["id"], since)

        daily = await conn.fetch(
            "SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads "
            "FROM uploads WHERE user_id = $1 AND created_at >= $2 "
            "GROUP BY DATE(created_at) ORDER BY date", 
            user["id"], since
        )
        platforms = await conn.fetch(
            "SELECT unnest(platforms) AS platform, COUNT(*)::int AS count "
            "FROM uploads WHERE user_id = $1 AND created_at >= $2 "
            "AND status IN ('completed','succeeded','partial') "
            "GROUP BY platform",
            user["id"], since
        )

        # ================================================================
        # TRILL TELEMETRY STATS
        # ================================================================
        trill_stats = None
        try:
            trill_data = await conn.fetchrow("""
                SELECT 
                    COUNT(*)::int AS trill_uploads,
                    COALESCE(AVG(trill_score), 0)::decimal AS avg_score,
                    COALESCE(MAX(trill_score), 0)::decimal AS max_score,
                    COALESCE(MAX(max_speed_mph), 0)::decimal AS max_speed_mph,
                    COALESCE(SUM(distance_miles), 0)::decimal AS total_distance_miles
                FROM uploads 
                WHERE user_id = $1 
                AND created_at >= $2 
                AND trill_score IS NOT NULL
            """, user["id"], since)

            if trill_data and trill_data["trill_uploads"] > 0:
                speed_buckets = await conn.fetch("""
                    SELECT speed_bucket, COUNT(*)::int AS count
                    FROM uploads
                    WHERE user_id = $1
                    AND created_at >= $2
                    AND speed_bucket IS NOT NULL
                    GROUP BY speed_bucket
                """, user["id"], since)

                bucket_counts = {
                    "gloryBoy": 0,
                    "euphoric": 0,
                    "sendIt": 0,
                    "spirited": 0,
                    "chill": 0
                }

                for bucket in speed_buckets:
                    if bucket["speed_bucket"] in bucket_counts:
                        bucket_counts[bucket["speed_bucket"]] = bucket["count"]

                trill_stats = {
                    "trill_uploads": trill_data["trill_uploads"],
                    "avg_score": float(trill_data["avg_score"]),
                    "max_score": float(trill_data["max_score"]),
                    "max_speed_mph": float(trill_data["max_speed_mph"]),
                    "total_distance_miles": float(trill_data["total_distance_miles"]),
                    "speed_buckets": bucket_counts
                }
        except Exception as e:
            logger.warning(f"Trill stats unavailable: {e}")
        # ================================================================

    result = {
        "total_uploads": stats["total"] if stats else 0,
        "completed": stats["completed"] if stats else 0,
        "views": stats["views"] if stats else 0,
        "likes": stats["likes"] if stats else 0,
        "put_used": stats["put_used"] if stats else 0,
        "aic_used": stats["aic_used"] if stats else 0,
        "daily": [{"date": str(d["date"]), "uploads": d["uploads"]} for d in daily],
        "platforms": {p["platform"]: p["count"] for p in platforms}
    }

    if trill_stats:
        result["trill"] = trill_stats

    return result

# ============================================================
# Platform Metrics — Live data from each platform API
# ============================================================
# Endpoints:
#   GET /api/analytics/platform-metrics         (live fetch + 3h cache)
#   GET /api/analytics/platform-metrics/cached  (DB cache only)
# Place this block AFTER the existing @app.get("/api/analytics") handler
# and BEFORE @app.get("/api/exports/excel").
# ============================================================

import time as _time
import asyncio as _asyncio

# In-memory cache per user  {user_id_str: {"fetched_at": float, "data": dict}}
_platform_metrics_cache: dict = {}
_PLATFORM_CACHE_TTL = 3 * 60 * 60  # 3 hours


async def _fetch_tiktok_metrics(access_token: str) -> dict:
    """TikTok Content API — video list totals + follower stats (requires video.list + user.info.stats)."""
    if not access_token:
        return {"status": "not_connected"}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # 1) Video list (requires video.list)
            resp = await client.post(
                "https://open.tiktokapis.com/v2/video/list/",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                # fields is REQUIRED — without it TikTok returns videos with every stat = 0
                json={
                    "max_count": 20,
                    "fields": ["id","title","view_count","like_count","comment_count","share_count","duration","create_time"],
                },
            )
            if resp.status_code != 200:
                logger.warning(f"TikTok video list HTTP {resp.status_code}: {resp.text[:200]}")
                return {"status": "error", "error": f"video_list_http_{resp.status_code}"}

            videos = (resp.json().get("data", {}) or {}).get("videos", []) or []

            def _i(v):
                try:
                    return int(v or 0)
                except Exception:
                    return 0

            views    = sum(_i(v.get("view_count"))    for v in videos)
            likes    = sum(_i(v.get("like_count"))    for v in videos)
            comments = sum(_i(v.get("comment_count")) for v in videos)
            shares   = sum(_i(v.get("share_count"))   for v in videos)

            durs      = [_i(v.get("duration")) for v in videos if v.get("duration") is not None]
            avg_watch = round(sum(durs) / len(durs), 1) if durs else None

            # 2) User info stats (requires user.info.stats scope)
            followers = following = total_likes = video_count = None
            ui = await client.get(
                "https://open.tiktokapis.com/v2/user/info/",
                params={"fields": "follower_count,following_count,likes_count,video_count"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if ui.status_code == 200:
                user_obj    = ((ui.json().get("data", {}) or {}).get("user", {}) or {})
                followers   = user_obj.get("follower_count")
                following   = user_obj.get("following_count")
                total_likes = user_obj.get("likes_count")
                video_count = user_obj.get("video_count")
            else:
                logger.warning(f"TikTok user.info.stats HTTP {ui.status_code}: {ui.text[:200]}")

            return {
                "status": "live",
                "analytics_source": "video.list+user.info.stats",
                "followers":   _i(followers)   if followers   is not None else None,
                "following":   _i(following)   if following   is not None else None,
                "total_likes": _i(total_likes) if total_likes is not None else None,
                "video_count": _i(video_count) if video_count is not None else len(videos),
                "views":    views,
                "likes":    likes,
                "comments": comments,
                "shares":   shares,
                "avg_watch_seconds": avg_watch,
            }

    except Exception as e:
        logger.error(f"TikTok metrics error: {e}")
        return {"status": "error", "error": str(e)}



async def _fetch_youtube_metrics(access_token: str) -> dict:
    """YouTube Data API v3 + (optional) YouTube Analytics API."""
    if not access_token:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            ch = await client.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={"part": "statistics", "mine": "true"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if ch.status_code != 200:
                return {"status": "error", "error": f"HTTP {ch.status_code}"}

            items = ch.json().get("items", []) or []
            stats = items[0].get("statistics", {}) if items else {}

            views = likes = comments = shares = 0
            avg_watch = minutes_watched = None
            analytics_source = "channel_stats_fallback"
            try:
                today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                thirty = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
                an = await client.get(
                    "https://youtubeanalytics.googleapis.com/v2/reports",
                    params={
                        "ids":        "channel==MINE",
                        "startDate":  thirty,
                        "endDate":    today,
                        # dimensions=day locks column order: [0]=day [1]=views [2]=likes
                        # [3]=comments [4]=shares [5]=avgDuration [6]=minutesWatched
                        # Without it r[0] returns a date string, not view count
                        "dimensions": "day",
                        "metrics":    "views,likes,comments,shares,averageViewDuration,estimatedMinutesWatched",
                        "sort":       "day",
                    },
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if an.status_code == 200:
                    rows = an.json().get("rows", []) or []
                    if rows:
                        views           = sum(int(r[1] or 0) for r in rows)
                        likes           = sum(int(r[2] or 0) for r in rows)
                        comments        = sum(int(r[3] or 0) for r in rows)
                        shares          = sum(int(r[4] or 0) for r in rows)
                        dur_vals        = [float(r[5]) for r in rows if r[5]]
                        avg_watch       = round(sum(dur_vals) / len(dur_vals), 1) if dur_vals else None
                        minutes_watched = int(sum(float(r[6] or 0) for r in rows))
                        analytics_source = "yt-analytics"
                    else:
                        views    = int(stats.get("viewCount",    0))
                        likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                        comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
                elif an.status_code == 403:
                    logger.warning("YouTube Analytics 403 — yt-analytics.readonly missing from token; user must reconnect")
                    views    = int(stats.get("viewCount",    0))
                    likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                    comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
                else:
                    views    = int(stats.get("viewCount",    0))
                    likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                    comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
            except Exception as ae:
                logger.warning(f"YouTube Analytics error (non-fatal): {ae}")
                views    = int(stats.get("viewCount",    0))
                likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0

            return {
                "status": "live",
                "analytics_source": analytics_source,
                "views":           views,
                "likes":           likes,
                "comments":        comments,
                "shares":          shares,
                "subscribers":     int(stats.get("subscriberCount", 0)) if "subscriberCount" in stats else 0,
                "avg_watch_seconds": avg_watch,
                "minutes_watched": minutes_watched,
                "video_count":     int(stats.get("videoCount", 0)) if "videoCount" in stats else 0,
            }
    except Exception as e:
        logger.error(f"YouTube metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_instagram_metrics(access_token: str, ig_user_id: str) -> dict:
    """Instagram Graph API — Reels insights."""
    if not access_token or not ig_user_id:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            media = await client.get(
                f"https://graph.facebook.com/v21.0/{ig_user_id}/media",
                params={"access_token": access_token, "fields": "id,media_type,timestamp", "limit": 20},
            )
            if media.status_code != 200:
                return {"status": "error", "error": f"HTTP {media.status_code}"}

            items = media.json().get("data", []) or []
            if not items:
                return {"status": "live", "views": 0, "likes": 0, "comments": 0, "saves": 0, "reach": 0, "shares": 0, "video_count": 0}

            total_views = total_likes = total_comments = 0
            total_saves = total_reach = total_shares = 0

            analytics_source = None
            for item in items[:10]:
                media_type = (item.get("media_type") or "IMAGE").upper()
                # "plays" on IMAGE/CAROUSEL silently kills the insights call
                if media_type in ("VIDEO", "REELS"):
                    metric_str = "plays,reach,saved,shares,comments,likes"
                    view_key   = "plays"
                else:
                    metric_str = "impressions,reach,saved,shares,comments,likes"
                    view_key   = "impressions"

                ins = await client.get(
                    f"https://graph.facebook.com/v21.0/{item['id']}/insights",
                    params={"access_token": access_token, "metric": metric_str},
                )
                if ins.status_code != 200:
                    continue
                analytics_source = "instagram_manage_insights"
                for m in ins.json().get("data", []) or []:
                    name = m.get("name", "")
                    vals = m.get("values", [])
                    val  = vals[-1].get("value", 0) if vals else m.get("value", 0)
                    if isinstance(val, dict):
                        val = sum(val.values())
                    val = int(val or 0)
                    if name == view_key:       total_views    += val
                    elif name == "likes":      total_likes    += val
                    elif name == "comments":   total_comments += val
                    elif name == "saved":      total_saves    += val
                    elif name == "reach":      total_reach    += val
                    elif name == "shares":     total_shares   += val

            return {
                "status": "live",
                "analytics_source": analytics_source,
                "views":    total_views,
                "likes":    total_likes,
                "comments": total_comments,
                "saves":    total_saves,
                "reach":    total_reach,
                "shares":   total_shares,
                "video_count": len(items),
            }
    except Exception as e:
        logger.error(f"Instagram metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_facebook_metrics(access_token: str, page_id: str) -> dict:
    """Facebook Graph API — Page video insights."""
    if not access_token or not page_id:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            vids = await client.get(
                f"https://graph.facebook.com/v21.0/{page_id}/videos",
                params={"access_token": access_token, "fields": "id,created_time", "limit": 15},
            )
            if vids.status_code != 200:
                return {"status": "error", "error": f"HTTP {vids.status_code}"}

            videos = vids.json().get("data", []) or []
            if not videos:
                return {"status": "live", "views": 0, "reactions": 0, "comments": 0, "shares": 0, "followers": 0, "video_count": 0}

            total_views = total_reactions = total_comments = total_shares = 0

            analytics_source = None
            for vid in videos[:10]:
                try:
                    ins = await client.get(
                        f"https://graph.facebook.com/v21.0/{vid['id']}",
                        params={
                            "access_token": access_token,
                            "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_shares,total_video_comments)",
                        },
                    )
                    if ins.status_code != 200:
                        continue
                    analytics_source = "read_insights+pages_read_engagement"
                    for m in ins.json().get("insights", {}).get("data", []) or []:
                        name = m.get("name", "")
                        vals = m.get("values", [{}])
                        val  = vals[-1].get("value", 0) if vals else 0
                        if isinstance(val, dict):
                            val = sum(val.values())
                        val = int(val or 0)
                        if   name == "total_video_views":                    total_views     += val
                        elif name == "total_video_reactions_by_type_total":  total_reactions += val
                        elif name == "total_video_shares":                   total_shares    += val
                        elif name == "total_video_comments":                 total_comments  += val
                except Exception as ve:
                    logger.warning(f"Facebook video insight error for {vid.get('id')} (skipping): {ve}")
                    continue

            followers = 0
            try:
                pg = await client.get(
                    f"https://graph.facebook.com/v21.0/{page_id}",
                    params={"access_token": access_token, "fields": "followers_count,fan_count"},
                )
                if pg.status_code == 200:
                    pg_data   = pg.json()
                    followers = pg_data.get("followers_count") or pg_data.get("fan_count") or 0
            except Exception as fe:
                logger.warning(f"Facebook followers fetch error (non-fatal): {fe}")

            return {
                "status": "live",
                "analytics_source": analytics_source,
                "views":     total_views,
                "reactions": total_reactions,
                "comments":  total_comments,
                "shares":    total_shares,
                "followers": followers,
                "video_count": len(videos),
            }
    except Exception as e:
        logger.error(f"Facebook metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _platform_metrics_db_cache_get(conn, user_id: str) -> Optional[dict]:
    try:
        row = await conn.fetchrow("SELECT fetched_at, data FROM platform_metrics_cache WHERE user_id = $1", user_id)
        if not row:
            return None
        data = row["data"]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = None
        if not isinstance(data, dict):
            return None
        out = dict(data)
        out["cached"] = True
        out["cache_source"] = "db"
        out["fetched_at"] = row["fetched_at"].isoformat() if row["fetched_at"] else out.get("fetched_at")
        return out
    except Exception:
        return None


async def _platform_metrics_db_cache_set(conn, user_id: str, output: dict) -> None:
    try:
        await conn.execute(
            """
            INSERT INTO platform_metrics_cache (user_id, fetched_at, data)
            VALUES ($1, NOW(), $2::jsonb)
            ON CONFLICT (user_id) DO UPDATE
            SET fetched_at = EXCLUDED.fetched_at,
                data = EXCLUDED.data
            """,
            user_id,
            json.dumps(output),
        )
    except Exception:
        pass


@app.get("/api/analytics/platform-metrics")
async def get_platform_metrics(force: bool = False, user: dict = Depends(get_current_user)):
    """
    Fetch live engagement metrics from all connected platform APIs.
    Cached 3 hours per user (memory + DB). Pass ?force=true to bypass cache.
    """
    user_id = str(user["id"])
    now = _time.time()

    # 1) in-memory cache
    cached = _platform_metrics_cache.get(user_id)
    if cached and not force:
        age = now - cached["fetched_at"]
        if age < _PLATFORM_CACHE_TTL:
            result = dict(cached["data"])
            result["cached"] = True
            result["cache_source"] = "memory"
            result["cache_age_minutes"] = int(age / 60)
            result["next_refresh_minutes"] = int((_PLATFORM_CACHE_TTL - age) / 60)
            return result

    # 2) DB cache (survives restarts)
    async with db_pool.acquire() as conn:
        db_cached = await _platform_metrics_db_cache_get(conn, user_id)
        if db_cached and not force:
            _platform_metrics_cache[user_id] = {"fetched_at": now, "data": db_cached}
            return db_cached

        token_rows = await conn.fetch(
            "SELECT platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )
        upload_counts = await conn.fetch(
            """SELECT unnest(platforms) AS platform, COUNT(*)::int AS cnt
               FROM uploads
               WHERE user_id = $1 AND status IN ('succeeded', 'completed', 'partial')
               GROUP BY platform""",
            user["id"],
        )

    upload_map = {r["platform"]: r["cnt"] for r in upload_counts}

    token_map: dict = {}
    for row in token_rows:
        plat = row["platform"]
        blob = row["token_blob"]
        if not blob:
            continue
        try:
            decrypted = decrypt_blob(blob)
        except Exception:
            continue
        if decrypted:
            if plat == "instagram" and not decrypted.get("ig_user_id") and row["account_id"]:
                decrypted["ig_user_id"] = str(row["account_id"])
            if plat == "facebook" and not decrypted.get("page_id") and row["account_id"]:
                decrypted["page_id"] = str(row["account_id"])
            token_map[plat] = decrypted

    async def run_tiktok():
        t = token_map.get("tiktok", {})
        return await _fetch_tiktok_metrics(t.get("access_token", ""))

    async def run_youtube():
        t = token_map.get("youtube", {})
        return await _fetch_youtube_metrics(t.get("access_token", ""))

    async def run_instagram():
        t = token_map.get("instagram", {})
        ig_id = (t.get("ig_user_id") or t.get("instagram_user_id") or t.get("instagram_page_id") or "")
        return await _fetch_instagram_metrics(t.get("access_token", ""), ig_id)

    async def run_facebook():
        t = token_map.get("facebook", {})
        page_id = (t.get("page_id") or t.get("facebook_page_id") or t.get("fb_page_id") or "")
        return await _fetch_facebook_metrics(t.get("access_token", ""), page_id)

    tasks = {}
    if "tiktok" in token_map:
        tasks["tiktok"] = run_tiktok()
    if "youtube" in token_map:
        tasks["youtube"] = run_youtube()
    if "instagram" in token_map:
        tasks["instagram"] = run_instagram()
    if "facebook" in token_map:
        tasks["facebook"] = run_facebook()

    platforms_result: dict = {}
    if tasks:
        results = await _asyncio.gather(*tasks.values(), return_exceptions=True)
        for plat, res in zip(tasks.keys(), results):
            platforms_result[plat] = {"status": "error", "error": str(res)} if isinstance(res, Exception) else res

    for plat in ["tiktok", "youtube", "instagram", "facebook"]:
        if plat not in platforms_result:
            platforms_result[plat] = {"status": "not_connected"}
        platforms_result[plat]["uploads"] = upload_map.get(plat, 0)

    output = {
        "platforms": platforms_result,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "cache_age_minutes": 0,
        "next_refresh_minutes": int(_PLATFORM_CACHE_TTL / 60),
    }

    _platform_metrics_cache[user_id] = {"fetched_at": now, "data": output}
    async with db_pool.acquire() as conn:
        await _platform_metrics_db_cache_set(conn, user_id, output)

    return output


@app.get("/api/analytics/platform-metrics/cached")
async def get_platform_metrics_cached(user: dict = Depends(get_current_user)):
    """Return DB-cached platform metrics only (no live API calls)."""
    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        cached = await _platform_metrics_db_cache_get(conn, user_id)
    if cached:
        return cached
    return {"platforms": {}, "cached": True, "cache_source": "db", "fetched_at": None}

@app.get("/api/exports/excel")
async def export_excel(type: str = "uploads", range: str = "30d", user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    if not plan.get("excel"): raise HTTPException(403, "Excel export requires Studio+ plan")
    
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
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
# ========================================@app.get("/api/analytics/overview")
async def analytics_overview(days: int = Query(30, ge=1, le=3650), user: dict = Depends(get_current_user)):
    """High-level KPI summary for analytics dashboard."""
    since = _now_utc() - timedelta(days=days)

    async with db_pool.acquire() as conn:
        # Upload KPIs (defensive against older schemas)
        try:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS uploads_total,
                    SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS uploads_completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS uploads_failed,
                    COALESCE(AVG(EXTRACT(EPOCH FROM (processing_finished_at - processing_started_at))), 0)::double precision AS avg_processing_seconds,
                    COALESCE(SUM(views), 0)::bigint AS views_total,
                    COALESCE(SUM(likes), 0)::bigint AS likes_total,
                    COALESCE(SUM(cost_attributed), 0)::double precision AS cost_total
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                """,
                user["id"], since
            )
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS uploads_total,
                    SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS uploads_completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS uploads_failed,
                    0::double precision AS avg_processing_seconds,
                    0::bigint AS views_total,
                    0::bigint AS likes_total,
                    0::double precision AS cost_total
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                """,
                user["id"], since
            )

        # Revenue (optional)
        revenue_total = 0.0
        try:
            rev = await conn.fetchval(
                "SELECT COALESCE(SUM(amount), 0)::decimal FROM revenue_tracking WHERE user_id = $1 AND created_at >= $2",
                user["id"], since
            )
            revenue_total = float(rev or 0)
        except Exception as e:
            if e.__class__.__name__ != "UndefinedTableError":
                raise

    return {
        "range_days": days,
        "since": since.isoformat(),
        "uploads": {
            "total": int(row["uploads_total"] or 0),
            "completed": int(row["uploads_completed"] or 0),
            "failed": int(row["uploads_failed"] or 0),
            "avg_processing_seconds": float(row["avg_processing_seconds"] or 0),
        },
        "engagement": {
            "views": int(row["views_total"] or 0),
            "likes": int(row["likes_total"] or 0),
        },
        "costs": {
            "cost_total": float(row["cost_total"] or 0),
        },
        "revenue": {
            "revenue_total": revenue_total,
        },
    }

@app.get("/api/analytics/my-avg-processing")
async def my_avg_processing(user: dict = Depends(get_current_user)):
    """Return this user's personal average processing time in seconds.
    Used by upload.html to calibrate the progress estimate instead of
    using a hardcoded 7-minute fallback.
    Falls back to 420s (7 min) when fewer than 3 completed uploads exist.
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS sample_count,
                    COALESCE(
                        AVG(EXTRACT(EPOCH FROM (processing_finished_at - processing_started_at))),
                        420
                    )::double precision AS avg_seconds
                FROM uploads
                WHERE user_id      = $1
                  AND status       IN ('succeeded', 'completed', 'partial')
                  AND processing_started_at  IS NOT NULL
                  AND processing_finished_at IS NOT NULL
                  AND processing_finished_at > processing_started_at
                """,
                user["id"],
            )
        sample_count = int(row["sample_count"] or 0)
        avg_seconds  = float(row["avg_seconds"] or 420)
        # Only trust personal average if we have at least 3 data points
        if sample_count < 3:
            avg_seconds = 420.0
        return {
            "avg_processing_seconds": round(avg_seconds, 1),
            "sample_count": sample_count,
            "reliable": sample_count >= 3,
        }
    except Exception as e:
        logger.warning(f"my_avg_processing failed: {e}")
        return {"avg_processing_seconds": 420.0, "sample_count": 0, "reliable": False}




@app.get("/api/analytics/export")
async def analytics_export(days: int = Query(30, ge=1, le=3650), format: str = Query("csv"), user: dict = Depends(get_current_user)):
    """Export analytics for the last N days as CSV (default) or JSON."""
    since = _now_utc() - timedelta(days=days)

    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT
                    id, filename, title, caption, platforms, privacy, status,
                    created_at, completed_at,
                    COALESCE(views, 0)::bigint AS views,
                    COALESCE(likes, 0)::bigint AS likes,
                    COALESCE(comments, 0)::bigint AS comments,
                    COALESCE(shares, 0)::bigint AS shares,
                    COALESCE(cost_attributed, 0)::double precision AS cost_attributed,
                    video_url
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                ORDER BY created_at DESC
                """,
                user["id"], since
            )
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise
            # Older schema fallback
            rows = await conn.fetch(
                """
                SELECT
                    id, filename, title, caption, platforms, privacy, status,
                    created_at, completed_at,
                    0::bigint AS views,
                    0::bigint AS likes,
                    0::bigint AS comments,
                    0::bigint AS shares,
                    0::double precision AS cost_attributed,
                    video_url
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2
                ORDER BY created_at DESC
                """,
                user["id"], since
            )

    data = [
        {
            "id": str(r["id"]),
            "filename": r["filename"],
            "title": r["title"],
            "caption": r["caption"],
            "platforms": list(r["platforms"]) if r["platforms"] else [],
            "privacy": r["privacy"],
            "status": r["status"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            "views": int(r["views"] or 0),
            "likes": int(r["likes"] or 0),
            "comments": int(r["comments"] or 0),
            "shares": int(r["shares"] or 0),
            "cost_attributed": float(r["cost_attributed"] or 0),
            "video_url": r.get("video_url"),
        }
        for r in rows
    ]

    if format.lower() == "json":
        return {"range_days": days, "since": since.isoformat(), "rows": data}

    # CSV default
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id","filename","title","caption","platforms","privacy","status",
            "created_at","completed_at",
            "views","likes","comments","shares",
            "cost_attributed","video_url",
        ],
    )
    writer.writeheader()
    for item in data:
        item = dict(item)
        item["platforms"] = ",".join(item.get("platforms") or [])
        writer.writerow(item)

    csv_bytes = output.getvalue().encode("utf-8")
    headers = {"Content-Disposition": f'attachment; filename="uploadm8-analytics-{days}d.csv"'}
    return Response(content=csv_bytes, media_type="text/csv", headers=headers)
# ====================
class SupportContactRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    subject: str
    message: str


@app.post("/api/support/contact")
async def support_contact(payload: SupportContactRequest, user: dict = Depends(get_current_user)):
    """Create a support ticket/message from the app."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO support_messages (user_id, name, email, subject, message)
            VALUES ($1, $2, $3, $4, $5)
            """,
            user["id"],
            (payload.name or user.get("name") or "").strip() or None,
            (payload.email or user.get("email") or "").strip() or None,
            payload.subject.strip(),
            payload.message.strip(),
        )

    # Optional admin notification
    if ADMIN_DISCORD_WEBHOOK_URL:
        await discord_notify(
            ADMIN_DISCORD_WEBHOOK_URL,
            embeds=[{
                "title": "🆘 Support Message",
                "color": 0xf97316,
                "fields": [
                    {"name": "User", "value": f"{user.get('email','')} ({user.get('id','')})"},
                    {"name": "Subject", "value": payload.subject[:256]},
                    {"name": "Message", "value": (payload.message[:900] + "…") if len(payload.message) > 900 else payload.message},
                ],
            }],
        )

    return {"status": "received"}


# ============================================================
# FRONTEND BUTTON-CLICK / UI ACTION AUDIT ENDPOINT
# ============================================================
class ActivityLogIn(BaseModel):
    action: str
    event_category: str = "UI_ACTION"
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict] = None
    session_id: Optional[str] = None

@app.post("/api/activity/log")
async def log_activity(data: ActivityLogIn, request: Request, user: dict = Depends(get_current_user)):
    """
    Frontend button-click and UI action audit trail.
    Called by JavaScript whenever a significant user action occurs.
    Stored in system_event_log with full context.
    """
    allowed_categories = {"UI_ACTION", "UPLOAD", "PLATFORM", "AUTH", "NAVIGATION"}
    category = data.event_category if data.event_category in allowed_categories else "UI_ACTION"

    # Sanitize — prevent log injection
    action_safe = str(data.action or "")[:100].strip()
    if not action_safe:
        return {"ok": False, "error": "action required"}

    await log_system_event(
        user_id=str(user["id"]),
        action=action_safe,
        event_category=category,
        resource_type=data.resource_type,
        resource_id=data.resource_id,
        details={**(data.details or {}), "session_id": data.session_id},
        request=request,
    )
    return {"ok": True}


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
async def admin_update_user(user_id: str, data: AdminUserUpdate, request: Request, user: dict = Depends(require_admin)):
    updates, params = [], [user_id]
    changes = {}
    if data.subscription_tier:
        updates.append(f"subscription_tier = ${len(params)+1}")
        params.append(data.subscription_tier)
        changes["subscription_tier"] = data.subscription_tier
    if data.role and user.get("role") == "master_admin":
        updates.append(f"role = ${len(params)+1}")
        params.append(data.role)
        changes["role"] = data.role
    if data.status:
        updates.append(f"status = ${len(params)+1}")
        params.append(data.status)
        changes["status"] = data.status
    if data.flex_enabled is not None:
        updates.append(f"flex_enabled = ${len(params)+1}")
        params.append(data.flex_enabled)
        changes["flex_enabled"] = data.flex_enabled
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
            await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_UPDATE_USER",
                                  details={"changes": changes}, request=request,
                                  resource_type="user", resource_id=user_id)
    return {"status": "updated"}

@app.post("/api/admin/users/{user_id}/ban")
async def admin_ban_user(user_id: str, request: Request, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET status = 'banned' WHERE id = $1", user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_BAN_USER",
                              details={"target_email": target["email"] if target else None},
                              request=request, resource_type="user", resource_id=user_id,
                              severity="WARNING")
    return {"status": "banned"}

@app.post("/api/admin/users/{user_id}/unban")
async def admin_unban_user(user_id: str, request: Request, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET status = 'active' WHERE id = $1", user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_UNBAN_USER",
                              details={"target_email": target["email"] if target else None},
                              request=request, resource_type="user", resource_id=user_id)
    return {"status": "unbanned"}


@app.put("/api/admin/users/{user_id}/email")
async def admin_change_email(user_id: str, payload: AdminUpdateEmailIn, request: Request, user: dict = Depends(get_current_user)):
    require_admin(user)
    new_email = payload.email.lower().strip()

    async with db_pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email)=LOWER($1) AND id <> $2",
            new_email,
            user_id,
        )
        if exists:
            raise HTTPException(status_code=409, detail="Email already in use")

        old = await conn.fetchrow("SELECT email FROM users WHERE id=$1", user_id)
        if not old:
            raise HTTPException(status_code=404, detail="User not found")

        verification_token = secrets.token_urlsafe(32)

        await conn.execute(
            """
            INSERT INTO email_changes (user_id, old_email, new_email, changed_by_admin_id, verification_token)
            VALUES ($1::uuid, $2, $3, $4::uuid, $5)
            """,
            user_id,
            old["email"],
            new_email,
            user["id"],
            verification_token,
        )

        await conn.execute(
            "UPDATE users SET email=$1, email_verified=false, updated_at=NOW() WHERE id=$2",
            new_email,
            user_id,
        )

        await log_admin_audit(
            conn,
            user_id=user_id,
            admin=user,
            action="ADMIN_CHANGE_EMAIL",
            details={"old_email": old["email"], "new_email": new_email},
            request=request,
        )

    return {"ok": True, "email": new_email}


class AdminResetPasswordIn(BaseModel):
    temp_password: str = Field(min_length=8, max_length=128)


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, payload: AdminResetPasswordIn, request: Request, user: dict = Depends(get_current_user)):
    require_admin(user)
    temp = payload.temp_password
    pw_hash = bcrypt.hashpw(temp.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT id, role FROM users WHERE id=$1", user_id)
        if not target:
            raise HTTPException(status_code=404, detail="User not found")
        if target["role"] == "master_admin" and user.get("role") != "master_admin":
            raise HTTPException(status_code=403, detail="Cannot reset master_admin password")

        await conn.execute(
            """
            UPDATE users
            SET password_hash=$1,
                must_reset_password=true,
                updated_at=NOW()
            WHERE id=$2
            """,
            pw_hash,
            user_id,
        )

        await conn.execute(
            """
            INSERT INTO password_resets (user_id, reset_by_admin_id, temp_password_hash, force_change, expires_at)
            VALUES ($1::uuid, $2::uuid, $3, TRUE, NOW() + INTERVAL '7 days')
            """,
            user_id,
            user["id"],
            pw_hash,
        )

        await log_admin_audit(
            conn,
            user_id=user_id,
            admin=user,
            action="ADMIN_RESET_PASSWORD",
            details={"must_reset_password": True},
            request=request,
        )

    return {"ok": True}


@app.get("/api/admin/audit")
async def admin_audit(
    user_id: Optional[str] = None,
    event_category: Optional[str] = None,
    action: Optional[str] = None,
    severity: Optional[str] = None,
    source: str = "all",           # "all" | "admin" | "system"
    limit: int = 100,
    offset: int = 0,
    user: dict = Depends(get_current_user)
):
    """
    Corporate-grade audit log endpoint.
    - Returns rolling 6-month window across both admin_audit_log and system_event_log
    - Supports filtering by category, action, severity, user, source table
    - Auto-purges records older than 6 months on each call (once per hour max via in-memory flag)
    - Pagination via limit/offset
    """
    require_admin(user)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    def _ser(v):
        if v is None: return None
        if hasattr(v, "isoformat"): return v.isoformat()
        if isinstance(v, uuid.UUID): return str(v)
        if isinstance(v, (dict, list)): return v
        return v

    def _ser_row(r: dict) -> dict:
        return {k: _ser(v) for k, v in r.items()}

    try:
        async with db_pool.acquire() as conn:
            # ── Background purge (rolling 6-month window) ────────────────────
            try:
                await conn.execute(
                    "DELETE FROM admin_audit_log WHERE created_at < NOW() - INTERVAL '6 months'"
                )
                await conn.execute(
                    "DELETE FROM system_event_log WHERE created_at < NOW() - INTERVAL '6 months'"
                )
            except Exception as purge_err:
                logger.warning(f"[audit] Purge skipped: {purge_err}")

            items = []

            # ── Build admin_audit_log query ───────────────────────────────────
            if source in ("all", "admin"):
                try:
                    aq = """
                        SELECT
                            id, 'admin_audit' AS log_source,
                            COALESCE(event_category, 'ADMIN') AS event_category,
                            action,
                            COALESCE(actor_user_id, admin_id) AS actor_id,
                            admin_email AS actor_email,
                            user_id AS target_user_id,
                            resource_type, resource_id,
                            details, ip_address, user_agent,
                            COALESCE(severity, 'INFO') AS severity,
                            COALESCE(outcome, 'SUCCESS') AS outcome,
                            created_at
                        FROM admin_audit_log
                        WHERE created_at >= NOW() - INTERVAL '6 months'
                    """
                    aargs: List[Any] = []
                    if user_id:
                        aargs.append(user_id)
                        aq += f" AND (user_id = ${len(aargs)}::uuid OR admin_id = ${len(aargs)}::uuid)"
                    if event_category:
                        aargs.append(event_category.upper())
                        aq += f" AND UPPER(COALESCE(event_category,'ADMIN')) = ${len(aargs)}"
                    if action:
                        aargs.append(f"%{action.upper()}%")
                        aq += f" AND UPPER(action) LIKE ${len(aargs)}"
                    if severity:
                        aargs.append(severity.upper())
                        aq += f" AND UPPER(COALESCE(severity,'INFO')) = ${len(aargs)}"

                    admin_rows = await conn.fetch(aq, *aargs)
                    items.extend([_ser_row(dict(r)) for r in admin_rows])
                except Exception as ae:
                    logger.warning(f"[audit] admin_audit_log query failed: {ae}")

            # ── Build system_event_log query ──────────────────────────────────
            if source in ("all", "system"):
                try:
                    sq = """
                        SELECT
                            id, 'system_event' AS log_source,
                            event_category,
                            action,
                            user_id AS actor_id,
                            NULL AS actor_email,
                            user_id AS target_user_id,
                            resource_type, resource_id,
                            details, ip_address, user_agent,
                            COALESCE(severity, 'INFO') AS severity,
                            COALESCE(outcome, 'SUCCESS') AS outcome,
                            created_at
                        FROM system_event_log
                        WHERE created_at >= NOW() - INTERVAL '6 months'
                    """
                    sargs: List[Any] = []
                    if user_id:
                        sargs.append(user_id)
                        sq += f" AND user_id = ${len(sargs)}::uuid"
                    if event_category:
                        sargs.append(event_category.upper())
                        sq += f" AND UPPER(event_category) = ${len(sargs)}"
                    if action:
                        sargs.append(f"%{action.upper()}%")
                        sq += f" AND UPPER(action) LIKE ${len(sargs)}"
                    if severity:
                        sargs.append(severity.upper())
                        sq += f" AND UPPER(COALESCE(severity,'INFO')) = ${len(sargs)}"

                    sys_rows = await conn.fetch(sq, *sargs)
                    items.extend([_ser_row(dict(r)) for r in sys_rows])
                except Exception as se:
                    logger.warning(f"[audit] system_event_log query failed: {se}")

            # Sort combined by created_at desc, paginate
            items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            total = len(items)
            page = items[offset: offset + limit]

            # Enrich with actor name from users table where possible
            actor_ids = list({x["actor_id"] for x in page if x.get("actor_id")})
            actor_map = {}
            if actor_ids:
                try:
                    urows = await conn.fetch(
                        "SELECT id, email, name FROM users WHERE id = ANY($1::uuid[])",
                        [str(a) for a in actor_ids]
                    )
                    for ur in urows:
                        actor_map[str(ur["id"])] = {"email": ur["email"], "name": ur["name"]}
                except Exception:
                    pass

            for item in page:
                aid = str(item.get("actor_id") or "")
                if aid in actor_map:
                    item["actor_email"] = item.get("actor_email") or actor_map[aid]["email"]
                    item["actor_name"] = actor_map[aid]["name"]

        return {
            "items": page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
        }

    except Exception as e:
        logger.error(f"[admin_audit] Error fetching audit log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit log error: {str(e)}")


@app.get("/api/admin/analytics/users")
async def admin_analytics_users(user: dict = Depends(get_current_user)):
    require_admin(user)

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        active_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status='active'")
        banned_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status='banned'")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier <> 'free'")
        new_users_30d = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '30 days'")
        new_users_7d = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '7 days'")
        new_users_24h = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= NOW() - INTERVAL '24 hours'")

    return {
        "total_users": int(total_users or 0),
        "active_users": int(active_users or 0),
        "banned_users": int(banned_users or 0),
        "paid_users": int(paid_users or 0),
        "new_users_30d": int(new_users_30d or 0),
        "new_users_7d": int(new_users_7d or 0),
        "new_users_24h": int(new_users_24h or 0),
    }


@app.get("/api/admin/analytics/revenue")
async def admin_analytics_revenue(user: dict = Depends(get_current_user)):
    require_admin(user)

    async with db_pool.acquire() as conn:
        launch_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='launch'")
        creator_pro_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='creator_pro'")
        studio_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='studio'")
        agency_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='agency'")

    return {
        "mrr_estimate": 0.0,
        "launch_count": int(launch_count or 0),
        "creator_pro_count": int(creator_pro_count or 0),
        "studio_count": int(studio_count or 0),
        "agency_count": int(agency_count or 0),
    }


@app.post("/api/admin/users/assign-tier")
async def admin_assign_tier(user_id: str = Query(...), tier: str = Query(...), request: Request = None, user: dict = Depends(require_master_admin)):
    if tier not in PLAN_CONFIG:
        raise HTTPException(400, "Invalid tier")
    async with db_pool.acquire() as conn:
        old = await conn.fetchrow("SELECT subscription_tier, email FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET subscription_tier = $1 WHERE id = $2", tier, user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_ASSIGN_TIER",
                              details={"old_tier": old["subscription_tier"] if old else None, "new_tier": tier,
                                       "target_email": old["email"] if old else None},
                              request=request, resource_type="user", resource_id=user_id,
                              severity="WARNING")
    return {"status": "assigned", "tier": tier}

# ============================================================
# Announcements
# ============================================================
@app.post("/api/admin/announcements/send")

# ---------------------------------------------------------------------------
# ANNOUNCEMENTS: idempotent delivery intents + multi-channel fanout
# - Keeps backwards compatibility with legacy AnnouncementRequest booleans.
# - Uses announcement_deliveries as the source of truth for what was attempted/sent.
# ---------------------------------------------------------------------------

def _normalize_announcement_channels(channels_in, send_email: bool, send_discord_community: bool, send_user_webhooks: bool):
    """Return normalized channel list: ['email','discord_community','user_webhook']"""
    out = []
    # New-style list/dict support (if frontend starts sending channels explicitly)
    if isinstance(channels_in, dict):
        for k, v in channels_in.items():
            if v is True:
                out.append(str(k))
    elif isinstance(channels_in, list):
        out.extend([str(c) for c in channels_in])

    # Legacy booleans always win (existing UI contract)
    if send_email and "email" not in out:
        out.append("email")
    if send_discord_community and "discord_community" not in out:
        out.append("discord_community")
    if send_user_webhooks and "user_webhook" not in out and "user_webhooks" not in out:
        out.append("user_webhook")

    # Normalize key spelling
    out = ["user_webhook" if c == "user_webhooks" else c for c in out]
    # De-dupe, preserve order
    seen = set()
    norm = []
    for c in out:
        c = (c or "").strip()
        if not c:
            continue
        if c in seen:
            continue
        if c not in ("email", "discord_community", "user_webhook"):
            continue
        seen.add(c)
        norm.append(c)
    return norm

def _channels_to_store_map(channels_list):
    """Persist in the same shape your DB currently uses (json text map)."""
    store = {"email": False, "discord_community": False, "user_webhooks": False}
    for c in channels_list:
        if c == "email":
            store["email"] = True
        elif c == "discord_community":
            store["discord_community"] = True
        elif c == "user_webhook":
            store["user_webhooks"] = True
    return json.dumps(store)

async def _discord_post_raw(webhook_url: str, *, content: str = None, embeds: list = None):
    """Return (ok, err). Unlike discord_notify(), this does not swallow failures."""
    if not webhook_url:
        return False, "missing webhook url"
    payload = {}
    if content:
        payload["content"] = content[:1999]
    if embeds:
        payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(webhook_url, json=payload)
        if r.status_code in (200, 204):
            return True, ""
        return False, f"{r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, str(e)

async def _insert_delivery_intents(conn, announcement_id: str, *, sender_user_id: str, recipients: list, channels_list: list, title: str, body: str):
    """
    Insert queued intents into announcement_deliveries, idempotent by (announcement_id,user_id,channel).
    - email: one row per recipient (destination=email)
    - user_webhook: one row per recipient with webhook configured (destination=webhook)
    - discord_community: single row owned by sender (destination=global webhook)
    """
    now = _now_utc() if "_now_utc" in globals() else datetime.now(timezone.utc)

    inserts = []

    # Community webhook is a single delivery owned by sender
    if "discord_community" in channels_list and COMMUNITY_DISCORD_WEBHOOK_URL:
        inserts.append((announcement_id, sender_user_id, "discord_community", COMMUNITY_DISCORD_WEBHOOK_URL, "queued", None, now))

    # Per-user deliveries
    for r in recipients:
        uid = str(r.get("id") or r.get("user_id") or "")
        email = (r.get("email") or "").strip()
        webhook = (r.get("discord_webhook") or "").strip()

        if not uid:
            continue

        if "email" in channels_list and email:
            inserts.append((announcement_id, uid, "email", email, "queued", None, now))

        if "user_webhook" in channels_list and webhook:
            inserts.append((announcement_id, uid, "user_webhook", webhook, "queued", None, now))

    if not inserts:
        return 0

    await conn.executemany(
        """
        INSERT INTO announcement_deliveries
          (announcement_id, user_id, channel, destination, status, error, created_at)
        SELECT $1::uuid, $2::uuid, $3::text, $4::text, $5::text, $6::text, $7::timestamptz
        WHERE NOT EXISTS (
          SELECT 1 FROM announcement_deliveries
          WHERE announcement_id = $1::uuid AND user_id = $2::uuid AND channel = $3::text
        )
        """,
        inserts,
    )
    return len(inserts)

async def _execute_announcement_deliveries(conn, announcement_id: str, title: str, body: str):
    """Execute queued deliveries for an announcement and update rollups."""
    rows = await conn.fetch(
        """
        SELECT id, channel, destination
        FROM announcement_deliveries
        WHERE announcement_id = $1::uuid AND status = 'queued'
        ORDER BY id ASC
        """,
        announcement_id,
    )

    email_sent = 0
    discord_sent = 0
    webhook_sent = 0

    for r in rows:
        delivery_id = r["id"]
        ch = r["channel"]
        dest = (r["destination"] or "").strip()

        ok = False
        err = ""

        if ch == "email":
            try:
                # send_email(to, subject, html_body) exists in your codebase
                await send_email(dest, title, f"<h1>{title}</h1><p>{body}</p>")
                ok = True
                email_sent += 1
            except Exception as e:
                ok = False
                err = str(e)

        elif ch == "discord_community":
            ok, err = await _discord_post_raw(dest, embeds=[{"title": f"📢 {title}", "description": body, "color": 0xf97316}])
            if ok:
                discord_sent += 1

        elif ch == "user_webhook":
            ok, err = await _discord_post_raw(dest, embeds=[{"title": f"📢 {title}", "description": body, "color": 0xf97316}])
            if ok:
                webhook_sent += 1

        status = "sent" if ok else "failed"
        await conn.execute(
            """
            UPDATE announcement_deliveries
            SET status = $2,
                error = $3,
                sent_at = CASE WHEN $2='sent' THEN NOW() ELSE sent_at END
            WHERE id = $1
            """,
            delivery_id,
            status,
            (err or None),
        )

    await conn.execute(
        """
        UPDATE announcements
        SET email_sent = COALESCE(email_sent,0) + $2,
            discord_sent = COALESCE(discord_sent,0) + $3,
            webhook_sent = COALESCE(webhook_sent,0) + $4
        WHERE id = $1::uuid
        """,
        announcement_id,
        int(email_sent),
        int(discord_sent),
        int(webhook_sent),
    )

    return {"email": email_sent, "discord_community": discord_sent, "user_webhook": webhook_sent}



async def send_announcement(data: AnnouncementRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    """Creates announcement + idempotent delivery intents, then executes queued deliveries."""
    title = (data.title or "").strip()
    body = (data.body or "").strip()
    if not title or not body:
        raise HTTPException(status_code=400, detail="title and body are required")

    async with db_pool.acquire() as conn:
        # ----------------------------
        # Resolve recipients (banned excluded)
        # ----------------------------
        query = """
            SELECT
              u.id::text AS id,
              u.email,
              u.subscription_tier,
              COALESCE(NULLIF(us.discord_webhook,''), '') AS discord_webhook
            FROM users u
            LEFT JOIN user_settings us ON us.user_id = u.id
            WHERE COALESCE(u.status,'active') <> 'banned'
        """
        params = []

        if getattr(data, "target", None) == "paid":
            query += " AND u.subscription_tier NOT IN ('free')"
        elif getattr(data, "target", None) == "free":
            query += " AND u.subscription_tier = 'free'"
        elif getattr(data, "target", None) in ("specific_tiers", "tiers") and getattr(data, "target_tiers", None):
            params.append(list(data.target_tiers))
            query += f" AND u.subscription_tier = ANY(${len(params)}::text[]) "

        recipients = await conn.fetch(query, *params) if params else await conn.fetch(query)
        recipients_list = [dict(r) for r in recipients]

        # ----------------------------
        # Normalize channels
        # ----------------------------
        channels_list = _normalize_announcement_channels(
            getattr(data, "channels", None) if hasattr(data, "channels") else None,
            bool(getattr(data, "send_email", False)),
            bool(getattr(data, "send_discord_community", False)),
            bool(getattr(data, "send_user_webhooks", False)),
        )
        if not channels_list:
            raise HTTPException(status_code=400, detail="No channels selected")

        # Persist in the DB using your current storage shape (json text map)
        channels_store = _channels_to_store_map(channels_list)

        # ----------------------------
        # Insert announcement row
        # ----------------------------
        ann_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO announcements (id, title, body, channels, target, target_tiers, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            ann_id, title, body, channels_store, getattr(data, "target", "all"), getattr(data, "target_tiers", None), user["id"]
        )

        # ----------------------------
        # Insert idempotent delivery intents
        # ----------------------------
        await _insert_delivery_intents(
            conn,
            ann_id,
            sender_user_id=str(user["id"]),
            recipients=recipients_list,
            channels_list=channels_list,
            title=title,
            body=body,
        )

        # ----------------------------
        # Execute fanout (background). For debugging, you can run inline.
        # ----------------------------
        async def _run():
            async with db_pool.acquire() as c2:
                return await _execute_announcement_deliveries(c2, ann_id, title, body)

        background_tasks.add_task(_run)

    return {"status": "queued", "announcement_id": ann_id, "recipients": len(recipients_list), "channels": channels_list}


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
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family')")
        
        upload_stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed,
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
@app.get("/api/admin/analytics/overview")
async def admin_analytics_overview(
    range: Optional[str] = Query(None, description="Time window: 7d|30d|90d|6m|1y|Nd (custom)"),
    days: Optional[int] = Query(None, ge=1, le=3650, description="Compatibility: days=N"),
    user: dict = Depends(require_admin),
):
    """
    Admin analytics overview for the selected time window.

    Contract (frontend expects):
      - total_users
      - new_users
      - paid_users
      - mrr_estimate
      - range

    Supports:
      - ?range=7d|30d|90d|6m|1y
      - ?range=45d (custom, guarded 1–3650)
      - ?days=45 (compat)
    """
    # -------- range parsing (authoritative) --------
    preset_map = {"7d": 7, "30d": 30, "90d": 90, "6m": 180, "1y": 365}
    window_days = None

    if days is not None:
        window_days = int(days)

    if range:
        r = str(range).strip().lower()
        if r in preset_map:
            window_days = preset_map[r]
        else:
            m = re.match(r"^(\d{1,4})d$", r)
            if m:
                window_days = int(m.group(1))
            else:
                raise HTTPException(status_code=400, detail="Invalid range. Use 7d|30d|90d|6m|1y or Nd (e.g. 45d).")

    if window_days is None:
        window_days = 30

    if window_days < 1 or window_days > 3650:
        raise HTTPException(status_code=400, detail="Range out of bounds. Use 1–3650 days.")

    since = _now_utc() - timedelta(days=window_days)

    # -------- KPI aggregation --------
    # We intentionally avoid a dependency on a billing_events table (not guaranteed to exist).
    # Instead, derive paid_users + mrr_estimate from users.subscription_tier + users.subscription_status.
    paid_tiers = ["launch", "creator_pro", "studio", "agency"]

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")

        new_users = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE created_at >= $1",
            since,
        )

        paid_rows = await conn.fetch(
            """
            SELECT subscription_tier, COUNT(*)::bigint AS c
            FROM users
            WHERE subscription_status = 'active'
              AND subscription_tier = ANY($1::text[])
            GROUP BY subscription_tier
            """,
            paid_tiers,
        )

    counts = {row["subscription_tier"]: int(row["c"]) for row in paid_rows}
    paid_users = sum(counts.values())

    # MRR estimate based on PLAN_CONFIG prices
    mrr_estimate = 0.0
    for tier, c in counts.items():
        price = float(get_plan(tier).get("price", 0) or 0)
        mrr_estimate += price * c

    return {
        "total_users": int(total_users or 0),
        "new_users": int(new_users or 0),
        "paid_users": int(paid_users or 0),
        "mrr_estimate": float(mrr_estimate),
        "range": f"{window_days}d",
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
            SELECT u.subscription_tier, COUNT(up.id)::int AS uploads, 0::decimal AS cost
            FROM users u LEFT JOIN uploads up ON up.user_id = u.id AND up.created_at >= $1
            GROUP BY u.subscription_tier
        """, since)
        
        platform_data = await conn.fetch("""
            SELECT unnest(platforms) AS platform, COUNT(*)::int AS uploads, 0::decimal AS cost
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
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed,
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


# ============================================================
# ADDITIONAL KPI ENDPOINTS (Added for frontend compatibility)
# ============================================================

# ------------------------------------------------------------
# Time range parsing (supports presets + custom 'Nd')
# ------------------------------------------------------------
_RANGE_PRESETS_MINUTES = {
    "24h": 24 * 60,
    "7d": 7 * 24 * 60,
    "30d": 30 * 24 * 60,
    "90d": 90 * 24 * 60,
    "6m": 180 * 24 * 60,
    "1y": 365 * 24 * 60,
}

def _range_to_minutes(range_str: str | None, default_minutes: int) -> int:
    r = (range_str or "").strip()
    if not r:
        return default_minutes
    if r in _RANGE_PRESETS_MINUTES:
        return _RANGE_PRESETS_MINUTES[r]
    m = re.fullmatch(r"(\d{1,4})d", r)
    if m:
        days = int(m.group(1))
        # Guardrails: 1 day .. 10 years
        days = max(1, min(days, 3650))
        return days * 24 * 60
    return default_minutes

def _range_label(range_str: str | None, fallback: str = "30d") -> str:
    r = (range_str or "").strip()
    return r if r else fallback

@app.get("/api/admin/kpis")
async def get_admin_kpis(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Combined KPI endpoint that returns all metrics in one call"""
    minutes = {"7d": 10080, "30d": 43200, "90d": 129600, "365d": 525600, "1y": 525600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    prev_since = since - timedelta(minutes=minutes)
    
    async with db_pool.acquire() as conn:
        # Users
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        prev_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1 AND created_at < $2", prev_since, since)
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active'")
        active_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)
        
        new_users_change = ((new_users - prev_users) / max(prev_users, 1)) * 100 if prev_users > 0 else 0
        
        # MRR
        mrr_data = await conn.fetch("""
            SELECT subscription_tier, COUNT(*) AS count FROM users 
            WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') 
            AND subscription_status = 'active' GROUP BY subscription_tier
        """)
        total_mrr = sum(get_plan(r["subscription_tier"]).get("price", 0) * r["count"] for r in mrr_data)
        mrr_by_tier = {r["subscription_tier"]: get_plan(r["subscription_tier"]).get("price", 0) * r["count"] for r in mrr_data}
        
        # Tier breakdown
        tier_data = await conn.fetch("SELECT COALESCE(subscription_tier, 'free') as tier, COUNT(*)::int AS count FROM users GROUP BY subscription_tier")
        tier_breakdown = {t["tier"] or "free": t["count"] for t in tier_data}
        
        # Revenue
        revenue = await conn.fetchrow("""
            SELECT COALESCE(SUM(amount), 0)::decimal AS total,
            COALESCE(SUM(CASE WHEN source = 'topup' THEN amount ELSE 0 END), 0)::decimal AS topups
            FROM revenue_tracking WHERE created_at >= $1
        """, since)
        
        # Costs
        costs = await conn.fetchrow("""
            SELECT COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
            COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
            COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        openai_cost = float(costs["openai"] or 0) if costs else 0
        storage_cost = float(costs["storage"] or 0) if costs else 0
        compute_cost = float(costs["compute"] or 0) if costs else 0
        total_costs = openai_cost + storage_cost + compute_cost
        
        gross_margin = ((total_mrr - total_costs) / max(total_mrr, 1)) * 100 if total_mrr > 0 else 0
        
        # Uploads
        upload_stats = await conn.fetchrow("""
            SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE created_at >= $1
        """, since)
        total_uploads = upload_stats["total"] if upload_stats else 0
        successful_uploads = upload_stats["completed"] if upload_stats else 0
        success_rate = (successful_uploads / max(total_uploads, 1)) * 100
        
        prev_uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1 AND created_at < $2", prev_since, since)
        uploads_change = ((total_uploads - prev_uploads) / max(prev_uploads, 1)) * 100 if prev_uploads > 0 else 0
        cost_per_upload = total_costs / max(successful_uploads, 1)
        
        # Platform distribution
        platform_data = await conn.fetch("SELECT unnest(platforms) AS platform, COUNT(*)::int AS uploads FROM uploads WHERE created_at >= $1 GROUP BY platform", since)
        platform_distribution = {p["platform"]: p["uploads"] for p in platform_data}
        
        queue_depth = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status IN ('pending', 'queued', 'processing')")
        
        # Funnels
        funnel_connected = await conn.fetchval("SELECT COUNT(DISTINCT u.id) FROM users u JOIN platform_tokens pt ON u.id = pt.user_id WHERE u.created_at >= $1", since)
        funnel_uploaded = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)
        funnel_signup_connect = (funnel_connected / max(new_users, 1)) * 100
        funnel_connect_upload = (funnel_uploaded / max(funnel_connected, 1)) * 100
        
        cancellations = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_status = 'cancelled' AND updated_at >= $1", since)
    
    return {
        "total_mrr": total_mrr, "mrr_change": 0, "mrr_by_tier": mrr_by_tier,
        "mrr_launch": mrr_by_tier.get("launch", 0), "mrr_creator_pro": mrr_by_tier.get("creator_pro", 0),
        "mrr_studio": mrr_by_tier.get("studio", 0), "mrr_agency": mrr_by_tier.get("agency", 0),
        "launch_users": tier_breakdown.get("launch", 0), "creator_pro_users": tier_breakdown.get("creator_pro", 0),
        "studio_users": tier_breakdown.get("studio", 0), "agency_users": tier_breakdown.get("agency", 0),
        "topup_revenue": float(revenue["topups"]) if revenue else 0, "topup_count": 0,
        "arpu": round(total_mrr / max(total_users, 1), 2), "arpa": round(total_mrr / max(paid_users, 1), 2),
        "refunds": 0, "refund_count": 0,
        "openai_cost": openai_cost, "storage_cost": storage_cost, "compute_cost": compute_cost,
        "total_costs": total_costs, "cost_per_upload": round(cost_per_upload, 4),
        "gross_margin": round(gross_margin, 1), "gross_margin_change": 0,
        "funnel_signups": new_users, "funnel_connected": funnel_connected, "funnel_uploaded": funnel_uploaded,
        "funnel_signup_connect": round(funnel_signup_connect, 1), "funnel_connect_upload": round(funnel_connect_upload, 1),
        "free_to_paid_rate": round((paid_users / max(total_users, 1)) * 100, 1), "free_to_paid_change": 0,
        "cancellations": cancellations, "cancellation_rate": round((cancellations / max(paid_users, 1)) * 100, 1),
        "failed_payments": 0, "payment_failure_rate": 0,
        "total_uploads": total_uploads, "successful_uploads": successful_uploads, "success_rate": round(success_rate, 1),
        "transcode_fail_rate": 0, "platform_fail_rate": 0, "retry_rate": 0,
        "avg_process_time": 0, "avg_transcode_time": 0, "cancel_rate": 0, "queue_depth": queue_depth or 0,
        "new_users": new_users, "new_users_change": round(new_users_change, 1), "uploads_change": round(uploads_change, 1),
        "active_users": active_users or 0, "total_views": upload_stats["views"] if upload_stats else 0,
        "total_likes": upload_stats["likes"] if upload_stats else 0,
        "avg_uploads_per_user": round(total_uploads / max(active_users or 1, 1), 1),
        "tier_breakdown": tier_breakdown, "platform_distribution": platform_distribution,
    }


@app.get("/api/admin/kpi/revenue")
async def get_kpi_revenue(user: dict = Depends(require_admin)):
    since = _now_utc() - timedelta(days=30)
    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        paid_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active'") or 1
        mrr_data = await conn.fetch("SELECT subscription_tier, COUNT(*) AS count FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active' GROUP BY subscription_tier")
        total_mrr = sum(get_plan(r["subscription_tier"]).get("price", 0) * r["count"] for r in mrr_data)
        topup = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM revenue_tracking WHERE source = 'topup' AND created_at >= $1", since)
    return {"total_mrr": total_mrr, "mrr_change": 0, "mrr_by_tier": {}, "topup_total": float(topup or 0),
            "arpu": round(total_mrr / max(total_users, 1), 2), "arpa": round(total_mrr / max(paid_users, 1), 2),
            "ltv": round((total_mrr / max(paid_users, 1)) * 12, 2), "refunds_total": 0, "refunds_count": 0, "refunds_change": 0}


@app.get("/api/admin/kpi/costs")
async def get_kpi_costs(user: dict = Depends(require_admin)):
    since = _now_utc() - timedelta(days=30)
    async with db_pool.acquire() as conn:
        costs = await conn.fetchrow("SELECT COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai, COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage, COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute FROM cost_tracking WHERE created_at >= $1", since)
        uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status IN ('completed','succeeded') AND created_at >= $1", since)
    o, s, c = (float(costs["openai"] or 0), float(costs["storage"] or 0), float(costs["compute"] or 0)) if costs else (0, 0, 0)
    return {"openai_cost": o, "storage_cost": s, "compute_cost": c, "total_costs": o+s+c, "costs_change": 0, "cost_per_upload": round((o+s+c) / max(uploads or 1, 1), 4), "successful_uploads": uploads or 0, "total_cogs": o+s+c}


@app.get("/api/admin/kpi/growth")
async def get_kpi_growth(user: dict = Depends(require_admin)):
    since = _now_utc() - timedelta(days=30)
    async with db_pool.acquire() as conn:
        signups = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        connected = await conn.fetchval("SELECT COUNT(DISTINCT u.id) FROM users u JOIN platform_tokens pt ON u.id = pt.user_id WHERE u.created_at >= $1", since)
        uploaded = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)
        paid = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier NOT IN ('free', 'master_admin', 'friends_family', 'lifetime') AND subscription_status = 'active' AND updated_at >= $1", since)
        cancellations = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_status = 'cancelled' AND updated_at >= $1", since)
    return {"activation": {"rate": round((uploaded / max(signups, 1)) * 100, 1), "signups": signups, "connected": connected, "firstUpload": uploaded},
            "conversion": {"freeToPaid": round((paid / max(signups, 1)) * 100, 1), "trialToPaid": 0, "avgDays": 7, "count30d": paid, "change": 0},
            "attach": {"ai": 0, "topups": 0, "flex": 0, "average": 0},
            "churn": {"rate": 0, "cancellations": cancellations, "failedPayments": 0, "downgrades": 0},
            "free_to_paid_rate": round((paid / max(signups, 1)) * 100, 1), "conversion_change": 0}


@app.get("/api/admin/kpi/reliability")
async def get_kpi_reliability(user: dict = Depends(require_admin)):
    since = _now_utc() - timedelta(days=30)
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN ('completed','succeeded') THEN 1 ELSE 0 END)::int AS completed FROM uploads WHERE created_at >= $1", since)
        queue = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status IN ('pending', 'queued', 'processing')")
    total, completed = (stats["total"] or 0, stats["completed"] or 0) if stats else (0, 0)
    sr = (completed / max(total, 1)) * 100
    return {"success_rate": round(sr, 1), "reliability_change": 0, "failRates": {"ingest": 0.5, "processing": 1, "upload": round(100-sr, 1), "publish": 0.5, "average": round(100-sr, 1)},
            "retries": {"rate": 5, "one": 3, "two": 1.5, "threePlus": 0.5}, "processingTime": {"ingest": 2, "transcode": 15, "upload": 8, "average": 25},
            "cancels": {"rate": 2, "beforeProcessing": 1.5, "duringProcessing": 0.5, "total30d": 0}, "queue_depth": queue or 0}


@app.get("/api/admin/kpi/usage")
async def get_kpi_usage(user: dict = Depends(require_admin)):
    since = _now_utc() - timedelta(days=30)
    prev_since = since - timedelta(days=30)
    async with db_pool.acquire() as conn:
        active = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM uploads WHERE created_at >= $1", since)
        uploads = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1", since)
        new_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since)
        prev_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1 AND created_at < $2", prev_since, since)
        engagement = await conn.fetchrow("SELECT COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes FROM uploads WHERE created_at >= $1", since)
    chg = ((new_users - prev_users) / max(prev_users, 1)) * 100 if prev_users > 0 else 0
    return {"active_users": active or 0, "active_users_change": 0, "total_uploads": uploads or 0, "uploads_change": 0,
            "new_users": new_users or 0, "new_users_change": round(chg, 1), "total_views": engagement["views"] if engagement else 0,
            "total_likes": engagement["likes"] if engagement else 0, "avg_uploads_per_user": round((uploads or 0) / max(active or 1, 1), 1)}


@app.get("/api/admin/leaderboard")
async def get_leaderboard(range: str = Query("30d"), sort: str = Query("uploads"), user: dict = Depends(require_admin)):
    minutes = {"7d": 10080, "30d": 43200, "90d": 129600}.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        if sort == "revenue":
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, COALESCE(SUM(r.amount), 0)::decimal AS revenue, COUNT(DISTINCT up.id)::int AS uploads FROM users u LEFT JOIN revenue_tracking r ON u.id = r.user_id AND r.created_at >= $1 LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1 GROUP BY u.id ORDER BY revenue DESC LIMIT 10", since)
        else:
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, 0::decimal AS revenue, COUNT(up.id)::int AS uploads FROM users u LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1 GROUP BY u.id ORDER BY uploads DESC LIMIT 10", since)
    return [{"id": str(r["id"]), "name": r["name"] or "Unknown", "email": r["email"], "tier": r["subscription_tier"] or "free", "uploads": r["uploads"] or 0, "revenue": float(r["revenue"] or 0), "views": 0} for r in rows]


@app.get("/api/admin/countries")
async def get_countries(range: str = Query("30d"), user: dict = Depends(require_admin)):
    return []  # Add country column to users table to enable this


@app.get("/api/admin/chart/revenue")
async def get_chart_revenue(period: str = Query("30d"), user: dict = Depends(require_admin)):
    days = int(period.replace("d", "")) if period.endswith("d") and period[:-1].isdigit() else 30
    since = _now_utc() - timedelta(days=days)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT DATE(created_at) as date, COALESCE(SUM(amount), 0)::decimal as revenue FROM revenue_tracking WHERE created_at >= $1 GROUP BY DATE(created_at) ORDER BY date", since)
    data = {r["date"]: float(r["revenue"]) for r in rows}
    labels, values, current, end = [], [], since.date(), _now_utc().date()
    while current <= end:
        labels.append(current.strftime("%b %d"))
        values.append(data.get(current, 0))
        current += timedelta(days=1)
    return {"labels": labels, "values": values}


@app.get("/api/admin/chart/users")
async def get_chart_users(period: str = Query("30d"), user: dict = Depends(require_admin)):
    days = int(period.replace("d", "")) if period.endswith("d") and period[:-1].isdigit() else 30
    since = _now_utc() - timedelta(days=days)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT DATE(created_at) as date, COUNT(*)::int as users FROM users WHERE created_at >= $1 GROUP BY DATE(created_at) ORDER BY date", since)
    data = {r["date"]: r["users"] for r in rows}
    labels, values, current, end = [], [], since.date(), _now_utc().date()
    while current <= end:
        labels.append(current.strftime("%b %d"))
        values.append(data.get(current, 0))
        current += timedelta(days=1)
    return {"labels": labels, "values": values}


@app.get("/api/admin/activity")
async def get_admin_activity(limit: int = Query(10), user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        signups = await conn.fetch("SELECT 'signup' as type, id as user_id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT $1", limit // 2)
        uploads = await conn.fetch("SELECT 'upload' as type, user_id, filename, status, created_at FROM uploads ORDER BY created_at DESC LIMIT $1", limit // 2)
        payments = await conn.fetch("SELECT 'payment' as type, user_id, amount, source, created_at FROM revenue_tracking ORDER BY created_at DESC LIMIT $1", limit // 2)
    activities = []
    for s in signups:
        activities.append({"id": str(s["user_id"]), "user_id": str(s["user_id"]), "type": "signup", "description": f"{s['name'] or s['email']} signed up", "created_at": s["created_at"].isoformat() if s["created_at"] else None})
    for u in uploads:
        activities.append({"id": str(uuid.uuid4()), "user_id": str(u["user_id"]), "type": "upload", "description": f"Uploaded {u['filename'] or 'video'} ({u['status']})", "created_at": u["created_at"].isoformat() if u["created_at"] else None})
    for p in payments:
        activities.append({"id": str(uuid.uuid4()), "user_id": str(p["user_id"]) if p["user_id"] else None, "type": "payment", "description": f"${float(p['amount']):.2f} - {p['source'] or 'payment'}", "created_at": p["created_at"].isoformat() if p["created_at"] else None})
    activities.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return activities[:limit]


@app.get("/api/admin/top-users")
async def get_admin_top_users(limit: int = Query(5), sort: str = Query("revenue"), user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        if sort == "revenue":
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, COALESCE(SUM(r.amount), 0)::decimal AS revenue, COUNT(DISTINCT up.id)::int AS uploads FROM users u LEFT JOIN revenue_tracking r ON u.id = r.user_id LEFT JOIN uploads up ON u.id = up.user_id GROUP BY u.id ORDER BY revenue DESC LIMIT $1", limit)
        else:
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, 0::decimal AS revenue, COUNT(up.id)::int AS uploads FROM users u LEFT JOIN uploads up ON u.id = up.user_id GROUP BY u.id ORDER BY uploads DESC LIMIT $1", limit)
    return [{"id": str(r["id"]), "name": r["name"] or "Unknown", "email": r["email"], "tier": r["subscription_tier"] or "free", "subscription_tier": r["subscription_tier"] or "free", "revenue": float(r["revenue"] or 0), "uploads": r["uploads"] or 0} for r in rows]


# FIX: POST /api/admin/announcements - frontend calls this path (not /send)
@app.post("/api/admin/announcements")
async def post_announcements(data: AnnouncementRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    """Announcement endpoint at the path frontend expects"""
    return await send_announcement(data, background_tasks, user)


# ============================================================
# DISCORD NOTIFICATION SETTINGS
# ============================================================

class NotificationSettings(BaseModel):
    notify_mrr_charge: bool = False
    notify_topup: bool = False
    notify_upgrade: bool = False
    notify_downgrade: bool = False
    notify_cancel: bool = False
    notify_refund: bool = False
    notify_openai_cost: bool = False
    notify_storage_cost: bool = False
    notify_compute_cost: bool = False
    notify_weekly_report: bool = False
    notify_stripe_payout: bool = False
    notify_cloud_billing: bool = False
    notify_render_renewal: bool = False
    stripe_payout_day: int = 15
    cloud_billing_day: int = 1
    render_renewal_day: int = 7
    admin_webhook_url: str = ""


@app.get("/api/admin/notification-settings")
async def get_notification_settings(user: dict = Depends(require_admin)):
    """Get Discord notification settings"""
    settings = admin_settings_cache.get("notifications", {})
    return {
        "notify_mrr_charge": settings.get("notify_mrr_charge", True),
        "notify_topup": settings.get("notify_topup", True),
        "notify_upgrade": settings.get("notify_upgrade", True),
        "notify_downgrade": settings.get("notify_downgrade", True),
        "notify_cancel": settings.get("notify_cancel", True),
        "notify_refund": settings.get("notify_refund", True),
        "notify_openai_cost": settings.get("notify_openai_cost", True),
        "notify_storage_cost": settings.get("notify_storage_cost", True),
        "notify_compute_cost": settings.get("notify_compute_cost", True),
        "notify_weekly_report": settings.get("notify_weekly_report", True),
        "notify_stripe_payout": settings.get("notify_stripe_payout", True),
        "notify_cloud_billing": settings.get("notify_cloud_billing", True),
        "notify_render_renewal": settings.get("notify_render_renewal", True),
        "stripe_payout_day": settings.get("stripe_payout_day", 15),
        "cloud_billing_day": settings.get("cloud_billing_day", 1),
        "render_renewal_day": settings.get("render_renewal_day", 7),
        "admin_webhook_url": settings.get("admin_webhook_url", ADMIN_DISCORD_WEBHOOK_URL or ""),
    }


@app.put("/api/admin/notification-settings")
async def update_notification_settings(settings: NotificationSettings, user: dict = Depends(require_admin)):
    """Update Discord notification settings"""
    global admin_settings_cache
    notif_settings = {
        "notify_mrr_charge": settings.notify_mrr_charge,
        "notify_topup": settings.notify_topup,
        "notify_upgrade": settings.notify_upgrade,
        "notify_downgrade": settings.notify_downgrade,
        "notify_cancel": settings.notify_cancel,
        "notify_refund": settings.notify_refund,
        "notify_openai_cost": settings.notify_openai_cost,
        "notify_storage_cost": settings.notify_storage_cost,
        "notify_compute_cost": settings.notify_compute_cost,
        "notify_weekly_report": settings.notify_weekly_report,
        "notify_stripe_payout": settings.notify_stripe_payout,
        "notify_cloud_billing": settings.notify_cloud_billing,
        "notify_render_renewal": settings.notify_render_renewal,
        "stripe_payout_day": settings.stripe_payout_day,
        "cloud_billing_day": settings.cloud_billing_day,
        "render_renewal_day": settings.render_renewal_day,
        "admin_webhook_url": settings.admin_webhook_url,
    }
    admin_settings_cache["notifications"] = notif_settings
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1", json.dumps(admin_settings_cache))
    return {"status": "updated", "settings": notif_settings}


@app.post("/api/admin/test-webhook")
async def test_webhook(data: dict, user: dict = Depends(require_admin)):
    """Send a test message to the provided Discord webhook"""
    webhook_url = data.get("webhook_url", "").strip()
    if not webhook_url:
        raise HTTPException(400, "Webhook URL required")
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        raise HTTPException(400, "Invalid Discord webhook URL")
    test_embed = {
        "title": "🔔 UploadM8 Webhook Test",
        "description": "If you see this message, your webhook is configured correctly!",
        "color": 0x22c55e,
        "fields": [
            {"name": "Status", "value": "✅ Connected", "inline": True},
            {"name": "Tested By", "value": user.get("email", "Admin"), "inline": True},
        ],
        "footer": {"text": "UploadM8 Admin Notifications"},
        "timestamp": _now_utc().isoformat()
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook_url, json={"embeds": [test_embed]})
            if r.status_code not in (200, 204):
                raise HTTPException(400, f"Discord returned status {r.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(400, "Webhook request timed out")
    except Exception as e:
        raise HTTPException(400, f"Failed to send: {str(e)}")
    return {"status": "sent"}


# ============================================================
# DISCORD NOTIFICATION HELPERS
# ============================================================

def get_notif_settings():
    return admin_settings_cache.get("notifications", {})

def get_admin_webhook():
    settings = get_notif_settings()
    return settings.get("admin_webhook_url") or ADMIN_DISCORD_WEBHOOK_URL

def mask_email(email: str) -> str:
    if not email or "@" not in email: return email
    local, domain = email.split("@", 1)
    return f"{local[:2]}***@{domain}" if len(local) > 2 else f"{local[0]}***@{domain}"


async def notify_revenue_event(event_type: str, email: str, tier: str, amount: float, stripe_id: str = None, extra_fields: list = None):
    """Send revenue event notification to Discord"""
    settings = get_notif_settings()
    webhook = get_admin_webhook()
    if not webhook: return
    setting_map = {"mrr_charge": "notify_mrr_charge", "topup": "notify_topup", "upgrade": "notify_upgrade", "downgrade": "notify_downgrade", "cancel": "notify_cancel", "refund": "notify_refund"}
    if setting_map.get(event_type) and not settings.get(setting_map[event_type], True): return
    event_config = {
        "mrr_charge": {"emoji": "💰", "color": 0x22c55e, "title": "MRR Charge"},
        "topup": {"emoji": "💳", "color": 0x8b5cf6, "title": "Top-up Purchase"},
        "upgrade": {"emoji": "⬆️", "color": 0x3b82f6, "title": "Plan Upgrade"},
        "downgrade": {"emoji": "⬇️", "color": 0xf59e0b, "title": "Plan Downgrade"},
        "cancel": {"emoji": "❌", "color": 0xef4444, "title": "Subscription Cancelled"},
        "refund": {"emoji": "↩️", "color": 0xf97316, "title": "Refund Processed"},
    }
    cfg = event_config.get(event_type, {"emoji": "📊", "color": 0x6b7280, "title": event_type.title()})
    fields = [{"name": "User", "value": mask_email(email), "inline": True}, {"name": "Tier", "value": tier.title(), "inline": True}, {"name": "Amount", "value": f"${amount:.2f}", "inline": True}]
    if stripe_id: fields.append({"name": "Stripe ID", "value": f"`{stripe_id[:20]}...`" if len(stripe_id) > 20 else f"`{stripe_id}`", "inline": False})
    if extra_fields: fields.extend(extra_fields)
    await discord_notify(webhook, embeds=[{"title": f"{cfg['emoji']} {cfg['title']}", "color": cfg["color"], "fields": fields, "footer": {"text": "UploadM8 Revenue Alert"}, "timestamp": _now_utc().isoformat()}])


async def notify_cost_report(report_type: str, costs: dict, period: str = "Weekly"):
    """Send cost report notification to Discord"""
    settings = get_notif_settings()
    webhook = get_admin_webhook()
    if not webhook: return
    setting_map = {"openai": "notify_openai_cost", "storage": "notify_storage_cost", "compute": "notify_compute_cost", "weekly": "notify_weekly_report"}
    if setting_map.get(report_type) and not settings.get(setting_map[report_type], True): return
    if report_type == "weekly":
        total_cost = costs.get("openai", 0) + costs.get("storage", 0) + costs.get("compute", 0)
        revenue = costs.get("revenue", 0)
        margin = revenue - total_cost
        margin_pct = (margin / max(revenue, 1)) * 100
        embed = {"title": "📊 Weekly Cost Report", "color": 0x3b82f6, "fields": [
            {"name": "OpenAI", "value": f"${costs.get('openai', 0):.2f}", "inline": True},
            {"name": "Storage", "value": f"${costs.get('storage', 0):.2f}", "inline": True},
            {"name": "Compute", "value": f"${costs.get('compute', 0):.2f}", "inline": True},
            {"name": "Total COGS", "value": f"${total_cost:.2f}", "inline": True},
            {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
            {"name": "Margin", "value": f"${margin:.2f} ({margin_pct:.1f}%)", "inline": True},
        ], "footer": {"text": f"UploadM8 {period} Report"}, "timestamp": _now_utc().isoformat()}
    else:
        titles = {"openai": "🤖 OpenAI Cost", "storage": "💾 Storage Cost", "compute": "⚡ Compute Cost"}
        embed = {"title": titles.get(report_type, "📈 Cost Report"), "color": 0xef4444, "fields": [
            {"name": "Cost", "value": f"${costs.get('amount', 0):.2f}", "inline": True},
            {"name": "Units", "value": str(costs.get("units", "N/A")), "inline": True},
            {"name": "vs Last Week", "value": f"{costs.get('change', 0):+.1f}%", "inline": True},
        ], "footer": {"text": f"UploadM8 {period} Report"}, "timestamp": _now_utc().isoformat()}
    await discord_notify(webhook, embeds=[embed])


async def notify_billing_reminder(reminder_type: str, date: str, amount: float = None, service: str = None):
    """Send billing calendar reminder to Discord"""
    settings = get_notif_settings()
    webhook = get_admin_webhook()
    if not webhook: return
    setting_map = {"stripe_payout": "notify_stripe_payout", "cloud_billing": "notify_cloud_billing", "render_renewal": "notify_render_renewal"}
    if setting_map.get(reminder_type) and not settings.get(setting_map[reminder_type], True): return
    config = {"stripe_payout": {"emoji": "💸", "color": 0x6366f1, "title": "Stripe Payout Coming"}, "cloud_billing": {"emoji": "☁️", "color": 0xf97316, "title": "Cloud Billing Reminder"}, "render_renewal": {"emoji": "🚀", "color": 0x06b6d4, "title": "Render Renewal Reminder"}}
    cfg = config.get(reminder_type, {"emoji": "📅", "color": 0x6b7280, "title": "Billing Reminder"})
    fields = [{"name": "Date", "value": date, "inline": True}]
    if service: fields.append({"name": "Service", "value": service, "inline": True})
    if amount: fields.append({"name": "Est. Amount", "value": f"${amount:.2f}", "inline": True})
    await discord_notify(webhook, embeds=[{"title": f"{cfg['emoji']} {cfg['title']}", "description": "Upcoming billing event in 2 days", "color": cfg["color"], "fields": fields, "footer": {"text": "UploadM8 Billing Calendar"}, "timestamp": _now_utc().isoformat()}])


@app.post("/api/admin/check-billing-reminders")
async def check_billing_reminders(user: dict = Depends(require_admin)):
    """Check and send billing reminders for upcoming dates"""
    settings = get_notif_settings()
    today = _now_utc().day
    reminders_sent = []
    stripe_day = settings.get("stripe_payout_day", 15)
    cloud_day = settings.get("cloud_billing_day", 1)
    render_day = settings.get("render_renewal_day", 7)
    if (stripe_day - 2) == today or (stripe_day - 2 + 28) % 28 == today:
        await notify_billing_reminder("stripe_payout", f"Day {stripe_day} of this month", service="Stripe Payouts")
        reminders_sent.append("stripe_payout")
    if (cloud_day - 2) == today or (cloud_day - 2 + 28) % 28 == today:
        await notify_billing_reminder("cloud_billing", f"Day {cloud_day} of this month", service="AWS/Cloudflare")
        reminders_sent.append("cloud_billing")
    if (render_day - 2) == today or (render_day - 2 + 28) % 28 == today:
        await notify_billing_reminder("render_renewal", f"Day {render_day} of this month", service="Render Hosting")
        reminders_sent.append("render_renewal")
    return {"status": "checked", "reminders_sent": reminders_sent}

# ============================================================
# TRILL TELEMETRY API ENDPOINTS
# ============================================================

@app.post("/api/trill/analyze/{upload_id}")
async def analyze_telemetry(upload_id: str, user: dict = Depends(get_current_user)):
    """Manually trigger trill analysis on an upload with telemetry data"""
    import tempfile
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        if not upload.get("telemetry_r2_key"):
            raise HTTPException(400, "No telemetry data for this upload")
        
        # Download files from R2
        s3 = get_s3_client()
        
        video_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=upload["r2_key"])
        video_data = video_obj["Body"].read()
        
        telem_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=upload["telemetry_r2_key"])
        telem_data = telem_obj["Body"].read()
        
        # Write to temp files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
            vf.write(video_data)
            video_path = vf.name
        
        with tempfile.NamedTemporaryFile(suffix=".map", delete=False) as tf:
            tf.write(telem_data)
            map_path = tf.name
        
        try:
            # Get user preferences
            prefs = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1",
                user["id"]
            )
            user_prefs = dict(prefs) if prefs else {}
            
            # Process
            result = await process_telemetry(
                conn, upload_id, user["id"],
                video_path, map_path, user_prefs
            )
            
            return {
                "success": True,
                "trill_score": result["trill_metadata"].get("trill_score"),
                "speed_bucket": result["trill_metadata"].get("speed_bucket"),
                "ai_enhanced": bool(result.get("ai_content")),
                "hud_generated": bool(result.get("hud_path")),
                "ai_content": result.get("ai_content")
            }
        finally:
            os.unlink(video_path)
            os.unlink(map_path)


@app.get("/api/trill/places")
async def get_trill_places(
    state: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    user: dict = Depends(get_current_user)
):
    """Get popular trill places for targeting"""
    async with db_pool.acquire() as conn:
        if state:
            places = await conn.fetch(
                "SELECT * FROM trill_places WHERE state = $1 ORDER BY popularity_score DESC LIMIT $2",
                state.upper(), limit
            )
        else:
            places = await conn.fetch(
                "SELECT * FROM trill_places ORDER BY popularity_score DESC LIMIT $1",
                limit
            )
    
    return [dict(p) for p in places]


@app.post("/api/trill/generate-preview")
async def generate_trill_preview(
    data: dict = Body(...),
    user: dict = Depends(get_current_user)
):
    """Preview AI-generated content without saving"""
    trill_metadata = data.get("trill_metadata", {})
    
    # Get user preferences
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow(
            "SELECT * FROM user_preferences WHERE user_id = $1",
            user["id"]
        )
        user_prefs = dict(prefs) if prefs else {}
    
    result = generate_trill_content(trill_metadata, user_prefs)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


@app.get("/api/uploads/{upload_id}")
async def get_upload_details(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Upload detail for current user.

    Contract (frontend-safe):
      - thumbnail_url: presigned R2 URL (if thumbnail_r2_key exists)
      - platform_results: always list
      - hashtags: always list[str]
      - title/caption: falls back to AI values when empty
      - ai_title/ai_caption/ai_hashtags always present
      - duration_seconds computed from processing timestamps when available
    """
    cols = await _load_uploads_columns(db_pool)
    wanted = [
        "id","user_id","r2_key","filename","file_size","platforms",
        "title","caption","hashtags","privacy","status",
        "scheduled_time","created_at","completed_at",
        "put_reserved","aic_reserved",
        "error_code","error_detail",
        "thumbnail_r2_key","platform_results",
        "processing_started_at","processing_finished_at",
        "processing_stage","processing_progress",
        "views","likes",
        # AI fields (older/newer schema variants)
        "ai_title","ai_caption",
        "ai_generated_title","ai_generated_caption","ai_generated_hashtags",
    ]
    select_cols = _pick_cols(wanted, cols) or ["id","user_id","r2_key","filename","platforms","status","created_at"]
    sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE id = $1 AND user_id = $2"

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(sql, upload_id, user["id"])

    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")

    d = dict(row)

    def _normalize_platform_results(raw):
        pr = _safe_json(raw, [])
        if isinstance(pr, list):
            items = [x for x in pr if isinstance(x, dict)]
        elif isinstance(pr, dict):
            items = []
            for k, v in pr.items():
                if isinstance(v, dict):
                    items.append({"platform": k, **v})
                else:
                    items.append({"platform": k, "value": v})
        else:
            return []
        out = []
        for item in items:
            d = dict(item)
            if d.get("platform_video_id") and not d.get("video_id"):
                d["video_id"] = d["platform_video_id"]
            if d.get("platform_url") and not d.get("url"):
                d["url"] = d["platform_url"]
            out.append(d)
        return out

    def _normalize_hashtags(raw):
        tags = _safe_json(raw, [])
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        if isinstance(tags, str) and tags.strip():
            # Support single string containing many tags
            s = tags.strip().replace('
', ' ').replace(',', ' ')
            parts = [p.strip() for p in s.split() if p.strip()]
            return parts
        return []

    ai_title = (d.get("ai_title") or d.get("ai_generated_title") or "") or ""
    ai_caption = (d.get("ai_caption") or d.get("ai_generated_caption") or "") or ""
    ai_hashtags = _normalize_hashtags(d.get("ai_generated_hashtags"))

    title = (d.get("title") or "").strip() or ai_title
    caption = (d.get("caption") or "").strip() or ai_caption
    hashtags = _normalize_hashtags(d.get("hashtags"))
    platform_results = _normalize_platform_results(d.get("platform_results"))

    thumbnail_url = None
    thumb_key = d.get("thumbnail_r2_key")
    if thumb_key:
        try:
            s3 = get_s3_client()
            thumbnail_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(thumb_key)},
                ExpiresIn=3600,
            )
        except Exception:
            thumbnail_url = None

    duration_seconds = None
    ps = d.get("processing_started_at")
    pf = d.get("processing_finished_at")
    if ps and pf:
        try:
            duration_seconds = int((pf - ps).total_seconds())
        except Exception:
            duration_seconds = None

    return {
        "id": str(d.get("id")),
        "filename": d.get("filename"),
        "r2_key": d.get("r2_key"),
        "platforms": list(d.get("platforms") or []),
        "status": d.get("status"),
        "privacy": d.get("privacy", "public"),

        "title": title,
        "caption": caption,
        "hashtags": hashtags,

        "ai_title": ai_title,
        "ai_caption": ai_caption,
        "ai_hashtags": ai_hashtags,

        "scheduled_time": d.get("scheduled_time").isoformat() if d.get("scheduled_time") else None,
        "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
        "completed_at": d.get("completed_at").isoformat() if d.get("completed_at") else None,

        "put_cost": int(d.get("put_reserved") or 0),
        "aic_cost": int(d.get("aic_reserved") or 0),

        "error_code": d.get("error_code"),
        "error": d.get("error_detail") or d.get("error_code") or None,

        "thumbnail_url": thumbnail_url,
        "platform_results": platform_results,

        "file_size": d.get("file_size"),
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),

        "progress": int(d.get("processing_progress") or 0),
        "current_stage": d.get("processing_stage"),
        "duration_seconds": duration_seconds,

        # camelCase aliases required by upload.html pollUpload() tick loop
        # The poller keys on processingStartedAt to flip from Queued -> Processing
        "processingStartedAt":  d.get("processing_started_at").isoformat() if d.get("processing_started_at") else None,
        "processingFinishedAt": d.get("processing_finished_at").isoformat() if d.get("processing_finished_at") else None,
        "processingProgress":   int(d.get("processing_progress") or 0),
        "processingStage":      d.get("processing_stage"),
    }


@app.get("/api/admin/users/{user_id}/wallet")
async def admin_get_user_wallet(
    user_id: str,
    admin: dict = Depends(require_admin),
):
    """Return the wallet + recent ledger for any user (admin only)."""
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT id, email, name, subscription_tier FROM users WHERE id = $1",
            user_id,
        )
        if not target:
            raise HTTPException(404, "User not found")

        wallet = await get_wallet(conn, target["id"])

        ledger = await conn.fetch(
            """
            SELECT token_type, delta, reason, upload_id, meta, created_at
            FROM   token_ledger
            WHERE  user_id = $1
            ORDER  BY created_at DESC
            LIMIT  100
            """,
            target["id"],
        )

    plan = get_plan(target["subscription_tier"] or "free")

    return {
        "user": {
            "id":    str(target["id"]),
            "email": target["email"],
            "name":  target["name"],
            "tier":  target["subscription_tier"],
        },
        "wallet": {
            "put_balance":  int(wallet.get("put_balance", 0)),
            "aic_balance":  int(wallet.get("aic_balance", 0)),
            "put_reserved": int(wallet.get("put_reserved", 0)),
            "aic_reserved": int(wallet.get("aic_reserved", 0)),
        },
        "plan_limits": {
            "put_daily":   int(plan.get("put_daily", 0)),
            "put_monthly": int(plan.get("put_monthly", 0)),
            "aic_monthly": int(plan.get("aic_monthly", 0)),
        },
        "ledger": [dict(r) for r in ledger],
    }

@app.post("/api/admin/users/{user_id}/wallet/adjust")
async def admin_adjust_wallet(
    user_id:  str,
    payload:  AdminWalletAdjust,
    request:  Request,
    admin:    dict = Depends(require_admin),
):
    """
    Manually adjust a user's PUT or AIC balance.

    Modes:
      set      → set balance to exactly this amount
      add      → add tokens to current balance
      subtract → subtract tokens (clamped to 0 — never goes negative)

    All changes are written to token_ledger and admin_audit_log.
    """
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT id, email FROM users WHERE id = $1", user_id
        )
        if not target:
            raise HTTPException(404, "User not found")

        wallet = await get_wallet(conn, target["id"])
        col    = "put_balance" if payload.wallet == "put" else "aic_balance"
        before = int(wallet.get(col, 0) or 0)

        if payload.mode == "set":
            new_val = int(payload.amount)
            delta   = new_val - before
        elif payload.mode == "add":
            new_val = before + int(payload.amount)
            delta   = int(payload.amount)
        else:  # subtract
            new_val = max(0, before - int(payload.amount))
            delta   = new_val - before  # negative

        await conn.execute(
            f"UPDATE wallets SET {col} = $1, updated_at = NOW() WHERE user_id = $2",
            new_val,
            target["id"],
        )

        await conn.execute(
            """
            INSERT INTO token_ledger
                (user_id, token_type, delta, reason, meta)
            VALUES
                ($1, $2, $3, $4,
                 jsonb_build_object(
                     'ref_type',    'admin_adjust',
                     'admin_id',    $5::text,
                     'admin_email', $6,
                     'mode',        $7,
                     'before',      $8,
                     'after',       $9,
                     'reason',      $10
                 ))
            """,
            target["id"],
            payload.wallet,
            int(delta),
            f"admin_{payload.mode}_{payload.wallet}",
            str(admin.get("id")),
            admin.get("email"),
            payload.mode,
            before,
            new_val,
            payload.reason,
        )

        try:
            await conn.execute(
                """
                INSERT INTO admin_audit_log
                    (user_id, admin_id, admin_email, action, details, ip_address)
                VALUES
                    ($1, $2, $3, $4, $5::jsonb, $6)
                """,
                target["id"],
                admin.get("id"),
                admin.get("email"),
                "ADMIN_WALLET_ADJUST",
                json.dumps({
                    "wallet": payload.wallet,
                    "mode":   payload.mode,
                    "amount": int(payload.amount),
                    "delta":  int(delta),
                    "before": before,
                    "after":  new_val,
                    "reason": payload.reason,
                }),
                request.client.host if request.client else None,
            )
        except Exception:
            # admin_audit_log may not exist in some environments; ledger is the source of truth
            pass

    return {
        "ok":     True,
        "wallet": payload.wallet,
        "mode":   payload.mode,
        "before": before,
        "after":  new_val,
        "delta":  int(delta),
    }

