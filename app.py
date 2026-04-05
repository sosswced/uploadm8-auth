"""
UploadM8 API Server - Production Build v4

Route map (handlers in this module; integrators may call /api/v1/* — aliased to /api/*):
  Auth:      /api/auth/* (register, login, refresh, logout, email confirm, password, …)
  Session:   /api/me, /api/me/preferences
  Settings:  /api/settings* — see canonical note above GET /api/me/preferences
  Uploads:   /api/uploads/* (presign, complete, list, queue, …)
  Billing:   /api/billing/*, /api/stripe/webhook (duplicate mount)
  OAuth:     /api/oauth/*, platform webhooks (TikTok, Meta, …)
  Admin:     /api/admin/*
  Catalog:   /api/catalog/*
  Ops:       /health, /ready, /metrics

Also: PUT/AIC wallet + ledger, announcements, KPI dashboards, Stripe.
"""

import os
import pathlib
import csv

# Load .env before any config reads (needed for local dev when running via uvicorn)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import io
import json
import logging
import re
import asyncio
import calendar
import math


# ---------------------------------------------------------------------------
# DB JSON CODECS — shared implementation (stages.asyncpg_json_codecs)
# ---------------------------------------------------------------------------
from stages.asyncpg_json_codecs import apply_asyncpg_json_codecs as _init_asyncpg_codecs

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
        except json.JSONDecodeError:
            return default
    return default
import secrets
import hashlib
import base64
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional
from decimal import Decimal
from html import escape
from ipaddress import ip_address
from urllib.parse import urlencode, quote, urlsplit, urlunsplit, parse_qsl, urlparse

# Sensitive query-param keys that must never appear in logs
_SENSITIVE_KEYS = {"access_token", "client_secret", "code", "refresh_token", "fb_exchange_token"}

def _valid_uuid(s: str) -> bool:
    """Return True if s is a valid UUID string (avoids 500 when frontend sends 'undefined' etc)."""
    if not s or not isinstance(s, str) or len(s) != 36:
        return False
    try:
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False

def redact_url(url: str) -> str:
    """Strip sensitive query params from a URL before logging it."""
    try:
        parts = urlsplit(url)
        q = parse_qsl(parts.query, keep_blank_values=True)
        redacted = [(k, "***" if k in _SENSITIVE_KEYS else v) for k, v in q]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted), parts.fragment))
    except Exception as e:
        logging.getLogger("uploadm8-api").debug("redact_url failed: %s", e)
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
from fastapi.responses import (
    RedirectResponse,
    Response,
    StreamingResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
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
    styled_thumbnails: bool = Field(True, alias="styledThumbnails")
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
    auth_security_alerts: bool = Field(True, alias="authSecurityAlerts")
    digest_emails: bool = Field(True, alias="digestEmails")
    scheduled_alert_emails: bool = Field(True, alias="scheduledAlertEmails")
    discord_webhook: Optional[str] = Field(None, alias="discordWebhook")
    # Caption & AI (stored in users.preferences; worker caption_stage reads these)
    caption_style: Literal["story", "punchy", "factual"] = Field("story", alias="captionStyle")
    caption_tone: Literal["hype", "calm", "cinematic", "authentic"] = Field("authentic", alias="captionTone")
    caption_voice: Literal["default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator"] = Field("default", alias="captionVoice")
    caption_frame_count: int = Field(6, ge=2, le=12, alias="captionFrameCount")

    class Config:
        populate_by_name = True
        extra = "ignore"



LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_JSON_LOGS = os.environ.get("JSON_LOGS", "1").strip().lower() in ("1", "true", "yes")

class _JsonFormatter(logging.Formatter):
    def format(self, record):
        import json as _j
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "trace_id") and record.trace_id:
            entry["trace_id"] = record.trace_id
        if record.exc_info and record.exc_info[0]:
            entry["exc"] = self.formatException(record.exc_info)
        return _j.dumps(entry, default=str, ensure_ascii=False)

_handler = logging.StreamHandler()
if _JSON_LOGS:
    _handler.setFormatter(_JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.root.handlers = [_handler]
logging.root.setLevel(LOG_LEVEL)
logger = logging.getLogger("uploadm8-api")


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers (mean Earth radius 6371 km). Canonical for Trill / geo helpers."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    a = min(1.0, max(0.0, a))
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Sentry (opt-in via SENTRY_DSN) ──────────────────────────
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.asyncpg import AsyncPGIntegration
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            environment=os.environ.get("SENTRY_ENV", "production"),
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_RATE", "0.1")),
            send_default_pii=False,
            integrations=[FastApiIntegration(), AsyncPGIntegration()],
        )
        logger.info("Sentry initialised (env=%s)", os.environ.get("SENTRY_ENV", "production"))
    except ImportError:
        logger.warning("SENTRY_DSN set but sentry-sdk not installed — skipping")

# ============================================================
# Configuration
# ============================================================
DATABASE_URL = os.environ.get("DATABASE_URL")
BASE_URL = os.environ.get("BASE_URL", "https://auth.uploadm8.com")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("Missing JWT_SECRET env var")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.uploadm8.com")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "uploadm8-app")
ACCESS_TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.environ.get("REFRESH_TOKEN_DAYS", "30"))
TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://app.uploadm8.com,https://uploadm8.com,"
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:8080,http://127.0.0.1:8080",
)
ALLOWED_ORIGINS_LIST = [x.strip() for x in ALLOWED_ORIGINS.split(",") if x.strip()]
# Empty ALLOWED_ORIGINS in .env (e.g. ALLOWED_ORIGINS=) breaks CORS entirely — preflight has no ACAO.
if not ALLOWED_ORIGINS_LIST:
    ALLOWED_ORIGINS_LIST = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
# Any port on loopback (npx serve -p 8080, Vite, etc.). Set CORS_STRICT_ORIGINS=1 in production to disable.
_CORS_STRICT = os.environ.get("CORS_STRICT_ORIGINS", "").strip().lower() in ("1", "true", "yes", "on")
_CORS_LOCAL_REGEX = None if _CORS_STRICT else r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"
# When strict, the regex above is off — only ALLOWED_ORIGINS_LIST applies. Production envs often list 8080
# but not 3000/5173, which breaks Vite / common dev servers. Merge these unless opted out.
_CORS_LOCAL_DEV_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)
_CORS_ALLOW_LOCAL_DEV = os.environ.get("CORS_ALLOW_LOCAL_DEV", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Always merge into the explicit list when enabled (not only under CORS_STRICT_ORIGINS).
# Production preflight was returning 400 with no ACAO when the regex path failed to match
# in some deployments; explicit origins are the reliable fix.
if _CORS_ALLOW_LOCAL_DEV:
    _cors_seen = set(ALLOWED_ORIGINS_LIST)
    for _o in _CORS_LOCAL_DEV_ORIGINS:
        if _o not in _cors_seen:
            ALLOWED_ORIGINS_LIST.append(_o)
            _cors_seen.add(_o)


def _cors_reflect_origin(request) -> str:
    """Pick Access-Control-Allow-Origin for error responses (matches CORSMiddleware rules)."""
    origin = (request.headers.get("origin") or "").strip()
    if origin in ALLOWED_ORIGINS_LIST:
        return origin
    if _CORS_LOCAL_REGEX and origin and re.fullmatch(_CORS_LOCAL_REGEX, origin):
        return origin
    return ALLOWED_ORIGINS_LIST[0] if ALLOWED_ORIGINS_LIST else "*"


BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()
TRUST_PROXY_HEADERS = os.environ.get("TRUST_PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes", "on")

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
# Presigned upload URL TTL (seconds). Default 2h for large/slow uploads; increase if users hit "R2 upload network error" due to expiry.
R2_PRESIGN_UPLOAD_TTL = int(os.environ.get("R2_PRESIGN_UPLOAD_TTL", "7200"))
# When true, presigned PUT does not bind Content-Type in the signature. Use if frontend sends file.type that differs from presign
# (empty or browser-specific) and R2 returns 403 — avoids "network error" from failed PUT. Object may need Content-Type set later if required.
R2_PRESIGN_PUT_UNSIGNED_CONTENT = os.environ.get("R2_PRESIGN_PUT_UNSIGNED_CONTENT", "").strip().lower() in ("1", "true", "yes", "on")

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

# "live" | "test" — explicit BILLING_MODE wins; else infer from Stripe secret (sk_live_* → live).
_billing_env = os.environ.get("BILLING_MODE", "").strip().lower()
if _billing_env in ("live", "test"):
    BILLING_MODE = _billing_env
else:
    BILLING_MODE = "live" if STRIPE_SECRET_KEY.startswith("sk_live") else "test"

# Discord Webhooks
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
SIGNUP_DISCORD_WEBHOOK_URL = os.environ.get("SIGNUP_DISCORD_WEBHOOK_URL", "")
MRR_DISCORD_WEBHOOK_URL = os.environ.get("MRR_DISCORD_WEBHOOK_URL", "")
COMMUNITY_DISCORD_WEBHOOK_URL = (
    os.getenv("DISCORD_COMMUNITY_WEBHOOK_URL", "").strip()
    or os.getenv("DISCORD_COMMUNITY_WEBHOOK", "").strip()
    or os.getenv("COMMUNITY_DISCORD_WEBHOOK_URL", "").strip()
)
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
- Titles: 1-2 emojis max (fire , lightning , eyes )
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
# Set True at the start of lifespan shutdown so HTTP middleware can 503 new work early.
app_shutting_down: bool = False
ENC_KEYS: Dict[str, bytes] = {}
CURRENT_KEY_ID = "v1"
admin_settings_cache: Dict[str, Any] = {
    "demo_data_enabled": False,
    "billing_mode": BILLING_MODE,
    "promo_burst_week_enabled": False,
    "promo_referral_enabled": False,
}

# Keep in sync with the highest migration tuple in run_migrations().
MIGRATIONS_LATEST_VERSION = 823
MIGRATIONS_CRITICAL_VERSIONS = [606, 607, 608, 609, 811]

# ============================================================
# Plan Configuration (PUT/AIC based)
# ============================================================
# ── Entitlements: single source of truth ──────────────────────
# PLAN_CONFIG removed. All tier data lives in entitlements.py.
# Import everything we need from there.
from stages.context import expand_hashtag_items
from stages.entitlements import (
    TIER_CONFIG,
    STRIPE_LOOKUP_TO_TIER,
    TOPUP_PRODUCTS,
    PRIORITY_QUEUE_CLASSES,
    TIER_SLUGS,
    ENTITLEMENT_KEYS,
    ADMIN_KPI_COUNTED_SUBSCRIPTION_TIERS,
    normalize_tier,
    get_tier_display_name,
    get_next_public_upgrade_tier,
    get_tiers_for_api,
    get_entitlements_for_tier,
    get_entitlements_from_user,
    entitlements_to_dict,
    check_queue_depth,
    can_user_connect_platform,
    compute_upload_cost,   # canonical PUT/AIC formula
)
from services.wallet import (
    get_wallet,
    ledger_entry,
    reserve_tokens,
    spend_tokens,
    refund_tokens,
    credit_wallet,
    transfer_tokens,
    daily_refill,
)
from services.wallet_marketing import build_wallet_marketing_payload
from services.billing import (
    _tier_is_upgrade,
    get_plan,
    ensure_stripe_customer,
    create_wallet_topup_checkout_session,
    create_billing_checkout_session,
)
from services.uploads import (
    calculate_smart_schedule,
    get_existing_scheduled_days,
)
from services.thumbnail_studio import (
    extract_youtube_video_id,
    fetch_youtube_title,
    estimate_studio_cost,
    estimate_pikzels_v2_call_cost,
    format_library_rows,
    generate_recreate_variants,
)
from services.notifications import (
    discord_notify as _discord_notify_service,
    notify_signup as _notify_signup_service,
    notify_mrr as _notify_mrr_service,
    notify_topup as _notify_topup_service,
    notify_weekly_costs as _notify_weekly_costs_service,
)
from services import metric_definitions as metric_defs
from services.api_errors import api_problem
from services.meta_oauth import (
    fetch_granted_permissions,
    meta_facebook_oauth_scope,
    meta_instagram_oauth_scope,
    meta_oauth_mode,
)
from services.meta_graph_metrics import (
    facebook_page_feed_reel_engagement_rollups,
    instagram_account_degraded_live,
)
from services.platform_channels import (
    list_analytics_platform_query_values,
    resolve_analytics_platform_filter,
)
from services.upload_metrics import (
    ANALYTICS_OVERVIEW_PLATFORMS,
    SUCCESSFUL_STATUS_SQL_IN,
    SUCCESSFUL_UPLOAD_STATUSES,
)

# ── Email notifications ───────────────────────────────────────────────────────
from stages.emails import (
    # Auth
    send_signup_confirmation_email as send_signup_confirmation_email_v2,
    send_welcome_email,
    send_fully_signed_up_guide_email,
    send_password_reset_email,
    send_password_changed_email,
    send_account_deleted_email,
    send_email_change_email,
    send_admin_email_change_notice_to_old_email,
    send_user_email_change_notice_to_old_email,
    send_admin_reset_password_email,
    send_login_anomaly_email,
    # Billing — Subscriptions
    send_subscription_started_email,
    send_trial_started_email,
    send_trial_cancelled_email,
    send_subscription_cancelled_email,
    send_renewal_receipt_email,
    # Billing — Changes
    send_plan_upgraded_email,
    send_plan_downgraded_email,
    send_topup_receipt_email,
    send_refund_receipt_email,
    # Announcements
    send_announcement_email,
    # Heartfelt welcomes
    send_friends_family_welcome_email,
    send_agency_welcome_email,
    send_master_admin_welcome_email,
    # Admin actions
    send_admin_wallet_topup_email,
    send_admin_tier_switch_email,
    send_admin_account_status_email,
    # Lifecycle
    send_payment_failed_email,
    send_trial_ending_reminder_email,
    send_low_token_warning_email,
    send_monthly_user_kpi_digest_email,
    send_admin_weekly_kpi_digest_email,
    send_report_ready_email,
    send_scheduled_publish_alert_email,
)
from stages.emails.base import send_email, MAIL_FROM_SUPPORT, URL_BILLING, URL_SETTINGS, SUPPORT_EMAIL

# ============================================================
# Helpers
# ============================================================
def _now_utc(): return datetime.now(timezone.utc)
def _sha256_hex(s: str): return hashlib.sha256(s.encode()).hexdigest()
def _req_id(): return f"req_{int(time.time())}_{secrets.token_hex(4)}"

# Email digest / reminder cron cadence
EMAIL_CRON_INTERVAL_SECONDS = int(os.environ.get("EMAIL_CRON_INTERVAL_SECONDS", "3600"))


async def _run_trial_ending_reminders_once():
    """Send 3-day trial ending reminders (idempotent per trial period)."""
    now = _now_utc()
    window_end = now + timedelta(days=4)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, email, name, subscription_tier, trial_end, current_period_end, trial_reminder_sent_at
            FROM users
            WHERE subscription_status = 'trialing'
              AND trial_end IS NOT NULL
              AND trial_end > $1
              AND trial_end <= $2
              AND status = 'active'
            """,
            now, window_end,
        )
        for r in rows:
            trial_end = r.get("trial_end")
            if not trial_end:
                continue
            days_left = max(1, (trial_end.date() - now.date()).days)
            if days_left > 3:
                continue
            sent_at = r.get("trial_reminder_sent_at")
            if sent_at and sent_at.date() >= (trial_end - timedelta(days=3)).date():
                continue
            try:
                amount = float(get_plan(r.get("subscription_tier") or "free").get("price", 0.0) or 0.0)
            except Exception:
                amount = 0.0
            await send_trial_ending_reminder_email(
                r["email"],
                r["name"] or "there",
                r.get("subscription_tier") or "free",
                trial_end.strftime("%B %d, %Y"),
                days_left=days_left,
                amount=amount,
            )
            await conn.execute(
                "UPDATE users SET trial_reminder_sent_at = NOW(), updated_at = NOW() WHERE id = $1",
                r["id"],
            )


async def _run_monthly_user_kpi_digests_once():
    """Send one monthly KPI digest per active user per month."""
    now = _now_utc()
    period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    if now.day != 1:
        return
    prev_end = period_start
    prev_start = datetime(prev_end.year, prev_end.month, 1, tzinfo=timezone.utc) - timedelta(days=1)
    prev_start = datetime(prev_start.year, prev_start.month, 1, tzinfo=timezone.utc)
    period_label = prev_start.strftime("%B %Y")

    async with db_pool.acquire() as conn:
        users = await conn.fetch(
            """
            SELECT u.id, u.email, u.name, u.subscription_tier, u.monthly_digest_period,
                   COALESCE(up.email_notifications, TRUE) AS email_notifications,
                   COALESCE(up.digest_emails, TRUE) AS digest_emails
            FROM users u
            LEFT JOIN user_preferences up ON up.user_id = u.id
            WHERE u.status = 'active'
              AND u.email_verified = TRUE
              AND COALESCE(up.email_notifications, TRUE) = TRUE
              AND COALESCE(up.digest_emails, TRUE) = TRUE
            """
        )
        for u in users:
            if u.get("monthly_digest_period") and u["monthly_digest_period"] >= period_start.date():
                continue
            stats = await conn.fetchrow(
                f"""
                SELECT
                  COUNT(*)::int AS uploads,
                  COUNT(*) FILTER (WHERE status IN {SUCCESSFUL_STATUS_SQL_IN})::int AS success_uploads,
                  COALESCE(SUM(put_spent),0)::bigint AS put_from_uploads,
                  COALESCE(SUM(aic_spent),0)::bigint AS aic_from_uploads
                FROM uploads
                WHERE user_id = $1
                  AND created_at >= $2
                  AND created_at < $3
                """,
                u["id"], prev_start, prev_end,
            )
            ledger = await conn.fetchrow(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN token_type = 'put' AND reason = 'spend' AND delta < 0
                    THEN ABS(delta::bigint) ELSE 0 END), 0)::bigint AS put_used,
                  COALESCE(SUM(CASE WHEN token_type = 'aic' AND reason = 'spend' AND delta < 0
                    THEN ABS(delta::bigint) ELSE 0 END), 0)::bigint AS aic_used
                FROM token_ledger
                WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
                """,
                u["id"], prev_start, prev_end,
            ) or {"put_used": 0, "aic_used": 0}
            engagement = await _compute_upload_engagement_totals(
                conn, str(u["id"]), since=prev_start, until=prev_end,
            )
            platform_rows = await conn.fetch(
                """
                SELECT lower(trim(p)) AS platform, COUNT(*)::int AS n
                  FROM uploads, unnest(COALESCE(platforms, ARRAY[]::varchar[])) AS p
                 WHERE user_id = $1
                   AND created_at >= $2 AND created_at < $3
                   AND length(trim(p)) > 0
                 GROUP BY 1
                 ORDER BY n DESC, 1 ASC
                 LIMIT 16
                """,
                u["id"], prev_start, prev_end,
            )
            wallet = await conn.fetchrow(
                "SELECT COALESCE(put_balance,0)::int AS put_balance, COALESCE(aic_balance,0)::int AS aic_balance FROM wallets WHERE user_id = $1",
                u["id"],
            ) or {"put_balance": 0, "aic_balance": 0}
            uploads = int((stats or {}).get("uploads") or 0)
            success_uploads = int((stats or {}).get("success_uploads") or 0)
            success_pct = int(round((success_uploads / max(uploads, 1)) * 100))
            put_ledger = int((ledger or {}).get("put_used") or 0)
            aic_ledger = int((ledger or {}).get("aic_used") or 0)
            put_u = int((stats or {}).get("put_from_uploads") or 0)
            aic_u = int((stats or {}).get("aic_from_uploads") or 0)
            put_used = max(put_ledger, put_u)
            aic_used = max(aic_ledger, aic_u)
            plat_break = (
                [(str(r["platform"]), f"{int(r['n']):,} uploads targeted") for r in platform_rows]
                if platform_rows
                else None
            )
            await send_monthly_user_kpi_digest_email(
                u["email"],
                u["name"] or "there",
                u.get("subscription_tier") or "free",
                period_label,
                uploads,
                success_pct,
                int(engagement.get("views") or 0),
                int(engagement.get("likes") or 0),
                put_used,
                aic_used,
                int(wallet.get("put_balance") or 0),
                int(wallet.get("aic_balance") or 0),
                comments=int(engagement.get("comments") or 0),
                shares=int(engagement.get("shares") or 0),
                platform_breakdown=plat_break,
            )
            await conn.execute(
                "UPDATE users SET monthly_digest_period = $2, updated_at = NOW() WHERE id = $1",
                u["id"], period_start.date(),
            )


def _admin_notifications_dict(settings: dict) -> dict:
    """settings_json.notifications — same shape as /api/admin/notification-settings."""
    n = settings.get("notifications")
    return dict(n) if isinstance(n, dict) else {}


async def _run_admin_weekly_kpi_digest_once():
    """Send one weekly KPI digest to admin/master_admin accounts."""
    now = _now_utc()
    week_key = now.strftime("%G-W%V")
    # Send on Mondays only.
    if now.weekday() != 0:
        return
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT settings_json FROM admin_settings WHERE id = 1")
        settings = {}
        if row and row.get("settings_json"):
            try:
                settings = dict(row["settings_json"])
            except Exception:
                settings = {}
        notif = _admin_notifications_dict(settings)
        # Match admin UI: toggles live under notifications.* (legacy root key fallback).
        weekly_on = notif.get("notify_weekly_report", settings.get("notify_weekly_report", True))
        if not weekly_on:
            return
        if settings.get("weekly_kpi_digest_week") == week_key:
            return

        since = now - timedelta(days=7)
        total_users = int(await conn.fetchval("SELECT COUNT(*) FROM users") or 0)
        new_users = int(await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at >= $1", since) or 0)
        paid_tiers = sorted(ADMIN_KPI_COUNTED_SUBSCRIPTION_TIERS)
        paid_users = int(
            await conn.fetchval(
                """
                SELECT COUNT(*) FROM users
                 WHERE status = 'active'
                   AND LOWER(COALESCE(subscription_tier, 'free')) = ANY($1::text[])
                   AND subscription_status IN ('active', 'trialing')
                """,
                paid_tiers,
            )
            or 0
        )
        trialing_paid = int(
            await conn.fetchval(
                """
                SELECT COUNT(*) FROM users
                 WHERE status = 'active'
                   AND subscription_status = 'trialing'
                   AND LOWER(COALESCE(subscription_tier, 'free')) = ANY($1::text[])
                """,
                paid_tiers,
            )
            or 0
        )
        uploads = int(await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE created_at >= $1", since) or 0)
        uploads_ok = int(
            await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM uploads
                 WHERE created_at >= $1
                   AND status IN {SUCCESSFUL_STATUS_SQL_IN}
                """,
                since,
            )
            or 0
        )
        upload_success_pct = int(round((uploads_ok / max(uploads, 1)) * 100))
        revenue = float(await conn.fetchval("SELECT COALESCE(SUM(amount),0)::decimal FROM revenue_tracking WHERE created_at >= $1", since) or 0.0)
        cost = float(await conn.fetchval("SELECT COALESCE(SUM(cost_usd),0)::decimal FROM cost_tracking WHERE created_at >= $1", since) or 0.0)
        margin_pct = ((revenue - cost) / revenue * 100.0) if revenue > 0 else 0.0
        platform_rollups_on = notif.get("weekly_digest_platform_kpi_rollups", True)
        platform_summary = None
        if platform_rollups_on:
            try:
                await _refresh_platform_kpi_rollups_for_utc_range(conn, since, now)
            except Exception as ex:
                logger.warning(f"platform_kpi_rollups refresh skipped: {ex}")
            plat_rows = await _fetch_platform_kpi_totals_between(conn, since, now)
            if plat_rows:
                lines = []
                for r in plat_rows[:8]:
                    p = str(r["platform"] or "unknown")
                    lines.append(
                        f"{p}: {int(r['uploads_targeted'] or 0):,} targeted / "
                        f"{int(r['uploads_completed'] or 0):,} completed — "
                        f"{int(r['views'] or 0):,} views, {int(r['likes'] or 0):,} likes"
                    )
                platform_summary = lines
        admins = await conn.fetch(
            "SELECT email, name FROM users WHERE role IN ('admin','master_admin') AND status='active' AND email_verified=TRUE"
        )
        week_label = f"{since.strftime('%b %d')} - {now.strftime('%b %d, %Y')}"
        for a in admins:
            await send_admin_weekly_kpi_digest_email(
                a["email"],
                a["name"] or "admin",
                week_label,
                total_users,
                new_users,
                paid_users,
                uploads,
                revenue,
                cost,
                margin_pct,
                upload_success_pct=upload_success_pct,
                trialing_paid_users=trialing_paid,
                platform_summary_lines=platform_summary,
            )
        settings["weekly_kpi_digest_week"] = week_key
        await conn.execute(
            "UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1",
            json.dumps(settings),
        )


async def _run_scheduled_publish_alerts_once():
    """Alert users when scheduled uploads are delayed or failed."""
    now = _now_utc()
    delayed_cutoff = now - timedelta(minutes=15)
    async with db_pool.acquire() as conn:
        delayed = await conn.fetch(
            """
            SELECT u.id, u.user_id, u.filename, u.status, u.scheduled_time, u.error_detail,
                   usr.email, usr.name,
                   COALESCE(up.email_notifications, TRUE) AS email_notifications,
                   COALESCE(up.scheduled_alert_emails, TRUE) AS scheduled_alert_emails
            FROM uploads u
            JOIN users usr ON usr.id = u.user_id
            LEFT JOIN user_preferences up ON up.user_id = u.user_id
            WHERE u.schedule_mode IN ('scheduled', 'smart')
              AND u.scheduled_time IS NOT NULL
              AND u.scheduled_time <= $1
              AND u.status IN ('pending','scheduled','queued','staged','ready_to_publish')
              AND u.schedule_warn_email_sent_at IS NULL
              AND usr.status = 'active'
              AND usr.email_verified = TRUE
              AND COALESCE(up.email_notifications, TRUE) = TRUE
              AND COALESCE(up.scheduled_alert_emails, TRUE) = TRUE
            """,
            delayed_cutoff,
        )
        for r in delayed:
            when = r["scheduled_time"].strftime("%B %d, %Y %H:%M UTC") if r.get("scheduled_time") else "scheduled time"
            await send_scheduled_publish_alert_email(
                r["email"],
                r["name"] or "there",
                r.get("filename") or "upload",
                when,
                r.get("status") or "pending",
                r.get("error_detail") or "",
                str(r["id"]),
            )
            await conn.execute("UPDATE uploads SET schedule_warn_email_sent_at = NOW() WHERE id = $1", r["id"])

        failed = await conn.fetch(
            """
            SELECT u.id, u.user_id, u.filename, u.status, u.scheduled_time, u.error_detail,
                   usr.email, usr.name,
                   COALESCE(up.email_notifications, TRUE) AS email_notifications,
                   COALESCE(up.scheduled_alert_emails, TRUE) AS scheduled_alert_emails
            FROM uploads u
            JOIN users usr ON usr.id = u.user_id
            LEFT JOIN user_preferences up ON up.user_id = u.user_id
            WHERE u.schedule_mode IN ('scheduled', 'smart')
              AND u.status = 'failed'
              AND u.schedule_fail_email_sent_at IS NULL
              AND usr.status = 'active'
              AND usr.email_verified = TRUE
              AND COALESCE(up.email_notifications, TRUE) = TRUE
              AND COALESCE(up.scheduled_alert_emails, TRUE) = TRUE
            """
        )
        for r in failed:
            when = r["scheduled_time"].strftime("%B %d, %Y %H:%M UTC") if r.get("scheduled_time") else "scheduled time"
            await send_scheduled_publish_alert_email(
                r["email"],
                r["name"] or "there",
                r.get("filename") or "upload",
                when,
                "failed",
                r.get("error_detail") or "",
                str(r["id"]),
            )
            await conn.execute("UPDATE uploads SET schedule_fail_email_sent_at = NOW() WHERE id = $1", r["id"])


async def _acquire_cron_lock(lock_name: str, ttl_seconds: int = 300) -> bool:
    """Try to acquire a distributed Redis lock for cron leadership.
    Returns True if this instance won the lock (should run the cron tick).
    Falls back to always-True when Redis is unavailable (single-instance mode)."""
    if not redis_client:
        return True
    try:
        acquired = await redis_client.set(f"cron_lock:{lock_name}", "1", nx=True, ex=ttl_seconds)
        return bool(acquired)
    except Exception:
        return True

async def _email_cron_loop():
    """Background loop for reminder/digest emails (leader-elected via Redis lock)."""
    try:
        while True:
            try:
                if await _acquire_cron_lock("email_cron", ttl_seconds=max(280, EMAIL_CRON_INTERVAL_SECONDS - 20)):
                    await _run_trial_ending_reminders_once()
                    await _run_monthly_user_kpi_digests_once()
                    await _run_admin_weekly_kpi_digest_once()
                    await _run_scheduled_publish_alerts_once()
            except Exception as e:
                logger.warning(f"email cron loop failed: {e}")
            await asyncio.sleep(max(300, EMAIL_CRON_INTERVAL_SECONDS))
    except asyncio.CancelledError:
        # Finish cleanly so lifespan await does not propagate cancellation to uvicorn.
        return

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
    """Return encrypted token envelope (dict). Persist to `platform_tokens.token_blob` as JSONB only —
    TEXT columns cause asyncpg to reject dict parameters (expected str, got dict)."""
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
    except Exception: return False

def create_access_jwt(user_id: str) -> str:
    now = _now_utc()
    return jwt.encode({"sub": user_id, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()), "iss": JWT_ISSUER, "aud": JWT_AUDIENCE}, JWT_SECRET, algorithm="HS256")

def _normalize_jwt_subject(raw_sub) -> Optional[str]:
    """
    Normalize JWT `sub` into a string user id.
    Defensively handles accidental object subjects so DB queries never receive dicts.
    """
    if isinstance(raw_sub, str):
        s = raw_sub.strip()
        return s or None

    data = raw_sub
    if isinstance(data, dict) and "kid" in data and "nonce" in data and "ciphertext" in data:
        try:
            data = decrypt_blob(data)
        except Exception:
            return None

    if isinstance(data, dict):
        for key in ("sub", "user_id", "id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None

def verify_access_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        subject = _normalize_jwt_subject(payload.get("sub"))
        if not subject:
            logger.warning("JWT verification failed: invalid subject type")
            return None
        return subject
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except (jwt.InvalidAudienceError, jwt.InvalidIssuerError) as e:
        logger.warning(f"JWT verification failed: {type(e).__name__}")
        return None
    except Exception as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

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
    await _discord_notify_service(webhook_url, content=content, embeds=embeds)

async def notify_signup(email: str, name: str):
    await _notify_signup_service(email, name, SIGNUP_DISCORD_WEBHOOK_URL, ADMIN_DISCORD_WEBHOOK_URL)

async def notify_mrr(amount: float, email: str, plan: str, event_type: str = "charge"):
    await _notify_mrr_service(amount, email, plan, event_type, MRR_DISCORD_WEBHOOK_URL, ADMIN_DISCORD_WEBHOOK_URL)

async def notify_topup(amount: float, email: str, wallet: str, tokens: int):
    await _notify_topup_service(amount, email, wallet, tokens, MRR_DISCORD_WEBHOOK_URL, ADMIN_DISCORD_WEBHOOK_URL)

async def notify_weekly_costs(openai_cost: float, storage_cost: float, compute_cost: float, revenue: float):
    await _notify_weekly_costs_service(openai_cost, storage_cost, compute_cost, revenue, ADMIN_DISCORD_WEBHOOK_URL)

async def send_signup_confirmation_email(email: str, name: str, token: str):
    """
    Send signup confirmation using the centralized stages.emails template set.
    This keeps all auth emails visually consistent and link-safe.
    """
    confirm_url = f"{FRONTEND_URL.rstrip('/')}/confirm-email.html?token={quote(token)}"
    await send_signup_confirmation_email_v2(email, name or "there", confirm_url)


# ============================================================
# R2 Storage
# ============================================================
def get_s3_client():
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4"), region_name="auto")



# --- R2 helpers (single source of truth: users.avatar_r2_key) ---

def r2_presign_get_url(r2_key: str, expires_in: int = 3600) -> str:

    """Generate a short-lived signed URL for a private R2 object."""
    return generate_presigned_download_url(r2_key, ttl=int(expires_in))


def _platform_account_avatar_to_url(stored: Optional[str]) -> str:
    """
    platform_tokens.account_avatar stores either a legacy HTTPS URL (external CDN)
    or an R2 object key under platform-avatars/... after OAuth import mirroring.
    """
    if not stored or not isinstance(stored, str):
        return ""
    s = stored.strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    try:
        return generate_presigned_download_url(s) or ""
    except Exception:
        return ""


async def _mirror_oauth_profile_image_to_r2(user_id: str, platform: str, source_url: str) -> Optional[str]:
    """
    Fetch a provider profile image (FB/IG/TikTok CDNs often 403 hotlinked from the browser)
    and store a private copy in R2. Returns the object key, or None if skipped/failed.
    """
    if not source_url or not str(source_url).startswith("http"):
        return None
    if not R2_BUCKET_NAME or not R2_ACCOUNT_ID:
        return None
    url = str(source_url).strip()
    max_bytes = 5 * 1024 * 1024
    ct_to_ext = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            r = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
        if r.status_code != 200:
            logger.debug(f"OAuth avatar fetch HTTP {r.status_code} for {platform} user={user_id}")
            return None
        body = r.content
        if not body or len(body) > max_bytes:
            return None
        ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        ext = ct_to_ext.get(ctype, "jpg")
        if ext == "jpg" and ctype and ctype not in ct_to_ext:
            # Unknown image type — still store as .jpg; browsers tolerate most blobs as jpeg label for small avatars
            ext = "jpg"
        key = f"platform-avatars/{user_id}/{platform}/{uuid.uuid4()}.{ext}"
        s3 = get_s3_client()
        put_ct = ctype if ctype.startswith("image/") else "image/jpeg"
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=_normalize_r2_key(key),
            Body=body,
            ContentType=put_ct,
        )
        logger.info(f"Mirrored OAuth avatar to R2 key={key[:64]}... platform={platform}")
        return key
    except Exception as e:
        logger.warning(f"OAuth avatar mirror failed ({platform}): {e}")
        return None

def generate_presigned_upload_url(key: str, content_type: str, ttl: int = None) -> str:
    ttl = int(ttl) if ttl is not None else R2_PRESIGN_UPLOAD_TTL
    key = _normalize_r2_key(key)
    s3 = get_s3_client()
    # Binding ContentType in the signature requires the browser to send the exact same header; mismatches → 403 → client often reports "network error".
    if R2_PRESIGN_PUT_UNSIGNED_CONTENT:
        params = {"Bucket": R2_BUCKET_NAME, "Key": key}
        logger.info(f"Presigned upload URL (unsigned Content-Type) for key={key[:80]}{'...' if len(key) > 80 else ''} ttl={ttl}s")
    else:
        params = {"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type}
        logger.info(f"Presigned upload URL generated for key={key[:80]}{'...' if len(key) > 80 else ''} ttl={ttl}s content_type={content_type}")
    url = s3.generate_presigned_url("put_object", Params=params, ExpiresIn=ttl)
    return url

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

class ResendConfirmationRequest(BaseModel):
    email: EmailStr

class UpdatePendingEmailRequest(BaseModel):
    current_email: EmailStr
    new_email: EmailStr


class UploadInit(BaseModel):
    filename: str
    file_size: int
    content_type: str
    platforms: List[str]
    target_accounts: List[str] = []  # platform_tokens.id UUIDs — publish to specific accounts
    title: str = ""
    caption: str = ""
    hashtags: List[str] = []
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None
    schedule_mode: str = "immediate"  # immediate | scheduled | smart
    has_telemetry: bool = False
    use_ai: bool = False
    smart_schedule_days: int = 7  # How many days to spread uploads across
    # Billing: optional client-reported duration (seconds) for AIC; else estimated from file_size
    duration_seconds: Optional[float] = None
    # If set, caps PUT + thumbnail_ai AIC; else derived from auto_thumbnails / use_ai
    thumbnail_count: Optional[int] = None
    # Thumbnail Studio renderer/persona per-upload overrides.
    thumbnail_use_studio_engine: Optional[bool] = None
    thumbnail_use_pikzels: Optional[bool] = None
    thumbnail_use_persona: Optional[bool] = None
    thumbnail_persona_id: Optional[str] = None
    thumbnail_persona_strength: Optional[int] = Field(default=None, ge=0, le=100)

class SettingsUpdate(BaseModel):
    discord_webhook: Optional[str] = Field(None, alias="discordWebhook")
    telemetry_enabled: Optional[bool] = Field(None, alias="telemetryEnabled")
    hud_enabled: Optional[bool] = Field(None, alias="hudEnabled")
    hud_position: Optional[str] = Field(None, alias="hudPosition")
    speeding_mph: Optional[int] = Field(None, alias="speedingMph")
    euphoria_mph: Optional[int] = Field(None, alias="euphoriaMph")
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

    class Config:
        populate_by_name = True

class CheckoutRequest(BaseModel):
    lookup_key: str
    kind: str = "subscription"  # subscription | topup | addon


class BillingSubscriptionActionRequest(BaseModel):
    action: Literal[
        "pause_payment_collection",
        "share_payment_update_link",
        "create_one_time_invoice",
        "cancel_subscription",
    ]
    amount_cents: Optional[int] = Field(default=None, ge=100, le=5000000)
    currency: str = "usd"
    description: Optional[str] = None


class UploadCostEstimateRequest(BaseModel):
    num_publish_targets: int = Field(default=1, ge=1, le=100)
    use_ai: bool = True
    use_hud: bool = False
    num_thumbnails: int = Field(default=1, ge=1, le=20)
    duration_seconds: Optional[float] = None
    file_size: Optional[int] = None
    has_telemetry: bool = False


class ThumbnailStudioEstimateRequest(BaseModel):
    variant_count: int = Field(default=4, ge=4, le=8)
    has_persona: bool = False
    competitor_gap_mode: bool = False
    has_channel_memory: bool = True


class ThumbnailPersonaCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    image_urls: List[str] = Field(min_length=3, max_length=20)
    expressions: List[str] = []
    lighting_presets: List[str] = []
    scene_prefs: List[str] = []


class ThumbnailRecreateRequest(BaseModel):
    youtube_url: str
    topic: Optional[str] = ""
    niche: str = "general"
    closeness: int = Field(default=55, ge=0, le=100)
    variant_count: int = Field(default=6, ge=4, le=8)
    persona_id: Optional[str] = None
    format_key: Optional[str] = None
    competitor_gap_mode: bool = False
    competitor_urls: List[str] = []


class ThumbnailFeedbackRequest(BaseModel):
    job_id: str
    variant_id: Optional[str] = None
    event_type: Literal["shown", "selected", "exported", "published_outcome"] = "shown"
    metadata: Dict[str, Any] = {}


class PikzelsV2PromptBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    model: str = "pkz_4"
    format: str = "16:9"
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None
    persona: Optional[str] = None
    style: Optional[str] = None


class PikzelsV2RecreateBody(BaseModel):
    """Pikzels v2 create-from-image (Recreate™)."""
    prompt: Optional[str] = Field(default=None, max_length=1000)
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None
    image_weight: Optional[str] = "medium"
    model: str = "pkz_4"
    format: str = "16:9"
    persona: Optional[str] = None
    style: Optional[str] = None


class PikzelsV2EditBody(BaseModel):
    """Pikzels v2 edit (Edit + One-Click Fix™ share this endpoint)."""
    prompt: str = Field(min_length=1, max_length=1000)
    format: str = "16:9"
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mask_url: Optional[str] = None
    mask_base64: Optional[str] = None
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None


class PikzelsV2FaceswapBody(BaseModel):
    format: str = "16:9"
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    face_image: Optional[str] = None
    face_image_base64: Optional[str] = None
    mask_url: Optional[str] = None
    mask_base64: Optional[str] = None


class PikzelsV2ScoreBody(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=200)


class PikzelsV2TitlesBody(BaseModel):
    prompt: Optional[str] = Field(default=None, max_length=2000)
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None


class PikzelsV2PikzonalityBody(BaseModel):
    name: str = Field(min_length=1, max_length=25)
    image_urls: Optional[List[str]] = None
    image_base64s: Optional[List[str]] = None


class PromoTogglesBody(BaseModel):
    """PATCH /api/admin/settings/promo-toggles — omit a field to leave it unchanged."""
    promo_burst_week_enabled: Optional[bool] = None
    promo_referral_enabled: Optional[bool] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

class ProfileUpdateSettings(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: Optional[str] = None
    avatar_r2_key: Optional[str] = None

class SettingsEmailChange(BaseModel):
    new_email: EmailStr
    current_password: str = Field(min_length=8)

class PreferencesUpdate(BaseModel):
    """Settings page / legacy prefs — includes Caption & AI card fields for save/load."""
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
    aiHashtagStyle: Optional[str] = None  # lowercase | capitalized | camelcase | mixed
    captionStyle: Optional[str] = None   # story | punchy | factual
    captionTone: Optional[str] = None    # hype | calm | cinematic | authentic
    captionVoice: Optional[str] = None   # default | mentor | hypebeast | best_friend | teacher | cinematic_narrator
    platformHashtags: Optional[dict] = None

class TransferRequest(BaseModel):
    from_platform: str
    to_platform: str
    amount: int

class MarketingEventIn(BaseModel):
    event_type: str = Field(..., pattern="^(shown|clicked|dismissed|converted)$")
    nudge_type: Optional[str] = Field(default="general", max_length=120)
    nudge_severity: Optional[str] = Field(default=None, max_length=32)
    cta_variant: Optional[str] = Field(default=None, max_length=8)
    urgency_variant: Optional[str] = Field(default=None, max_length=8)
    ordering_variant: Optional[str] = Field(default=None, max_length=8)
    page: Optional[str] = Field(default=None, max_length=255)
    session_id: Optional[str] = Field(default=None, max_length=120)
    metadata: Optional[Dict[str, Any]] = None

class MarketingCampaignIn(BaseModel):
    name: str = Field(..., min_length=3, max_length=160)
    objective: str = Field(..., min_length=3, max_length=400)
    channel: str = Field(..., pattern="^(in_app|email|discount|mixed)$")
    range: str = Field(default="30d", max_length=16)
    tiers: List[str] = []
    min_uploads_30d: int = Field(default=0, ge=0, le=10000)
    min_enterprise_fit_score: float = Field(default=0, ge=0, le=100)
    min_nudge_ctr_pct: float = Field(default=0, ge=0, le=100)
    require_no_revenue_7d: bool = False
    schedule_at: Optional[datetime] = None
    notes: Optional[str] = Field(default="", max_length=4000)

class MarketingCampaignStatusIn(BaseModel):
    status: str = Field(..., pattern="^(draft|scheduled|active|paused|completed|cancelled)$")

class MarketingAIGenerateIn(BaseModel):
    range: str = Field(default="30d", max_length=16)
    objective: str = Field(default="revenue_growth", max_length=120)
    tone: str = Field(default="executive_clear", max_length=80)
    offer_style: str = Field(default="value_first", max_length=80)
    channel_mix: str = Field(default="mixed", max_length=40)
    force_deploy: bool = False

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

class SmartScheduleOnlyUpdate(BaseModel):
    """PATCH /api/scheduled/{id} - only smart_schedule (platform -> ISO datetime string)."""
    smart_schedule: Dict[str, str] = Field(..., description="Platform -> ISO datetime string")

class UploadUpdate(BaseModel):
    """PATCH /api/uploads/{id} - title, caption, hashtags, scheduled_time, smart_schedule."""
    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    scheduled_time: Optional[datetime] = None
    smart_schedule: Optional[Dict[str, str]] = Field(None, description="Platform -> ISO datetime string")

class ColorPreferencesUpdate(BaseModel):
    tiktok_color: Optional[str] = None
    youtube_color: Optional[str] = None
    instagram_color: Optional[str] = None
    facebook_color: Optional[str] = None
    accent_color: Optional[str] = None

# App Lifespan & Migrations
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, admin_settings_cache, app_shutting_down
    app_shutting_down = False
    init_enc_keys()
    if STRIPE_SECRET_KEY: stripe.api_key = STRIPE_SECRET_KEY
    
    _db_min = int(os.environ.get("DB_POOL_MIN", "5"))
    _db_max = int(os.environ.get("DB_POOL_MAX", "20"))
    db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=_db_min,
        max_size=_db_max,
        command_timeout=30,
        init=_init_asyncpg_codecs,
    )
    await _load_uploads_columns(db_pool)
    logger.info("Database connected")
    
    await run_migrations()

    logger.info(
        "Rate limits: profile=%s window=%ss global=%s login=%s auth=%s admin=%s presign=%s "
        "localhost_bypass=%s enabled=%s redis_rl=%s",
        _RATE_LIMIT_CFG.get("profile"),
        _RATE_LIMIT_CFG.get("window_sec"),
        _RATE_LIMIT_CFG.get("global"),
        _RATE_LIMIT_CFG.get("login"),
        _RATE_LIMIT_CFG.get("auth"),
        _RATE_LIMIT_CFG.get("admin"),
        _RATE_LIMIT_CFG.get("presign"),
        _RL_LOCALHOST_BYPASS,
        _RATE_LIMIT_ENABLED,
        bool(REDIS_URL),
    )

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
                admin_settings_cache.setdefault("promo_burst_week_enabled", False)
                admin_settings_cache.setdefault("promo_referral_enabled", False)
                _n = admin_settings_cache.get("notifications")
                if isinstance(_n, dict):
                    _n.setdefault("weekly_digest_platform_kpi_rollups", True)
    except Exception as e:
        logger.debug("lifespan: admin_settings preload skipped: %s", e)

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
    
    email_cron_task = asyncio.create_task(_email_cron_loop())

    try:
        yield
    finally:
        app_shutting_down = True
        try:
            try:
                drain = float(os.environ.get("SHUTDOWN_DRAIN_SECONDS", "2"))
            except ValueError:
                drain = 2.0
            if drain > 0:
                try:
                    await asyncio.sleep(drain)
                except asyncio.CancelledError:
                    pass
            email_cron_task.cancel()
            # return_exceptions=True: avoid re-raising CancelledError from the child task
            # (some asyncio/uvicorn versions surface it to run_until_complete).
            _cron_results = await asyncio.gather(email_cron_task, return_exceptions=True)
            for _cr in _cron_results:
                if isinstance(_cr, BaseException) and not isinstance(_cr, asyncio.CancelledError):
                    logger.warning("email_cron_task finished with: %r", _cr)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("shutdown cleanup before pool close: %s", e)
        if db_pool:
            try:
                await db_pool.close()
            except Exception as e:
                logger.debug("db_pool.close during shutdown: %s", e)
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.debug("redis close during shutdown: %s", e)

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
            (205, """
                CREATE TABLE IF NOT EXISTS marketing_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    session_id VARCHAR(120),
                    event_type VARCHAR(32) NOT NULL,
                    nudge_type VARCHAR(120),
                    nudge_severity VARCHAR(32),
                    cta_variant VARCHAR(8),
                    urgency_variant VARCHAR(8),
                    ordering_variant VARCHAR(8),
                    page VARCHAR(255),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_marketing_events_created ON marketing_events(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_marketing_events_user_created ON marketing_events(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_marketing_events_type_created ON marketing_events(event_type, created_at DESC);
            """),
            (206, """
                CREATE TABLE IF NOT EXISTS marketing_campaigns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    name VARCHAR(160) NOT NULL,
                    objective VARCHAR(400) NOT NULL,
                    channel VARCHAR(24) NOT NULL,
                    status VARCHAR(24) NOT NULL DEFAULT 'draft',
                    range_key VARCHAR(16) NOT NULL DEFAULT '30d',
                    targeting JSONB NOT NULL DEFAULT '{}'::jsonb,
                    estimated_audience INT NOT NULL DEFAULT 0,
                    schedule_at TIMESTAMPTZ,
                    notes TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_marketing_campaigns_created ON marketing_campaigns(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_marketing_campaigns_status ON marketing_campaigns(status, schedule_at);
            """),
            (207, """
                CREATE TABLE IF NOT EXISTS ai_marketing_decisions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    action VARCHAR(32) NOT NULL, -- generate | deploy
                    objective VARCHAR(120),
                    range_key VARCHAR(16),
                    used_openai BOOLEAN DEFAULT FALSE,
                    deploy_allowed BOOLEAN,
                    forced BOOLEAN DEFAULT FALSE,
                    confidence_score INT,
                    status VARCHAR(24) NOT NULL, -- ok | blocked | error
                    blocked_reasons JSONB DEFAULT '[]'::jsonb,
                    snapshot JSONB DEFAULT '{}'::jsonb,
                    decision JSONB DEFAULT '{}'::jsonb,
                    plan JSONB DEFAULT '{}'::jsonb,
                    campaign_id UUID,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_ai_marketing_decisions_created ON ai_marketing_decisions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ai_marketing_decisions_action ON ai_marketing_decisions(action, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ai_marketing_decisions_campaign ON ai_marketing_decisions(campaign_id);
            """),
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
                styled_thumbnails BOOLEAN DEFAULT TRUE,
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
                use_audio_context BOOLEAN DEFAULT TRUE,
                audio_transcription BOOLEAN DEFAULT TRUE,
                ai_service_telemetry BOOLEAN DEFAULT TRUE,
                ai_service_audio_signals BOOLEAN DEFAULT TRUE,
                ai_service_music_detection BOOLEAN DEFAULT TRUE,
                ai_service_audio_summary BOOLEAN DEFAULT TRUE,
                ai_service_emotion_signals BOOLEAN DEFAULT TRUE,
                ai_service_caption_writer BOOLEAN DEFAULT TRUE,
                ai_service_thumbnail_designer BOOLEAN DEFAULT TRUE,
                ai_service_frame_inspector BOOLEAN DEFAULT TRUE,
                ai_service_speech_to_text BOOLEAN DEFAULT TRUE,
                ai_service_video_analyzer BOOLEAN DEFAULT TRUE,
                ai_service_scene_understanding BOOLEAN DEFAULT TRUE,
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
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS use_audio_context BOOLEAN DEFAULT TRUE;
                ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS audio_transcription BOOLEAN DEFAULT TRUE;
            """),
            (707, "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'"),
            (1030, "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS styled_thumbnails BOOLEAN DEFAULT TRUE"),
            (1031, "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'"),
            (1900, """
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
            (1901, """
                UPDATE users
                SET subscription_tier = 'creator_lite'
                WHERE LOWER(COALESCE(subscription_tier, '')) = 'launch';
            """),
            (1902, """
                ALTER TABLE wallets ADD COLUMN IF NOT EXISTS subscription_drip_month VARCHAR(7);
                ALTER TABLE wallets ADD COLUMN IF NOT EXISTS put_drip_granted INT DEFAULT 0;
                ALTER TABLE wallets ADD COLUMN IF NOT EXISTS aic_drip_granted INT DEFAULT 0;
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

# ── Per-account metrics events + daily rollups (scalable analytics backbone) ──
(606, """
    CREATE TABLE IF NOT EXISTS platform_account_metrics_events (
        id           BIGSERIAL PRIMARY KEY,
        fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_row_id UUID,
        platform     VARCHAR(50) NOT NULL,
        account_id   TEXT,
        metrics      JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    CREATE INDEX IF NOT EXISTS idx_pame_user_time ON platform_account_metrics_events(user_id, fetched_at DESC);
    CREATE INDEX IF NOT EXISTS idx_pame_plat_time ON platform_account_metrics_events(platform, fetched_at DESC);
"""),

(607, """
    CREATE TABLE IF NOT EXISTS platform_user_metrics_rollups_daily (
        user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        day            DATE NOT NULL,
        views          BIGINT NOT NULL DEFAULT 0,
        likes          BIGINT NOT NULL DEFAULT 0,
        comments       BIGINT NOT NULL DEFAULT 0,
        shares         BIGINT NOT NULL DEFAULT 0,
        platforms_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (user_id, day)
    );
    CREATE INDEX IF NOT EXISTS idx_pumrd_day ON platform_user_metrics_rollups_daily(day DESC);
"""),

# ── ML-ready feature/outcome datasets ─────────────────────────────────────────
(608, """
    CREATE TABLE IF NOT EXISTS upload_feature_events (
        id                    BIGSERIAL PRIMARY KEY,
        created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        user_id               UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        upload_id             UUID NOT NULL,
        category              TEXT,
        audio_context         JSONB,
        vision_context        JSONB,
        video_understanding   JSONB,
        thumbnail_brief       JSONB,
        output_artifacts      JSONB,
        ai_title              TEXT,
        ai_caption            TEXT,
        ai_hashtags           JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_ufe_user_time ON upload_feature_events(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_ufe_upload ON upload_feature_events(upload_id);
"""),

(609, """
    CREATE TABLE IF NOT EXISTS upload_quality_scores_daily (
        user_id            UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        day                DATE NOT NULL,
        platform           VARCHAR(50) NOT NULL DEFAULT 'all',
        strategy_key       TEXT NOT NULL DEFAULT 'default',
        samples            INT NOT NULL DEFAULT 0,
        mean_engagement    DOUBLE PRECISION NOT NULL DEFAULT 0,
        mean_views         DOUBLE PRECISION NOT NULL DEFAULT 0,
        engagement_stddev  DOUBLE PRECISION NOT NULL DEFAULT 0,
        ci95_low           DOUBLE PRECISION NOT NULL DEFAULT 0,
        ci95_high          DOUBLE PRECISION NOT NULL DEFAULT 0,
        updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (user_id, day, platform, strategy_key)
    );
    CREATE INDEX IF NOT EXISTS idx_uqsd_day ON upload_quality_scores_daily(day DESC);
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

# ── Country column for geo-analytics ─────────────────────────────────────────
(701, """
    ALTER TABLE users ADD COLUMN IF NOT EXISTS country VARCHAR(2);
    CREATE INDEX IF NOT EXISTS idx_users_country ON users(country) WHERE country IS NOT NULL;
"""),

(702, """
    -- password_resets was originally created for admin-forced resets only.
    -- The user-initiated forgot-password flow requires token_hash and used_at.
    -- Adding both as nullable so existing admin-reset rows are unaffected.
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS token_hash TEXT;
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS used_at TIMESTAMPTZ;
    -- Index for fast token lookup on reset-password endpoint
    CREATE INDEX IF NOT EXISTS idx_password_resets_token_hash ON password_resets(token_hash)
        WHERE token_hash IS NOT NULL;
    -- Index for fast invalidation of unused tokens per user
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

# ═══════════════════════════════════════════════════════════════
# ENTERPRISE SCALE MIGRATIONS (10K users)
# ═══════════════════════════════════════════════════════════════

(800, """
    -- Performance indexes for high-read queries
    CREATE INDEX IF NOT EXISTS idx_uploads_user_created     ON uploads(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_uploads_status_created   ON uploads(status, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_uploads_scheduled_time   ON uploads(scheduled_time)
        WHERE scheduled_time IS NOT NULL AND status IN ('staged', 'ready_to_publish');
    CREATE INDEX IF NOT EXISTS idx_uploads_user_completed   ON uploads(user_id, completed_at DESC)
        WHERE status IN ('completed', 'succeeded', 'partial');
    CREATE INDEX IF NOT EXISTS idx_users_email_lower        ON users(LOWER(email));
    CREATE INDEX IF NOT EXISTS idx_users_role               ON users(role) WHERE role != 'user';
    CREATE INDEX IF NOT EXISTS idx_users_tier               ON users(subscription_tier);
    CREATE INDEX IF NOT EXISTS idx_users_active             ON users(last_active_at DESC);
    CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user      ON refresh_tokens(user_id)
        WHERE revoked_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires   ON refresh_tokens(expires_at)
        WHERE revoked_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_ledger_user_type         ON token_ledger(user_id, token_type, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_cost_tracking_category   ON cost_tracking(category, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_revenue_tracking_created ON revenue_tracking(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_wallets_low_balance      ON wallets(user_id)
        WHERE put_balance <= 5 OR aic_balance <= 5;
"""),

(801, """
    -- Dead-letter queue for failed jobs that exhausted retries
    CREATE TABLE IF NOT EXISTS dead_letter_queue (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        upload_id       UUID REFERENCES uploads(id) ON DELETE SET NULL,
        user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
        job_data        JSONB NOT NULL,
        error_code      VARCHAR(100),
        error_message   TEXT,
        retry_count     INT DEFAULT 0,
        max_retries     INT DEFAULT 3,
        last_attempt_at TIMESTAMPTZ,
        resolved_at     TIMESTAMPTZ,
        resolved_by     VARCHAR(50),
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_dlq_unresolved ON dead_letter_queue(created_at)
        WHERE resolved_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_dlq_user ON dead_letter_queue(user_id);
"""),

(802, """
    -- API keys for enterprise integrations
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
    CREATE INDEX IF NOT EXISTS idx_api_keys_user    ON api_keys(user_id) WHERE revoked_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_api_keys_hash    ON api_keys(key_hash) WHERE revoked_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_api_keys_prefix  ON api_keys(key_prefix);
"""),

(803, """
    -- Upload processing metrics for SLA monitoring
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS put_cost INT DEFAULT 0;
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS aic_cost INT DEFAULT 0;
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS hold_status VARCHAR(20) DEFAULT 'none';

    -- Wallet holds (pending deductions until job finishes)
    CREATE TABLE IF NOT EXISTS wallet_holds (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        upload_id       UUID NOT NULL,
        user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        put_amount      INT DEFAULT 0,
        aic_amount      INT DEFAULT 0,
        status          VARCHAR(20) DEFAULT 'held',
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        resolved_at     TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_wallet_holds_user    ON wallet_holds(user_id) WHERE status = 'held';
    CREATE INDEX IF NOT EXISTS idx_wallet_holds_upload  ON wallet_holds(upload_id);
"""),

(804, """
    -- Caption memory for few-shot AI learning
    CREATE TABLE IF NOT EXISTS upload_caption_memory (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        upload_id       UUID,
        category        VARCHAR(50) DEFAULT 'general',
        platforms       JSONB DEFAULT '[]'::jsonb,
        ai_title        TEXT,
        ai_caption      TEXT,
        ai_hashtags     JSONB DEFAULT '[]'::jsonb,
        caption_voice   VARCHAR(50),
        caption_tone    VARCHAR(50),
        caption_style   VARCHAR(50),
        source          VARCHAR(20) DEFAULT 'auto',
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_caption_memory_user ON upload_caption_memory(user_id, category, created_at DESC);
"""),

(805, """
    -- Publish attempts ledger for delivery verification
    CREATE TABLE IF NOT EXISTS publish_attempts (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        upload_id           UUID NOT NULL,
        user_id             UUID NOT NULL,
        platform            VARCHAR(50) NOT NULL,
        status              VARCHAR(20) DEFAULT 'pending',
        platform_post_id    TEXT,
        platform_url        TEXT,
        publish_id          TEXT,
        http_status         INT,
        response_payload    JSONB,
        error_code          VARCHAR(100),
        error_message       TEXT,
        verify_status       VARCHAR(20),
        verified_at         TIMESTAMPTZ,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        updated_at          TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_publish_attempts_upload ON publish_attempts(upload_id);
    CREATE INDEX IF NOT EXISTS idx_publish_attempts_verify ON publish_attempts(status, verify_status)
        WHERE status = 'accepted' AND (verify_status IS NULL OR verify_status = 'pending');
"""),

(806, """
    -- Entitlement overrides (admin can grant features per-user)
    CREATE TABLE IF NOT EXISTS entitlement_overrides (
        user_id             UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        max_thumbnails      INT,
        max_caption_frames  INT,
        can_burn_hud        BOOLEAN,
        can_watermark       BOOLEAN,
        can_ai              BOOLEAN,
        can_schedule        BOOLEAN,
        put_monthly         INT,
        aic_monthly         INT,
        notes               TEXT,
        created_by          UUID,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        updated_at          TIMESTAMPTZ DEFAULT NOW()
    );
"""),

(807, """
    -- Email confirmation for signup (users must verify before full access)
    ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT TRUE;
    UPDATE users SET email_verified = TRUE WHERE email_verified IS NULL;
    CREATE TABLE IF NOT EXISTS email_confirmations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash VARCHAR(255) NOT NULL UNIQUE,
        expires_at TIMESTAMPTZ NOT NULL,
        used_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_email_confirmations_token ON email_confirmations(token_hash) WHERE used_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_email_confirmations_user ON email_confirmations(user_id) WHERE used_at IS NULL;
"""),

(808, """
    -- Self-serve forgot-password uses token_hash + used_at (migration 702). Some DBs differ:
    -- missing columns, or no temp_password_hash column at all — unconditional DROP NOT NULL fails.
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS token_hash TEXT;
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS used_at TIMESTAMPTZ;
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS temp_password_hash TEXT;
    DO $migration808$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'password_resets'
              AND column_name = 'temp_password_hash' AND is_nullable = 'NO'
        ) THEN
            ALTER TABLE password_resets ALTER COLUMN temp_password_hash DROP NOT NULL;
        END IF;
    END
    $migration808$;
    CREATE INDEX IF NOT EXISTS idx_password_resets_token_hash ON password_resets(token_hash)
        WHERE token_hash IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_password_resets_user_unused ON password_resets(user_id)
        WHERE used_at IS NULL;
"""),

(809, """
    -- OAuth reconnection time (distinct from token refresh / identity backfill updated_at)
    ALTER TABLE platform_tokens ADD COLUMN IF NOT EXISTS last_oauth_reconnect_at TIMESTAMPTZ;
"""),

(810, """
    -- Stripe invoice dedup + billing period log (wallet refills); calculator assumptions
    CREATE TABLE IF NOT EXISTS stripe_invoice_log (
        invoice_id    TEXT PRIMARY KEY,
        user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        tier_slug     TEXT,
        put_credited  INT DEFAULT 0,
        aic_credited  INT DEFAULT 0,
        period_start  TIMESTAMPTZ,
        period_end    TIMESTAMPTZ,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_stripe_invoice_log_user ON stripe_invoice_log(user_id);

    CREATE TABLE IF NOT EXISTS cost_model_config (
        id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        effective_date          DATE NOT NULL DEFAULT CURRENT_DATE,
        worker_costs            JSONB NOT NULL DEFAULT '{}'::jsonb,
        storage_costs           JSONB NOT NULL DEFAULT '{}'::jsonb,
        ai_cost_per_aic         NUMERIC(14, 6),
        ops_costs               JSONB NOT NULL DEFAULT '{}'::jsonb,
        utilization_target      NUMERIC(6, 4),
        lookahead_default_hours INT,
        prep_limit_default      INT,
        notes                   TEXT,
        updated_at              TIMESTAMPTZ DEFAULT NOW()
    );
"""),

(811, """
    -- platform_tokens.token_blob must be JSONB. Legacy TEXT rows may be invalid as a single JSON
    -- value (e.g. two objects concatenated) - plain ::jsonb then fails. Parse first balanced {...}
    -- or [...] or wrap as _legacy_parse for the rest.
    DO $m811$
    BEGIN
      IF EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'platform_tokens'
           AND column_name = 'token_blob'
           AND data_type IN ('text', 'character varying')
      ) THEN
        CREATE OR REPLACE FUNCTION uploadm8_token_blob_text_to_jsonb(t text)
        RETURNS jsonb
        LANGUAGE plpgsql
        IMMUTABLE
        AS $tokfn$
        DECLARE
          s text;
          len int;
          i int;
          depth int;
          start1 int;
          c text;
        BEGIN
          IF t IS NULL THEN RETURN NULL; END IF;
          s := trim(t);
          IF s = '' THEN RETURN NULL; END IF;
          BEGIN
            RETURN s::jsonb;
          EXCEPTION WHEN OTHERS THEN
            PERFORM 1;
          END;
          -- First balanced {...}
          depth := 0;
          start1 := NULL;
          len := length(s);
          FOR i IN 1..len LOOP
            c := substring(s FROM i FOR 1);
            IF c = '{' THEN
              IF depth = 0 THEN start1 := i; END IF;
              depth := depth + 1;
            ELSIF c = '}' THEN
              depth := depth - 1;
              IF depth = 0 AND start1 IS NOT NULL THEN
                BEGIN
                  RETURN substring(s FROM start1 FOR (i - start1 + 1))::jsonb;
                EXCEPTION WHEN OTHERS THEN
                  start1 := NULL;
                  depth := 0;
                  PERFORM 1;
                END;
              END IF;
            END IF;
          END LOOP;
          -- First balanced [...]
          depth := 0;
          start1 := NULL;
          FOR i IN 1..len LOOP
            c := substring(s FROM i FOR 1);
            IF c = '[' THEN
              IF depth = 0 THEN start1 := i; END IF;
              depth := depth + 1;
            ELSIF c = ']' THEN
              depth := depth - 1;
              IF depth = 0 AND start1 IS NOT NULL THEN
                BEGIN
                  RETURN substring(s FROM start1 FOR (i - start1 + 1))::jsonb;
                EXCEPTION WHEN OTHERS THEN
                  PERFORM 1;
                END;
              END IF;
            END IF;
          END LOOP;
          RETURN jsonb_build_object('_legacy_parse', to_jsonb(s));
        END;
        $tokfn$;

        ALTER TABLE platform_tokens
          ALTER COLUMN token_blob TYPE jsonb USING uploadm8_token_blob_text_to_jsonb(token_blob::text);

        DROP FUNCTION uploadm8_token_blob_text_to_jsonb(text);
      END IF;
    END
    $m811$;
"""),

(812, """
    -- Force password reset flag (admin reset-password); email change completion tracking
    ALTER TABLE users ADD COLUMN IF NOT EXISTS must_reset_password BOOLEAN DEFAULT FALSE;
    UPDATE users SET must_reset_password = FALSE WHERE must_reset_password IS NULL;
    ALTER TABLE email_changes ADD COLUMN IF NOT EXISTS used_at TIMESTAMPTZ;
"""),

(813, """
    -- Ensure admin email/password reset columns exist on older databases
    ALTER TABLE email_changes ADD COLUMN IF NOT EXISTS changed_by_admin_id UUID REFERENCES users(id) ON DELETE SET NULL;
    ALTER TABLE email_changes ADD COLUMN IF NOT EXISTS verification_token TEXT;
    ALTER TABLE email_changes ADD COLUMN IF NOT EXISTS used_at TIMESTAMPTZ;

    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS reset_by_admin_id UUID REFERENCES users(id) ON DELETE SET NULL;
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS temp_password_hash TEXT;
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS force_change BOOLEAN DEFAULT TRUE;
    ALTER TABLE password_resets ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
"""),

(814, """
    -- Email reminder/digest cursors
    ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_reminder_sent_at TIMESTAMPTZ;
    ALTER TABLE users ADD COLUMN IF NOT EXISTS monthly_digest_period DATE;
"""),

(815, """
    -- Async analytics export jobs (secure one-time download link)
    CREATE TABLE IF NOT EXISTS export_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        token_hash TEXT UNIQUE NOT NULL,
        report_type VARCHAR(50) NOT NULL DEFAULT 'analytics',
        format VARCHAR(20) NOT NULL DEFAULT 'csv',
        days INT NOT NULL DEFAULT 30,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        file_blob BYTEA,
        content_type TEXT,
        filename TEXT,
        expires_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        ready_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_export_jobs_user_created ON export_jobs(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_export_jobs_expires ON export_jobs(expires_at);
"""),

(816, """
    -- Scheduled publish alert email flags
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS schedule_warn_email_sent_at TIMESTAMPTZ;
    ALTER TABLE uploads ADD COLUMN IF NOT EXISTS schedule_fail_email_sent_at TIMESTAMPTZ;
    CREATE INDEX IF NOT EXISTS idx_uploads_sched_warn_email ON uploads(schedule_warn_email_sent_at);
    CREATE INDEX IF NOT EXISTS idx_uploads_sched_fail_email ON uploads(schedule_fail_email_sent_at);
"""),

(817, """
    -- Login anomaly fingerprint fields
    ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_ip TEXT;
    ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_country VARCHAR(2);
    ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_user_agent TEXT;
"""),

(818, """
    -- Email category toggles
    ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS auth_security_alerts BOOLEAN DEFAULT TRUE;
    ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS digest_emails BOOLEAN DEFAULT TRUE;
    ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS scheduled_alert_emails BOOLEAN DEFAULT TRUE;
"""),

# ── Unified content catalog — one canonical row per real video ────────────
# Covers UploadM8-published videos AND externally discovered videos from each
# connected account.  Keyed by (user_id, platform, account_id, platform_video_id)
# so multi-account users never collide and we never store duplicates.
(819, """
    CREATE TABLE IF NOT EXISTS platform_content_items (
        id                 UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id            UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        platform_token_id  UUID         REFERENCES platform_tokens(id) ON DELETE SET NULL,
        platform           VARCHAR(32)  NOT NULL,
        account_id         VARCHAR(255) NOT NULL DEFAULT '',
        platform_video_id  VARCHAR(255) NOT NULL,
        upload_id          UUID         REFERENCES uploads(id) ON DELETE SET NULL,
        -- source: 'external' = discovered by catalog scan only
        --         'uploadm8' = published through the pipeline
        --         'linked'   = external catalog row later matched to an upload
        source             VARCHAR(24)  NOT NULL DEFAULT 'external',
        content_kind       VARCHAR(32),           -- short|long|reel|story|unknown
        title              TEXT,
        published_at       TIMESTAMPTZ,
        thumbnail_url      TEXT,
        platform_url       TEXT,
        duration_seconds   INT,
        views              BIGINT       NOT NULL DEFAULT 0,
        likes              BIGINT       NOT NULL DEFAULT 0,
        comments           BIGINT       NOT NULL DEFAULT 0,
        shares             BIGINT       NOT NULL DEFAULT 0,
        metrics_synced_at  TIMESTAMPTZ,
        extra              JSONB        NOT NULL DEFAULT '{}'::jsonb,
        created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        updated_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        UNIQUE (user_id, platform, account_id, platform_video_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pci_user_platform  ON platform_content_items(user_id, platform);
    CREATE INDEX IF NOT EXISTS idx_pci_user_views     ON platform_content_items(user_id, views DESC);
    CREATE INDEX IF NOT EXISTS idx_pci_user_published ON platform_content_items(user_id, published_at DESC NULLS LAST);
    CREATE INDEX IF NOT EXISTS idx_pci_upload         ON platform_content_items(upload_id) WHERE upload_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_pci_source         ON platform_content_items(user_id, source);
    CREATE INDEX IF NOT EXISTS idx_pci_synced         ON platform_content_items(metrics_synced_at NULLS FIRST);
"""),

(820, """
    -- Tracks per-token catalog sync state: pagination cursor + last run info.
    CREATE TABLE IF NOT EXISTS platform_content_sync_state (
        id                 UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id            UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        platform_token_id  UUID         NOT NULL REFERENCES platform_tokens(id) ON DELETE CASCADE,
        platform           VARCHAR(32)  NOT NULL,
        account_id         VARCHAR(255) NOT NULL DEFAULT '',
        last_synced_at     TIMESTAMPTZ,
        next_cursor        TEXT,             -- opaque cursor / page token for next incremental page
        total_discovered   INT          NOT NULL DEFAULT 0,
        total_linked       INT          NOT NULL DEFAULT 0,
        status             VARCHAR(24)  NOT NULL DEFAULT 'idle',  -- idle|syncing|done|error
        error_detail       TEXT,
        updated_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        UNIQUE (user_id, platform_token_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pcss_user ON platform_content_sync_state(user_id);
"""),

(821, """
    -- Global per-platform KPI rollups (UTC day) for admin digests & analytics; refreshed from uploads + platform_results.
    CREATE TABLE IF NOT EXISTS platform_kpi_rollups_daily (
        day                 DATE         NOT NULL,
        platform            VARCHAR(64)  NOT NULL,
        uploads_targeted    BIGINT       NOT NULL DEFAULT 0,
        uploads_completed   BIGINT       NOT NULL DEFAULT 0,
        views               BIGINT       NOT NULL DEFAULT 0,
        likes               BIGINT       NOT NULL DEFAULT 0,
        comments            BIGINT       NOT NULL DEFAULT 0,
        shares              BIGINT       NOT NULL DEFAULT 0,
        updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        PRIMARY KEY (day, platform)
    );
    CREATE INDEX IF NOT EXISTS idx_pkrd_day ON platform_kpi_rollups_daily(day DESC);
    CREATE INDEX IF NOT EXISTS idx_pkrd_platform ON platform_kpi_rollups_daily(platform);
"""),

(822, """
    -- Repair schema drift: admin_audit_log must have user_id (target of admin action).
    -- CREATE TABLE IF NOT EXISTS in v104 skips if a partial/legacy table already existed.
    ALTER TABLE admin_audit_log ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE;
    CREATE INDEX IF NOT EXISTS idx_admin_audit_log_user ON admin_audit_log(user_id);
"""),

(823, """
    CREATE TABLE IF NOT EXISTS thumbnail_format_library (
        key                 TEXT PRIMARY KEY,
        niche               TEXT NOT NULL,
        name                TEXT NOT NULL,
        pattern             TEXT NOT NULL,
        social_proof        TEXT DEFAULT '',
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS creator_personas (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name                TEXT NOT NULL,
        profile_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
        image_count         INT NOT NULL DEFAULT 0,
        quality_score       DOUBLE PRECISION NOT NULL DEFAULT 0,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_creator_personas_user ON creator_personas(user_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS creator_persona_images (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        persona_id          UUID NOT NULL REFERENCES creator_personas(id) ON DELETE CASCADE,
        user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        image_url           TEXT NOT NULL,
        quality_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_persona_images_persona ON creator_persona_images(persona_id, created_at ASC);

    CREATE TABLE IF NOT EXISTS thumbnail_recreate_jobs (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        youtube_url         TEXT NOT NULL,
        youtube_video_id    TEXT,
        source_title        TEXT,
        topic               TEXT,
        niche               TEXT NOT NULL DEFAULT 'general',
        closeness           INT NOT NULL DEFAULT 50,
        variant_count       INT NOT NULL DEFAULT 4,
        persona_id          UUID REFERENCES creator_personas(id) ON DELETE SET NULL,
        competitor_gap_mode BOOLEAN NOT NULL DEFAULT FALSE,
        put_cost            INT NOT NULL DEFAULT 0,
        aic_cost            INT NOT NULL DEFAULT 0,
        breakdown_json      JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_thumb_recreate_jobs_user ON thumbnail_recreate_jobs(user_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS thumbnail_recreate_variants (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id              UUID NOT NULL REFERENCES thumbnail_recreate_jobs(id) ON DELETE CASCADE,
        user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        rank_idx            INT NOT NULL DEFAULT 1,
        variant_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
        selected            BOOLEAN NOT NULL DEFAULT FALSE,
        publish_outcome     JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_thumb_recreate_variants_job ON thumbnail_recreate_variants(job_id, rank_idx);

    CREATE TABLE IF NOT EXISTS thumbnail_recreate_feedback (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        job_id              UUID NOT NULL REFERENCES thumbnail_recreate_jobs(id) ON DELETE CASCADE,
        variant_id          UUID REFERENCES thumbnail_recreate_variants(id) ON DELETE SET NULL,
        event_type          TEXT NOT NULL,
        metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_thumb_feedback_user ON thumbnail_recreate_feedback(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_thumb_feedback_job ON thumbnail_recreate_feedback(job_id, created_at DESC);
"""),

(824, """
    CREATE TABLE IF NOT EXISTS stripe_webhook_events (
        id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL DEFAULT '',
        received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_stripe_webhook_events_received ON stripe_webhook_events(received_at DESC);
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
                    raise

app = FastAPI(title="UploadM8 API", version="4.0.0", lifespan=lifespan)
# NOTE: Do NOT add CORSMiddleware here. FastAPI prepends each add_middleware();
# registering CORS first puts it *inside* later middlewares. The rate-limit layer
# would then see OPTIONS preflights before CORS short-circuits them → 429 breaks CORS.
# CORS is registered once after all @app.middleware hooks (see below).

# ── CORS-safe exception handler ──────────────────────────────────────────────
# FastAPI's CORSMiddleware does NOT add Access-Control-Allow-Origin to 500
# responses when an unhandled exception propagates — the browser then reports
# a CORS error instead of the real HTTP 500.  This handler catches every
# unhandled exception and returns a proper JSON 500 with CORS headers so the
# browser (and developer tools) always see the real error.
@app.exception_handler(Exception)
async def _cors_safe_500_handler(request: Request, exc: Exception) -> JSONResponse:
    cors_origin = _cors_reflect_origin(request)

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

async def rate_limit_allowed(key: str, limit: int, window_sec: int) -> bool:
    """
    Fixed-window rate limit. Prefers Redis (distributed) when configured,
    falls back to in-memory buckets for single-instance dev.
    Set limit <= 0 to disable that bucket (always allow).
    """
    if limit <= 0:
        return True
    if redis_client is not None:
        try:
            count = await redis_client.incr(key)
            if count == 1:
                await redis_client.expire(key, window_sec)
            return int(count) <= int(limit)
        except Exception as e:
            logger.warning(f"Redis rate limit failed, falling back to memory: {e}")

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
    if TRUST_PROXY_HEADERS:
        for hdr in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
            val = (req.headers.get(hdr) or "").strip()
            if val:
                try:
                    return str(ip_address(val))
                except Exception as e:
                    logger.debug("client_ip: ignoring invalid trusted header %s=%r: %s", hdr, val, e)

        xff = (req.headers.get("x-forwarded-for") or "").strip()
        if xff:
            candidate = xff.split(",")[0].strip()
            try:
                return str(ip_address(candidate))
            except Exception as e:
                logger.debug("client_ip: ignoring invalid x-forwarded-for first hop %r: %s", candidate, e)

    return (req.client.host if req.client else "unknown")


def _client_is_loopback(ip: str) -> bool:
    """True for 127.0.0.1, ::1, ::ffff:127.0.0.1, etc. (Windows/ASGI may use mapped forms)."""
    if ip in ("127.0.0.1", "::1", "localhost"):
        return True
    try:
        return bool(ip_address(ip).is_loopback)
    except Exception:
        return False


def _parse_int_env(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _load_rate_limit_config() -> Dict[str, Any]:
    """
    Dynamic HTTP rate limits (per IP / per bucket).

    RATE_LIMIT_PROFILE: strict | standard | relaxed | enterprise — baseline presets.
    Override any bucket with RL_*_LIMIT or RL_WINDOW_SEC (seconds per window).

    Set RL_*_LIMIT=0 to disable that bucket only. RATE_LIMIT_ENABLED=false disables middleware.
    """
    profile = os.environ.get("RATE_LIMIT_PROFILE", "standard").strip().lower()
    presets: Dict[str, tuple] = {
        "strict": (150, 8, 25, 40, 20),
        "standard": (300, 15, 40, 60, 30),
        "relaxed": (2000, 60, 200, 300, 120),
        "enterprise": (5000, 120, 500, 800, 400),
    }
    g, lg, au, ad, pr = presets.get(profile, presets["standard"])
    win = _parse_int_env("RL_WINDOW_SEC", 60)
    return {
        "profile": profile,
        "window_sec": win if win > 0 else 60,
        "global": _parse_int_env("RL_GLOBAL_LIMIT", g),
        "login": _parse_int_env("RL_LOGIN_LIMIT", lg),
        "auth": _parse_int_env("RL_AUTH_LIMIT", au),
        "admin": _parse_int_env("RL_ADMIN_LIMIT", ad),
        "presign": _parse_int_env("RL_PRESIGN_LIMIT", pr),
    }


_RATE_LIMIT_CFG: Dict[str, Any] = _load_rate_limit_config()
_RL_LOCALHOST_BYPASS = os.environ.get("RL_LOCALHOST_BYPASS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_RL_BYPASS_SECRET = os.environ.get("E2E_RATE_LIMIT_BYPASS_SECRET", "").strip()
_RL_BYPASS_HEADER = (
    os.environ.get("E2E_RATE_LIMIT_BYPASS_HEADER", "X-UploadM8-RL-Bypass").strip()
    or "X-UploadM8-RL-Bypass"
)


def _parse_trusted_rl_ips() -> frozenset:
    raw = os.environ.get("RATE_LIMIT_TRUSTED_IPS", "").strip()
    if not raw:
        return frozenset()
    out = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(str(ip_address(x)))
        except Exception:
            out.append(x)
    return frozenset(out)


_RL_TRUSTED_IPS: frozenset = _parse_trusted_rl_ips()


def _rate_limit_request_bypass(request: Request) -> bool:
    """E2E/CI secret header, or optional trusted egress IPs (e.g. fixed runner)."""
    if _RL_BYPASS_SECRET and (request.headers.get(_RL_BYPASS_HEADER) or "").strip() == _RL_BYPASS_SECRET:
        return True
    if _RL_TRUSTED_IPS:
        ip = client_ip(request)
        if ip in _RL_TRUSTED_IPS:
            return True
    return False


def _json_429(detail: str) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": detail})

_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")

def install_rate_limit_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def rl_middleware(request: Request, call_next):
        if not _RATE_LIMIT_ENABLED:
            return await call_next(request)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        # Probes must not depend on Redis rate-limit or client_ip heuristics (proxy headers,
        # ::ffff:127.0.0.1, etc.). Otherwise a stuck INCR or mis-identified IP can hang /health.
        _path = request.url.path or ""
        if _path in ("/health", "/ready"):
            return await call_next(request)

        ip = client_ip(request)

        if _rate_limit_request_bypass(request):
            return await call_next(request)

        if _RL_LOCALHOST_BYPASS and _client_is_loopback(ip):
            return await call_next(request)

        cfg = _RATE_LIMIT_CFG
        w = int(cfg["window_sec"])

        path = request.url.path

        if not await rate_limit_allowed(f"ip:{ip}:global", limit=int(cfg["global"]), window_sec=w):
            return _json_429("Rate limit exceeded (global)")

        if path.startswith("/api/auth/"):
            if path in ("/api/auth/login", "/api/auth/refresh"):
                if not await rate_limit_allowed(f"ip:{ip}:login", limit=int(cfg["login"]), window_sec=w):
                    return _json_429("Rate limit exceeded (login)")
            elif not await rate_limit_allowed(f"ip:{ip}:auth", limit=int(cfg["auth"]), window_sec=w):
                return _json_429("Rate limit exceeded (auth)")
        elif path.startswith("/api/admin/"):
            if not await rate_limit_allowed(f"ip:{ip}:admin", limit=int(cfg["admin"]), window_sec=w):
                return _json_429("Rate limit exceeded (admin)")
        elif path.startswith("/api/presign"):
            if not await rate_limit_allowed(f"ip:{ip}:presign", limit=int(cfg["presign"]), window_sec=w):
                return _json_429("Rate limit exceeded (uploads)")

        response = await call_next(request)
        return response

install_rate_limit_middleware(app)


# ============================================================
# AUDIT LOGGING HELPER
# ============================================================
async def audit_log(
    user_id: str,
    action: str,
    *,
    event_category: str = "SYSTEM",
    resource_type: str = None,
    resource_id: str = None,
    details: dict = None,
    ip_address: str = None,
    user_agent: str = None,
    severity: str = "INFO",
    outcome: str = "SUCCESS",
):
    """Write to system_event_log for full audit trail. Non-blocking, non-fatal."""
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO system_event_log
                       (user_id, event_category, action, resource_type, resource_id,
                        details, ip_address, user_agent, severity, outcome)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10)""",
                user_id, event_category, action, resource_type, resource_id,
                json.dumps(details or {}), ip_address, user_agent, severity, outcome,
            )
    except Exception as e:
        logger.debug(f"audit_log write failed (non-fatal): {e}")


# ============================================================
# REDIS CACHE HELPERS
# ============================================================
CACHE_TTL_SHORT = int(os.environ.get("CACHE_TTL_SHORT", "60"))
CACHE_TTL_MEDIUM = int(os.environ.get("CACHE_TTL_MEDIUM", "300"))
CACHE_TTL_LONG = int(os.environ.get("CACHE_TTL_LONG", "3600"))
# Redis JSON cache for GET /api/me (Bearer JWT). Skips platform-token aggregation + heavy payload rebuild on hits.
# Set to 0 to disable. Invalidated on profile/password/avatar updates.
ME_API_CACHE_TTL_SEC = int(os.environ.get("ME_API_CACHE_TTL_SEC", "30"))

async def cache_get(key: str):
    """Get from Redis cache. Returns None on miss or if Redis unavailable."""
    if not redis_client:
        return None
    try:
        val = await redis_client.get(f"cache:{key}")
        return json.loads(val) if val else None
    except Exception as e:
        logger.debug("cache_get miss/failure key=%s: %s", key, e)
        return None

async def cache_set(key: str, value, ttl: int = CACHE_TTL_SHORT):
    """Set in Redis cache. Non-fatal on failure."""
    if not redis_client:
        return
    try:
        await redis_client.setex(f"cache:{key}", ttl, json.dumps(value, default=str))
    except Exception as e:
        logger.debug("cache_set failure key=%s: %s", key, e)

async def cache_delete(key: str):
    """Delete from Redis cache."""
    if not redis_client:
        return
    try:
        await redis_client.delete(f"cache:{key}")
    except Exception as e:
        logger.debug("cache_delete failure key=%s: %s", key, e)

async def cache_delete_pattern(pattern: str):
    """Delete all keys matching pattern from Redis cache."""
    if not redis_client:
        return
    try:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"cache:{pattern}", count=100)
            if keys:
                await redis_client.delete(*keys)
            if cursor == 0:
                break
    except Exception as e:
        logger.debug("cache_delete_pattern failure pattern=%r: %s", pattern, e)


async def invalidate_me_api_cache(user_id: Optional[str]) -> None:
    """Bust GET /api/me Redis payload for this user (call after profile/tier/role-affecting writes)."""
    if not user_id or not redis_client:
        return
    try:
        await cache_delete(f"api_me:{user_id}")
    except Exception as e:
        logger.debug("invalidate_me_api_cache failure user_id=%s: %s", user_id, e)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or _req_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


import contextvars as _contextvars
_trace_id_var: _contextvars.ContextVar[str] = _contextvars.ContextVar("trace_id", default="")

class _TraceFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = _trace_id_var.get("")
        return True

for _h in logging.root.handlers:
    _h.addFilter(_TraceFilter())

@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Request-ID") or secrets.token_hex(8)
    _trace_id_var.set(trace_id)
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = trace_id
    return resp

@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # CSP is set here (server-wide) so ALL pages get font-src for Font Awesome (cdnjs) and
    # Google Fonts (fonts.gstatic.com). Without font-src, browsers fall back to default-src
    # 'self' and block every CDN webfont, making all icons render as blank squares.
    # Pages that also carry <meta http-equiv="Content-Security-Policy"> (e.g. dashboard.html)
    # add further per-page restrictions on top of this header — both must permit a resource.
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com data:; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://static.cloudflareinsights.com; "
        "connect-src 'self' http: https:; "
        "frame-ancestors 'none';"
    )
    return resp


@app.middleware("http")
async def shutdown_guard_middleware(request: Request, call_next):
    """Reject new HTTP work early while lifespan is closing the DB pool (outermost — registered last)."""
    if app_shutting_down and request.url.path not in ("/health",):
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is shutting down"},
            headers={"Retry-After": "5"},
        )
    return await call_next(request)


@app.middleware("http")
async def api_v1_path_alias_middleware(request: Request, call_next):
    """
    /api/v1/<path> uses the same handlers as /api/<path>. Responses on API routes get X-API-Version: 1.
    GET /api/v1 (no trailing path) is a small contract index — not rewritten.
    """
    orig_path = request.scope.get("path") or ""
    if orig_path.startswith("/api/v1/"):
        suffix = orig_path[len("/api/v1/") :]
        new_path = "/api/" + suffix
        request.scope["path"] = new_path
        if "raw_path" in request.scope:
            try:
                request.scope["raw_path"] = new_path.encode("latin-1")
            except Exception:
                pass
    response = await call_next(request)
    if orig_path.startswith("/api/") or orig_path.startswith("/api/v1"):
        response.headers.setdefault("X-API-Version", "1")
    return response


# CORS must be registered *after* other add_middleware / @app.middleware hooks so it sits
# *outside* rate limiting on the request path. Browser OPTIONS preflights are answered
# here without consuming auth/login rate-limit buckets (fixes 429 + bogus CORS errors).
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_origin_regex=_CORS_LOCAL_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1", tags=["meta"])
async def api_v1_contract():
    """Stable entry for external integrators; OpenAPI for all routes remains at /openapi.json and /docs."""
    return {
        "api_version": "1",
        "openapi_json": "/openapi.json",
        "docs": "/docs",
        "alias": "Every /api/v1/... request is handled by the same route as /api/...",
        "examples": [
            "/api/v1/auth/login",
            "/api/v1/me",
            "/api/v1/uploads/presign",
            "/api/v1/billing/checkout",
        ],
    }


# ============================================================
# Auth Dependencies
# ============================================================
def _pool_temporarily_unavailable(exc: BaseException) -> bool:
    """True when asyncpg pool is closed or mid-close (common during uvicorn shutdown)."""
    if not isinstance(exc, asyncpg.exceptions.InterfaceError):
        return False
    msg = str(exc).lower()
    return "pool is closed" in msg or "pool is closing" in msg


async def _auth_via_api_key(api_key: str) -> Optional[dict]:
    """Authenticate via API key (um8_... prefix). Returns user dict or None."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT ak.user_id, ak.scopes, ak.rate_limit, ak.expires_at
                   FROM api_keys ak
                   WHERE ak.key_hash = $1 AND ak.revoked_at IS NULL""",
                key_hash,
            )
            if not row:
                return None
            if row["expires_at"] and row["expires_at"] < _now_utc():
                return None
            await conn.execute(
                "UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = $1",
                key_hash,
            )
            user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", row["user_id"])
            if not user or user["status"] == "banned":
                return None
            wallet = await get_wallet(conn, row["user_id"])
            result = {**dict(user), "wallet": wallet, "_api_key": True, "_scopes": list(row["scopes"] or [])}
            return result
    except asyncpg.exceptions.InterfaceError as e:
        if _pool_temporarily_unavailable(e):
            raise HTTPException(503, "Server is shutting down")
        raise
    except Exception:
        return None


def _last_active_touch_interval_sec() -> int:
    """Min seconds between last_active_at bumps; 0 = update every request."""
    try:
        return max(0, int(os.environ.get("LAST_ACTIVE_TOUCH_MIN_INTERVAL_SEC", "120")))
    except ValueError:
        return 120


def _should_touch_last_active(last_active_at: Any, interval_sec: int) -> bool:
    if interval_sec <= 0:
        return True
    if last_active_at is None:
        return True
    la = last_active_at
    if getattr(la, "tzinfo", None) is None:
        la = la.replace(tzinfo=timezone.utc)
    return (_now_utc() - la).total_seconds() >= float(interval_sec)


async def _load_user_session_from_db(
    user_id: str,
    *,
    run_daily_refill: bool = True,
    touch_last_active: bool = True,
) -> dict:
    """Load full user row + wallet; optional last_active bump and free-tier daily_refill."""
    try:
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            if not user:
                raise HTTPException(401, "User not found")
            if user["status"] == "banned":
                raise HTTPException(403, "Account suspended")
            if touch_last_active and _should_touch_last_active(
                user.get("last_active_at"), _last_active_touch_interval_sec()
            ):
                await conn.execute("UPDATE users SET last_active_at = NOW() WHERE id = $1", user_id)
            if run_daily_refill:
                await daily_refill(conn, user_id, user["subscription_tier"])
            wallet = await get_wallet(conn, user_id)
            return {**dict(user), "wallet": wallet}
    except asyncpg.exceptions.InterfaceError as e:
        if _pool_temporarily_unavailable(e):
            raise HTTPException(503, "Server is shutting down")
        raise


async def _resolve_current_user_bearer(
    authorization: Optional[str],
    *,
    run_daily_refill: bool = True,
    touch_last_active: bool = True,
) -> dict:
    if not authorization:
        raise HTTPException(401, "Missing authorization")

    if authorization.startswith("um8_"):
        user = await _auth_via_api_key(authorization)
        if not user:
            raise HTTPException(401, "Invalid API key")
        return user

    auth_token = authorization[7:] if authorization.startswith("Bearer ") else None
    if not auth_token:
        raise HTTPException(401, "Missing authorization")
    user_id = verify_access_jwt(auth_token)
    if not user_id:
        raise HTTPException(401, "Invalid token")

    return await _load_user_session_from_db(
        user_id,
        run_daily_refill=run_daily_refill,
        touch_last_active=touch_last_active,
    )


async def get_current_user(request: Request, authorization: Optional[str] = Header(None)):
    return await _resolve_current_user_bearer(authorization)


async def get_current_user_readonly(request: Request, authorization: Optional[str] = Header(None)):
    """Same as get_current_user but skips free-tier daily_refill (read-heavy / polling routes)."""
    return await _resolve_current_user_bearer(
        authorization,
        run_daily_refill=False,
        touch_last_active=True,
    )

async def require_admin(user: dict = Depends(get_current_user)):
    role = str(user.get("role") or "").lower()
    tier = str(user.get("subscription_tier") or "").lower()
    if role not in ("admin", "master_admin") and tier != "master_admin":
        raise HTTPException(403, "Admin required")
    return user

async def require_master_admin(user: dict = Depends(get_current_user)):
    role = str(user.get("role") or "").lower()
    tier = str(user.get("subscription_tier") or "").lower()
    if role != "master_admin" and tier != "master_admin":
        raise HTTPException(403, "Master admin required")
    return user


@app.get("/api/admin/migrations/status")
async def admin_migrations_status(user: dict = Depends(require_master_admin)):
    """
    DB/schema migration status from live app DB.
    """
    async with db_pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW())"
        )
        rows = await conn.fetch(
            "SELECT version, applied_at FROM schema_migrations ORDER BY version"
        )
    applied_versions = [int(r["version"]) for r in rows]
    applied_set = set(applied_versions)
    latest_applied = max(applied_versions) if applied_versions else 0
    critical = [
        {
            "version": v,
            "applied": v in applied_set,
        }
        for v in MIGRATIONS_CRITICAL_VERSIONS
    ]
    return {
        "latest_code_version": MIGRATIONS_LATEST_VERSION,
        "latest_applied_version": latest_applied,
        "applied_count": len(applied_versions),
        "applied_versions": applied_versions,
        "critical_versions": critical,
        "missing_critical_versions": [v for v in MIGRATIONS_CRITICAL_VERSIONS if v not in applied_set],
        "in_sync_to_latest": latest_applied >= MIGRATIONS_LATEST_VERSION,
        "applied_at": {
            str(int(r["version"])): (r["applied_at"].isoformat() if r["applied_at"] else None)
            for r in rows
        },
    }


@app.post("/api/admin/migrations/run")
async def admin_run_migrations(user: dict = Depends(require_master_admin)):
    """
    Trigger run_migrations from the live app process.
    """
    async with db_pool.acquire() as conn:
        before_rows = await conn.fetch("SELECT version FROM schema_migrations")
    before_set = {int(r["version"]) for r in before_rows}

    await run_migrations()

    async with db_pool.acquire() as conn:
        after_rows = await conn.fetch(
            "SELECT version, applied_at FROM schema_migrations ORDER BY version"
        )
    after_set = {int(r["version"]) for r in after_rows}
    newly_applied = sorted(after_set - before_set)
    return {
        "ok": True,
        "newly_applied_versions": newly_applied,
        "latest_applied_version": (max(after_set) if after_set else 0),
        "latest_code_version": MIGRATIONS_LATEST_VERSION,
        "in_sync_to_latest": (max(after_set) if after_set else 0) >= MIGRATIONS_LATEST_VERSION,
    }


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
            ip_address = client_ip(request)
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
   - Examples: "This road changed my perspective " or "POV: You find the perfect line "
   
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
    places = await conn.fetch("SELECT * FROM trill_places")

    best = None
    min_dist = float("inf")

    for p in places:
        dist = haversine_distance_km(lat, lon, float(p["lat"]), float(p["lon"]))
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
# Health, Readiness & Metrics
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now_utc().isoformat()}

@app.get("/ready")
async def readiness():
    """Deep readiness probe — checks DB and Redis connectivity."""
    checks = {}
    healthy = True

    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"
        healthy = False

    if redis_client:
        try:
            await redis_client.ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {e}"
            healthy = False
    else:
        checks["redis"] = "not_configured"

    status = 200 if healthy else 503
    return JSONResponse(
        status_code=status,
        content={"status": "ok" if healthy else "degraded", "checks": checks, "timestamp": _now_utc().isoformat()},
    )

@app.get("/metrics")
async def metrics(request: Request):
    """Lightweight metrics for monitoring dashboards. Admin-only or internal."""
    auth = request.headers.get("authorization", "")
    metrics_key = os.environ.get("METRICS_API_KEY", "")
    if metrics_key and auth != f"Bearer {metrics_key}":
        raise HTTPException(403, "Forbidden")
    try:
        async with db_pool.acquire() as conn:
            total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
            active_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM users WHERE last_active_at > NOW() - INTERVAL '24 hours'"
            )
            uploads_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
            processing_now = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE status = 'processing'"
            )
            queued_now = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE status IN ('queued', 'staged')"
            )
            failed_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE status = 'failed' AND updated_at > NOW() - INTERVAL '24 hours'"
            )
        pool_size = db_pool.get_size() if hasattr(db_pool, 'get_size') else -1
        pool_free = db_pool.get_idle_size() if hasattr(db_pool, 'get_idle_size') else -1
        return {
            "users": {"total": total_users, "active_24h": active_24h},
            "uploads": {"last_24h": uploads_24h, "processing": processing_now, "queued": queued_now, "failed_24h": failed_24h},
            "db_pool": {"size": pool_size, "idle": pool_free},
            "timestamp": _now_utc().isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, f"Metrics error: {e}")

# ============================================================
# Auth Endpoints
# ============================================================
@app.post("/api/auth/register")
async def register(data: UserCreate, background_tasks: BackgroundTasks, request: Request):
    async with db_pool.acquire() as conn:
        if await conn.fetchrow("SELECT id FROM users WHERE LOWER(email) = $1", data.email.lower()):
            raise api_problem(
                409,
                code="email_already_registered",
                message="Email already registered",
            )
        user_id = str(uuid.uuid4())
        # Capture country from Cloudflare header if present
        country_code = (request.headers.get("CF-IPCountry") or "")[:2].upper() or None
        if country_code in ("XX", "T1", ""):
            country_code = None
        await conn.execute(
            "INSERT INTO users (id, email, password_hash, name, country, email_verified) VALUES ($1, $2, $3, $4, $5, FALSE)",
            user_id, data.email.lower(), hash_password(data.password), data.name, country_code
        )
        await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1)", user_id)
        # Default credits from canonical free-tier entitlements.
        ent = get_entitlements_for_tier("free")
        now_utc = datetime.now(timezone.utc)
        days_in_month = calendar.monthrange(now_utc.year, now_utc.month)[1]
        signup_put = int(math.ceil((int(ent.put_monthly or 0)) / max(1, days_in_month)))
        signup_aic = int(math.ceil((int(ent.aic_monthly or 0)) / max(1, days_in_month)))
        drip_month = f"{now_utc.year}-{now_utc.month:02d}"
        await conn.execute(
            """
            INSERT INTO wallets (
                user_id, put_balance, aic_balance,
                subscription_drip_month, put_drip_granted, aic_drip_granted
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            user_id,
            signup_put,
            signup_aic,
            drip_month,
            signup_put,
            signup_aic,
        )
        await ledger_entry(conn, user_id, "put", signup_put, "signup_bonus")
        await ledger_entry(conn, user_id, "aic", signup_aic, "signup_bonus")

        # Email confirmation token (24h expiry)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        await conn.execute(
            "INSERT INTO email_confirmations (user_id, token_hash, expires_at) VALUES ($1, $2, $3)",
            user_id, token_hash, expires_at
        )

        background_tasks.add_task(notify_signup, data.email, data.name)
        background_tasks.add_task(send_signup_confirmation_email, data.email, data.name, token)

    return {"ok": True, "email": data.email.lower()}

@app.post("/api/auth/resend-confirmation")
async def resend_confirmation(payload: ResendConfirmationRequest, background_tasks: BackgroundTasks):
    """
    Resend signup confirmation email for an unverified account.
    Returns ok even when account does not exist to reduce enumeration.
    """
    email = payload.email.lower().strip()
    async with db_pool.acquire() as conn:
        user_row = await conn.fetchrow(
            "SELECT id, email, name, email_verified, status FROM users WHERE LOWER(email) = $1",
            email,
        )
        if not user_row:
            return {"ok": True}
        if user_row.get("email_verified", False):
            return {"ok": True, "already_verified": True}
        if user_row.get("status") in ("banned", "disabled"):
            return {"ok": True}

        # Invalidate old outstanding links and issue a fresh one.
        await conn.execute(
            "UPDATE email_confirmations SET used_at = NOW() WHERE user_id = $1 AND used_at IS NULL",
            user_row["id"],
        )
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        await conn.execute(
            "INSERT INTO email_confirmations (user_id, token_hash, expires_at) VALUES ($1, $2, $3)",
            user_row["id"], token_hash, expires_at
        )

    background_tasks.add_task(
        send_signup_confirmation_email,
        user_row["email"],
        user_row["name"] or "there",
        token,
    )
    return {"ok": True}

@app.post("/api/auth/update-pending-email")
async def update_pending_email(payload: UpdatePendingEmailRequest, background_tasks: BackgroundTasks):
    """
    Let users fix a typo in their signup email before verification,
    while keeping the same pending account.
    """
    current_email = payload.current_email.lower().strip()
    new_email = payload.new_email.lower().strip()
    if current_email == new_email:
        return {"ok": True, "email": new_email}

    async with db_pool.acquire() as conn:
        user_row = await conn.fetchrow(
            "SELECT id, name, email_verified, status FROM users WHERE LOWER(email) = $1",
            current_email,
        )
        if not user_row:
            raise api_problem(
                404,
                code="pending_account_not_found",
                message="Pending account not found",
            )
        if user_row.get("email_verified", False):
            raise api_problem(
                409,
                code="account_already_verified",
                message="This account is already verified",
            )
        if user_row.get("status") in ("banned", "disabled"):
            raise api_problem(
                403,
                code="account_not_eligible",
                message="Account is not eligible for email updates",
            )

        existing = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email) = $1 AND id <> $2",
            new_email,
            user_row["id"],
        )
        if existing:
            raise api_problem(
                409,
                code="email_in_use",
                message="Email already in use",
            )

        await conn.execute(
            "UPDATE users SET email = $1, updated_at = NOW() WHERE id = $2",
            new_email,
            user_row["id"],
        )
        await conn.execute(
            "UPDATE email_confirmations SET used_at = NOW() WHERE user_id = $1 AND used_at IS NULL",
            user_row["id"],
        )

        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        await conn.execute(
            "INSERT INTO email_confirmations (user_id, token_hash, expires_at) VALUES ($1, $2, $3)",
            user_row["id"], token_hash, expires_at
        )

    background_tasks.add_task(
        send_signup_confirmation_email,
        new_email,
        user_row["name"] or "there",
        token,
    )
    return {"ok": True, "email": new_email}

@app.post("/api/auth/login")
async def login(data: UserLogin, request: Request, background_tasks: BackgroundTasks):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, email, name, password_hash, status, email_verified, must_reset_password, subscription_tier, last_login_ip, last_login_country, last_login_user_agent FROM users WHERE LOWER(email) = $1",
            data.email.lower(),
        )
        if not user or not verify_password(data.password, user["password_hash"]):
            raise api_problem(401, code="invalid_credentials", message="Invalid credentials")
        if user["status"] == "banned":
            raise api_problem(403, code="account_suspended", message="Account suspended")
        if not user.get("email_verified", True):
            raise api_problem(
                403,
                code="email_not_verified",
                message="Please verify your email before signing in. Check your inbox for the confirmation link.",
            )
        # Ensure daily refresh is applied on successful sign-in.
        await daily_refill(conn, str(user["id"]), user.get("subscription_tier") or "free")
        ip_now = client_ip(request)
        country_now = ((request.headers.get("cf-ipcountry") or "").strip().upper()[:2] or None)
        ua_now = (request.headers.get("user-agent") or "").strip()[:500]
        prev_ip = (user.get("last_login_ip") or "").strip()
        prev_country = (user.get("last_login_country") or "").strip().upper()
        prev_ua = (user.get("last_login_user_agent") or "").strip()
        ip_changed = bool(prev_ip and ip_now and prev_ip != ip_now)
        country_changed = bool(prev_country and country_now and prev_country != country_now)
        ua_changed = bool(prev_ua and ua_now and prev_ua != ua_now)
        likely_new_device = ip_changed or country_changed or ua_changed
        try:
            security_alerts_enabled = bool(
                await conn.fetchval(
                    "SELECT COALESCE(auth_security_alerts, TRUE) FROM user_preferences WHERE user_id = $1",
                    user["id"],
                )
            )
        except Exception:
            security_alerts_enabled = True
        localhost_now = ip_now in ("127.0.0.1", "::1")
        if likely_new_device and not localhost_now and security_alerts_enabled:
            background_tasks.add_task(
                send_login_anomaly_email,
                user["email"],
                user.get("name") or "there",
                ip_now,
                country_now or "",
                ua_now,
                prev_ip,
            )
        await conn.execute(
            "UPDATE users SET last_login_ip = $1, last_login_country = $2, last_login_user_agent = $3, last_active_at = NOW(), updated_at = NOW() WHERE id = $4",
            ip_now, country_now, ua_now, user["id"]
        )
        must_reset = bool(user.get("must_reset_password"))
        return {
            "access_token": create_access_jwt(str(user["id"])),
            "refresh_token": await create_refresh_token(conn, str(user["id"])),
            "token_type": "bearer",
            "must_reset_password": must_reset,
        }

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

            reset_link = f"{FRONTEND_URL.rstrip('/')}/reset-password.html?token={quote(token)}"
            background.add_task(send_password_reset_email, user_row["email"], reset_link)

    return {"ok": True}

@app.post("/api/auth/reset-password")
async def reset_password(payload: ResetPasswordRequest, background: BackgroundTasks):
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
            "UPDATE users SET password_hash=$1, must_reset_password=FALSE, updated_at=NOW() WHERE id=$2",
            new_hash, pr["user_id"],
        )
        await conn.execute("UPDATE password_resets SET used_at=NOW() WHERE id=$1", pr["id"])

        # Force logout across devices/sessions
        await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id=$1 AND revoked_at IS NULL", pr["user_id"])

        # Fetch email+name for the security confirmation email
        _u = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1", pr["user_id"])

    if _u:
        background.add_task(send_password_changed_email, _u["email"], _u["name"] or "there")

    return {"ok": True}


@app.get("/api/auth/confirm-email")
async def confirm_email(background_tasks: BackgroundTasks, token: str = Query(...)):
    """Verify signup email. Token from confirmation link. On success, returns tokens for auto-login."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    async with db_pool.acquire() as conn:
        ec = await conn.fetchrow(
            """
            SELECT ec.id, ec.user_id, ec.expires_at, ec.used_at
            FROM email_confirmations ec
            WHERE ec.token_hash = $1
            ORDER BY ec.created_at DESC
            LIMIT 1
            """,
            token_hash,
        )
        if not ec or ec["used_at"] is not None:
            raise HTTPException(status_code=410, detail="Link expired or already used")
        if ec["expires_at"] < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Link expired")

        await conn.execute("UPDATE users SET email_verified = TRUE, updated_at = NOW() WHERE id = $1", ec["user_id"])
        await conn.execute("UPDATE email_confirmations SET used_at = NOW() WHERE id = $1", ec["id"])

        user_row = await conn.fetchrow("SELECT email, name, subscription_tier FROM users WHERE id = $1", ec["user_id"])
        access = create_access_jwt(str(ec["user_id"]))
        refresh = await create_refresh_token(conn, str(ec["user_id"]))

    if user_row:
        background_tasks.add_task(send_welcome_email, user_row["email"], user_row["name"] or "there")
        ent = get_entitlements_for_tier(user_row.get("subscription_tier") or "free")
        background_tasks.add_task(
            send_fully_signed_up_guide_email,
            user_row["email"],
            user_row["name"] or "there",
            user_row.get("subscription_tier") or "free",
            int(ent.put_monthly or 0),
            int(ent.aic_monthly or 0),
            int(ent.max_accounts or 0),
            int(ent.max_accounts_per_platform or 0),
        )

    return {
        "ok": True,
        "email": user_row["email"] if user_row else None,
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
    }


@app.get("/api/auth/verify-email")
async def verify_email_change(token: str = Query(...)):
    """
    Complete admin-initiated email change: user clicks link in email to verify the new address.
    (Signup confirmation uses GET /api/auth/confirm-email instead.)
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, old_email, new_email, verification_token, used_at
              FROM email_changes
             WHERE verification_token = $1
             ORDER BY created_at DESC
             LIMIT 1
            """,
            token,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Invalid verification link")
        if row["used_at"] is not None:
            raise HTTPException(status_code=410, detail="This link was already used")

        # Re-check uniqueness at verification time to avoid race collisions.
        taken = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email)=LOWER($1) AND id <> $2",
            row["new_email"],
            row["user_id"],
        )
        if taken:
            raise HTTPException(status_code=409, detail="Email already in use")

        await conn.execute(
            "UPDATE users SET email = $1, email_verified = TRUE, updated_at = NOW() WHERE id = $2",
            row["new_email"],
            row["user_id"],
        )
        await conn.execute(
            "UPDATE email_changes SET used_at = NOW() WHERE id = $1",
            row["id"],
        )

    await invalidate_me_api_cache(str(row["user_id"]))
    return {"ok": True, "email": row["new_email"], "new_email": row["new_email"]}


# ============================================================
# Entitlements Schema API (no auth — for frontend tier/entitlement sync)
# ============================================================
def _entitlements_tiers_payload():
    """Canonical tier list and entitlement schema. Keys match ENTITLEMENT_KEYS in stages/entitlements.py."""
    return {
        "tiers": get_tiers_for_api(),
        "tier_slugs": list(TIER_SLUGS),
        "entitlement_keys": list(ENTITLEMENT_KEYS),
    }

@app.get("/api/entitlements/tiers")
async def get_entitlements_tiers():
    """Canonical tier list. Frontend uses this as single source."""
    return _entitlements_tiers_payload()

@app.get("/api/entitlements")
async def get_entitlements():
    """Alias for /api/entitlements/tiers — backward compatibility."""
    return _entitlements_tiers_payload()


# ============================================================
# Public Pricing API (no auth — for index.html, settings.html)
# ============================================================
@app.get("/api/pricing")
async def get_public_pricing():
    """
    Public pricing and entitlements for landing page and billing UI.
    Returns tiers with PUT/AIC, lookahead, queue_depth, max_thumbnails, max_caption_frames,
    trial_days, Stripe lookup keys, plus top-up packs (with suggested prices).
    """
    STRIPE_LOOKUP = {
        "creator_lite": "uploadm8_creator_lite_monthly",
        "creator_pro": "uploadm8_creator_pro_monthly",
        "studio": "uploadm8_studio_monthly",
        "agency": "uploadm8_agency_monthly",
    }
    tiers = []
    for slug in ("free", "creator_lite", "creator_pro", "studio", "agency"):
        cfg = TIER_CONFIG.get(slug, {})
        per_pf = cfg.get("max_accounts_per_platform", cfg.get("per_platform", 0))
        tiers.append({
            "slug": slug,
            "name": cfg.get("name", slug.replace("_", " ").title()),
            "price": float(cfg.get("price", 0)),
            "put_monthly": cfg.get("put_monthly", 0),
            "aic_monthly": cfg.get("aic_monthly", 0),
            "max_accounts": cfg.get("max_accounts", 0),
            "max_accounts_per_platform": int(per_pf or 0),
            "lookahead_hours": cfg.get("lookahead_hours", 0),
            "queue_depth": cfg.get("queue_depth", 0),
            "max_thumbnails": int(cfg.get("max_thumbnails", 1) or 1),
            "max_caption_frames": int(
                cfg.get("max_caption_frames", cfg.get("caption_frames", 3)) or 3
            ),
            "trial_days": int(cfg.get("trial_days", 0) or 0),
            "stripe_lookup_key": STRIPE_LOOKUP.get(slug),
        })
    topups = []
    for lookup_key, meta in TOPUP_PRODUCTS.items():
        topups.append({
            "lookup_key": lookup_key,
            "wallet": meta.get("wallet", ""),
            "amount": meta.get("amount", 0),
            "price_usd": meta.get("price_usd") or meta.get("price"),
            "label": f"{meta.get('wallet', '').upper()} {meta.get('amount', 0)}",
        })
    return {"tiers": tiers, "topups": topups}

# ============================================================
# User Profile & Wallet
# ============================================================
async def _build_me_response_dict(user: dict) -> dict:
    """Build GET /api/me JSON from a session user dict (row + wallet)."""
    raw_tier = user.get("subscription_tier", "free")
    ent = get_entitlements_from_user(dict(user))
    plan = entitlements_to_dict(ent)  # plan = entitlements (single source)
    wallet = user.get("wallet", {})
    role = user.get("role", "user")

    avatar_r2_key = user.get("avatar_r2_key")
    avatar_signed_url = None
    if avatar_r2_key:
        try:
            avatar_signed_url = generate_presigned_download_url(avatar_r2_key)
        except Exception as e:
            logger.warning(f"Failed to presign avatar for user {user.get('id')}: {e}")

    raw_name = user.get("name")
    first = (user.get("first_name") or "").strip()
    last = (user.get("last_name") or "").strip()
    combined = f"{first} {last}".strip() if (first or last) else None
    email_prefix = (user.get("email") or "").split("@")[0] if user.get("email") else None
    display_name = raw_name or combined or email_prefix or "User"
    next_upgrade_tier = get_next_public_upgrade_tier(raw_tier)

    accounts_connected = 0
    accounts_by_platform: Dict[str, int] = {}
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT platform, COUNT(*)::int AS n
                FROM platform_tokens
                WHERE user_id = $1
                  AND revoked_at IS NULL
                GROUP BY platform
                """,
                user["id"],
            )
            for r in rows:
                p = (r.get("platform") or "").lower()
                n = int(r.get("n") or 0)
                if not p:
                    continue
                accounts_by_platform[p] = n
                accounts_connected += n
    except Exception as e:
        logger.debug("_build_me_response_dict: platform token counts unavailable: %s", e)

    return {
        "id": user["id"],
        "email": user["email"],
        "name": display_name,
        "role": role,
        "timezone": user.get("timezone") or "America/Chicago",
        "avatar_r2_key": avatar_r2_key,
        "avatar_url": avatar_signed_url,
        "avatarUrl": avatar_signed_url,
        "avatar_signed_url": avatar_signed_url,
        "avatarSignedUrl": avatar_signed_url,
        "subscription_tier":      raw_tier,
        "tier":                  ent.tier,
        "tier_display":          ent.tier_display,
        "subscription_status":     user.get("subscription_status"),
        "current_period_end":      user.get("current_period_end").isoformat() if user.get("current_period_end") else None,
        "trial_end":               user.get("trial_end").isoformat() if user.get("trial_end") else None,
        "stripe_subscription_id":  user.get("stripe_subscription_id"),
        "stripe_customer_id":      user.get("stripe_customer_id"),
        "billing_mode":            BILLING_MODE,
        "accounts_connected":      accounts_connected,
        "accounts_by_platform":    accounts_by_platform,
        "wallet": {
            "put_balance":  float(wallet.get("put_balance", 0.0) or 0.0),
            "aic_balance":  float(wallet.get("aic_balance", 0.0) or 0.0),
            "put_reserved": float(wallet.get("put_reserved", 0.0) or 0.0),
            "aic_reserved": float(wallet.get("aic_reserved", 0.0) or 0.0),
            "updated_at":   wallet.get("updated_at"),
        },
        "plan": plan,
        "features": {
            "uploads":     plan.get("put_monthly", 0) > 0,
            "scheduler":   plan.get("can_schedule", False),
            "analytics":   bool(plan.get("analytics") and plan.get("analytics") != "basic"),
            "watermark":   plan.get("can_watermark", True),
            "white_label": plan.get("can_white_label", False),
            "support":     True,
        },
        "entitlements": entitlements_to_dict(
            get_entitlements_from_user(dict(user))
        ),
        "must_reset_password": bool(user.get("must_reset_password")),
        "billing_upgrade_hint": {
            "next_tier": next_upgrade_tier,
            "next_tier_display": get_tier_display_name(next_upgrade_tier),
            "billing_url": "/settings.html#billing",
            "pricing_url": "/index.html#pricing",
        } if next_upgrade_tier else None,
    }


@app.get("/api/me")
async def get_me(authorization: Optional[str] = Header(None)):
    """
    Session snapshot for the SPA. Bearer responses are cached in Redis (ME_API_CACHE_TTL_SEC) to cut
    repeated platform-token aggregation and payload assembly; still runs last_active, daily_refill,
    and fresh wallet on every request. API-key clients skip the response cache.
    """
    if not authorization:
        raise HTTPException(401, "Missing authorization")

    if authorization.startswith("um8_"):
        user = await _auth_via_api_key(authorization)
        if not user:
            raise HTTPException(401, "Invalid API key")
        return await _build_me_response_dict(user)

    auth_token = authorization[7:] if authorization.startswith("Bearer ") else None
    if not auth_token:
        raise HTTPException(401, "Missing authorization")
    user_id = verify_access_jwt(auth_token)
    if not user_id:
        raise HTTPException(401, "Invalid token")

    ttl = ME_API_CACHE_TTL_SEC
    if ttl > 0 and redis_client:
        cached = await cache_get(f"api_me:{user_id}")
        if isinstance(cached, dict) and cached.get("_me_cache_v") == 1:
            row = None
            wallet = None
            try:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT id, status, subscription_tier, role, must_reset_password, avatar_r2_key
                        FROM users WHERE id = $1
                        """,
                        user_id,
                    )
                    if not row:
                        await invalidate_me_api_cache(user_id)
                        raise HTTPException(401, "User not found")
                    if row["status"] == "banned":
                        raise HTTPException(403, "Account suspended")
                    await conn.execute("UPDATE users SET last_active_at = NOW() WHERE id = $1", user_id)
                    await daily_refill(conn, user_id, row["subscription_tier"])
                    wallet = await get_wallet(conn, user_id)
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"/api/me cache fast-path failed user={user_id}: {e}")
                row = None

            if row is not None and wallet is not None:
                st = str(row.get("subscription_tier") or "")
                sr = str(row.get("role") or "")
                if st == str(cached.get("subscription_tier") or "") and sr == str(cached.get("role") or ""):
                    out = {k: v for k, v in cached.items() if k != "_me_cache_v"}
                    out["must_reset_password"] = bool(row.get("must_reset_password"))
                    out["wallet"] = {
                        "put_balance": float(wallet.get("put_balance", 0.0) or 0.0),
                        "aic_balance": float(wallet.get("aic_balance", 0.0) or 0.0),
                        "put_reserved": float(wallet.get("put_reserved", 0.0) or 0.0),
                        "aic_reserved": float(wallet.get("aic_reserved", 0.0) or 0.0),
                        "updated_at": wallet.get("updated_at"),
                    }
                    ar2 = row.get("avatar_r2_key")
                    out["avatar_r2_key"] = ar2
                    avatar_signed_url = None
                    if ar2:
                        try:
                            avatar_signed_url = generate_presigned_download_url(ar2)
                        except Exception as e:
                            logger.warning(f"Failed to presign avatar for user {user_id}: {e}")
                    out["avatar_url"] = avatar_signed_url
                    out["avatarUrl"] = avatar_signed_url
                    out["avatar_signed_url"] = avatar_signed_url
                    out["avatarSignedUrl"] = avatar_signed_url
                    return out
                await invalidate_me_api_cache(user_id)

    user = await _load_user_session_from_db(user_id)
    body = await _build_me_response_dict(user)
    if ttl > 0 and redis_client:
        try:
            await cache_set(f"api_me:{user_id}", {**body, "_me_cache_v": 1}, ttl)
        except Exception as e:
            logger.debug("get_me: cache_set body failed user_id=%s: %s", user_id, e)
    return body


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
        await invalidate_me_api_cache(str(user["id"]))
    return {"status": "updated"}

@app.post("/api/auth/change-password")
async def change_password(data: PasswordChange, background: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Change user password"""
    async with db_pool.acquire() as conn:
        # Verify current password
        user_row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")
        
        # Update to new password
        new_hash = hash_password(data.new_password)
        await conn.execute(
            "UPDATE users SET password_hash = $1, must_reset_password = FALSE, updated_at = NOW() WHERE id = $2",
            new_hash,
            user["id"],
        )

        # Optionally invalidate other sessions (refresh tokens)
        await conn.execute("DELETE FROM refresh_tokens WHERE user_id = $1", user["id"])

    logger.info(f"Password changed for user {user['id']}")
    background.add_task(send_password_changed_email, user["email"], user.get("name") or "there")
    await invalidate_me_api_cache(str(user["id"]))
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

    if data.avatar_r2_key is not None:
        updates.append(f"avatar_r2_key = ${len(params)+1}")
        params.append((data.avatar_r2_key or "").strip() or None)
    
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
        await invalidate_me_api_cache(str(user["id"]))
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
        if data.captionStyle is not None:
            prefs["captionStyle"] = data.captionStyle
        if data.captionTone is not None:
            prefs["captionTone"] = data.captionTone
        if data.captionVoice is not None:
            prefs["captionVoice"] = data.captionVoice
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
async def update_password_settings(data: PasswordChange, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Change user password (settings endpoint version)"""
    async with db_pool.acquire() as conn:
        # Verify current password
        user_row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")

        # Update to new password; clear admin force-reset flag
        new_hash = hash_password(data.new_password)
        await conn.execute(
            "UPDATE users SET password_hash = $1, must_reset_password = FALSE, updated_at = NOW() WHERE id = $2",
            new_hash,
            user["id"],
        )

        # Optionally invalidate other sessions (refresh tokens)
        await conn.execute("DELETE FROM refresh_tokens WHERE user_id = $1", user["id"])

    logger.info(f"Password changed via settings for user {user['id']}")
    background_tasks.add_task(send_password_changed_email, user["email"], user.get("name") or "there")
    await invalidate_me_api_cache(str(user["id"]))
    return {"status": "success", "message": "Password changed successfully"}


@app.put("/api/settings/email")
async def update_settings_email(data: SettingsEmailChange, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Self-serve email change from settings, secured by current password."""
    new_email = data.new_email.lower().strip()
    old_email = (user.get("email") or "").lower().strip()
    if not old_email:
        raise HTTPException(status_code=400, detail="Current email not found")
    if new_email == old_email:
        return {"status": "success", "email": new_email, "message": "Email unchanged"}

    async with db_pool.acquire() as conn:
        user_row = await conn.fetchrow(
            "SELECT password_hash, name FROM users WHERE id = $1",
            user["id"],
        )
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        exists = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email)=LOWER($1) AND id <> $2",
            new_email,
            user["id"],
        )
        if exists:
            raise HTTPException(status_code=409, detail="Email already in use")

        verification_token = secrets.token_urlsafe(32)
        await conn.execute(
            "UPDATE email_changes SET used_at = NOW() WHERE user_id = $1::uuid AND used_at IS NULL",
            user["id"],
        )
        await conn.execute(
            """
            INSERT INTO email_changes (user_id, old_email, new_email, changed_by_admin_id, verification_token)
            VALUES ($1::uuid, $2, $3, NULL, $4)
            """,
            user["id"],
            old_email,
            new_email,
            verification_token,
        )
        # Keep current login email until verification link is used.
        await conn.execute("UPDATE users SET updated_at=NOW() WHERE id=$1", user["id"])

    verify_link = f"{FRONTEND_URL.rstrip('/')}/verify-email.html?token={verification_token}"
    target_name = (user_row["name"] if user_row else None) or "there"
    background_tasks.add_task(send_email_change_email, new_email, old_email, target_name, verify_link)
    background_tasks.add_task(send_user_email_change_notice_to_old_email, old_email, new_email, target_name)
    return {"status": "success", "email": new_email, "message": "Verification email sent"}

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

        await invalidate_me_api_cache(str(user["id"]))
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
# ──────────────────────────────────────────────────────────────────────────────
# Account deletion helper (TOS: paid users keep access until period end)
# ──────────────────────────────────────────────────────────────────────────────

async def _execute_account_deletion(conn, user: dict, ip_addr: str = None, initiated_by: str = "self") -> dict:
    """
    Performs full account deletion: revoke platform tokens, delete DB rows, purge R2.
    Called from DELETE /api/me (immediate) or customer.subscription.deleted (deferred).
    """
    user_id = str(user["id"])
    r2_rows = await conn.fetch(
        "SELECT r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key FROM uploads WHERE user_id = $1",
        user["id"],
    )
    r2_keys = []
    for row in r2_rows:
        for col in ("r2_key", "telemetry_r2_key", "processed_r2_key", "thumbnail_r2_key"):
            v = row.get(col) if col in row.keys() else None
            if v:
                r2_keys.append(v)
    avatar_key = user.get("avatar_r2_key") or ""
    if avatar_key:
        r2_keys.append(avatar_key)

    token_rows = await conn.fetch(
        "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE user_id = $1",
        user["id"],
    )
    tokens_revoked = 0
    for trow in token_rows:
        ok, err = await _revoke_platform_token(trow["platform"], trow["token_blob"])
        if ok:
            tokens_revoked += 1
        await conn.execute(
            """
            INSERT INTO platform_disconnect_log
                (user_id, platform, account_id, account_name,
                 revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            user_id,
            trow["platform"],
            trow["account_id"],
            trow["account_name"],
            ok,
            err or None,
            initiated_by,
            ip_addr,
        )

    rows_deleted = {}
    for tbl in ("uploads", "platform_tokens", "token_ledger", "wallets",
                "user_settings", "user_preferences", "refresh_tokens",
                "user_color_preferences", "account_groups", "white_label_settings"):
        try:
            n = await conn.fetchval(f"SELECT COUNT(*) FROM {tbl} WHERE user_id = $1", user["id"])
            rows_deleted[tbl] = int(n)
        except Exception as e:
            logger.debug("_execute_account_deletion: count skip table=%s: %s", tbl, e)
    rows_deleted["users"] = 1

    import asyncio as _aio
    _aio.ensure_future(send_account_deleted_email(
        user.get("email", ""),
        user.get("name") or "there",
    ))

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
    except Exception as e:
        logger.debug("_execute_account_deletion: white_label_settings delete skip: %s", e)
    await conn.execute("DELETE FROM uploads                  WHERE user_id = $1", user["id"])
    try:
        await conn.execute(
            "UPDATE support_messages SET name = '[deleted]', email = '[deleted]' WHERE user_id = $1",
            user["id"],
        )
    except Exception as e:
        logger.debug("_execute_account_deletion: support_messages anonymize skip: %s", e)
    await conn.execute("DELETE FROM users WHERE id = $1", user["id"])

    r2_deleted = await _delete_r2_objects(r2_keys)
    return {"r2_deleted": r2_deleted, "tokens_revoked": tokens_revoked, "rows_deleted": rows_deleted}


# Self-serve account deletion  DELETE /api/me
# ──────────────────────────────────────────────────────────────────────────────

@app.delete("/api/me")
async def delete_account(request: Request, user: dict = Depends(get_current_user)):
    """
    Self-serve account deletion. TOS-aligned:
      • Free users: deletion is immediate.
      • Paid users: subscription cancelled (no future charges), access until period end,
        then full deletion when Stripe sends subscription.deleted.
    """
    user_id = str(user["id"])
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)

    if user.get("role") == "master_admin":
        raise HTTPException(403, "Master admin accounts cannot be deleted via this endpoint.")

    async with db_pool.acquire() as conn:
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

        stripe_sub_id = user.get("stripe_subscription_id")
        has_active_paid_sub = bool(stripe_sub_id and STRIPE_SECRET_KEY)

        if has_active_paid_sub:
            try:
                sub = stripe.Subscription.retrieve(stripe_sub_id)
                if sub.status in ("active", "trialing"):
                    has_active_paid_sub = True
                else:
                    has_active_paid_sub = False
            except Exception as e:
                logger.debug("delete_account: Stripe subscription retrieve failed: %s", e)
                has_active_paid_sub = False

        if has_active_paid_sub:
            # TOS: paid users keep access until end of billing period
            await conn.execute(
                "UPDATE users SET deletion_requested_at = NOW() WHERE id = $1",
                user["id"],
            )
            try:
                stripe.Subscription.cancel(stripe_sub_id)
            except Exception as e:
                logger.warning(f"Stripe cancel failed for {user_id}: {e}")

            period_end = user.get("current_period_end")
            access_until = period_end.strftime("%B %d, %Y") if period_end and hasattr(period_end, "strftime") else "end of billing period"

            logger.info(f"[DELETION SCHEDULED] user={user_id} access_until={access_until}")
            return {
                "status": "deletion_scheduled",
                "message": "Your account will be deleted at the end of your billing period. You retain access until then.",
                "access_until": access_until,
            }
        else:
            # Free user or no active subscription: delete immediately
            result = await _execute_account_deletion(conn, user, ip_addr=ip_addr, initiated_by="account_deletion")
            await conn.execute(
                """
                UPDATE account_deletion_log
                SET completed_at = NOW(), r2_keys_deleted = $2, tokens_revoked = $3,
                    stripe_cancelled = $4, rows_deleted = $5
                WHERE id = $1
                """,
                deletion_log_id,
                result["r2_deleted"],
                result["tokens_revoked"],
                False,
                json.dumps(result["rows_deleted"]),
            )
            logger.info(f"[DELETION COMPLETE] user={user_id} r2={result['r2_deleted']} tokens={result['tokens_revoked']}")
            return {
                "status": "account_deleted",
                "summary": {
                    "r2_objects_deleted": result["r2_deleted"],
                    "platform_tokens_revoked": result["tokens_revoked"],
                    "rows_deleted": result["rows_deleted"],
                },
            }

@app.get("/api/wallet")
async def get_wallet_endpoint(user: dict = Depends(get_current_user)):
    plan = get_plan(user.get("subscription_tier", "free"))
    promo_defaults = {
        "promo_burst_week_enabled": bool(admin_settings_cache.get("promo_burst_week_enabled", False)),
        "promo_referral_enabled": bool(admin_settings_cache.get("promo_referral_enabled", False)),
    }
    fallback_wallet = {"put_balance": 0, "aic_balance": 0, "put_reserved": 0, "aic_reserved": 0}
    fallback_marketing = {
        "burn_put_pct": 0.0,
        "burn_aic_pct": 0.0,
        "put_capacity": int(plan.get("put_monthly", 30) or 30),
        "aic_capacity": int(plan.get("aic_monthly", 0) or 0),
        "ai_enabled": True,
        "banners": [],
        "links": {
            "topup": "/settings.html#billing",
            "topup_put": "/settings.html?topup=uploadm8_put_500#billing",
            "topup_aic": "/settings.html?topup=uploadm8_aic_1000#billing",
            "upgrade": "/settings.html#billing",
        },
        "period_start": None,
        "put_spent_period": 0,
        "aic_spent_period": 0,
        "put_available": 0,
        "aic_available": 0,
        "sales_opportunities": [],
        "experiments": {},
        "suppression": {},
    }

    try:
        async with db_pool.acquire() as conn:
            wallet = await get_wallet(conn, user["id"])
            try:
                ledger = await conn.fetch(
                    "SELECT * FROM token_ledger WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50",
                    user["id"],
                )
            except Exception as e:
                logger.debug("/api/wallet: token_ledger history unavailable: %s", e)
                ledger = []

            settings_row = None
            try:
                settings_row = await conn.fetchrow(
                    """
                    SELECT auto_generate_captions, auto_generate_hashtags, auto_generate_thumbnails
                    FROM user_settings WHERE user_id = $1
                    """,
                    user["id"],
                )
            except Exception as e:
                logger.debug("/api/wallet: user_settings row unavailable: %s", e)
            try:
                marketing = await build_wallet_marketing_payload(
                    conn,
                    str(user["id"]),
                    user.get("subscription_tier"),
                    wallet,
                    plan,
                    dict(settings_row) if settings_row else None,
                    {**promo_defaults, **admin_settings_cache},
                    bool(user.get("flex_enabled")),
                )
            except Exception as e:
                logger.error(f"/api/wallet marketing payload failed user={user.get('id')}: {e}")
                marketing = fallback_marketing
    except Exception as e:
        logger.error(f"/api/wallet failed user={user.get('id')}: {e}")
        wallet = fallback_wallet
        ledger = []
        marketing = fallback_marketing

    # Daily PUT + AIC refill cadence (UTC date-based; same semantics as services.wallet.daily_refill()).
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    next_utc_midnight = datetime.combine(today_utc + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    seconds_until_next = max(0, int((next_utc_midnight - now_utc).total_seconds()))
    _days_in_month = calendar.monthrange(today_utc.year, today_utc.month)[1]
    put_daily = max(0, int(math.ceil((int(plan.get("put_monthly", 0) or 0)) / max(1, _days_in_month))))
    aic_daily = max(0, int(math.ceil((int(plan.get("aic_monthly", 0) or 0)) / max(1, _days_in_month))))
    put_monthly = int(plan.get("put_monthly", 0) or 0)
    aic_monthly = int(plan.get("aic_monthly", 0) or 0)
    put_balance_now = int((wallet or {}).get("put_balance") or 0)
    aic_balance_now = int((wallet or {}).get("aic_balance") or 0)
    last_refill = (wallet or {}).get("last_refill_date")
    if isinstance(last_refill, str):
        try:
            last_refill = datetime.fromisoformat(last_refill).date()
        except Exception:
            last_refill = None
    refilled_today = bool(last_refill and last_refill >= today_utc)
    month_key = f"{today_utc.year}-{today_utc.month:02d}"
    wk = wallet or {}
    if (wk.get("subscription_drip_month") or "") != month_key:
        put_drip_m = 0
        aic_drip_m = 0
    else:
        put_drip_m = int(wk.get("put_drip_granted") or 0)
        aic_drip_m = int(wk.get("aic_drip_granted") or 0)
    put_sub_remaining = max(0, put_monthly - put_drip_m)
    aic_sub_remaining = max(0, aic_monthly - aic_drip_m)
    tier_slug = str(plan.get("tier", "free") or "free")
    is_free_tier = tier_slug == "free"
    can_refill_now = bool(
        is_free_tier
        and (
            (put_daily > 0 and put_sub_remaining > 0)
            or (aic_daily > 0 and aic_sub_remaining > 0)
        )
        and not refilled_today
    )
    daily_topup = {
        "enabled": bool(is_free_tier and (put_daily > 0 or aic_daily > 0)),
        "token_type": "both",
        "amount": put_daily,  # backward-compatible alias (legacy PUT-only widget)
        "amount_put": put_daily,
        "amount_aic": aic_daily,
        "cap_monthly_put": put_monthly,
        "cap_monthly_aic": aic_monthly,
        "put_subscription_granted_month": put_drip_m,
        "aic_subscription_granted_month": aic_drip_m,
        "put_subscription_remaining_month": put_sub_remaining,
        "aic_subscription_remaining_month": aic_sub_remaining,
        "rollover_unlimited": True,
        "refill_policy": "free_daily" if is_free_tier else "paid_on_invoice",
        "last_refill_date": last_refill.isoformat() if hasattr(last_refill, "isoformat") else None,
        "refilled_today": refilled_today,
        "can_refill_now": can_refill_now,
        "next_refill_at": (now_utc if can_refill_now else next_utc_midnight).isoformat(),
        "seconds_until_refill": 0 if can_refill_now else seconds_until_next,
    }

    _links = dict(marketing.get("links", {}) or {})
    _links.setdefault("topup", "/settings.html#billing")
    _links.setdefault("topup_put", "/settings.html?topup=uploadm8_put_500#billing")
    _links.setdefault("topup_aic", "/settings.html?topup=uploadm8_aic_1000#billing")
    _links.setdefault("upgrade", "/settings.html#billing")

    return {
        "wallet": wallet,
        "plan_limits": {
            "put_daily": plan.get("put_daily", 1),
            "put_monthly": plan.get("put_monthly", 30),
            "aic_monthly": plan.get("aic_monthly", 0),
        },
        "ledger": [dict(l) for l in ledger],
        "burn_put_pct": marketing.get("burn_put_pct", 0.0),
        "burn_aic_pct": marketing.get("burn_aic_pct", 0.0),
        "put_capacity": marketing.get("put_capacity", int(plan.get("put_monthly", 30) or 30)),
        "aic_capacity": marketing.get("aic_capacity", int(plan.get("aic_monthly", 0) or 0)),
        "ai_enabled": bool(marketing.get("ai_enabled", True)),
        "banners": marketing.get("banners", []),
        "links": _links,
        "daily_topup": daily_topup,
        "period_start": marketing.get("period_start"),
        "put_spent_period": marketing.get("put_spent_period", 0),
        "aic_spent_period": marketing.get("aic_spent_period", 0),
        "put_available": marketing.get("put_available"),
        "aic_available": marketing.get("aic_available"),
        "sales_opportunities": marketing.get("sales_opportunities", []),
        "experiments": marketing.get("experiments", {}),
        "suppression": marketing.get("suppression", {}),
    }

@app.post("/api/wallet/topup")
async def wallet_topup(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        customer_id = await ensure_stripe_customer(conn, user, stripe)

    session = create_wallet_topup_checkout_session(
        stripe_client=stripe,
        customer_id=customer_id,
        lookup_key=data.lookup_key,
        topup_products=TOPUP_PRODUCTS,
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
        user_id=str(user["id"]),
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

@app.post("/api/marketing/events")
async def ingest_marketing_event(data: MarketingEventIn, user: dict = Depends(get_current_user)):
    sid = (data.session_id or "").strip()
    if not sid:
        sid = hashlib.sha256(
            f"{user.get('id')}|{user.get('email')}|{datetime.now(timezone.utc).strftime('%Y-%m-%d')}".encode("utf-8")
        ).hexdigest()[:24]
    page = ((data.page or "").strip() or None)
    meta = data.metadata if isinstance(data.metadata, dict) else {}
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO marketing_events (
                user_id, session_id, event_type, nudge_type, nudge_severity,
                cta_variant, urgency_variant, ordering_variant, page, metadata
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            """,
            str(user["id"]),
            sid,
            data.event_type,
            (data.nudge_type or "general")[:120],
            (data.nudge_severity or "")[:32] or None,
            (data.cta_variant or "")[:8] or None,
            (data.urgency_variant or "")[:8] or None,
            (data.ordering_variant or "")[:8] or None,
            page[:255] if page else None,
            json.dumps(meta),
        )
    return {"status": "ok"}


@app.get("/api/admin/users/{user_id}/daily-refill-preview")
async def admin_daily_refill_preview(user_id: str, admin: dict = Depends(require_admin)):
    """
    Dry-run view of daily refill logic for a user.
    No wallet mutations are performed.
    """
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT id, email, subscription_tier, status FROM users WHERE id = $1",
            user_id,
        )
        if not target:
            raise HTTPException(status_code=404, detail="User not found")

        wallet = await get_wallet(conn, user_id)
        ent = get_entitlements_for_tier(target.get("subscription_tier") or "free")
        today = _now_utc().date()
        last_refill = wallet.get("last_refill_date")
        current_put = int(wallet.get("put_balance") or 0)
        monthly_cap = int(ent.put_monthly or 0)
        aic_monthly_cap = int(ent.aic_monthly or 0)
        _dim = calendar.monthrange(today.year, today.month)[1]
        put_drip = max(0, int(math.ceil(monthly_cap / max(1, _dim))))
        aic_drip = max(0, int(math.ceil(aic_monthly_cap / max(1, _dim))))
        month_key = f"{today.year}-{today.month:02d}"
        if (wallet.get("subscription_drip_month") or "") != month_key:
            put_g = 0
            aic_g = 0
        else:
            put_g = int(wallet.get("put_drip_granted") or 0)
            aic_g = int(wallet.get("aic_drip_granted") or 0)
        put_sub_rem = max(0, monthly_cap - put_g)
        aic_sub_rem = max(0, aic_monthly_cap - aic_g)
        can_refill_today = not (last_refill and last_refill >= today)
        if getattr(ent, "is_internal", False):
            projected_put_add = 0
            projected_aic_add = 0
            would_refill = False
            reason = "internal_tier_invoice_grant"
        elif str(getattr(ent, "tier", "")) != "free":
            projected_put_add = 0
            projected_aic_add = 0
            would_refill = False
            reason = "paid_tier_refills_on_invoice_paid"
        else:
            projected_put_add = min(put_drip, put_sub_rem) if can_refill_today else 0
            projected_aic_add = min(aic_drip, aic_sub_rem) if can_refill_today else 0
            would_refill = (projected_put_add > 0) or (projected_aic_add > 0)
            if not can_refill_today:
                reason = "already_refilled_today"
            elif put_sub_rem <= 0 and aic_sub_rem <= 0:
                reason = "subscription_monthly_allowance_fully_granted"
            elif projected_put_add <= 0 and projected_aic_add <= 0:
                reason = "subscription_monthly_allowance_fully_granted"
            else:
                reason = "eligible"

    return {
        "ok": True,
        "user_id": str(target["id"]),
        "email": target["email"],
        "status": target.get("status"),
        "tier": target.get("subscription_tier") or "free",
        "today_utc": today.isoformat(),
        "last_refill_date": last_refill.isoformat() if hasattr(last_refill, "isoformat") else None,
        "wallet": {
            "put_balance": current_put,
            "put_reserved": int(wallet.get("put_reserved") or 0),
            "aic_balance": int(wallet.get("aic_balance") or 0),
            "aic_reserved": int(wallet.get("aic_reserved") or 0),
            "subscription_drip_month": wallet.get("subscription_drip_month"),
            "put_drip_granted": int(wallet.get("put_drip_granted") or 0),
            "aic_drip_granted": int(wallet.get("aic_drip_granted") or 0),
        },
        "limits": {
            "put_daily_drip": put_drip,
            "aic_daily_drip": aic_drip,
            "put_monthly": monthly_cap,
            "aic_monthly": aic_monthly_cap,
        },
        "decision": {
            "can_refill_today": can_refill_today,
            "put_subscription_remaining_month": put_sub_rem,
            "aic_subscription_remaining_month": aic_sub_rem,
            "projected_put_add": projected_put_add,
            "projected_aic_add": projected_aic_add,
            "would_refill": would_refill,
            "projected_balance_after_refill": current_put + projected_put_add,
            "reason": reason,
        },
        "requested_by_admin_id": str(admin.get("id")),
    }

# ============================================================
# Settings
# ============================================================
_SETTINGS_DEFAULTS = {
    "discord_webhook": None,
    "telemetry_enabled": True,
    "hud_enabled": True,
    "hud_position": "bottom-left",
    "speeding_mph": 80,
    "euphoria_mph": 100,
    "hud_speed_unit": "mph",
    "hud_color": "#FFFFFF",
    "hud_font_family": "Arial",
    "hud_font_size": 24,
    "ffmpeg_screenshot_interval": 5,
    "auto_generate_thumbnails": True,
    "auto_generate_captions": True,
    "auto_generate_hashtags": True,
    "default_hashtag_count": 5,
    "always_use_hashtags": False,
}

@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    """Get user settings including Trill preferences"""
    async with db_pool.acquire() as conn:
        try:
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
        except Exception as e:
            # Fallback if extended columns not yet migrated (e.g. pre-707)
            logger.warning(f"Full settings SELECT failed ({e}), using base columns")
            settings = await conn.fetchrow("""
                SELECT discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                    speeding_mph, euphoria_mph, hud_speed_unit, hud_color
                FROM user_settings WHERE user_id = $1
            """, user["id"])
            if settings:
                settings = dict(settings)
                for k, v in _SETTINGS_DEFAULTS.items():
                    settings.setdefault(k, v)
                return settings
            return dict(_SETTINGS_DEFAULTS)

        if not settings:
            return dict(_SETTINGS_DEFAULTS)
        result = dict(settings)
        for k, v in _SETTINGS_DEFAULTS.items():
            result.setdefault(k, v)
        return result

# Base columns that exist in user_settings from migration 5 (before 707)
_SETTINGS_BASE_FIELDS = [
    "discord_webhook", "telemetry_enabled", "hud_enabled",
    "hud_position", "speeding_mph", "euphoria_mph",
    "hud_speed_unit", "hud_color",
]
# Extended columns added in migration 707
_SETTINGS_EXTENDED_FIELDS = [
    "hud_font_family", "hud_font_size", "ffmpeg_screenshot_interval",
    "auto_generate_thumbnails", "auto_generate_captions", "auto_generate_hashtags",
    "default_hashtag_count", "always_use_hashtags",
]

@app.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    """Update user settings including Trill thresholds"""
    all_fields = _SETTINGS_BASE_FIELDS + _SETTINGS_EXTENDED_FIELDS
    updates, params = [], [user["id"]]

    for field in all_fields:
        val = getattr(data, field, None)
        if val is not None:
            updates.append(f"{field} = ${len(params)+1}")
            params.append(val)

    if not updates:
        return {"status": "updated"}

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO user_settings (user_id) 
            VALUES ($1) 
            ON CONFLICT (user_id) DO NOTHING
        """, user["id"])

        try:
            await conn.execute(
                f"UPDATE user_settings SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = $1",
                *params
            )
        except Exception as e:
            # Fallback: update only base columns if extended columns not yet migrated
            base_updates, base_params = [], [user["id"]]
            for field in _SETTINGS_BASE_FIELDS:
                val = getattr(data, field, None)
                if val is not None:
                    base_updates.append(f"{field} = ${len(base_params)+1}")
                    base_params.append(val)
            if base_updates:
                await conn.execute(
                    f"UPDATE user_settings SET {', '.join(base_updates)}, updated_at = NOW() WHERE user_id = $1",
                    *base_params
                )
                logger.warning(f"Settings update fell back to base columns after: {e}")
            else:
                raise

    logger.info(f"Updated settings for user {user['id']}: {updates}")
    return {"status": "updated"}


@app.post("/api/settings/test-discord-webhook")
async def test_user_discord_webhook(data: dict, user: dict = Depends(get_current_user)):
    """Send a test message to the user's Discord webhook (for Settings page Test Webhook button).
    Accepts webhookUrl or webhook_url in body. If empty, uses the user's saved webhook from settings.
    The same saved URL is used when admin sends via 'Send to user webhooks'."""
    webhook_url = (data.get("webhookUrl") or data.get("webhook_url") or "").strip()
    if not webhook_url:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(
                  NULLIF(TRIM(us.discord_webhook), ''),
                  NULLIF(TRIM(up.discord_webhook), ''),
                  NULLIF(TRIM(COALESCE(u.preferences->>'discordWebhook', u.preferences->>'discord_webhook')), '')
                ) AS url
                FROM users u
                LEFT JOIN user_settings us ON us.user_id = u.id
                LEFT JOIN user_preferences up ON up.user_id = u.id
                WHERE u.id = $1
                """,
                user["id"],
            )
            webhook_url = (row["url"] or "").strip() if row else ""
    if not webhook_url:
        raise HTTPException(400, "Webhook URL required. Save your webhook in Settings first, or pass webhookUrl in the request.")
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        raise HTTPException(400, "Invalid Discord webhook URL")
    test_embed = {
        "title": " UploadM8 Webhook Test",
        "description": "If you see this message, your webhook is configured correctly!",
        "color": 0x22c55e,
        "fields": [
            {"name": "Status", "value": " Connected", "inline": True},
            {"name": "Tested By", "value": user.get("email", "User"), "inline": True},
        ],
        "footer": {"text": "UploadM8 Notifications"},
        "timestamp": _now_utc().isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook_url, json={"embeds": [test_embed]})
            if r.status_code not in (200, 204):
                raise HTTPException(400, f"Discord returned status {r.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(504, "Webhook request timed out")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Failed to reach Discord: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Failed to send: {str(e)}")
    return {"status": "sent"}


# Canonical settings surfaces (avoid adding parallel preference stores):
#   users.preferences (JSON) — caption / hashtag / thumbnail style; primary UI: PUT /api/me/preferences
#   user_preferences (row) — processing toggles, AI flags; GET/POST/PUT /api/settings/preferences*
#   user_settings (row) — HUD, legacy discord on settings; GET/PUT /api/settings
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
    
    # Validate and sanitize hashtag data (string-safe; avoids list('#tag') char-splitting)
    def _split_tags(v):
        if v is None:
            return []
        # Accept JSON string list
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                maybe = json.loads(s)
                if isinstance(maybe, list):
                    v = maybe
                else:
                    v = s
            except Exception:
                v = s
        if isinstance(v, str):
            parts = re.split(r"[\s,]+", v.strip())
            return [p for p in parts if p]
        if isinstance(v, (list, tuple, set)):
            out = []
            for x in v:
                sx = str(x).strip()
                if sx:
                    out.append(sx)
            return out
        sx = str(v).strip()
        return [sx] if sx else []

    def _clean_tag_list(v, limit):
        out = []
        for t in _split_tags(v)[:limit]:
            t = str(t).strip().lower().lstrip("#")[:50]
            if t:
                out.append(t)
        return out

    # Support both camelCase (frontend) and snake_case (backend/worker) keys
    if "alwaysHashtags" in prefs or "always_hashtags" in prefs:
        v = prefs.get("alwaysHashtags", None)
        if v is None:
            v = prefs.get("always_hashtags", None)
        clean = _clean_tag_list(v, 100)
        prefs["alwaysHashtags"] = [f"#{t}" for t in clean]  # UI-friendly
        prefs["always_hashtags"] = clean                   # worker-friendly (no '#')

    if "blockedHashtags" in prefs or "blocked_hashtags" in prefs:
        v = prefs.get("blockedHashtags", None)
        if v is None:
            v = prefs.get("blocked_hashtags", None)
        clean = _clean_tag_list(v, 100)
        prefs["blockedHashtags"] = [f"#{t}" for t in clean]
        prefs["blocked_hashtags"] = clean

    if "platformHashtags" in prefs or "platform_hashtags" in prefs:
        ph = prefs.get("platformHashtags", None)
        if ph is None:
            ph = prefs.get("platform_hashtags", None)

        if isinstance(ph, str):
            try:
                ph = json.loads(ph)
            except Exception:
                ph = {}
        if not isinstance(ph, dict):
            ph = {}

        cleaned_ui = {}
        cleaned_worker = {}
        for platform in ["tiktok", "youtube", "instagram", "facebook"]:
            raw = ph.get(platform) or ph.get(platform.title()) or ph.get(platform.upper())
            clean = _clean_tag_list(raw, 50)
            cleaned_ui[platform] = [f"#{t}" for t in clean]
            cleaned_worker[platform] = clean

        prefs["platformHashtags"] = cleaned_ui
        prefs["platform_hashtags"] = cleaned_worker
    # Validate numeric hashtag settings
    if "maxHashtags" in prefs:
        prefs["maxHashtags"] = max(1, min(50, int(prefs["maxHashtags"])))
    if "aiHashtagCount" in prefs:
        prefs["aiHashtagCount"] = max(1, min(30, int(prefs["aiHashtagCount"])))
    
    # Validate hashtag position
    if "hashtagPosition" in prefs and prefs["hashtagPosition"] not in ["start", "end", "caption", "comment"]:
        prefs["hashtagPosition"] = "end"
    
    # Validate AI hashtag style (must match caption_stage + UI schema)
    if "aiHashtagStyle" in prefs and prefs["aiHashtagStyle"] not in ["lowercase", "capitalized", "camelcase", "mixed"]:
        prefs["aiHashtagStyle"] = "mixed"
    if "ai_hashtag_style" in prefs and prefs["ai_hashtag_style"] not in ["lowercase", "capitalized", "camelcase", "mixed"]:
        prefs["ai_hashtag_style"] = "mixed"

    # Caption & AI Settings — style / tone / voice (worker caption_stage reads these)
    _CAPTION_STYLES = ("story", "punchy", "factual")
    _CAPTION_TONES = ("hype", "calm", "cinematic", "authentic")
    _CAPTION_VOICES = (
        "default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator",
    )
    if "captionStyle" in prefs or "caption_style" in prefs:
        v = str(prefs.get("captionStyle") or prefs.get("caption_style") or "story").strip().lower()
        if v not in _CAPTION_STYLES:
            v = "story"
        prefs["captionStyle"] = prefs["caption_style"] = v
    if "captionTone" in prefs or "caption_tone" in prefs:
        v = str(prefs.get("captionTone") or prefs.get("caption_tone") or "authentic").strip().lower()
        if v not in _CAPTION_TONES:
            v = "authentic"
        prefs["captionTone"] = prefs["caption_tone"] = v
    if "captionVoice" in prefs or "caption_voice" in prefs:
        v = str(prefs.get("captionVoice") or prefs.get("caption_voice") or "default").strip().lower()
        if v not in _CAPTION_VOICES:
            v = "default"
        prefs["captionVoice"] = prefs["caption_voice"] = v

    if "audioTranscription" in prefs or "audio_transcription" in prefs:
        v = prefs.get("audioTranscription", prefs.get("audio_transcription", True))
        prefs["audioTranscription"] = prefs["audio_transcription"] = bool(v)

    async with db_pool.acquire() as conn:
        # MERGE with existing preferences — never replace entirely (frontend may send partial updates)
        existing_row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
        existing = {}
        if existing_row and existing_row.get("preferences"):
            raw = existing_row["preferences"]
            if isinstance(raw, str):
                try:
                    existing = json.loads(raw) if raw else {}
                except Exception:
                    existing = {}
            elif isinstance(raw, dict):
                existing = dict(raw)
        merged = {**existing, **prefs}
        _ge = _engine_enabled_from_mixed_prefs(merged)
        _ue = _use_studio_engine_from_mixed_prefs(merged)
        _strip_legacy_thumbnail_engine_keys(merged)
        if _ge is not None:
            merged["thumbnailStudioEngineEnabled"] = merged["thumbnail_studio_engine_enabled"] = bool(_ge)
        if _ue is not None:
            merged["thumbnailUseStudioEngine"] = merged["thumbnail_use_studio_engine"] = bool(_ue)
        await conn.execute(
            "UPDATE users SET preferences = $1, updated_at = NOW() WHERE id = $2",
            json.dumps(merged), user["id"]
        )
        # Sync discord_webhook to user_settings and user_preferences so:
        # - Admin "Send to user webhooks" finds it (announcement query reads from these tables)
        # - Worker load_user_settings can use either source
        discord_webhook = (prefs.get("discordWebhook") or prefs.get("discord_webhook") or "").strip() or None
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, discord_webhook) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET discord_webhook = $2, updated_at = NOW()
            """,
            user["id"],
            discord_webhook,
        )
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            user["id"],
        )
        await conn.execute(
            "UPDATE user_preferences SET discord_webhook = $1, updated_at = NOW() WHERE user_id = $2",
            discord_webhook,
            user["id"],
        )
    return {"status": "updated"}

# ============================================================
# Uploads
# ============================================================
@app.post("/api/uploads/presign")
async def presign_upload(data: UploadInit, request: Request, user: dict = Depends(get_current_user)):
    """Create upload with user preferences applied"""
    plan = get_plan(user.get("subscription_tier", "free"))

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
        # Step 1: Merge form hashtags + always_hashtags into upload record.
        #         Platform-specific hashtags are applied per-platform at publish time
        #         (context.get_effective_hashtags(platform)), not here.
        def _split_tags(v):
            if v is None:
                return []
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return []
                try:
                    maybe = json.loads(s)
                    if isinstance(maybe, list):
                        v = maybe
                    else:
                        v = s
                except Exception:
                    v = s
            if isinstance(v, str):
                return [p for p in re.split(r"[\s,]+", v.strip()) if p]
            if isinstance(v, (list, tuple, set)):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            return [s] if s else []

        def _to_hash_tags(v):
            out = []
            seen = set()
            for t in _split_tags(v):
                token = _sanitize_hashtag_token(t)
                if not token or token in seen:
                    continue
                seen.add(token)
                out.append(f"#{token}")
            return out

        # Store ONLY form/manual hashtags here. always_hashtags and platform_hashtags
        # are applied per-platform at publish via context.get_effective_hashtags(platform).
        combined = _to_hash_tags(getattr(data, "hashtags", []) or [])

        # Step 2: AI-generated hashtag injection — gated on plan + user toggle.
        #         Actual AI generation happens later in caption_stage.
        if user_prefs.get("ai_hashtags_enabled") and plan.get("ai"):
            pass  # ai_hashtags merged into ctx later by caption_stage

        # Step 3: Deduplicate + enforce limit.
        # NOTE: For single-video manual metadata, do not remove user-entered tags
        # based on saved blocked settings. Blocked tags are an AI/settings guardrail
        # and are applied in context layering for non-manual sources.
        data.hashtags = list(dict.fromkeys(combined))[: int(user_prefs.get("max_hashtags", 15))]

        # Each account has unique platform_tokens.id; dedupe target_accounts to avoid duplicate publishes
        target_accounts = list(dict.fromkeys(str(t) for t in (data.target_accounts or []) if t))

        # ── Compute PUT/AIC cost — canonical formula from entitlements ──
        ent_cost = get_entitlements_for_tier(user.get("subscription_tier", "free"))
        use_ai  = bool(getattr(data, "use_ai", False)) and ent_cost.can_ai
        use_hud = bool(user_prefs.get("hud_enabled", False)) and ent_cost.can_burn_hud
        upload_prefs = dict(user_prefs or {})
        if not use_ai:
            # Per-upload AI OFF: prevent AI stages + AIC debit for this upload.
            upload_prefs.update({
                "auto_captions": False,
                "auto_thumbnails": False,
                "ai_hashtags_enabled": False,
                "use_audio_context": False,
                "audio_transcription": False,
                "aiServiceTelemetry": False,
                "aiServiceAudioSignals": False,
                "aiServiceMusicDetection": False,
                "aiServiceAudioSummary": False,
                "aiServiceEmotionSignals": False,
                "aiServiceCaptionWriter": False,
                "aiServiceThumbnailDesigner": False,
                "aiServiceFrameInspector": False,
                "aiServiceSpeechToText": False,
                "aiServiceVideoAnalyzer": False,
                "aiServiceSceneUnderstanding": False,
            })

        # Thumbnail Studio gates (saved in users.preferences) + per-upload overrides.
        gate_studio = bool(upload_prefs.get("thumbnailStudioEnabled", upload_prefs.get("thumbnail_studio_enabled", False)))
        gate_engine = bool(
            upload_prefs.get(
                "thumbnailStudioEngineEnabled",
                upload_prefs.get(
                    "thumbnail_studio_engine_enabled",
                    upload_prefs.get("thumbnailPikzelsEnabled", upload_prefs.get("thumbnail_pikzels_enabled", False)),
                ),
            )
        )
        gate_persona = bool(upload_prefs.get("thumbnailPersonaEnabled", upload_prefs.get("thumbnail_persona_enabled", False)))

        req_use_engine = (
            bool(data.thumbnail_use_studio_engine)
            if data.thumbnail_use_studio_engine is not None
            else (bool(data.thumbnail_use_pikzels) if data.thumbnail_use_pikzels is not None else gate_engine)
        )
        req_use_persona = bool(data.thumbnail_use_persona) if data.thumbnail_use_persona is not None else gate_persona
        persona_id = (
            (data.thumbnail_persona_id or "").strip()
            or str(upload_prefs.get("thumbnailDefaultPersonaId") or upload_prefs.get("thumbnail_default_persona_id") or "").strip()
        )
        if persona_id and not _valid_uuid(persona_id):
            persona_id = ""
        try:
            persona_strength = int(
                data.thumbnail_persona_strength
                if data.thumbnail_persona_strength is not None
                else (upload_prefs.get("thumbnailPersonaStrength", upload_prefs.get("thumbnail_persona_strength", 70)) or 70)
            )
        except Exception:
            persona_strength = 70
        persona_strength = max(0, min(100, persona_strength))

        if (not gate_studio) or (not use_ai):
            req_use_engine = False
            req_use_persona = False
            persona_id = ""
        if not gate_engine:
            req_use_engine = False
        if not gate_persona:
            req_use_persona = False
            persona_id = ""

        if req_use_persona and persona_id:
            persona_exists = await conn.fetchval(
                "SELECT 1 FROM creator_personas WHERE id = $1::uuid AND user_id = $2::uuid",
                persona_id,
                user["id"],
            )
            if not persona_exists:
                req_use_persona = False
                persona_id = ""

        upload_prefs.update({
            "thumbnailStudioEnabled": gate_studio,
            "thumbnail_studio_enabled": gate_studio,
            "thumbnailStudioEngineEnabled": gate_engine,
            "thumbnail_studio_engine_enabled": gate_engine,
            "thumbnailPersonaEnabled": gate_persona,
            "thumbnail_persona_enabled": gate_persona,
            "thumbnailUseStudioEngine": bool(req_use_engine),
            "thumbnail_use_studio_engine": bool(req_use_engine),
            "thumbnailUsePersona": bool(req_use_persona and bool(persona_id)),
            "thumbnail_use_persona": bool(req_use_persona and bool(persona_id)),
            "thumbnailPersonaId": persona_id or None,
            "thumbnail_persona_id": persona_id or None,
            "thumbnailPersonaStrength": persona_strength,
            "thumbnail_persona_strength": persona_strength,
        })
        _strip_legacy_thumbnail_engine_keys(upload_prefs)

        # Each target account counts as a separate publish (costs +2 PUT per extra beyond 1)
        # When target_accounts provided: user selected specific accounts.
        # When empty: legacy one-per-platform.
        num_publish_targets = len(target_accounts) if target_accounts else len(data.platforms)
        put_cost, aic_cost = compute_upload_cost(
            entitlements=ent_cost,
            num_platforms=num_publish_targets,
            use_ai=use_ai,
            use_hud=use_hud,
            num_thumbnails=getattr(data, "thumbnail_count", None),
            duration_seconds=getattr(data, "duration_seconds", None),
            duration_hint=getattr(data, "duration_seconds", None),
            file_size=int(data.file_size) if getattr(data, "file_size", None) else None,
            user_prefs=upload_prefs,
            has_telemetry=bool(getattr(data, "has_telemetry", False)),
        )

        # Fresh wallet row — avoids racing another presign after JWT-time wallet snapshot
        wrow = await get_wallet(conn, str(user["id"]))
        put_avail = int(wrow.get("put_balance", 0) or 0) - int(wrow.get("put_reserved", 0) or 0)
        aic_avail = int(wrow.get("aic_balance", 0) or 0) - int(wrow.get("aic_reserved", 0) or 0)

        if not getattr(ent_cost, "is_internal", False):
            if put_avail < put_cost:
                raise api_problem(
                    429,
                    code="insufficient_put",
                    message=f"Insufficient PUT tokens ({put_avail} available, {put_cost} needed).",
                    topup_url="/settings.html#billing",
                )
            if aic_cost > 0 and aic_avail < aic_cost:
                raise api_problem(
                    429,
                    code="insufficient_aic",
                    message=f"Insufficient AIC credits ({aic_avail} available, {aic_cost} needed).",
                    topup_url="/settings.html#billing",
                )

        # ── Queue depth guard ─────────────────────────────────────────
        pending_count = await conn.fetchval(
            """SELECT COUNT(*) FROM uploads
               WHERE user_id = $1
               AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
            user["id"]
        )
        ent_check = get_entitlements_for_tier(user.get("subscription_tier", "free"))
        if pending_count >= ent_check.queue_depth:
            raise api_problem(
                429,
                code="queue_depth_exceeded",
                message=(
                    f"Queue limit reached ({pending_count}/{ent_check.queue_depth} uploads pending). "
                    "Wait for existing uploads to complete or upgrade your plan."
                ),
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

        # Store upload with preferences metadata (put_reserved/aic_reserved = quoted cost at presign)
        await conn.execute("""
            INSERT INTO uploads (
                id, user_id, r2_key, filename, file_size, platforms,
                title, caption, hashtags, privacy, status, scheduled_time,
                schedule_mode, put_reserved, aic_reserved, put_cost, aic_cost, schedule_metadata,
                user_preferences, target_accounts
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'pending', $11, $12, $13, $14, $15, $16, $17, $18, $19)
        """,
            upload_id, user["id"], r2_key, data.filename, data.file_size,
            data.platforms, data.title, data.caption, data.hashtags,
            data.privacy, scheduled_time, data.schedule_mode, put_cost,
            aic_cost, put_cost, aic_cost, json.dumps(schedule_metadata) if schedule_metadata else None,
            json.dumps(upload_prefs), target_accounts
        )

        reserved = await reserve_tokens(conn, str(user["id"]), put_cost, aic_cost, upload_id)
        if not reserved:
            await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
            raise api_problem(
                429,
                code="insufficient_credits",
                message="Could not reserve PUT/AIC (balance changed). Retry or check billing.",
                topup_url="/settings.html#billing",
            )
        try:
            await conn.execute(
                """
                INSERT INTO wallet_holds (upload_id, user_id, put_amount, aic_amount, status)
                VALUES ($1, $2, $3, $4, 'held')
                """,
                upload_id,
                user["id"],
                put_cost,
                aic_cost,
            )
        except Exception as e:
            logger.debug("upload init: wallet_holds insert skipped (non-fatal): %s", e)

    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    result = {
        "upload_id": upload_id,
        "presigned_url": presigned_url,
        "r2_key": r2_key,
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "schedule_mode": data.schedule_mode,
        "target_accounts": target_accounts,
        "preferences_applied": {
            "auto_captions": bool(upload_prefs.get("auto_captions")),
            "auto_thumbnails": bool(upload_prefs.get("auto_thumbnails")),
            "thumbnail_use_studio_engine": bool(upload_prefs.get("thumbnail_use_studio_engine")),
            "ai_hashtags": bool(upload_prefs.get("ai_hashtags_enabled")),
            "thumbnail_use_persona": bool(upload_prefs.get("thumbnail_use_persona")),
            "thumbnail_persona_id": upload_prefs.get("thumbnail_persona_id"),
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

class CompleteUploadBody(BaseModel):
    """Optional metadata from upload page (single-file manual title/caption/hashtags)."""
    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    privacy: Optional[str] = None
    target_accounts: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None


@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    """
    Complete upload and either enqueue immediately (immediate mode) or stage
    for deferred processing (scheduled / smart mode).

    IMMEDIATE  → status=queued, pushed to Redis → worker fires NOW
    SCHEDULED  → status=staged, NOT pushed to Redis → scheduler fires at scheduled_time - processing_window
    SMART      → status=staged, NOT pushed to Redis → scheduler fires at first scheduled_time - processing_window

    Request body may include title, caption, hashtags from the upload page (manual metadata for single-file uploads).
    Title and caption override when provided. Hashtags are merged with any tags already on the upload row (deduped).
    """
    body = {}
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw) or {}
    except Exception as e:
        logger.debug("complete_upload: optional JSON body parse failed upload_id=%s: %s", upload_id, e)

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

        # ── Apply manual metadata from upload page (single-file) ─────────────────
        updates = []
        params = []
        idx = 1
        if body.get("title") is not None:
            updates.append(f"title = ${idx}")
            params.append(str(body["title"])[:512])
            idx += 1
        if body.get("caption") is not None:
            updates.append(f"caption = ${idx}")
            params.append(str(body["caption"])[:10000])
            idx += 1
        if body.get("hashtags") is not None:
            raw_tags = body["hashtags"]
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in re.split(r"[\s,]+", str(raw_tags)) if t.strip()]
            incoming = _sanitize_hashtag_list(raw_tags if isinstance(raw_tags, (list, tuple)) else [])
            blocked = set(
                _sanitize_hashtag_token(x)
                for x in (user_prefs.get("blocked_hashtags") or user_prefs.get("blockedHashtags") or [])
            )
            existing = upload.get("hashtags")
            merged_raw: List[str] = []
            for tag in expand_hashtag_items(existing) + expand_hashtag_items(incoming):
                t = _sanitize_hashtag_token(tag)
                if not t or t in blocked:
                    continue
                merged_raw.append(f"#{t}")
            seen: set = set()
            tags: List[str] = []
            for t in merged_raw:
                k = t.lstrip("#").lower()
                if k in seen:
                    continue
                seen.add(k)
                tags.append(t)
            tags = tags[: int(user_prefs.get("max_hashtags", 15))]
            updates.append(f"hashtags = ${idx}")
            params.append(tags)  # TEXT[] expects list
            idx += 1

        if body.get("target_accounts") is not None:
            # Each account has unique platform_tokens.id; dedupe to avoid duplicate publishes
            target_ids = list(dict.fromkeys(str(t) for t in (body["target_accounts"] or []) if t))
            updates.append(f"target_accounts = ${idx}")
            params.append(target_ids)
            idx += 1

        if updates:
            params.append(upload_id)
            await conn.execute(
                f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = ${idx}",
                *params
            )

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
        except Exception as e:
            logger.debug("complete_upload: smart schedule_metadata parse failed upload_id=%s: %s", upload_id, e)

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
        upload = await conn.fetchrow(
            "SELECT put_reserved, aic_reserved, status, r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
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
            # Remove video and related assets from R2 so they don't persist
            r2_keys = [k for k in (
                upload.get("r2_key"),
                upload.get("telemetry_r2_key"),
                upload.get("processed_r2_key"),
                upload.get("thumbnail_r2_key"),
            ) if k]
            if r2_keys:
                await _delete_r2_objects(r2_keys)
            return {"status": "cancelled"}


# Status view groupings for queue/dashboard (simplified UX)
# pending: waiting to process (includes smart + scheduled)
# processing: actively being processed
# completed: done (succeeded, partial, or legacy completed)
# failed: publish failed
_UPLOAD_VIEW_STATUS = {
    "pending": ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
    "processing": ("processing",),
    "completed": ("completed", "succeeded", "partial"),
    "failed": ("failed",),
    "staged": ("pending", "staged", "queued", "scheduled", "ready_to_publish"),  # alias for pending
    "smart_schedule": None,  # special: schedule_mode='smart' + pending statuses
}
_STATUS_LABEL = {
    "pending": "Pending",
    "staged": "Scheduled",
    "queued": "Queued",
    "scheduled": "Scheduled",
    "ready_to_publish": "Ready to publish",
    "processing": "Processing",
    "completed": "Completed",
    "succeeded": "Succeeded",
    "partial": "Partial",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def _rollup_engagement_from_platform_results(
    entries: list,
    *,
    shortform_only: bool = False,
    successful_only: bool = True,
) -> dict[str, int]:
    """Sum per-platform metrics stored on platform_results when uploads.views/likes are stale."""
    tv = tl = tc = ts = 0
    if not entries:
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}

    def _pick_int(d: dict, *keys: str) -> int:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return int(d[k] or 0)
                except Exception:
                    return 0
        return 0

    shortform_platforms = {"tiktok", "youtube", "instagram", "facebook"}
    successful_statuses = {"published", "succeeded", "success", "completed", "partial"}

    for e in entries:
        if not isinstance(e, dict):
            continue
        plat = str(e.get("platform") or "").strip().lower()
        if shortform_only and plat and plat not in shortform_platforms:
            continue
        if successful_only:
            ok = bool(e.get("success") is True)
            st = str(e.get("status") or "").strip().lower()
            if (not ok) and (st not in successful_statuses):
                continue
        tv += _pick_int(e, "views", "view_count", "play_count", "playCount", "video_views")
        tl += _pick_int(e, "likes", "like_count", "likeCount")
        tc += _pick_int(e, "comments", "comment_count", "commentCount")
        ts += _pick_int(e, "shares", "share_count", "shareCount")
    return {"views": tv, "likes": tl, "comments": tc, "shares": ts}


def _normalize_upload_platform_results_list(raw: Any) -> list:
    pr = _safe_json(raw, [])
    if isinstance(pr, dict):
        return [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in pr.items()]
    if isinstance(pr, list):
        return pr
    return []


def _pr_entry_matches_catalog_video(e: dict, plat: str, catalog_vid: str) -> bool:
    """Match catalog platform_video_id to a platform_results element (handles Facebook id shapes)."""
    if str(e.get("platform") or "").lower() != plat:
        return False
    c = str(catalog_vid or "").strip()
    if not c:
        return False
    candidates: list[str] = []
    for k in (
        "platform_video_id",
        "video_id",
        "youtube_video_id",
        "tiktok_video_id",
        "facebook_video_id",
        "fb_video_id",
        "videoId",
        "post_id",
    ):
        v = e.get(k)
        if v is not None and str(v).strip():
            candidates.append(str(v).strip())
    eid = e.get("id")
    if eid is not None and str(eid).strip():
        candidates.append(str(eid).strip())
    for cand in candidates:
        if cand == c:
            return True
    if eid is not None and str(eid).strip() == c:
        return True
    if plat == "facebook":
        for cand in candidates:
            if not cand:
                continue
            if c == cand:
                return True
            if c in cand or cand in c:
                return True
            c_digits = "".join(ch for ch in c if ch.isdigit())
            d_digits = "".join(ch for ch in cand if ch.isdigit())
            if c_digits and c_digits == d_digits:
                return True
            if "_" in c:
                tail = c.split("_")[-1]
                if tail and (tail in cand or cand.endswith(tail)):
                    return True
    return False


def _platform_result_title_from_entry(e: dict) -> Optional[str]:
    for k in ("title", "video_title", "name"):
        v = e.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:500]
    for k in ("caption", "description"):
        v = e.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:500]
    return None


def _catalog_engagement_from_pr_entry(e: dict) -> dict[str, int]:
    """
    Like _engagement_from_single_platform_result_entry but:
    - Treats reactions as likes (Facebook)
    - If numeric metrics exist, do not drop the row when success/status is missing
    """
    if not isinstance(e, dict):
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}

    def _pick_int(d: dict, *keys: str) -> int:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return int(d[k] or 0)
                except Exception:
                    return 0
        return 0

    shortform_platforms = {"tiktok", "youtube", "instagram", "facebook"}
    successful_statuses = {
        "published",
        "succeeded",
        "success",
        "completed",
        "partial",
        "ok",
        "live",
        "done",
        "active",
        "complete",
        "available",
        "public",
    }
    plat = str(e.get("platform") or "").strip().lower()
    if plat and plat not in shortform_platforms:
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    views = _pick_int(
        e, "views", "view_count", "viewCount", "play_count", "playCount", "video_views", "plays",
    )
    likes = _pick_int(
        e, "likes", "like_count", "likeCount", "reactions", "reaction_count", "total_reactions",
    )
    comments = _pick_int(e, "comments", "comment_count", "commentCount")
    shares = _pick_int(e, "shares", "share_count", "shareCount")
    ok = bool(e.get("success") is True)
    st = str(e.get("status") or "").strip().lower()
    if (not ok) and (st not in successful_statuses):
        if views or likes or comments or shares:
            pass
        else:
            return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    return {"views": views, "likes": likes, "comments": comments, "shares": shares}


def _catalog_title_and_metrics_from_upload_pr(
    platform_results_raw: Any,
    platform: str,
    platform_video_id: str,
) -> tuple[Optional[str], dict[str, int]]:
    """Title + per-video metrics from uploads.platform_results (sync-analytics / publish pipeline)."""
    entries = _normalize_upload_platform_results_list(platform_results_raw)
    plat = (platform or "").lower()
    plat_entries = [
        e
        for e in entries
        if isinstance(e, dict) and str(e.get("platform") or "").lower() == plat
    ]
    for e in plat_entries:
        if _pr_entry_matches_catalog_video(e, plat, platform_video_id):
            t = _platform_result_title_from_entry(e)
            m = _catalog_engagement_from_pr_entry(e)
            return t, m
    # One publish result for this platform — use it even if catalog video id drifted (IDs can mismatch).
    if len(plat_entries) == 1:
        e = plat_entries[0]
        t = _platform_result_title_from_entry(e)
        m = _catalog_engagement_from_pr_entry(e)
        return t, m
    return None, {"views": 0, "likes": 0, "comments": 0, "shares": 0}


def _engagement_from_single_platform_result_entry(e: dict) -> dict[str, int]:
    """Metrics for one platform_results element (same field keys as rollup)."""
    if not isinstance(e, dict):
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}

    def _pick_int(d: dict, *keys: str) -> int:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return int(d[k] or 0)
                except Exception:
                    return 0
        return 0

    shortform_platforms = {"tiktok", "youtube", "instagram", "facebook"}
    successful_statuses = {"published", "succeeded", "success", "completed", "partial"}
    plat = str(e.get("platform") or "").strip().lower()
    if plat and plat not in shortform_platforms:
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    ok = bool(e.get("success") is True)
    st = str(e.get("status") or "").strip().lower()
    if (not ok) and (st not in successful_statuses):
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    return {
        "views": _pick_int(e, "views", "view_count", "play_count", "playCount", "video_views"),
        "likes": _pick_int(e, "likes", "like_count", "likeCount"),
        "comments": _pick_int(e, "comments", "comment_count", "commentCount"),
        "shares": _pick_int(e, "shares", "share_count", "shareCount"),
    }


def _utc_day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


async def _refresh_platform_kpi_rollups_for_utc_day(conn, d: date) -> None:
    """
    Rebuild platform_kpi_rollups_daily for one UTC calendar day from uploads.
    Counts uploads by targeted platforms[]; engagement from platform_results with
    row-level fallback split across targets when results are empty.
    """
    day_start, day_end = _utc_day_bounds(d)
    await conn.execute("DELETE FROM platform_kpi_rollups_daily WHERE day = $1::date", d)
    rows = await conn.fetch(
        """
        SELECT platforms, platform_results, status, views, likes, comments, shares
          FROM uploads
         WHERE created_at >= $1 AND created_at < $2
        """,
        day_start,
        day_end,
    )
    from collections import defaultdict

    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "uploads_targeted": 0,
            "uploads_completed": 0,
            "views": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0,
        }
    )
    done_status = {"completed", "succeeded", "partial"}

    for r in rows:
        plats = [str(p).strip().lower() for p in (r.get("platforms") or []) if str(p).strip()]
        if not plats:
            plats = ["unknown"]
        for p in plats:
            agg[p]["uploads_targeted"] += 1
            if str(r.get("status") or "").lower() in done_status:
                agg[p]["uploads_completed"] += 1

        entries = _normalize_upload_platform_results_list(r.get("platform_results"))
        entry_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"views": 0, "likes": 0, "comments": 0, "shares": 0})
        for e in entries:
            if not isinstance(e, dict):
                continue
            pk = str(e.get("platform") or "").strip().lower() or "unknown"
            m = _engagement_from_single_platform_result_entry(e)
            for k in m:
                entry_metrics[pk][k] += int(m[k] or 0)

        roll_all = _rollup_engagement_from_platform_results(
            entries, shortform_only=True, successful_only=True,
        )
        row_totals = {
            "views": max(int(r.get("views") or 0), int(roll_all["views"] or 0)),
            "likes": max(int(r.get("likes") or 0), int(roll_all["likes"] or 0)),
            "comments": max(int(r.get("comments") or 0), int(roll_all["comments"] or 0)),
            "shares": max(int(r.get("shares") or 0), int(roll_all["shares"] or 0)),
        }
        entry_sum = {k: 0 for k in row_totals}
        for pm in entry_metrics.values():
            for k in row_totals:
                entry_sum[k] += int(pm.get(k) or 0)

        if sum(entry_sum.values()) > 0:
            for pk, pm in entry_metrics.items():
                for k in row_totals:
                    agg[pk][k] += int(pm.get(k) or 0)
        else:
            n = len(plats)
            if n <= 0:
                n = 1
            for k in row_totals:
                base, rem = divmod(int(row_totals[k] or 0), n)
                for i, p in enumerate(plats):
                    agg[p][k] += base + (1 if i < rem else 0)

    for platform, m in agg.items():
        await conn.execute(
            """
            INSERT INTO platform_kpi_rollups_daily
                (day, platform, uploads_targeted, uploads_completed, views, likes, comments, shares, updated_at)
            VALUES ($1::date, $2, $3, $4, $5, $6, $7, $8, NOW())
            """,
            d,
            platform[:64],
            m["uploads_targeted"],
            m["uploads_completed"],
            m["views"],
            m["likes"],
            m["comments"],
            m["shares"],
        )


async def _refresh_platform_kpi_rollups_for_utc_range(conn, start: datetime, end: datetime) -> None:
    """Refresh daily rollups for each UTC day touched by [start, end)."""
    if end <= start:
        return
    d = start.date()
    end_d = (end - timedelta(microseconds=1)).date() if end > start else start.date()
    while d <= end_d:
        await _refresh_platform_kpi_rollups_for_utc_day(conn, d)
        d = d + timedelta(days=1)


async def _fetch_platform_kpi_totals_between(
    conn, start: datetime, end: datetime,
) -> list:
    """Aggregate rollup rows for [start, end) in UTC (days fully or partially covered)."""
    day0 = start.date()
    day1 = (end - timedelta(microseconds=1)).date() if end > start else start.date()
    return await conn.fetch(
        """
        SELECT platform,
               SUM(uploads_targeted)::bigint  AS uploads_targeted,
               SUM(uploads_completed)::bigint AS uploads_completed,
               SUM(views)::bigint AS views,
               SUM(likes)::bigint AS likes,
               SUM(comments)::bigint AS comments,
               SUM(shares)::bigint AS shares
          FROM platform_kpi_rollups_daily
         WHERE day >= $1::date AND day <= $2::date
         GROUP BY platform
         ORDER BY uploads_targeted DESC, platform ASC
        """,
        day0,
        day1,
    )


async def _compute_upload_engagement_totals(
    conn,
    user_id: str,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    platform: Optional[str] = None,
) -> dict[str, int]:
    """
    Compute user-scoped engagement from upload rows, using per-upload rollups when
    DB columns are stale. This stays strictly upload-specific (no account-level bleed).
    ``platform`` filters to uploads whose platforms[] contains that slug (case-insensitive).
    """
    where = f"WHERE user_id = $1 AND status IN {SUCCESSFUL_STATUS_SQL_IN}"
    params: list[Any] = [user_id]
    if since is not None:
        where += f" AND created_at >= ${len(params) + 1}"
        params.append(since)
    if until is not None:
        where += f" AND created_at < ${len(params) + 1}"
        params.append(until)
    if platform:
        where += (
            f" AND EXISTS (SELECT 1 FROM unnest(COALESCE(platforms, ARRAY[]::text[])) AS _plat "
            f"WHERE lower(_plat::text) = ${len(params) + 1})"
        )
        params.append(platform)
    rows = await conn.fetch(
        f"""
        SELECT views, likes, comments, shares, platform_results
          FROM uploads
          {where}
        """,
        *params,
    )
    totals = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    for r in rows:
        pr = _normalize_upload_platform_results_list(r.get("platform_results"))
        roll = _rollup_engagement_from_platform_results(
            pr,
            shortform_only=True,
            successful_only=True,
        )
        totals["views"] += max(int(r.get("views") or 0), int(roll["views"] or 0))
        totals["likes"] += max(int(r.get("likes") or 0), int(roll["likes"] or 0))
        totals["comments"] += max(int(r.get("comments") or 0), int(roll["comments"] or 0))
        totals["shares"] += max(int(r.get("shares") or 0), int(roll["shares"] or 0))
    return totals


async def _user_upload_kpi_bundle(
    conn,
    user_id: str,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    platform: Optional[str] = None,
) -> dict[str, Any]:
    """
    Single source for user-scoped upload counts + engagement (matches GET /api/analytics).
    SUCCESSFUL_UPLOAD_STATUSES is documented in services.upload_metrics.
    """
    where = ["user_id = $1"]
    params: list[Any] = [user_id]
    if since is not None:
        where.append(f"created_at >= ${len(params) + 1}")
        params.append(since)
    if until is not None:
        where.append(f"created_at < ${len(params) + 1}")
        params.append(until)
    if platform:
        where.append(
            f"EXISTS (SELECT 1 FROM unnest(COALESCE(platforms, ARRAY[]::text[])) AS _plat "
            f"WHERE lower(_plat::text) = ${len(params) + 1})"
        )
        params.append(platform)
    wh = " AND ".join(where)
    q_full = f"""
        SELECT
            COUNT(*)::int AS total,
            SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS successful,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            SUM(CASE WHEN status IN ('pending','queued','processing','staged','scheduled','ready_to_publish')
                THEN 1 ELSE 0 END)::int AS in_queue,
            COALESCE(SUM(put_spent), 0)::int AS put_used,
            COALESCE(SUM(aic_spent), 0)::int AS aic_used
        FROM uploads
        WHERE {wh}
    """
    q_min = f"""
        SELECT
            COUNT(*)::int AS total,
            SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS successful,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
            SUM(CASE WHEN status IN ('pending','queued','processing','staged','scheduled','ready_to_publish')
                THEN 1 ELSE 0 END)::int AS in_queue,
            0::int AS put_used,
            0::int AS aic_used
        FROM uploads
        WHERE {wh}
    """
    try:
        row = await conn.fetchrow(q_full, *params)
    except Exception as e:
        if e.__class__.__name__ != "UndefinedColumnError":
            raise
        row = await conn.fetchrow(q_min, *params)
    eng = await _compute_upload_engagement_totals(
        conn, user_id, since=since, until=until, platform=platform
    )
    tot = int(row["total"] or 0)
    succ = int(row["successful"] or 0)
    return {
        "total": tot,
        "successful": succ,
        "failed": int(row["failed"] or 0),
        "in_queue": int(row["in_queue"] or 0),
        "success_rate_pct": round((succ / max(tot, 1)) * 100, 1) if tot else 0.0,
        "put_used": int(row["put_used"] or 0),
        "aic_used": int(row["aic_used"] or 0),
        "engagement": eng,
        "successful_statuses": list(SUCCESSFUL_UPLOAD_STATUSES),
    }


async def _enrich_platform_results(conn, upload_row: dict, user_id: str) -> list:
    """
    Return platform_results as a flat list. Each entry enriched with
    account_name/username/avatar.

    Priority:
      1. Fields already stored IN the entry (set by worker after Fix 2/3)
      2. target_accounts UUID → platform_tokens JOIN (for uploads before Fix 3)
      3. Primary account per platform (last resort for very old uploads)
    """
    raw = upload_row.get("platform_results") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []

    if isinstance(raw, dict):
        items = [{"platform": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    elif isinstance(raw, list):
        items = list(raw)
    else:
        items = []

    if not items:
        return items

    # If ALL successful entries already have full identity including avatar, skip DB join
    successful = [e for e in items if e.get("success") is not False]
    already_enriched = successful and all(
        (e.get("account_username") or e.get("account_name") or e.get("account_id"))
        and (e.get("account_avatar") or e.get("avatar"))
        for e in successful
    )
    if already_enriched:
        for e in items:
            if e.get("account_avatar"):
                e["account_avatar"] = _platform_account_avatar_to_url(e["account_avatar"])
            if e.get("avatar"):
                e["avatar"] = _platform_account_avatar_to_url(e["avatar"])
        return items

    # Build token_row_id → identity map from target_accounts
    token_map = {}
    target_ids = [str(t) for t in (upload_row.get("target_accounts") or []) if t]

    if target_ids:
        try:
            rows = await conn.fetch(
                """SELECT id, platform, account_id, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND id = ANY($2::uuid[]) AND revoked_at IS NULL""",
                user_id, target_ids
            )
            for r in rows:
                token_map[str(r["id"])] = {
                    "token_row_id":     str(r["id"]),
                    "account_id":       r["account_id"]       or "",
                    "account_name":     r["account_name"]     or "",
                    "account_username": r["account_username"] or "",
                    "account_avatar":   r["account_avatar"]   or "",
                    "platform":         r["platform"],
                }
        except Exception as e:
            logger.warning(f"_enrich_platform_results target lookup failed: {e}")

    # Fallback: primary token per platform for old uploads
    platform_fallback = {}
    if not token_map:
        try:
            platforms_needed = list({(e.get("platform") or "").lower() for e in items if e.get("platform")})
            if platforms_needed:
                rows = await conn.fetch(
                    """SELECT DISTINCT ON (platform)
                              id, platform, account_id, account_name, account_username, account_avatar
                       FROM platform_tokens
                       WHERE user_id = $1 AND platform = ANY($2::text[]) AND revoked_at IS NULL
                       ORDER BY platform, is_primary DESC NULLS LAST, updated_at DESC""",
                    user_id, platforms_needed
                )
                for r in rows:
                    platform_fallback[r["platform"]] = {
                        "token_row_id":     str(r["id"]),
                        "account_id":       r["account_id"]       or "",
                        "account_name":     r["account_name"]     or "",
                        "account_username": r["account_username"] or "",
                        "account_avatar":   r["account_avatar"]   or "",
                    }
        except Exception as e:
            logger.warning(f"_enrich_platform_results fallback lookup failed: {e}")

    # Track which token_ids we've assigned to avoid giving same identity to multiple entries
    used_token_ids: set[str] = set()
    enriched = []
    for entry in items:
        p = (entry.get("platform") or "").lower()

        stored_token_id = entry.get("token_row_id") or ""
        if stored_token_id and stored_token_id in token_map:
            acct = token_map[stored_token_id]
        elif token_map:
            # Multi-account: pick first unused token for this platform so each entry gets distinct identity
            candidates = [v for v in token_map.values() if v.get("platform") == p and v.get("token_row_id") not in used_token_ids]
            acct = candidates[0] if candidates else next((v for v in token_map.values() if v.get("platform") == p), {})
            if acct.get("token_row_id"):
                used_token_ids.add(acct["token_row_id"])
        else:
            acct = platform_fallback.get(p, {})

        merged = {**entry}
        for field in ("token_row_id", "account_id", "account_name", "account_username", "account_avatar"):
            if not merged.get(field) and acct.get(field):
                merged[field] = acct[field]

        if merged.get("account_avatar"):
            merged["account_avatar"] = _platform_account_avatar_to_url(merged["account_avatar"])
        if merged.get("avatar"):
            merged["avatar"] = _platform_account_avatar_to_url(merged["avatar"])

        enriched.append(merged)

    return enriched


async def _batch_enrich_platform_results(conn, rows: list, user_id: str) -> dict:
    """Pre-fetch all platform_tokens needed by a batch of upload rows in ONE query.
    Returns {upload_id: enriched_list}. Eliminates N+1 DB calls."""
    all_target_ids: set[str] = set()
    all_platforms: set[str] = set()
    per_upload_raw: dict[str, list] = {}

    for r in rows:
        uid = str(r.get("id") or r["id"])
        raw = r.get("platform_results") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        if isinstance(raw, dict):
            items = [{"platform": k, **v} for k, v in raw.items() if isinstance(v, dict)]
        elif isinstance(raw, list):
            items = list(raw)
        else:
            items = []

        per_upload_raw[uid] = items

        for tid in (r.get("target_accounts") or []):
            if tid:
                all_target_ids.add(str(tid))
        for e in items:
            p = (e.get("platform") or "").lower()
            if p:
                all_platforms.add(p)

    token_map: dict[str, dict] = {}
    if all_target_ids:
        try:
            tok_rows = await conn.fetch(
                """SELECT id, platform, account_id, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND id = ANY($2::uuid[]) AND revoked_at IS NULL""",
                user_id, list(all_target_ids)
            )
            for tr in tok_rows:
                token_map[str(tr["id"])] = {
                    "token_row_id":     str(tr["id"]),
                    "account_id":       tr["account_id"]       or "",
                    "account_name":     tr["account_name"]     or "",
                    "account_username": tr["account_username"] or "",
                    "account_avatar":   tr["account_avatar"]   or "",
                    "platform":         tr["platform"],
                }
        except Exception as e:
            logger.warning(f"_batch_enrich target lookup: {e}")

    platform_fallback: dict[str, dict] = {}
    if not token_map and all_platforms:
        try:
            fb_rows = await conn.fetch(
                """SELECT DISTINCT ON (platform)
                          id, platform, account_id, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND platform = ANY($2::text[]) AND revoked_at IS NULL
                   ORDER BY platform, is_primary DESC NULLS LAST, updated_at DESC""",
                user_id, list(all_platforms)
            )
            for r in fb_rows:
                platform_fallback[r["platform"]] = {
                    "token_row_id":     str(r["id"]),
                    "account_id":       r["account_id"]       or "",
                    "account_name":     r["account_name"]     or "",
                    "account_username": r["account_username"] or "",
                    "account_avatar":   r["account_avatar"]   or "",
                }
        except Exception as e:
            logger.warning(f"_batch_enrich fallback lookup: {e}")

    result: dict[str, list] = {}
    for uid, items in per_upload_raw.items():
        if not items:
            result[uid] = items
            continue

        successful = [e for e in items if e.get("success") is not False]
        already_enriched = successful and all(
            (e.get("account_username") or e.get("account_name") or e.get("account_id"))
            and (e.get("account_avatar") or e.get("avatar"))
            for e in successful
        )
        if already_enriched:
            for e in items:
                if e.get("account_avatar"):
                    e["account_avatar"] = _platform_account_avatar_to_url(e["account_avatar"])
                if e.get("avatar"):
                    e["avatar"] = _platform_account_avatar_to_url(e["avatar"])
            result[uid] = items
            continue

        used_token_ids: set[str] = set()
        enriched = []
        for entry in items:
            p = (entry.get("platform") or "").lower()
            stored_token_id = entry.get("token_row_id") or ""
            if stored_token_id and stored_token_id in token_map:
                acct = token_map[stored_token_id]
            elif token_map:
                candidates = [v for v in token_map.values() if v.get("platform") == p and v.get("token_row_id") not in used_token_ids]
                acct = candidates[0] if candidates else next((v for v in token_map.values() if v.get("platform") == p), {})
                if acct.get("token_row_id"):
                    used_token_ids.add(acct["token_row_id"])
            else:
                acct = platform_fallback.get(p, {})

            merged = {**entry}
            for field in ("token_row_id", "account_id", "account_name", "account_username", "account_avatar"):
                if not merged.get(field) and acct.get(field):
                    merged[field] = acct[field]
            if merged.get("account_avatar"):
                merged["account_avatar"] = _platform_account_avatar_to_url(merged["account_avatar"])
            if merged.get("avatar"):
                merged["avatar"] = _platform_account_avatar_to_url(merged["avatar"])
            enriched.append(merged)

        result[uid] = enriched

    return result


@app.get("/api/uploads")
async def get_uploads(
    status: Optional[str] = None,
    view: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    cursor: Optional[str] = None,
    trill_only: bool = False,
    meta: bool = False,
    user: dict = Depends(get_current_user),
):
    """
    Upload queue list for current user.

    Filter by status (exact) or view (semantic group):
      view=pending   → status IN (pending, staged, queued, scheduled, ready_to_publish) — waiting uploads incl. smart/scheduled
      view=processing → status = processing
      view=completed → status IN (completed, succeeded, partial)
      view=failed   → status = failed
      view=staged   → same as pending (alias for Staged/Pending filter)
      view=smart_schedule → schedule_mode='smart' AND status IN pending group

    Contract (frontend-safe):
      - status_label: human-readable label for display (fixes ? succeeded, ? staged)
      - thumbnail_url, platform_results, hashtags, etc.
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
        "schedule_mode","schedule_metadata",
        "video_url",
        "ai_title","ai_caption",
        "ai_generated_title","ai_generated_caption","ai_generated_hashtags",
        "target_accounts",
    ]
    select_cols = _pick_cols(wanted, cols) or ["id","filename","platforms","status","created_at"]
    select_sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE user_id = $1"
    count_sql = "SELECT COUNT(*) FROM uploads WHERE user_id = $1"
    params = [user["id"]]
    count_params = [user["id"]]

    # view takes precedence over status for semantic filtering
    # view=all: no status filter — show all uploads
    if view == "all":
        pass  # no status filter
    elif view and view in _UPLOAD_VIEW_STATUS:
        statuses = _UPLOAD_VIEW_STATUS[view]
        if statuses is not None:
            placeholders = ", ".join(f"${i}" for i in range(len(params) + 1, len(params) + 1 + len(statuses)))
            params.extend(statuses)
            count_params.extend(statuses)
            select_sql += f" AND status IN ({placeholders})"
            count_sql += f" AND status IN ({placeholders})"
        else:
            # smart_schedule: schedule_mode='smart' AND status IN pending group
            pending = _UPLOAD_VIEW_STATUS["pending"]
            ph = ", ".join(f"${i}" for i in range(len(params) + 1, len(params) + 1 + len(pending)))
            params.extend(pending)
            count_params.extend(pending)
            select_sql += f" AND schedule_mode = 'smart' AND status IN ({ph})"
            count_sql += f" AND schedule_mode = 'smart' AND status IN ({ph})"
    elif status:
        params.append(status)
        count_params.append(status)
        select_sql += f" AND status = ${len(params)}"
        count_sql += f" AND status = ${len(count_params)}"

    if trill_only and "trill_score" in cols:
        select_sql += " AND trill_score IS NOT NULL"
        count_sql += " AND trill_score IS NOT NULL"

    cursor_ts = None
    cursor_id = None
    if cursor:
        try:
            parts = base64.b64decode(cursor).decode().split("|", 1)
            cursor_ts = datetime.fromisoformat(parts[0])
            cursor_id = parts[1] if len(parts) > 1 else None
        except Exception as e:
            logger.debug("uploads list: invalid cursor ignored: %s", e)

    if cursor_ts and cursor_id:
        params.append(cursor_ts)
        params.append(cursor_id)
        select_sql += f" AND (created_at, id) < (${len(params)-1}, ${len(params)}::uuid)"
    elif cursor_ts:
        params.append(cursor_ts)
        select_sql += f" AND created_at < ${len(params)}"

    params.append(limit)
    select_sql += f" ORDER BY created_at DESC, id DESC LIMIT ${len(params)}"
    if not cursor:
        params.append(offset)
        select_sql += f" OFFSET ${len(params)}"

    def _normalize_hashtags(raw):
        tags = _safe_json(raw, [])
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        if isinstance(tags, str) and tags.strip():
            return [tags.strip()]
        return []

    s3 = None
    out = []
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(select_sql, *params)
        total = await conn.fetchval(count_sql, *count_params) if meta else None

        row_dicts = [dict(r) for r in rows]

        enriched_map = await _batch_enrich_platform_results(conn, row_dicts, str(user["id"]))

        thumb_keys = [(i, d.get("thumbnail_r2_key")) for i, d in enumerate(row_dicts)]
        thumb_urls: dict[int, str | None] = {}
        keys_to_sign = [(i, _normalize_r2_key(k)) for i, k in thumb_keys if k]
        if keys_to_sign:
            try:
                s3 = get_s3_client()
                for i, nk in keys_to_sign:
                    try:
                        thumb_urls[i] = s3.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": R2_BUCKET_NAME, "Key": nk},
                            ExpiresIn=3600,
                        )
                    except Exception as e:
                        logger.debug("uploads list: thumbnail presign failed idx=%s: %s", i, e)
                        thumb_urls[i] = None
            except Exception as e:
                logger.debug("uploads list: batch thumbnail presign unavailable: %s", e)

        for idx, d in enumerate(row_dicts):
            ai_title = (d.get("ai_title") or d.get("ai_generated_title") or "") or ""
            ai_caption = (d.get("ai_caption") or d.get("ai_generated_caption") or "") or ""
            ai_hashtags = _normalize_hashtags(d.get("ai_generated_hashtags"))

            title = (d.get("title") or "").strip() or ai_title
            caption = (d.get("caption") or "").strip() or ai_caption

            hashtags = _normalize_hashtags(d.get("hashtags"))
            platform_results = enriched_map.get(str(d.get("id")), [])
            rollup = _rollup_engagement_from_platform_results(platform_results)
            v_row = int(d.get("views") or 0)
            l_row = int(d.get("likes") or 0)
            c_row = int(d.get("comments") or 0)
            s_row = int(d.get("shares") or 0)

            raw_status = d.get("status") or ""
            item = {
                "id": str(d.get("id")),
                "filename": d.get("filename"),
                "platforms": list(d.get("platforms") or []),
                "status": raw_status,
                "status_label": _STATUS_LABEL.get(raw_status, raw_status.replace("_", " ").title() if raw_status else "Unknown"),
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

                "thumbnail_url": thumb_urls.get(idx),
                "platform_results": platform_results,

                "file_size": d.get("file_size"),
                "views":    max(v_row, rollup["views"]),
                "likes":    max(l_row, rollup["likes"]),
                "comments": max(c_row, rollup["comments"]),
                "shares":   max(s_row, rollup["shares"]),

                "progress": int(d.get("processing_progress") or 0),
                "current_stage": d.get("processing_stage"),

                "schedule_mode":     d.get("schedule_mode") or "immediate",
                "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
                "smart_schedule":    _safe_json(d.get("schedule_metadata"), None),

                "is_editable": d.get("status") in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),

                "video_url": d.get("video_url"),
            }
            out.append(item)

    if not meta:
        return out

    next_cursor = None
    if out:
        last = row_dicts[-1]
        last_ts = last.get("created_at")
        last_id = str(last.get("id") or "")
        if last_ts:
            raw_cursor = f"{last_ts.isoformat()}|{last_id}"
            next_cursor = base64.b64encode(raw_cursor.encode()).decode()

    return {
        "uploads": out,
        "total": int(total or 0),
        "limit": limit,
        "offset": offset,
        "next_cursor": next_cursor,
    }


@app.get("/api/uploads/queue-stats")
async def get_uploads_queue_stats(user: dict = Depends(get_current_user)):
    """
    Queue summary counts for queue.html and dashboard.html.
    Use these counts for Pending, Processing, Completed, Failed cards.
    Pending includes staged, queued, scheduled, ready_to_publish (smart + scheduled).
    """
    pending_statuses = _UPLOAD_VIEW_STATUS["pending"]
    completed_statuses = _UPLOAD_VIEW_STATUS["completed"]
    n_p, n_c = len(pending_statuses), len(completed_statuses)
    ph_p = ", ".join(f"${i}" for i in range(2, 2 + n_p))
    ph_c = ", ".join(f"${i}" for i in range(2 + n_p, 2 + n_p + n_c))
    params = [user["id"]] + list(pending_statuses) + list(completed_statuses)

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT
                SUM(CASE WHEN status IN ({ph_p}) THEN 1 ELSE 0 END)::int AS pending,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END)::int AS processing,
                SUM(CASE WHEN status IN ({ph_c}) THEN 1 ELSE 0 END)::int AS completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed
            FROM uploads WHERE user_id = $1
            """,
            *params,
        )
    return {
        "pending": row["pending"] or 0,
        "processing": row["processing"] or 0,
        "completed": row["completed"] or 0,
        "failed": row["failed"] or 0,
    }


@app.get("/api/scheduled")
async def get_scheduled(user: dict = Depends(get_current_user)):
    """Get scheduled uploads (scheduled + smart modes, all pending statuses)"""
    async with db_pool.acquire() as conn:
        uploads = await conn.fetch("""
            SELECT * FROM uploads
            WHERE user_id = $1
              AND status IN ('pending', 'queued', 'scheduled', 'staged', 'ready_to_publish')
            ORDER BY scheduled_time ASC NULLS LAST, created_at ASC
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
        except Exception as e:
            logger.debug("regenerate thumbnail: existing presign failed, regenerating: %s", e)

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
        except Exception as e:
            logger.debug("regenerate thumbnail: ffprobe duration fallback offset=5s: %s", e)
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


@app.post("/api/uploads/{upload_id}/thumbnail-presign")
async def presign_thumbnail_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Get a presigned URL for uploading a custom thumbnail.
    After uploading, call PATCH /api/uploads/{upload_id} with thumbnail_r2_key in the body (if supported).
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
    if not row:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if row["status"] not in editable:
        raise HTTPException(400, "Cannot change thumbnail after upload is processing or published")

    thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/custom.jpg"
    presigned_url = generate_presigned_upload_url(thumb_r2_key, "image/jpeg")
    return {"presigned_url": presigned_url, "r2_key": thumb_r2_key}


def _extract_platform_video_id_from_url(platform: str, raw_url: str) -> str:
    """Best-effort extraction of platform post/video IDs from canonical URLs."""
    if not raw_url:
        return ""
    try:
        url = str(raw_url).strip()
        if not url:
            return ""
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        q = dict(parse_qsl(parsed.query or ""))
        p = str(platform or "").strip().lower()

        if p == "youtube" or "youtu" in host:
            if host.endswith("youtu.be"):
                m = re.match(r"^/([A-Za-z0-9_-]{6,})", path)
                if m:
                    return m.group(1)
            if q.get("v"):
                return str(q.get("v"))
            m = re.search(r"/(?:shorts|embed|watch)/([A-Za-z0-9_-]{6,})", path)
            if m:
                return m.group(1)
            m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", path)
            if m:
                return m.group(1)

        if p == "tiktok" or "tiktok.com" in host:
            m = re.search(r"/video/(\d+)", path)
            if m:
                return m.group(1)

        if p == "facebook" or "facebook.com" in host or "fb.watch" in host:
            m = re.search(r"/videos/(?:[^/]+/)?(\d+)", path)
            if m:
                return m.group(1)
            if q.get("v"):
                return str(q.get("v"))

        if p == "instagram" or "instagram.com" in host:
            # Often shortcode, but still useful when present in stored URLs.
            m = re.search(r"/(?:reel|p|tv)/([^/?#]+)", path)
            if m:
                return m.group(1)
    except Exception:
        return ""
    return ""


@app.post("/api/uploads/{upload_id}/sync-analytics")
async def sync_upload_analytics(
    upload_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Fetch latest engagement stats for a single completed upload from platform APIs.
    Uses the video IDs stored in platform_results to query per-video metrics.
    Updates the uploads table (views, likes, comments, shares) and returns fresh data.
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platforms, platform_results, status, title, caption, filename, views, likes, comments, shares, analytics_synced_at FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if row["status"] not in ("completed", "succeeded", "partial"):
        return {
            "synced": False,
            "reason": "not_completed",
            "message": "Stats sync is only available after the upload has finished processing successfully.",
            "views": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0,
        }

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
        if not pr.get("video_id"):
            plat = str(pr.get("platform") or "").strip().lower()
            for u in (
                pr.get("platform_url"),
                pr.get("url"),
                pr.get("permalink"),
                pr.get("link"),
                pr.get("share_url"),
            ):
                inferred = _extract_platform_video_id_from_url(plat, str(u or ""))
                if inferred:
                    pr["video_id"] = inferred
                    if not pr.get("platform_video_id"):
                        pr["platform_video_id"] = inferred
                    break

    # Get tokens for all connected platforms (include id for multi-account lookup)
    async with db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )

    token_map_by_id = {}
    token_map_by_platform = {}
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if dec:
                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                    dec["ig_user_id"] = str(tr["account_id"])
                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                    dec["page_id"] = str(tr["account_id"])
                token_id = str(tr["id"])
                token_map_by_id[token_id] = dec
                token_map_by_platform.setdefault(tr["platform"], []).append(dec)
        except Exception as e:
            logger.debug("sync-analytics: token decrypt/map skip row id=%s: %s", tr.get("id"), e)

    from services.sync_analytics_helpers import (
        build_plat_account_token_map,
        resolve_token_candidates_for_platform_result,
    )

    token_map_by_plat_account = build_plat_account_token_map(token_rows, decrypt_blob)

    # Refresh OAuth for every connected row for platforms present on this upload (multi-account safe).
    # Keep this bounded so manual sync cannot stall for long.
    uid_str = str(user["id"])
    plats_in_upload = {str(pr.get("platform") or "").lower() for pr in pr_list}
    try:
        from stages.publish_stage import _refresh_tiktok_token, _refresh_youtube_token, _refresh_meta_token

        refresh_sem = asyncio.Semaphore(4)

        async def _refresh_token_row(tr: Any) -> None:
            plat = str(tr.get("platform") or "").lower()
            if plat not in plats_in_upload:
                return
            try:
                dec = decrypt_blob(tr["token_blob"])
                if not dec:
                    return
            except Exception:
                return
            async with refresh_sem:
                try:
                    if plat == "tiktok":
                        await _refresh_tiktok_token(dict(dec), db_pool=db_pool, user_id=uid_str)
                    elif plat == "youtube":
                        await _refresh_youtube_token(dict(dec), db_pool=db_pool, user_id=uid_str)
                    elif plat in ("instagram", "facebook"):
                        await _refresh_meta_token(dict(dec), platform=plat, db_pool=db_pool, user_id=uid_str)
                except Exception:
                    return

        await asyncio.wait_for(
            asyncio.gather(*[_refresh_token_row(tr) for tr in token_rows], return_exceptions=True),
            timeout=8.0,
        )
        async with db_pool.acquire() as conn:
            trs = await conn.fetch(
                "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
                user["id"],
            )
        token_map_by_id = {}
        token_map_by_platform = {}
        for tr in trs:
            try:
                dec = decrypt_blob(tr["token_blob"])
                if dec:
                    if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                        dec["ig_user_id"] = str(tr["account_id"])
                    if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                        dec["page_id"] = str(tr["account_id"])
                    token_map_by_id[str(tr["id"])] = dec
                    token_map_by_platform.setdefault(tr["platform"], []).append(dec)
            except Exception as e:
                logger.debug("sync-analytics: token remap after refresh skip id=%s: %s", tr.get("id"), e)
        token_map_by_plat_account = build_plat_account_token_map(trs, decrypt_blob)
    except Exception as _sync_ref_e:
        logger.warning(f"sync-analytics OAuth refresh: {_sync_ref_e}")

    total_views = total_likes = total_comments = total_shares = 0
    resolved_platform_count = 0
    platform_stats = {}
    upload_id_str = str(upload_id or "")
    enable_profile_scan = str(os.environ.get("ENABLE_ANALYTICS_PROFILE_SCAN", "0") or "0").strip() == "1"

    def _norm_text(v: Any) -> str:
        s = str(v or "").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    match_phrases = []
    for raw in (row.get("title"), row.get("caption"), row.get("filename")):
        n = _norm_text(raw)
        if len(n) >= 6:
            match_phrases.append(n)
    upload_title_norm = _norm_text(row.get("title"))
    match_keywords = set()
    for p in match_phrases:
        for w in p.split():
            if len(w) >= 5:
                match_keywords.add(w)
    upload_id_compact = re.sub(r"[^a-z0-9]", "", upload_id_str.lower())
    upload_id_needles = set(
        x for x in (
            upload_id_compact,
            upload_id_compact[:8] if len(upload_id_compact) >= 8 else "",
            upload_id_compact[-8:] if len(upload_id_compact) >= 8 else "",
        ) if x
    )

    def _candidate_match_meta(text: Any, *, candidate_id: Any = None, candidate_url: Any = None) -> dict[str, Any]:
        nt = _norm_text(text)
        compact = re.sub(r"[^a-z0-9]", "", nt)
        score = 0
        upload_id_hit = False
        for n in upload_id_needles:
            if n and (n in compact or n in str(candidate_id or "").lower() or n in str(candidate_url or "").lower()):
                score += 7
                upload_id_hit = True
                break
        title_exact = False
        if upload_title_norm:
            title_exact = (nt == upload_title_norm) or (upload_title_norm in nt) or (nt in upload_title_norm)
        for ph in match_phrases:
            if ph and ph in nt:
                score += 4
                break
        kw_hits = 0
        for kw in match_keywords:
            if kw in nt:
                kw_hits += 1
                if kw_hits >= 4:
                    break
        score += kw_hits
        confidence = min(98, 40 + (score * 8))
        if title_exact or upload_id_hit:
            confidence = 99
        return {
            "score": int(score),
            "confidence": int(confidence),
            "title_exact": bool(title_exact),
            "upload_id_hit": bool(upload_id_hit),
        }

    def _candidate_is_accepted(meta: dict[str, Any]) -> bool:
        # When an upload has a title, require near-certain title-level match.
        if upload_title_norm:
            return bool(meta.get("title_exact")) and int(meta.get("confidence") or 0) >= 99
        # Titleless uploads: still require strong evidence.
        return int(meta.get("confidence") or 0) >= 90

    def _accum_platform_stats(pstat: dict, p: str, s: dict) -> None:
        if p not in pstat:
            pstat[p] = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
        for k in ("views", "likes", "comments", "shares"):
            pstat[p][k] = int(pstat[p].get(k, 0)) + int(s.get(k, 0) or 0)

    pr_list_meta_dirty = False
    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            ok_statuses = {"published", "succeeded", "success", "completed", "accepted", "live", "verified"}
            ok = (pr.get("success") is True) or (str(pr.get("status", "")).lower() in ok_statuses)
            if not ok:
                continue

            # platform_video_id is the canonical field written by db.py/mark_processing_completed
            # video_id / media_id / share_id etc. are legacy / webhook-written variants
            video_id = (
                pr.get("platform_video_id")  # canonical (worker pipeline)
                or pr.get("video_id") or pr.get("videoId") or pr.get("id")
                or pr.get("media_id") or pr.get("post_id") or pr.get("share_id")
            )

            try:
                candidates = resolve_token_candidates_for_platform_result(
                    pr, token_map_by_id, token_map_by_plat_account, token_map_by_platform
                )
                if not candidates:
                    continue

                # Rehydrate canonical permalinks from Graph (fixes legacy bad synthetic URLs on IG/FB).
                if plat == "instagram" and video_id:
                    mid = str(video_id).strip()
                    if mid.isdigit():
                        for _tok in (candidates or [])[:1]:
                            at = str(_tok.get("access_token") or "").strip()
                            if not at:
                                continue
                            try:
                                hr = await client.get(
                                    f"https://graph.facebook.com/v21.0/{mid}",
                                    params={"access_token": at, "fields": "permalink,shortcode"},
                                )
                                if hr.status_code == 200:
                                    hj = hr.json() or {}
                                    new_url = hj.get("permalink")
                                    if not new_url and hj.get("shortcode"):
                                        new_url = f"https://www.instagram.com/p/{hj['shortcode']}/"
                                    old_u = (pr.get("platform_url") or pr.get("url") or "").strip()
                                    if new_url and new_url != old_u:
                                        pr["platform_url"] = new_url
                                        pr["url"] = new_url
                                        pr_list_meta_dirty = True
                            except Exception:
                                pass
                            break
                elif plat == "facebook" and video_id:
                    fvid = str(video_id).strip()
                    if fvid.isdigit():
                        for _tok in (candidates or [])[:1]:
                            at = str(_tok.get("access_token") or "").strip()
                            if not at:
                                continue
                            try:
                                hr = await client.get(
                                    f"https://graph.facebook.com/v21.0/{fvid}",
                                    params={"access_token": at, "fields": "permalink_url"},
                                )
                                if hr.status_code == 200:
                                    new_url = (hr.json() or {}).get("permalink_url")
                                    old_u = (pr.get("platform_url") or pr.get("url") or "").strip()
                                    if new_url and new_url != old_u:
                                        pr["platform_url"] = new_url
                                        pr["url"] = new_url
                                        pr_list_meta_dirty = True
                            except Exception:
                                pass
                            break

                resolved = False
                for tok in (candidates or [])[:3]:
                    access_token = tok.get("access_token", "")
                    if not access_token:
                        continue

                    if plat == "tiktok" and video_id:
                        from services.tiktok_api import tiktok_envelope_error, tiktok_video_query_url

                        qurl = tiktok_video_query_url()
                        resp = await client.post(
                            qurl,
                            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                            json={"filters": {"video_ids": [str(video_id)]}},
                        )
                        try:
                            body = resp.json()
                        except Exception:
                            body = {}
                        if tiktok_envelope_error(body):
                            continue
                        if resp.status_code == 200:
                            vids = body.get("data", {}).get("videos", []) or []
                            if vids:
                                v = vids[0]
                                s = {
                                    "views": int(v.get("view_count") or 0),
                                    "likes": int(v.get("like_count") or 0),
                                    "comments": int(v.get("comment_count") or 0),
                                    "shares": int(v.get("share_count") or 0),
                                }
                                _accum_platform_stats(platform_stats, "tiktok", s)
                                total_views    += s["views"];    total_likes   += s["likes"]
                                total_comments += s["comments"]; total_shares  += s["shares"]
                                resolved = True
                                resolved_platform_count += 1
                                break

                    elif plat == "youtube" and video_id:
                        # statistics covers long-form and Shorts; snippet/contentDetails optional for debugging
                        resp = await client.get(
                            "https://www.googleapis.com/youtube/v3/videos",
                            params={"part": "statistics,snippet", "id": str(video_id)},
                            headers={"Authorization": f"Bearer {access_token}"},
                        )
                        if resp.status_code == 200:
                            items = resp.json().get("items", []) or []
                            if items:
                                st = items[0].get("statistics", {}) or {}
                                s = {
                                    "views": int(st.get("viewCount") or 0),
                                    "likes": int(st.get("likeCount") or 0),
                                    "comments": int(st.get("commentCount") or 0),
                                    "shares": 0,
                                }
                                _accum_platform_stats(platform_stats, "youtube", s)
                                total_views    += s["views"];    total_likes   += s["likes"]
                                total_comments += s["comments"]
                                resolved = True
                                resolved_platform_count += 1
                                break

                    elif plat == "instagram" and video_id:
                        # Instagram Insights API requires numeric media_id (not shortcode)
                        media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
                        resp = await client.get(
                            f"https://graph.facebook.com/v21.0/{media_id}/insights",
                            params={
                                "access_token": access_token,
                                "metric": "views,plays,likes,comments,saved,shares,reach",
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json().get("data", []) or []
                            if not data:
                                continue  # probably no permission for this token
                            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                            ig_views = ig_plays = 0
                            for m in data:
                                name = m.get("name", "")
                                vals = m.get("values", [])
                                val = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                                if name == "views":       ig_views     = val
                                elif name == "plays":     ig_plays     = val  # deprecated fallback
                                elif name == "likes":     s["likes"]   += val
                                elif name == "comments":  s["comments"] += val
                                elif name == "shares":    s["shares"]  += val
                            s["views"] = ig_views or ig_plays  # prefer views over deprecated plays
                            _accum_platform_stats(platform_stats, "instagram", s)
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]; total_shares  += s["shares"]
                            resolved = True
                            resolved_platform_count += 1
                            break
                        # Fallback when insights scope is missing: basic media fields.
                        media_resp = await client.get(
                            f"https://graph.facebook.com/v21.0/{media_id}",
                            params={
                                "access_token": access_token,
                                "fields": "media_type,video_view_count,play_count,like_count,comments_count",
                            },
                        )
                        if media_resp.status_code == 200:
                            md = media_resp.json() or {}
                            mtype = str(md.get("media_type") or "").upper()
                            if mtype in ("VIDEO", "REELS"):
                                s = {
                                    "views": int(md.get("video_view_count") or md.get("play_count") or 0),
                                    "likes": int(md.get("like_count") or 0),
                                    "comments": int(md.get("comments_count") or 0),
                                    "shares": 0,
                                }
                                _accum_platform_stats(platform_stats, "instagram", s)
                                total_views += s["views"]; total_likes += s["likes"]
                                total_comments += s["comments"]; total_shares += s["shares"]
                                resolved = True
                                resolved_platform_count += 1
                                break

                    elif plat == "facebook" and video_id:
                        resp = await client.get(
                            f"https://graph.facebook.com/v21.0/{video_id}",
                            params={
                                "access_token": access_token,
                                "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)",
                            },
                        )
                        if resp.status_code == 200:
                            data = (resp.json().get("insights", {}) or {}).get("data", []) or []
                            if not data:
                                continue
                            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                            for m in data:
                                name = m.get("name", "")
                                vals = m.get("values", [{}])
                                val  = vals[-1].get("value", 0) if vals else 0
                                if isinstance(val, dict):
                                    val = sum(val.values())
                                val = int(val or 0)
                                if name == "total_video_views":                      s["views"]    += val
                                elif name == "total_video_reactions_by_type_total":  s["likes"]    += val
                                elif name == "total_video_comments":                  s["comments"] += val
                                elif name == "total_video_shares":                    s["shares"]   += val
                            _accum_platform_stats(platform_stats, "facebook", s)
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]; total_shares  += s["shares"]
                            resolved = True
                            resolved_platform_count += 1
                            break
                        # Fallback without insights scope: basic counters.
                        fb_basic = await client.get(
                            f"https://graph.facebook.com/v21.0/{video_id}",
                            params={
                                "access_token": access_token,
                                "fields": "views,reactions.summary(true),comments.summary(true),shares",
                            },
                        )
                        if fb_basic.status_code == 200:
                            bd = fb_basic.json() or {}
                            s = {
                                "views": int(bd.get("views") or 0),
                                "likes": int(((bd.get("reactions") or {}).get("summary") or {}).get("total_count") or 0),
                                "comments": int(((bd.get("comments") or {}).get("summary") or {}).get("total_count") or 0),
                                "shares": int(((bd.get("shares") or {}).get("count")) or 0),
                            }
                            _accum_platform_stats(platform_stats, "facebook", s)
                            total_views += s["views"]; total_likes += s["likes"]
                            total_comments += s["comments"]; total_shares += s["shares"]
                            resolved = True
                            resolved_platform_count += 1
                            break

                    # TikTok video-list fallback: always active when no video_id is known.
                    # TikTok's Content Posting API processes uploads asynchronously, so
                    # verify_stage often can't capture the video_id in time. We match by
                    # title and save the id back so future syncs work without a list scan.
                    if (not resolved) and (not video_id) and plat == "tiktok":
                        from services.tiktok_api import tiktok_envelope_error, tiktok_video_list_url
                        lurl = tiktok_video_list_url()
                        lresp = await client.post(
                            lurl,
                            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                            json={"max_count": 20},
                        )
                        try:
                            lbody = lresp.json() if lresp.content else {}
                        except Exception:
                            lbody = {}
                        if lresp.status_code == 200 and not tiktok_envelope_error(lbody):
                            vids = ((lbody.get("data") or {}).get("videos") or [])
                            best = None
                            best_meta = None
                            best_score = -1
                            for v in vids:
                                cid = str(v.get("id") or "")
                                ctext = f"{v.get('title') or v.get('video_description') or ''} {cid}"
                                meta = _candidate_match_meta(ctext, candidate_id=cid, candidate_url=f"https://www.tiktok.com/video/{cid}")
                                sscore = int(meta.get("score") or 0)
                                if sscore > best_score:
                                    best = v
                                    best_meta = meta
                                    best_score = sscore
                            if best and _candidate_is_accepted(best_meta or {}):
                                found_vid_id = str(best.get("id") or "").strip()
                                s = {
                                    "views": int(best.get("view_count") or 0),
                                    "likes": int(best.get("like_count") or 0),
                                    "comments": int(best.get("comment_count") or 0),
                                    "shares": int(best.get("share_count") or 0),
                                }
                                _accum_platform_stats(platform_stats, "tiktok", s)
                                total_views += s["views"]; total_likes += s["likes"]
                                total_comments += s["comments"]; total_shares += s["shares"]
                                # Persist discovered video_id back to platform_results so
                                # future sync-analytics calls use the direct query path.
                                if found_vid_id:
                                    pr["platform_video_id"] = found_vid_id
                                    pr["video_id"] = found_vid_id
                                    uname = (pr.get("account_username") or "").strip().lstrip("@")
                                    if uname:
                                        pr["platform_url"] = f"https://www.tiktok.com/@{uname}/video/{found_vid_id}"
                                    else:
                                        pr["platform_url"] = f"https://www.tiktok.com/video/{found_vid_id}"
                                    pr_list_meta_dirty = True
                                resolved = True
                                resolved_platform_count += 1
                                break

                    if (not resolved) and (not video_id) and enable_profile_scan and plat == "youtube":
                        sr = await client.get(
                            "https://www.googleapis.com/youtube/v3/search",
                            params={"part": "snippet", "forMine": "true", "type": "video", "order": "date", "maxResults": 25},
                            headers={"Authorization": f"Bearer {access_token}"},
                        )
                        if sr.status_code == 200:
                            items = sr.json().get("items", []) or []
                            best_id = ""
                            best_meta = None
                            best_score = -1
                            for it in items:
                                vid = (((it.get("id") or {}).get("videoId")) or "").strip()
                                sn = it.get("snippet") or {}
                                txt = f"{sn.get('title') or ''} {sn.get('description') or ''}"
                                surl = f"https://www.youtube.com/watch?v={vid}" if vid else ""
                                meta = _candidate_match_meta(txt, candidate_id=vid, candidate_url=surl)
                                sscore = int(meta.get("score") or 0)
                                if vid and sscore > best_score:
                                    best_id = vid
                                    best_meta = meta
                                    best_score = sscore
                            if best_id and _candidate_is_accepted(best_meta or {}):
                                vr = await client.get(
                                    "https://www.googleapis.com/youtube/v3/videos",
                                    params={"part": "statistics", "id": best_id},
                                    headers={"Authorization": f"Bearer {access_token}"},
                                )
                                if vr.status_code == 200:
                                    vitems = vr.json().get("items", []) or []
                                    if vitems:
                                        st = vitems[0].get("statistics", {}) or {}
                                        s = {
                                            "views": int(st.get("viewCount") or 0),
                                            "likes": int(st.get("likeCount") or 0),
                                            "comments": int(st.get("commentCount") or 0),
                                            "shares": 0,
                                        }
                                        _accum_platform_stats(platform_stats, "youtube", s)
                                        total_views += s["views"]; total_likes += s["likes"]; total_comments += s["comments"]
                                        resolved = True
                                        resolved_platform_count += 1
                                        break

                    if (not resolved) and (not video_id) and enable_profile_scan and plat == "instagram":
                        ig_uid = str(tok.get("ig_user_id") or tok.get("account_id") or "").strip()
                        if ig_uid:
                            mr = await client.get(
                                f"https://graph.facebook.com/v21.0/{ig_uid}/media",
                                params={"access_token": access_token, "fields": "id,caption,media_type,media_product_type,permalink,timestamp", "limit": 30},
                            )
                            if mr.status_code == 200:
                                mitems = mr.json().get("data", []) or []
                                best_media = None
                                best_meta = None
                                best_score = -1
                                for m in mitems:
                                    mtype = str(m.get("media_type") or "").upper()
                                    mprod = str(m.get("media_product_type") or "").upper()
                                    if mtype not in ("VIDEO", "REELS") and mprod != "REELS":
                                        continue
                                    mid = str(m.get("id") or "")
                                    murl = str(m.get("permalink") or "")
                                    txt = f"{m.get('caption') or ''} {murl}"
                                    meta = _candidate_match_meta(txt, candidate_id=mid, candidate_url=murl)
                                    sscore = int(meta.get("score") or 0)
                                    if sscore > best_score:
                                        best_media = m
                                        best_meta = meta
                                        best_score = sscore
                                if best_media and _candidate_is_accepted(best_meta or {}):
                                    media_id = str(best_media.get("id") or "")
                                    ir = await client.get(
                                        f"https://graph.facebook.com/v21.0/{media_id}/insights",
                                        params={"access_token": access_token, "metric": "views,plays,likes,comments,shares"},
                                    )
                                    if ir.status_code == 200:
                                        data = ir.json().get("data", []) or []
                                        if data:
                                            s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                                            for m in data:
                                                name = str(m.get("name") or "")
                                                vals = m.get("values", [])
                                                val = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                                                if name in ("views", "plays"): s["views"] = max(s["views"], val)
                                                elif name == "likes": s["likes"] += val
                                                elif name == "comments": s["comments"] += val
                                                elif name == "shares": s["shares"] += val
                                            _accum_platform_stats(platform_stats, "instagram", s)
                                            total_views += s["views"]; total_likes += s["likes"]
                                            total_comments += s["comments"]; total_shares += s["shares"]
                                            resolved = True
                                            resolved_platform_count += 1
                                            break

                    if (not resolved) and (not video_id) and enable_profile_scan and plat == "facebook":
                        page_id = str(tok.get("page_id") or tok.get("account_id") or "").strip()
                        if page_id:
                            fr = await client.get(
                                f"https://graph.facebook.com/v21.0/{page_id}/videos",
                                params={"access_token": access_token, "fields": "id,description,permalink_url,views,reactions.summary(true),comments.summary(true),shares,created_time", "limit": 25},
                            )
                            if fr.status_code == 200:
                                vids = fr.json().get("data", []) or []
                                best = None
                                best_meta = None
                                best_score = -1
                                for v in vids:
                                    vid = str(v.get("id") or "")
                                    vurl = str(v.get("permalink_url") or "")
                                    txt = f"{v.get('description') or ''} {vurl}"
                                    meta = _candidate_match_meta(txt, candidate_id=vid, candidate_url=vurl)
                                    sscore = int(meta.get("score") or 0)
                                    if sscore > best_score:
                                        best = v
                                        best_meta = meta
                                        best_score = sscore
                                if best and _candidate_is_accepted(best_meta or {}):
                                    s = {
                                        "views": int(best.get("views") or 0),
                                        "likes": int(((best.get("reactions") or {}).get("summary") or {}).get("total_count") or 0),
                                        "comments": int(((best.get("comments") or {}).get("summary") or {}).get("total_count") or 0),
                                        "shares": int(((best.get("shares") or {}).get("count")) or 0),
                                    }
                                    _accum_platform_stats(platform_stats, "facebook", s)
                                    total_views += s["views"]; total_likes += s["likes"]
                                    total_comments += s["comments"]; total_shares += s["shares"]
                                    resolved = True
                                    resolved_platform_count += 1
                                    break

                # If we couldn't resolve for any candidate token, keep zeros.
                if resolved:
                    continue

            except Exception as e:
                logger.warning(f"sync-analytics error for {plat}/{video_id}: {e}")
                continue

        if pr_list_meta_dirty:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET platform_results = $1::jsonb, updated_at = NOW()
                        WHERE id = $2 AND user_id = $3
                        """,
                        json.dumps(pr_list),
                        str(upload_id),
                        user["id"],
                    )
            except Exception as ex:
                logger.warning("sync-analytics: could not persist rehydrated platform URLs: %s", ex)

    # Do not zero-out existing analytics when no platform could be resolved.
    if int(resolved_platform_count or 0) <= 0:
        prev_views = int(row.get("views") or 0)
        prev_likes = int(row.get("likes") or 0)
        prev_comments = int(row.get("comments") or 0)
        prev_shares = int(row.get("shares") or 0)
        return {
            "synced": False,
            "reason": "no_platform_metrics_resolved",
            "message": (
                "Could not sync stats from any platform API. "
                "Check that the post is live, the correct account is connected in Platforms, "
                "and this upload has a platform video/post ID—or reconnect the account and try again."
            ),
            "views": prev_views,
            "likes": prev_likes,
            "comments": prev_comments,
            "shares": prev_shares,
            "platform_stats": platform_stats,
            "resolved_platforms": 0,
        }

    # Persist to DB
    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE uploads
               SET views=$1, likes=$2, comments=$3, shares=$4, analytics_synced_at=NOW(), updated_at=NOW()
               WHERE id=$5 AND user_id=$6""",
            total_views, total_likes, total_comments, total_shares,
            upload_id, user["id"],
        )
        # Also write fresh metrics back to platform_content_items so the canonical rollup
        # always has the latest per-platform values even before catalog sync runs.
        try:
            from services.catalog_sync import _upsert_content_item as _upsert_pci
            uid_str_pci = str(user["id"])
            for pr_entry in pr_list:
                p_plat = str(pr_entry.get("platform") or "").strip().lower()
                p_vid = (
                    str(pr_entry.get("platform_video_id") or "")
                    or str(pr_entry.get("video_id") or "")
                ).strip()
                p_acct = str(pr_entry.get("account_id") or "").strip()
                p_tok = str(pr_entry.get("token_row_id") or "").strip()
                if not (p_plat and p_vid and p_acct and p_tok):
                    continue
                ps = platform_stats.get(p_plat)
                if not ps:
                    continue
                await _upsert_pci(
                    conn,
                    user_id=uid_str_pci,
                    platform_token_id=p_tok,
                    platform=p_plat,
                    account_id=p_acct,
                    platform_video_id=p_vid,
                    source="sync_analytics",
                    platform_url=str(pr_entry.get("platform_url") or pr_entry.get("url") or "") or None,
                    views=int(ps.get("views") or 0),
                    likes=int(ps.get("likes") or 0),
                    comments=int(ps.get("comments") or 0),
                    shares=int(ps.get("shares") or 0),
                )
        except Exception as _pci_ex:
            logger.warning("sync-analytics: platform_content_items upsert failed: %s", _pci_ex)

    async def _refresh_account_platform_cache():
        try:
            from services.platform_metrics_job import refresh_platform_metrics_for_user

            await refresh_platform_metrics_for_user(db_pool, user["id"])
        except Exception as ex:
            logger.warning(f"post-sync platform_metrics_cache refresh: {ex}")

    background_tasks.add_task(_refresh_account_platform_cache)

    return {
        "synced": True,
        "views": total_views, "likes": total_likes,
        "comments": total_comments, "shares": total_shares,
        "platform_stats": platform_stats,
        "resolved_platforms": int(resolved_platform_count or 0),
    }


@app.api_route("/api/uploads/sync-analytics/all", methods=["POST", "GET"])
async def sync_all_upload_analytics(
    background_tasks: BackgroundTasks,
    max_uploads: int = Query(500, ge=1, le=5000),
    async_mode: bool = Query(True),
    stale_minutes: int = Query(180, ge=5, le=24 * 60),
    user: dict = Depends(get_current_user),
):
    """
    Backfill analytics for all successful uploads that already have platform video IDs.
    Used after account connect/reconnect and for full aggregate refresh.
    """
    uid = str(user["id"])

    def _row_has_video_id(raw_pr: Any) -> bool:
        pr = _safe_json(raw_pr, [])
        entries: list[dict[str, Any]] = []
        if isinstance(pr, list):
            entries = [x for x in pr if isinstance(x, dict)]
        elif isinstance(pr, dict):
            entries = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in pr.items()]
        for e in entries:
            vid = (
                e.get("platform_video_id")
                or e.get("video_id")
                or e.get("videoId")
                or e.get("id")
                or e.get("media_id")
                or e.get("post_id")
                or e.get("share_id")
            )
            if str(vid or "").strip():
                return True
            plat = str(e.get("platform") or "").strip().lower()
            for u in (
                e.get("platform_url"),
                e.get("url"),
                e.get("permalink"),
                e.get("link"),
                e.get("share_url"),
            ):
                if _extract_platform_video_id_from_url(plat, str(u or "")):
                    return True
        return False

    async def _run_sync_all() -> dict[str, Any]:
        t_bulk = _time.time()
        started_bulk = datetime.now(timezone.utc).isoformat()
        await _analytics_sync_status_patch(
            uid,
            "upload_analytics",
            {
                "status": "running",
                "queued": False,
                "started_at": started_bulk,
                "finished_at": None,
                "duration_ms": None,
                "error": None,
                "stale_minutes": int(stale_minutes),
                "max_uploads": int(max_uploads),
            },
        )
        try:
            stale_cutoff = _now_utc() - timedelta(minutes=int(stale_minutes))
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT id, platform_results, analytics_synced_at
                      FROM uploads
                     WHERE user_id = $1
                       AND status IN {SUCCESSFUL_STATUS_SQL_IN}
                       AND (analytics_synced_at IS NULL OR analytics_synced_at < $3)
                     ORDER BY created_at DESC
                     LIMIT $2
                    """,
                    user["id"],
                    int(max_uploads),
                    stale_cutoff,
                )

            candidates = [str(r["id"]) for r in rows if _row_has_video_id(r.get("platform_results"))]
            await _analytics_sync_status_patch(
                uid,
                "upload_analytics",
                {
                    "eligible_rows": len(rows),
                    "candidates": len(candidates),
                },
            )
            if not candidates:
                finished = datetime.now(timezone.utc).isoformat()
                out = {"status": "ok", "processed": 0, "synced": 0, "failed": 0, "skipped": len(rows)}
                await _analytics_sync_status_patch(
                    uid,
                    "upload_analytics",
                    {
                        "status": "completed",
                        "finished_at": finished,
                        "duration_ms": int((_time.time() - t_bulk) * 1000),
                        **{k: out[k] for k in ("processed", "synced", "failed", "skipped") if k in out},
                    },
                )
                return out

            sem = asyncio.Semaphore(max(2, int(os.environ.get("UPLOAD_ANALYTICS_SYNC_CONCURRENCY", "4") or 4)))
            synced = 0
            failed = 0

            async def _run_one(upload_id: str) -> None:
                nonlocal synced, failed
                async with sem:
                    try:
                        res = await sync_upload_analytics(
                            upload_id=upload_id,
                            background_tasks=BackgroundTasks(),
                            user=user,
                        )
                        if isinstance(res, dict) and res.get("synced"):
                            synced += 1
                        else:
                            failed += 1
                    except Exception as ex:
                        failed += 1
                        logger.warning(f"sync-all analytics failed for {uid}/{upload_id}: {ex}")

            await asyncio.gather(*[_run_one(u) for u in candidates], return_exceptions=True)

            try:
                await _analytics_track_platform_metrics_refresh(user["id"], "sync-upload-analytics-all-tail")
            except Exception as ex:
                logger.warning(f"sync-all platform_metrics_cache refresh failed for {uid}: {ex}")

            out = {
                "status": "ok",
                "processed": len(candidates),
                "synced": int(synced),
                "failed": int(failed),
                "skipped": max(0, int(len(rows) - len(candidates))),
            }
            finished = datetime.now(timezone.utc).isoformat()
            await _analytics_sync_status_patch(
                uid,
                "upload_analytics",
                {
                    "status": "completed",
                    "finished_at": finished,
                    "duration_ms": int((_time.time() - t_bulk) * 1000),
                    "processed": out["processed"],
                    "synced": out["synced"],
                    "failed": out["failed"],
                    "skipped": out["skipped"],
                },
            )
            return out
        except Exception as ex:
            finished = datetime.now(timezone.utc).isoformat()
            await _analytics_sync_status_patch(
                uid,
                "upload_analytics",
                {
                    "status": "failed",
                    "finished_at": finished,
                    "duration_ms": int((_time.time() - t_bulk) * 1000),
                    "error": str(ex)[:500],
                },
            )
            raise

    if async_mode:
        await _analytics_sync_status_patch(
            uid,
            "upload_analytics",
            {
                "status": "queued",
                "queued": True,
                "started_at": None,
                "finished_at": None,
                "duration_ms": None,
                "error": None,
                "stale_minutes": int(stale_minutes),
                "max_uploads": int(max_uploads),
            },
        )

        async def _bg_job() -> None:
            try:
                await _run_sync_all()
            except Exception as ex:
                logger.warning(f"sync-all async job failed for {uid}: {ex}")

        background_tasks.add_task(_bg_job)
        return {"status": "accepted", "queued": True, "max_uploads": int(max_uploads)}

    return await _run_sync_all()


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
    """Get scheduled upload statistics for the current user."""
    cache_key = f"sched_stats:{user['id']}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    async with db_pool.acquire() as conn:
        now = _now_utc()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        week_end = now + timedelta(days=7)

        pending_count = await conn.fetchval("""
            SELECT COUNT(*) FROM uploads
            WHERE user_id = $1
              AND scheduled_time IS NOT NULL
              AND scheduled_time > $2
              AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
        """, user["id"], now)

        today_count = await conn.fetchval("""
            SELECT COUNT(*) FROM uploads
            WHERE user_id = $1
              AND scheduled_time >= $2
              AND scheduled_time < $3
              AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
        """, user["id"], today_start, today_end)

        week_count = await conn.fetchval("""
            SELECT COUNT(*) FROM uploads
            WHERE user_id = $1
              AND scheduled_time >= $2
              AND scheduled_time < $3
              AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
        """, user["id"], now, week_end)

    result = {
        "pending": pending_count or 0,
        "today": today_count or 0,
        "week": week_count or 0,
    }
    await cache_set(cache_key, result, CACHE_TTL_SHORT)
    return result

@app.get("/api/scheduled/list")
async def get_scheduled_list(user: dict = Depends(get_current_user)):
    """Get list of all scheduled uploads for the current user."""
    try:
        async with db_pool.acquire() as conn:
            uploads = await conn.fetch("""
                SELECT
                    id, filename, title, scheduled_time, platforms, target_accounts,
                    thumbnail_r2_key, caption, status,
                    created_at, timezone, schedule_mode, schedule_metadata
                FROM uploads
                WHERE user_id = $1
                  AND status IN ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish')
                ORDER BY scheduled_time ASC NULLS LAST, created_at ASC
            """, user["id"])

            result = []
            for upload in uploads:
                thumbnail_url = None
                if upload["thumbnail_r2_key"]:
                    try:
                        s3 = get_s3_client()
                        thumbnail_url = s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': R2_BUCKET_NAME, 'Key': _normalize_r2_key(upload["thumbnail_r2_key"])},
                            ExpiresIn=3600
                        )
                    except Exception as e:
                        logger.debug("scheduled/list: thumbnail presign failed upload=%s: %s", upload.get("id"), e)

                smart_schedule = None
                try:
                    sm = upload["schedule_metadata"]
                    if sm and upload["schedule_mode"] == "smart":
                        smart_schedule = sm if isinstance(sm, dict) else json.loads(sm)
                except Exception as e:
                    logger.debug("scheduled/list: schedule_metadata parse failed upload=%s: %s", upload.get("id"), e)

                target_ids = [str(x) for x in (upload.get("target_accounts") or []) if x]
                target_account_details = []
                if target_ids:
                    rows = await conn.fetch(
                        """SELECT id, platform, account_name, account_username, account_avatar
                           FROM platform_tokens
                           WHERE user_id = $1 AND id = ANY($2::uuid[]) AND revoked_at IS NULL""",
                        user["id"], target_ids
                    )
                    for r in rows:
                        target_account_details.append({
                            "id": str(r["id"]),
                            "platform": r["platform"],
                            "name": r["account_name"] or "",
                            "username": r["account_username"] or "",
                            "avatar": _platform_account_avatar_to_url(r["account_avatar"]) or "",
                        })

                result.append({
                    "id": str(upload["id"]),
                    "filename": upload.get("filename") or "",
                    "title": upload["title"] or "Untitled",
                    "scheduled_time": upload["scheduled_time"].isoformat() if upload["scheduled_time"] else None,
                    "timezone": upload["timezone"] or "UTC",
                    "platforms": list(upload["platforms"]) if upload["platforms"] else [],
                    "target_accounts": target_ids,
                    "target_account_details": target_account_details,
                    "thumbnail": thumbnail_url,
                    "caption": upload["caption"],
                    "status": upload["status"],
                    "schedule_mode": upload["schedule_mode"] or "scheduled",
                    "smart_schedule": smart_schedule,
                    "is_editable": upload["status"] in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
                    "created_at": upload["created_at"].isoformat() if upload["created_at"] else None
                })

        return result
    except Exception as _ex:
        logger.exception("GET /api/scheduled/list failed")
        raise HTTPException(500, f"Scheduled list error: {_ex!s}")

@app.get("/api/scheduled/{upload_id}")
async def get_scheduled_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Get details of a specific scheduled upload (for edit form)"""
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("""
            SELECT 
                id, title, scheduled_time, platforms, timezone,
                thumbnail_r2_key, caption, hashtags, privacy,
                status, created_at, schedule_mode, schedule_metadata
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
                    Params={'Bucket': R2_BUCKET_NAME, 'Key': _normalize_r2_key(upload["thumbnail_r2_key"])},
                    ExpiresIn=3600
                )
            except Exception as e:
                logger.debug("scheduled/detail: thumbnail presign failed upload=%s: %s", upload_id, e)

        sm = upload.get("schedule_metadata")
        if sm and isinstance(sm, str):
            try:
                sm = json.loads(sm)
            except Exception:
                sm = None
        elif not isinstance(sm, dict):
            sm = None
        
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
        "schedule_mode": upload.get("schedule_mode") or "scheduled",
        "schedule_metadata": sm,
        "smart_schedule": sm,  # alias for scheduled.html saveScheduledUpload()
        "is_editable": upload.get("status") in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
        "created_at": upload["created_at"].isoformat() if upload["created_at"] else None
    }

def _parse_smart_schedule(sm: dict, upload_platforms: list) -> tuple:
    """
    Parse smart_schedule (platform -> ISO datetime string).
    Returns (schedule_metadata_json, scheduled_time_dt).
    Same logic as create flow: validate platforms, parse ISO, set scheduled_time = min.
    """
    if not isinstance(sm, dict):
        raise HTTPException(400, "smart_schedule must be a dict of platform -> ISO datetime string")
    if not sm:
        raise HTTPException(400, "smart_schedule requires per-platform times (non-empty object)")
    platforms = list(upload_platforms or [])
    for k in sm:
        if k not in platforms:
            raise HTTPException(400, f"smart_schedule platform '{k}' not in upload platforms")
    dts = []
    for v in sm.values():
        if not v:
            continue
        s = str(v).replace("Z", "+00:00").replace("z", "+00:00")
        try:
            dts.append(datetime.fromisoformat(s))
        except ValueError:
            raise HTTPException(400, "smart_schedule values must be valid ISO datetime strings")
    metadata = {k: v for k, v in sm.items() if v}
    scheduled_dt = min(dts) if dts else None
    return metadata, scheduled_dt


async def _apply_smart_schedule(conn, upload_id: str, user_id: str, sm: dict) -> None:
    """Apply smart_schedule to upload (same logic as create)."""
    upload = await conn.fetchrow(
        "SELECT id, status, platforms FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id, user_id
    )
    if not upload:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if upload["status"] not in editable:
        raise HTTPException(400, "Cannot edit upload that is already processing or published")

    metadata, scheduled_dt = _parse_smart_schedule(sm, upload["platforms"])
    await conn.execute("""
        UPDATE uploads SET
            schedule_metadata = $1::jsonb,
            scheduled_time = $2,
            schedule_mode = 'smart',
            updated_at = NOW()
        WHERE id = $3 AND user_id = $4
    """, json.dumps(metadata), scheduled_dt, upload_id, user_id)


def _sanitize_hashtag_token(raw: Any) -> str:
    """Normalize hashtag token to a plain lowercase word/identifier."""
    token = re.sub(r"[^a-z0-9_]", "", str(raw or "").strip().lower().lstrip("#"))
    return token[:50]


def _sanitize_hashtag_list(raw_items: Any) -> List[str]:
    items = expand_hashtag_items(raw_items)
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        token = _sanitize_hashtag_token(item)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(f"#{token}")
    return out


async def _update_upload_metadata(conn, upload_id: str, user_id: str, update_data: UploadUpdate) -> None:
    """PATCH /api/uploads - title, caption, hashtags, scheduled_time, smart_schedule."""
    upload = await conn.fetchrow(
        "SELECT id, status, platforms FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id, user_id
    )
    if not upload:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if upload["status"] not in editable:
        raise HTTPException(400, "Cannot edit upload that is already processing or published")

    updates = []
    params: list = [upload_id, user_id]
    param_count = 2

    if update_data.title is not None:
        param_count += 1
        updates.append(f"title = ${param_count}")
        params.append(update_data.title)

    if update_data.caption is not None:
        param_count += 1
        updates.append(f"caption = ${param_count}")
        params.append(update_data.caption)

    if update_data.hashtags is not None:
        cleaned = _sanitize_hashtag_list(update_data.hashtags)
        param_count += 1
        updates.append(f"hashtags = ${param_count}")
        params.append(cleaned)

    if update_data.scheduled_time is not None:
        param_count += 1
        updates.append(f"scheduled_time = ${param_count}")
        params.append(update_data.scheduled_time)

    if update_data.smart_schedule is not None:
        metadata, scheduled_dt = _parse_smart_schedule(update_data.smart_schedule, upload["platforms"])
        param_count += 1
        updates.append(f"schedule_metadata = ${param_count}::jsonb")
        params.append(json.dumps(metadata))
        if scheduled_dt is not None:
            param_count += 1
            updates.append(f"scheduled_time = ${param_count}")
            params.append(scheduled_dt)
        param_count += 1
        updates.append(f"schedule_mode = ${param_count}")
        params.append("smart")

    if not updates:
        raise HTTPException(400, "No updates provided")

    param_count += 1
    updates.append(f"updated_at = ${param_count}")
    params.append(_now_utc())

    await conn.execute(f"UPDATE uploads SET {', '.join(updates)} WHERE id = $1 AND user_id = $2", *params)


@app.patch("/api/scheduled/{upload_id}")
async def update_scheduled_upload(
    upload_id: str,
    update_data: SmartScheduleOnlyUpdate,
    user: dict = Depends(get_current_user),
):
    """Update a scheduled upload's smart_schedule only (platform -> ISO datetime string)."""
    async with db_pool.acquire() as conn:
        await _apply_smart_schedule(conn, upload_id, user["id"], update_data.smart_schedule)
    return {"status": "updated", "id": upload_id}


@app.patch("/api/uploads/{upload_id}")
async def update_upload(
    upload_id: str,
    update_data: UploadUpdate,
    user: dict = Depends(get_current_user),
):
    """Update an upload's metadata: title, caption, hashtags, scheduled_time, smart_schedule."""
    async with db_pool.acquire() as conn:
        await _update_upload_metadata(conn, upload_id, user["id"], update_data)
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
        
        # Allow cancel for all pending/scheduled states shown on scheduled page
        if upload["status"] not in ('pending', 'scheduled', 'queued', 'staged', 'ready_to_publish'):
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

    pc = plan.get("priority_class", "p4")
    await enqueue_job(job_data, lane="process", priority_class=pc)
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
    try:
        async with db_pool.acquire() as conn:
            try:
                prefs = await conn.fetchrow(
                    "SELECT * FROM user_preferences WHERE user_id = $1",
                    user["id"]
                )
            except Exception:
                prefs = None  # fall through to INSERT-on-demand

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
                except Exception:
                    always_tags = []
            if isinstance(blocked_tags, str):
                try:
                    blocked_tags = json.loads(blocked_tags)
                except Exception:
                    blocked_tags = []
            if isinstance(platform_tags, str):
                try:
                    platform_tags = json.loads(platform_tags)
                except Exception:
                    platform_tags = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}

            out = {
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
                "authSecurityAlerts": d.get("auth_security_alerts", True),
                "digestEmails": d.get("digest_emails", True),
                "scheduledAlertEmails": d.get("scheduled_alert_emails", True),
                "discordWebhook": d.get("discord_webhook"),
                "trillEnabled": bool(d.get("trill_enabled", False)),
                "trillMinScore": int(d.get("trill_min_score", 0) or 0),
                "trillHudEnabled": bool(d.get("trill_hud_enabled", False)),
                "trillAiEnhance": bool(d.get("trill_ai_enhance", False)),
                "trillOpenaiModel": d.get("trill_openai_model", "gpt-4o-mini"),
                "styledThumbnails": d.get("styled_thumbnails", True),
                "useAudioContext": bool(d.get("use_audio_context", True)),
                "audioTranscription": bool(d.get("audio_transcription", True)),
                "aiServiceTelemetry": bool(d.get("ai_service_telemetry", True)),
                "aiServiceAudioSignals": bool(d.get("ai_service_audio_signals", True)),
                "aiServiceMusicDetection": bool(d.get("ai_service_music_detection", True)),
                "aiServiceAudioSummary": bool(d.get("ai_service_audio_summary", True)),
                "aiServiceEmotionSignals": bool(d.get("ai_service_emotion_signals", True)),
                "aiServiceCaptionWriter": bool(d.get("ai_service_caption_writer", True)),
                "aiServiceThumbnailDesigner": bool(d.get("ai_service_thumbnail_designer", True)),
                "aiServiceFrameInspector": bool(d.get("ai_service_frame_inspector", True)),
                "aiServiceSpeechToText": bool(d.get("ai_service_speech_to_text", True)),
                "aiServiceVideoAnalyzer": bool(d.get("ai_service_video_analyzer", True)),
                "aiServiceSceneUnderstanding": bool(d.get("ai_service_scene_understanding", True)),
                "thumbnailStudioEnabled": False,
                "thumbnailStudioEngineEnabled": False,
                "thumbnailPersonaEnabled": False,
                "thumbnailDefaultPersonaId": None,
                "thumbnailPersonaStrength": 70,
            }
            # Overlay users.preferences — source of truth for hashtags + caption (PUT /api/me/preferences)
            users_prefs = None
            try:
                users_prefs = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", user["id"])
            except Exception as e:
                logger.debug("settings merge: users.preferences fetch failed: %s", e)
            up = _parse_users_preferences(users_prefs) if users_prefs else {}
            if up:
                def _has_content(v, is_platform_map: bool = False) -> bool:
                    if v is None:
                        return False
                    if is_platform_map and isinstance(v, dict):
                        return any(
                            (isinstance(x, list) and len(x) > 0)
                            or (isinstance(x, str) and x.strip())
                            for x in (v.values() or [])
                        )
                    if isinstance(v, (list, tuple)):
                        return len(v) > 0
                    if isinstance(v, dict):
                        return len(v) > 0
                    return bool(v)

                if up.get("alwaysHashtags") is not None or up.get("always_hashtags") is not None:
                    v = up.get("alwaysHashtags") or up.get("always_hashtags")
                    if _has_content(v):
                        out["alwaysHashtags"] = v if isinstance(v, list) else []
                if up.get("blockedHashtags") is not None or up.get("blocked_hashtags") is not None:
                    v = up.get("blockedHashtags") or up.get("blocked_hashtags")
                    if _has_content(v):
                        out["blockedHashtags"] = v if isinstance(v, list) else []
                if up.get("platformHashtags") is not None or up.get("platform_hashtags") is not None:
                    v = up.get("platformHashtags") or up.get("platform_hashtags")
                    if _has_content(v, is_platform_map=True):
                        out["platformHashtags"] = v if isinstance(v, dict) else {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
                if up.get("maxHashtags") is not None or up.get("max_hashtags") is not None:
                    out["maxHashtags"] = str(up.get("maxHashtags") or up.get("max_hashtags") or 15)
                if up.get("aiHashtagCount") is not None or up.get("ai_hashtag_count") is not None:
                    out["aiHashtagCount"] = str(up.get("aiHashtagCount") or up.get("ai_hashtag_count") or 5)
                out["captionStyle"] = up.get("captionStyle") or up.get("caption_style") or "story"
                out["captionTone"] = up.get("captionTone") or up.get("caption_tone") or "authentic"
                out["captionVoice"] = up.get("captionVoice") or up.get("caption_voice") or "default"
                out["captionFrameCount"] = up.get("captionFrameCount") or up.get("caption_frame_count") or 6
                if up.get("useAudioContext") is not None or up.get("use_audio_context") is not None:
                    out["useAudioContext"] = bool(up.get("useAudioContext", up.get("use_audio_context", True)))
                if up.get("audioTranscription") is not None or up.get("audio_transcription") is not None:
                    out["audioTranscription"] = bool(up.get("audioTranscription", up.get("audio_transcription", True)))
                if up.get("thumbnailStudioEnabled") is not None or up.get("thumbnail_studio_enabled") is not None:
                    out["thumbnailStudioEnabled"] = bool(up.get("thumbnailStudioEnabled", up.get("thumbnail_studio_enabled", False)))
                _ge = _engine_enabled_from_mixed_prefs(up)
                if _ge is not None:
                    out["thumbnailStudioEngineEnabled"] = bool(_ge)
                if up.get("thumbnailPersonaEnabled") is not None or up.get("thumbnail_persona_enabled") is not None:
                    out["thumbnailPersonaEnabled"] = bool(up.get("thumbnailPersonaEnabled", up.get("thumbnail_persona_enabled", False)))
                if up.get("thumbnailDefaultPersonaId") is not None or up.get("thumbnail_default_persona_id") is not None:
                    out["thumbnailDefaultPersonaId"] = str(up.get("thumbnailDefaultPersonaId") or up.get("thumbnail_default_persona_id") or "").strip() or None
                if up.get("thumbnailPersonaStrength") is not None or up.get("thumbnail_persona_strength") is not None:
                    try:
                        out["thumbnailPersonaStrength"] = max(0, min(100, int(up.get("thumbnailPersonaStrength", up.get("thumbnail_persona_strength", 70)) or 70)))
                    except Exception:
                        out["thumbnailPersonaStrength"] = 70
            else:
                out.setdefault("captionStyle", "story")
                out.setdefault("captionTone", "authentic")
                out.setdefault("captionVoice", "default")
                out.setdefault("captionFrameCount", 6)
                out.setdefault("thumbnailStudioEnabled", False)
                out.setdefault("thumbnailStudioEngineEnabled", False)
                out.setdefault("thumbnailPersonaEnabled", False)
                out.setdefault("thumbnailDefaultPersonaId", None)
                out.setdefault("thumbnailPersonaStrength", 70)
            _strip_legacy_thumbnail_engine_keys(out)
            return out
    except Exception as e:
        logger.exception("get_user_preferences failed: %s", e)
        # Return defaults so settings page loads; avoid 500 when DB schema mismatch or migration not run
        return {
            "autoCaptions": False, "autoThumbnails": False, "thumbnailInterval": "5",
            "defaultPrivacy": "public", "aiHashtagsEnabled": False, "aiHashtagCount": "5",
            "aiHashtagStyle": "mixed", "hashtagPosition": "end", "maxHashtags": "15",
            "alwaysHashtags": [], "blockedHashtags": [],
            "platformHashtags": {"tiktok": [], "youtube": [], "instagram": [], "facebook": []},
            "emailNotifications": True, "discordWebhook": None,
            "authSecurityAlerts": True, "digestEmails": True, "scheduledAlertEmails": True,
            "trillEnabled": False, "trillMinScore": 60, "trillHudEnabled": False,
            "trillAiEnhance": True, "trillOpenaiModel": "gpt-4o-mini", "styledThumbnails": True,
            "captionStyle": "story", "captionTone": "authentic", "captionVoice": "default", "captionFrameCount": 6,
            "useAudioContext": True,
            "audioTranscription": True,
            "aiServiceTelemetry": True,
            "aiServiceAudioSignals": True,
            "aiServiceMusicDetection": True,
            "aiServiceAudioSummary": True,
            "aiServiceEmotionSignals": True,
            "aiServiceCaptionWriter": True,
            "aiServiceThumbnailDesigner": True,
            "aiServiceFrameInspector": True,
            "aiServiceSpeechToText": True,
            "aiServiceVideoAnalyzer": True,
            "aiServiceSceneUnderstanding": True,
            "thumbnailStudioEnabled": False,
            "thumbnailStudioEngineEnabled": False,
            "thumbnailPersonaEnabled": False,
            "thumbnailDefaultPersonaId": None,
            "thumbnailPersonaStrength": 70,
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
        "authSecurityAlerts": "auth_security_alerts",
        "digestEmails": "digest_emails",
        "scheduledAlertEmails": "scheduled_alert_emails",
        "discordWebhook": "discord_webhook",
        "trillEnabled": "trill_enabled",
        "trillMinScore": "trill_min_score",
        "trillHudEnabled": "trill_hud_enabled",
        "trillAiEnhance": "trill_ai_enhance",
        "trillOpenaiModel": "trill_openai_model",
        "styledThumbnails": "styled_thumbnails",
        "captionStyle": "caption_style",
        "captionTone": "caption_tone",
        "captionVoice": "caption_voice",
        "captionFrameCount": "caption_frame_count",
        "useAudioContext": "use_audio_context",
        "audioTranscription": "audio_transcription",
        "aiServiceTelemetry": "ai_service_telemetry",
        "aiServiceAudioSignals": "ai_service_audio_signals",
        "aiServiceMusicDetection": "ai_service_music_detection",
        "aiServiceAudioSummary": "ai_service_audio_summary",
        "aiServiceEmotionSignals": "ai_service_emotion_signals",
        "aiServiceCaptionWriter": "ai_service_caption_writer",
        "aiServiceThumbnailDesigner": "ai_service_thumbnail_designer",
        "aiServiceFrameInspector": "ai_service_frame_inspector",
        "aiServiceSpeechToText": "ai_service_speech_to_text",
        "aiServiceVideoAnalyzer": "ai_service_video_analyzer",
        "aiServiceSceneUnderstanding": "ai_service_scene_understanding",
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
                    clean = re.sub(r"[^a-z0-9_]", "", item.strip().lower().replace('#', ''))
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
            parts = [re.sub(r"[^a-z0-9_]", "", p.strip().lower().replace('#', '')) for p in s.split(',')]
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

    always = _coerce_hashtag_list(p.get("always_hashtags"))
    blocked = _coerce_hashtag_list(p.get("blocked_hashtags"))
    platform = _coerce_platform_map(p.get("platform_hashtags"))
    
    # DEBUG: Log what we're about to save
    logger.info(f"Saving preferences for user {user['id']}")
    logger.info(f"always_hashtags: {always} (type: {type(always)})")
    logger.info(f"blocked_hashtags: {blocked} (type: {type(blocked)})")
    logger.info(f"platform_hashtags: {platform} (type: {type(platform)})")

    # core scalar coercions
    auto_captions = bool(p.get("auto_captions", False))
    auto_thumbnails = bool(p.get("auto_thumbnails", False))
    styled_thumbnails = bool(p.get("styled_thumbnails", True))

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
    auth_security_alerts = bool(p.get("auth_security_alerts", True))
    digest_emails = bool(p.get("digest_emails", True))
    scheduled_alert_emails = bool(p.get("scheduled_alert_emails", True))
    discord_webhook = p.get("discord_webhook")

    # Trill fields (user_preferences)
    trill_enabled = bool(p.get("trill_enabled", False))
    try:
        trill_min_score = int(p.get("trill_min_score", 60))
        trill_min_score = max(0, min(100, trill_min_score))
    except (TypeError, ValueError):
        trill_min_score = 60
    trill_hud_enabled = bool(p.get("trill_hud_enabled", False))
    trill_ai_enhance = bool(p.get("trill_ai_enhance", True))
    trill_openai_model = str(p.get("trill_openai_model", "gpt-4o-mini") or "gpt-4o-mini")[:50]
    use_audio_context = bool(p.get("useAudioContext", p.get("use_audio_context", True)))
    audio_transcription = bool(p.get("audioTranscription", p.get("audio_transcription", True)))
    ai_service_telemetry = bool(p.get("aiServiceTelemetry", p.get("ai_service_telemetry", True)))
    ai_service_audio_signals = bool(p.get("aiServiceAudioSignals", p.get("ai_service_audio_signals", True)))
    ai_service_music_detection = bool(p.get("aiServiceMusicDetection", p.get("ai_service_music_detection", True)))
    ai_service_audio_summary = bool(p.get("aiServiceAudioSummary", p.get("ai_service_audio_summary", True)))
    ai_service_emotion_signals = bool(p.get("aiServiceEmotionSignals", p.get("ai_service_emotion_signals", True)))
    ai_service_caption_writer = bool(p.get("aiServiceCaptionWriter", p.get("ai_service_caption_writer", True)))
    ai_service_thumbnail_designer = bool(p.get("aiServiceThumbnailDesigner", p.get("ai_service_thumbnail_designer", True)))
    ai_service_frame_inspector = bool(p.get("aiServiceFrameInspector", p.get("ai_service_frame_inspector", True)))
    ai_service_speech_to_text = bool(p.get("aiServiceSpeechToText", p.get("ai_service_speech_to_text", True)))
    ai_service_video_analyzer = bool(p.get("aiServiceVideoAnalyzer", p.get("ai_service_video_analyzer", True)))
    ai_service_scene_understanding = bool(p.get("aiServiceSceneUnderstanding", p.get("ai_service_scene_understanding", True)))

    try:
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
                styled_thumbnails = $3,
                thumbnail_interval = $4,
                default_privacy = $5,
                ai_hashtags_enabled = $6,
                ai_hashtag_count = $7,
                ai_hashtag_style = $8,
                hashtag_position = $9,
                max_hashtags = $10,
                always_hashtags = $11::jsonb,
                blocked_hashtags = $12::jsonb,
                platform_hashtags = $13::jsonb,
                email_notifications = $14,
                auth_security_alerts = $15,
                digest_emails = $16,
                scheduled_alert_emails = $17,
                discord_webhook = $18,
                trill_enabled = $19,
                trill_min_score = $20,
                trill_hud_enabled = $21,
                trill_ai_enhance = $22,
                trill_openai_model = $23,
                use_audio_context = $24,
                audio_transcription = $25,
                ai_service_telemetry = $26,
                ai_service_audio_signals = $27,
                ai_service_music_detection = $28,
                ai_service_audio_summary = $29,
                ai_service_emotion_signals = $30,
                ai_service_caption_writer = $31,
                ai_service_thumbnail_designer = $32,
                ai_service_frame_inspector = $33,
                ai_service_speech_to_text = $34,
                ai_service_video_analyzer = $35,
                ai_service_scene_understanding = $36,
                updated_at = NOW()
            WHERE user_id = $37
            """,
            auto_captions,
            auto_thumbnails,
            styled_thumbnails,
            thumbnail_interval,
            default_privacy,
            ai_hashtags_enabled,
            ai_hashtag_count,
            ai_hashtag_style,
            hashtag_position,
            max_hashtags,
            json.dumps(always),
            json.dumps(blocked),
            json.dumps(platform),
            email_notifications,
            auth_security_alerts,
            digest_emails,
            scheduled_alert_emails,
            discord_webhook,
            trill_enabled,
            trill_min_score,
            trill_hud_enabled,
            trill_ai_enhance,
            trill_openai_model,
            use_audio_context,
            audio_transcription,
            ai_service_telemetry,
            ai_service_audio_signals,
            ai_service_music_detection,
            ai_service_audio_summary,
            ai_service_emotion_signals,
            ai_service_caption_writer,
            ai_service_thumbnail_designer,
            ai_service_frame_inspector,
            ai_service_speech_to_text,
            ai_service_video_analyzer,
            ai_service_scene_understanding,
            user["id"],
        )

        # Sync discord_webhook to user_settings so worker (load_user_settings) gets it
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, discord_webhook) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET discord_webhook = $2, updated_at = NOW()
            """,
            user["id"],
            discord_webhook,
        )

        # Sync caption fields to users.preferences (worker caption_stage reads from there)
        caption_keys = ("captionStyle", "captionTone", "captionVoice", "captionFrameCount")
        caption_snake = ("caption_style", "caption_tone", "caption_voice", "caption_frame_count")
        if any(k in p or sk in p for k, sk in zip(caption_keys, caption_snake)):
            try:
                _CAPTION_STYLES = ("story", "punchy", "factual")
                _CAPTION_TONES = ("hype", "calm", "cinematic", "authentic")
                _CAPTION_VOICES = ("default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator")
                users_prefs_row = await conn.fetchval("SELECT preferences FROM users WHERE id = $1", user["id"])
                users_prefs = {}
                if users_prefs_row:
                    users_prefs = json.loads(users_prefs_row) if isinstance(users_prefs_row, str) else (users_prefs_row or {})
                if not isinstance(users_prefs, dict):
                    users_prefs = {}
                if "captionStyle" in p or "caption_style" in p:
                    v = str(p.get("captionStyle") or p.get("caption_style") or "story").strip().lower()
                    users_prefs["captionStyle"] = users_prefs["caption_style"] = v if v in _CAPTION_STYLES else "story"
                if "captionTone" in p or "caption_tone" in p:
                    v = str(p.get("captionTone") or p.get("caption_tone") or "authentic").strip().lower()
                    users_prefs["captionTone"] = users_prefs["caption_tone"] = v if v in _CAPTION_TONES else "authentic"
                if "captionVoice" in p or "caption_voice" in p:
                    v = str(p.get("captionVoice") or p.get("caption_voice") or "default").strip().lower()
                    users_prefs["captionVoice"] = users_prefs["caption_voice"] = v if v in _CAPTION_VOICES else "default"
                if "captionFrameCount" in p or "caption_frame_count" in p:
                    try:
                        ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))
                        max_frames = ent.max_caption_frames or 20
                        v = int(p.get("captionFrameCount") or p.get("caption_frame_count") or 6)
                        v = max(2, min(v, max_frames))
                        users_prefs["captionFrameCount"] = users_prefs["caption_frame_count"] = v
                    except (TypeError, ValueError):
                        users_prefs["captionFrameCount"] = users_prefs["caption_frame_count"] = 6
                await conn.execute(
                    "UPDATE users SET preferences = $1, updated_at = NOW() WHERE id = $2",
                    json.dumps(users_prefs),
                    user["id"],
                )
            except Exception as _cap_err:
                logger.warning(f"Caption sync to users.preferences failed (column may not exist): {_cap_err}")

        # immediate read-after-write to validate persistence (helps front-end debugging)
        row = await conn.fetchrow(
            "SELECT updated_at FROM user_preferences WHERE user_id = $1",
            user["id"],
        )

        return {"ok": True, "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None)}
    except Exception as _save_err:
        import traceback
        logger.error(f"save_user_preferences UPDATE failed: {type(_save_err).__name__}: {_save_err}\n{traceback.format_exc()[-800:]}")
        # Retry without trill columns (handles pre-migration schemas)
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """UPDATE user_preferences SET
                        auto_captions=$1, auto_thumbnails=$2, thumbnail_interval=$3,
                        default_privacy=$4, ai_hashtags_enabled=$5, ai_hashtag_count=$6,
                        ai_hashtag_style=$7, hashtag_position=$8, max_hashtags=$9,
                        always_hashtags=$10::jsonb, blocked_hashtags=$11::jsonb,
                        platform_hashtags=$12::jsonb, email_notifications=$13,
                        auth_security_alerts=$14, digest_emails=$15, scheduled_alert_emails=$16,
                        discord_webhook=$17, updated_at=NOW()
                       WHERE user_id=$18""",
                    auto_captions, auto_thumbnails, thumbnail_interval, default_privacy,
                    ai_hashtags_enabled, ai_hashtag_count, ai_hashtag_style, hashtag_position,
                    max_hashtags, json.dumps(always), json.dumps(blocked), json.dumps(platform),
                    email_notifications, auth_security_alerts, digest_emails, scheduled_alert_emails,
                    discord_webhook, user["id"],
                )
                row = await conn.fetchrow("SELECT updated_at FROM user_preferences WHERE user_id=$1", user["id"])
            return {"ok": True, "updatedAt": (row["updated_at"].isoformat() if row and row.get("updated_at") else None)}
        except Exception as _retry_err:
            logger.error(f"save_user_preferences retry also failed: {_retry_err}")
            raise HTTPException(500, f"Could not save preferences: {type(_retry_err).__name__}")



@app.put("/api/settings/preferences")
async def save_user_preferences_put(
    prefs: UserPreferencesUpdate,
    user: dict = Depends(get_current_user)
):
    """Backward-compatible alias for clients that still call PUT"""
    return await save_user_preferences(prefs.model_dump(by_alias=True), user)

def _parse_users_preferences(raw) -> dict:
    """Parse users.preferences JSONB into a dict."""
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw) or {}
        except Exception:
            return {}
    return dict(raw) if hasattr(raw, "keys") else {}


_LEGACY_THUMB_ENGINE_OUT_KEYS = frozenset(
    {
        "thumbnailPikzelsEnabled",
        "thumbnail_pikzels_enabled",
        "thumbnailUsePikzels",
        "thumbnail_use_pikzels",
    }
)


def _strip_legacy_thumbnail_engine_keys(d: Optional[dict]) -> None:
    """Remove deprecated engine key aliases (still accepted on input)."""
    if not isinstance(d, dict):
        return
    for k in _LEGACY_THUMB_ENGINE_OUT_KEYS:
        d.pop(k, None)


def _engine_enabled_from_mixed_prefs(p: dict):
    v = p.get("thumbnailStudioEngineEnabled")
    if v is None:
        v = p.get("thumbnail_studio_engine_enabled")
    if v is None:
        v = p.get("thumbnailPikzelsEnabled")
    if v is None:
        v = p.get("thumbnail_pikzels_enabled")
    return v


def _use_studio_engine_from_mixed_prefs(p: dict):
    v = p.get("thumbnailUseStudioEngine")
    if v is None:
        v = p.get("thumbnail_use_studio_engine")
    if v is None:
        v = p.get("thumbnailUsePikzels")
    if v is None:
        v = p.get("thumbnail_use_pikzels")
    return v


def _overlay_users_prefs_on_result(result: dict, up: dict) -> None:
    """Overlay users.preferences onto result. users.preferences wins for all fields."""
    if not isinstance(up, dict) or not up:
        return

    def _has_content(v, is_platform_map: bool = False) -> bool:
        if v is None:
            return False
        if is_platform_map and isinstance(v, dict):
            return any(
                (isinstance(x, list) and len(x) > 0)
                or (isinstance(x, str) and x.strip())
                for x in (v.values() or [])
            )
        if isinstance(v, (list, tuple)):
            return len(v) > 0
        if isinstance(v, dict):
            return len(v) > 0
        return bool(v)

    # Hashtag fields
    for camel, snake, key in [
        ("alwaysHashtags", "always_hashtags", "always_hashtags"),
        ("blockedHashtags", "blocked_hashtags", "blocked_hashtags"),
        ("platformHashtags", "platform_hashtags", "platform_hashtags"),
    ]:
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        is_platform_map = key == "platform_hashtags"
        if _has_content(v, is_platform_map=is_platform_map):
            result[camel] = result[key] = v
    # Scalar prefs from users.preferences
    for camel, snake in [
        ("maxHashtags", "max_hashtags"), ("aiHashtagCount", "ai_hashtag_count"),
        ("aiHashtagsEnabled", "ai_hashtags_enabled"), ("captionStyle", "caption_style"),
        ("captionTone", "caption_tone"), ("captionVoice", "caption_voice"),
        ("captionFrameCount", "caption_frame_count"), ("aiHashtagStyle", "ai_hashtag_style"),
        ("hashtagPosition", "hashtag_position"), ("autoCaptions", "auto_captions"),
        ("autoThumbnails", "auto_thumbnails"), ("styledThumbnails", "styled_thumbnails"),
        ("defaultPrivacy", "default_privacy"), ("thumbnailInterval", "thumbnail_interval"),
        ("useAudioContext", "use_audio_context"), ("audioTranscription", "audio_transcription"),
        ("aiServiceTelemetry", "ai_service_telemetry"),
        ("aiServiceAudioSignals", "ai_service_audio_signals"),
        ("aiServiceMusicDetection", "ai_service_music_detection"),
        ("aiServiceAudioSummary", "ai_service_audio_summary"),
        ("aiServiceEmotionSignals", "ai_service_emotion_signals"),
        ("aiServiceCaptionWriter", "ai_service_caption_writer"),
        ("aiServiceThumbnailDesigner", "ai_service_thumbnail_designer"),
        ("aiServiceFrameInspector", "ai_service_frame_inspector"),
        ("aiServiceSpeechToText", "ai_service_speech_to_text"),
        ("aiServiceVideoAnalyzer", "ai_service_video_analyzer"),
        ("aiServiceSceneUnderstanding", "ai_service_scene_understanding"),
        ("thumbnailStudioEnabled", "thumbnail_studio_enabled"),
        ("thumbnailStudioEngineEnabled", "thumbnail_studio_engine_enabled"),
        ("thumbnailPersonaEnabled", "thumbnail_persona_enabled"),
        ("thumbnailDefaultPersonaId", "thumbnail_default_persona_id"),
        ("thumbnailPersonaStrength", "thumbnail_persona_strength"),
        ("thumbnailUseStudioEngine", "thumbnail_use_studio_engine"),
        ("thumbnailUsePersona", "thumbnail_use_persona"),
        ("thumbnailPersonaId", "thumbnail_persona_id"),
    ]:
        v = up.get(camel) if up.get(camel) is not None else up.get(snake)
        if v is not None:
            result[camel] = result[snake] = v

    # Engine toggle / per-upload use: canonical keys only; still read legacy aliases from `up`.
    _ge = _engine_enabled_from_mixed_prefs(up)
    if _ge is not None:
        result["thumbnailStudioEngineEnabled"] = result["thumbnail_studio_engine_enabled"] = bool(_ge)
    _ue = _use_studio_engine_from_mixed_prefs(up)
    if _ue is not None:
        result["thumbnailUseStudioEngine"] = result["thumbnail_use_studio_engine"] = bool(_ue)
    _strip_legacy_thumbnail_engine_keys(result)


async def get_user_prefs_for_upload(conn, user_id: int) -> dict:
    """Helper to fetch user preferences for upload processing.
    Merges user_preferences table + users.preferences + user_settings so:
    - PUT /api/me/preferences (users.preferences) wins for hashtag/caption fields
    - POST /api/settings/preferences (user_preferences) provides defaults
    - user_settings provides hud_enabled for Trill HUD cost calculation
    """
    result = {}
    # Read from user_preferences table
    prefs_row = await conn.fetchrow(
        "SELECT * FROM user_preferences WHERE user_id = $1",
        user_id
    )
    if prefs_row:
        styled = prefs_row.get("styled_thumbnails", True)
        result = {
            "auto_captions": prefs_row["auto_captions"],
            "auto_thumbnails": prefs_row["auto_thumbnails"],
            "styled_thumbnails": styled,
            "styledThumbnails": styled,
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
            "auth_security_alerts": prefs_row.get("auth_security_alerts", True),
            "digest_emails": prefs_row.get("digest_emails", True),
            "scheduled_alert_emails": prefs_row.get("scheduled_alert_emails", True),
            "discord_webhook": prefs_row["discord_webhook"],
            "use_audio_context": prefs_row.get("use_audio_context", True),
            "useAudioContext": prefs_row.get("use_audio_context", True),
            "audio_transcription": prefs_row.get("audio_transcription", True),
            "audioTranscription": prefs_row.get("audio_transcription", True),
            "ai_service_telemetry": prefs_row.get("ai_service_telemetry", True),
            "aiServiceTelemetry": prefs_row.get("ai_service_telemetry", True),
            "ai_service_audio_signals": prefs_row.get("ai_service_audio_signals", True),
            "aiServiceAudioSignals": prefs_row.get("ai_service_audio_signals", True),
            "ai_service_music_detection": prefs_row.get("ai_service_music_detection", True),
            "aiServiceMusicDetection": prefs_row.get("ai_service_music_detection", True),
            "ai_service_audio_summary": prefs_row.get("ai_service_audio_summary", True),
            "aiServiceAudioSummary": prefs_row.get("ai_service_audio_summary", True),
            "ai_service_emotion_signals": prefs_row.get("ai_service_emotion_signals", True),
            "aiServiceEmotionSignals": prefs_row.get("ai_service_emotion_signals", True),
            "ai_service_caption_writer": prefs_row.get("ai_service_caption_writer", True),
            "aiServiceCaptionWriter": prefs_row.get("ai_service_caption_writer", True),
            "ai_service_thumbnail_designer": prefs_row.get("ai_service_thumbnail_designer", True),
            "aiServiceThumbnailDesigner": prefs_row.get("ai_service_thumbnail_designer", True),
            "ai_service_frame_inspector": prefs_row.get("ai_service_frame_inspector", True),
            "aiServiceFrameInspector": prefs_row.get("ai_service_frame_inspector", True),
            "ai_service_speech_to_text": prefs_row.get("ai_service_speech_to_text", True),
            "aiServiceSpeechToText": prefs_row.get("ai_service_speech_to_text", True),
            "ai_service_video_analyzer": prefs_row.get("ai_service_video_analyzer", True),
            "aiServiceVideoAnalyzer": prefs_row.get("ai_service_video_analyzer", True),
            "ai_service_scene_understanding": prefs_row.get("ai_service_scene_understanding", True),
            "aiServiceSceneUnderstanding": prefs_row.get("ai_service_scene_understanding", True),
            "thumbnail_studio_enabled": False,
            "thumbnailStudioEnabled": False,
            "thumbnail_studio_engine_enabled": False,
            "thumbnailStudioEngineEnabled": False,
            "thumbnail_persona_enabled": False,
            "thumbnailPersonaEnabled": False,
            "thumbnail_default_persona_id": None,
            "thumbnailDefaultPersonaId": None,
            "thumbnail_persona_strength": 70,
            "thumbnailPersonaStrength": 70,
            "thumbnail_use_studio_engine": False,
            "thumbnailUseStudioEngine": False,
            "thumbnail_use_persona": False,
            "thumbnailUsePersona": False,
            "thumbnail_persona_id": None,
            "thumbnailPersonaId": None,
        }

    # Overlay users.preferences (PUT /api/me/preferences writes here) — full overlay
    users_prefs_row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user_id)
    up = _parse_users_preferences(users_prefs_row["preferences"] if users_prefs_row else None)
    _overlay_users_prefs_on_result(result, up)

    # Overlay user_settings for hud_enabled (Trill HUD; PUT /api/settings writes here)
    try:
        us_row = await conn.fetchrow("SELECT hud_enabled FROM user_settings WHERE user_id = $1", user_id)
        if us_row and us_row.get("hud_enabled") is not None:
            result["hud_enabled"] = bool(us_row["hud_enabled"])
    except Exception as e:
        logger.debug("get_user_settings: hud_enabled overlay failed user_id=%s: %s", user_id, e)

    if result:
        _strip_legacy_thumbnail_engine_keys(result)
        return result

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
    styled = prefs.get("styledThumbnails", prefs.get("styled_thumbnails", True))
    _thumb_eng = _engine_enabled_from_mixed_prefs(prefs)
    _thumb_use_eng = _use_studio_engine_from_mixed_prefs(prefs)
    out = {
        "auto_captions": prefs.get("autoCaptions", False),
        "auto_thumbnails": prefs.get("autoThumbnails", False),
        "styled_thumbnails": styled,
        "styledThumbnails": styled,
        "thumbnail_interval": prefs.get("thumbnailInterval", 5),
        "default_privacy": prefs.get("defaultPrivacy", "public"),
        "ai_hashtags_enabled": prefs.get("aiHashtagsEnabled", False),
        "ai_hashtag_count": prefs.get("aiHashtagCount", 5),
        "ai_hashtag_style": prefs.get("aiHashtagStyle", "mixed"),
        "hashtag_position": prefs.get("hashtagPosition", "end"),
        "max_hashtags": prefs.get("maxHashtags", 15),
        "always_hashtags": prefs.get("alwaysHashtags", []),
        "blocked_hashtags": prefs.get("blockedHashtags", []),
        "platform_hashtags": prefs.get("platformHashtags", {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}),
        "email_notifications": prefs.get("emailNotifications", True),
        "auth_security_alerts": prefs.get("authSecurityAlerts", True),
        "digest_emails": prefs.get("digestEmails", True),
        "scheduled_alert_emails": prefs.get("scheduledAlertEmails", True),
        "discord_webhook": prefs.get("discordWebhook", None),
        "use_audio_context": prefs.get("useAudioContext", prefs.get("use_audio_context", True)),
        "useAudioContext": prefs.get("useAudioContext", prefs.get("use_audio_context", True)),
        "audio_transcription": prefs.get("audioTranscription", prefs.get("audio_transcription", True)),
        "audioTranscription": prefs.get("audioTranscription", prefs.get("audio_transcription", True)),
        "ai_service_telemetry": prefs.get("aiServiceTelemetry", prefs.get("ai_service_telemetry", True)),
        "aiServiceTelemetry": prefs.get("aiServiceTelemetry", prefs.get("ai_service_telemetry", True)),
        "ai_service_audio_signals": prefs.get("aiServiceAudioSignals", prefs.get("ai_service_audio_signals", True)),
        "aiServiceAudioSignals": prefs.get("aiServiceAudioSignals", prefs.get("ai_service_audio_signals", True)),
        "ai_service_music_detection": prefs.get("aiServiceMusicDetection", prefs.get("ai_service_music_detection", True)),
        "aiServiceMusicDetection": prefs.get("aiServiceMusicDetection", prefs.get("ai_service_music_detection", True)),
        "ai_service_audio_summary": prefs.get("aiServiceAudioSummary", prefs.get("ai_service_audio_summary", True)),
        "aiServiceAudioSummary": prefs.get("aiServiceAudioSummary", prefs.get("ai_service_audio_summary", True)),
        "ai_service_emotion_signals": prefs.get("aiServiceEmotionSignals", prefs.get("ai_service_emotion_signals", True)),
        "aiServiceEmotionSignals": prefs.get("aiServiceEmotionSignals", prefs.get("ai_service_emotion_signals", True)),
        "ai_service_caption_writer": prefs.get("aiServiceCaptionWriter", prefs.get("ai_service_caption_writer", True)),
        "aiServiceCaptionWriter": prefs.get("aiServiceCaptionWriter", prefs.get("ai_service_caption_writer", True)),
        "ai_service_thumbnail_designer": prefs.get("aiServiceThumbnailDesigner", prefs.get("ai_service_thumbnail_designer", True)),
        "aiServiceThumbnailDesigner": prefs.get("aiServiceThumbnailDesigner", prefs.get("ai_service_thumbnail_designer", True)),
        "ai_service_frame_inspector": prefs.get("aiServiceFrameInspector", prefs.get("ai_service_frame_inspector", True)),
        "aiServiceFrameInspector": prefs.get("aiServiceFrameInspector", prefs.get("ai_service_frame_inspector", True)),
        "ai_service_speech_to_text": prefs.get("aiServiceSpeechToText", prefs.get("ai_service_speech_to_text", True)),
        "aiServiceSpeechToText": prefs.get("aiServiceSpeechToText", prefs.get("ai_service_speech_to_text", True)),
        "ai_service_video_analyzer": prefs.get("aiServiceVideoAnalyzer", prefs.get("ai_service_video_analyzer", True)),
        "aiServiceVideoAnalyzer": prefs.get("aiServiceVideoAnalyzer", prefs.get("ai_service_video_analyzer", True)),
        "ai_service_scene_understanding": prefs.get("aiServiceSceneUnderstanding", prefs.get("ai_service_scene_understanding", True)),
        "aiServiceSceneUnderstanding": prefs.get("aiServiceSceneUnderstanding", prefs.get("ai_service_scene_understanding", True)),
        "thumbnail_studio_enabled": prefs.get("thumbnailStudioEnabled", prefs.get("thumbnail_studio_enabled", False)),
        "thumbnailStudioEnabled": prefs.get("thumbnailStudioEnabled", prefs.get("thumbnail_studio_enabled", False)),
        "thumbnail_studio_engine_enabled": bool(_thumb_eng) if _thumb_eng is not None else False,
        "thumbnailStudioEngineEnabled": bool(_thumb_eng) if _thumb_eng is not None else False,
        "thumbnail_persona_enabled": prefs.get("thumbnailPersonaEnabled", prefs.get("thumbnail_persona_enabled", False)),
        "thumbnailPersonaEnabled": prefs.get("thumbnailPersonaEnabled", prefs.get("thumbnail_persona_enabled", False)),
        "thumbnail_default_persona_id": prefs.get("thumbnailDefaultPersonaId", prefs.get("thumbnail_default_persona_id")),
        "thumbnailDefaultPersonaId": prefs.get("thumbnailDefaultPersonaId", prefs.get("thumbnail_default_persona_id")),
        "thumbnail_persona_strength": int(prefs.get("thumbnailPersonaStrength", prefs.get("thumbnail_persona_strength", 70)) or 70),
        "thumbnailPersonaStrength": int(prefs.get("thumbnailPersonaStrength", prefs.get("thumbnail_persona_strength", 70)) or 70),
        "thumbnail_use_studio_engine": bool(_thumb_use_eng) if _thumb_use_eng is not None else False,
        "thumbnailUseStudioEngine": bool(_thumb_use_eng) if _thumb_use_eng is not None else False,
        "thumbnail_use_persona": prefs.get("thumbnailUsePersona", prefs.get("thumbnail_use_persona", False)),
        "thumbnailUsePersona": prefs.get("thumbnailUsePersona", prefs.get("thumbnail_use_persona", False)),
        "thumbnail_persona_id": prefs.get("thumbnailPersonaId", prefs.get("thumbnail_persona_id")),
        "thumbnailPersonaId": prefs.get("thumbnailPersonaId", prefs.get("thumbnail_persona_id")),
    }
    # Add hud_enabled from user_settings for fallback path
    try:
        us_row = await conn.fetchrow("SELECT hud_enabled FROM user_settings WHERE user_id = $1", user_id)
        if us_row and us_row.get("hud_enabled") is not None:
            out["hud_enabled"] = bool(us_row["hud_enabled"])
    except Exception as e:
        logger.debug("get_user_settings fallback: hud_enabled overlay failed user_id=%s: %s", user_id, e)
    _strip_legacy_thumbnail_engine_keys(out)
    return out



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
async def get_platforms(user: dict = Depends(get_current_user_readonly)):
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at, last_oauth_reconnect_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])
    
    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms: platforms[p] = []
        first_at = acc["created_at"].isoformat() if acc["created_at"] else None
        _lr = acc.get("last_oauth_reconnect_at") or acc.get("created_at")
        reconnect_at = _lr.isoformat() if _lr else None
        platforms[p].append({
            "id": str(acc["id"]),
            "account_id": acc["account_id"],
            "name": acc["account_name"],
            "username": acc["account_username"],
            "avatar": _platform_account_avatar_to_url(acc["account_avatar"]),
            "is_primary": acc["is_primary"],
            "status": "active",
            "connected_at": first_at,
            "first_connected_at": first_at,
            "last_reconnected_at": reconnect_at,
        })
    
    ent = get_entitlements_from_user(dict(user))
    plan = entitlements_to_dict(ent)
    total = sum(len(v) for v in platforms.values())
    max_accounts = int(plan.get("max_accounts", 1) or 1)
    return {"platforms": platforms, "total_accounts": total, "max_accounts": max_accounts, "can_add_more": total < max_accounts}

# Alias endpoint for frontend compatibility
@app.get("/api/platform-accounts")
async def get_platform_accounts(user: dict = Depends(get_current_user_readonly)):
    """Returns flat list of accounts for frontend compatibility"""
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at, last_oauth_reconnect_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])
    
    result = []
    for acc in accounts:
        first_at = acc["created_at"].isoformat() if acc["created_at"] else None
        _lr = acc.get("last_oauth_reconnect_at") or acc.get("created_at")
        reconnect_at = _lr.isoformat() if _lr else None
        result.append({
            "id": str(acc["id"]),
            "platform": acc["platform"],
            "account_id": acc["account_id"],
            "account_name": acc["account_name"],
            "account_username": acc["account_username"],
            "account_avatar_url": _platform_account_avatar_to_url(acc["account_avatar"]),
            "is_primary": acc["is_primary"],
            "status": "active",
            "connected_at": first_at,
            "first_connected_at": first_at,
            "last_reconnected_at": reconnect_at,
        })
    return {"accounts": result}

@app.get("/api/accounts")
async def get_accounts_simple(user: dict = Depends(get_current_user_readonly)):
    """Simple accounts list for dashboard"""
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch(
            """SELECT DISTINCT ON (platform, account_id) id, platform, account_name, account_username, account_avatar
               FROM platform_tokens
               WHERE user_id = $1 AND revoked_at IS NULL
                 AND account_id IS NOT NULL AND account_id <> ''
               ORDER BY platform, account_id, created_at DESC""",
            user["id"],
        )
    return [{"id": str(a["id"]), "platform": a["platform"], "name": a["account_name"], "username": a["account_username"], "avatar": _platform_account_avatar_to_url(a["account_avatar"]), "status": "active"} for a in accounts]

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
        "scope": "user.info.basic,user.info.stats,video.list,video.publish,video.upload",
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
        # Effective scope from services.meta_oauth (META_OAUTH_MODE / custom env).
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


# OAuth state storage — Redis-backed when available for multi-instance scaling,
# falls back to in-memory dict for single-instance / local dev.
_oauth_states_mem: Dict[str, dict] = {}
_OAUTH_STATE_TTL = 600  # 10 minutes

async def _oauth_state_set(key: str, data: dict):
    if redis_client:
        try:
            await redis_client.setex(f"oauth_state:{key}", _OAUTH_STATE_TTL, json.dumps(data))
            return
        except Exception as e:
            logger.debug("oauth_state_set: redis failed, using memory: %s", e)
    _oauth_states_mem[key] = data

async def _oauth_state_pop(key: str) -> dict | None:
    if redis_client:
        try:
            raw = await redis_client.getdel(f"oauth_state:{key}")
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug("oauth_state_pop: redis getdel failed: %s", e)
    return _oauth_states_mem.pop(key, None)

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
    # Register this exact URL with each provider. Use OAUTH_PUBLIC_BASE_URL when the app listens on
    # localhost but TikTok/Google/Meta must redirect to a public tunnel (ngrok, cloudflared, etc.).
    pub = (os.environ.get("OAUTH_PUBLIC_BASE_URL") or "").strip() or BASE_URL
    return f"{pub.rstrip('/')}/api/oauth/{platform}/callback"


def _tiktok_pkce_verifier_and_challenge() -> tuple[str, str]:
    """RFC 7636 S256: TikTok may require code_challenge on authorize and code_verifier on token exchange."""
    verifier = secrets.token_urlsafe(48)
    if len(verifier) < 43:
        verifier = secrets.token_urlsafe(32) + secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _sanitize_oauth_parent_origin(raw: Optional[str]) -> str:
    """
    Target origin for window.opener.postMessage after OAuth. Must match the page that opened the popup
    (typically window.location.origin), not necessarily FRONTEND_URL.
    """
    default = FRONTEND_URL.rstrip("/")
    if not raw or not str(raw).strip():
        return default
    cand = str(raw).strip().rstrip("/")
    try:
        p = urlparse(cand if "://" in cand else f"http://{cand}")
        if p.scheme not in ("http", "https") or not p.hostname:
            return default
        host = (p.hostname or "").lower()
        cand_norm = f"{p.scheme}://{p.netloc}".rstrip("/")
        fe = urlparse(FRONTEND_URL if "://" in FRONTEND_URL else f"https://{FRONTEND_URL}")
        if (p.scheme, p.netloc) == (fe.scheme, fe.netloc):
            return cand_norm
        bu = urlparse(BASE_URL if "://" in BASE_URL else f"https://{BASE_URL}")
        if (p.scheme, p.netloc) == (bu.scheme, bu.netloc):
            return cand_norm
        if host in ("localhost", "127.0.0.1"):
            return cand_norm
        if host.endswith("uploadm8.com"):
            return cand_norm
    except Exception as e:
        logger.debug("_sanitize_oauth_parent_origin: parse failed raw=%r: %s", raw, e)
    return default


@app.get("/api/oauth/{platform}/start")
async def oauth_start(
    platform: str,
    parent_origin: Optional[str] = Query(None),
    force_login: bool = Query(False, description="Force account chooser/reauth where provider supports it"),
    reconnect_account_id: Optional[str] = Query(None, description="Existing platform_tokens.id to reconnect"),
    user: dict = Depends(get_current_user),
):
    """Start OAuth flow for a platform"""
    if platform not in OAUTH_CONFIG:
        raise HTTPException(400, f"Unsupported platform: {platform}")

    # Account limits are enforced in the OAuth callback on *new* rows only, so users at the limit
    # can still reconnect (UPDATE) existing platform identities.

    config = OAUTH_CONFIG[platform]
    state = secrets.token_urlsafe(32)

    reconnect_target = None
    if reconnect_account_id:
        async with db_pool.acquire() as conn:
            reconnect_target = await conn.fetchrow(
                """
                SELECT id, account_id
                FROM platform_tokens
                WHERE id = $1
                  AND user_id = $2
                  AND platform = $3
                  AND revoked_at IS NULL
                """,
                reconnect_account_id,
                str(user["id"]),
                platform,
            )
        if not reconnect_target:
            raise HTTPException(404, "Reconnect target account not found")

    tiktok_code_verifier: str | None = None
    if platform == "tiktok":
        tiktok_code_verifier, tiktok_code_challenge = _tiktok_pkce_verifier_and_challenge()

    # Store state with user info
    state_payload: dict = {
        "user_id": str(user["id"]),
        "platform": platform,
        "created_at": _now_utc().isoformat(),
        "parent_origin": _sanitize_oauth_parent_origin(parent_origin),
        "reconnect_account_id": str(reconnect_target["id"]) if reconnect_target else None,
        "reconnect_expected_provider_account_id": str(reconnect_target["account_id"]) if reconnect_target else None,
    }
    if tiktok_code_verifier:
        state_payload["tiktok_code_verifier"] = tiktok_code_verifier
    await _oauth_state_set(state, state_payload)

    redirect_uri = get_oauth_redirect_uri(platform)

    if platform == "tiktok":
        params = {
            "client_key": TIKTOK_CLIENT_KEY,
            "scope": config["scope"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": tiktok_code_challenge,
            "code_challenge_method": "S256",
        }
        if force_login:
            params["prompt"] = "login"
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
            "scope": meta_instagram_oauth_scope(),
            "response_type": "code",
            "state": state,
            "auth_type": "rerequest",  # Force re-authentication
        }
        if force_login:
            params["auth_nonce"] = secrets.token_hex(12)
    elif platform == "facebook":
        params = {
            "client_id": FACEBOOK_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": meta_facebook_oauth_scope(),
            "response_type": "code",
            "state": state,
            "auth_type": "rerequest",  # Force re-authentication
        }
        if force_login:
            params["auth_nonce"] = secrets.token_hex(12)
    
    auth_url = f"{config['auth_url']}?{urlencode(params)}"
    return {"auth_url": auth_url, "state": state}


@app.get("/api/oauth/meta/config")
async def meta_oauth_config_public():
    """
    Which Meta OAuth scope bundle the server requests (for App Review demos vs production).
    Does not expose app secrets.
    """
    return {
        "meta_oauth_mode": meta_oauth_mode(),
        "instagram_scope": meta_instagram_oauth_scope(),
        "facebook_scope": meta_facebook_oauth_scope(),
        "notes": (
            "META_OAUTH_MODE=minimal requests only pages_show_list, pages_read_engagement, business_management "
            "for reviewer login and listing Pages; publishing and most insights require full mode after approval."
        ),
    }


@app.get("/api/oauth/{platform}/callback")
async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):
    """Handle OAuth callback - returns HTML that communicates with parent window"""
    post_target = FRONTEND_URL.rstrip("/")
    state_data = None
    if state:
        state_data = await _oauth_state_pop(state)
        if state_data:
            post_target = _sanitize_oauth_parent_origin(state_data.get("parent_origin"))

    def popup_response(success: bool, platform: str, error_msg: str = None):
        """Generate HTML that posts message to parent window and closes popup"""
        if success:
            message = f'{{"type": "oauth_success", "platform": "{platform}"}}'
        else:
            safe_error = (error_msg or "Unknown error").replace('"', '\\"').replace('\n', ' ')[:200]
            message = f'{{"type": "oauth_error", "platform": "{platform}", "error": "{safe_error}"}}'
        target_js = json.dumps(post_target)

        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Connecting...</title></head>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #1a1a2e; color: white;">
            <div style="text-align: center;">
                <p>{" Connected successfully!" if success else " Connection failed"}</p>
                <p style="color: #888; font-size: 14px;">This window will close automatically...</p>
            </div>
            <script>
                if (window.opener) {{
                    window.opener.postMessage({message}, {target_js});
                }}
                setTimeout(() => window.close(), 1500);
            </script>
        </body>
        </html>
        """)

    if error:
        return popup_response(False, platform, error)

    if not state_data:
        return popup_response(False, platform, "Invalid or expired session. Please try again.")

    user_id = state_data["user_id"]
    reconnect_row_id = state_data.get("reconnect_account_id")
    existing_reconnect_profile = None
    if reconnect_row_id:
        try:
            async with db_pool.acquire() as conn:
                existing_reconnect_profile = await conn.fetchrow(
                    """
                    SELECT account_id, account_name, account_username, account_avatar
                    FROM platform_tokens
                    WHERE id = $1
                      AND user_id = $2
                      AND platform = $3
                    """,
                    reconnect_row_id,
                    user_id,
                    platform,
                )
        except Exception:
            existing_reconnect_profile = None

    if not code:
        return popup_response(False, platform, "No authorization code received")
    
    config = OAUTH_CONFIG[platform]
    redirect_uri = get_oauth_redirect_uri(platform)
    
    try:
        # follow_redirects: TikTok docs use curl -L for user/info; redirects would otherwise yield non-JSON bodies.
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # Exchange code for tokens based on platform
            token_payload = {}
            if platform == "tiktok":
                from services.tiktok_api import (
                    fetch_tiktok_user_profile_for_oauth,
                    tiktok_parse_oauth_token_response,
                )

                token_body = {
                    "client_key": TIKTOK_CLIENT_KEY,
                    "client_secret": TIKTOK_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                }
                cv = (state_data or {}).get("tiktok_code_verifier")
                if cv:
                    token_body["code_verifier"] = cv
                token_response = await client.post(config["token_url"], data=token_body)
                try:
                    token_data = token_response.json() if token_response.content else {}
                except Exception:
                    token_data = {}
                if token_response.status_code >= 400:
                    hint = (
                        token_data.get("error_description")
                        or token_data.get("description")
                        or token_data.get("error")
                        or (token_response.text or "")[:300]
                    )
                    raise Exception(f"TikTok token exchange failed ({token_response.status_code}): {hint}")

                access_token, token_open_id, _, token_payload = tiktok_parse_oauth_token_response(token_data)
                if not access_token:
                    raise Exception(
                        "TikTok did not return an access_token. Check client credentials, "
                        "redirect URI (must match portal, including OAUTH_PUBLIC_BASE_URL if used), and PKCE."
                    )

                account_id = token_open_id or secrets.token_hex(8)
                account_name = "TikTok User"
                account_username = ""
                account_avatar = ""

                prof = await fetch_tiktok_user_profile_for_oauth(client, access_token)
                if prof.get("account_id"):
                    account_id = str(prof["account_id"]).strip() or account_id
                if prof.get("account_name"):
                    account_name = str(prof["account_name"]).strip() or account_name
                if prof.get("account_username"):
                    account_username = str(prof["account_username"]).strip()
                if prof.get("account_avatar"):
                    account_avatar = str(prof["account_avatar"]).strip()

                # Reconnect fallback: keep prior profile fields if provider didn't return them.
                if reconnect_row_id and existing_reconnect_profile:
                    if not account_name or account_name == "TikTok User":
                        prev_nm = (existing_reconnect_profile.get("account_name") or "").strip()
                        if prev_nm and prev_nm != "TikTok User":
                            account_name = prev_nm
                    if (not account_username or not str(account_username).strip()) and existing_reconnect_profile.get(
                        "account_username"
                    ):
                        account_username = existing_reconnect_profile.get("account_username")
                    if (not account_avatar or not str(account_avatar).strip()) and existing_reconnect_profile.get(
                        "account_avatar"
                    ):
                        account_avatar = existing_reconnect_profile.get("account_avatar")
                if account_name == "TikTok User":
                    if account_username:
                        account_name = account_username
                    elif account_id:
                        aid = str(account_id)
                        account_name = f"TikTok {aid[-6:]}" if len(aid) > 6 else f"TikTok {aid}"

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
                    # customUrl can be empty for channels without custom URL; use channel title as fallback
                    account_username = (snippet.get("customUrl") or "").strip() or account_name
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

                meta_perms = await fetch_granted_permissions(client, user_access_token)

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

                        # Profile fields require instagram_basic; minimal OAuth may only grant pages_* + business_management.
                        ig_details_response = await client.get(
                            f"https://graph.facebook.com/v18.0/{ig_account_id}?fields=id,username,name,profile_picture_url&access_token={page_token}"
                        )
                        if ig_details_response.status_code == 200:
                            instagram_account = ig_details_response.json()
                        else:
                            logger.warning(
                                "Instagram profile fetch HTTP %s (degraded identity): %s",
                                ig_details_response.status_code,
                                (ig_details_response.text or "")[:240],
                            )
                            instagram_account = {
                                "id": ig_account_id,
                                "username": "",
                                "name": f"Instagram account {ig_account_id}",
                                "profile_picture_url": "",
                            }

                        page_access_token = page_token
                        break
                
                if not instagram_account:
                    raise Exception("No Instagram Business account found connected to your Facebook Pages. Connect your Instagram Business/Creator account to a Facebook Page first.")
                
                account_id = instagram_account.get("id")
                account_name = instagram_account.get("name") or instagram_account.get("username", "Instagram Account")
                # username can be empty for some accounts; use name as fallback for display
                account_username = (instagram_account.get("username") or "").strip() or account_name
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

                meta_perms_fb = await fetch_granted_permissions(client, user_access_token)

                # Facebook Reels require a Page token, not a user token.
                # Fetch the user's Pages and use the first one.
                pages_response = await client.get(
                    "https://graph.facebook.com/v18.0/me/accounts",
                    params={"access_token": user_access_token, "fields": "id,name,username,access_token,picture"},
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
                # Page username (e.g. for facebook.com/PageName); fallback to page name when empty
                account_username = (page.get("username") or "").strip() or account_name
                account_avatar   = page.get("picture", {}).get("data", {}).get("url", "")
                access_token     = page["access_token"]                            # Page token

            # Copy profile image into R2 — FB/IG/TikTok CDN URLs often 403 when hotlinked from the browser.
            _avatar_before_mirror = str(account_avatar)[:120] if account_avatar else ""
            if account_avatar and str(account_avatar).startswith("http") and platform in (
                "facebook",
                "instagram",
                "tiktok",
            ):
                try:
                    mirrored_key = await _mirror_oauth_profile_image_to_r2(
                        str(user_id), platform, str(account_avatar)
                    )
                    if mirrored_key:
                        account_avatar = mirrored_key
                except Exception as _av_e:
                    logger.debug(f"OAuth avatar mirror skipped ({platform}): {_av_e}")
            
            # Refuse to persist "ghost" connections with no identity
            if not account_id or (isinstance(account_id, str) and account_id.strip() == ""):
                raise Exception("Provider did not return account_id; refusing to store token (prevents phantom accounts).")

            # Store the token — include platform-specific IDs in the blob so
            # publish_stage can read them without needing a separate DB lookup.
            blob_payload = {
                "access_token": access_token,
                "refresh_token": token_data.get("refresh_token") or token_payload.get("refresh_token"),
                "expires_at": token_data.get("expires_in") or token_payload.get("expires_in"),
            }
            if platform in ("instagram", "facebook"):
                blob_payload["meta_oauth_mode"] = meta_oauth_mode()
                if platform == "instagram":
                    blob_payload["meta_permissions"] = meta_perms
                elif platform == "facebook":
                    blob_payload["meta_permissions"] = meta_perms_fb
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
                        account_avatar = $4, updated_at = NOW(), last_oauth_reconnect_at = NOW() WHERE id = $5
                    """, token_blob, account_name, account_username, account_avatar, existing["id"])
                    connect_action = "PLATFORM_RECONNECTED"
                else:
                    reconnect_row_id = state_data.get("reconnect_account_id")
                    reconnect_expected_provider_id = state_data.get("reconnect_expected_provider_account_id")
                    if reconnect_row_id:
                        if reconnect_expected_provider_id and str(reconnect_expected_provider_id) != str(account_id):
                            return popup_response(
                                False,
                                platform,
                                "You authenticated a different account. Please sign in to the same account you selected for reconnect.",
                            )
                        await conn.execute(
                            """
                            UPDATE platform_tokens
                            SET token_blob = $1,
                                account_name = $2,
                                account_username = $3,
                                account_avatar = $4,
                                account_id = $5,
                                updated_at = NOW(),
                                last_oauth_reconnect_at = NOW()
                            WHERE id = $6
                              AND user_id = $7
                              AND platform = $8
                            """,
                            token_blob,
                            account_name,
                            account_username,
                            account_avatar,
                            account_id,
                            reconnect_row_id,
                            user_id,
                            platform,
                        )
                        connect_action = "PLATFORM_RECONNECTED"
                        await log_system_event(
                            conn,
                            user_id=str(user_id),
                            action=connect_action,
                            event_category="PLATFORM",
                            resource_type="platform",
                            resource_id=f"{platform}:{account_id}",
                            details={
                                "platform": platform,
                                "account_name": account_name,
                                "account_username": account_username,
                                "reconnect_row_id": reconnect_row_id,
                            },
                        )
                        return popup_response(True, platform)
                    user_row = await conn.fetchrow(
                        "SELECT id, role, subscription_tier FROM users WHERE id = $1",
                        user_id,
                    )
                    current_count = int(await conn.fetchval(
                        "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
                        user_id,
                    ) or 0)
                    current_for_platform = int(await conn.fetchval(
                        "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND platform = $2 AND revoked_at IS NULL",
                        user_id, platform,
                    ) or 0)
                    allowed, reason = can_user_connect_platform(
                        dict(user_row or {}),
                        current_total=current_count,
                        current_for_platform=current_for_platform,
                    )
                    if not allowed:
                        return popup_response(
                            False,
                            platform,
                            reason,
                        )
                    await conn.execute("""
                        INSERT INTO platform_tokens (user_id, platform, account_id, account_name, account_username, account_avatar, token_blob, last_oauth_reconnect_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """, user_id, platform, account_id, account_name, account_username, account_avatar, token_blob)
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

    try:
        async with db_pool.acquire() as conn:
            # ── Double-subscribe guard ────────────────────────────────────
            if data.kind == "subscription":
                existing_sub_id = user.get("stripe_subscription_id")
                existing_status = (user.get("subscription_status") or "").lower()
                if existing_sub_id and existing_status in ("active", "trialing"):
                    cust = user.get("stripe_customer_id")
                    if not cust:
                        cust = await ensure_stripe_customer(conn, user, stripe)
                    try:
                        portal = stripe.billing_portal.Session.create(
                            customer=cust,
                            return_url=f"{FRONTEND_URL}/settings.html#billing",
                        )
                        return {"checkout_url": portal.url, "session_id": None, "portal_redirect": True}
                    except stripe.error.StripeError as e:
                        logger.warning(f"Stripe billing portal redirect failed user={user.get('id')}: {e}")

            customer_id = await ensure_stripe_customer(conn, user, stripe)

        session = create_billing_checkout_session(
            stripe_client=stripe,
            customer_id=customer_id,
            kind=data.kind,
            lookup_key=data.lookup_key,
            user_id=str(user["id"]),
            success_url=STRIPE_SUCCESS_URL,
            cancel_url=STRIPE_CANCEL_URL,
            topup_products=TOPUP_PRODUCTS,
        )

        return {"checkout_url": session.url, "session_id": session.id}
    except HTTPException:
        raise
    except stripe.error.StripeError as e:
        logger.error(f"Stripe checkout error user={user.get('id')}: {e}")
        msg = getattr(e, "user_message", None) or str(e) or "Billing provider error"
        raise HTTPException(502, msg)

@app.post("/api/billing/portal")
async def create_portal(background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    if not user.get("stripe_customer_id"): raise HTTPException(400, "No billing account")
    session = stripe.billing_portal.Session.create(customer=user["stripe_customer_id"], return_url=f"{FRONTEND_URL}/settings.html#billing")
    if user.get("email"):
        background_tasks.add_task(
            _send_billing_action_email,
            user.get("email"),
            user.get("name") or user.get("email") or "there",
            "Billing portal opened",
            "A secure Stripe billing portal session was requested from your account.",
        )
    return {"portal_url": session.url}


async def _send_billing_action_email(email: str, name: str, action_title: str, details_html: str):
    """Simple transactional note for billing manager actions initiated by the user."""
    try:
        html = f"""
        <div style="font-family:Inter,Arial,sans-serif;background:#0f172a;color:#e5e7eb;padding:24px">
          <div style="max-width:640px;margin:0 auto;background:#111827;border:1px solid #374151;border-radius:12px;padding:24px">
            <h2 style="margin:0 0 8px;color:#f8fafc;">{action_title}</h2>
            <p style="margin:0 0 16px;color:#9ca3af;">Hi {escape(email if not name else name)}, this confirms your recent billing action in UploadM8.</p>
            <div style="background:#0b1220;border:1px solid #334155;border-radius:10px;padding:14px 16px;line-height:1.6;color:#d1d5db;">
              {details_html}
            </div>
            <p style="margin:18px 0 0;">
              <a href="{URL_BILLING}" style="color:#fb923c;text-decoration:none;">Open Billing</a>
              &nbsp;·&nbsp;
              <a href="{URL_SETTINGS}" style="color:#fb923c;text-decoration:none;">Settings</a>
            </p>
          </div>
        </div>
        """
        await send_email(
            email,
            f"UploadM8 billing update: {action_title}",
            html,
            from_addr=MAIL_FROM_SUPPORT,
            reply_to=SUPPORT_EMAIL,
        )
    except Exception as e:
        logger.warning(f"billing action email failed for {email}: {e}")


@app.get("/api/billing/subscription/actions")
async def get_subscription_actions(user: dict = Depends(get_current_user)):
    """Actions shown in Settings > Billing manager dropdown."""
    can_manage = bool(user.get("stripe_subscription_id"))
    actions = [
        {
            "id": "pause_payment_collection",
            "label": "Pause payment collection",
            "requires_subscription": True,
        },
        {
            "id": "share_payment_update_link",
            "label": "Share payment update link",
            "requires_subscription": False,
        },
        {
            "id": "create_one_time_invoice",
            "label": "Create one-time invoice",
            "requires_subscription": False,
        },
        {
            "id": "cancel_subscription",
            "label": "Cancel subscription",
            "requires_subscription": True,
        },
    ]
    return {
        "can_manage_subscription": can_manage,
        "actions": actions,
    }


@app.post("/api/billing/subscription/action")
async def run_subscription_action(
    data: BillingSubscriptionActionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured")

    action = (data.action or "").strip().lower()
    sub_id = user.get("stripe_subscription_id")
    user_name = (user.get("name") or user.get("email") or "there")
    user_email = (user.get("email") or "").strip()

    async with db_pool.acquire() as conn:
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer_id = await ensure_stripe_customer(conn, user, stripe)

    try:
        if action == "pause_payment_collection":
            if not sub_id:
                raise HTTPException(400, "No active subscription to pause")
            sub = stripe.Subscription.modify(
                sub_id,
                pause_collection={"behavior": "keep_as_draft"},
            )
            if user_email:
                background_tasks.add_task(
                    _send_billing_action_email,
                    user_email,
                    user_name,
                    "Payment collection paused",
                    "Your subscription is still active, but automatic collection is paused until resumed in Stripe.",
                )
            return {"ok": True, "action": action, "status": sub.get("status"), "pause_collection": sub.get("pause_collection")}

        if action == "share_payment_update_link":
            try:
                portal = stripe.billing_portal.Session.create(
                    customer=customer_id,
                    return_url=f"{FRONTEND_URL}/settings.html#billing",
                    flow_data={"type": "payment_method_update"},
                )
            except Exception:
                portal = stripe.billing_portal.Session.create(
                    customer=customer_id,
                    return_url=f"{FRONTEND_URL}/settings.html#billing",
                )
            if user_email:
                background_tasks.add_task(
                    _send_billing_action_email,
                    user_email,
                    user_name,
                    "Payment update link created",
                    f"A secure payment-method update link was created.<br><br><a href='{escape(str(portal.url))}' style='color:#fb923c;'>Open payment update link</a>",
                )
            return {"ok": True, "action": action, "share_url": portal.url}

        if action == "create_one_time_invoice":
            amount_cents = int(data.amount_cents or 0)
            if amount_cents < 100:
                raise HTTPException(400, "amount_cents must be at least 100")
            currency = (data.currency or "usd").lower()
            desc = (data.description or "One-time UploadM8 invoice").strip()[:240]
            stripe.InvoiceItem.create(
                customer=customer_id,
                amount=amount_cents,
                currency=currency,
                description=desc,
            )
            inv = stripe.Invoice.create(
                customer=customer_id,
                auto_advance=True,
                collection_method="charge_automatically",
                description=desc,
            )
            finalized = stripe.Invoice.finalize_invoice(inv.id)
            if user_email:
                background_tasks.add_task(
                    _send_billing_action_email,
                    user_email,
                    user_name,
                    "One-time invoice created",
                    f"A one-time invoice for ${(amount_cents/100):.2f} {currency.upper()} was created with reference <code>{escape(str(finalized.id))}</code>.",
                )
            return {"ok": True, "action": action, "invoice_id": finalized.id, "invoice_url": finalized.get("hosted_invoice_url")}

        if action == "cancel_subscription":
            if not sub_id:
                raise HTTPException(400, "No active subscription to cancel")
            sub = stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
            period_end = sub.get("current_period_end")
            pretty_end = (
                datetime.fromtimestamp(period_end, tz=timezone.utc).strftime("%B %d, %Y")
                if period_end else "the period end"
            )
            if user_email:
                background_tasks.add_task(
                    _send_billing_action_email,
                    user_email,
                    user_name,
                    "Subscription cancellation scheduled",
                    f"Your subscription is set to cancel at period end ({escape(pretty_end)}). You retain access until then.",
                )
            return {"ok": True, "action": action, "cancel_at_period_end": bool(sub.get("cancel_at_period_end")), "current_period_end": period_end}

        raise HTTPException(400, "Unsupported action")
    except HTTPException:
        raise
    except stripe.error.StripeError as e:
        logger.error(f"billing action {action} failed user={user.get('id')}: {e}")
        msg = getattr(e, "user_message", None) or str(e) or "Stripe error"
        raise HTTPException(502, msg)


@app.get("/api/billing/overview")
async def get_billing_overview(user: dict = Depends(get_current_user)):
    """
    Consolidated billing snapshot for Settings > Billing.
    Includes current Stripe subscription status, payment method, and invoices.
    """
    out = {
        "customer_id": user.get("stripe_customer_id"),
        "subscription_id": user.get("stripe_subscription_id"),
        "subscription": None,
        "default_payment_method": None,
        "invoices": [],
        "portal_available": bool(user.get("stripe_customer_id")),
    }
    if not STRIPE_SECRET_KEY:
        return out

    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        return out

    try:
        customer = stripe.Customer.retrieve(
            customer_id,
            expand=["invoice_settings.default_payment_method"],
        )
        pm = (customer.get("invoice_settings") or {}).get("default_payment_method")
        if pm:
            card = pm.get("card") or {}
            out["default_payment_method"] = {
                "brand": card.get("brand"),
                "last4": card.get("last4"),
                "exp_month": card.get("exp_month"),
                "exp_year": card.get("exp_year"),
            }
    except Exception as e:
        logger.warning(f"billing overview: customer fetch failed for {customer_id}: {e}")

    sub_id = user.get("stripe_subscription_id")
    if sub_id:
        try:
            sub = stripe.Subscription.retrieve(
                sub_id,
                expand=["items.data.price", "default_payment_method"],
            )
            price = None
            try:
                price = sub["items"]["data"][0]["price"]
            except Exception:
                price = None
            tier = _tier_from_stripe_price(price or {}, user.get("subscription_tier") or "free")
            out["subscription"] = {
                "id": sub.get("id"),
                "status": sub.get("status"),
                "cancel_at_period_end": bool(sub.get("cancel_at_period_end")),
                "trial_end": sub.get("trial_end"),
                "current_period_start": sub.get("current_period_start"),
                "current_period_end": sub.get("current_period_end"),
                "tier": tier,
                "lookup_key": (price or {}).get("lookup_key"),
            }
        except Exception as e:
            logger.warning(f"billing overview: subscription fetch failed for {sub_id}: {e}")

    try:
        invoices = stripe.Invoice.list(customer=customer_id, limit=12)
        out["invoices"] = [
            {
                "id": inv.get("id"),
                "status": inv.get("status"),
                "amount_paid": ((inv.get("amount_paid") or 0) / 100.0),
                "amount_due": ((inv.get("amount_due") or 0) / 100.0),
                "currency": (inv.get("currency") or "usd").upper(),
                "created": inv.get("created"),
                "period_start": inv.get("period_start"),
                "period_end": inv.get("period_end"),
                "hosted_invoice_url": inv.get("hosted_invoice_url"),
                "invoice_pdf": inv.get("invoice_pdf"),
            }
            for inv in (invoices.data or [])
        ]
    except Exception as e:
        logger.warning(f"billing overview: invoice fetch failed for {customer_id}: {e}")

    return out


@app.post("/api/billing/upload-estimate")
async def estimate_upload_cost(
    data: UploadCostEstimateRequest,
    user: dict = Depends(get_current_user),
):
    """Canonical PUT/AIC estimator based on backend entitlement + per-service AIC model."""
    ent = get_entitlements_from_user(dict(user))
    async with db_pool.acquire() as conn:
        prefs = await get_user_prefs_for_upload(conn, user["id"])
    if not bool(data.use_ai):
        prefs = {
            **dict(prefs or {}),
            "auto_captions": False,
            "auto_thumbnails": False,
            "ai_hashtags_enabled": False,
            "use_audio_context": False,
            "audio_transcription": False,
            "aiServiceTelemetry": False,
            "aiServiceAudioSignals": False,
            "aiServiceMusicDetection": False,
            "aiServiceAudioSummary": False,
            "aiServiceEmotionSignals": False,
            "aiServiceCaptionWriter": False,
            "aiServiceThumbnailDesigner": False,
            "aiServiceFrameInspector": False,
            "aiServiceSpeechToText": False,
            "aiServiceVideoAnalyzer": False,
            "aiServiceSceneUnderstanding": False,
        }
    put_cost, aic_cost, breakdown = compute_upload_cost(
        entitlements=ent,
        num_platforms=max(1, int(data.num_publish_targets or 1)),
        use_ai=bool(data.use_ai),
        use_hud=bool(data.use_hud),
        num_thumbnails=max(1, int(data.num_thumbnails or 1)),
        duration_seconds=float(data.duration_seconds) if data.duration_seconds is not None else None,
        duration_hint=float(data.duration_seconds) if data.duration_seconds is not None else None,
        file_size=int(data.file_size) if data.file_size is not None else None,
        user_prefs=prefs,
        has_telemetry=bool(data.has_telemetry),
        return_breakdown=True,
    )
    return {
        "put_cost": int(put_cost),
        "aic_cost": int(aic_cost),
        "tier": ent.tier,
        "ai_depth": ent.ai_depth,
        "max_thumbnails": int(ent.max_thumbnails or 1),
        "aic_breakdown": breakdown,
    }


async def _thumbnail_channel_memory_hint(conn, user_id: str, niche: str) -> str:
    try:
        rows = await conn.fetch(
            """
            SELECT trv.variant_json
            FROM thumbnail_recreate_feedback tf
            JOIN thumbnail_recreate_variants trv ON trv.id = tf.variant_id
            JOIN thumbnail_recreate_jobs trj ON trj.id = tf.job_id
            WHERE tf.user_id = $1
              AND tf.event_type = 'selected'
              AND COALESCE(trj.niche, 'general') = $2
            ORDER BY tf.created_at DESC
            LIMIT 8
            """,
            user_id,
            (niche or "general").lower(),
        )
    except Exception:
        return ""
    bits: List[str] = []
    for r in rows or []:
        v = r.get("variant_json") if isinstance(r, dict) else r["variant_json"]
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                v = {}
        if isinstance(v, dict):
            bits.extend([str(v.get("emotion") or ""), str(v.get("text_position") or "")])
    cleaned = [b for b in bits if b]
    return ",".join(cleaned[:10])


async def _thumbnail_studio_debit(conn, user: dict, put_cost: int, aic_cost: int, meta: Dict[str, Any]) -> None:
    ent = get_entitlements_from_user(dict(user))
    if getattr(ent, "is_internal", False):
        return
    wallet = await get_wallet(conn, str(user["id"]))
    put_avail = int(wallet.get("put_balance", 0) or 0) - int(wallet.get("put_reserved", 0) or 0)
    aic_avail = int(wallet.get("aic_balance", 0) or 0) - int(wallet.get("aic_reserved", 0) or 0)
    if put_avail < put_cost:
        raise HTTPException(429, {"code": "insufficient_put", "message": f"Need {put_cost} PUT ({put_avail} available)."})
    if aic_avail < aic_cost:
        raise HTTPException(429, {"code": "insufficient_aic", "message": f"Need {aic_cost} AIC ({aic_avail} available)."})

    await conn.execute(
        "UPDATE wallets SET put_balance = put_balance - $1, aic_balance = aic_balance - $2 WHERE user_id = $3",
        int(put_cost),
        int(aic_cost),
        user["id"],
    )
    if put_cost > 0:
        await ledger_entry(conn, str(user["id"]), "put", -int(put_cost), "thumbnail_studio", meta=meta)
    if aic_cost > 0:
        await ledger_entry(conn, str(user["id"]), "aic", -int(aic_cost), "thumbnail_studio", meta=meta)


async def _pikzels_v2_user_call(user: dict, op: str, path: str, body: Dict[str, Any]) -> JSONResponse:
    from services.pikzels_v2_client import pikzels_api_key, pikzels_v2_post

    if not pikzels_api_key():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "pikzels_unconfigured",
                "message": "Set PIKZELS_API_KEY (canonical for api.pikzels.com) or THUMB_RENDER_API_KEY.",
            },
        )
    put_cost, aic_cost, _ = estimate_pikzels_v2_call_cost(op)
    async with db_pool.acquire() as conn:
        await _thumbnail_studio_debit(
            conn,
            user,
            put_cost,
            aic_cost,
            {"pikzels_v2_op": op, "source": "pikzels_v2_proxy"},
        )
    status, data = await pikzels_v2_post(path, body)
    return JSONResponse(status_code=status, content=data)


async def _pikzels_v2_admin_call(path: str, body: Dict[str, Any]) -> JSONResponse:
    from services.pikzels_v2_client import pikzels_api_key, pikzels_v2_post

    if not pikzels_api_key():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "pikzels_unconfigured",
                "message": "Set PIKZELS_API_KEY (canonical for api.pikzels.com) or THUMB_RENDER_API_KEY.",
            },
        )
    status, data = await pikzels_v2_post(path, body)
    return JSONResponse(status_code=status, content=data)


async def _pikzels_v2_get_response(path: str) -> JSONResponse:
    from services.pikzels_v2_client import pikzels_api_key, pikzels_v2_get

    if not pikzels_api_key():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "pikzels_unconfigured",
                "message": "Set PIKZELS_API_KEY (canonical for api.pikzels.com) or THUMB_RENDER_API_KEY.",
            },
        )
    status, data = await pikzels_v2_get(path)
    return JSONResponse(status_code=status, content=data)


@app.get("/api/thumbnail-studio/formats")
async def get_thumbnail_studio_formats(
    niche: Optional[str] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    rows: List[Dict[str, Any]] = []
    async with db_pool.acquire() as conn:
        existing = await conn.fetch("SELECT key, niche, name, pattern, social_proof FROM thumbnail_format_library ORDER BY niche, name")
        if not existing:
            for row in format_library_rows():
                await conn.execute(
                    """
                    INSERT INTO thumbnail_format_library (key, niche, name, pattern, social_proof)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (key) DO NOTHING
                    """,
                    row["key"],
                    row["niche"],
                    row["name"],
                    row["pattern"],
                    row.get("social_proof", ""),
                )
            existing = await conn.fetch("SELECT key, niche, name, pattern, social_proof FROM thumbnail_format_library ORDER BY niche, name")
        for r in existing or []:
            rows.append(
                {
                    "key": r["key"],
                    "niche": r["niche"],
                    "name": r["name"],
                    "pattern": r["pattern"],
                    "social_proof": r.get("social_proof") or "",
                }
            )
    n = (niche or "").strip().lower()
    if n:
        rows = [r for r in rows if str(r.get("niche") or "").lower() == n]
    return {"formats": rows}


@app.post("/api/thumbnail-studio/estimate")
async def estimate_thumbnail_studio(
    payload: ThumbnailStudioEstimateRequest,
    user: dict = Depends(get_current_user),
):
    put_cost, aic_cost, breakdown = estimate_studio_cost(
        variant_count=payload.variant_count,
        has_persona=bool(payload.has_persona),
        competitor_gap_mode=bool(payload.competitor_gap_mode),
        has_channel_memory=bool(payload.has_channel_memory),
    )
    return {"put_cost": put_cost, "aic_cost": aic_cost, "breakdown": breakdown}


@app.get("/api/thumbnail-studio/pikzels-v2-map")
async def thumbnail_studio_pikzels_v2_map(user: dict = Depends(get_current_user)):
    """Canonical map of product features → Pikzels public v2 paths (for integration planning)."""
    _ = user
    from services.pikzels_v2 import PUBLIC_BASE, feature_map_for_docs, public_api_key_source, resolve_public_api_key

    return {
        "public_base": PUBLIC_BASE,
        "docs_index": "https://docs.pikzels.com/llms.txt",
        "openapi": "https://docs.pikzels.com/openapi.json",
        "features": feature_map_for_docs(),
        "v2_api_key_configured": bool(resolve_public_api_key()),
        "v2_api_key_source": public_api_key_source(),
        "note": (
            "Local persona library may store 3–20 reference images; POST /api/thumbnail-studio/pikzels-v2/persona "
            "forwards the first 3 to Pikzels. User routes debit PUT/AIC per call (see estimate_pikzels_v2_call_cost)."
        ),
        "proxy_routes": "/api/thumbnail-studio/pikzels-v2/* (authenticated) · /api/admin/pikzels-v2/* (admin, no debit)",
    }


@app.post("/api/thumbnail-studio/pikzels-v2/prompt")
async def ts_pikzels_v2_prompt(body: PikzelsV2PromptBody, user: dict = Depends(get_current_user)):
    """Pikzels Prompt — text-to-thumbnail (POST /v2/thumbnail/text)."""
    from services.pikzels_v2 import V2_THUMBNAIL_TEXT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    return await _pikzels_v2_user_call(user, "prompt", V2_THUMBNAIL_TEXT, j)


@app.post("/api/thumbnail-studio/pikzels-v2/recreate")
async def ts_pikzels_v2_recreate(body: PikzelsV2RecreateBody, user: dict = Depends(get_current_user)):
    """Pikzels Recreate™ — create-from-image (POST /v2/thumbnail/image)."""
    from services.pikzels_v2 import V2_THUMBNAIL_IMAGE
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_user_call(user, "recreate", V2_THUMBNAIL_IMAGE, j)


@app.post("/api/thumbnail-studio/pikzels-v2/edit")
async def ts_pikzels_v2_edit(body: PikzelsV2EditBody, user: dict = Depends(get_current_user)):
    """Pikzels Edit — inpaint / refine (POST /v2/thumbnail/edit)."""
    from services.pikzels_v2 import V2_THUMBNAIL_EDIT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "mask_url", "mask_base64")
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_user_call(user, "edit", V2_THUMBNAIL_EDIT, j)


@app.post("/api/thumbnail-studio/pikzels-v2/one-click-fix")
async def ts_pikzels_v2_one_click_fix(body: PikzelsV2EditBody, user: dict = Depends(get_current_user)):
    """One-Click Fix™ — same v2 edit endpoint; distinct UX label + ledger op."""
    from services.pikzels_v2 import V2_THUMBNAIL_EDIT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "mask_url", "mask_base64")
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_user_call(user, "one_click_fix", V2_THUMBNAIL_EDIT, j)


@app.post("/api/thumbnail-studio/pikzels-v2/faceswap")
async def ts_pikzels_v2_faceswap(body: PikzelsV2FaceswapBody, user: dict = Depends(get_current_user)):
    from services.pikzels_v2 import V2_THUMBNAIL_FACESWAP
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "face_image", "face_image_base64")
    normalize_url_or_base64(j, "mask_url", "mask_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64 for target image.")
    if not j.get("face_image") and not j.get("face_image_base64"):
        raise HTTPException(400, "Provide face_image or face_image_base64.")
    return await _pikzels_v2_user_call(user, "faceswap", V2_THUMBNAIL_FACESWAP, j)


@app.post("/api/thumbnail-studio/pikzels-v2/score")
async def ts_pikzels_v2_score(body: PikzelsV2ScoreBody, user: dict = Depends(get_current_user)):
    """Pikzels Score™ (POST /v2/thumbnail/score)."""
    from services.pikzels_v2 import V2_THUMBNAIL_SCORE
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_user_call(user, "score", V2_THUMBNAIL_SCORE, j)


@app.post("/api/thumbnail-studio/pikzels-v2/titles")
async def ts_pikzels_v2_titles(body: PikzelsV2TitlesBody, user: dict = Depends(get_current_user)):
    """Title suggestions (POST /v2/title/text)."""
    from services.pikzels_v2 import V2_TITLE_TEXT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not (j.get("prompt") or j.get("support_image_url") or j.get("support_image_base64")):
        raise HTTPException(400, "Provide prompt and/or a support image (URL or base64 data URL).")
    return await _pikzels_v2_user_call(user, "titles", V2_TITLE_TEXT, j)


@app.post("/api/thumbnail-studio/pikzels-v2/persona")
async def ts_pikzels_v2_persona(body: PikzelsV2PikzonalityBody, user: dict = Depends(get_current_user)):
    """Pikzonality Persona (POST /v2/pikzonality/persona) — first 3 images sent to Pikzels."""
    from services.pikzels_v2 import V2_PIKZONALITY_PERSONA
    from services.pikzels_v2_client import trim_pikzonality_images

    j = body.model_dump(exclude_none=True)
    trim_pikzonality_images(j)
    nu, nb = j.get("image_urls") or [], j.get("image_base64s") or []
    if len(nu) < 3 and len(nb) < 3:
        raise HTTPException(400, "Provide at least 3 image_urls or 3 image_base64s.")
    return await _pikzels_v2_user_call(user, "persona", V2_PIKZONALITY_PERSONA, j)


@app.post("/api/thumbnail-studio/pikzels-v2/style")
async def ts_pikzels_v2_style(body: PikzelsV2PikzonalityBody, user: dict = Depends(get_current_user)):
    """Pikzonality Style (POST /v2/pikzonality/style)."""
    from services.pikzels_v2 import V2_PIKZONALITY_STYLE
    from services.pikzels_v2_client import trim_pikzonality_images

    j = body.model_dump(exclude_none=True)
    trim_pikzonality_images(j)
    nu, nb = j.get("image_urls") or [], j.get("image_base64s") or []
    if len(nu) < 3 and len(nb) < 3:
        raise HTTPException(400, "Provide at least 3 image_urls or 3 image_base64s.")
    return await _pikzels_v2_user_call(user, "style", V2_PIKZONALITY_STYLE, j)


@app.get("/api/thumbnail-studio/pikzels-v2/pikzonality/{pikzonality_id}")
async def ts_pikzels_v2_pikzonality_poll(pikzonality_id: str, user: dict = Depends(get_current_user)):
    """Poll Pikzonality status (GET /v2/pikzonality/{id}). No token debit."""
    _ = user
    from services.pikzels_v2 import V2_PIKZONALITY_BY_ID

    path = V2_PIKZONALITY_BY_ID.format(id=pikzonality_id.strip())
    return await _pikzels_v2_get_response(path)


@app.post("/api/admin/pikzels-v2/prompt")
async def admin_pikzels_v2_prompt(body: PikzelsV2PromptBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_THUMBNAIL_TEXT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    return await _pikzels_v2_admin_call(V2_THUMBNAIL_TEXT, j)


@app.post("/api/admin/pikzels-v2/recreate")
async def admin_pikzels_v2_recreate(body: PikzelsV2RecreateBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_THUMBNAIL_IMAGE
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_admin_call(V2_THUMBNAIL_IMAGE, j)


@app.post("/api/admin/pikzels-v2/edit")
async def admin_pikzels_v2_edit(body: PikzelsV2EditBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_THUMBNAIL_EDIT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "mask_url", "mask_base64")
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_admin_call(V2_THUMBNAIL_EDIT, j)


@app.post("/api/admin/pikzels-v2/one-click-fix")
async def admin_pikzels_v2_one_click_fix(body: PikzelsV2EditBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_THUMBNAIL_EDIT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "mask_url", "mask_base64")
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_admin_call(V2_THUMBNAIL_EDIT, j)


@app.post("/api/admin/pikzels-v2/faceswap")
async def admin_pikzels_v2_faceswap(body: PikzelsV2FaceswapBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_THUMBNAIL_FACESWAP
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    normalize_url_or_base64(j, "face_image", "face_image_base64")
    normalize_url_or_base64(j, "mask_url", "mask_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64 for target image.")
    if not j.get("face_image") and not j.get("face_image_base64"):
        raise HTTPException(400, "Provide face_image or face_image_base64.")
    return await _pikzels_v2_admin_call(V2_THUMBNAIL_FACESWAP, j)


@app.post("/api/admin/pikzels-v2/score")
async def admin_pikzels_v2_score(body: PikzelsV2ScoreBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_THUMBNAIL_SCORE
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "image_url", "image_base64")
    if not j.get("image_url") and not j.get("image_base64"):
        raise HTTPException(400, "Provide image_url or image_base64.")
    return await _pikzels_v2_admin_call(V2_THUMBNAIL_SCORE, j)


@app.post("/api/admin/pikzels-v2/titles")
async def admin_pikzels_v2_titles(body: PikzelsV2TitlesBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_TITLE_TEXT
    from services.pikzels_v2_client import normalize_url_or_base64

    j = body.model_dump(exclude_none=True)
    normalize_url_or_base64(j, "support_image_url", "support_image_base64")
    if not (j.get("prompt") or j.get("support_image_url") or j.get("support_image_base64")):
        raise HTTPException(400, "Provide prompt and/or a support image.")
    return await _pikzels_v2_admin_call(V2_TITLE_TEXT, j)


@app.post("/api/admin/pikzels-v2/persona")
async def admin_pikzels_v2_persona(body: PikzelsV2PikzonalityBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_PIKZONALITY_PERSONA
    from services.pikzels_v2_client import trim_pikzonality_images

    j = body.model_dump(exclude_none=True)
    trim_pikzonality_images(j)
    nu, nb = j.get("image_urls") or [], j.get("image_base64s") or []
    if len(nu) < 3 and len(nb) < 3:
        raise HTTPException(400, "Provide at least 3 image_urls or 3 image_base64s.")
    return await _pikzels_v2_admin_call(V2_PIKZONALITY_PERSONA, j)


@app.post("/api/admin/pikzels-v2/style")
async def admin_pikzels_v2_style(body: PikzelsV2PikzonalityBody, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_PIKZONALITY_STYLE
    from services.pikzels_v2_client import trim_pikzonality_images

    j = body.model_dump(exclude_none=True)
    trim_pikzonality_images(j)
    nu, nb = j.get("image_urls") or [], j.get("image_base64s") or []
    if len(nu) < 3 and len(nb) < 3:
        raise HTTPException(400, "Provide at least 3 image_urls or 3 image_base64s.")
    return await _pikzels_v2_admin_call(V2_PIKZONALITY_STYLE, j)


@app.get("/api/admin/pikzels-v2/pikzonality/{pikzonality_id}")
async def admin_pikzels_v2_pikzonality_poll(pikzonality_id: str, user: dict = Depends(require_admin)):
    _ = user
    from services.pikzels_v2 import V2_PIKZONALITY_BY_ID

    path = V2_PIKZONALITY_BY_ID.format(id=pikzonality_id.strip())
    return await _pikzels_v2_get_response(path)


@app.get("/api/thumbnail-studio/personas")
async def list_thumbnail_personas(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, profile_json, image_count, quality_score, created_at, updated_at
            FROM creator_personas
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT 100
            """,
            user["id"],
        )
    out = []
    for r in rows or []:
        out.append(
            {
                "id": str(r["id"]),
                "name": r["name"],
                "profile": dict(r["profile_json"] or {}),
                "image_count": int(r.get("image_count") or 0),
                "quality_score": float(r.get("quality_score") or 0),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            }
        )
    return {"personas": out}


@app.post("/api/thumbnail-studio/personas")
async def create_thumbnail_persona(payload: ThumbnailPersonaCreateRequest, user: dict = Depends(get_current_user)):
    image_urls = [str(x).strip() for x in (payload.image_urls or []) if str(x).strip()]
    if len(image_urls) < 3 or len(image_urls) > 20:
        raise HTTPException(400, "Provide between 3 and 20 selfie images.")

    persona_id = str(uuid.uuid4())
    profile_json = {
        "expressions": payload.expressions or [],
        "lighting_presets": payload.lighting_presets or [],
        "scene_prefs": payload.scene_prefs or [],
    }
    quality_score = round(72.0 + min(22.0, len(image_urls) * 1.8), 2)

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO creator_personas (id, user_id, name, profile_json, image_count, quality_score)
            VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5, $6)
            """,
            persona_id,
            user["id"],
            payload.name.strip(),
            json.dumps(profile_json),
            len(image_urls),
            quality_score,
        )
        for u in image_urls:
            await conn.execute(
                """
                INSERT INTO creator_persona_images (persona_id, user_id, image_url, quality_json)
                VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
                """,
                persona_id,
                user["id"],
                u,
                json.dumps({"ok": True}),
            )

    return {
        "id": persona_id,
        "name": payload.name.strip(),
        "image_count": len(image_urls),
        "quality_score": quality_score,
        "profile": profile_json,
    }


@app.post("/api/thumbnail-studio/recreate")
async def create_thumbnail_recreate_job(payload: ThumbnailRecreateRequest, user: dict = Depends(get_current_user)):
    youtube_url = (payload.youtube_url or "").strip()
    video_id = extract_youtube_video_id(youtube_url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL.")

    source_title = await fetch_youtube_title(youtube_url)
    topic = (payload.topic or source_title or "").strip()
    if not topic:
        topic = "Untitled concept"
    niche = (payload.niche or "general").strip().lower()

    persona_name = ""
    persona_id = payload.persona_id if payload.persona_id and _valid_uuid(payload.persona_id) else None

    async with db_pool.acquire() as conn:
        if persona_id:
            prow = await conn.fetchrow(
                "SELECT id, name FROM creator_personas WHERE id = $1::uuid AND user_id = $2::uuid",
                persona_id,
                user["id"],
            )
            if not prow:
                raise HTTPException(404, "Persona not found.")
            persona_name = str(prow.get("name") or "")

        channel_memory_hint = await _thumbnail_channel_memory_hint(conn, str(user["id"]), niche)
        put_cost, aic_cost, breakdown = estimate_studio_cost(
            variant_count=payload.variant_count,
            has_persona=bool(persona_id),
            competitor_gap_mode=bool(payload.competitor_gap_mode),
            has_channel_memory=bool(channel_memory_hint),
        )
        await _thumbnail_studio_debit(
            conn,
            user,
            put_cost,
            aic_cost,
            {"feature": "thumbnail_studio", "video_id": video_id, "variant_count": payload.variant_count},
        )

        variants = generate_recreate_variants(
            youtube_title=source_title,
            topic=topic,
            niche=niche,
            closeness=payload.closeness,
            variant_count=payload.variant_count,
            persona_name=persona_name,
            competitor_gap_mode=bool(payload.competitor_gap_mode),
            channel_memory_hint=channel_memory_hint,
        )
        job_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO thumbnail_recreate_jobs
            (id, user_id, youtube_url, youtube_video_id, source_title, topic, niche, closeness, variant_count,
             persona_id, competitor_gap_mode, put_cost, aic_cost, breakdown_json)
            VALUES
            ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10::uuid, $11, $12, $13, $14::jsonb)
            """,
            job_id,
            user["id"],
            youtube_url,
            video_id,
            source_title or None,
            topic,
            niche,
            int(payload.closeness),
            int(payload.variant_count),
            persona_id,
            bool(payload.competitor_gap_mode),
            put_cost,
            aic_cost,
            json.dumps(breakdown),
        )

        for idx, v in enumerate(variants, start=1):
            await conn.execute(
                """
                INSERT INTO thumbnail_recreate_variants (job_id, user_id, rank_idx, variant_json)
                VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
                """,
                job_id,
                user["id"],
                idx,
                json.dumps(v),
            )

    return {
        "job_id": job_id,
        "youtube_video_id": video_id,
        "source_title": source_title,
        "topic": topic,
        "niche": niche,
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "breakdown": breakdown,
        "variants": variants,
        "channel_memory_hint_used": bool(channel_memory_hint),
        "competitor_gap_mode": bool(payload.competitor_gap_mode),
    }


@app.get("/api/thumbnail-studio/jobs/{job_id}")
async def get_thumbnail_recreate_job(job_id: str, user: dict = Depends(get_current_user)):
    if not _valid_uuid(job_id):
        raise HTTPException(400, "Invalid job id.")
    async with db_pool.acquire() as conn:
        j = await conn.fetchrow(
            """
            SELECT id, youtube_url, youtube_video_id, source_title, topic, niche, closeness, variant_count,
                   persona_id, competitor_gap_mode, put_cost, aic_cost, breakdown_json, created_at
            FROM thumbnail_recreate_jobs
            WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            job_id,
            user["id"],
        )
        if not j:
            raise HTTPException(404, "Job not found.")
        rows = await conn.fetch(
            """
            SELECT id, rank_idx, variant_json, selected, publish_outcome, created_at
            FROM thumbnail_recreate_variants
            WHERE job_id = $1::uuid AND user_id = $2::uuid
            ORDER BY rank_idx ASC
            """,
            job_id,
            user["id"],
        )
    variants = []
    for r in rows or []:
        data = dict(r.get("variant_json") or {})
        data["variant_id"] = str(r["id"])
        data["selected"] = bool(r.get("selected"))
        data["publish_outcome"] = dict(r.get("publish_outcome") or {})
        variants.append(data)
    return {
        "job": {
            "id": str(j["id"]),
            "youtube_url": j["youtube_url"],
            "youtube_video_id": j["youtube_video_id"],
            "source_title": j["source_title"],
            "topic": j["topic"],
            "niche": j["niche"],
            "closeness": int(j["closeness"] or 0),
            "variant_count": int(j["variant_count"] or 0),
            "persona_id": str(j["persona_id"]) if j.get("persona_id") else None,
            "competitor_gap_mode": bool(j.get("competitor_gap_mode")),
            "put_cost": int(j.get("put_cost") or 0),
            "aic_cost": int(j.get("aic_cost") or 0),
            "breakdown": dict(j.get("breakdown_json") or {}),
            "created_at": j["created_at"].isoformat() if j.get("created_at") else None,
        },
        "variants": variants,
    }


@app.get("/api/thumbnail-studio/ab-export/{job_id}")
async def thumbnail_studio_ab_export(job_id: str, user: dict = Depends(get_current_user)):
    if not _valid_uuid(job_id):
        raise HTTPException(400, "Invalid job id.")
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, rank_idx, variant_json
            FROM thumbnail_recreate_variants
            WHERE job_id = $1::uuid AND user_id = $2::uuid
            ORDER BY rank_idx ASC
            LIMIT 3
            """,
            job_id,
            user["id"],
        )
    if not rows:
        raise HTTPException(404, "No variants found.")
    export_rows = []
    for i, r in enumerate(rows, start=1):
        v = dict(r.get("variant_json") or {})
        label = f"A/B-{i}"
        export_rows.append(
            {
                "variant_id": str(r["id"]),
                "label": label,
                "filename": f"thumb_{job_id[:8]}_{label.lower()}.jpg",
                "headline": v.get("headline"),
                "ctr_score": v.get("ctr_score"),
                "experiment_meta": {
                    "job_id": job_id,
                    "arm": label,
                    "niche": v.get("name"),
                },
            }
        )
    return {"job_id": job_id, "exports": export_rows}


@app.post("/api/thumbnail-studio/feedback")
async def thumbnail_studio_feedback(payload: ThumbnailFeedbackRequest, user: dict = Depends(get_current_user)):
    if not _valid_uuid(payload.job_id):
        raise HTTPException(400, "Invalid job id.")
    if payload.variant_id and not _valid_uuid(payload.variant_id):
        raise HTTPException(400, "Invalid variant id.")

    async with db_pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM thumbnail_recreate_jobs WHERE id = $1::uuid AND user_id = $2::uuid",
            payload.job_id,
            user["id"],
        )
        if not exists:
            raise HTTPException(404, "Job not found.")
        await conn.execute(
            """
            INSERT INTO thumbnail_recreate_feedback (user_id, job_id, variant_id, event_type, metadata)
            VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5::jsonb)
            """,
            user["id"],
            payload.job_id,
            payload.variant_id,
            payload.event_type,
            json.dumps(payload.metadata or {}),
        )
        if payload.event_type == "selected" and payload.variant_id:
            await conn.execute(
                "UPDATE thumbnail_recreate_variants SET selected = (id = $1::uuid) WHERE job_id = $2::uuid",
                payload.variant_id,
                payload.job_id,
            )
        if payload.event_type == "published_outcome" and payload.variant_id:
            await conn.execute(
                """
                UPDATE thumbnail_recreate_variants
                SET publish_outcome = COALESCE(publish_outcome, '{}'::jsonb) || $2::jsonb
                WHERE id = $1::uuid
                """,
                payload.variant_id,
                json.dumps(payload.metadata or {}),
            )
    return {"status": "ok"}


@app.get("/api/thumbnail-studio/weekly-digest")
async def thumbnail_studio_weekly_digest(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT trj.niche,
                   COUNT(*)::int AS variants_seen,
                   AVG(COALESCE((trv.variant_json->>'ctr_score')::double precision, 0))::double precision AS avg_ctr_score
            FROM thumbnail_recreate_variants trv
            JOIN thumbnail_recreate_jobs trj ON trj.id = trv.job_id
            WHERE trv.user_id = $1::uuid
              AND trv.created_at >= NOW() - INTERVAL '7 days'
            GROUP BY trj.niche
            ORDER BY avg_ctr_score DESC, variants_seen DESC
            LIMIT 6
            """,
            user["id"],
        )
        selected = await conn.fetch(
            """
            SELECT COUNT(*)::int AS selected_count
            FROM thumbnail_recreate_feedback
            WHERE user_id = $1::uuid
              AND event_type = 'selected'
              AND created_at >= NOW() - INTERVAL '7 days'
            """,
            user["id"],
        )
    top_patterns = [
        {
            "niche": str(r.get("niche") or "general"),
            "variants_seen": int(r.get("variants_seen") or 0),
            "avg_ctr_score": round(float(r.get("avg_ctr_score") or 0.0), 2),
        }
        for r in (rows or [])
    ]
    return {
        "window_days": 7,
        "selected_variants": int((selected[0].get("selected_count") if selected else 0) or 0),
        "top_patterns": top_patterns,
        "note": "Trend summary is based on your last 7 days of thumbnail studio usage.",
    }

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
            except Exception as e:
                logger.debug("billing session success: subscription price/tier parse skip: %s", e)

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

    # Fallback sync path: if webhook is delayed or misconfigured, hydrate subscription
    # state from Stripe session directly so UI and entitlements are immediately coherent.
    if mode == "subscription" and tier and sub_status in ("active", "trialing", "past_due"):
        try:
            ent = get_entitlements_for_tier(tier)
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE users SET
                        subscription_tier      = $1,
                        stripe_customer_id     = COALESCE(stripe_customer_id, $2),
                        stripe_subscription_id = COALESCE($3, stripe_subscription_id),
                        subscription_status    = $4,
                        current_period_end     = COALESCE($5, current_period_end),
                        trial_end              = COALESCE($6, trial_end),
                        updated_at             = NOW()
                    WHERE id = $7
                    """,
                    tier,
                    sess.get("customer"),
                    (sess.get("subscription") or {}).get("id") if isinstance(sess.get("subscription"), dict) else sess.get("subscription"),
                    sub_status,
                    (datetime.fromtimestamp(current_period_end_ts, tz=timezone.utc) if current_period_end_ts else None),
                    (datetime.fromtimestamp(trial_end_ts, tz=timezone.utc) if trial_end_ts else None),
                    user["id"],
                )
                await _ensure_wallet_floor_for_tier(
                    conn,
                    str(user["id"]),
                    ent,
                    "tier_floor_adjustment",
                    stripe_event_id=session_id,
                )
        except Exception as e:
            logger.warning(f"billing session fallback sync failed for user={user.get('id')}: {e}")

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


def _tier_from_stripe_price(price_obj: dict, fallback_tier: Optional[str] = None) -> str:
    """Resolve internal tier from Stripe Price metadata (tier=...) or lookup_key."""
    if not price_obj:
        return normalize_tier(fallback_tier or "free")
    pmd = price_obj.get("metadata") or {}
    t = (pmd.get("tier") or "").strip().lower()
    if t == "launch":
        t = "creator_lite"
    if t:
        return normalize_tier(t)
    lk = (price_obj.get("lookup_key") or "").strip()
    if lk in STRIPE_LOOKUP_TO_TIER:
        return normalize_tier(STRIPE_LOOKUP_TO_TIER[lk])
    return normalize_tier(fallback_tier or "free")


async def _ensure_wallet_floor_for_tier(conn, user_id: str, ent, reason: str, stripe_event_id: Optional[str] = None):
    """
    Ensure wallet balances have at least the minimum refill floor.
    Free tier: daily drip floor (monthly / days in month).
    Paid tiers: refill on invoice.paid only (no automatic floor grants).
    Internal tiers: full monthly floor.
    Preserve any higher balances (topups/carryover).
    """
    wallet = await get_wallet(conn, user_id)
    put_now = int(wallet.get("put_balance") or 0)
    aic_now = int(wallet.get("aic_balance") or 0)
    if bool(getattr(ent, "is_internal", False)):
        put_floor = int(getattr(ent, "put_monthly", 0) or 0)
        aic_floor = int(getattr(ent, "aic_monthly", 0) or 0)
    elif str(getattr(ent, "tier", "")) == "free":
        now_utc = datetime.now(timezone.utc)
        days_in_month = calendar.monthrange(now_utc.year, now_utc.month)[1]
        put_floor = int(math.ceil((int(getattr(ent, "put_monthly", 0) or 0)) / max(1, days_in_month)))
        aic_floor = int(math.ceil((int(getattr(ent, "aic_monthly", 0) or 0)) / max(1, days_in_month)))
    else:
        put_floor = 0
        aic_floor = 0

    if put_now < put_floor:
        delta = put_floor - put_now
        await conn.execute(
            "UPDATE wallets SET put_balance = put_balance + $1, updated_at = NOW() WHERE user_id = $2",
            delta, user_id,
        )
        await ledger_entry(conn, user_id, "put", delta, reason, stripe_event_id=stripe_event_id)

    if aic_now < aic_floor:
        delta = aic_floor - aic_now
        await conn.execute(
            "UPDATE wallets SET aic_balance = aic_balance + $1, updated_at = NOW() WHERE user_id = $2",
            delta, user_id,
        )
        await ledger_entry(conn, user_id, "aic", delta, reason, stripe_event_id=stripe_event_id)


@app.post("/api/billing/webhook")
@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Stripe billing webhook — signature-verified; duplicate event ids are acked without reprocessing."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.warning("Stripe webhook signature verification failed: %s", e)
        raise api_problem(
            400,
            code="stripe_invalid_signature",
            message="Invalid webhook signature",
        )

    ev_id = event.get("id")
    ev_type = event.get("type") or ""
    if ev_id and db_pool:
        try:
            async with db_pool.acquire() as conn:
                inserted = await conn.fetchval(
                    """
                    INSERT INTO stripe_webhook_events (id, event_type)
                    VALUES ($1, $2)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                    """,
                    str(ev_id),
                    str(ev_type)[:200],
                )
            if inserted is None:
                logger.info("Stripe webhook duplicate event id=%s (ack only)", ev_id)
                return JSONResponse(status_code=200, content={"received": True, "duplicate": True})
        except Exception as e:
            logger.warning("Stripe webhook idempotency insert failed (continuing): %s", e)

    etype = event.type

    # ── checkout.session.completed ──────────────────────────────────────
    if etype == "checkout.session.completed":
        session = event.data.object
        user_id = session.metadata.get("user_id")
        if not user_id:
            return {"status": "no_user_id"}

        async with db_pool.acquire() as conn:
            user_row = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1", user_id)
            email = user_row["email"] if user_row else ""
            uname = (user_row["name"] if user_row else None) or "there"

            if session.mode == "subscription":
                sub = stripe.Subscription.retrieve(
                    session.subscription,
                    expand=["latest_invoice"],
                )
                price_obj = sub["items"]["data"][0]["price"]
                tier = _tier_from_stripe_price(price_obj, None)
                ent  = get_entitlements_for_tier(tier)
                status = sub.status

                period_start = datetime.fromtimestamp(sub.current_period_start, tz=timezone.utc) if sub.get("current_period_start") else _now_utc()
                period_end   = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
                trial_end    = datetime.fromtimestamp(sub.trial_end, tz=timezone.utc) if sub.get("trial_end") else None

                await conn.execute("""
                    UPDATE users SET
                        subscription_tier      = $1,
                        stripe_customer_id     = COALESCE(stripe_customer_id, $2),
                        stripe_subscription_id = $3,
                        subscription_status    = $4,
                        current_period_end     = $5,
                        trial_end              = $6,
                        updated_at             = NOW()
                    WHERE id = $7
                """, tier, session.customer, session.subscription, status, period_end, trial_end, user_id)

                # Paid tiers refill only on confirmed invoice.paid.
                # Keep any existing wallet balance while trialing / awaiting payment.
                await _ensure_wallet_floor_for_tier(
                    conn, user_id, ent, "tier_floor_adjustment", stripe_event_id=session.id
                )

                amount = (session.amount_total or 0) / 100
                await conn.execute(
                    "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                    "VALUES ($1,$2,'subscription',$3,$4) ON CONFLICT DO NOTHING",
                    user_id, amount, session.id, tier
                )
                background_tasks.add_task(notify_mrr, amount, email, tier, status)

                # ── Welcome email: trial vs paid ──────────────────────────────
                if trial_end:
                    trial_days = sub.get("trial_period_days") or 14
                    background_tasks.add_task(
                        send_trial_started_email,
                        email, uname, tier,
                        trial_end.strftime("%B %d, %Y"),
                        trial_days,
                    )
                else:
                    next_date = period_end.strftime("%B %d, %Y")
                    background_tasks.add_task(
                        send_subscription_started_email,
                        email, uname, tier, amount, next_date,
                    )

            elif session.mode == "payment":
                wallet_type   = (session.metadata or {}).get("wallet", "put")
                amount_tokens = int((session.metadata or {}).get("amount", 0) or 0)
                try:
                    sess_exp = stripe.checkout.Session.retrieve(
                        session.id,
                        expand=["line_items.data.price"],
                    )
                    li = sess_exp.line_items.data[0] if sess_exp.line_items and sess_exp.line_items.data else None
                    if li and li.price:
                        pmd = li.price.metadata or {}
                        wallet_type = pmd.get("wallet") or wallet_type
                        if pmd.get("amount") not in (None, ""):
                            amount_tokens = int(pmd.get("amount") or 0)
                except Exception as e:
                    logger.warning(f"checkout topup: price metadata expand failed, using session metadata: {e}")
                if amount_tokens > 0:
                    # First top-up bonus: +25% to incentivize trying paid credits
                    prior = await conn.fetchval(
                        """
                        SELECT 1 FROM token_ledger
                        WHERE user_id = $1 AND reason IN ('topup', 'topup_purchase')
                        LIMIT 1
                        """,
                        user_id,
                    )
                    bonus = int(amount_tokens * 0.25) if not prior else 0
                    total = amount_tokens + bonus
                    await credit_wallet(conn, user_id, wallet_type, total, "topup", session.id)
                    amount = (session.amount_total or 0) / 100
                    await conn.execute(
                        "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                        "VALUES ($1,$2,'topup',$3,$4) ON CONFLICT DO NOTHING",
                        user_id, amount, session.id, f"{wallet_type}_{amount_tokens}"
                    )
                    background_tasks.add_task(notify_topup, amount, email, wallet_type, total)
                    background_tasks.add_task(
                        send_topup_receipt_email,
                        email, uname, wallet_type, total, amount, 0, session.id, bonus_tokens=bonus,
                    )

    # ── invoice.paid — monthly wallet refill on every renewal ──────────
    elif etype == "invoice.paid":
        invoice = event.data.object
        sub_id  = invoice.get("subscription")
        if not sub_id:
            return {"status": "no_subscription"}

        async with db_pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT id, email, name, subscription_tier FROM users WHERE stripe_subscription_id = $1", sub_id
            )
            if not user_row:
                logger.warning(f"invoice.paid: no user for subscription {sub_id}")
                return {"status": "user_not_found"}

            user_id = str(user_row["id"])
            email   = user_row["email"]
            uname   = user_row["name"] or "there"
            sub = stripe.Subscription.retrieve(sub_id)
            price_obj = sub["items"]["data"][0]["price"]
            tier = _tier_from_stripe_price(price_obj, user_row["subscription_tier"] or "free")
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
            await _ensure_wallet_floor_for_tier(conn, user_id, ent, "tier_floor_adjustment", stripe_event_id=invoice_id)

            amount = (invoice.amount_paid or 0) / 100
            await conn.execute(
                "INSERT INTO revenue_tracking (user_id, amount, source, stripe_event_id, plan) "
                "VALUES ($1,$2,'renewal',$3,$4) ON CONFLICT DO NOTHING",
                user_id, amount, invoice_id, tier
            )
            background_tasks.add_task(notify_mrr, amount, email, tier, "renewal")
            background_tasks.add_task(
                send_renewal_receipt_email,
                email, uname, tier, amount,
                invoice_id,
                f"{period_start.strftime('%b %d')} – {period_end.strftime('%b %d, %Y')}",
                period_end.strftime("%B %d, %Y"),
            )

    # ── subscription.updated — status changes, upgrades, downgrades ────
    elif etype == "customer.subscription.updated":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            price_obj = sub["items"]["data"][0]["price"]
            period_end = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
            trial_end = datetime.fromtimestamp(sub.trial_end, tz=timezone.utc) if sub.get("trial_end") else None

            # Fetch user before updating so we have old_tier for comparison
            user_row = await conn.fetchrow(
                "SELECT id, email, name, subscription_tier, subscription_status FROM users WHERE stripe_subscription_id = $1", sub.id
            )
            new_tier = _tier_from_stripe_price(
                price_obj,
                user_row["subscription_tier"] if user_row else "free",
            )
            old_tier = user_row["subscription_tier"] if user_row else new_tier
            old_status = user_row["subscription_status"] if user_row else None
            ent = get_entitlements_for_tier(new_tier)

            await conn.execute("""
                UPDATE users SET
                    subscription_tier   = $1,
                    subscription_status = $2,
                    current_period_end  = $3,
                    trial_end           = $4,
                    updated_at          = NOW()
                WHERE stripe_subscription_id = $5
            """, new_tier, sub.status, period_end, trial_end, sub.id)

            if user_row:
                await _ensure_wallet_floor_for_tier(
                    conn,
                    str(user_row["id"]),
                    ent,
                    "tier_floor_adjustment",
                    stripe_event_id=sub.id,
                )

            # Send tier change email only when tier actually changed
            if user_row and old_tier != new_tier:
                _email = user_row["email"]
                _name  = user_row["name"] or "there"
                _amount = 0.0  # Stripe doesn't provide amount here directly
                if _tier_is_upgrade(old_tier, new_tier):
                    background_tasks.add_task(
                        send_plan_upgraded_email,
                        _email, _name, old_tier, new_tier, _amount,
                        period_end.strftime("%B %d, %Y"),
                    )
                else:
                    background_tasks.add_task(
                        send_plan_downgraded_email,
                        _email, _name, old_tier, new_tier, _amount,
                        period_end.strftime("%B %d, %Y"),
                    )

            if user_row and old_status != sub.status:
                _email = user_row["email"]
                _name = user_row["name"] or "there"
                if sub.status == "trialing":
                    td = ent.trial_days or 7
                    t_end = trial_end.strftime("%B %d, %Y") if trial_end else "in 7 days"
                    background_tasks.add_task(
                        send_trial_started_email,
                        _email, _name, new_tier, t_end, td,
                    )
                elif sub.status == "active" and old_status not in ("active",):
                    amount = float(get_plan(new_tier).get("price", 0.0) or 0.0)
                    next_date = period_end.strftime("%B %d, %Y")
                    background_tasks.add_task(
                        send_subscription_started_email,
                        _email, _name, new_tier, amount, next_date,
                    )

    # ── invoice.payment_failed — notify user to update payment method ────
    elif etype == "invoice.payment_failed":
        inv = event.data.object
        sub_id = inv.get("subscription")
        if sub_id:
            async with db_pool.acquire() as conn:
                user_row = await conn.fetchrow(
                    "SELECT email, name, subscription_tier FROM users WHERE stripe_subscription_id = $1",
                    sub_id,
                )
            if user_row:
                retry_ts = inv.get("next_payment_attempt")
                retry_date = (
                    datetime.fromtimestamp(retry_ts, tz=timezone.utc).strftime("%B %d, %Y")
                    if retry_ts else ""
                )
                failure_reason = ""
                try:
                    err = inv.get("last_finalization_error") or {}
                    failure_reason = err.get("message", "") if isinstance(err, dict) else str(err)
                except Exception as e:
                    logger.debug("stripe webhook invoice.payment_failed: failure_reason parse skip: %s", e)
                background_tasks.add_task(
                    send_payment_failed_email,
                    user_row["email"],
                    user_row["name"] or "there",
                    user_row["subscription_tier"] or "free",
                    (inv.get("amount_due") or 0) / 100,
                    retry_date,
                    inv.get("id", ""),
                    failure_reason,
                )

    # ── charge.refunded — send billing adjustment receipt ────────────────
    elif etype == "charge.refunded":
        ch = event.data.object
        customer_id = ch.get("customer")
        if customer_id:
            async with db_pool.acquire() as conn:
                user_row = await conn.fetchrow(
                    "SELECT email, name FROM users WHERE stripe_customer_id = $1",
                    customer_id,
                )
            if user_row:
                amount = float((ch.get("amount_refunded") or 0) / 100.0)
                if amount > 0:
                    background_tasks.add_task(
                        send_refund_receipt_email,
                        user_row["email"],
                        user_row["name"] or "there",
                        amount,
                        (ch.get("currency") or "usd"),
                        ch.get("id") or "",
                        ch.get("reason") or "",
                        "refund",
                    )

    # ── charge.dispute.created — notify potential chargeback ─────────────
    elif etype == "charge.dispute.created":
        dispute = event.data.object
        charge_id = dispute.get("charge")
        if charge_id:
            try:
                charge = stripe.Charge.retrieve(charge_id)
            except Exception as e:
                logger.debug("stripe webhook dispute: Charge.retrieve failed id=%s: %s", charge_id, e)
                charge = None
            customer_id = charge.get("customer") if charge else None
            if customer_id:
                async with db_pool.acquire() as conn:
                    user_row = await conn.fetchrow(
                        "SELECT email, name FROM users WHERE stripe_customer_id = $1",
                        customer_id,
                    )
                if user_row:
                    amount = float((dispute.get("amount") or 0) / 100.0)
                    background_tasks.add_task(
                        send_refund_receipt_email,
                        user_row["email"],
                        user_row["name"] or "there",
                        amount,
                        (dispute.get("currency") or "usd"),
                        dispute.get("id") or "",
                        dispute.get("reason") or "Card dispute opened",
                        "chargeback",
                    )

    # ── subscription.deleted — downgrade to free OR execute deferred account deletion ─
    elif etype == "customer.subscription.deleted":
        sub = event.data.object
        async with db_pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT * FROM users WHERE stripe_subscription_id = $1", sub.id
            )
            if not user_row:
                return {"status": "user_not_found"}

            user_dict = dict(user_row)

            if user_row.get("deletion_requested_at"):
                # User requested account deletion; period ended — execute full deletion now
                deletion_log = await conn.fetchrow(
                    "SELECT id FROM account_deletion_log WHERE user_id = $1 AND completed_at IS NULL ORDER BY requested_at DESC LIMIT 1",
                    str(user_row["id"]),
                )
                result = await _execute_account_deletion(conn, user_dict, initiated_by="account_deletion")
                if deletion_log:
                    await conn.execute(
                        """
                        UPDATE account_deletion_log
                        SET completed_at = NOW(), r2_keys_deleted = $2, tokens_revoked = $3,
                            stripe_cancelled = TRUE, rows_deleted = $4
                        WHERE id = $1
                        """,
                        deletion_log["id"],
                        result["r2_deleted"],
                        result["tokens_revoked"],
                        json.dumps(result["rows_deleted"]),
                    )
                logger.info(
                    f"[DELETION COMPLETE via subscription.deleted] user={user_row['id']} "
                    f"r2={result['r2_deleted']} tokens={result['tokens_revoked']}"
                )
            else:
                # Normal cancellation: downgrade to free, send emails
                await conn.execute("""
                    UPDATE users SET
                        subscription_tier   = 'free',
                        subscription_status = 'cancelled',
                        updated_at          = NOW()
                    WHERE stripe_subscription_id = $1
                """, sub.id)

                _email    = user_row["email"]
                _name     = user_row["name"] or "there"
                _old_tier = user_row["subscription_tier"] or "free"
                _trial_end = user_row.get("trial_end")
                _access_until = datetime.fromtimestamp(
                    sub.current_period_end, tz=timezone.utc
                ).strftime("%B %d, %Y") if sub.get("current_period_end") else "now"

                if _trial_end and _trial_end > _now_utc():
                    background_tasks.add_task(
                        send_trial_cancelled_email,
                        _email, _name, _old_tier, _access_until,
                    )
                else:
                    background_tasks.add_task(
                        send_subscription_cancelled_email,
                        _email, _name, _old_tier, _access_until,
                    )

    return {"status": "ok"}


async def _do_monthly_refill(conn, user_id, tier, ent, invoice_id, period_start, period_end):
    """Renewal handler. Deduped by invoice_id — safe to call on webhook retry.

    Paid tiers: full monthly PUT+AIC credited on confirmed invoice.paid.
    Free tier: uses daily drip budget logic (no invoice credit path).

    Internal tiers: full monthly PUT+AIC credited on each paid invoice.
    """
    # Check dedup table
    try:
        existing = await conn.fetchrow(
            "SELECT invoice_id FROM stripe_invoice_log WHERE invoice_id = $1", invoice_id
        )
        if existing:
            logger.info(f"Monthly refill already processed for {invoice_id}, skipping.")
            return False
    except Exception as e:
        logger.debug("_do_monthly_refill: stripe_invoice_log dedup check skipped (table may be missing): %s", e)

    is_internal = bool(getattr(ent, "is_internal", False))
    is_paid_public = str(getattr(ent, "tier", "")) in {"creator_lite", "creator_pro", "studio", "agency"}
    put_amount = 0
    aic_amount = 0
    if is_internal or is_paid_public:
        put_amount = int(ent.put_monthly or 0)
        aic_amount = int(ent.aic_monthly or 0)
        if put_amount > 0:
            await conn.execute(
                "UPDATE wallets SET put_balance = put_balance + $1, updated_at = NOW() WHERE user_id = $2",
                put_amount,
                user_id,
            )
            await ledger_entry(
                conn, user_id, "put", put_amount, "monthly_refill", stripe_event_id=invoice_id
            )

        if aic_amount > 0:
            await conn.execute(
                "UPDATE wallets SET aic_balance = aic_balance + $1, updated_at = NOW() WHERE user_id = $2",
                aic_amount,
                user_id,
            )
            await ledger_entry(
                conn, user_id, "aic", aic_amount, "monthly_refill", stripe_event_id=invoice_id
            )

    try:
        await conn.execute(
            """
            INSERT INTO stripe_invoice_log
                (invoice_id, user_id, tier_slug, put_credited, aic_credited, period_start, period_end)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (invoice_id) DO NOTHING
            """,
            invoice_id,
            user_id,
            tier,
            put_amount,
            aic_amount,
            period_start,
            period_end,
        )
    except Exception as e:
        logger.debug("_do_monthly_refill: stripe_invoice_log insert failed (non-fatal): %s", e)

    logger.info(
        f"Monthly refill: user={user_id} tier={tier} internal={is_internal} "
        f"+{put_amount} PUT +{aic_amount} AIC invoice={invoice_id}"
    )
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

    If TIKTOK_WEBHOOK_SECRET is empty the check is skipped in development
    (localhost/127.0.0.1). In production, missing secret rejects the event.
    """
    if not secret:
        base = os.environ.get("BASE_URL", "")
        if "localhost" in base or "127.0.0.1" in base or not base:
            return True, "sig-check-skipped-no-secret-dev"
        return False, "webhook-secret-not-configured"

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

                # Find the most recent TikTok upload for this open_id that either
                # (a) is still in-flight, OR (b) succeeded/completed but never received
                # a platform_video_id (TikTok fires the webhook after our pipeline
                # already marks the job succeeded, so we must handle both states).
                upload = await conn.fetchrow(
                    """
                    SELECT u.id, u.user_id, u.platform_results
                    FROM uploads u
                    JOIN platform_tokens pt
                        ON u.user_id = pt.user_id
                       AND pt.platform = 'tiktok'
                       AND pt.account_id = $1
                       AND pt.revoked_at IS NULL
                    WHERE u.status IN ('succeeded', 'completed', 'partial',
                                       'processing', 'queued', 'pending', 'accepted')
                      AND (
                        -- Prefer in-flight uploads first
                        u.status NOT IN ('succeeded', 'completed', 'partial')
                        OR
                        -- Fall back to succeeded uploads missing platform_video_id
                        NOT EXISTS (
                          SELECT 1
                          FROM jsonb_array_elements(
                            CASE WHEN jsonb_typeof(u.platform_results) = 'array'
                                 THEN u.platform_results ELSE '[]'::jsonb END
                          ) AS item
                          WHERE item->>'platform' = 'tiktok'
                            AND (item->>'platform_video_id') IS NOT NULL
                            AND (item->>'platform_video_id') != ''
                            AND (item->>'platform_video_id') != 'null'
                        )
                      )
                    ORDER BY
                      CASE WHEN u.status NOT IN ('succeeded','completed','partial') THEN 0 ELSE 1 END,
                      u.created_at DESC
                    LIMIT 1
                    """,
                    user_openid,
                )

                if upload:
                    raw_pr = _safe_json(upload["platform_results"], [])
                    # Preserve array format (one entry per account); merge TikTok update into matching entry
                    if isinstance(raw_pr, list):
                        tiktok_updated = False
                        for item in raw_pr:
                            if isinstance(item, dict) and str(item.get("platform", "")).lower() == "tiktok":
                                if str(item.get("account_id", "")) == user_openid:
                                    item["platform_video_id"] = video_id
                                    item["video_id"] = video_id
                                    item["platform_url"] = f"https://www.tiktok.com/video/{video_id}"
                                    item["url"] = item["platform_url"]
                                    item["success"] = True
                                    tiktok_updated = True
                                    break
                        if not tiktok_updated:
                            raw_pr.append({"platform": "tiktok", "account_id": user_openid, "platform_video_id": video_id, "video_id": video_id, "success": True})
                        updated_pr = raw_pr
                    else:
                        existing = raw_pr if isinstance(raw_pr, dict) else {}
                        existing["tiktok"] = {"platform": "tiktok", "status": "published", "video_id": video_id, "share_id": share_id, "published_at": _now_utc().isoformat()}
                        updated_pr = existing
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status           = 'completed',
                            completed_at     = NOW(),
                            platform_results = $1::jsonb,
                            updated_at       = NOW()
                        WHERE id = $2
                        """,
                        json.dumps(updated_pr),
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
                    raw_pr = _safe_json(upload["platform_results"], [])
                    if isinstance(raw_pr, list):
                        for item in raw_pr:
                            if isinstance(item, dict) and str(item.get("platform", "")).lower() == "tiktok" and str(item.get("account_id", "")) == user_openid:
                                item["success"] = False
                                item["error_message"] = "TikTok reported upload failure via webhook"
                                break
                        updated_pr = raw_pr
                    else:
                        existing = raw_pr if isinstance(raw_pr, dict) else {}
                        existing["tiktok"] = {"platform": "tiktok", "status": "failed", "share_id": share_id, "failed_at": _now_utc().isoformat()}
                        updated_pr = existing
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status           = 'failed',
                            error_code       = 'tiktok_upload_failed',
                            error_detail     = 'TikTok reported upload failure via webhook',
                            platform_results = $1::jsonb,
                            updated_at       = NOW()
                        WHERE id = $2
                        """,
                        json.dumps(updated_pr),
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
        logger.warning("[facebook-webhook] missing X-Hub-Signature-256 header")
        raise HTTPException(400, "Missing signature")
    elif not META_APP_SECRET:
        base = os.environ.get("BASE_URL", "")
        if base and "localhost" not in base and "127.0.0.1" not in base:
            logger.error("[facebook-webhook] META_APP_SECRET not configured in production — rejecting")
            raise HTTPException(500, "Webhook secret not configured")

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
    # Bounded ranges: same UTC half-open window [since, until) on uploads.created_at
    # and catalog published_at (see services.canonical_engagement).
    from services.canonical_engagement import (
        ROLLUP_VERSION,
        compute_canonical_engagement_rollup,
        engagement_time_window_for_analytics_range,
        engagement_window_api_dict,
    )

    now = _now_utc()
    since, until_excl = engagement_time_window_for_analytics_range(range, now=now)

    async with db_pool.acquire() as conn:
        bundle = await _user_upload_kpi_bundle(
            conn, str(user["id"]), since=since, until=until_excl
        )
        upload_engagement = bundle["engagement"]

        if since is None:
            daily = await conn.fetch(
                "SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads "
                "FROM uploads WHERE user_id = $1 "
                "GROUP BY DATE(created_at) ORDER BY date",
                user["id"],
            )
            platforms = await conn.fetch(
                f"SELECT unnest(platforms) AS platform, COUNT(*)::int AS count "
                f"FROM uploads WHERE user_id = $1 "
                f"AND status IN {SUCCESSFUL_STATUS_SQL_IN} "
                f"GROUP BY platform",
                user["id"],
            )
        else:
            daily = await conn.fetch(
                "SELECT DATE(created_at) AS date, COUNT(*)::int AS uploads "
                "FROM uploads WHERE user_id = $1 AND created_at >= $2 AND created_at < $3 "
                "GROUP BY DATE(created_at) ORDER BY date",
                user["id"], since, until_excl,
            )
            platforms = await conn.fetch(
                f"SELECT unnest(platforms) AS platform, COUNT(*)::int AS count "
                f"FROM uploads WHERE user_id = $1 AND created_at >= $2 AND created_at < $3 "
                f"AND status IN {SUCCESSFUL_STATUS_SQL_IN} "
                f"GROUP BY platform",
                user["id"], since, until_excl,
            )

        # ================================================================
        # TRILL TELEMETRY STATS
        # ================================================================
        trill_stats = None
        try:
            if since is None:
                trill_data = await conn.fetchrow("""
                    SELECT
                        COUNT(*)::int AS trill_uploads,
                        COALESCE(AVG(trill_score), 0)::decimal AS avg_score,
                        COALESCE(MAX(trill_score), 0)::decimal AS max_score,
                        COALESCE(MAX(max_speed_mph), 0)::decimal AS max_speed_mph,
                        COALESCE(SUM(distance_miles), 0)::decimal AS total_distance_miles
                    FROM uploads
                    WHERE user_id = $1
                    AND trill_score IS NOT NULL
                """, user["id"])
            else:
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
                    AND created_at < $3
                    AND trill_score IS NOT NULL
                """, user["id"], since, until_excl)

            if trill_data and trill_data["trill_uploads"] > 0:
                if since is None:
                    speed_buckets = await conn.fetch("""
                        SELECT speed_bucket, COUNT(*)::int AS count
                        FROM uploads
                        WHERE user_id = $1
                        AND speed_bucket IS NOT NULL
                        GROUP BY speed_bucket
                    """, user["id"])
                else:
                    speed_buckets = await conn.fetch("""
                        SELECT speed_bucket, COUNT(*)::int AS count
                        FROM uploads
                        WHERE user_id = $1
                        AND created_at >= $2
                        AND created_at < $3
                        AND speed_bucket IS NOT NULL
                        GROUP BY speed_bucket
                    """, user["id"], since, until_excl)

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

        live_aggregate = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}
        try:
            connected_accounts = await conn.fetchval(
                "SELECT COUNT(*)::int FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
                user["id"],
            )
        except Exception as e:
            logger.debug("get_analytics: connected_accounts count failed: %s", e)
            connected_accounts = 0
        if int(connected_accounts or 0) > 0:
            try:
                prow = await conn.fetchrow(
                    "SELECT data FROM platform_metrics_cache WHERE user_id = $1",
                    user["id"],
                )
                if prow and prow["data"] is not None:
                    pdata = prow["data"]
                    if isinstance(pdata, str):
                        pdata = json.loads(pdata)
                    if isinstance(pdata, dict):
                        pl = pdata.get("platforms") or {}
                        live_aggregate = _aggregate_platform_metrics_live(pl)
            except Exception as e:
                logger.debug("get_analytics: platform_metrics_cache read failed: %s", e)

        try:
            cr = await compute_canonical_engagement_rollup(
                conn,
                str(user["id"]),
                window_start=since,
                window_end_exclusive=until_excl,
                platform=None,
            )
        except Exception as e:
            logger.warning("get_analytics: canonical engagement rollup failed: %s", e)
            cr = {
                "views": int(upload_engagement.get("views") or 0),
                "likes": int(upload_engagement.get("likes") or 0),
                "comments": int(upload_engagement.get("comments") or 0),
                "shares": int(upload_engagement.get("shares") or 0),
                "breakdown": {
                    "compute": {
                        "rollup_version": ROLLUP_VERSION,
                        "complete": False,
                        "warnings": ["rollup_exception"],
                        "error_detail": str(e)[:500],
                    },
                },
                "catalog_tracked_videos": 0,
                "rollup_version": ROLLUP_VERSION,
                "rollup_rule": "fallback_upload_bundle_only",
                "kpi_sources": {"error": str(e), "rollup_version": ROLLUP_VERSION},
            }
        win_meta = engagement_window_api_dict(start=since, end_exclusive=until_excl)
        canon = dict(cr)
        canon["engagement_window_utc"] = win_meta

    result = {
        "total_uploads": bundle["total"],
        "completed": bundle["successful"],
        "success_rate_pct": bundle["success_rate_pct"],
        "views": int(canon["views"] or 0),
        "likes": int(canon["likes"] or 0),
        "comments": int(canon["comments"] or 0),
        "shares": int(canon["shares"] or 0),
        "engagement_breakdown": canon["breakdown"],
        "engagement_rollup_rule": canon["rollup_rule"],
        "engagement_rollup_version": canon.get("rollup_version"),
        "engagement_window_utc": canon.get("engagement_window_utc"),
        "catalog_tracked_videos": int(canon["catalog_tracked_videos"] or 0),
        "put_used": bundle["put_used"],
        "aic_used": bundle["aic_used"],
        "daily": [{"date": str(d["date"]), "uploads": d["uploads"]} for d in daily],
        "platforms": {p["platform"]: p["count"] for p in platforms},
        "live_aggregate": live_aggregate,
        "kpi_sources": canon.get("kpi_sources"),
        "engagement_crosswalk": metric_defs.engagement_crosswalk(),
        "metric_definitions": metric_defs.for_get_analytics(),
    }

    if trill_stats:
        result["trill"] = trill_stats

    return result


@app.get("/api/analytics/quality-scores")
async def get_quality_scores(
    days: int = 30,
    platform: str = "all",
    user: dict = Depends(get_current_user),
):
    """
    ML strategy score rows (daily aggregates + confidence intervals).
    Used to inspect what's working over time and to drive model/prompt biasing.
    """
    days = max(1, min(int(days or 30), 365))
    platform = (platform or "all").strip().lower()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT day, platform, strategy_key, samples,
                   mean_engagement, mean_views, engagement_stddev, ci95_low, ci95_high, updated_at
              FROM upload_quality_scores_daily
             WHERE user_id = $1
               AND day >= (CURRENT_DATE - ($2::int || ' days')::interval)::date
               AND ($3 = 'all' OR platform = $3)
             ORDER BY day DESC, mean_engagement DESC, samples DESC
             LIMIT 5000
            """,
            user["id"],
            days,
            platform,
        )
    return {
        "days": days,
        "platform": platform,
        "rows": [dict(r) for r in rows],
        "metric_definitions": metric_defs.for_quality_scores(),
    }

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
_platform_metrics_refresh_markers: dict = {}
_PLATFORM_CACHE_TTL = 3 * 60 * 60  # 3 hours
_PLATFORM_CACHE_STALE_REFRESH_SEC = 2 * 60 * 60  # 2 hours
_PLATFORM_CACHE_REFRESH_THROTTLE_SEC = 10 * 60


def _parse_iso_ts(ts: Any) -> Optional[datetime]:
    try:
        if not ts:
            return None
        s = str(ts).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _should_background_refresh_platform_cache(cached_payload: Optional[dict]) -> bool:
    if not isinstance(cached_payload, dict):
        return True
    fetched_at = _parse_iso_ts(cached_payload.get("fetched_at"))
    if not fetched_at:
        return True
    age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
    return age >= float(_PLATFORM_CACHE_STALE_REFRESH_SEC)


# Per-user analytics job visibility (in-memory; resets on process restart).
_analytics_sync_status: dict[str, dict[str, Any]] = {}
_analytics_sync_status_lock = asyncio.Lock()


def _empty_analytics_sync_snapshot() -> dict[str, Any]:
    return {
        "upload_analytics": {
            "status": "idle",
            "queued": False,
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "processed": 0,
            "synced": 0,
            "failed": 0,
            "skipped": 0,
            "eligible_rows": None,
            "candidates": None,
            "stale_minutes": None,
            "max_uploads": None,
            "error": None,
        },
        "platform_metrics": {
            "status": "idle",
            "queued": False,
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "ok": None,
            "error": None,
            "trigger": None,
        },
    }


async def _analytics_sync_status_get(uid: str) -> dict[str, Any]:
    base = _empty_analytics_sync_snapshot()
    async with _analytics_sync_status_lock:
        cur = _analytics_sync_status.get(str(uid)) or {}
    for section in ("upload_analytics", "platform_metrics"):
        if isinstance(cur.get(section), dict):
            base[section].update(cur[section])
    return base


async def _analytics_sync_status_patch(uid: str, section: str, patch: dict) -> None:
    uid_s = str(uid)
    async with _analytics_sync_status_lock:
        if uid_s not in _analytics_sync_status:
            _analytics_sync_status[uid_s] = _empty_analytics_sync_snapshot()
        snap = _analytics_sync_status[uid_s]
        if section not in snap or not isinstance(snap[section], dict):
            snap[section] = _empty_analytics_sync_snapshot()[section]
        snap[section].update(patch)


async def _analytics_track_platform_metrics_refresh(user_id: str, trigger: str) -> bool:
    """Run platform_metrics_job refresh with status tracking for /api/analytics/sync-status."""
    uid_s = str(user_id)
    t0 = _time.time()
    started = datetime.now(timezone.utc).isoformat()
    await _analytics_sync_status_patch(
        uid_s,
        "platform_metrics",
        {
            "status": "running",
            "queued": False,
            "started_at": started,
            "finished_at": None,
            "duration_ms": None,
            "ok": None,
            "error": None,
            "trigger": str(trigger or "unknown"),
        },
    )
    try:
        from services.platform_metrics_job import refresh_platform_metrics_for_user

        ok = bool(await refresh_platform_metrics_for_user(db_pool, user_id))
        finished = datetime.now(timezone.utc).isoformat()
        await _analytics_sync_status_patch(
            uid_s,
            "platform_metrics",
            {
                "status": "completed",
                "finished_at": finished,
                "duration_ms": int((_time.time() - t0) * 1000),
                "ok": ok,
                "error": None,
            },
        )
        return ok
    except Exception as e:
        finished = datetime.now(timezone.utc).isoformat()
        await _analytics_sync_status_patch(
            uid_s,
            "platform_metrics",
            {
                "status": "failed",
                "finished_at": finished,
                "duration_ms": int((_time.time() - t0) * 1000),
                "ok": False,
                "error": str(e)[:500],
            },
        )
        raise


def _schedule_platform_cache_refresh_if_needed(user_id: str) -> None:
    now = _time.time()
    last = float(_platform_metrics_refresh_markers.get(user_id) or 0.0)
    if (now - last) < float(_PLATFORM_CACHE_REFRESH_THROTTLE_SEC):
        return
    _platform_metrics_refresh_markers[user_id] = now

    async def _bg_refresh() -> None:
        try:
            await _analytics_track_platform_metrics_refresh(user_id, "stale-cache-read")
        except Exception as e:
            logger.warning(f"platform-metrics stale background refresh failed user={user_id}: {e}")

    try:
        _asyncio.create_task(_bg_refresh())
    except Exception as e:
        logger.debug("_schedule_platform_cache_refresh_if_needed: create_task failed user=%s: %s", user_id, e)


def _aggregate_platform_metrics_live(platforms_result: dict) -> dict:
    """
    Sum engagement from platforms that returned status=='live' (TikTok/YouTube/etc.).
    Meta platforms that only return errors do not contribute — avoids zeroing the sum.
    """
    agg = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}

    def _n(x) -> int:
        try:
            return int(x or 0)
        except Exception:
            return 0

    if not isinstance(platforms_result, dict):
        return agg

    for plat, d in platforms_result.items():
        if not isinstance(d, dict) or d.get("status") != "live":
            continue
        if plat == "youtube":
            views = _n(d.get("shorts_views")) or _n(d.get("views"))
            likes = _n(d.get("shorts_likes")) or _n(d.get("likes"))
            comments = _n(d.get("shorts_comments")) or _n(d.get("comments"))
        else:
            views = _n(d.get("views"))
            likes = _n(d.get("reactions")) if plat == "facebook" else _n(d.get("likes"))
            comments = _n(d.get("comments"))
        # Facebook card uses reactions as the like-like metric
        shares = _n(d.get("shares"))
        if views or likes or comments or shares:
            agg["views"] += views
            agg["likes"] += likes
            agg["comments"] += comments
            agg["shares"] += shares
            agg["platforms_included"].append(plat)
    return agg


def _attach_aggregate_to_metrics_payload(payload: dict) -> dict:
    """Ensure every platform-metrics response includes aggregate (for older DB cache rows)."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    pl = out.get("platforms")
    if isinstance(pl, dict):
        out["aggregate"] = _aggregate_platform_metrics_live(pl)
    return out


def _sanitize_aggregate_numbers(aggregate: dict) -> tuple[dict, bool]:
    changed = False
    clean = {}
    for key in ("views", "likes", "comments", "shares"):
        raw = (aggregate or {}).get(key, 0)
        try:
            val = int(raw or 0)
        except Exception:
            val = 0
        if val < 0:
            val = 0
        if raw != val:
            changed = True
        clean[key] = val
    plats = (aggregate or {}).get("platforms_included", [])
    if not isinstance(plats, list):
        plats = []
        changed = True
    clean["platforms_included"] = [str(x) for x in plats if x]
    return clean, changed


async def _enforce_metrics_rollup_standards(conn, user_id: str, payload: dict) -> tuple[dict, dict]:
    """
    Post-rollup standards guardrail:
    - recompute aggregate from platform payload
    - sanitize numeric fields (non-negative ints)
    - persist corrected payload/rollup when drift is detected
    """
    out = dict(payload or {})
    platforms = out.get("platforms")
    computed = _aggregate_platform_metrics_live(platforms if isinstance(platforms, dict) else {})
    sanitized, sanitized_changed = _sanitize_aggregate_numbers(computed)
    if not isinstance(platforms, dict):
        await audit_log(
            str(user_id),
            "METRICS_PAYLOAD_MISSING_PLATFORMS",
            event_category="DATA_INTEGRITY",
            resource_type="platform_metrics_cache",
            resource_id=str(user_id),
            details={"reason": "missing_or_invalid_platforms_payload"},
            severity="WARNING",
            outcome="FAILED",
        )
    current = out.get("aggregate") if isinstance(out.get("aggregate"), dict) else {}
    current_sanitized, _ = _sanitize_aggregate_numbers(current)
    corrected = current_sanitized != sanitized
    if corrected or sanitized_changed:
        out["aggregate"] = sanitized
        await _platform_metrics_db_cache_set(conn, str(user_id), out)
        await audit_log(
            str(user_id),
            "METRICS_ROLLUP_CORRECTED",
            event_category="DATA_INTEGRITY",
            resource_type="platform_metrics_cache",
            resource_id=str(user_id),
            details={
                "before": current_sanitized,
                "after": sanitized,
                "sanitized_changed": bool(sanitized_changed),
            },
            severity="WARNING",
            outcome="SUCCESS",
        )
        try:
            await conn.execute(
                """
                INSERT INTO platform_user_metrics_rollups_daily
                    (user_id, day, views, likes, comments, shares, platforms_json, updated_at)
                VALUES
                    ($1, CURRENT_DATE, $2, $3, $4, $5, $6::jsonb, NOW())
                ON CONFLICT (user_id, day) DO UPDATE
                SET views = EXCLUDED.views,
                    likes = EXCLUDED.likes,
                    comments = EXCLUDED.comments,
                    shares = EXCLUDED.shares,
                    platforms_json = EXCLUDED.platforms_json,
                    updated_at = NOW()
                """,
                str(user_id),
                int(sanitized.get("views", 0)),
                int(sanitized.get("likes", 0)),
                int(sanitized.get("comments", 0)),
                int(sanitized.get("shares", 0)),
                json.dumps(platforms if isinstance(platforms, dict) else {}),
            )
        except Exception as e:
            logger.warning(f"rollup standards update skipped user={user_id}: {e}")
            await audit_log(
                str(user_id),
                "ROLLUP_DAILY_UPSERT_FAILED",
                event_category="DATA_INTEGRITY",
                resource_type="platform_user_metrics_rollups_daily",
                resource_id=str(user_id),
                details={"error": str(e)},
                severity="ERROR",
                outcome="FAILED",
            )
    return out, {
        "corrected": bool(corrected or sanitized_changed),
        "aggregate": sanitized,
    }


async def _apply_metrics_standards_for_user(user_id: str, payload: dict) -> tuple[dict, dict]:
    """Single standards path for any platform-metrics payload returned to UI."""
    async with db_pool.acquire() as conn:
        return await _enforce_metrics_rollup_standards(conn, str(user_id), _attach_aggregate_to_metrics_payload(payload or {}))


def _combine_platform_live_results(platform: str, results: List[dict]) -> dict:
    """Combine multiple account-level metric responses into one platform payload."""
    platform = (platform or "").lower()
    live = [r for r in (results or []) if isinstance(r, dict) and r.get("status") == "live"]
    if not live:
        err = next((r for r in (results or []) if isinstance(r, dict) and r.get("status") == "error"), None)
        if err:
            return {"status": "error", "error": err.get("error", "unknown_error"), "error_detail": err.get("error_detail")}
        return {"status": "not_connected"}

    def _n(v):
        try:
            return int(v or 0)
        except Exception:
            return 0

    out = {
        "status": "live",
        "analytics_source": ",".join(sorted({str(r.get("analytics_source") or "").strip() for r in live if r.get("analytics_source")})),
        "views": sum(_n(r.get("views")) for r in live),
        "likes": sum(_n(r.get("likes")) for r in live),
        "comments": sum(_n(r.get("comments")) for r in live),
        "shares": sum(_n(r.get("shares")) for r in live),
        "accounts": len(live),
    }

    if platform == "youtube":
        out["subscribers"] = sum(_n(r.get("subscribers")) for r in live)
        out["minutes_watched"] = sum(_n(r.get("minutes_watched")) for r in live)
        out["video_count"] = sum(_n(r.get("video_count")) for r in live)
        durations = [float(r.get("avg_watch_seconds")) for r in live if r.get("avg_watch_seconds") is not None]
        out["avg_watch_seconds"] = round(sum(durations) / len(durations), 1) if durations else None
    elif platform == "tiktok":
        out["followers"] = sum(_n(r.get("followers")) for r in live if r.get("followers") is not None)
        out["following"] = sum(_n(r.get("following")) for r in live if r.get("following") is not None)
        out["total_likes"] = sum(_n(r.get("total_likes")) for r in live if r.get("total_likes") is not None)
        out["video_count"] = sum(_n(r.get("video_count")) for r in live)
        durations = [float(r.get("avg_watch_seconds")) for r in live if r.get("avg_watch_seconds") is not None]
        out["avg_watch_seconds"] = round(sum(durations) / len(durations), 1) if durations else None
    elif platform == "instagram":
        out["saves"] = sum(_n(r.get("saves")) for r in live)
        out["reach"] = sum(_n(r.get("reach")) for r in live)
        out["video_count"] = sum(_n(r.get("video_count")) for r in live)
    elif platform == "facebook":
        out["reactions"] = sum(_n(r.get("reactions")) for r in live)
        out["followers"] = sum(_n(r.get("followers")) for r in live)
        out["video_count"] = sum(_n(r.get("video_count")) for r in live)

    return out


async def _fetch_tiktok_metrics(access_token: str) -> dict:
    """TikTok Content API — video list totals + follower stats (requires video.list + user.info.stats)."""
    if not access_token:
        return {"status": "not_connected"}

    try:
        from services.tiktok_api import tiktok_envelope_error, tiktok_video_list_url

        async with httpx.AsyncClient(timeout=20) as client:
            # 1) Video list — fields MUST be query params per TikTok docs (not JSON body).
            list_url = tiktok_video_list_url()
            resp = await client.post(
                list_url,
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json={"max_count": 20},
            )
            try:
                body = resp.json()
            except Exception:
                body = {}
            env_err = tiktok_envelope_error(body)
            if env_err:
                logger.warning(f"TikTok video list API error: {env_err}")
                return {"status": "error", "error": env_err, "error_detail": env_err}
            if resp.status_code != 200:
                logger.warning(f"TikTok video list HTTP {resp.status_code}: {resp.text[:200]}")
                return {
                    "status": "error",
                    "error": f"video_list_http_{resp.status_code}",
                    "error_detail": resp.text[:300],
                }

            videos = (body.get("data", {}) or {}).get("videos", []) or []

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

            # 2) User info stats + profile (requires user.info.basic + user.info.stats scopes)
            followers = following = total_likes = video_count = None
            display_name = username = avatar_url = None
            ui = await client.get(
                "https://open.tiktokapis.com/v2/user/info/",
                params={"fields": "follower_count,following_count,likes_count,video_count,display_name,username,avatar_url"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            try:
                ubody = ui.json() if ui.content else {}
            except Exception:
                ubody = {}
            uenv = tiktok_envelope_error(ubody)
            if ui.status_code == 200 and not uenv:
                user_obj     = ((ubody.get("data", {}) or {}).get("user", {}) or {})
                followers    = user_obj.get("follower_count")
                following    = user_obj.get("following_count")
                total_likes  = user_obj.get("likes_count")
                video_count  = user_obj.get("video_count")
                display_name = (user_obj.get("display_name") or "").strip() or None
                username     = (user_obj.get("username") or "").strip() or None
                avatar_url   = (user_obj.get("avatar_url") or "").strip() or None
            else:
                logger.warning(
                    f"TikTok user.info HTTP {ui.status_code}: {uenv or ui.text[:200]}"
                )

            return {
                "status": "live",
                "analytics_source": "video.list+user.info.stats",
                "display_name": display_name,
                "username":    username,
                "avatar_url":  avatar_url,
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
                detail = ch.text[:400] if ch.text else ""
                return {
                    "status": "error",
                    "error": f"channels_http_{ch.status_code}",
                    "error_detail": detail or f"HTTP {ch.status_code}",
                }

            items = ch.json().get("items", []) or []
            if not items:
                return {
                    "status": "error",
                    "error": "no_youtube_channel",
                    "error_detail": "No channel returned for mine=true — reconnect Google/YouTube.",
                }
            stats = items[0].get("statistics", {}) if items else {}

            views = likes = comments = shares = 0
            avg_watch = minutes_watched = None
            analytics_source = "channel_stats_fallback"
            analytics_diagnostic = ""
            shorts_views = shorts_likes = shorts_comments = shorts_count = 0

            def _dur_to_seconds(iso: str) -> int:
                # Parse a subset of ISO-8601 durations (PT#H#M#S) without extra deps.
                if not iso or not isinstance(iso, str) or not iso.startswith("PT"):
                    return 0
                h = m = s = 0
                try:
                    import re as _re

                    hh = _re.search(r"(\d+)H", iso)
                    mm = _re.search(r"(\d+)M", iso)
                    ss = _re.search(r"(\d+)S", iso)
                    if hh:
                        h = int(hh.group(1))
                    if mm:
                        m = int(mm.group(1))
                    if ss:
                        s = int(ss.group(1))
                except Exception:
                    return 0
                return h * 3600 + m * 60 + s
            try:
                today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                thirty = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
                # Primary: aggregated report (no dimensions) is more tolerant
                # across channel/account states than day-dimension queries.
                an = await client.get(
                    "https://youtubeanalytics.googleapis.com/v2/reports",
                    params={
                        "ids":        "channel==MINE",
                        "startDate":  thirty,
                        "endDate":    today,
                        "metrics":    "views,likes,comments,shares,averageViewDuration,estimatedMinutesWatched",
                    },
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if an.status_code == 200:
                    rows = an.json().get("rows", []) or []
                    if rows:
                        r0 = rows[0] if isinstance(rows[0], list) else []
                        views           = int(r0[0] or 0) if len(r0) > 0 else 0
                        likes           = int(r0[1] or 0) if len(r0) > 1 else 0
                        comments        = int(r0[2] or 0) if len(r0) > 2 else 0
                        shares          = int(r0[3] or 0) if len(r0) > 3 else 0
                        avg_watch       = round(float(r0[4]), 1) if len(r0) > 4 and r0[4] is not None else None
                        minutes_watched = int(float(r0[5] or 0)) if len(r0) > 5 else 0
                        analytics_source = "yt-analytics"
                    else:
                        # Fallback attempt: day-dimension report to avoid edge-case
                        # empty aggregate responses.
                        an_day = await client.get(
                            "https://youtubeanalytics.googleapis.com/v2/reports",
                            params={
                                "ids":        "channel==MINE",
                                "startDate":  thirty,
                                "endDate":    today,
                                "dimensions": "day",
                                "metrics":    "views,likes,comments,shares,averageViewDuration,estimatedMinutesWatched",
                                "sort":       "day",
                            },
                            headers={"Authorization": f"Bearer {access_token}"},
                        )
                        if an_day.status_code == 200:
                            drows = an_day.json().get("rows", []) or []
                            if drows:
                                views           = sum(int(r[1] or 0) for r in drows if isinstance(r, list) and len(r) > 1)
                                likes           = sum(int(r[2] or 0) for r in drows if isinstance(r, list) and len(r) > 2)
                                comments        = sum(int(r[3] or 0) for r in drows if isinstance(r, list) and len(r) > 3)
                                shares          = sum(int(r[4] or 0) for r in drows if isinstance(r, list) and len(r) > 4)
                                dur_vals        = [float(r[5]) for r in drows if isinstance(r, list) and len(r) > 5 and r[5] is not None]
                                avg_watch       = round(sum(dur_vals) / len(dur_vals), 1) if dur_vals else None
                                minutes_watched = int(sum(float(r[6] or 0) for r in drows if isinstance(r, list) and len(r) > 6))
                                analytics_source = "yt-analytics"
                            else:
                                analytics_diagnostic = "yt_analytics_200_empty_rows"
                        else:
                            analytics_diagnostic = f"yt_analytics_day_http_{an_day.status_code}"
                elif an.status_code == 403:
                    logger.warning(
                        "YouTube Analytics 403 - yt-analytics.readonly missing from token; user must reconnect"
                    )
                    analytics_diagnostic = "yt_analytics_http_403"
                    views    = int(stats.get("viewCount",    0))
                    likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                    comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
                else:
                    analytics_diagnostic = f"yt_analytics_http_{an.status_code}"
                    views    = int(stats.get("viewCount",    0))
                    likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                    comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
            except Exception as ae:
                logger.warning(f"YouTube Analytics error (non-fatal): {ae}")
                analytics_diagnostic = f"yt_analytics_exception:{type(ae).__name__}"
                views    = int(stats.get("viewCount",    0))
                likes    = int(stats.get("likeCount",    0)) if "likeCount"    in stats else 0
                comments = int(stats.get("commentCount", 0)) if "commentCount" in stats else 0

            # Shorts-focused rollup for high-scale analytics quality.
            # We classify Shorts as <= 60s duration from recent uploads playlist.
            try:
                ch_details = await client.get(
                    "https://www.googleapis.com/youtube/v3/channels",
                    params={"part": "contentDetails", "mine": "true"},
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if ch_details.status_code == 200:
                    items2 = (ch_details.json() or {}).get("items") or []
                    uploads_playlist = (((items2[0] if items2 else {}).get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
                    if uploads_playlist:
                        pl = await client.get(
                            "https://www.googleapis.com/youtube/v3/playlistItems",
                            params={"part": "contentDetails", "playlistId": uploads_playlist, "maxResults": 50},
                            headers={"Authorization": f"Bearer {access_token}"},
                        )
                        if pl.status_code == 200:
                            ids = [
                                ((it.get("contentDetails") or {}).get("videoId") or "")
                                for it in ((pl.json() or {}).get("items") or [])
                            ]
                            ids = [v for v in ids if v]
                            if ids:
                                vids = await client.get(
                                    "https://www.googleapis.com/youtube/v3/videos",
                                    params={"part": "contentDetails,statistics", "id": ",".join(ids[:50]), "maxResults": 50},
                                    headers={"Authorization": f"Bearer {access_token}"},
                                )
                                if vids.status_code == 200:
                                    for v in (vids.json() or {}).get("items") or []:
                                        cd = v.get("contentDetails") or {}
                                        st = v.get("statistics") or {}
                                        if _dur_to_seconds(str(cd.get("duration") or "")) <= 60:
                                            shorts_count += 1
                                            shorts_views += int(st.get("viewCount") or 0)
                                            shorts_likes += int(st.get("likeCount") or 0)
                                            shorts_comments += int(st.get("commentCount") or 0)
            except Exception as _shorts_e:
                logger.debug(f"YouTube Shorts rollup skipped: {_shorts_e}")

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
                "shorts_views": shorts_views,
                "shorts_likes": shorts_likes,
                "shorts_comments": shorts_comments,
                "shorts_count": shorts_count,
                "analytics_diagnostic": analytics_diagnostic,
            }
    except Exception as e:
        logger.error(f"YouTube metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_instagram_metrics(access_token: str, ig_user_id: str) -> dict:
    """
    Instagram Graph API — Reels insights.
    Attempt 1: instagram_manage_insights (plays, reach, saved, shares, likes, comments)
    Attempt 2: basic instagram_basic fields (like_count, comments_count) — no advanced scope needed.
    """
    if not access_token or not ig_user_id:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            media = await client.get(
                f"https://graph.facebook.com/v21.0/{ig_user_id}/media",
                params={"access_token": access_token, "fields": "id,media_type,media_product_type,permalink,timestamp", "limit": 30},
            )
            if media.status_code != 200:
                # Without instagram_basic, /media often 403/400 — keep status live so rollups still count other platforms.
                return instagram_account_degraded_live(http_status=media.status_code, ig_user_id=ig_user_id)

            items = media.json().get("data", []) or []
            reels = [
                it for it in items
                if str(it.get("media_product_type") or "").upper() == "REELS"
                or "/reel/" in str(it.get("permalink") or "").lower()
            ]
            if not reels:
                return {"status": "live", "views": 0, "likes": 0, "comments": 0,
                        "saves": 0, "reach": 0, "shares": 0, "video_count": 0,
                        "analytics_source": None}

            total_views = total_likes = total_comments = 0
            total_saves = total_reach = total_shares = 0
            analytics_source = None
            used_fallback = False

            video_items = [it for it in reels if str(it.get("media_type") or "").upper() in ("VIDEO", "REELS")]
            for item in video_items[:10]:
                media_type = (item.get("media_type") or "IMAGE").upper()
                # "plays" on IMAGE/CAROUSEL silently kills the insights call
                if media_type in ("VIDEO", "REELS"):
                    metric_str = "plays,reach,saved,shares,comments,likes"
                    view_key   = "plays"
                else:
                    metric_str = "impressions,reach,saved,shares,comments,likes"
                    view_key   = "impressions"

                # ── Attempt 1: instagram_manage_insights ──────────────────────
                ins = await client.get(
                    f"https://graph.facebook.com/v21.0/{item['id']}/insights",
                    params={"access_token": access_token, "metric": metric_str},
                )
                if ins.status_code == 200:
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
                else:
                    # ── Attempt 2: basic media fields (instagram_basic only) ───
                    # like_count and comments_count are always available.
                    # views/plays unavailable without manage_insights.
                    fallback = await client.get(
                        f"https://graph.facebook.com/v21.0/{item['id']}",
                        params={"access_token": access_token,
                                "fields": "like_count,comments_count"},
                    )
                    if fallback.status_code == 200:
                        fb = fallback.json()
                        total_likes    += int(fb.get("like_count")     or 0)
                        total_comments += int(fb.get("comments_count") or 0)
                        used_fallback   = True

            if analytics_source is None and used_fallback:
                analytics_source = "instagram_basic_fallback"

            return {
                "status":           "live",
                "analytics_source": analytics_source,
                "views":            total_views,
                "likes":            total_likes,
                "comments":         total_comments,
                "saves":            total_saves,
                "reach":            total_reach,
                "shares":           total_shares,
                "video_count":      len(video_items),
            }
    except Exception as e:
        logger.error(f"Instagram metrics error: {e}")
        return {"status": "error", "error": str(e)}


async def _fetch_facebook_metrics(access_token: str, page_id: str) -> dict:
    """
    Facebook Graph API — Page video insights.
    Attempt 1: read_insights scope (total_video_views, reactions, comments, shares)
    Attempt 2: basic video fields (video_views, reactions.summary, comments.summary, shares)
               — available with just a page access token, no advanced scope needed.
    """
    if not access_token or not page_id:
        return {"status": "not_connected"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            vids = await client.get(
                f"https://graph.facebook.com/v21.0/{page_id}/videos",
                params={"access_token": access_token, "fields": "id,created_time,permalink_url", "limit": 30},
            )
            if vids.status_code != 200:
                fb_feed = await facebook_page_feed_reel_engagement_rollups(client, access_token, page_id)
                if fb_feed:
                    return fb_feed
                return {"status": "error", "error": f"HTTP {vids.status_code}"}

            all_videos = vids.json().get("data", []) or []
            videos = [v for v in all_videos if "/reel/" in str(v.get("permalink_url") or "").lower()]
            if not videos:
                fb_feed = await facebook_page_feed_reel_engagement_rollups(client, access_token, page_id)
                if fb_feed:
                    return fb_feed
                followers = 0
                try:
                    pg = await client.get(
                        f"https://graph.facebook.com/v21.0/{page_id}",
                        params={"access_token": access_token, "fields": "followers_count,fan_count"},
                    )
                    if pg.status_code == 200:
                        pg_data = pg.json()
                        followers = pg_data.get("followers_count") or pg_data.get("fan_count") or 0
                except Exception as e:
                    logger.debug("FB metrics: followers fetch failed page_id=%s: %s", page_id, e)
                return {"status": "live", "views": 0, "reactions": 0, "comments": 0,
                        "shares": 0, "followers": followers, "video_count": 0,
                        "analytics_source": None}

            total_views = total_reactions = total_comments = total_shares = 0
            analytics_source = None

            for vid in videos[:10]:
                try:
                    got_vid_stats = False

                    # ── Attempt 1: read_insights scope ────────────────────────
                    ins = await client.get(
                        f"https://graph.facebook.com/v21.0/{vid['id']}",
                        params={
                            "access_token": access_token,
                            "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_shares,total_video_comments)",
                        },
                    )
                    if ins.status_code == 200:
                        insights_data = ins.json().get("insights", {}).get("data", []) or []
                        if insights_data:
                            analytics_source = "read_insights+pages_read_engagement"
                            for m in insights_data:
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
                            got_vid_stats = True

                    # ── Attempt 2: basic video fields (no read_insights needed) ─
                    if not got_vid_stats:
                        fallback = await client.get(
                            f"https://graph.facebook.com/v21.0/{vid['id']}",
                            params={
                                "access_token": access_token,
                                "fields": "video_views,reactions.summary(true),comments.summary(true),shares",
                            },
                        )
                        if fallback.status_code == 200:
                            fb = fallback.json()
                            total_views     += int(fb.get("video_views") or 0)
                            total_reactions += int((fb.get("reactions") or {}).get("summary", {}).get("total_count") or 0)
                            total_comments  += int((fb.get("comments")  or {}).get("summary", {}).get("total_count") or 0)
                            total_shares    += int((fb.get("shares")    or {}).get("count") or 0)
                            if analytics_source is None:
                                analytics_source = "basic_video_fields_fallback"

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
                "status":           "live",
                "analytics_source": analytics_source,
                "views":            total_views,
                "reactions":        total_reactions,
                "comments":         total_comments,
                "shares":           total_shares,
                "followers":        followers,
                "video_count":      len(videos),
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
    except Exception as e:
        logger.debug("platform_metrics_cache write failed user_id=%s: %s", user_id, e)


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
            corrected_payload, _ = await _apply_metrics_standards_for_user(user_id, result)
            _platform_metrics_cache[user_id] = {"fetched_at": now, "data": corrected_payload}
            return corrected_payload

    # 2) DB cache (survives restarts)
    async with db_pool.acquire() as conn:
        db_cached = await _platform_metrics_db_cache_get(conn, user_id)
        if db_cached and not force:
            corrected_payload, _ = await _apply_metrics_standards_for_user(user_id, db_cached)
            _platform_metrics_cache[user_id] = {"fetched_at": now, "data": corrected_payload}
            return corrected_payload

    # Scale-safe path: API reads cache/rollups; polling runs in background job service.
    if not force:
        async with db_pool.acquire() as conn:
            roll = await conn.fetchrow(
                """
                SELECT views, likes, comments, shares
                  FROM platform_user_metrics_rollups_daily
                 WHERE user_id = $1
                 ORDER BY day DESC
                 LIMIT 1
                """,
                user["id"],
            )
        agg = {
            "views": int((roll or {}).get("views") or 0),
            "likes": int((roll or {}).get("likes") or 0),
            "comments": int((roll or {}).get("comments") or 0),
            "shares": int((roll or {}).get("shares") or 0),
            "platforms_included": [],
        }
        # No platform payload here — only clamp/sanitize DB rollup (same non-negative rules)
        agg_sanitized, _ = _sanitize_aggregate_numbers(agg)
        return {
            "platforms": {},
            "aggregate": agg_sanitized,
            "cached": True,
            "cache_source": "rollup",
            "fetched_at": None,
            "cache_age_minutes": 9999,
            "next_refresh_minutes": int(_PLATFORM_CACHE_TTL / 60),
        }

    # Manual refresh path (?force=true): run scalable poller, then read DB cache.
    try:
        from services.platform_metrics_job import refresh_platform_metrics_for_user

        await refresh_platform_metrics_for_user(db_pool, user["id"])
    except Exception as e:
        logger.warning(f"platform-metrics forced refresh failed: {e}")

    async with db_pool.acquire() as conn:
        db_cached_after = await _platform_metrics_db_cache_get(conn, user_id)
    if db_cached_after:
        corrected_payload, _ = await _apply_metrics_standards_for_user(user_id, db_cached_after)
        _platform_metrics_cache[user_id] = {"fetched_at": now, "data": corrected_payload}
        return corrected_payload

    async with db_pool.acquire() as conn:
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
    token_lists: dict = {"tiktok": [], "youtube": [], "instagram": [], "facebook": []}
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
            token_lists.setdefault(plat, []).append(decrypted)
            token_map[plat] = decrypted

    # Refresh OAuth tokens before live API calls (YouTube ~1h; TikTok ~24h; Meta ~60 days).
    uid_str = str(user["id"])
    try:
        from stages.publish_stage import _refresh_tiktok_token, _refresh_youtube_token, _refresh_meta_token

        if token_lists.get("tiktok"):
            refreshed = []
            for tok in token_lists["tiktok"]:
                try:
                    refreshed.append(await _refresh_tiktok_token(dict(tok), db_pool=db_pool, user_id=uid_str))
                except Exception:
                    refreshed.append(tok)
            token_lists["tiktok"] = refreshed
            token_map["tiktok"] = refreshed[-1]
        if token_lists.get("youtube"):
            refreshed = []
            for tok in token_lists["youtube"]:
                try:
                    refreshed.append(await _refresh_youtube_token(dict(tok), db_pool=db_pool, user_id=uid_str))
                except Exception:
                    refreshed.append(tok)
            token_lists["youtube"] = refreshed
            token_map["youtube"] = refreshed[-1]
        if token_lists.get("instagram"):
            refreshed = []
            for tok in token_lists["instagram"]:
                try:
                    refreshed.append(await _refresh_meta_token(dict(tok), platform="instagram", db_pool=db_pool, user_id=uid_str))
                except Exception:
                    refreshed.append(tok)
            token_lists["instagram"] = refreshed
            token_map["instagram"] = refreshed[-1]
        if token_lists.get("facebook"):
            refreshed = []
            for tok in token_lists["facebook"]:
                try:
                    refreshed.append(await _refresh_meta_token(dict(tok), platform="facebook", db_pool=db_pool, user_id=uid_str))
                except Exception:
                    refreshed.append(tok)
            token_lists["facebook"] = refreshed
            token_map["facebook"] = refreshed[-1]
    except Exception as _tok_e:
        logger.warning(f"Platform metrics OAuth refresh skipped: {_tok_e}")

    async def run_tiktok():
        tokens = token_lists.get("tiktok") or []
        if not tokens:
            return {"status": "not_connected"}
        rs = await _asyncio.gather(*[_fetch_tiktok_metrics((t or {}).get("access_token", "")) for t in tokens], return_exceptions=True)
        norm = [{"status": "error", "error": str(r)} if isinstance(r, Exception) else r for r in rs]
        return _combine_platform_live_results("tiktok", norm)

    async def run_youtube():
        tokens = token_lists.get("youtube") or []
        if not tokens:
            return {"status": "not_connected"}
        rs = await _asyncio.gather(*[_fetch_youtube_metrics((t or {}).get("access_token", "")) for t in tokens], return_exceptions=True)
        norm = [{"status": "error", "error": str(r)} if isinstance(r, Exception) else r for r in rs]
        return _combine_platform_live_results("youtube", norm)

    async def run_instagram():
        tokens = token_lists.get("instagram") or []
        if not tokens:
            return {"status": "not_connected"}
        coros = []
        for t in tokens:
            ig_id = ((t or {}).get("ig_user_id") or (t or {}).get("instagram_user_id") or (t or {}).get("instagram_page_id") or "")
            coros.append(_fetch_instagram_metrics((t or {}).get("access_token", ""), ig_id))
        rs = await _asyncio.gather(*coros, return_exceptions=True)
        norm = [{"status": "error", "error": str(r)} if isinstance(r, Exception) else r for r in rs]
        return _combine_platform_live_results("instagram", norm)

    async def run_facebook():
        tokens = token_lists.get("facebook") or []
        if not tokens:
            return {"status": "not_connected"}
        coros = []
        for t in tokens:
            page_id = ((t or {}).get("page_id") or (t or {}).get("facebook_page_id") or (t or {}).get("fb_page_id") or "")
            coros.append(_fetch_facebook_metrics((t or {}).get("access_token", ""), page_id))
        rs = await _asyncio.gather(*coros, return_exceptions=True)
        norm = [{"status": "error", "error": str(r)} if isinstance(r, Exception) else r for r in rs]
        return _combine_platform_live_results("facebook", norm)

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
        "aggregate": _aggregate_platform_metrics_live(platforms_result),
    }

    _platform_metrics_cache[user_id] = {"fetched_at": now, "data": output}
    async with db_pool.acquire() as conn:
        await _platform_metrics_db_cache_set(conn, user_id, output)
    corrected_payload, _ = await _apply_metrics_standards_for_user(user_id, output)
    _platform_metrics_cache[user_id] = {"fetched_at": now, "data": corrected_payload}
    return corrected_payload


@app.get("/api/analytics/platform-metrics/cached")
async def get_platform_metrics_cached(user: dict = Depends(get_current_user_readonly)):
    """Return DB-cached platform metrics only (no live API calls)."""
    user_id = str(user["id"])
    async with db_pool.acquire() as conn:
        cached = await _platform_metrics_db_cache_get(conn, user_id)
    if cached:
        corrected_payload, _ = await _apply_metrics_standards_for_user(user_id, cached)
        _platform_metrics_cache[user_id] = {"fetched_at": _time.time(), "data": corrected_payload}
        if _should_background_refresh_platform_cache(corrected_payload):
            _schedule_platform_cache_refresh_if_needed(user_id)
        return corrected_payload
    _schedule_platform_cache_refresh_if_needed(user_id)
    return {"platforms": {}, "aggregate": {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}, "cached": True, "cache_source": "db", "fetched_at": None}


@app.get("/api/analytics/sync-status")
async def get_analytics_sync_status(user: dict = Depends(get_current_user_readonly)):
    """
    Last-known analytics job state for this user (in-memory; resets on API restart).

    Tracks:
      - upload_analytics: bulk POST /api/uploads/sync-analytics/all
      - platform_metrics: POST /api/analytics/refresh-all, stale-cache background refresh, tail refresh after bulk sync
    """
    uid = str(user["id"])
    snap = await _analytics_sync_status_get(uid)
    return {
        "user_id": uid,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "upload_analytics": snap.get("upload_analytics") or {},
        "platform_metrics": snap.get("platform_metrics") or {},
    }


@app.get("/api/analytics/pikzels-v2-usage")
async def analytics_pikzels_v2_usage(range: str = "30d", user: dict = Depends(get_current_user_readonly)):
    """Your Pikzels v2 proxy calls (from wallet ledger rows with meta.pikzels_v2_op)."""
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    uid = str(user["id"])
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT COALESCE(meta->>'pikzels_v2_op', 'unknown') AS op,
                   COUNT(*)::int AS n
            FROM token_ledger
            WHERE user_id = $1::uuid
              AND created_at >= $2
              AND reason = 'thumbnail_studio'
              AND meta->>'pikzels_v2_op' IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC
            """,
            uid,
            since,
        )
        total = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM token_ledger
            WHERE user_id = $1::uuid
              AND created_at >= $2
              AND reason = 'thumbnail_studio'
              AND meta->>'pikzels_v2_op' IS NOT NULL
            """,
            uid,
            since,
        )
    return {
        "range": range,
        "since": since.isoformat(),
        "total_calls": int(total or 0),
        "by_operation": [{"op": str(r["op"]), "count": int(r["n"])} for r in rows or []],
    }


@app.post("/api/analytics/refresh-all")
async def refresh_all_analytics(
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Force-refresh cross-platform aggregate cache for this user.
    This keeps dashboard/queue/analytics numerics aligned after a manual refresh.
    """
    uid_s = str(user["id"])

    async def _do_refresh(trigger: str = "refresh-all") -> bool:
        try:
            return await _analytics_track_platform_metrics_refresh(uid_s, trigger)
        except Exception as e:
            logger.warning(f"/api/analytics/refresh-all failed user={user.get('id')}: {e}")
            await audit_log(
                str(user.get("id")),
                "ANALYTICS_REFRESH_ALL_FAILED",
                event_category="DATA_INTEGRITY",
                resource_type="platform_metrics_cache",
                resource_id=str(user.get("id")),
                details={"error": str(e)},
                severity="ERROR",
                outcome="FAILED",
            )
            return False

    if async_mode:
        await _analytics_sync_status_patch(
            uid_s,
            "platform_metrics",
            {
                "status": "queued",
                "queued": True,
                "started_at": None,
                "finished_at": None,
                "duration_ms": None,
                "ok": None,
                "error": None,
                "trigger": "refresh-all-async",
            },
        )

        async def _bg_refresh() -> None:
            await _do_refresh("refresh-all-async")

        background_tasks.add_task(_bg_refresh)
        return {"ok": True, "queued": True}

    refreshed = await _do_refresh("refresh-all")

    standards = {"corrected": False, "aggregate": {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}}
    async with db_pool.acquire() as conn:
        cached = await _platform_metrics_db_cache_get(conn, str(user["id"]))
        if not cached:
            await audit_log(
                str(user.get("id")),
                "ANALYTICS_REFRESH_ALL_CACHE_MISS",
                event_category="DATA_INTEGRITY",
                resource_type="platform_metrics_cache",
                resource_id=str(user.get("id")),
                details={"note": "no cached payload available after refresh"},
                severity="WARNING",
                outcome="FAILED",
            )
        payload, standards = await _enforce_metrics_rollup_standards(
            conn,
            str(user["id"]),
            _attach_aggregate_to_metrics_payload(cached or {}),
        )
    if standards.get("corrected"):
        await audit_log(
            str(user.get("id")),
            "ANALYTICS_REFRESH_ALL_CORRECTED",
            event_category="DATA_INTEGRITY",
            resource_type="platform_metrics_cache",
            resource_id=str(user.get("id")),
            details={"aggregate": standards.get("aggregate") or {}},
            severity="WARNING",
            outcome="SUCCESS",
        )
    return {
        "ok": bool(refreshed),
        "refreshed": bool(refreshed),
        "standards": standards,
        "fetched_at": payload.get("fetched_at"),
        "aggregate": payload.get("aggregate") or {
            "views": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "platforms_included": [],
        },
    }

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
# User analytics overview (KPI dashboard / kpi.html)
# ============================================================
@app.get("/api/analytics/overview")
async def analytics_overview(
    days: int = Query(30, ge=1, le=3650),
    platform: str = Query(
        "all",
        description=(
            "Canonical slug (tiktok, youtube, instagram, facebook, …) or alias: "
            "instagram_reels, facebook_reels (reel catalog slice). all = no filter."
        ),
    ),
    user: dict = Depends(get_current_user),
):
    """High-level KPI summary for analytics dashboard (same upload/engagement definitions as GET /api/analytics)."""
    from services.canonical_engagement import (
        ROLLUP_VERSION,
        compute_canonical_engagement_rollup,
        engagement_time_window_for_overview_days,
        engagement_window_api_dict,
    )

    now = _now_utc()
    since, until_excl = engagement_time_window_for_overview_days(days, now=now)
    try:
        plat_filter = resolve_analytics_platform_filter(platform)
    except ValueError as e:
        allowed = ", ".join(list_analytics_platform_query_values())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform. Use all or one of: {allowed}",
        ) from e
    plat = plat_filter.platform

    async with db_pool.acquire() as conn:
        bundle = await _user_upload_kpi_bundle(
            conn, str(user["id"]), since=since, until=until_excl, platform=plat
        )
        eng = bundle["engagement"]

        avg_processing_seconds = 0.0
        try:
            params_avg: list[Any] = [user["id"], since, until_excl]
            plat_sql = ""
            if plat:
                plat_sql = (
                    f" AND EXISTS (SELECT 1 FROM unnest(COALESCE(platforms, ARRAY[]::text[])) AS _plat "
                    f"WHERE lower(_plat::text) = ${len(params_avg) + 1})"
                )
                params_avg.append(plat)
            ravg = await conn.fetchrow(
                f"""
                SELECT COALESCE(AVG(EXTRACT(EPOCH FROM (processing_finished_at - processing_started_at))), 0)::double precision AS s
                FROM uploads
                WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
                  AND status IN {SUCCESSFUL_STATUS_SQL_IN}
                  AND processing_started_at IS NOT NULL
                  AND processing_finished_at IS NOT NULL
                  AND processing_finished_at > processing_started_at
                  {plat_sql}
                """,
                *params_avg,
            )
            avg_processing_seconds = float(ravg["s"] or 0) if ravg else 0.0
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise

        cost_total = 0.0
        try:
            params_cost: list[Any] = [user["id"], since, until_excl]
            plat_sql_c = ""
            if plat:
                plat_sql_c = (
                    f" AND EXISTS (SELECT 1 FROM unnest(COALESCE(platforms, ARRAY[]::text[])) AS _plat "
                    f"WHERE lower(_plat::text) = ${len(params_cost) + 1})"
                )
                params_cost.append(plat)
            cost_total = float(
                await conn.fetchval(
                    f"""
                    SELECT COALESCE(SUM(cost_attributed), 0)::double precision
                    FROM uploads
                    WHERE user_id = $1 AND created_at >= $2 AND created_at < $3
                    {plat_sql_c}
                    """,
                    *params_cost,
                )
                or 0
            )
        except Exception as e:
            if e.__class__.__name__ != "UndefinedColumnError":
                raise

        revenue_total = 0.0
        try:
            rev = await conn.fetchval(
                "SELECT COALESCE(SUM(amount), 0)::decimal FROM revenue_tracking WHERE user_id = $1 AND created_at >= $2 AND created_at < $3",
                user["id"], since, until_excl,
            )
            revenue_total = float(rev or 0)
        except Exception as e:
            if e.__class__.__name__ != "UndefinedTableError":
                raise

        try:
            cr = await compute_canonical_engagement_rollup(
                conn,
                str(user["id"]),
                window_start=since,
                window_end_exclusive=until_excl,
                platform=plat,
                catalog_content_kind=plat_filter.catalog_content_kind,
            )
        except Exception as e:
            logger.warning("analytics_overview: canonical engagement rollup failed: %s", e)
            cr = {
                "views": int(eng.get("views") or 0),
                "likes": int(eng.get("likes") or 0),
                "comments": int(eng.get("comments") or 0),
                "shares": int(eng.get("shares") or 0),
                "breakdown": {
                    "compute": {
                        "rollup_version": ROLLUP_VERSION,
                        "complete": False,
                        "warnings": ["rollup_exception"],
                        "error_detail": str(e)[:500],
                    },
                },
                "catalog_tracked_videos": 0,
                "rollup_version": ROLLUP_VERSION,
                "rollup_rule": "fallback_upload_bundle_only",
                "kpi_sources": {"error": str(e), "rollup_version": ROLLUP_VERSION},
            }
        win_meta = engagement_window_api_dict(start=since, end_exclusive=until_excl)
        canon = dict(cr)
        canon["engagement_window_utc"] = win_meta

    pf = plat or "all"
    catalog_surface = "reels" if plat_filter.catalog_content_kind else "all"
    return {
        "range_days": days,
        "since": since.isoformat(),
        "until_exclusive": until_excl.isoformat(),
        "filters": {
            "platform": pf,
            "platform_display": plat_filter.display_name,
            "catalog_surface": catalog_surface,
            "platform_query": plat_filter.raw_query or pf,
        },
        "uploads": {
            "total": bundle["total"],
            "completed": bundle["successful"],
            "failed": bundle["failed"],
            "in_queue": bundle["in_queue"],
            "success_rate_pct": bundle["success_rate_pct"],
            "avg_processing_seconds": avg_processing_seconds,
        },
        "engagement": {
            "views": int(canon["views"] or 0),
            "likes": int(canon["likes"] or 0),
            "comments": int(canon["comments"] or 0),
            "shares": int(canon["shares"] or 0),
            "breakdown": canon["breakdown"],
            "rollup_rule": canon["rollup_rule"],
            "rollup_version": canon.get("rollup_version"),
            "catalog_tracked_videos": int(canon["catalog_tracked_videos"] or 0),
            "engagement_window_utc": canon.get("engagement_window_utc"),
            "kpi_sources": canon.get("kpi_sources"),
        },
        "costs": {
            "cost_total": cost_total,
        },
        "revenue": {
            "revenue_total": revenue_total,
            "scope_note": metric_defs.REVENUE_SCOPE_NOTE_ANALYTICS_OVERVIEW,
        },
        "metric_definitions": metric_defs.for_analytics_overview(),
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




async def _analytics_export_rows(conn, user_id, since):
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
            user_id, since
        )
    except Exception as e:
        if e.__class__.__name__ != "UndefinedColumnError":
            raise
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
            user_id, since
        )
    return rows


def _analytics_export_data(rows):
    return [
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


def _analytics_export_csv_bytes(data):
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
    return output.getvalue().encode("utf-8")


@app.get("/api/analytics/export")
async def analytics_export(days: int = Query(30, ge=1, le=3650), format: str = Query("csv"), user: dict = Depends(get_current_user)):
    """Export analytics for the last N days as CSV (default) or JSON."""
    since = _now_utc() - timedelta(days=days)
    async with db_pool.acquire() as conn:
        rows = await _analytics_export_rows(conn, user["id"], since)
    data = _analytics_export_data(rows)

    if format.lower() == "json":
        return {"range_days": days, "since": since.isoformat(), "rows": data}

    # CSV default
    csv_bytes = _analytics_export_csv_bytes(data)
    headers = {"Content-Disposition": f'attachment; filename="uploadm8-analytics-{days}d.csv"'}
    return Response(content=csv_bytes, media_type="text/csv", headers=headers)


async def _process_analytics_export_job(job_id: str, user_id: str, email: str, name: str, token_plain: str, days: int, fmt: str):
    since = _now_utc() - timedelta(days=days)
    fmt = (fmt or "csv").lower()
    if fmt not in ("csv", "json"):
        fmt = "csv"
    try:
        async with db_pool.acquire() as conn:
            rows = await _analytics_export_rows(conn, user_id, since)
            data = _analytics_export_data(rows)
            if fmt == "json":
                blob = json.dumps({"range_days": days, "since": since.isoformat(), "rows": data}).encode("utf-8")
                content_type = "application/json"
                filename = f"uploadm8-analytics-{days}d.json"
            else:
                blob = _analytics_export_csv_bytes(data)
                content_type = "text/csv"
                filename = f"uploadm8-analytics-{days}d.csv"
            await conn.execute(
                """
                UPDATE export_jobs
                SET status = 'ready', file_blob = $2, content_type = $3, filename = $4, ready_at = NOW()
                WHERE id = $1
                """,
                job_id, blob, content_type, filename,
            )
            exp = await conn.fetchval("SELECT expires_at FROM export_jobs WHERE id = $1", job_id)
        download_url = f"{FRONTEND_URL.rstrip('/')}/api/exports/download?token={quote(token_plain)}"
        expires_lbl = exp.strftime("%B %d, %Y %H:%M UTC") if exp else "within 24 hours"
        await send_report_ready_email(
            email,
            name or "there",
            f"Analytics export ({days}d)",
            download_url,
            expires_lbl,
        )
    except Exception as e:
        logger.warning(f"analytics export job failed {job_id}: {e}")
        try:
            async with db_pool.acquire() as conn:
                await conn.execute("UPDATE export_jobs SET status = 'failed' WHERE id = $1", job_id)
        except Exception as e:
            logger.warning("analytics export: could not mark job failed id=%s: %s", job_id, e)


@app.post("/api/analytics/export/request")
async def request_analytics_export(
    background_tasks: BackgroundTasks,
    days: int = Query(30, ge=1, le=3650),
    format: str = Query("csv"),
    user: dict = Depends(get_current_user),
):
    """Async analytics export: sends email when report is ready."""
    token_plain = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token_plain.encode()).hexdigest()
    expires_at = _now_utc() + timedelta(hours=24)
    async with db_pool.acquire() as conn:
        urow = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1", user["id"])
        job = await conn.fetchrow(
            """
            INSERT INTO export_jobs (user_id, token_hash, report_type, format, days, status, expires_at)
            VALUES ($1, $2, 'analytics', $3, $4, 'pending', $5)
            RETURNING id
            """,
            user["id"], token_hash, (format or "csv").lower(), int(days), expires_at,
        )
    background_tasks.add_task(
        _process_analytics_export_job,
        str(job["id"]),
        str(user["id"]),
        (urow["email"] if urow else user.get("email") or ""),
        (urow["name"] if urow else user.get("name") or "there"),
        token_plain,
        int(days),
        format,
    )
    return {"ok": True, "status": "queued"}


@app.get("/api/exports/download")
async def download_export(token: str = Query(...)):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT status, file_blob, content_type, filename, expires_at
            FROM export_jobs
            WHERE token_hash = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            token_hash,
        )
    if not row:
        raise HTTPException(404, "Export not found")
    if row["expires_at"] < _now_utc():
        raise HTTPException(410, "Export link expired")
    if row["status"] != "ready" or not row.get("file_blob"):
        raise HTTPException(409, "Export is not ready yet")
    blob = bytes(row["file_blob"])
    headers = {"Content-Disposition": f'attachment; filename="{row.get("filename") or "export.bin"}"'}
    return Response(content=blob, media_type=row.get("content_type") or "application/octet-stream", headers=headers)
# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED CONTENT CATALOG  /api/catalog/...
#
# Design keys
# -----------
# • platform_content_items: one row per (user, platform, account_id, video_id)
# • source = external | uploadm8 | linked
# • /api/catalog/sync        — trigger incremental (or full) catalog refresh
# • /api/catalog/sync-status — per-token progress
# • /api/catalog/content     — paginated list (sortable, filterable)
# • /api/catalog/aggregate   — aggregated views/likes/comments/shares for chips
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/catalog/sync")
async def trigger_catalog_sync(
    background_tasks: BackgroundTasks,
    force_full: bool = Query(False, description="Clear cursors and re-scan from the beginning"),
    async_mode: bool = Query(True),
    user: dict = Depends(get_current_user),
):
    """
    Trigger a catalog sync for the current user.
    async_mode=true (default): queues in background, returns immediately.
    async_mode=false:          runs synchronously, returns totals (max 30 s).
    """
    from services.catalog_sync import sync_catalog_for_user

    if async_mode:
        background_tasks.add_task(sync_catalog_for_user, db_pool, str(user["id"]), force_full)
        return {"ok": True, "status": "queued", "async_mode": True}

    try:
        totals = await asyncio.wait_for(
            sync_catalog_for_user(db_pool, str(user["id"]), force_full),
            timeout=30.0,
        )
        return {"ok": True, "status": "done", "async_mode": False, **totals}
    except asyncio.TimeoutError:
        return {"ok": True, "status": "running", "async_mode": False, "note": "still running in background"}


@app.get("/api/catalog/sync-status")
async def get_catalog_sync_status(user: dict = Depends(get_current_user)):
    """Return per-token sync state (platform, status, last_synced_at, cursor, counts)."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.platform, s.account_id, s.status, s.last_synced_at,
                   s.total_discovered, s.total_linked, s.error_detail,
                   s.next_cursor IS NOT NULL AS has_more_pages
            FROM platform_content_sync_state s
            WHERE s.user_id = $1
            ORDER BY s.last_synced_at DESC NULLS LAST
            """,
            user["id"],
        )
    return [
        {
            "platform": r["platform"],
            "account_id": r["account_id"],
            "status": r["status"],
            "last_synced_at": r["last_synced_at"].isoformat() if r["last_synced_at"] else None,
            "total_discovered": r["total_discovered"],
            "total_linked": r["total_linked"],
            "has_more_pages": bool(r["has_more_pages"]),
            "error_detail": r["error_detail"],
        }
        for r in rows
    ]


@app.get("/api/catalog/content")
async def get_catalog_content(
    platform: Optional[str] = Query(None),
    source: Optional[str] = Query(None, description="external|uploadm8|linked|all"),
    account_id: Optional[str] = Query(None, description="Filter to one connected account (catalog account_id)"),
    sort: str = Query("views", description="views|likes|published_at|engagement"),
    order: str = Query("desc"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    days: Optional[int] = Query(None, description="Filter to last N days (published_at)"),
    start: Optional[str] = Query(None, description="ISO-8601 UTC inclusive start (use with end; excludes rolling days)"),
    end: Optional[str] = Query(None, description="ISO-8601 UTC exclusive end"),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Paginated list of all known content items — UploadM8 + external.
    Each row includes source badge, upload_id if linked, and per-platform URL.
    """
    uid = str(user["id"])
    conditions = ["pci.user_id = $1"]
    params: List[Any] = [uid]

    if platform:
        conditions.append(f"pci.platform = ${len(params)+1}")
        params.append(platform.lower())
    if source and source != "all":
        conditions.append(f"pci.source = ${len(params)+1}")
        params.append(source.lower())
    if account_id and str(account_id).strip():
        conditions.append(f"pci.account_id = ${len(params)+1}")
        params.append(str(account_id).strip())
    win_start = _parse_iso_ts(start)
    win_end = _parse_iso_ts(end)
    # Use upload completion / created time when catalog publish date is missing or placeholder.
    _eff_ts = "COALESCE(pci.published_at, u.completed_at, u.created_at)"
    if win_start is not None and win_end is not None:
        conditions.append(f"{_eff_ts} >= ${len(params)+1}")
        params.append(win_start)
        conditions.append(f"{_eff_ts} < ${len(params)+1}")
        params.append(win_end)
        conditions.append(f"{_eff_ts} IS NOT NULL")
    elif days and days > 0:
        conditions.append(f"{_eff_ts} >= NOW() - INTERVAL '{int(days)} days'")
        conditions.append(f"{_eff_ts} IS NOT NULL")

    where = " AND ".join(conditions)

    sort_col_map = {
        "views": "pci.views",
        "likes": "pci.likes",
        "published_at": _eff_ts,
        "engagement": "(CASE WHEN pci.views > 0 THEN (pci.likes + pci.comments + pci.shares)::float / pci.views ELSE 0 END)",
    }
    sort_col = sort_col_map.get(sort, "pci.views")
    sort_dir = "DESC" if order.lower() != "asc" else "ASC"
    null_pos = "NULLS LAST" if sort_dir == "DESC" else "NULLS FIRST"

    async with db_pool.acquire() as conn:
        total_row = await conn.fetchrow(
            f"""
            SELECT COUNT(*) FROM platform_content_items pci
            LEFT JOIN uploads u ON u.id = pci.upload_id AND u.user_id = pci.user_id
            WHERE {where}
            """,
            *params,
        )
        rows = await conn.fetch(
            f"""
            SELECT pci.id, pci.platform, pci.account_id, pci.platform_video_id, pci.upload_id, pci.source,
                   pci.content_kind, pci.title, pci.published_at, pci.thumbnail_url, pci.platform_url,
                   pci.duration_seconds, pci.views, pci.likes, pci.comments, pci.shares, pci.metrics_synced_at,
                   pci.created_at,
                   u.title AS upload_title, u.thumbnail_r2_key, u.platform_results AS upload_pr,
                   u.views AS upload_views, u.likes AS upload_likes, u.comments AS upload_comments, u.shares AS upload_shares,
                   u.platforms AS upload_platforms,
                   u.filename AS upload_filename, u.caption AS upload_caption,
                   u.ai_title AS upload_ai_title, u.ai_generated_title AS upload_ai_generated_title,
                   u.completed_at AS upload_completed_at, u.created_at AS upload_created_at,
                   pt.account_name AS token_account_name, pt.account_username AS token_account_username
            FROM platform_content_items pci
            LEFT JOIN uploads u ON u.id = pci.upload_id AND u.user_id = pci.user_id
            LEFT JOIN LATERAL (
                SELECT account_name, account_username
                FROM platform_tokens
                WHERE user_id = pci.user_id
                  AND platform = pci.platform
                  AND account_id IS NOT DISTINCT FROM pci.account_id
                  AND revoked_at IS NULL
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1
            ) pt ON TRUE
            WHERE {where}
            ORDER BY {sort_col} {sort_dir} {null_pos}
            LIMIT ${len(params)+1} OFFSET ${len(params)+2}
            """,
            *params, limit, offset,
        )

    row_dicts = [dict(r) for r in rows]
    thumb_urls: dict[int, str | None] = {}
    keys_to_sign: list[tuple[int, str]] = []
    for i, d in enumerate(row_dicts):
        if d.get("thumbnail_r2_key"):
            nk = _normalize_r2_key(d["thumbnail_r2_key"])
            if nk:
                keys_to_sign.append((i, nk))
    if keys_to_sign:
        try:
            s3 = get_s3_client()
            for i, nk in keys_to_sign:
                try:
                    thumb_urls[i] = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": R2_BUCKET_NAME, "Key": nk},
                        ExpiresIn=3600,
                    )
                except Exception as e:
                    logger.debug("catalog content: thumbnail presign failed idx=%s: %s", i, e)
                    thumb_urls[i] = None
        except Exception as e:
            logger.debug("catalog content: batch thumbnail presign unavailable: %s", e)

    items = []
    for idx, r in enumerate(row_dicts):
        pr_title, pr_m = _catalog_title_and_metrics_from_upload_pr(
            r.get("upload_pr"), r.get("platform"), r.get("platform_video_id")
        )
        v = max(int(r["views"] or 0), int(pr_m["views"] or 0))
        l = max(int(r["likes"] or 0), int(pr_m["likes"] or 0))
        c = max(int(r["comments"] or 0), int(pr_m["comments"] or 0))
        s = max(int(r["shares"] or 0), int(pr_m["shares"] or 0))
        plat_row = str(r.get("platform") or "").lower()
        upl = r.get("upload_platforms") or []
        if isinstance(upl, str):
            try:
                upl = json.loads(upl)
            except Exception:
                upl = []
        targets = [str(x).lower() for x in upl if x]
        merge_u = bool(r.get("upload_id")) and (
            not targets or (len(targets) == 1 and targets[0] == plat_row)
        )
        if merge_u:
            v = max(v, int(r.get("upload_views") or 0))
            l = max(l, int(r.get("upload_likes") or 0))
            c = max(c, int(r.get("upload_comments") or 0))
            s = max(s, int(r.get("upload_shares") or 0))
        eng = round((l + c + s) / v * 100, 2) if v > 0 else 0.0
        raw_title = (r.get("title") or "").strip()
        up_t = (r.get("upload_title") or "").strip()
        pr_t = (pr_title or "").strip() if pr_title else ""
        ai_t = (r.get("upload_ai_title") or r.get("upload_ai_generated_title") or "").strip()
        fn_t = (r.get("upload_filename") or "").strip()
        cap_raw = r.get("upload_caption")
        cap_t = ""
        if isinstance(cap_raw, str) and cap_raw.strip():
            cap_t = cap_raw.strip().split("\n")[0][:500]
        title_out = raw_title or up_t or pr_t or ai_t or fn_t or cap_t or None
        thumb_out = (r.get("thumbnail_url") or "").strip() or thumb_urls.get(idx)
        acct_label = (
            (r.get("token_account_name") or "").strip()
            or (r.get("token_account_username") or "").strip()
            or ""
        )
        eff_pub = r.get("published_at") or r.get("upload_completed_at") or r.get("upload_created_at")
        items.append({
            "id": str(r["id"]),
            "platform": r["platform"],
            "account_id": r["account_id"],
            "account_label": acct_label or None,
            "platform_video_id": r["platform_video_id"],
            "upload_id": str(r["upload_id"]) if r["upload_id"] else None,
            "source": r["source"],
            "content_kind": r["content_kind"],
            "title": title_out,
            "published_at": eff_pub.isoformat() if eff_pub else None,
            "thumbnail_url": thumb_out or None,
            "platform_url": r["platform_url"],
            "duration_seconds": r["duration_seconds"],
            "views": v, "likes": l, "comments": c, "shares": s,
            "engagement_rate": eng,
            "metrics_synced_at": r["metrics_synced_at"].isoformat() if r["metrics_synced_at"] else None,
        })

    return {
        "items": items,
        "total": int(total_row[0] or 0),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/catalog/aggregate")
async def get_catalog_aggregate_endpoint(
    period: Optional[str] = Query(None, description="Time window: '7d','30d','7h','24h','90m','all'. Overrides days."),
    days: Optional[int] = Query(None, description="Legacy: last N days (use period instead)"),
    platform: Optional[str] = Query(None),
    source: Optional[str] = Query(None, description="external|uploadm8|linked"),
    account_id: Optional[str] = Query(None, description="Restrict to one catalog account_id"),
    start: Optional[str] = Query(None, description="ISO-8601 UTC inclusive start (use with end)"),
    end: Optional[str] = Query(None, description="ISO-8601 UTC exclusive end"),
    user: dict = Depends(get_current_user),
):
    """
    Aggregated views / likes / comments / shares + per-platform + per-source
    breakdown for the current user's entire content catalog.

    Time window via `period` (preferred) or legacy `days` integer:
      period=7d   → last 7 days
      period=7h   → last 7 hours
      period=30m  → last 30 minutes
      period=all  → all time (default when omitted)

    Custom absolute window: pass `start` and `end` (half-open [start, end)) — overrides period/days.

    Used by analytics chips / KPI cards to show ALL activity on connected
    accounts — not just UploadM8-originated videos.
    """
    from services.catalog_sync import get_catalog_aggregate
    ws = _parse_iso_ts(start)
    we = _parse_iso_ts(end)
    if ws is not None and we is not None:
        result = await get_catalog_aggregate(
            db_pool,
            str(user["id"]),
            period=None,
            days=None,
            platform=platform,
            source=source,
            account_id=account_id,
            window_start=ws,
            window_end_exclusive=we,
        )
    else:
        result = await get_catalog_aggregate(
            db_pool,
            str(user["id"]),
            period=period,
            days=days,
            platform=platform,
            source=source,
            account_id=account_id,
        )
    return result


@app.get("/api/analytics/upload-counts-by-token")
async def get_upload_counts_by_token(user: dict = Depends(get_current_user)):
    """
    Completed UploadM8 uploads per connected account (platform_tokens.id), when
    target_accounts explicitly lists that token. Used for multi-account CRM views.
    """
    uid = str(user["id"])
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                pt.platform::text AS platform,
                pt.id::text AS token_id,
                COUNT(u.id) FILTER (
                    WHERE u.status IN ('succeeded', 'completed', 'partial')
                      AND pt.platform = ANY(u.platforms)
                      AND pt.id::text = ANY(COALESCE(u.target_accounts, ARRAY[]::text[]))
                )::bigint AS cnt
            FROM platform_tokens pt
            LEFT JOIN uploads u ON u.user_id = pt.user_id
            WHERE pt.user_id = $1::uuid AND pt.revoked_at IS NULL
            GROUP BY pt.platform, pt.id
            """,
            uid,
        )
    by_platform: Dict[str, Dict[str, int]] = {}
    for r in rows:
        plat = (r["platform"] or "").lower()
        by_platform.setdefault(plat, {})[r["token_id"]] = int(r["cnt"] or 0)
    return {"by_platform": by_platform}


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
    query = (
        "SELECT id, email, name, role, subscription_tier, subscription_status, status, created_at, "
        "last_active_at, stripe_subscription_id FROM users WHERE 1=1"
    )
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
        active_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status = 'active'")
        banned_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status = 'banned'")
        # "Paying" = Stripe subscription exists and is active or past_due (excludes trialing = no successful charge yet).
        stripe_paying_users = await conn.fetchval(
            """
            SELECT COUNT(*) FROM users
            WHERE stripe_subscription_id IS NOT NULL
              AND LOWER(COALESCE(subscription_status, '')) IN ('active', 'past_due')
            """
        )
    return {
        "users": [dict(u) for u in users],
        "total": total,
        "summary": {
            "total_users": int(total or 0),
            "active_users": int(active_users or 0),
            "banned_users": int(banned_users or 0),
            "stripe_paying_users": int(stripe_paying_users or 0),
        },
    }

@app.put("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, data: AdminUserUpdate, request: Request, background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    updates, params = [], [user_id]
    changes = {}
    if data.subscription_tier:
        raw_tier = (data.subscription_tier or "").strip().lower()
        # Master admin is a role, not a billable tier; legacy tiers are blocked in admin UI.
        allowed_admin_tiers = {"free", "creator_lite", "creator_pro", "studio", "agency", "friends_family"}
        if raw_tier not in allowed_admin_tiers:
            raise HTTPException(status_code=400, detail=f"Invalid subscription_tier: {data.subscription_tier}")
        normalized_tier = normalize_tier(raw_tier)
        updates.append(f"subscription_tier = ${len(params)+1}")
        params.append(normalized_tier)
        changes["subscription_tier"] = normalized_tier
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
            # Fetch target before updating so we have old tier
            _target = await conn.fetchrow("SELECT email, name, subscription_tier FROM users WHERE id = $1", user_id)
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
            await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_UPDATE_USER",
                                  details={"changes": changes}, request=request,
                                  resource_type="user", resource_id=user_id)

        # Fire tier-change and special welcome emails when subscription_tier changed
        if changes.get("subscription_tier") and _target:
            _te = _target["email"]
            _tn = _target["name"] or "there"
            _old = _target["subscription_tier"] or "free"
            _new = changes["subscription_tier"]
            if _old != _new:
                background_tasks.add_task(
                    send_admin_tier_switch_email,
                    _te, _tn, _old, _new, "", _tier_is_upgrade(_old, _new),
                )
            # Special heartfelt welcome for privileged tiers
            if _new == "friends_family":
                background_tasks.add_task(send_friends_family_welcome_email, _te, _tn)
            elif _new == "agency":
                background_tasks.add_task(send_agency_welcome_email, _te, _tn)
            # master_admin is role-only; no subscription_tier path here.

    return {"status": "updated"}

@app.post("/api/admin/users/{user_id}/ban")
async def admin_ban_user(
    user_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
):
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email, name, status FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET status = 'banned' WHERE id = $1", user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_BAN_USER",
                              details={"target_email": target["email"] if target else None},
                              request=request, resource_type="user", resource_id=user_id,
                              severity="WARNING")
    if target and target.get("email") and (target.get("status") != "banned"):
        background_tasks.add_task(
            send_admin_account_status_email,
            target["email"],
            target.get("name") or "there",
            "banned",
            "Your account was restricted by an UploadM8 administrator.",
        )
    return {"status": "banned"}

@app.post("/api/admin/users/{user_id}/unban")
async def admin_unban_user(
    user_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
):
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow("SELECT email, name, status FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET status = 'active' WHERE id = $1", user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_UNBAN_USER",
                              details={"target_email": target["email"] if target else None},
                              request=request, resource_type="user", resource_id=user_id)
    if target and target.get("email") and (target.get("status") != "active"):
        background_tasks.add_task(
            send_admin_account_status_email,
            target["email"],
            target.get("name") or "there",
            "active",
            "Your account access has been restored.",
        )
    return {"status": "unbanned"}


@app.put("/api/admin/users/{user_id}/email")
async def admin_change_email(
    user_id: uuid.UUID,
    payload: AdminUpdateEmailIn,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
):
    new_email = payload.email.lower().strip()
    user_id_str = str(user_id)

    async with db_pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email)=LOWER($1) AND id <> $2",
            new_email,
            user_id_str,
        )
        if exists:
            raise HTTPException(status_code=409, detail="Email already in use")

        old = await conn.fetchrow("SELECT email, name FROM users WHERE id=$1", user_id_str)
        if not old:
            raise HTTPException(status_code=404, detail="User not found")

        verification_token = secrets.token_urlsafe(32)
        await conn.execute(
            "UPDATE email_changes SET used_at = NOW() WHERE user_id = $1::uuid AND used_at IS NULL",
            user_id_str,
        )

        await conn.execute(
            """
            INSERT INTO email_changes (user_id, old_email, new_email, changed_by_admin_id, verification_token)
            VALUES ($1::uuid, $2, $3, $4::uuid, $5)
            """,
            user_id_str,
            old["email"],
            new_email,
            user["id"],
            verification_token,
        )

        # Keep current login email until verification link is used.
        await conn.execute("UPDATE users SET updated_at=NOW() WHERE id=$1", user_id_str)

        await log_admin_audit(
            conn,
            user_id=user_id_str,
            admin=user,
            action="ADMIN_CHANGE_EMAIL",
            details={"old_email": old["email"], "new_email": new_email},
            request=request,
        )

    # New address: verify link. Old address: security notice (admin-initiated change).
    _verify_link = f"{FRONTEND_URL.rstrip('/')}/verify-email.html?token={verification_token}"
    _target_name = old.get("name") or "there"
    background_tasks.add_task(
        send_email_change_email,
        new_email, old["email"], _target_name, _verify_link,
    )
    background_tasks.add_task(
        send_admin_email_change_notice_to_old_email,
        old["email"], new_email, _target_name,
    )

    return {"ok": True, "email": new_email}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(
    user_id: uuid.UUID,
    payload: AdminResetPasswordIn,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
):
    temp = payload.temp_password
    pw_hash = bcrypt.hashpw(temp.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user_id_str = str(user_id)

    async with db_pool.acquire() as conn:
        try:
            target = await conn.fetchrow("SELECT id, role FROM users WHERE id=$1", user_id_str)
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
                user_id_str,
            )

            # Prefer full admin reset audit row; gracefully degrade for legacy schemas.
            admin_reset_token = secrets.token_urlsafe(32)
            admin_token_hash = _sha256_hex(admin_reset_token)
            try:
                await conn.execute(
                    """
                    INSERT INTO password_resets (user_id, reset_by_admin_id, temp_password_hash, token_hash, force_change, expires_at)
                    VALUES ($1::uuid, $2::uuid, $3, $4, TRUE, NOW() + INTERVAL '7 days')
                    """,
                    user_id_str,
                    user["id"],
                    pw_hash,
                    admin_token_hash,
                )
            except Exception as e:
                logger.warning(f"admin reset_password legacy insert fallback user={user_id_str}: {e}")
                await conn.execute(
                    """
                    INSERT INTO password_resets (user_id, temp_password_hash, token_hash, expires_at)
                    VALUES ($1::uuid, $2, $3, NOW() + INTERVAL '7 days')
                    """,
                    user_id_str,
                    pw_hash,
                    admin_token_hash,
                )

            await log_admin_audit(
                conn,
                user_id=user_id_str,
                admin=user,
                action="ADMIN_RESET_PASSWORD",
                details={"must_reset_password": True},
                request=request,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"ADMIN_RESET_PASSWORD failed user={user_id_str}: {e}")
            raise HTTPException(status_code=500, detail="Failed to reset password")

    # Email the user their temporary password
    async with db_pool.acquire() as _ec:
        _tgt = await _ec.fetchrow("SELECT email, name FROM users WHERE id=$1", user_id_str)
    if _tgt:
        background_tasks.add_task(send_admin_reset_password_email, _tgt["email"], _tgt["name"] or "there", payload.temp_password)

    return {"ok": True}


@app.get("/api/admin/audit")
async def admin_audit(
    user_id: Optional[str] = None,
    event_category: Optional[str] = None,
    action: Optional[str] = None,
    severity: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    source: str = "all",           # "all" | "admin" | "system"
    limit: int = 100,
    offset: int = 0,
    user: dict = Depends(get_current_user)
):
    """
    Corporate-grade audit log endpoint.
    - Returns rolling 6-month window across both admin_audit_log and system_event_log
    - Supports filtering by category, action, severity, user, source table
    - Optional resource_type + resource_id (e.g. resource_type=upload & resource_id=<upload uuid>)
      for publish metadata / per-upload trails in system_event_log
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
                    if resource_type:
                        aargs.append(resource_type.strip())
                        aq += f" AND resource_type = ${len(aargs)}"
                    if resource_id:
                        aargs.append(str(resource_id).strip())
                        aq += f" AND resource_id = ${len(aargs)}"

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
                    if resource_type:
                        sargs.append(resource_type.strip())
                        sq += f" AND resource_type = ${len(sargs)}"
                    if resource_id:
                        sargs.append(str(resource_id).strip())
                        sq += f" AND resource_id = ${len(sargs)}"

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
                except Exception as e:
                    logger.debug("[admin_audit] actor name lookup failed: %s", e)

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


@app.get("/api/admin/audit/publish-events")
async def admin_audit_publish_events(
    upload_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user: dict = Depends(get_current_user),
):
    """
    Publish metadata + outcome audit trail (worker `system_event_log`).

    Rows are written per platform attempt:
    - PUBLISH_METADATA_RESOLVED — effective title/caption/hashtags, privacy, account
    - PUBLISH_ATTEMPT_RESULT — success/failure, error, publish_id, platform_url

    Query examples:
    - By upload:   ?upload_id=<uuid>
    - By user:     ?user_id=<uuid>
    - Both:        narrow to that user's events for one upload
    """
    require_admin(user)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    actions = ("PUBLISH_METADATA_RESOLVED", "PUBLISH_ATTEMPT_RESULT")
    try:
        async with db_pool.acquire() as conn:
            where_sql = """
                created_at >= NOW() - INTERVAL '6 months'
                  AND event_category = 'UPLOAD'
                  AND action = ANY($1::text[])
            """
            args: List[Any] = [list(actions)]
            p = 2
            if upload_id:
                args.append(str(upload_id).strip())
                where_sql += f" AND resource_type = 'upload' AND resource_id = ${p}"
                p += 1
            if user_id:
                args.append(str(user_id).strip())
                where_sql += f" AND user_id = ${p}::uuid"
                p += 1

            total = int(
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM system_event_log WHERE {where_sql}",
                    *args,
                )
                or 0
            )

            lim_p, off_p = p, p + 1
            q = f"""
                SELECT
                    id,
                    user_id,
                    event_category,
                    action,
                    resource_type,
                    resource_id,
                    details,
                    severity,
                    outcome,
                    created_at
                FROM system_event_log
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT ${lim_p} OFFSET ${off_p}
            """
            rows = await conn.fetch(q, *args, limit, offset)
            page = [dict(r) for r in rows]

            def _ser(v):
                if v is None:
                    return None
                if hasattr(v, "isoformat"):
                    return v.isoformat()
                if isinstance(v, uuid.UUID):
                    return str(v)
                return v

            for it in page:
                for k, v in list(it.items()):
                    it[k] = _ser(v)
                d = it.get("details")
                if isinstance(d, str):
                    try:
                        it["details"] = json.loads(d)
                    except Exception as e:
                        logger.debug("[admin_audit_publish_events] details JSON parse skip: %s", e)

        return {
            "items": page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
            "actions": list(actions),
        }
    except Exception as e:
        logger.error(f"[admin_audit_publish_events] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Publish audit query failed: {str(e)}")


@app.get("/api/admin/audit/data-integrity")
async def admin_data_integrity_audit(
    limit: int = 200,
    offset: int = 0,
    severity: Optional[str] = None,
    since_hours: int = 72,
    user: dict = Depends(get_current_user),
):
    """Dedicated admin feed for data-integrity and rollup correction events."""
    require_admin(user)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    since_hours = max(1, min(int(since_hours or 72), 24 * 180))
    sev_filter = (severity or "").upper().strip() or None

    async with db_pool.acquire() as conn:
        q = """
            SELECT
                id, event_category, action, user_id, resource_type, resource_id,
                details, severity, outcome, created_at
            FROM system_event_log
            WHERE created_at >= (NOW() - ($1::int * INTERVAL '1 hour'))
              AND UPPER(COALESCE(event_category, '')) = 'DATA_INTEGRITY'
        """
        args: List[Any] = [since_hours]
        if sev_filter:
            args.append(sev_filter)
            q += f" AND UPPER(COALESCE(severity,'INFO')) = ${len(args)}"
        q += " ORDER BY created_at DESC"
        rows = await conn.fetch(q, *args)
        items = [dict(r) for r in rows]
        total = len(items)
        page = items[offset: offset + limit]
        summary = {
            "total": total,
            "failed": sum(1 for x in items if str(x.get("outcome") or "").upper() == "FAILED"),
            "corrected": sum(1 for x in items if "CORRECTED" in str(x.get("action") or "").upper()),
            "errors": sum(1 for x in items if str(x.get("severity") or "").upper() == "ERROR"),
            "warnings": sum(1 for x in items if str(x.get("severity") or "").upper() == "WARNING"),
        }

    return {
        "summary": summary,
        "items": page,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
    }


@app.get("/api/admin/ml/priors/latest")
async def admin_ml_priors_latest(
    limit: int = 100,
    offset: int = 0,
    since_hours: int = 72,
    user_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """
    Debug feed for latest ML priors actually applied per upload.
    Includes both:
      - thumbnail bias decisions (_ml_thumbnail_selection_bias)
      - M8 strategy priors (from upload_quality_scores_daily, injected into m8_engine_json)
    """
    require_admin(user)
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    since_hours = max(1, min(int(since_hours or 72), 24 * 180))
    uid = (user_id or "").strip() or None

    def _to_dict(val: Any) -> Dict[str, Any]:
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT
                    fe.created_at,
                    fe.user_id,
                    u.email AS user_email,
                    fe.upload_id,
                    fe.category,
                    fe.output_artifacts
                FROM upload_feature_events fe
                LEFT JOIN users u ON u.id = fe.user_id
                WHERE fe.created_at >= (NOW() - ($1::int * INTERVAL '1 hour'))
                  AND ($2::uuid IS NULL OR fe.user_id = $2::uuid)
                ORDER BY fe.created_at DESC
                LIMIT $3 OFFSET $4
                """,
                since_hours,
                uid,
                limit,
                offset,
            )
        except Exception as e:
            if e.__class__.__name__ in ("UndefinedTableError", "UndefinedColumnError"):
                return {
                    "summary": {
                        "total": 0,
                        "thumbnail_bias_present": 0,
                        "m8_strategy_priors_present": 0,
                    },
                    "items": [],
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "note": "ML debug tables/columns not available in this environment yet.",
                }
            raise

    items: List[Dict[str, Any]] = []
    thumb_present = 0
    m8_present = 0
    for r in rows:
        d = dict(r)
        oa = _to_dict(d.get("output_artifacts"))
        thumb_bias = _to_dict(oa.get("_ml_thumbnail_selection_bias"))
        if thumb_bias:
            thumb_present += 1

        m8_json = _to_dict(oa.get("m8_engine_json"))
        strategy_priors = _to_dict(m8_json.get("strategy_priors"))
        if strategy_priors:
            m8_present += 1

        items.append(
            {
                "created_at": d.get("created_at"),
                "user_id": str(d.get("user_id")) if d.get("user_id") else None,
                "user_email": d.get("user_email"),
                "upload_id": str(d.get("upload_id")) if d.get("upload_id") else None,
                "category": d.get("category"),
                "thumbnail_selection_method": oa.get("thumbnail_selection_method"),
                "thumbnail_render_method": oa.get("thumbnail_render_method"),
                "thumbnail_bias": thumb_bias,
                "m8_strategy_priors": strategy_priors,
            }
        )

    return {
        "summary": {
            "total": len(items),
            "thumbnail_bias_present": thumb_present,
            "m8_strategy_priors_present": m8_present,
        },
        "items": items,
        "limit": limit,
        "offset": offset,
        "has_more": len(items) >= limit,
    }


@app.get("/api/admin/analytics/users")
async def admin_analytics_users(user: dict = Depends(get_current_user)):
    require_admin(user)

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        active_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status='active'")
        banned_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status='banned'")
        paid_users = await conn.fetchval(
            """
            SELECT COUNT(*) FROM users
            WHERE stripe_subscription_id IS NOT NULL
              AND LOWER(COALESCE(subscription_status, '')) IN ('active', 'past_due')
            """
        )
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
        creator_lite_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='creator_lite'")
        creator_pro_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='creator_pro'")
        studio_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='studio'")
        agency_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE subscription_tier='agency'")

    return {
        "mrr_estimate": 0.0,
        "creator_lite_count": int(creator_lite_count or 0),
        "creator_pro_count": int(creator_pro_count or 0),
        "studio_count": int(studio_count or 0),
        "agency_count": int(agency_count or 0),
    }


@app.post("/api/admin/users/assign-tier")
async def admin_assign_tier(user_id: str = Query(...), tier: str = Query(...), request: Request = None, background_tasks: BackgroundTasks = None, user: dict = Depends(require_master_admin)):
    raw = (tier or "").strip().lower()
    if raw not in TIER_CONFIG:
        raise HTTPException(400, "Invalid tier")
    tier_norm = normalize_tier(raw)
    async with db_pool.acquire() as conn:
        old = await conn.fetchrow("SELECT subscription_tier, email, name FROM users WHERE id = $1", user_id)
        await conn.execute("UPDATE users SET subscription_tier = $1 WHERE id = $2", tier_norm, user_id)
        await log_admin_audit(conn, user_id=user_id, admin=user, action="ADMIN_ASSIGN_TIER",
                              details={"old_tier": old["subscription_tier"] if old else None, "new_tier": tier_norm,
                                       "target_email": old["email"] if old else None},
                              request=request, resource_type="user", resource_id=user_id,
                              severity="WARNING")

    if old and background_tasks:
        _te  = old["email"]
        _tn  = old["name"] or "there"
        _old = old["subscription_tier"] or "free"
        if _old != tier_norm:
            background_tasks.add_task(
                send_admin_tier_switch_email,
                _te, _tn, _old, tier_norm, "", _tier_is_upgrade(_old, tier_norm),
            )
        if tier_norm == "friends_family":
            background_tasks.add_task(send_friends_family_welcome_email, _te, _tn)
        elif tier_norm == "agency":
            background_tasks.add_task(send_agency_welcome_email, _te, _tn)
        elif tier_norm == "master_admin":
            background_tasks.add_task(send_master_admin_welcome_email, _te, _tn)

    return {"status": "assigned", "tier": tier_norm}

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
                await send_announcement_email(dest, title, body)
                ok = True
                email_sent += 1
            except Exception as e:
                ok = False
                err = str(e)

        elif ch == "discord_community":
            ok, err = await _discord_post_raw(dest, embeds=[{"title": f" {title}", "description": body, "color": 0xf97316}])
            if ok:
                discord_sent += 1

        elif ch == "user_webhook":
            ok, err = await _discord_post_raw(dest, embeds=[{"title": f" {title}", "description": body, "color": 0xf97316}])
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
              COALESCE(
                NULLIF(TRIM(us.discord_webhook), ''),
                NULLIF(TRIM(up.discord_webhook), ''),
                NULLIF(TRIM(COALESCE(u.preferences->>'discordWebhook', u.preferences->>'discord_webhook')), ''),
                ''
              ) AS discord_webhook
            FROM users u
            LEFT JOIN user_settings us ON us.user_id = u.id
            LEFT JOIN user_preferences up ON up.user_id = u.id
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


@app.get("/api/admin/integrations/thumbnail-provider")
async def admin_thumbnail_provider_integration(user: dict = Depends(require_admin)):
    """
    Ops visibility: thumbnail generation wiring (legacy single-call vs public v2 per-feature API).
    """
    _ = user
    from services.pikzels_v2 import PUBLIC_BASE, feature_map_for_docs, public_api_key_source, resolve_public_api_key
    from stages.pikzels_api import THUMB_RENDER_API_URL, studio_renderer_enabled

    key = resolve_public_api_key()
    legacy_on = studio_renderer_enabled()
    host = None
    try:
        if legacy_on and THUMB_RENDER_API_URL:
            host = urlparse(str(THUMB_RENDER_API_URL)).netloc or None
    except Exception:
        host = None

    return {
        "v2_public_api_base": PUBLIC_BASE,
        "v2_api_key_configured": bool(key),
        "v2_api_key_source": public_api_key_source(),
        "legacy_one_shot_renderer_enabled": legacy_on,
        "legacy_renderer_url_host": host,
        "upload_pipeline_mode": "legacy_single_post_with_frame_brief" if legacy_on else "internal_only",
        "v2_features_individual_routes": feature_map_for_docs(),
        "note": (
            "Legacy worker path (when enabled): one POST with frame + brief (stages/pikzels_api). "
            "Per-feature public v2 calls are proxied at /api/thumbnail-studio/pikzels-v2/* and /api/admin/pikzels-v2/* "
            "(see ledger meta pikzels_v2_op for usage analytics)."
        ),
        "documentation": {
            "marketing_site": "https://pikzels.com/",
            "api_index": "https://docs.pikzels.com/llms.txt",
            "openapi": "https://docs.pikzels.com/openapi.json",
        },
    }


@app.get("/api/admin/kpi/pikzels-v2-usage")
async def admin_kpi_pikzels_v2_usage(range: str = "30d", user: dict = Depends(require_admin)):
    """Aggregate Pikzels v2 proxy usage from token_ledger (meta.pikzels_v2_op)."""
    _ = user
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT COALESCE(meta->>'pikzels_v2_op', 'unknown') AS op,
                   COUNT(*)::int AS n
            FROM token_ledger
            WHERE created_at >= $1
              AND reason = 'thumbnail_studio'
              AND meta->>'pikzels_v2_op' IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC
            """,
            since,
        )
        total = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM token_ledger
            WHERE created_at >= $1
              AND reason = 'thumbnail_studio'
              AND meta->>'pikzels_v2_op' IS NOT NULL
            """,
            since,
        )
    return {
        "range": range,
        "since": since.isoformat(),
        "total_calls": int(total or 0),
        "by_operation": [{"op": str(r["op"]), "count": int(r["n"])} for r in rows or []],
    }


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
        
        upload_stats = await conn.fetchrow(
            f"""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
                   COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE created_at >= $1
            """,
            since,
        )
        
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
    paid_tiers = ["creator_lite", "creator_pro", "studio", "agency"]

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
            SELECT
                COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
                COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
                COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute,
                COALESCE(SUM(CASE WHEN category IN ('stripe_fees','mailgun','bandwidth','postgres','redis') THEN cost_usd ELSE 0 END), 0)::decimal AS other
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
    
    total_cost = float(costs["openai"] or 0) + float(costs["storage"] or 0) + float(costs["compute"] or 0) + float(costs.get("other") or 0)
    gross_margin = float(revenue or 0) - total_cost
    
    return {
        "costs": {"openai": float(costs["openai"] or 0), "storage": float(costs["storage"] or 0), "compute": float(costs["compute"] or 0), "other": float(costs.get("other") or 0), "total": total_cost},
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

        free_active_uploaders_7d = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT u.id)::int
            FROM users u
            INNER JOIN uploads up ON up.user_id = u.id AND up.created_at >= NOW() - INTERVAL '7 days'
            WHERE u.subscription_tier = 'free' AND u.status = 'active'
            """
        ) or 0

        low_put_available = await conn.fetchval(
            """
            SELECT COUNT(*)::int
            FROM users u
            INNER JOIN wallets w ON w.user_id = u.id
            WHERE u.status = 'active'
              AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
              AND (w.put_balance - w.put_reserved) BETWEEN 0 AND 29
            """
        ) or 0

        aic_starved = await conn.fetchval(
            """
            SELECT COUNT(*)::int
            FROM users u
            INNER JOIN wallets w ON w.user_id = u.id
            WHERE u.status = 'active'
              AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
              AND (w.aic_balance - w.aic_reserved) BETWEEN 0 AND 9
            """
        ) or 0

        multi_platform_users = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM (
              SELECT user_id FROM platform_tokens GROUP BY user_id HAVING COUNT(*) >= 3
            ) s
            """
        ) or 0
    
    return {
        "put_spent": token_stats["put_spent"] if token_stats else 0,
        "aic_spent": token_stats["aic_spent"] if token_stats else 0,
        "tokens_purchased": token_stats["tokens_purchased"] if token_stats else 0,
        "users_hitting_quota": hitting_quota,
        "total_active_users": total_active,
        "quota_hit_pct": (hitting_quota / max(total_active, 1)) * 100,
        "sales_opportunity_levers": {
            "free_users_uploading_last_7d": int(free_active_uploaders_7d),
            "users_low_put_available_0_29": int(low_put_available),
            "users_low_aic_available_0_9": int(aic_starved),
            "users_3plus_platform_connections": int(multi_platform_users),
        },
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


@app.get("/api/admin/calculator/pricing")
async def get_admin_calculator_pricing(user: dict = Depends(require_master_admin)):
    """
    Live pricing and entitlements for the admin Business Calculator.
    Single source of truth from stages/entitlements.py.
    Call on page load to prefill inputs with current tiers (including Friends & Family).
    """
    # Revenue tiers (public pricing page)
    revenue_tiers = {}
    for slug in ("free", "creator_lite", "creator_pro", "studio", "agency", "enterprise"):
        cfg = TIER_CONFIG.get(slug, {})
        revenue_tiers[slug] = {
            "name": cfg.get("name", slug.replace("_", " ").title()),
            "price": float(cfg.get("price", 0)),
            "put_monthly": cfg.get("put_monthly", 0),
            "aic_monthly": cfg.get("aic_monthly", 0),
            "queue_depth": cfg.get("queue_depth", 0),
            "lookahead_hours": cfg.get("lookahead_hours", 0),
            "max_accounts_per_platform": cfg.get("max_accounts_per_platform", 0),
        }

    # Internal tiers (Friends & Family, Lifetime, Master Admin) — $0 revenue, full infra cost
    internal_tiers = {}
    for slug in ("friends_family", "lifetime", "master_admin"):
        cfg = TIER_CONFIG.get(slug, {})
        internal_tiers[slug] = {
            "name": cfg.get("name", slug.replace("_", " ").title()),
            "price": 0,
            "put_monthly": cfg.get("put_monthly", 0),
            "aic_monthly": cfg.get("aic_monthly", 0),
        }

    # Top-up packs (amounts from entitlements; prices come from Stripe)
    topup_packs = []
    for lookup_key, meta in TOPUP_PRODUCTS.items():
        wallet = meta.get("wallet", "")
        amount = meta.get("amount", 0)
        _price = meta.get("price_usd") or meta.get("price")
        topup_packs.append({
            "lookup_key": lookup_key,
            "wallet": wallet,
            "amount": amount,
            "price_usd": _price,   # canonical field (new frontend reads this)
            "price": _price,       # legacy alias  (older clients / backcompat)
            "label": f"{wallet.upper()} {amount} Pack",
        })

    return {
        "revenue_tiers": revenue_tiers,
        "internal_tiers": internal_tiers,
        "topup_packs": topup_packs,
    }


@app.put("/api/admin/settings")
async def update_admin_settings(settings: dict, user: dict = Depends(require_master_admin)):
    global admin_settings_cache
    admin_settings_cache.update(settings)
    admin_settings_cache.setdefault("promo_burst_week_enabled", False)
    admin_settings_cache.setdefault("promo_referral_enabled", False)
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1", json.dumps(admin_settings_cache))
    return {"status": "updated", "settings": admin_settings_cache}


@app.patch("/api/admin/settings/promo-toggles")
async def update_promo_toggles(
    data: PromoTogglesBody,
    user: dict = Depends(require_master_admin),
):
    """Toggle Burst Week / Referral promo flags without sending full admin settings JSON."""
    global admin_settings_cache
    if data.promo_burst_week_enabled is None and data.promo_referral_enabled is None:
        raise HTTPException(
            400,
            "Provide at least one of: promo_burst_week_enabled, promo_referral_enabled",
        )
    if data.promo_burst_week_enabled is not None:
        admin_settings_cache["promo_burst_week_enabled"] = bool(data.promo_burst_week_enabled)
    if data.promo_referral_enabled is not None:
        admin_settings_cache["promo_referral_enabled"] = bool(data.promo_referral_enabled)
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE admin_settings SET settings_json = $1, updated_at = NOW() WHERE id = 1",
            json.dumps(admin_settings_cache),
        )
    return {
        "status": "updated",
        "promo_burst_week_enabled": admin_settings_cache.get("promo_burst_week_enabled", False),
        "promo_referral_enabled": admin_settings_cache.get("promo_referral_enabled", False),
    }


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


class AdminEmailJobRunRequest(BaseModel):
    job: Literal["trial_reminders", "monthly_user_digest", "weekly_admin_digest", "scheduled_publish_alerts", "all"]


@app.post("/api/admin/email-jobs/run")
async def admin_run_email_jobs(data: AdminEmailJobRunRequest, user: dict = Depends(require_master_admin)):
    """Force-run email cron jobs for live smoke testing."""
    ran = []
    if data.job in ("trial_reminders", "all"):
        await _run_trial_ending_reminders_once()
        ran.append("trial_reminders")
    if data.job in ("monthly_user_digest", "all"):
        await _run_monthly_user_kpi_digests_once()
        ran.append("monthly_user_digest")
    if data.job in ("weekly_admin_digest", "all"):
        await _run_admin_weekly_kpi_digest_once()
        ran.append("weekly_admin_digest")
    if data.job in ("scheduled_publish_alerts", "all"):
        await _run_scheduled_publish_alerts_once()
        ran.append("scheduled_publish_alerts")
    return {"status": "ok", "ran": ran}

# ============================================================
# Dashboard Stats
# ============================================================
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(user: dict = Depends(get_current_user_readonly)):
    """Dashboard stats for user: uploads, quota, success rate, accounts, scheduled."""
    ent = get_entitlements_from_user(dict(user))
    plan = entitlements_to_dict(ent)
    wallet = user.get("wallet", {})
    
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow(
            f"""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status IN ('pending','queued','processing','staged','scheduled','ready_to_publish') THEN 1 ELSE 0 END)::int AS in_queue,
                   COALESCE(SUM(views), 0)::bigint AS views,
                   COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE user_id = $1
            """,
            user["id"],
        )
        # Scheduled: pending/staged/ready_to_publish with schedule_mode in scheduled/smart
        scheduled = await conn.fetchval("""
            SELECT COUNT(*)::int FROM uploads
            WHERE user_id = $1
              AND status IN ('pending','staged','queued','scheduled','ready_to_publish')
              AND schedule_mode IN ('scheduled','smart')
              AND scheduled_time IS NOT NULL
        """, user["id"])
        # Monthly PUT used (current month)
        try:
            put_used_month = await conn.fetchval("""
                SELECT COALESCE(SUM(put_spent), 0)::int FROM uploads
                WHERE user_id = $1 AND created_at >= date_trunc('month', CURRENT_DATE)
            """, user["id"])
        except Exception:
            put_used_month = 0
        # Monthly upload count (true upload quota semantics).
        try:
            uploads_used_month = await conn.fetchval(
                """
                SELECT COUNT(*)::int
                  FROM uploads
                 WHERE user_id = $1
                   AND created_at >= date_trunc('month', CURRENT_DATE)
                """,
                user["id"],
            )
        except Exception:
            uploads_used_month = 0
        # Connected accounts (exclude revoked)
        try:
            accounts = await conn.fetchval(
                "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND (revoked_at IS NULL OR revoked_at > NOW())",
                user["id"],
            )
        except Exception as e:
            logger.debug("dashboard: platform_tokens count with revoked clause failed, fallback: %s", e)
            accounts = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])
        recent = await conn.fetch(
            "SELECT id, filename, platforms, status, created_at FROM uploads WHERE user_id = $1 ORDER BY created_at DESC LIMIT 5",
            user["id"],
        )

        dash_live = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}
        if int(accounts or 0) > 0:
            try:
                drow = await conn.fetchrow(
                    "SELECT data FROM platform_metrics_cache WHERE user_id = $1",
                    user["id"],
                )
                if drow and drow["data"] is not None:
                    pdata = drow["data"]
                    if isinstance(pdata, str):
                        pdata = json.loads(pdata)
                    if isinstance(pdata, dict):
                        dash_live = _aggregate_platform_metrics_live(pdata.get("platforms") or {})
            except Exception as e:
                logger.debug("dashboard: platform_metrics_cache read failed: %s", e)
        upload_engagement = await _compute_upload_engagement_totals(conn, str(user["id"]))

        from services.canonical_engagement import (
            ROLLUP_VERSION,
            compute_canonical_engagement_rollup,
            engagement_window_api_dict,
        )

        try:
            cr = await compute_canonical_engagement_rollup(
                conn,
                str(user["id"]),
                window_start=None,
                window_end_exclusive=None,
                platform=None,
            )
        except Exception as e:
            logger.warning("dashboard: canonical engagement rollup failed: %s", e)
            cr = {
                "views": int(upload_engagement.get("views") or 0),
                "likes": int(upload_engagement.get("likes") or 0),
                "comments": int(upload_engagement.get("comments") or 0),
                "shares": int(upload_engagement.get("shares") or 0),
                "breakdown": {
                    "compute": {
                        "rollup_version": ROLLUP_VERSION,
                        "complete": False,
                        "warnings": ["rollup_exception"],
                        "error_detail": str(e)[:500],
                    },
                },
                "catalog_tracked_videos": 0,
                "rollup_version": ROLLUP_VERSION,
                "rollup_rule": "fallback_upload_table_only",
                "kpi_sources": {"error": str(e), "rollup_version": ROLLUP_VERSION},
            }
        dash_win = engagement_window_api_dict(start=None, end_exclusive=None)
        canon = dict(cr)
        canon["engagement_window_utc"] = dash_win

    total = stats["total"] if stats else 0
    completed = stats["completed"] if stats else 0
    put_avail = wallet.get("put_balance", 0) - wallet.get("put_reserved", 0)
    aic_avail = wallet.get("aic_balance", 0) - wallet.get("aic_reserved", 0)
    put_monthly = int(plan.get("put_monthly", 60) or 0)
    uploads_limit = int(
        plan.get("monthly_uploads")
        or plan.get("max_uploads_monthly")
        or plan.get("put_monthly")
        or 0
    )
    role = str(user.get("role") or "").lower()
    tier = str(user.get("subscription_tier") or "").lower()
    unlimited_uploads = bool(
        role == "master_admin"
        or tier in ("master_admin", "friends_family", "lifetime")
        or int(uploads_limit or 0) >= 999999
        or int(put_monthly or 0) >= 999999
    )
    success_rate = (completed / max(total, 1)) * 100 if total else 0
    
    # Credits display: PUT/AIC wallet balances (not monthly quota)
    put_reserved = float(wallet.get("put_reserved", 0) or 0)
    aic_reserved = float(wallet.get("aic_reserved", 0) or 0)
    put_total = float(wallet.get("put_balance", 0) or 0)
    aic_total = float(wallet.get("aic_balance", 0) or 0)

    db_views = int(upload_engagement.get("views") or 0)
    db_likes = int(upload_engagement.get("likes") or 0)
    live_v = int(dash_live.get("views") or 0)
    live_l = int(dash_live.get("likes") or 0)

    return {
        "uploads": {"total": total, "completed": completed, "in_queue": stats["in_queue"] if stats else 0},
        "engagement": {
            "views": int(canon["views"] or 0),
            "likes": int(canon["likes"] or 0),
            "comments": int(canon["comments"] or 0),
            "shares": int(canon["shares"] or 0),
            "breakdown": canon["breakdown"],
            "rollup_rule": canon["rollup_rule"],
            "rollup_version": canon.get("rollup_version"),
            "catalog_tracked_videos": int(canon["catalog_tracked_videos"] or 0),
            "engagement_window_utc": canon.get("engagement_window_utc"),
            "views_db": db_views,
            "likes_db": db_likes,
            "live_views": live_v,
            "live_likes": live_l,
            "live_platforms": dash_live.get("platforms_included") or [],
            "kpi_sources": canon.get("kpi_sources"),
        },
        "success_rate": round(success_rate, 1),
        "scheduled": scheduled or 0,
        "quota": {
            "put_used": put_used_month or 0,
            "put_limit": put_monthly,
            "uploads_used": int(uploads_used_month or 0),
            "uploads_limit": (-1 if unlimited_uploads else int(uploads_limit or 0)),
            "uploads_unlimited": unlimited_uploads,
        },
        "wallet": {"put_available": put_avail, "put_total": put_total, "aic_available": aic_avail, "aic_total": aic_total},
        "credits": {
            "put": {"available": put_avail, "reserved": put_reserved, "total": put_total, "monthly_allowance": put_monthly},
            "aic": {"available": aic_avail, "reserved": aic_reserved, "total": aic_total, "monthly_allowance": plan.get("aic_monthly", 0)},
        },
        "accounts": {"connected": accounts or 0, "limit": int(plan.get("max_accounts", 1) or 1)},
        "recent": [{"id": str(r["id"]), "filename": r["filename"], "platforms": r["platforms"], "status": r["status"]} for r in recent],
        "plan": plan,
    }


@app.get("/api/dashboard")
async def get_dashboard_alias(user: dict = Depends(get_current_user_readonly)):
    """Alias for GET /api/dashboard/stats — some frontends call /api/dashboard."""
    return await get_dashboard_stats(user)


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


def _estimate_whisper_stt_cost_usd(successful_uploads: int) -> tuple[float, dict]:
    """
    Rough Whisper spend for the selected period: successful uploads × assumed minutes × $/min.
    Tunable via WHISPER_USD_PER_MINUTE, WHISPER_ESTIMATE_AVG_SECONDS, WHISPER_ESTIMATE_AVG_CAP_MINUTES.
    """
    try:
        usd_per_min = float(os.environ.get("WHISPER_USD_PER_MINUTE", "0.006"))
    except ValueError:
        usd_per_min = 0.006
    try:
        avg_sec = float(os.environ.get("WHISPER_ESTIMATE_AVG_SECONDS", "60"))
    except ValueError:
        avg_sec = 60.0
    cap_raw = os.environ.get("WHISPER_ESTIMATE_AVG_CAP_MINUTES", "").strip()
    cap_min: float | None = None
    if cap_raw:
        try:
            cap_min = float(cap_raw)
        except ValueError:
            cap_min = None
    eff_min = max(avg_sec / 60.0, 0.0)
    if cap_min is not None and cap_min >= 0:
        eff_min = min(eff_min, cap_min)
    total = float(max(successful_uploads, 0)) * eff_min * usd_per_min
    meta = {
        "usd_per_minute": usd_per_min,
        "avg_seconds_assumed": avg_sec,
        "effective_minutes_per_upload": eff_min,
    }
    return round(total, 4), meta


@app.get("/api/admin/kpis")
async def get_admin_kpis(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Combined KPI endpoint that returns all metrics in one call"""
    from services.canonical_engagement import ROLLUP_VERSION

    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
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
        
        # Costs (openai, storage, compute, stripe_fees, mailgun, bandwidth, postgres, redis)
        costs = await conn.fetchrow("""
            SELECT
                COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
                COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
                COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute,
                COALESCE(SUM(CASE WHEN category = 'stripe_fees' THEN cost_usd ELSE 0 END), 0)::decimal AS stripe_fees,
                COALESCE(SUM(CASE WHEN category = 'mailgun' THEN cost_usd ELSE 0 END), 0)::decimal AS mailgun,
                COALESCE(SUM(CASE WHEN category = 'bandwidth' THEN cost_usd ELSE 0 END), 0)::decimal AS bandwidth,
                COALESCE(SUM(CASE WHEN category = 'postgres' THEN cost_usd ELSE 0 END), 0)::decimal AS postgres,
                COALESCE(SUM(CASE WHEN category = 'redis' THEN cost_usd ELSE 0 END), 0)::decimal AS redis,
                COALESCE(SUM(CASE WHEN category = 'tool_estimate' THEN cost_usd ELSE 0 END), 0)::decimal AS tool_estimate
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        openai_cost = float(costs["openai"] or 0) if costs else 0
        storage_cost = float(costs["storage"] or 0) if costs else 0
        compute_cost = float(costs["compute"] or 0) if costs else 0
        stripe_fees = float(costs.get("stripe_fees") or 0) if costs else 0
        mailgun_cost = float(costs.get("mailgun") or 0) if costs else 0
        bandwidth_cost = float(costs.get("bandwidth") or 0) if costs else 0
        postgres_cost = float(costs.get("postgres") or 0) if costs else 0
        redis_cost = float(costs.get("redis") or 0) if costs else 0
        tool_estimate_cost = float(costs.get("tool_estimate") or 0) if costs else 0
        total_costs = openai_cost + storage_cost + compute_cost + stripe_fees + mailgun_cost + bandwidth_cost + postgres_cost + redis_cost + tool_estimate_cost
        
        gross_margin = ((total_mrr - total_costs) / max(total_mrr, 1)) * 100 if total_mrr > 0 else 0
        
        # Uploads
        upload_stats = await conn.fetchrow(
            f"""
            SELECT COUNT(*)::int AS total,
                   SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed,
                   COALESCE(SUM(views), 0)::bigint AS views, COALESCE(SUM(likes), 0)::bigint AS likes
            FROM uploads WHERE created_at >= $1
            """,
            since,
        )
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

        kpi_sales_free_active_7d = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT u.id)::int
            FROM users u
            INNER JOIN uploads up ON up.user_id = u.id AND up.created_at >= NOW() - INTERVAL '7 days'
            WHERE u.subscription_tier = 'free' AND u.status = 'active'
            """
        ) or 0
        kpi_sales_low_put = await conn.fetchval(
            """
            SELECT COUNT(*)::int
            FROM users u
            INNER JOIN wallets w ON w.user_id = u.id
            WHERE u.status = 'active'
              AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
              AND (w.put_balance - w.put_reserved) BETWEEN 0 AND 29
            """
        ) or 0
        kpi_sales_low_aic = await conn.fetchval(
            """
            SELECT COUNT(*)::int
            FROM users u
            INNER JOIN wallets w ON w.user_id = u.id
            WHERE u.status = 'active'
              AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
              AND (w.aic_balance - w.aic_reserved) BETWEEN 0 AND 9
            """
        ) or 0
        kpi_sales_multipf = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM (
              SELECT user_id FROM platform_tokens GROUP BY user_id HAVING COUNT(*) >= 3
            ) s
            """
        ) or 0

        # Creative freshness (thumbnail style entropy)
        thumb_entropy_avg = 0.0
        thumb_low_entropy_entities = 0
        thumb_entities_measured = 0
        try:
            trows = await conn.fetch(
                """
                WITH ranked AS (
                    SELECT
                        user_id,
                        platform,
                        COALESCE(NULLIF(style_pack, ''), 'unknown') AS style_pack,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id, platform
                            ORDER BY created_at DESC
                        ) AS rn
                    FROM upload_thumbnail_style_memory
                    WHERE created_at >= $1
                ),
                limited AS (
                    SELECT * FROM ranked WHERE rn <= 30
                ),
                counts AS (
                    SELECT user_id, platform, style_pack, COUNT(*)::int AS c
                    FROM limited
                    GROUP BY user_id, platform, style_pack
                ),
                totals AS (
                    SELECT user_id, platform, COUNT(*)::int AS n, COUNT(DISTINCT style_pack)::int AS distinct_packs
                    FROM limited
                    GROUP BY user_id, platform
                ),
                entropy AS (
                    SELECT
                        c.user_id,
                        c.platform,
                        SUM(
                            - (c.c::double precision / NULLIF(t.n::double precision, 0))
                              * LN(c.c::double precision / NULLIF(t.n::double precision, 0))
                        ) AS h_nat,
                        t.n,
                        t.distinct_packs
                    FROM counts c
                    JOIN totals t
                      ON t.user_id = c.user_id AND t.platform = c.platform
                    GROUP BY c.user_id, c.platform, t.n, t.distinct_packs
                )
                SELECT
                    n,
                    CASE
                        WHEN distinct_packs <= 1 THEN 0.0
                        ELSE h_nat / NULLIF(LN(distinct_packs::double precision), 0)
                    END AS entropy_norm
                FROM entropy
                """
                ,
                since,
            )
            entropies = [float(r.get("entropy_norm") or 0.0) for r in (trows or []) if int(r.get("n") or 0) >= 5]
            thumb_entities_measured = len(entropies)
            if entropies:
                thumb_entropy_avg = float(sum(entropies) / max(len(entropies), 1))
                thumb_low_entropy_entities = sum(1 for v in entropies if v < 0.55)
        except asyncpg.exceptions.UndefinedTableError:
            pass
        except Exception as _thumb_err:
            logger.debug(f"thumbnail entropy KPI skipped: {_thumb_err}")

    whisper_est_usd, whisper_meta = _estimate_whisper_stt_cost_usd(successful_uploads)

    return {
        "total_mrr": total_mrr, "mrr_change": 0, "mrr_by_tier": mrr_by_tier,
        "mrr_creator_lite": mrr_by_tier.get("creator_lite", 0), "mrr_creator_pro": mrr_by_tier.get("creator_pro", 0),
        "mrr_studio": mrr_by_tier.get("studio", 0), "mrr_agency": mrr_by_tier.get("agency", 0),
        "creator_lite_users": tier_breakdown.get("creator_lite", 0), "creator_pro_users": tier_breakdown.get("creator_pro", 0),
        "studio_users": tier_breakdown.get("studio", 0), "agency_users": tier_breakdown.get("agency", 0),
        "topup_revenue": float(revenue["topups"]) if revenue else 0, "topup_count": 0,
        "arpu": round(total_mrr / max(total_users, 1), 2), "arpa": round(total_mrr / max(paid_users, 1), 2),
        "refunds": 0, "refund_count": 0,
        "openai_cost": openai_cost, "storage_cost": storage_cost, "compute_cost": compute_cost,
        "stripe_fees": stripe_fees, "mailgun_cost": mailgun_cost, "bandwidth_cost": bandwidth_cost,
        "postgres_cost": postgres_cost, "redis_cost": redis_cost, "tool_estimate_cost": tool_estimate_cost,
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
        "whisper_cost_estimate_usd": whisper_est_usd,
        "whisper_estimate": whisper_meta,
        "creative_freshness_score": round(thumb_entropy_avg * 100.0, 1),
        "thumbnail_entropy_norm": round(thumb_entropy_avg, 4),
        "thumbnail_low_entropy_entities": int(thumb_low_entropy_entities),
        "thumbnail_entities_measured": int(thumb_entities_measured),
        "sales_opportunity_levers": {
            "free_users_uploading_last_7d": int(kpi_sales_free_active_7d),
            "users_low_put_available_0_29": int(kpi_sales_low_put),
            "users_low_aic_available_0_9": int(kpi_sales_low_aic),
            "users_3plus_platform_connections": int(kpi_sales_multipf),
        },
        "engagement_crosswalk": metric_defs.engagement_crosswalk(),
        "metric_definitions": metric_defs.for_admin_kpis(),
        "data_provenance": metric_defs.admin_kpi_data_provenance(rollup_version=ROLLUP_VERSION),
    }


@app.get("/api/admin/kpi/revenue")
async def get_kpi_revenue(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
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
async def get_kpi_costs(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        costs = await conn.fetchrow("""
            SELECT
                COALESCE(SUM(CASE WHEN category = 'openai' THEN cost_usd ELSE 0 END), 0)::decimal AS openai,
                COALESCE(SUM(CASE WHEN category = 'storage' THEN cost_usd ELSE 0 END), 0)::decimal AS storage,
                COALESCE(SUM(CASE WHEN category = 'compute' THEN cost_usd ELSE 0 END), 0)::decimal AS compute,
                COALESCE(SUM(CASE WHEN category = 'stripe_fees' THEN cost_usd ELSE 0 END), 0)::decimal AS stripe_fees,
                COALESCE(SUM(CASE WHEN category = 'mailgun' THEN cost_usd ELSE 0 END), 0)::decimal AS mailgun,
                COALESCE(SUM(CASE WHEN category = 'bandwidth' THEN cost_usd ELSE 0 END), 0)::decimal AS bandwidth,
                COALESCE(SUM(CASE WHEN category = 'postgres' THEN cost_usd ELSE 0 END), 0)::decimal AS postgres,
                COALESCE(SUM(CASE WHEN category = 'redis' THEN cost_usd ELSE 0 END), 0)::decimal AS redis,
                COALESCE(SUM(CASE WHEN category = 'tool_estimate' THEN cost_usd ELSE 0 END), 0)::decimal AS tool_estimate
            FROM cost_tracking WHERE created_at >= $1
        """, since)
        uploads = await conn.fetchval(
            f"SELECT COUNT(*)::int FROM uploads WHERE status IN {SUCCESSFUL_STATUS_SQL_IN} AND created_at >= $1",
            since,
        )
    if not costs:
        return {"openai_cost": 0, "storage_cost": 0, "compute_cost": 0, "stripe_fees": 0, "mailgun_cost": 0, "bandwidth_cost": 0, "postgres_cost": 0, "redis_cost": 0, "tool_estimate_cost": 0, "total_costs": 0, "costs_change": 0, "cost_per_upload": 0, "successful_uploads": uploads or 0, "total_cogs": 0}
    o = float(costs["openai"] or 0)
    s = float(costs["storage"] or 0)
    c = float(costs["compute"] or 0)
    sf = float(costs.get("stripe_fees") or 0)
    mg = float(costs.get("mailgun") or 0)
    bw = float(costs.get("bandwidth") or 0)
    pg = float(costs.get("postgres") or 0)
    rd = float(costs.get("redis") or 0)
    te = float(costs.get("tool_estimate") or 0)
    total = o + s + c + sf + mg + bw + pg + rd + te
    return {"openai_cost": o, "storage_cost": s, "compute_cost": c, "stripe_fees": sf, "mailgun_cost": mg, "bandwidth_cost": bw, "postgres_cost": pg, "redis_cost": rd, "tool_estimate_cost": te, "total_costs": total, "costs_change": 0, "cost_per_upload": round(total / max(uploads or 1, 1), 4), "successful_uploads": uploads or 0, "total_cogs": total}


@app.get("/api/admin/kpi/cost-tracker")
async def get_admin_cost_tracker(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """
    Per-upload tool cost tracker used by the admin costs menu.
    Combines static per-upload assumptions with observed successful upload volume.
    """
    from stages.kpi_collector import estimate_tool_costs

    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        successful_uploads = await conn.fetchval(
            f"""
            SELECT COUNT(*)::int
            FROM uploads
            WHERE status IN {SUCCESSFUL_STATUS_SQL_IN}
              AND created_at >= $1
            """,
            since,
        )

    tracker = estimate_tool_costs(int(successful_uploads or 0))
    return {
        "range": _range_label(range),
        "successful_uploads": int(successful_uploads or 0),
        "estimated_total_per_upload_usd": tracker["total_usd_per_upload"],
        "estimated_total_window_usd": tracker["total_window_cost_usd"],
        "tools": tracker["tools"],
    }


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
async def get_kpi_reliability(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow(
            f"SELECT COUNT(*)::int AS total, SUM(CASE WHEN status IN {SUCCESSFUL_STATUS_SQL_IN} THEN 1 ELSE 0 END)::int AS completed FROM uploads WHERE created_at >= $1",
            since,
        )
        queue = await conn.fetchval("SELECT COUNT(*) FROM uploads WHERE status IN ('pending', 'queued', 'processing')")
    total, completed = (stats["total"] or 0, stats["completed"] or 0) if stats else (0, 0)
    sr = (completed / max(total, 1)) * 100
    return {"success_rate": round(sr, 1), "reliability_change": 0, "failRates": {"ingest": 0.5, "processing": 1, "upload": round(100-sr, 1), "publish": 0.5, "average": round(100-sr, 1)},
            "retries": {"rate": 5, "one": 3, "two": 1.5, "threePlus": 0.5}, "processingTime": {"ingest": 2, "transcode": 15, "upload": 8, "average": 25},
            "cancels": {"rate": 2, "beforeProcessing": 1.5, "duringProcessing": 0.5, "total30d": 0}, "queue_depth": queue or 0}


@app.get("/api/admin/kpi/thumbnail-diversity")
async def get_kpi_thumbnail_diversity(
    range: str = Query("30d"),
    per_platform_limit: int = Query(30, ge=10, le=120),
    user: dict = Depends(require_admin),
):
    """Style freshness KPI: normalized entropy of thumbnail style packs."""
    _ = user
    minutes = {
        "24h": 1440,
        "7d": 10080,
        "30d": 43200,
        "90d": 129600,
        "6m": 259200,
        "365d": 525600,
        "1y": 525600,
    }.get(range, 43200)
    since = _now_utc() - timedelta(minutes=minutes)

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH ranked AS (
                    SELECT
                        user_id,
                        platform,
                        COALESCE(NULLIF(style_pack, ''), 'unknown') AS style_pack,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id, platform
                            ORDER BY created_at DESC
                        ) AS rn
                    FROM upload_thumbnail_style_memory
                    WHERE created_at >= $1
                ),
                limited AS (
                    SELECT * FROM ranked WHERE rn <= $2
                ),
                counts AS (
                    SELECT user_id, platform, style_pack, COUNT(*)::int AS c
                    FROM limited
                    GROUP BY user_id, platform, style_pack
                ),
                totals AS (
                    SELECT user_id, platform, COUNT(*)::int AS n, COUNT(DISTINCT style_pack)::int AS distinct_packs
                    FROM limited
                    GROUP BY user_id, platform
                ),
                entropy AS (
                    SELECT
                        c.user_id,
                        c.platform,
                        SUM(
                            - (c.c::double precision / NULLIF(t.n::double precision, 0))
                              * LN(c.c::double precision / NULLIF(t.n::double precision, 0))
                        ) AS h_nat,
                        t.n,
                        t.distinct_packs
                    FROM counts c
                    JOIN totals t
                      ON t.user_id = c.user_id AND t.platform = c.platform
                    GROUP BY c.user_id, c.platform, t.n, t.distinct_packs
                )
                SELECT
                    e.user_id,
                    e.platform,
                    e.n,
                    e.distinct_packs,
                    CASE
                        WHEN e.distinct_packs <= 1 THEN 0.0
                        ELSE e.h_nat / NULLIF(LN(e.distinct_packs::double precision), 0)
                    END AS entropy_norm
                FROM entropy e
                ORDER BY entropy_norm ASC, e.n DESC
                """,
                since,
                per_platform_limit,
            )
    except asyncpg.exceptions.UndefinedTableError:
        return {
            "range": range,
            "since": since.isoformat(),
            "window_per_platform": per_platform_limit,
            "entities_measured": 0,
            "avg_entropy_norm": 0.0,
            "low_entropy_entities": 0,
            "by_platform": {},
            "riskiest": [],
            "note": "upload_thumbnail_style_memory table is missing",
        }
    except Exception as e:
        logger.warning(f"/api/admin/kpi/thumbnail-diversity failed: {e}")
        return {
            "range": range,
            "since": since.isoformat(),
            "window_per_platform": per_platform_limit,
            "entities_measured": 0,
            "avg_entropy_norm": 0.0,
            "low_entropy_entities": 0,
            "by_platform": {},
            "riskiest": [],
            "error": "thumbnail_diversity_unavailable",
        }

    entities = []
    plat_bucket: Dict[str, List[float]] = {}
    low_entropy_count = 0
    for r in rows or []:
        entropy = float(r.get("entropy_norm") or 0.0)
        n = int(r.get("n") or 0)
        if n < 5:
            continue
        platform = str(r.get("platform") or "unknown").lower()
        user_id = str(r.get("user_id") or "")
        distinct_packs = int(r.get("distinct_packs") or 0)
        entities.append(
            {
                "user_id": user_id,
                "platform": platform,
                "sample_n": n,
                "distinct_packs": distinct_packs,
                "entropy_norm": round(entropy, 4),
                "risk": "high" if entropy < 0.55 else ("medium" if entropy < 0.72 else "low"),
            }
        )
        plat_bucket.setdefault(platform, []).append(entropy)
        if entropy < 0.55:
            low_entropy_count += 1

    entities.sort(key=lambda x: (x["entropy_norm"], -x["sample_n"]))
    by_platform = {
        p: {
            "avg_entropy_norm": round(sum(vals) / max(len(vals), 1), 4),
            "entities": len(vals),
        }
        for p, vals in plat_bucket.items()
    }
    avg_entropy = (
        round(sum(x["entropy_norm"] for x in entities) / max(len(entities), 1), 4)
        if entities
        else 0.0
    )

    return {
        "range": range,
        "since": since.isoformat(),
        "window_per_platform": per_platform_limit,
        "entities_measured": len(entities),
        "avg_entropy_norm": avg_entropy,
        "low_entropy_entities": low_entropy_count,
        "by_platform": by_platform,
        "riskiest": entities[:50],
    }


@app.get("/api/admin/kpi/provider-costs")
async def admin_kpi_provider_costs(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Live provider estimates (Render, R2, Redis, Mailgun, Stripe fees) for admin KPI cards."""
    from stages.kpi_collector import fetch_provider_costs_for_dashboard
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    data = await fetch_provider_costs_for_dashboard(window_minutes=minutes)
    data["range"] = _range_label(range)
    return data


@app.post("/api/admin/kpi/refresh")
async def trigger_kpi_refresh(background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    """
    Manually trigger KPI data collection from Stripe, OpenAI, Mailgun, etc.
    Runs in background; results appear in cost_tracking within ~30s.
    """
    from stages.kpi_collector import run_kpi_collect
    async def _run():
        try:
            summary = await run_kpi_collect(db_pool)
            logger.info(f"KPI refresh complete: {summary}")
        except Exception as e:
            logger.warning(f"KPI refresh failed: {e}")
    background_tasks.add_task(_run)
    return {"status": "started", "message": "KPI collection running in background"}


@app.get("/api/admin/kpi/usage")
async def get_kpi_usage(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    prev_since = since - timedelta(minutes=minutes)
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


@app.get("/api/admin/marketing/intel")
async def get_admin_marketing_intel(range: str = Query("30d"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        levers = {
            "free_users_uploading_last_7d": int(await conn.fetchval(
                """
                SELECT COUNT(DISTINCT u.id)::int
                FROM users u
                INNER JOIN uploads up ON up.user_id = u.id AND up.created_at >= NOW() - INTERVAL '7 days'
                WHERE u.subscription_tier = 'free' AND u.status = 'active'
                """
            ) or 0),
            "users_low_put_available_0_29": int(await conn.fetchval(
                """
                SELECT COUNT(*)::int
                FROM users u
                INNER JOIN wallets w ON w.user_id = u.id
                WHERE u.status = 'active'
                  AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
                  AND (w.put_balance - w.put_reserved) BETWEEN 0 AND 29
                """
            ) or 0),
            "users_low_aic_available_0_9": int(await conn.fetchval(
                """
                SELECT COUNT(*)::int
                FROM users u
                INNER JOIN wallets w ON w.user_id = u.id
                WHERE u.status = 'active'
                  AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
                  AND (w.aic_balance - w.aic_reserved) BETWEEN 0 AND 9
                """
            ) or 0),
            "users_3plus_platform_connections": int(await conn.fetchval(
                "SELECT COUNT(*)::int FROM (SELECT user_id FROM platform_tokens GROUP BY user_id HAVING COUNT(*) >= 3) s"
            ) or 0),
        }

        counts = await conn.fetchrow(
            """
            SELECT
              COALESCE(SUM(CASE WHEN event_type = 'shown' THEN 1 ELSE 0 END), 0)::bigint AS shown,
              COALESCE(SUM(CASE WHEN event_type = 'clicked' THEN 1 ELSE 0 END), 0)::bigint AS clicked,
              COALESCE(SUM(CASE WHEN event_type = 'dismissed' THEN 1 ELSE 0 END), 0)::bigint AS dismissed,
              COALESCE(SUM(CASE WHEN event_type = 'converted' THEN 1 ELSE 0 END), 0)::bigint AS converted
            FROM marketing_events
            WHERE created_at >= $1
            """,
            since,
        )
        shown = int((counts["shown"] if counts else 0) or 0)
        clicked = int((counts["clicked"] if counts else 0) or 0)
        dismissed = int((counts["dismissed"] if counts else 0) or 0)
        converted = int((counts["converted"] if counts else 0) or 0)

        same_session = await conn.fetchval(
            """
            SELECT COALESCE(SUM(r.amount), 0)::decimal
            FROM revenue_tracking r
            WHERE r.created_at >= $1
              AND EXISTS (
                SELECT 1 FROM marketing_events e
                WHERE e.user_id = r.user_id
                  AND e.event_type IN ('shown', 'clicked')
                  AND e.session_id IS NOT NULL
                  AND e.session_id <> ''
                  AND e.created_at <= r.created_at
                  AND e.created_at >= r.created_at - INTERVAL '8 hours'
              )
            """,
            since,
        ) or 0
        view_through_7d = await conn.fetchval(
            """
            SELECT COALESCE(SUM(r.amount), 0)::decimal
            FROM revenue_tracking r
            WHERE r.created_at >= $1
              AND EXISTS (
                SELECT 1 FROM marketing_events e
                WHERE e.user_id = r.user_id
                  AND e.event_type IN ('shown', 'clicked')
                  AND e.created_at <= r.created_at
                  AND e.created_at >= r.created_at - INTERVAL '7 days'
              )
            """,
            since,
        ) or 0

        schedule_rows = await conn.fetch(
            """
            WITH clicks AS (
              SELECT EXTRACT(DOW FROM created_at)::int AS dow, EXTRACT(HOUR FROM created_at)::int AS hr, COUNT(*)::bigint AS n
              FROM marketing_events
              WHERE event_type = 'clicked' AND created_at >= $1
              GROUP BY 1,2
            ),
            conv AS (
              SELECT EXTRACT(DOW FROM e.created_at)::int AS dow, EXTRACT(HOUR FROM e.created_at)::int AS hr, COUNT(*)::bigint AS n
              FROM marketing_events e
              WHERE e.event_type = 'clicked'
                AND e.created_at >= $1
                AND EXISTS (
                  SELECT 1 FROM revenue_tracking r
                  WHERE r.user_id = e.user_id
                    AND r.created_at >= e.created_at
                    AND r.created_at <= e.created_at + INTERVAL '7 days'
                )
              GROUP BY 1,2
            )
            SELECT c.dow, c.hr, c.n::bigint AS clicks, COALESCE(v.n, 0)::bigint AS conversions,
                   (COALESCE(v.n, 0)::double precision / NULLIF(c.n::double precision, 0)) AS conv_rate
            FROM clicks c
            LEFT JOIN conv v ON v.dow = c.dow AND v.hr = c.hr
            ORDER BY conv_rate DESC NULLS LAST, clicks DESC
            LIMIT 6
            """,
            since,
        )
    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    schedule = [{
        "day": dow_names[int(r["dow"]) % 7],
        "hour_utc": int(r["hr"]),
        "clicks": int(r["clicks"] or 0),
        "conversions_7d": int(r["conversions"] or 0),
        "conv_rate_7d": round(float(r["conv_rate"] or 0.0) * 100.0, 2),
    } for r in (schedule_rows or [])]
    return {
        "range": range,
        "sales_opportunity_levers": levers,
        "marketing_funnel": {
            "shown": shown,
            "clicked": clicked,
            "dismissed": dismissed,
            "converted": converted,
            "ctr_pct": round((clicked / max(shown, 1)) * 100.0, 2),
            "dismiss_rate_pct": round((dismissed / max(shown, 1)) * 100.0, 2),
            "same_session_attributed_revenue": float(same_session or 0),
            "view_through_7d_attributed_revenue": float(view_through_7d or 0),
        },
        "promo_schedule_recommendations": schedule,
        "recommended_comms_plan": [
            {"channel": "in_app_banner", "cadence": "always_on", "trigger": "quota pressure + feature gap"},
            {"channel": "email_upgrade", "cadence": "twice_weekly", "trigger": "3+ clicks without conversion"},
            {"channel": "discount_offer", "cadence": "end_of_cycle", "trigger": "high intent + no conversion in 7d"},
        ],
        "metric_definitions": metric_defs.for_marketing_intel(),
    }


@app.get("/api/admin/marketing/accounts")
async def get_admin_marketing_accounts(
    range: str = Query("30d"),
    q: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=300),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_admin),
):
    """
    Account-level marketing intelligence for outreach and enterprise expansion.
    """
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    needle = (q or "").strip()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH u AS (
              SELECT id, email, name, subscription_tier, subscription_status, status, role, created_at
              FROM users
              WHERE ($3::text = '' OR email ILIKE ('%' || $3 || '%') OR COALESCE(name, '') ILIKE ('%' || $3 || '%'))
            ),
            up AS (
              SELECT user_id, COUNT(*)::int AS uploads_30d
              FROM uploads
              WHERE created_at >= $1
              GROUP BY user_id
            ),
            rev AS (
              SELECT user_id,
                     COALESCE(SUM(amount), 0)::decimal AS revenue_30d,
                     COALESCE(SUM(CASE WHEN source = 'topup' THEN amount ELSE 0 END), 0)::decimal AS topup_30d
              FROM revenue_tracking
              WHERE created_at >= $1
              GROUP BY user_id
            ),
            me AS (
              SELECT user_id,
                     COALESCE(SUM(CASE WHEN event_type = 'shown' THEN 1 ELSE 0 END), 0)::int AS shown,
                     COALESCE(SUM(CASE WHEN event_type = 'clicked' THEN 1 ELSE 0 END), 0)::int AS clicked,
                     COALESCE(SUM(CASE WHEN event_type = 'dismissed' THEN 1 ELSE 0 END), 0)::int AS dismissed,
                     COALESCE(SUM(CASE WHEN event_type = 'converted' THEN 1 ELSE 0 END), 0)::int AS converted
              FROM marketing_events
              WHERE created_at >= $1
              GROUP BY user_id
            ),
            pf AS (
              SELECT user_id, COUNT(*)::int AS connected_accounts
              FROM platform_tokens
              GROUP BY user_id
            )
            SELECT
              u.id,
              u.email,
              u.name,
              u.subscription_tier,
              u.subscription_status,
              u.status,
              u.role,
              u.created_at,
              COALESCE(up.uploads_30d, 0)::int AS uploads_30d,
              COALESCE(rev.revenue_30d, 0)::decimal AS revenue_30d,
              COALESCE(rev.topup_30d, 0)::decimal AS topup_30d,
              COALESCE(me.shown, 0)::int AS nudge_shown,
              COALESCE(me.clicked, 0)::int AS nudge_clicked,
              COALESCE(me.dismissed, 0)::int AS nudge_dismissed,
              COALESCE(me.converted, 0)::int AS nudge_converted,
              COALESCE(pf.connected_accounts, 0)::int AS connected_accounts
            FROM u
            LEFT JOIN up  ON up.user_id = u.id
            LEFT JOIN rev ON rev.user_id = u.id
            LEFT JOIN me  ON me.user_id = u.id
            LEFT JOIN pf  ON pf.user_id = u.id
            ORDER BY revenue_30d DESC, uploads_30d DESC, created_at DESC
            LIMIT $2 OFFSET $4
            """,
            since,
            limit,
            needle,
            offset,
        )
        total = await conn.fetchval(
            """
            SELECT COUNT(*)::bigint
            FROM users
            WHERE ($1::text = '' OR email ILIKE ('%' || $1 || '%') OR COALESCE(name, '') ILIKE ('%' || $1 || '%'))
            """,
            needle,
        )
    out = []
    for r in rows:
        tier = normalize_tier((r.get("subscription_tier") or "free"))
        paid = tier not in ("free", "master_admin", "friends_family", "lifetime")
        shown = int(r.get("nudge_shown") or 0)
        clicked = int(r.get("nudge_clicked") or 0)
        ctr = (clicked / max(shown, 1)) * 100.0
        uploads_30d = int(r.get("uploads_30d") or 0)
        connected_accounts = int(r.get("connected_accounts") or 0)
        # Heuristic enterprise-fit score (0..100)
        score = 0.0
        score += min(40.0, uploads_30d * 2.2)
        score += min(25.0, connected_accounts * 4.5)
        score += min(20.0, ctr * 0.6)
        if paid:
            score += 10.0
        if tier in ("studio", "agency"):
            score += 10.0
        out.append({
            "id": str(r["id"]),
            "email": r.get("email"),
            "name": r.get("name"),
            "subscription_tier": tier,
            "subscription_status": r.get("subscription_status"),
            "status": r.get("status"),
            "role": r.get("role"),
            "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
            "uploads_30d": uploads_30d,
            "revenue_30d": float(r.get("revenue_30d") or 0),
            "topup_30d": float(r.get("topup_30d") or 0),
            "nudge_shown": shown,
            "nudge_clicked": clicked,
            "nudge_dismissed": int(r.get("nudge_dismissed") or 0),
            "nudge_converted": int(r.get("nudge_converted") or 0),
            "nudge_ctr_pct": round(ctr, 2),
            "connected_accounts": connected_accounts,
            "enterprise_fit_score": round(max(0.0, min(score, 100.0)), 1),
        })
    return {"accounts": out, "total": int(total or 0), "range": range}


async def _estimate_marketing_audience(conn, payload: MarketingCampaignIn) -> int:
    minutes = _range_to_minutes(payload.range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    tiers = [normalize_tier(t) for t in (payload.tiers or []) if normalize_tier(t)]
    rows = await conn.fetch(
        """
        WITH u AS (
          SELECT id, subscription_tier
          FROM users
          WHERE status = 'active'
            AND ($2::text[] IS NULL OR cardinality($2::text[]) = 0 OR COALESCE(subscription_tier, 'free') = ANY($2::text[]))
        ),
        up AS (
          SELECT user_id, COUNT(*)::int AS uploads_30d
          FROM uploads
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        rv AS (
          SELECT user_id, COALESCE(SUM(amount), 0)::decimal AS revenue_7d
          FROM revenue_tracking
          WHERE created_at >= NOW() - INTERVAL '7 days'
          GROUP BY user_id
        ),
        me AS (
          SELECT user_id,
                 COALESCE(SUM(CASE WHEN event_type='shown' THEN 1 ELSE 0 END),0)::int AS shown,
                 COALESCE(SUM(CASE WHEN event_type='clicked' THEN 1 ELSE 0 END),0)::int AS clicked
          FROM marketing_events
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        pf AS (
          SELECT user_id, COUNT(*)::int AS connected_accounts
          FROM platform_tokens
          GROUP BY user_id
        )
        SELECT
          u.id,
          COALESCE(up.uploads_30d, 0)::int AS uploads_30d,
          COALESCE(rv.revenue_7d, 0)::decimal AS revenue_7d,
          COALESCE(me.shown, 0)::int AS shown,
          COALESCE(me.clicked, 0)::int AS clicked,
          COALESCE(pf.connected_accounts, 0)::int AS connected_accounts,
          COALESCE(u.subscription_tier, 'free') AS tier
        FROM u
        LEFT JOIN up ON up.user_id = u.id
        LEFT JOIN rv ON rv.user_id = u.id
        LEFT JOIN me ON me.user_id = u.id
        LEFT JOIN pf ON pf.user_id = u.id
        """,
        since,
        tiers if tiers else None,
    )
    count = 0
    for r in rows:
        uploads_30d = int(r.get("uploads_30d") or 0)
        shown = int(r.get("shown") or 0)
        clicked = int(r.get("clicked") or 0)
        ctr = (clicked / max(shown, 1)) * 100.0
        connected_accounts = int(r.get("connected_accounts") or 0)
        tier = normalize_tier(str(r.get("tier") or "free"))
        score = 0.0
        score += min(40.0, uploads_30d * 2.2)
        score += min(25.0, connected_accounts * 4.5)
        score += min(20.0, ctr * 0.6)
        if tier not in ("free", "master_admin", "friends_family", "lifetime"):
            score += 10.0
        if tier in ("studio", "agency"):
            score += 10.0

        if uploads_30d < int(payload.min_uploads_30d or 0):
            continue
        if ctr < float(payload.min_nudge_ctr_pct or 0):
            continue
        if score < float(payload.min_enterprise_fit_score or 0):
            continue
        if payload.require_no_revenue_7d and float(r.get("revenue_7d") or 0) > 0:
            continue
        count += 1
    return int(count)


async def _resolve_marketing_campaign_audience(conn, range_key: str, targeting: Dict[str, Any], limit: int = 5000) -> List[Dict[str, Any]]:
    minutes = _range_to_minutes(range_key, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    tiers = [normalize_tier(t) for t in (targeting.get("tiers") or []) if normalize_tier(t)]
    min_uploads = int(targeting.get("min_uploads_30d") or 0)
    min_score = float(targeting.get("min_enterprise_fit_score") or 0)
    min_ctr = float(targeting.get("min_nudge_ctr_pct") or 0)
    no_rev_7d = bool(targeting.get("require_no_revenue_7d"))
    rows = await conn.fetch(
        """
        WITH u AS (
          SELECT id, email, name, subscription_tier, subscription_status, status
          FROM users
          WHERE status = 'active'
            AND ($2::text[] IS NULL OR cardinality($2::text[]) = 0 OR COALESCE(subscription_tier, 'free') = ANY($2::text[]))
        ),
        up AS (
          SELECT user_id, COUNT(*)::int AS uploads_30d
          FROM uploads
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        rv AS (
          SELECT user_id, COALESCE(SUM(amount), 0)::decimal AS revenue_7d
          FROM revenue_tracking
          WHERE created_at >= NOW() - INTERVAL '7 days'
          GROUP BY user_id
        ),
        me AS (
          SELECT user_id,
                 COALESCE(SUM(CASE WHEN event_type='shown' THEN 1 ELSE 0 END),0)::int AS shown,
                 COALESCE(SUM(CASE WHEN event_type='clicked' THEN 1 ELSE 0 END),0)::int AS clicked,
                 COALESCE(SUM(CASE WHEN event_type='dismissed' THEN 1 ELSE 0 END),0)::int AS dismissed,
                 COALESCE(SUM(CASE WHEN event_type='converted' THEN 1 ELSE 0 END),0)::int AS converted
          FROM marketing_events
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        pf AS (
          SELECT user_id, COUNT(*)::int AS connected_accounts
          FROM platform_tokens
          GROUP BY user_id
        )
        SELECT
          u.id, u.email, u.name, u.subscription_tier, u.subscription_status, u.status,
          COALESCE(up.uploads_30d, 0)::int AS uploads_30d,
          COALESCE(rv.revenue_7d, 0)::decimal AS revenue_7d,
          COALESCE(me.shown, 0)::int AS shown,
          COALESCE(me.clicked, 0)::int AS clicked,
          COALESCE(me.dismissed, 0)::int AS dismissed,
          COALESCE(me.converted, 0)::int AS converted,
          COALESCE(pf.connected_accounts, 0)::int AS connected_accounts
        FROM u
        LEFT JOIN up ON up.user_id = u.id
        LEFT JOIN rv ON rv.user_id = u.id
        LEFT JOIN me ON me.user_id = u.id
        LEFT JOIN pf ON pf.user_id = u.id
        ORDER BY COALESCE(up.uploads_30d, 0) DESC, u.created_at DESC
        LIMIT $3
        """,
        since,
        tiers if tiers else None,
        max(1, min(limit, 10000)),
    )
    out = []
    for r in rows:
        uploads_30d = int(r.get("uploads_30d") or 0)
        shown = int(r.get("shown") or 0)
        clicked = int(r.get("clicked") or 0)
        ctr = (clicked / max(shown, 1)) * 100.0
        connected_accounts = int(r.get("connected_accounts") or 0)
        tier = normalize_tier(str(r.get("subscription_tier") or "free"))
        score = 0.0
        score += min(40.0, uploads_30d * 2.2)
        score += min(25.0, connected_accounts * 4.5)
        score += min(20.0, ctr * 0.6)
        if tier not in ("free", "master_admin", "friends_family", "lifetime"):
            score += 10.0
        if tier in ("studio", "agency"):
            score += 10.0
        if uploads_30d < min_uploads:
            continue
        if ctr < min_ctr:
            continue
        if score < min_score:
            continue
        if no_rev_7d and float(r.get("revenue_7d") or 0) > 0:
            continue
        out.append({
            "id": str(r["id"]),
            "email": r.get("email"),
            "name": r.get("name"),
            "subscription_tier": tier,
            "subscription_status": r.get("subscription_status"),
            "uploads_30d": uploads_30d,
            "revenue_7d": float(r.get("revenue_7d") or 0),
            "nudge_shown": shown,
            "nudge_clicked": clicked,
            "nudge_dismissed": int(r.get("dismissed") or 0),
            "nudge_converted": int(r.get("converted") or 0),
            "nudge_ctr_pct": round(ctr, 2),
            "connected_accounts": connected_accounts,
            "enterprise_fit_score": round(max(0.0, min(score, 100.0)), 1),
        })
    return out


async def _marketing_ai_snapshot(conn, range_key: str) -> Dict[str, Any]:
    minutes = _range_to_minutes(range_key, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    users = await conn.fetchrow(
        """
        SELECT
          COUNT(*)::int AS total_users,
          COUNT(*) FILTER (WHERE status = 'active')::int AS active_users,
          COUNT(*) FILTER (WHERE COALESCE(subscription_tier,'free') NOT IN ('free','master_admin','friends_family','lifetime')
                           AND subscription_status = 'active')::int AS paid_users
        FROM users
        """
    )
    uploads = await conn.fetchrow(
        f"""
        SELECT
          COUNT(*)::int AS total_uploads,
          COUNT(*) FILTER (WHERE status IN {SUCCESSFUL_STATUS_SQL_IN})::int AS successful_uploads,
          COALESCE(SUM(views), 0)::bigint AS views,
          COALESCE(SUM(likes), 0)::bigint AS likes
        FROM uploads
        WHERE created_at >= $1
        """,
        since,
    )
    revenue = await conn.fetchrow(
        """
        SELECT
          COALESCE(SUM(amount), 0)::decimal AS total_revenue,
          COALESCE(SUM(CASE WHEN source='topup' THEN amount ELSE 0 END), 0)::decimal AS topup_revenue
        FROM revenue_tracking
        WHERE created_at >= $1
        """,
        since,
    )
    funnel = await conn.fetchrow(
        """
        SELECT
          COALESCE(SUM(CASE WHEN event_type='shown' THEN 1 ELSE 0 END), 0)::bigint AS shown,
          COALESCE(SUM(CASE WHEN event_type='clicked' THEN 1 ELSE 0 END), 0)::bigint AS clicked,
          COALESCE(SUM(CASE WHEN event_type='dismissed' THEN 1 ELSE 0 END), 0)::bigint AS dismissed,
          COALESCE(SUM(CASE WHEN event_type='converted' THEN 1 ELSE 0 END), 0)::bigint AS converted
        FROM marketing_events
        WHERE created_at >= $1
        """,
        since,
    )
    pressure = await conn.fetchrow(
        """
        SELECT
          COUNT(*) FILTER (WHERE (w.put_balance - w.put_reserved) BETWEEN 0 AND 29)::int AS low_put_users,
          COUNT(*) FILTER (WHERE (w.aic_balance - w.aic_reserved) BETWEEN 0 AND 9)::int AS low_aic_users
        FROM users u
        JOIN wallets w ON w.user_id = u.id
        WHERE u.status = 'active'
          AND COALESCE(u.subscription_tier, 'free') NOT IN ('master_admin', 'friends_family', 'lifetime')
        """
    )
    multi_pf = await conn.fetchval(
        "SELECT COUNT(*)::int FROM (SELECT user_id FROM platform_tokens GROUP BY user_id HAVING COUNT(*) >= 3) s"
    )
    top_accounts = await conn.fetch(
        """
        SELECT
          u.id, u.email, COALESCE(u.name, split_part(u.email,'@',1)) AS name,
          COALESCE(u.subscription_tier,'free') AS tier,
          COALESCE(r.rev,0)::decimal AS revenue_window,
          COALESCE(up.cnt,0)::int AS uploads_window
        FROM users u
        LEFT JOIN (
          SELECT user_id, SUM(amount) AS rev
          FROM revenue_tracking
          WHERE created_at >= $1
          GROUP BY user_id
        ) r ON r.user_id = u.id
        LEFT JOIN (
          SELECT user_id, COUNT(*)::int AS cnt
          FROM uploads
          WHERE created_at >= $1
          GROUP BY user_id
        ) up ON up.user_id = u.id
        WHERE u.status = 'active'
        ORDER BY COALESCE(r.rev,0) DESC, COALESCE(up.cnt,0) DESC
        LIMIT 10
        """,
        since,
    )
    tier_rows = await conn.fetch(
        """
        SELECT COALESCE(subscription_tier, 'free') AS tier, COUNT(*)::int AS users
        FROM users
        WHERE status = 'active'
        GROUP BY COALESCE(subscription_tier, 'free')
        ORDER BY users DESC
        """
    )
    segment_rows = await conn.fetchrow(
        """
        WITH active_users AS (
          SELECT id, COALESCE(subscription_tier, 'free') AS tier
          FROM users
          WHERE status = 'active'
        ),
        up AS (
          SELECT user_id, COUNT(*)::int AS uploads_window
          FROM uploads
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        rev AS (
          SELECT user_id, COALESCE(SUM(amount), 0)::decimal AS revenue_window
          FROM revenue_tracking
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        m AS (
          SELECT user_id,
                 COALESCE(SUM(CASE WHEN event_type = 'shown' THEN 1 ELSE 0 END), 0)::int AS shown,
                 COALESCE(SUM(CASE WHEN event_type = 'clicked' THEN 1 ELSE 0 END), 0)::int AS clicked
          FROM marketing_events
          WHERE created_at >= $1
          GROUP BY user_id
        ),
        pf AS (
          SELECT user_id, COUNT(*)::int AS connected_accounts
          FROM platform_tokens
          GROUP BY user_id
        ),
        w AS (
          SELECT user_id,
                 (put_balance - put_reserved)::bigint AS put_available,
                 (aic_balance - aic_reserved)::bigint AS aic_available
          FROM wallets
        )
        SELECT
          COUNT(*) FILTER (
              WHERE COALESCE(up.uploads_window, 0) >= 4
                AND au.tier = 'free'
          )::int AS free_high_intent_uploaders,
          COUNT(*) FILTER (
              WHERE COALESCE(w.put_available, 0) BETWEEN 0 AND 29
                 OR COALESCE(w.aic_available, 0) BETWEEN 0 AND 9
          )::int AS token_pressure_accounts,
          COUNT(*) FILTER (
              WHERE COALESCE(pf.connected_accounts, 0) >= 3
                AND au.tier NOT IN ('agency', 'master_admin', 'friends_family', 'lifetime')
          )::int AS expansion_ready_accounts,
          COUNT(*) FILTER (
              WHERE COALESCE(m.clicked, 0) >= 2
                AND COALESCE(rev.revenue_window, 0) = 0
          )::int AS engaged_no_purchase_accounts
        FROM active_users au
        LEFT JOIN up ON up.user_id = au.id
        LEFT JOIN rev ON rev.user_id = au.id
        LEFT JOIN m ON m.user_id = au.id
        LEFT JOIN pf ON pf.user_id = au.id
        LEFT JOIN w ON w.user_id = au.id
        """,
        since,
    )
    source_rows = await conn.fetch(
        """
        SELECT COALESCE(source, 'unknown') AS source, COALESCE(SUM(amount), 0)::decimal AS amount
        FROM revenue_tracking
        WHERE created_at >= $1
        GROUP BY COALESCE(source, 'unknown')
        ORDER BY amount DESC
        LIMIT 8
        """,
        since,
    )
    platform_rows = await conn.fetch(
        """
        WITH p AS (
          SELECT
            LOWER(TRIM(unnest(platforms))) AS platform,
            status,
            COALESCE(views, 0)::bigint AS views,
            COALESCE(likes, 0)::bigint AS likes
          FROM uploads
          WHERE created_at >= $1
        )
        SELECT
          platform,
          COUNT(*)::int AS uploads,
          COUNT(*) FILTER (WHERE status IN ('completed', 'succeeded'))::int AS successful_uploads,
          COALESCE(SUM(views), 0)::bigint AS views,
          COALESCE(SUM(likes), 0)::bigint AS likes
        FROM p
        WHERE platform IS NOT NULL AND platform <> ''
        GROUP BY platform
        ORDER BY uploads DESC
        LIMIT 8
        """,
        since,
    )
    platform_kpis = await conn.fetchrow(
        """
        WITH connected AS (
          SELECT COUNT(DISTINCT user_id)::int AS connected_users
          FROM platform_tokens
          WHERE revoked_at IS NULL
        ),
        cached AS (
          SELECT
            COUNT(*)::int AS cached_users,
            COALESCE(SUM(
              CASE WHEN COALESCE(data->'aggregate'->>'views','') ~ '^[0-9]+$'
                   THEN (data->'aggregate'->>'views')::bigint ELSE 0 END
            ), 0)::bigint AS platform_views,
            COALESCE(SUM(
              CASE WHEN COALESCE(data->'aggregate'->>'likes','') ~ '^[0-9]+$'
                   THEN (data->'aggregate'->>'likes')::bigint ELSE 0 END
            ), 0)::bigint AS platform_likes,
            COALESCE(SUM(
              CASE WHEN COALESCE(data->'aggregate'->>'comments','') ~ '^[0-9]+$'
                   THEN (data->'aggregate'->>'comments')::bigint ELSE 0 END
            ), 0)::bigint AS platform_comments,
            COALESCE(SUM(
              CASE WHEN COALESCE(data->'aggregate'->>'shares','') ~ '^[0-9]+$'
                   THEN (data->'aggregate'->>'shares')::bigint ELSE 0 END
            ), 0)::bigint AS platform_shares
          FROM platform_metrics_cache
          WHERE fetched_at >= $1
        )
        SELECT
          connected.connected_users,
          cached.cached_users,
          cached.platform_views,
          cached.platform_likes,
          cached.platform_comments,
          cached.platform_shares
        FROM connected
        CROSS JOIN cached
        """,
        since,
    )
    mrr_rows = await conn.fetch(
        """
        SELECT COALESCE(subscription_tier, 'free') AS tier, COUNT(*)::int AS users
        FROM users
        WHERE subscription_status = 'active'
          AND COALESCE(subscription_tier, 'free') NOT IN ('free', 'master_admin', 'friends_family', 'lifetime')
        GROUP BY COALESCE(subscription_tier, 'free')
        """
    )
    costs_row = await conn.fetchrow(
        """
        SELECT
          COALESCE(SUM(CASE WHEN category='openai' THEN cost_usd ELSE 0 END),0)::decimal AS openai,
          COALESCE(SUM(CASE WHEN category='storage' THEN cost_usd ELSE 0 END),0)::decimal AS storage,
          COALESCE(SUM(CASE WHEN category='compute' THEN cost_usd ELSE 0 END),0)::decimal AS compute,
          COALESCE(SUM(CASE WHEN category IN ('stripe_fees','mailgun','bandwidth','postgres','redis','tool_estimate') THEN cost_usd ELSE 0 END),0)::decimal AS other
        FROM cost_tracking
        WHERE created_at >= $1
        """,
        since,
    )
    cancellations = await conn.fetchval(
        "SELECT COUNT(*)::int FROM users WHERE subscription_status = 'cancelled' AND updated_at >= $1",
        since,
    )
    same_session_attr = await conn.fetchval(
        """
        SELECT COALESCE(SUM(r.amount), 0)::decimal
        FROM revenue_tracking r
        WHERE r.created_at >= $1
          AND EXISTS (
            SELECT 1 FROM marketing_events e
            WHERE e.user_id = r.user_id
              AND e.event_type IN ('shown', 'clicked')
              AND e.created_at <= r.created_at
              AND e.created_at >= r.created_at - INTERVAL '8 hours'
          )
        """,
        since,
    ) or 0
    view_through_7d = await conn.fetchval(
        """
        SELECT COALESCE(SUM(r.amount), 0)::decimal
        FROM revenue_tracking r
        WHERE r.created_at >= $1
          AND EXISTS (
            SELECT 1 FROM marketing_events e
            WHERE e.user_id = r.user_id
              AND e.event_type IN ('shown', 'clicked')
              AND e.created_at <= r.created_at
              AND e.created_at >= r.created_at - INTERVAL '7 days'
          )
        """,
        since,
    ) or 0
    promo_window_rows = await conn.fetch(
        """
        WITH clicks AS (
          SELECT EXTRACT(DOW FROM created_at)::int AS dow, EXTRACT(HOUR FROM created_at)::int AS hr, COUNT(*)::bigint AS n
          FROM marketing_events
          WHERE event_type = 'clicked' AND created_at >= $1
          GROUP BY 1,2
        ),
        conv AS (
          SELECT EXTRACT(DOW FROM e.created_at)::int AS dow, EXTRACT(HOUR FROM e.created_at)::int AS hr, COUNT(*)::bigint AS n
          FROM marketing_events e
          WHERE e.event_type = 'clicked' AND e.created_at >= $1
            AND EXISTS (
              SELECT 1 FROM revenue_tracking r
              WHERE r.user_id = e.user_id
                AND r.created_at >= e.created_at
                AND r.created_at <= e.created_at + INTERVAL '7 days'
            )
          GROUP BY 1,2
        )
        SELECT c.dow, c.hr, c.n::bigint AS clicks, COALESCE(v.n,0)::bigint AS conversions,
               (COALESCE(v.n,0)::double precision / NULLIF(c.n::double precision,0)) AS conv_rate
        FROM clicks c
        LEFT JOIN conv v ON v.dow = c.dow AND v.hr = c.hr
        ORDER BY conv_rate DESC NULLS LAST, clicks DESC
        LIMIT 8
        """,
        since,
    )
    shown = int((funnel or {}).get("shown") or 0)
    clicked = int((funnel or {}).get("clicked") or 0)
    ctr = (clicked / max(shown, 1)) * 100.0
    total_revenue = float((revenue or {}).get("total_revenue") or 0)
    total_costs = float((costs_row or {}).get("openai") or 0) + float((costs_row or {}).get("storage") or 0) + float((costs_row or {}).get("compute") or 0) + float((costs_row or {}).get("other") or 0)
    gross_margin_pct = ((total_revenue - total_costs) / max(total_revenue, 1.0)) * 100.0 if total_revenue > 0 else 0.0
    platform_views = int((platform_kpis or {}).get("platform_views") or 0)
    platform_likes = int((platform_kpis or {}).get("platform_likes") or 0)
    platform_comments = int((platform_kpis or {}).get("platform_comments") or 0)
    platform_shares = int((platform_kpis or {}).get("platform_shares") or 0)
    platform_engagements = platform_likes + platform_comments + platform_shares
    platform_engagement_rate_pct = (platform_engagements / max(platform_views, 1)) * 100.0 if platform_views > 0 else 0.0
    mrr_by_tier: Dict[str, float] = {}
    mrr_estimate = 0.0
    for r in (mrr_rows or []):
        t = normalize_tier(str(r.get("tier") or "free"))
        amt = float(get_plan(t).get("price", 0) or 0) * int(r.get("users") or 0)
        mrr_by_tier[t] = round(amt, 2)
        mrr_estimate += amt
    # ML truth: top-performing creative strategies from empirical outcomes.
    ml_truth_top_strategies: List[Dict[str, Any]] = []
    try:
        ml_days = max(1, int(minutes / (60.0 * 24.0)))
        ml_rows = await conn.fetch(
            """
            SELECT
                strategy_key,
                SUM(samples)::bigint AS samples,
                CASE
                  WHEN SUM(samples) > 0
                  THEN SUM(mean_engagement * samples)::double precision / NULLIF(SUM(samples)::double precision, 0)
                  ELSE 0.0
                END AS weighted_mean_engagement,
                MAX(ci95_high)::double precision AS max_ci95_high,
                COUNT(DISTINCT day)::int AS days_with_data
            FROM upload_quality_scores_daily
            WHERE day >= (CURRENT_DATE - ($1::int || ' days')::interval)::date
              AND platform = 'all'
            GROUP BY strategy_key
            ORDER BY weighted_mean_engagement DESC, samples DESC
            LIMIT 8
            """,
            ml_days,
        )
        ml_truth_top_strategies = [
            {
                "strategy_key": str(r.get("strategy_key") or "default"),
                "samples": int(r.get("samples") or 0),
                "weighted_mean_engagement": float(r.get("weighted_mean_engagement") or 0),
                "max_ci95_high": float(r.get("max_ci95_high") or 0),
                "days_with_data": int(r.get("days_with_data") or 0),
            }
            for r in (ml_rows or [])
        ]
    except Exception as e:
        logger.debug(f"ml_truth_top_strategies query failed: {e}")

    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    tier_mix: Dict[str, int] = {}
    for r in (tier_rows or []):
        t = normalize_tier(str(r.get("tier") or "free"))
        tier_mix[t] = tier_mix.get(t, 0) + int(r.get("users") or 0)
    return {
        "range": range_key,
        "ml_truth": {"top_strategies": ml_truth_top_strategies},
        "kpis": {
            "total_users": int((users or {}).get("total_users") or 0),
            "active_users": int((users or {}).get("active_users") or 0),
            "paid_users": int((users or {}).get("paid_users") or 0),
            "total_uploads": int((uploads or {}).get("total_uploads") or 0),
            "successful_uploads": int((uploads or {}).get("successful_uploads") or 0),
            "views": int((uploads or {}).get("views") or 0),
            "likes": int((uploads or {}).get("likes") or 0),
            "total_revenue": float((revenue or {}).get("total_revenue") or 0),
            "topup_revenue": float((revenue or {}).get("topup_revenue") or 0),
            "mrr_estimate": round(mrr_estimate, 2),
            "mrr_by_tier": mrr_by_tier,
            "cancellations_window": int(cancellations or 0),
            "nudge_shown": shown,
            "nudge_clicked": clicked,
            "nudge_dismissed": int((funnel or {}).get("dismissed") or 0),
            "nudge_converted": int((funnel or {}).get("converted") or 0),
            "nudge_ctr_pct": round(ctr, 2),
            "same_session_attributed_revenue": float(same_session_attr or 0),
            "view_through_7d_attributed_revenue": float(view_through_7d or 0),
            "openai_cost": float((costs_row or {}).get("openai") or 0),
            "storage_cost": float((costs_row or {}).get("storage") or 0),
            "compute_cost": float((costs_row or {}).get("compute") or 0),
            "other_costs": float((costs_row or {}).get("other") or 0),
            "total_costs": round(total_costs, 2),
            "gross_margin_pct": round(gross_margin_pct, 2),
            "low_put_users": int((pressure or {}).get("low_put_users") or 0),
            "low_aic_users": int((pressure or {}).get("low_aic_users") or 0),
            "users_3plus_platform_connections": int(multi_pf or 0),
            "platform_connected_users": int((platform_kpis or {}).get("connected_users") or 0),
            "platform_cached_users": int((platform_kpis or {}).get("cached_users") or 0),
            "platform_views": platform_views,
            "platform_likes": platform_likes,
            "platform_comments": platform_comments,
            "platform_shares": platform_shares,
            "platform_engagements": platform_engagements,
            "platform_engagement_rate_pct": round(platform_engagement_rate_pct, 2),
        },
        "best_promo_windows_utc": [{
            "day": dow_names[int(r.get("dow") or 0) % 7],
            "hour_utc": int(r.get("hr") or 0),
            "clicks": int(r.get("clicks") or 0),
            "conversions_7d": int(r.get("conversions") or 0),
            "conv_rate_7d": round(float(r.get("conv_rate") or 0.0) * 100.0, 2),
        } for r in (promo_window_rows or [])],
        "tier_mix_active_users": tier_mix,
        "segment_signals": {
            "free_high_intent_uploaders": int((segment_rows or {}).get("free_high_intent_uploaders") or 0),
            "token_pressure_accounts": int((segment_rows or {}).get("token_pressure_accounts") or 0),
            "expansion_ready_accounts": int((segment_rows or {}).get("expansion_ready_accounts") or 0),
            "engaged_no_purchase_accounts": int((segment_rows or {}).get("engaged_no_purchase_accounts") or 0),
        },
        "revenue_source_mix": [
            {"source": str(r.get("source") or "unknown"), "amount": float(r.get("amount") or 0)}
            for r in (source_rows or [])
        ],
        "platform_kpis": [{
            "platform": str(r.get("platform") or "unknown"),
            "uploads": int(r.get("uploads") or 0),
            "successful_uploads": int(r.get("successful_uploads") or 0),
            "success_rate_pct": round((int(r.get("successful_uploads") or 0) / max(int(r.get("uploads") or 0), 1)) * 100.0, 2),
            "views": int(r.get("views") or 0),
            "likes": int(r.get("likes") or 0),
            "engagement_per_upload": round((int(r.get("likes") or 0) / max(int(r.get("uploads") or 0), 1)), 3),
        } for r in (platform_rows or [])],
        "top_accounts": [{
            "id": str(r["id"]),
            "name": r.get("name"),
            "email": r.get("email"),
            "tier": normalize_tier(str(r.get("tier") or "free")),
            "revenue_window": float(r.get("revenue_window") or 0),
            "uploads_window": int(r.get("uploads_window") or 0),
        } for r in (top_accounts or [])],
    }


async def _log_ai_marketing_decision(
    conn,
    *,
    created_by: str,
    action: str,
    objective: str,
    range_key: str,
    used_openai: bool,
    status: str,
    snapshot: Dict[str, Any],
    plan: Dict[str, Any],
    decision: Optional[Dict[str, Any]] = None,
    campaign_id: Optional[str] = None,
) -> None:
    d = decision or {}
    await conn.execute(
        """
        INSERT INTO ai_marketing_decisions (
            created_by, action, objective, range_key, used_openai, deploy_allowed, forced, confidence_score,
            status, blocked_reasons, snapshot, decision, plan, campaign_id
        )
        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb, $12::jsonb, $13::jsonb, $14::uuid)
        """,
        str(created_by),
        action,
        objective,
        range_key,
        bool(used_openai),
        d.get("deploy_allowed"),
        bool(d.get("force_deploy", False)),
        int(d.get("confidence_score") or 0) if d.get("confidence_score") is not None else None,
        status,
        json.dumps(d.get("blocked_reasons") or []),
        json.dumps(snapshot or {}),
        json.dumps(d or {}),
        json.dumps(plan or {}),
        campaign_id,
    )


def _ai_marketing_metric_sources() -> List[Dict[str, str]]:
    return [
        {
            "metric_group": "core_kpis",
            "sql_source": "users, uploads, revenue_tracking, marketing_events, cost_tracking",
            "description": "Windowed KPIs: active users, uploads, revenue, nudge funnel, and margin inputs",
        },
        {
            "metric_group": "segment_signals",
            "sql_source": "users + uploads + revenue_tracking + marketing_events + platform_tokens + wallets",
            "description": "Opportunity cohorts: free high-intent, token pressure, expansion-ready, engaged-no-purchase",
        },
        {
            "metric_group": "tier_and_revenue_mix",
            "sql_source": "users + revenue_tracking",
            "description": "Active tier distribution, MRR estimate by tier, revenue source mix",
        },
        {
            "metric_group": "platform_analytics",
            "sql_source": "platform_metrics_cache + uploads(unnest(platforms)) + platform_tokens",
            "description": "Coverage and platform KPI rollups used for factual plan weighting",
        },
        {
            "metric_group": "promo_windows",
            "sql_source": "marketing_events joined to revenue_tracking conversion window (7d)",
            "description": "Best day/hour UTC windows ranked by click->conversion performance",
        },
    ]


@app.post("/api/admin/marketing/campaigns/preview")
async def preview_marketing_campaign(payload: MarketingCampaignIn, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        audience = await _estimate_marketing_audience(conn, payload)
    return {"estimated_audience": audience}


@app.get("/api/admin/marketing/campaigns")
async def list_marketing_campaigns(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, objective, channel, status, range_key, targeting, estimated_audience,
                   schedule_at, notes, created_at, updated_at
            FROM marketing_campaigns
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    return {"campaigns": [dict(r) for r in rows]}


@app.post("/api/admin/marketing/campaigns")
async def create_marketing_campaign(payload: MarketingCampaignIn, user: dict = Depends(require_admin)):
    targeting = {
        "tiers": [normalize_tier(t) for t in payload.tiers or []],
        "min_uploads_30d": int(payload.min_uploads_30d or 0),
        "min_enterprise_fit_score": float(payload.min_enterprise_fit_score or 0),
        "min_nudge_ctr_pct": float(payload.min_nudge_ctr_pct or 0),
        "require_no_revenue_7d": bool(payload.require_no_revenue_7d),
    }
    status = "scheduled" if payload.schedule_at else "draft"
    async with db_pool.acquire() as conn:
        audience = await _estimate_marketing_audience(conn, payload)
        row = await conn.fetchrow(
            """
            INSERT INTO marketing_campaigns (
                created_by, name, objective, channel, status, range_key, targeting,
                estimated_audience, schedule_at, notes
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
            RETURNING id, name, objective, channel, status, range_key, targeting, estimated_audience, schedule_at, notes, created_at, updated_at
            """,
            str(user["id"]),
            payload.name,
            payload.objective,
            payload.channel,
            status,
            payload.range,
            json.dumps(targeting),
            audience,
            payload.schedule_at,
            payload.notes or "",
        )
    return {"campaign": dict(row)}


@app.post("/api/admin/marketing/campaigns/{campaign_id}/status")
async def update_marketing_campaign_status(campaign_id: str, payload: MarketingCampaignStatusIn, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE marketing_campaigns
            SET status = $2, updated_at = NOW()
            WHERE id = $1::uuid
            RETURNING id, name, status, updated_at
            """,
            campaign_id,
            payload.status,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"campaign": dict(row)}


@app.get("/api/admin/marketing/campaigns/{campaign_id}/audience.csv")
async def export_marketing_campaign_audience_csv(campaign_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        c = await conn.fetchrow(
            """
            SELECT id, name, range_key, targeting
            FROM marketing_campaigns
            WHERE id = $1::uuid
            """,
            campaign_id,
        )
        if not c:
            raise HTTPException(status_code=404, detail="Campaign not found")
        targeting = dict(c.get("targeting") or {})
        rows = await _resolve_marketing_campaign_audience(conn, c.get("range_key") or "30d", targeting, limit=10000)
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow([
        "user_id", "email", "name", "subscription_tier", "subscription_status", "uploads_30d",
        "revenue_7d", "nudge_shown", "nudge_clicked", "nudge_ctr_pct", "connected_accounts", "enterprise_fit_score",
    ])
    for r in rows:
        w.writerow([
            r.get("id"), r.get("email"), r.get("name"), r.get("subscription_tier"), r.get("subscription_status"),
            r.get("uploads_30d"), r.get("revenue_7d"), r.get("nudge_shown"), r.get("nudge_clicked"),
            r.get("nudge_ctr_pct"), r.get("connected_accounts"), r.get("enterprise_fit_score"),
        ])
    data = output.getvalue()
    headers = {
        "Content-Disposition": f'attachment; filename="marketing_campaign_{campaign_id}_audience.csv"',
        "Cache-Control": "no-store",
    }
    return PlainTextResponse(content=data, headers=headers, media_type="text/csv")


@app.get("/api/admin/marketing/campaigns/{campaign_id}/handoff")
async def handoff_marketing_campaign(campaign_id: str, user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        c = await conn.fetchrow(
            """
            SELECT id, name, objective, channel, range_key, targeting, estimated_audience
            FROM marketing_campaigns
            WHERE id = $1::uuid
            """,
            campaign_id,
        )
        if not c:
            raise HTTPException(status_code=404, detail="Campaign not found")
        targeting = dict(c.get("targeting") or {})
        audience = await _resolve_marketing_campaign_audience(conn, c.get("range_key") or "30d", targeting, limit=500)
    audience_ids = [a["id"] for a in audience if a.get("id")]
    tiers = targeting.get("tiers") or []
    title = f"{c.get('name')} - campaign announcement"
    body = (
        f"{c.get('objective')}\n\n"
        f"Channel: {c.get('channel')}\n"
        f"Targeting tiers: {', '.join(tiers) if tiers else 'all active tiers'}\n"
        f"Estimated audience: {int(c.get('estimated_audience') or 0)}\n\n"
        "This message is generated from Marketing Ops handoff and should be finalized before send."
    )
    return {
        "campaign_id": str(c["id"]),
        "title": title,
        "body": body,
        "target": "specific" if audience_ids else "all",
        "selected_user_ids": audience_ids,
        "selected_user_count": len(audience_ids),
    }


def _marketing_ai_fortune500_plan(payload: MarketingAIGenerateIn, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic lifecycle plan inspired by enterprise GTM playbooks:
    segment -> pressure -> offer -> cadence -> measurable KPI.
    """
    k = snapshot.get("kpis", {}) or {}
    seg = snapshot.get("segment_signals", {}) or {}
    tier_mix = snapshot.get("tier_mix_active_users", {}) or {}
    revenue_mix = snapshot.get("revenue_source_mix", []) or []
    total_rev = float(k.get("total_revenue", 0) or 0)
    topup_rev = float(k.get("topup_revenue", 0) or 0)
    rev_share_topup = round((topup_rev / max(total_rev, 1.0)) * 100.0, 2)
    active_users = int(k.get("active_users", 0) or 0)
    paid_users = int(k.get("paid_users", 0) or 0)
    paid_rate = round((paid_users / max(active_users, 1)) * 100.0, 2)
    nudge_ctr = float(k.get("nudge_ctr_pct", 0) or 0)
    platform_connected_users = int(k.get("platform_connected_users", 0) or 0)
    platform_cached_users = int(k.get("platform_cached_users", 0) or 0)
    platform_views = int(k.get("platform_views", 0) or 0)
    platform_engagements = int(k.get("platform_engagements", 0) or 0)
    platform_er = float(k.get("platform_engagement_rate_pct", 0) or 0)
    platform_coverage_pct = round((platform_cached_users / max(platform_connected_users, 1)) * 100.0, 2) if platform_connected_users > 0 else 0.0

    # Cohort sizing for plan confidence
    free_high_intent = int(seg.get("free_high_intent_uploaders", 0) or 0)
    token_pressure = int(seg.get("token_pressure_accounts", 0) or 0)
    expansion_ready = int(seg.get("expansion_ready_accounts", 0) or 0)
    engaged_no_purchase = int(seg.get("engaged_no_purchase_accounts", 0) or 0)

    dominant_tier = "free"
    if tier_mix:
        dominant_tier = max(tier_mix, key=lambda x: int(tier_mix.get(x) or 0))

    confidence_score = 45.0
    if active_users >= 100:
        confidence_score += 12.0
    elif active_users >= 25:
        confidence_score += 7.0
    if platform_connected_users > 0:
        confidence_score += min(20.0, platform_coverage_pct * 0.2)
    if total_rev > 0:
        confidence_score += 10.0
    if nudge_ctr > 0:
        confidence_score += 8.0
    confidence_score = round(max(0.0, min(confidence_score, 95.0)), 1)

    offer_stack: List[Dict[str, Any]] = [
        {
            "name": "Activation-to-Upgrade Ladder",
            "target": "free_high_intent_uploaders",
            "cta": "Start Creator Lite",
            "value_prop": "Unlock higher PUT/AIC and reduce upload bottlenecks for active creators.",
            "kpi_target": "Free->paid conversion lift +15% in 30 days",
            "proof_points": {
                "cohort_size": free_high_intent,
                "current_paid_rate_pct": paid_rate,
            },
        },
        {
            "name": "Token Pressure Relief Pack",
            "target": "token_pressure_accounts",
            "cta": "Buy PUT/AIC top-up",
            "value_prop": "Prevent stalled uploads and keep AI outputs fully available.",
            "kpi_target": "Top-up conversion +12% with 7-day repeat usage",
            "proof_points": {
                "cohort_size": token_pressure,
                "topup_revenue_share_pct": rev_share_topup,
            },
        },
        {
            "name": "Enterprise Expansion Motion",
            "target": "expansion_ready_accounts",
            "cta": "Book growth architecture call",
            "value_prop": "Scale teams, multi-account workflows, and white-label operations.",
            "kpi_target": "Pipeline meetings booked from top 10% fit cohort",
            "proof_points": {
                "cohort_size": expansion_ready,
                "users_3plus_platform_connections": int(k.get("users_3plus_platform_connections", 0) or 0),
            },
        },
    ]
    if engaged_no_purchase > 0:
        offer_stack.append(
            {
                "name": "High-Intent Rescue Offer",
                "target": "engaged_no_purchase_accounts",
                "cta": "Claim 72-hour conversion offer",
                "value_prop": "Short-window incentive for users repeatedly clicking but not converting.",
                "kpi_target": "Recover 8-12% of clickers with no purchase in window",
                "proof_points": {"cohort_size": engaged_no_purchase, "current_nudge_ctr_pct": nudge_ctr},
            }
        )

    revenue_mix_lines = []
    for item in revenue_mix[:4]:
        revenue_mix_lines.append(f"{item.get('source')}: ${int(float(item.get('amount') or 0)):,}")
    revenue_mix_text = ", ".join(revenue_mix_lines) if revenue_mix_lines else "limited attribution data yet"

    newsletter_subjects = [
        f"Revenue playbook for {payload.range}: convert intent, not volume",
        f"Where your next growth wins are hiding ({dominant_tier} majority cohort)",
        f"Executive growth brief: {int(k.get('total_uploads', 0) or 0):,} uploads, ${int(total_rev):,} revenue",
        "Top signals this cycle and the exact offer cadence to deploy",
    ]
    body_outline = [
        f"North-star metrics: active users {active_users:,}, paid rate {paid_rate:.2f}%, nudge CTR {nudge_ctr:.2f}%",
        f"Cohort priorities: free high-intent {free_high_intent:,}, token pressure {token_pressure:,}, expansion-ready {expansion_ready:,}",
        f"Platform KPIs: {platform_views:,} views, {platform_engagements:,} engagements, ER {platform_er:.2f}% (coverage {platform_coverage_pct:.2f}%)",
        f"Revenue source mix: {revenue_mix_text}",
        "Channel sequence: in-app trigger -> email follow-up -> conditional offer (with suppression safeguards)",
        "Weekly experiment protocol: rotate subject + CTA by segment and keep only positive-LTV winners",
    ]

    cadence = [
        "Daily: event-triggered in-app nudges on quota pressure and feature unlock moments.",
        "Twice weekly: lifecycle newsletters split by segment with one primary CTA each.",
        "Weekly: executive review of conversion, retention, and payback by cohort.",
        "End-of-cycle: rescue promotion only for engaged users with no purchase and no recent conversion.",
    ]
    if payload.channel_mix == "email":
        cadence[0] = "Daily: behavior-triggered email flows replacing most in-app nudges for this run."
    elif payload.channel_mix == "in_app":
        cadence[1] = "Weekly: minimal email recap; keep most pressure/upsell orchestration inside product."

    suggested_campaign = {
        "name": f"Elite {payload.objective.replace('_', ' ').title()} Engine ({payload.range})",
        "objective": "Maximize conversion and expansion using cohort-specific offer sequencing with suppression.",
        "channel": payload.channel_mix if payload.channel_mix in ("in_app", "email", "discount", "mixed") else "mixed",
        "range": payload.range,
        "tiers": ["free", "creator_lite", "creator_pro", "studio"],
        "min_uploads_30d": 3 if free_high_intent > 0 else 1,
        "min_enterprise_fit_score": 55 if expansion_ready > 20 else 45,
        "min_nudge_ctr_pct": max(5, int(nudge_ctr)),
        "require_no_revenue_7d": True,
        "notes": (
            f"Tone={payload.tone}; offer_style={payload.offer_style}; objective={payload.objective}; "
            f"prioritize cohorts in order: free_high_intent -> token_pressure -> expansion_ready."
        ),
    }

    return {
        "newsletter": {
            "subject_lines": newsletter_subjects,
            "body_outline": body_outline,
        },
        "offers": offer_stack,
        "execution_plan": cadence,
        "suggested_campaign": suggested_campaign,
        "game_plan": {
            "framework": "SEGMENT -> SIGNAL -> OFFER -> CHANNEL -> MEASURE",
            "confidence_score": confidence_score,
            "north_star": {
                "paid_rate_pct": paid_rate,
                "nudge_ctr_pct": round(nudge_ctr, 2),
                "total_revenue": round(total_rev, 2),
                "platform_views": platform_views,
                "platform_engagement_rate_pct": round(platform_er, 2),
            },
            "data_quality": {
                "platform_connected_users": platform_connected_users,
                "platform_cached_users": platform_cached_users,
                "platform_coverage_pct": platform_coverage_pct,
            },
            "priority_order": [
                "Monetize active free users first",
                "Relieve token pressure second",
                "Expand multi-platform accounts third",
                "Use rescue offers only with strict suppression",
            ],
        },
    }


@app.post("/api/admin/marketing/ai/generate")
async def generate_marketing_ai_plan(payload: MarketingAIGenerateIn, user: dict = Depends(require_admin)):
    """
    AI strategist for newsletters, offers, promos, and campaign plans.
    Uses a DB-wide business snapshot and can optionally call OpenAI.
    """
    async with db_pool.acquire() as conn:
        snapshot = await _marketing_ai_snapshot(conn, payload.range)

    base_plan = _marketing_ai_fortune500_plan(payload, snapshot)

    used_openai = False
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            model = os.environ.get("MARKETING_AI_MODEL", "gpt-4o-mini")
            system_prompt = (
                "You are a Fortune-500 lifecycle marketing strategist. "
                "Return strict JSON with keys: newsletter, offers, execution_plan, suggested_campaign. "
                "Keep recommendations concrete, measurable, and ethical. "
                "Use segment-aware, testable playbooks (activation, monetization pressure relief, expansion, rescue)."
            )
            user_prompt = json.dumps({
                "objective": payload.objective,
                "tone": payload.tone,
                "offer_style": payload.offer_style,
                "channel_mix": payload.channel_mix,
                "snapshot": snapshot,
                "deterministic_playbook": base_plan,
                "constraints": {
                    "avoid_spam": True,
                    "respect_suppression": True,
                    "focus_on_revenue_and_retention": True,
                },
            }, ensure_ascii=True)
            resp = openai.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            txt = (resp.choices[0].message.content or "").strip()
            parsed = None
            try:
                parsed = json.loads(txt)
            except Exception:
                a = txt.find("{")
                b = txt.rfind("}")
                if a >= 0 and b > a:
                    parsed = json.loads(txt[a:b + 1])
            if isinstance(parsed, dict):
                base_plan = {
                    "newsletter": parsed.get("newsletter", base_plan["newsletter"]),
                    "offers": parsed.get("offers", base_plan["offers"]),
                    "execution_plan": parsed.get("execution_plan", base_plan["execution_plan"]),
                    "suggested_campaign": parsed.get("suggested_campaign", base_plan["suggested_campaign"]),
                }
                used_openai = True
        except Exception as e:
            logger.warning(f"Marketing AI generation fallback used: {e}")

    result = {
        "status": "ok",
        "used_openai": used_openai,
        "snapshot": snapshot,
        "plan": base_plan,
    }
    try:
        async with db_pool.acquire() as conn:
            await _log_ai_marketing_decision(
                conn,
                created_by=str(user["id"]),
                action="generate",
                objective=payload.objective,
                range_key=payload.range,
                used_openai=used_openai,
                status="ok",
                snapshot=snapshot,
                plan=base_plan,
                decision={"deploy_allowed": None, "force_deploy": bool(payload.force_deploy), "confidence_score": None, "blocked_reasons": []},
                campaign_id=None,
            )
    except Exception as e:
        logger.warning(f"AI marketing decision log (generate) failed: {e}")
    return result


@app.post("/api/admin/marketing/ai/deploy")
async def deploy_marketing_ai_plan(payload: MarketingAIGenerateIn, user: dict = Depends(require_admin)):
    """
    Generate AI strategist plan and immediately deploy suggested campaign draft/schedule.
    """
    generated = await generate_marketing_ai_plan(payload, user)
    snapshot = generated.get("snapshot", {}) or {}
    k = snapshot.get("kpis", {}) or {}
    seg = snapshot.get("segment_signals", {}) or {}
    reasons_blocked: List[str] = []
    if int(k.get("active_users", 0) or 0) < 20:
        reasons_blocked.append("Not enough active users for statistically useful deployment (<20).")
    if int(k.get("total_uploads", 0) or 0) < 30:
        reasons_blocked.append("Insufficient recent upload volume for reliable signal (<30).")
    if int(k.get("nudge_shown", 0) or 0) < 50:
        reasons_blocked.append("Insufficient nudge impressions for confidence (<50).")
    if (
        int(seg.get("free_high_intent_uploaders", 0) or 0) <= 0
        and int(seg.get("token_pressure_accounts", 0) or 0) <= 0
        and int(seg.get("expansion_ready_accounts", 0) or 0) <= 0
    ):
        reasons_blocked.append("No material opportunity cohort currently detected.")

    confidence = 100
    confidence -= 30 if int(k.get("nudge_shown", 0) or 0) < 50 else 0
    confidence -= 25 if int(k.get("total_uploads", 0) or 0) < 30 else 0
    confidence -= 20 if int(k.get("active_users", 0) or 0) < 20 else 0
    confidence -= 15 if int(k.get("nudge_clicked", 0) or 0) < 8 else 0
    confidence = max(0, min(confidence, 100))
    decision = {
        "deploy_allowed": (len(reasons_blocked) == 0) or bool(payload.force_deploy),
        "force_deploy": bool(payload.force_deploy),
        "confidence_score": confidence,
        "blocked_reasons": reasons_blocked,
    }
    if not decision["deploy_allowed"]:
        blocked_result = {
            "status": "blocked",
            "used_openai": generated.get("used_openai", False),
            "snapshot": snapshot,
            "plan": generated.get("plan", {}),
            "decision": decision,
        }
        try:
            async with db_pool.acquire() as conn:
                await _log_ai_marketing_decision(
                    conn,
                    created_by=str(user["id"]),
                    action="deploy",
                    objective=payload.objective,
                    range_key=payload.range,
                    used_openai=bool(generated.get("used_openai", False)),
                    status="blocked",
                    snapshot=snapshot,
                    plan=generated.get("plan", {}) or {},
                    decision=decision,
                    campaign_id=None,
                )
        except Exception as e:
            logger.warning(f"AI marketing decision log (blocked deploy) failed: {e}")
        return blocked_result
    plan = generated.get("plan") or {}
    suggested = plan.get("suggested_campaign") or {}
    campaign_payload = MarketingCampaignIn(
        name=str(suggested.get("name") or f"AI Campaign {payload.objective}"),
        objective=str(suggested.get("objective") or "AI-generated lifecycle growth campaign"),
        channel=str(suggested.get("channel") or payload.channel_mix or "mixed"),
        range=str(suggested.get("range") or payload.range or "30d"),
        tiers=list(suggested.get("tiers") or []),
        min_uploads_30d=int(suggested.get("min_uploads_30d") or 0),
        min_enterprise_fit_score=float(suggested.get("min_enterprise_fit_score") or 0),
        min_nudge_ctr_pct=float(suggested.get("min_nudge_ctr_pct") or 0),
        require_no_revenue_7d=bool(suggested.get("require_no_revenue_7d") or False),
        schedule_at=suggested.get("schedule_at"),
        notes=str(suggested.get("notes") or "Generated by AI strategist deploy"),
    )
    created = await create_marketing_campaign(campaign_payload, user)
    ok_result = {
        "status": "ok",
        "used_openai": generated.get("used_openai", False),
        "snapshot": snapshot,
        "plan": plan,
        "deployed_campaign": created.get("campaign"),
        "decision": decision,
    }
    try:
        async with db_pool.acquire() as conn:
            await _log_ai_marketing_decision(
                conn,
                created_by=str(user["id"]),
                action="deploy",
                objective=payload.objective,
                range_key=payload.range,
                used_openai=bool(generated.get("used_openai", False)),
                status="ok",
                snapshot=snapshot,
                plan=plan,
                decision=decision,
                campaign_id=str((created.get("campaign") or {}).get("id")) if (created.get("campaign") or {}).get("id") else None,
            )
    except Exception as e:
        logger.warning(f"AI marketing decision log (deploy) failed: {e}")
    return ok_result


@app.get("/api/admin/marketing/ai/decisions")
async def list_ai_marketing_decisions(limit: int = Query(25, ge=1, le=200), user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, action, objective, range_key, used_openai, deploy_allowed, forced, confidence_score,
                   status, blocked_reasons, snapshot, decision, plan, campaign_id, created_at
            FROM ai_marketing_decisions
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    return {"decisions": [dict(r) for r in rows]}


@app.get("/api/admin/marketing/ai/decisions.csv")
async def export_ai_marketing_decisions_csv(limit: int = Query(500, ge=1, le=5000), user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, action, objective, range_key, used_openai, deploy_allowed, forced, confidence_score,
                   status, blocked_reasons, campaign_id, created_at
            FROM ai_marketing_decisions
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([
        "id",
        "action",
        "objective",
        "range",
        "used_openai",
        "deploy_allowed",
        "forced",
        "confidence_score",
        "status",
        "blocked_reasons",
        "campaign_id",
        "created_at",
    ])
    for r in rows:
        d = dict(r)
        w.writerow([
            str(d.get("id") or ""),
            d.get("action") or "",
            d.get("objective") or "",
            d.get("range_key") or "",
            bool(d.get("used_openai")),
            d.get("deploy_allowed"),
            bool(d.get("forced")),
            (int(d.get("confidence_score")) if d.get("confidence_score") is not None else ""),
            d.get("status") or "",
            json.dumps(d.get("blocked_reasons") or []),
            (str(d.get("campaign_id")) if d.get("campaign_id") else ""),
            (d.get("created_at").isoformat() if d.get("created_at") else ""),
        ])
    headers = {
        "Content-Disposition": 'attachment; filename="ai_marketing_decisions.csv"',
        "Cache-Control": "no-store",
    }
    return PlainTextResponse(content=out.getvalue(), headers=headers, media_type="text/csv")


@app.get("/api/admin/marketing/ai/truth")
async def get_ai_marketing_truth_dashboard(user: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, action, objective, range_key, used_openai, deploy_allowed, forced, confidence_score,
                   status, blocked_reasons, snapshot, decision, plan, campaign_id, created_at
            FROM ai_marketing_decisions
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
    if not row:
        return {
            "last_decision": None,
            "metrics_used": {},
            "metric_sources": _ai_marketing_metric_sources(),
            "notes": "No AI decisions logged yet.",
        }
    d = dict(row)
    snap = d.get("snapshot") or {}
    ml_truth = snap.get("ml_truth") or {}
    try:
        if not isinstance(ml_truth, dict) or not ml_truth.get("top_strategies"):
            # Backfill: if older decisions were logged before ML truth was added,
            # compute it on-demand so the UI still shows something useful.
            ml_rows = await conn.fetch(
                """
                SELECT
                    strategy_key,
                    SUM(samples)::bigint AS samples,
                    CASE
                      WHEN SUM(samples) > 0
                      THEN SUM(mean_engagement * samples)::double precision / NULLIF(SUM(samples)::double precision, 0)
                      ELSE 0.0
                    END AS weighted_mean_engagement,
                    MAX(ci95_high)::double precision AS max_ci95_high,
                    COUNT(DISTINCT day)::int AS days_with_data
                FROM upload_quality_scores_daily
                WHERE day >= (CURRENT_DATE - (30::int || ' days')::interval)::date
                  AND platform = 'all'
                GROUP BY strategy_key
                ORDER BY weighted_mean_engagement DESC, samples DESC
                LIMIT 8
                """
            )
            ml_truth = {
                "top_strategies": [
                    {
                        "strategy_key": str(r.get("strategy_key") or "default"),
                        "samples": int(r.get("samples") or 0),
                        "weighted_mean_engagement": float(r.get("weighted_mean_engagement") or 0),
                        "max_ci95_high": float(r.get("max_ci95_high") or 0),
                        "days_with_data": int(r.get("days_with_data") or 0),
                    }
                    for r in (ml_rows or [])
                ]
            }
    except Exception as e:
        logger.debug(f"ml_truth backfill failed: {e}")
    metrics_used = {
        "kpis": (snap.get("kpis") or {}),
        "segment_signals": (snap.get("segment_signals") or {}),
        "tier_mix_active_users": (snap.get("tier_mix_active_users") or {}),
        "revenue_source_mix": (snap.get("revenue_source_mix") or []),
        "platform_kpis": (snap.get("platform_kpis") or []),
        "best_promo_windows_utc": (snap.get("best_promo_windows_utc") or []),
        "ml_truth": ml_truth,
    }
    return {
        "last_decision": d,
        "metrics_used": metrics_used,
        "metric_sources": _ai_marketing_metric_sources(),
        "notes": "These are the exact snapshot metrics used at decision time.",
    }


@app.get("/api/admin/leaderboard")
async def get_leaderboard(range: str = Query("30d"), sort: str = Query("uploads"), user: dict = Depends(require_admin)):
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    async with db_pool.acquire() as conn:
        if sort == "revenue":
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, COALESCE(SUM(r.amount), 0)::decimal AS revenue, COUNT(DISTINCT up.id)::int AS uploads FROM users u LEFT JOIN revenue_tracking r ON u.id = r.user_id AND r.created_at >= $1 LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1 GROUP BY u.id ORDER BY revenue DESC LIMIT 10", since)
        else:
            rows = await conn.fetch("SELECT u.id, u.name, u.email, u.subscription_tier, 0::decimal AS revenue, COUNT(up.id)::int AS uploads FROM users u LEFT JOIN uploads up ON u.id = up.user_id AND up.created_at >= $1 GROUP BY u.id ORDER BY uploads DESC LIMIT 10", since)
    return [{"id": str(r["id"]), "name": r["name"] or "Unknown", "email": r["email"], "tier": r["subscription_tier"] or "free", "uploads": r["uploads"] or 0, "revenue": float(r["revenue"] or 0), "views": 0} for r in rows]


@app.get("/api/admin/countries")
async def get_countries(range: str = Query("30d"), user: dict = Depends(require_admin)):
    """Return user count by country. Populated from CF-IPCountry header at registration."""
    minutes = _range_to_minutes(range, default_minutes=30 * 24 * 60)
    since = _now_utc() - timedelta(minutes=minutes)
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT country, COUNT(*) AS users
                FROM users
                WHERE country IS NOT NULL
                  AND created_at >= $1
                GROUP BY country
                ORDER BY users DESC
                LIMIT 50
                """,
                since,
            )
        return [{"country": r["country"], "users": int(r["users"])} for r in rows]
    except Exception:
        return []  # column may not exist yet on older deployments


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
    weekly_digest_platform_kpi_rollups: bool = True
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
        "weekly_digest_platform_kpi_rollups": settings.get("weekly_digest_platform_kpi_rollups", True),
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
        "weekly_digest_platform_kpi_rollups": settings.weekly_digest_platform_kpi_rollups,
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


@app.post("/api/admin/platform-kpi-rollups/refresh")
async def admin_refresh_platform_kpi_rollups(
    days: int = Query(7, ge=1, le=90),
    user: dict = Depends(require_admin),
):
    """Rebuild platform_kpi_rollups_daily for each UTC day in the last `days` (for ops / dashboards)."""
    now = _now_utc()
    since = now - timedelta(days=days)
    async with db_pool.acquire() as conn:
        await _refresh_platform_kpi_rollups_for_utc_range(conn, since, now)
    return {"status": "ok", "refreshed_days": days}


@app.post("/api/admin/test-webhook")
async def test_webhook(data: dict, user: dict = Depends(require_admin)):
    """Send a test message to the provided Discord webhook"""
    webhook_url = data.get("webhook_url", "").strip()
    if not webhook_url:
        raise HTTPException(400, "Webhook URL required")
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        raise HTTPException(400, "Invalid Discord webhook URL")
    test_embed = {
        "title": " UploadM8 Webhook Test",
        "description": "If you see this message, your webhook is configured correctly!",
        "color": 0x22c55e,
        "fields": [
            {"name": "Status", "value": " Connected", "inline": True},
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
        "mrr_charge": {"emoji": "", "color": 0x22c55e, "title": "MRR Charge"},
        "topup": {"emoji": "", "color": 0x8b5cf6, "title": "Top-up Purchase"},
        "upgrade": {"emoji": "⬆️", "color": 0x3b82f6, "title": "Plan Upgrade"},
        "downgrade": {"emoji": "⬇️", "color": 0xf59e0b, "title": "Plan Downgrade"},
        "cancel": {"emoji": "", "color": 0xef4444, "title": "Subscription Cancelled"},
        "refund": {"emoji": "↩️", "color": 0xf97316, "title": "Refund Processed"},
    }
    cfg = event_config.get(event_type, {"emoji": "", "color": 0x6b7280, "title": event_type.title()})
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
        embed = {"title": " Weekly Cost Report", "color": 0x3b82f6, "fields": [
            {"name": "OpenAI", "value": f"${costs.get('openai', 0):.2f}", "inline": True},
            {"name": "Storage", "value": f"${costs.get('storage', 0):.2f}", "inline": True},
            {"name": "Compute", "value": f"${costs.get('compute', 0):.2f}", "inline": True},
            {"name": "Total COGS", "value": f"${total_cost:.2f}", "inline": True},
            {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
            {"name": "Margin", "value": f"${margin:.2f} ({margin_pct:.1f}%)", "inline": True},
        ], "footer": {"text": f"UploadM8 {period} Report"}, "timestamp": _now_utc().isoformat()}
    else:
        titles = {"openai": " OpenAI Cost", "storage": " Storage Cost", "compute": " Compute Cost"}
        embed = {"title": titles.get(report_type, " Cost Report"), "color": 0xef4444, "fields": [
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
    config = {"stripe_payout": {"emoji": "", "color": 0x6366f1, "title": "Stripe Payout Coming"}, "cloud_billing": {"emoji": "️", "color": 0xf97316, "title": "Cloud Billing Reminder"}, "render_renewal": {"emoji": "", "color": 0x06b6d4, "title": "Render Renewal Reminder"}}
    cfg = config.get(reminder_type, {"emoji": "", "color": 0x6b7280, "title": "Billing Reminder"})
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
        "thumbnail_r2_key","platform_results","target_accounts",
        "processing_started_at","processing_finished_at",
        "processing_stage","processing_progress",
        "views","likes",
        "schedule_mode","schedule_metadata","timezone",
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

        # Enrich platform_results with account names/avatars from platform_tokens
        enriched_pr = await _enrich_platform_results(conn, d, str(user["id"]))

    def _normalize_hashtags(raw):
        tags = _safe_json(raw, [])
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        if isinstance(tags, str) and tags.strip():
            return [tags.strip()]
        return []

    ai_title = (d.get("ai_title") or d.get("ai_generated_title") or "") or ""
    ai_caption = (d.get("ai_caption") or d.get("ai_generated_caption") or "") or ""
    ai_hashtags = _normalize_hashtags(d.get("ai_generated_hashtags"))

    title = (d.get("title") or "").strip() or ai_title
    caption = (d.get("caption") or "").strip() or ai_caption
    hashtags = _normalize_hashtags(d.get("hashtags"))
    platform_results = enriched_pr

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
        "schedule_mode": d.get("schedule_mode") or "immediate",
        "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
        "smart_schedule": _safe_json(d.get("schedule_metadata"), None),  # alias for queue.html edit modal
        "timezone": d.get("timezone") or "UTC",
        "is_editable": d.get("status") in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
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
    if not _valid_uuid(user_id):
        raise HTTPException(400, "Invalid user ID")
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
        "ledger": [
            {
                "token_type": r.get("token_type"),
                "delta": int(r.get("delta", 0)),
                "reason": r.get("reason"),
                "upload_id": str(r["upload_id"]) if r.get("upload_id") else None,
                "meta": r.get("meta"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            for r in ledger
        ],
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
    if not _valid_uuid(user_id):
        raise HTTPException(400, "Invalid user ID")
    async with db_pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT id, email, name FROM users WHERE id = $1", user_id
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

        meta_json = json.dumps({
            "ref_type": "admin_adjust",
            "admin_id": str(admin.get("id", "")),
            "admin_email": admin.get("email") or "",
            "mode": payload.mode,
            "before": before,
            "after": new_val,
            "reason": payload.reason,
        })
        await conn.execute(
            """
            INSERT INTO token_ledger (user_id, token_type, delta, reason, meta)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            target["id"],
            payload.wallet,
            int(delta),
            f"admin_{payload.mode}_{payload.wallet}",
            meta_json,
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

    # Send notification email to user for grants (add/set with positive delta only)
    if delta > 0 and payload.mode in ("add", "set"):
        import asyncio as _aio
        _aio.ensure_future(send_admin_wallet_topup_email(
            target["email"],
            target["name"] or "there",
            payload.wallet,
            int(delta),
            new_val,
            payload.reason or "Tokens credited to your account by the UploadM8 team.",
        ))

    return {
        "ok":     True,
        "wallet": payload.wallet,
        "mode":   payload.mode,
        "before": before,
        "after":  new_val,
        "delta":  int(delta),
    }


# ════════════════════════════════════════════════════════════════
# ENTERPRISE: API KEY MANAGEMENT
# ════════════════════════════════════════════════════════════════

@app.get("/api/keys")
async def list_api_keys(user: dict = Depends(get_current_user)):
    """List all API keys for the current user."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, key_prefix, name, scopes, rate_limit,
                      last_used_at, expires_at, created_at
               FROM api_keys
               WHERE user_id = $1 AND revoked_at IS NULL
               ORDER BY created_at DESC""",
            user["id"],
        )
    return [
        {
            "id": str(r["id"]),
            "key_prefix": r["key_prefix"],
            "name": r["name"],
            "scopes": list(r["scopes"] or []),
            "rate_limit": r["rate_limit"],
            "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
            "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


class CreateApiKeyRequest(BaseModel):
    name: str = Field("Default", max_length=255)
    scopes: List[str] = Field(default_factory=lambda: ["read"])
    expires_days: Optional[int] = Field(None, ge=1, le=365)


@app.post("/api/keys")
async def create_api_key(data: CreateApiKeyRequest, request: Request, user: dict = Depends(get_current_user)):
    """Create a new API key. Returns the raw key only once."""
    raw_key = f"um8_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:12]
    expires_at = (_now_utc() + timedelta(days=data.expires_days)) if data.expires_days else None

    async with db_pool.acquire() as conn:
        key_count = await conn.fetchval(
            "SELECT COUNT(*) FROM api_keys WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )
        if key_count >= 10:
            raise HTTPException(400, "Maximum 10 active API keys allowed")

        key_id = str(uuid.uuid4())
        await conn.execute(
            """INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, scopes, expires_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            key_id, user["id"], key_hash, key_prefix, data.name, data.scopes, expires_at,
        )

    await audit_log(
        user["id"], "API_KEY_CREATED",
        event_category="AUTH", resource_type="api_key", resource_id=key_id,
        details={"name": data.name, "scopes": data.scopes},
        ip_address=client_ip(request),
    )

    return {"id": key_id, "key": raw_key, "prefix": key_prefix, "name": data.name}


@app.delete("/api/keys/{key_id}")
async def revoke_api_key(key_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Revoke an API key."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchval(
            "UPDATE api_keys SET revoked_at = NOW() WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL RETURNING id",
            key_id, user["id"],
        )
    if not result:
        raise HTTPException(404, "API key not found")

    await audit_log(
        user["id"], "API_KEY_REVOKED",
        event_category="AUTH", resource_type="api_key", resource_id=key_id,
        ip_address=client_ip(request),
    )
    return {"status": "revoked", "id": key_id}


# ════════════════════════════════════════════════════════════════
# ENTERPRISE: AUDIT LOG EXPORT (CSV)
# ════════════════════════════════════════════════════════════════

@app.get("/api/admin/audit/export")
async def export_audit_csv(
    days: int = Query(30, ge=1, le=365),
    event_category: Optional[str] = None,
    user: dict = Depends(require_admin),
):
    """Export audit logs as CSV for compliance/enterprise customers."""
    async with db_pool.acquire() as conn:
        q = """
            SELECT
                created_at, event_category, action,
                COALESCE(actor_user_id::text, admin_id::text) AS actor_id,
                admin_email AS actor_email,
                user_id::text AS target_user_id,
                resource_type, resource_id,
                severity, outcome,
                ip_address, details::text
            FROM admin_audit_log
            WHERE created_at >= NOW() - ($1 || ' days')::interval
        """
        args: list = [str(days)]
        if event_category:
            args.append(event_category.upper())
            q += f" AND UPPER(COALESCE(event_category,'ADMIN')) = ${len(args)}"
        q += " ORDER BY created_at DESC LIMIT 10000"

        rows = await conn.fetch(q, *args)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "category", "action", "actor_id", "actor_email",
        "target_user_id", "resource_type", "resource_id",
        "severity", "outcome", "ip_address", "details",
    ])
    for r in rows:
        writer.writerow([
            r["created_at"].isoformat() if r["created_at"] else "",
            r["event_category"], r["action"],
            r["actor_id"] or "", r["actor_email"] or "",
            r["target_user_id"] or "",
            r["resource_type"] or "", r["resource_id"] or "",
            r["severity"], r["outcome"],
            r["ip_address"] or "", r["details"] or "",
        ])

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=audit-log-{days}d.csv"},
    )


# ════════════════════════════════════════════════════════════════
# ENTERPRISE: DEAD LETTER QUEUE ADMIN
# ════════════════════════════════════════════════════════════════

@app.get("/api/admin/dead-letter")
async def list_dead_letters(
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(require_admin),
):
    """List unresolved dead-letter queue items."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, upload_id, user_id, error_code, error_message,
                      retry_count, max_retries, last_attempt_at, created_at
               FROM dead_letter_queue
               WHERE resolved_at IS NULL
               ORDER BY created_at DESC
               LIMIT $1""",
            limit,
        )
    return [
        {k: (str(v) if isinstance(v, uuid.UUID) else v.isoformat() if hasattr(v, 'isoformat') else v)
         for k, v in dict(r).items()}
        for r in rows
    ]


@app.post("/api/admin/dead-letter/{dlq_id}/retry")
async def retry_dead_letter(dlq_id: str, request: Request, user: dict = Depends(require_admin)):
    """Re-enqueue a dead-letter item for processing."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM dead_letter_queue WHERE id = $1 AND resolved_at IS NULL",
            dlq_id,
        )
        if not row:
            raise HTTPException(404, "Dead letter not found or already resolved")

        job_data = row["job_data"] if isinstance(row["job_data"], dict) else json.loads(row["job_data"])

        if redis_client:
            await redis_client.lpush(PROCESS_NORMAL_QUEUE, json.dumps(job_data))

        await conn.execute(
            "UPDATE dead_letter_queue SET resolved_at = NOW(), resolved_by = 'admin_retry' WHERE id = $1",
            dlq_id,
        )

    await audit_log(
        user["id"], "DLQ_RETRY",
        event_category="ADMIN", resource_type="dead_letter", resource_id=dlq_id,
        ip_address=client_ip(request),
    )
    return {"status": "re-queued", "id": dlq_id}


@app.post("/api/admin/dead-letter/{dlq_id}/resolve")
async def resolve_dead_letter(dlq_id: str, request: Request, user: dict = Depends(require_admin)):
    """Mark a dead-letter item as manually resolved (discard)."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchval(
            "UPDATE dead_letter_queue SET resolved_at = NOW(), resolved_by = 'admin_discard' WHERE id = $1 AND resolved_at IS NULL RETURNING id",
            dlq_id,
        )
    if not result:
        raise HTTPException(404, "Dead letter not found or already resolved")

    await audit_log(
        user["id"], "DLQ_RESOLVED",
        event_category="ADMIN", resource_type="dead_letter", resource_id=dlq_id,
        ip_address=client_ip(request),
    )
    return {"status": "resolved", "id": dlq_id}


# ============================================================
# Cloudflare Web Analytics (beacon.min.js) — local / direct-origin
# On Cloudflare-proxied traffic, the edge handles POST /cdn-cgi/rum. Without the
# edge (127.0.0.1, direct to origin), the beacon posts same-origin → StaticFiles
# returned 501. Accept and discard so DevTools stays clean.
# ============================================================
@app.post("/cdn-cgi/rum")
async def _cf_rum_beacon_sink() -> Response:
    return Response(status_code=204)


# ============================================================
# Frontend static (local dev & single-process deploys)
# Registered last so /api/* and /docs keep precedence.
# ============================================================
_frontend_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(_frontend_static_dir):
    try:
        app.mount("/", StaticFiles(directory=_frontend_static_dir, html=True), name="frontend")
    except Exception as _fe_mount_err:
        logger.warning("Frontend static mount skipped: %s", _fe_mount_err)
