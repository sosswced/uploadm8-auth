"""
UploadM8 shared configuration — env vars, logging, constants.
Imported by routers and core modules; no mutable runtime state lives here.
"""

import os
from pathlib import Path

# Load .env before any config reads (needed for local dev when running via uvicorn)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging

# ============================================================
# Logging
# ============================================================
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
# Repo `frontend/` — optional static mount on the API app (same origin as /api/*).
_REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_STATIC_DIR = _REPO_ROOT / "frontend"
_serve_fe = os.environ.get("SERVE_FRONTEND", "1").strip().lower()
SERVE_FRONTEND_STATIC = _serve_fe not in ("0", "false", "no", "off")
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("Missing JWT_SECRET env var")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.uploadm8.com")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "uploadm8-app")
ACCESS_TOKEN_MINUTES = int(os.environ.get("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.environ.get("REFRESH_TOKEN_DAYS", "30"))
TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")

# HttpOnly auth cookies (see core.cookie_auth). Bearer header still works for API / cross-host dev.
AUTH_ACCESS_COOKIE = os.environ.get("AUTH_ACCESS_COOKIE", "uploadm8_access")
AUTH_REFRESH_COOKIE = os.environ.get("AUTH_REFRESH_COOKIE", "uploadm8_refresh")
AUTH_COOKIE_PATH = (os.environ.get("AUTH_COOKIE_PATH", "/") or "/").strip() or "/"
_cd = os.environ.get("AUTH_COOKIE_DOMAIN", "").strip()
AUTH_COOKIE_DOMAIN = _cd if _cd else None  # e.g. .uploadm8.com for all subdomains
_raw_cookie_secure = os.environ.get("AUTH_COOKIE_SECURE")
if _raw_cookie_secure is None:
    AUTH_COOKIE_SECURE = str(BASE_URL).lower().startswith("https://")
else:
    AUTH_COOKIE_SECURE = _raw_cookie_secure.strip().lower() in ("1", "true", "yes", "on")
AUTH_COOKIE_SAMESITE = (os.environ.get("AUTH_COOKIE_SAMESITE", "lax") or "lax").strip().lower()
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://app.uploadm8.com,https://uploadm8.com,"
    "http://localhost:3000,http://localhost:8080,"
    "http://localhost:8000,http://localhost:8001,http://localhost:8002,"
    "http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002",
)
ALLOWED_ORIGINS_LIST = [x.strip() for x in ALLOWED_ORIGINS.split(",") if x.strip()]
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()
# Ops alerts: upload failures, worker errors, user bug reports (Mailgun → this inbox)
ADMIN_OPS_EMAIL = (os.environ.get("ADMIN_OPS_EMAIL") or os.environ.get("BOOTSTRAP_ADMIN_EMAIL") or "").strip()
TRUST_PROXY_HEADERS = os.environ.get("TRUST_PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes", "on")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# Optional signup email confirmation (stores rows in signup_verifications).
SIGNUP_EMAIL_VERIFICATION = _env_truthy("SIGNUP_EMAIL_VERIFICATION")
# Legacy env: unverified users (email_verified=false) are always blocked at login, refresh,
# and get_current_user — no separate flag required.
REQUIRE_VERIFIED_EMAIL = _env_truthy("REQUIRE_VERIFIED_EMAIL")

# R2/S3
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")
# Presigned upload URL TTL (seconds). Default 2h for large/slow uploads; increase if users hit "R2 upload network error" due to expiry.
R2_PRESIGN_UPLOAD_TTL = int(os.environ.get("R2_PRESIGN_UPLOAD_TTL", "7200"))
# When true, presigned PUT does not bind Content-Type in the signature. Use if frontend sends file.type that differs from presign
# (empty or browser-specific) and R2 returns 403 — avoids "network error" from failed PUT. Object may need Content-Type set later if required.
R2_PRESIGN_PUT_UNSIGNED_CONTENT = os.environ.get("R2_PRESIGN_PUT_UNSIGNED_CONTENT", "").strip().lower() in ("1", "true", "yes", "on")

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "")

# HTTP rate limiting (see core/security.py). Keys in Redis are
# "{RATE_LIMIT_KEY_PREFIX}:ip:<addr>:<surface>". When several dev machines
# point at the same Redis, they all share 127.0.0.1 unless you set a unique
# RATE_LIMIT_KEY_PREFIX per machine or workspace.
RATE_LIMIT_KEY_PREFIX = (
    os.environ.get("RATE_LIMIT_KEY_PREFIX", "uploadm8:rl").strip().rstrip(":") or "uploadm8:rl"
)
RATE_LIMIT_DISABLED = os.environ.get("RATE_LIMIT_DISABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RATE_LIMIT_GLOBAL_PER_MIN = int(os.environ.get("RATE_LIMIT_GLOBAL_PER_MIN", "300"))
RATE_LIMIT_AUTH_PER_MIN = int(os.environ.get("RATE_LIMIT_AUTH_PER_MIN", "30"))
RATE_LIMIT_ADMIN_PER_MIN = int(os.environ.get("RATE_LIMIT_ADMIN_PER_MIN", "60"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
# Optional disambiguator when many devs share one Redis (e.g. set to %COMPUTERNAME%).
RATE_LIMIT_INSTANCE_ID = os.environ.get("RATE_LIMIT_INSTANCE_ID", "").strip()
# When true (default), requests whose client IP is loopback (::1 / 127.0.0.1) skip HTTP rate limits.
# Turn off (RATE_LIMIT_LOOPBACK_BYPASS=false) to test limits locally, or if a reverse proxy on the
# same host connects to Uvicorn without forwarding the real client IP (everyone would look loopback).
RATE_LIMIT_LOOPBACK_BYPASS = os.environ.get("RATE_LIMIT_LOOPBACK_BYPASS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

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
- Titles: 1-2 emojis max (fire, lightning, eyes)
- Captions: 2-3 emojis strategically placed
- Never excessive or spammy

HASHTAG STRATEGY:
- Mix viral mega-tags (#fyp, #viral, #trending)
- Niche community tags (#spiriteddrive, #roadtrip)
- Location-based tags (#Utah, #Moab)
- Motion tags when relevant (#curvyroads, #switchbacks)
- Protected lands tags ONLY when verified
"""

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

# OAuth credentials from environment
# TikTok
TIKTOK_CLIENT_KEY    = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
# Separate secret used to verify TikTok webhook payloads (HMAC-SHA256).
# Set this to the value shown in TikTok Developer Portal -> your app ->
# Webhooks -> "Client Secret". Falls back to TIKTOK_CLIENT_SECRET if
# you haven't configured a separate one yet (the common starting point).
TIKTOK_WEBHOOK_SECRET = (
    os.environ.get("TIKTOK_WEBHOOK_SECRET", "")
    or TIKTOK_CLIENT_SECRET
)
# YouTube/Google
YOUTUBE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "") or os.environ.get("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "") or os.environ.get("YOUTUBE_CLIENT_SECRET", "")
# Meta (Facebook & Instagram have SEPARATE credentials)
META_APP_ID = os.environ.get("META_APP_ID", "")
META_APP_SECRET = os.environ.get("META_APP_SECRET", "")
# Instagram Graph API authenticates via Facebook OAuth (uses META credentials)
INSTAGRAM_CLIENT_ID = META_APP_ID
INSTAGRAM_CLIENT_SECRET = META_APP_SECRET
# Facebook uses the main Meta App ID
FACEBOOK_CLIENT_ID = os.environ.get("FACEBOOK_CLIENT_ID", "") or META_APP_ID
FACEBOOK_CLIENT_SECRET = os.environ.get("FACEBOOK_CLIENT_SECRET", "") or META_APP_SECRET
