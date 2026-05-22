#!/usr/bin/env python3
"""Verify production env vars and service flags (values never printed). Exit 0 if OK."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=True)
except ImportError:
    pass

REQUIRED = ("JWT_SECRET", "DATABASE_URL")
PROD_RECOMMENDED = (
    "REDIS_URL",
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET_NAME",
    "BASE_URL",
    "FRONTEND_URL",
    "ALLOWED_ORIGINS",
    "TOKEN_ENC_KEYS",
)
OAUTH_ANY = (
    ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"),
    ("META_APP_ID", "META_APP_SECRET"),
    ("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET"),
)
PLACEHOLDERS = frozenset({"change-me-min-32-chars-random", "change-me"})
DISABLED = frozenset({"0", "false", "no", "off", "disable", "disabled", "2"})

# name -> (expect_active, description)
SERVICE_FLAGS = (
    ("SERVE_FRONTEND", True, "static UI mount"),
    ("UPLOADM8_HYDRATION_PAYLOAD", True, "worker hydration snapshot"),
    ("RATE_LIMIT_ENABLED", True, "HTTP rate limiting"),
    ("TRUST_PROXY_HEADERS", True, "reverse-proxy client IP"),
    ("WORKER_LEADER_LOCK", True, "single-replica worker loops"),
    ("WORKER_ENABLE_SCHEDULER", True, "scheduled publish loop"),
    ("WORKER_ENABLE_ANALYTICS_SYNC", True, "analytics sync loop"),
    ("WORKER_ENABLE_KPI_COLLECTOR", True, "KPI collector loop"),
    ("WORKER_ENABLE_ML_SCORING", True, "ML scoring loop"),
    ("STALE_JOB_RECOVERY_ENABLED", True, "stale job recovery"),
    ("JSON_LOGS", True, "structured JSON logs"),
    ("MARKETING_AUTOMATION_ENABLED", True, "marketing automation"),
    ("M8_ENGINE_ENABLED", True, "M8 caption engine"),
    ("AUDIO_STAGE_ENABLED", True, "audio pipeline stage"),
    ("VISION_STAGE_ENABLED", True, "vision pipeline stage"),
    ("VIDEO_INTELLIGENCE_ENABLED", True, "video intelligence stage"),
    ("YAMNET_ENABLED", True, "YAMNet audio labels"),
    ("TWELVELABS_ENABLED", True, "Twelve Labs stage"),
)


def _set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _status(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        return "MISSING"
    if name == "JWT_SECRET" and v in PLACEHOLDERS:
        return "PLACEHOLDER"
    if name == "DATABASE_URL" and ("localhost" in v or "127.0.0.1" in v):
        return "SET (local)"
    if name in ("BASE_URL", "FRONTEND_URL") and ("127.0.0.1" in v or "localhost" in v):
        return "SET (local)"
    return "SET"


def _flag_active(name: str, default_active: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        if name == "RATE_LIMIT_ENABLED":
            disabled = os.environ.get("RATE_LIMIT_DISABLED", "").strip().lower()
            if disabled in ("1", "true", "yes", "on"):
                return False
            return True
        return default_active
    v = str(raw).strip().lower()
    if name == "RATE_LIMIT_ENABLED":
        if v in DISABLED:
            return False
        return True
    if default_active:
        return v not in DISABLED
    return v in ("1", "true", "yes", "on")


async def _check_runtime() -> list[str]:
    issues: list[str] = []
    try:
        import asyncpg
        from core.config import DATABASE_URL

        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1, timeout=15)
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        await pool.close()
        print("  [runtime] PostgreSQL: OK")
    except Exception as e:
        issues.append(f"PostgreSQL: {type(e).__name__}")
        print(f"  [runtime] PostgreSQL: FAIL ({type(e).__name__})")

    try:
        import redis.asyncio as aioredis
        from core.config import REDIS_URL

        if not REDIS_URL:
            issues.append("REDIS_URL empty")
            print("  [runtime] Redis: MISSING URL")
        else:
            client = aioredis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
            await client.ping()
            await client.aclose()
            print("  [runtime] Redis: OK")
    except Exception as e:
        issues.append(f"Redis: {type(e).__name__}")
        print(f"  [runtime] Redis: FAIL ({type(e).__name__})")

    return issues


def main() -> int:
    check_runtime = "--check-runtime" in sys.argv
    errors: list[str] = []
    warnings: list[str] = []

    print("UploadM8 production env check (values not shown)\n")

    for k in REQUIRED:
        st = _status(k)
        print(f"  [required] {k}: {st}")
        if st in ("MISSING", "PLACEHOLDER"):
            errors.append(k)

    for k in PROD_RECOMMENDED:
        st = _status(k)
        print(f"  [recommended] {k}: {st}")
        if st == "MISSING":
            warnings.append(k)

    oauth_ok = any(_set(a) and _set(b) for a, b in OAUTH_ANY)
    print(f"  [oauth] at least one platform pair: {'OK' if oauth_ok else 'MISSING'}")
    if not oauth_ok:
        warnings.append("oauth (no complete platform pair)")

    for k in (
        "AUTH_COOKIE_SECURE",
        "AUTH_COOKIE_DOMAIN",
        "BILLING_MODE",
        "STRIPE_SECRET_KEY",
        "SENTRY_DSN",
        "MAILGUN_API_KEY",
        "OPENAI_API_KEY",
    ):
        print(f"  [prod] {k}: {_status(k)}")

    try:
        from core import config

        print(f"  [resolved] AUTH_COOKIE_SECURE: {config.AUTH_COOKIE_SECURE}")
        print(f"  [resolved] SERVE_FRONTEND_STATIC: {config.SERVE_FRONTEND_STATIC}")
        print(f"  [resolved] RATE_LIMIT_DISABLED: {config.RATE_LIMIT_DISABLED}")
        print(f"  [resolved] BILLING_MODE: {config.BILLING_MODE}")
        if config.BILLING_MODE == "test" and _set("STRIPE_SECRET_KEY"):
            sk = os.environ.get("STRIPE_SECRET_KEY", "")
            if sk.startswith("sk_live_"):
                warnings.append("BILLING_MODE=test but Stripe key is live")
    except Exception as e:
        warnings.append(f"config import: {type(e).__name__}")

    print("\n  Service flags:")
    for name, expect, desc in SERVICE_FLAGS:
        active = _flag_active(name, expect)
        mark = "ON" if active else "OFF"
        print(f"    {name}: {mark} ({desc})")
        if expect and not active:
            warnings.append(f"{name} is OFF")

    base = os.environ.get("BASE_URL", "")
    fe = os.environ.get("FRONTEND_URL", "")
    prod_urls = base.startswith("https://auth.uploadm8.com") and fe.startswith(
        "https://app.uploadm8.com"
    )
    print(f"\n  Prod URLs (auth + app): {'OK' if prod_urls else 'LOCAL/DEV'}")

    if check_runtime:
        print("\n  Runtime connectivity:")
        runtime_issues = asyncio.run(_check_runtime())
        warnings.extend(runtime_issues)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nErrors (fix before deploy):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nRequired vars OK.")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
