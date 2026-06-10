"""
Admin KPI finance helpers: windowed aggregates from ``cost_tracking`` plus optional
live reads (Stripe fees, Cloudflare R2, Upstash) for ``/api/admin/kpi/provider-costs``
and tool estimates for ``/api/admin/kpi/cost-tracker``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("uploadm8.admin_kpi_finance")

_PROVIDER_COSTS_CACHE_TTL_S = 90.0
_provider_costs_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# cost_tracking.category → admin KPI provider bucket
_GOOGLE_CATEGORIES = frozenset({
    "google_cloud", "google", "gcp", "vision_google", "video_intelligence",
    "google_vision", "google_speech", "dashcam_osd", "google_video_intelligence",
})
_PIKZELS_CATEGORIES = frozenset({"pikzels", "pikzels_v2", "thumbnail_pikzels"})
_TWELVELABS_CATEGORIES = frozenset({"twelvelabs", "twelve_labs"})
_SERPAPI_CATEGORIES = frozenset({"serpapi", "trend_intel", "serp", "serp_api"})
_OPENAI_CATEGORIES = frozenset({"openai", "openai_whisper", "audio_whisper", "whisper"})
_INFRA_CATEGORIES = frozenset({
    "storage", "bandwidth", "compute", "stripe_fees", "mailgun", "postgres", "redis",
})


def provider_bucket_for_category(category: str) -> str:
    cat = (category or "").strip().lower()
    if cat in _OPENAI_CATEGORIES:
        return "openai"
    if cat in _GOOGLE_CATEGORIES:
        return "google_cloud"
    if cat in _PIKZELS_CATEGORIES:
        return "pikzels"
    if cat in _TWELVELABS_CATEGORIES:
        return "twelvelabs"
    if cat in _SERPAPI_CATEGORIES:
        return "serpapi"
    if cat == "stripe_fees":
        return "stripe"
    if cat == "mailgun":
        return "mailgun"
    if cat == "storage":
        return "cloudflare_storage"
    if cat == "bandwidth":
        return "cloudflare_bandwidth"
    if cat == "compute":
        return "render_compute"
    if cat == "postgres":
        return "postgres"
    if cat == "redis":
        return "redis_upstash"
    return "other"


_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "google_cloud": "Google Cloud (Vision / Video Intelligence)",
    "pikzels": "Pikzels",
    "twelvelabs": "TwelveLabs",
    "serpapi": "SerpAPI / trend intel",
    "stripe": "Stripe fees",
    "mailgun": "Mailgun",
    "cloudflare_storage": "Cloudflare R2 storage",
    "cloudflare_bandwidth": "Cloudflare bandwidth",
    "render_compute": "Render compute",
    "postgres": "Postgres (Render)",
    "redis_upstash": "Upstash Redis",
    "other": "Other providers",
}

_RANGE_PRESETS_MINUTES = {
    "24h": 24 * 60,
    "7d": 7 * 24 * 60,
    "30d": 30 * 24 * 60,
    "90d": 90 * 24 * 60,
    "6m": 180 * 24 * 60,
    "1y": 365 * 24 * 60,
    "365d": 365 * 24 * 60,
}


def range_key_to_minutes(range_key: str | None, default_minutes: int = 30 * 24 * 60) -> int:
    r = (range_key or "").strip()
    if not r:
        return default_minutes
    if r in _RANGE_PRESETS_MINUTES:
        return _RANGE_PRESETS_MINUTES[r]
    m = re.fullmatch(r"(\d{1,4})d", r)
    if m:
        days = max(1, min(int(m.group(1)), 3650))
        return days * 24 * 60
    return default_minutes


def window_from_range_key(range_key: str | None) -> Tuple[datetime, datetime]:
    until = datetime.now(timezone.utc)
    mins = range_key_to_minutes(range_key)
    return until - timedelta(minutes=mins), until


def _prorate_monthly_cost(monthly: float, since: datetime, until: datetime) -> float:
    if monthly <= 0 or not math.isfinite(monthly):
        return 0.0
    delta = max(0.0, (until - since).total_seconds())
    month_sec = 30 * 24 * 3600
    return round(float(monthly) * (delta / month_sec), 4)


def _stripe_balance_fees_window_sync(since: datetime, until: datetime) -> Tuple[float, int]:
    key = (os.environ.get("STRIPE_SECRET_KEY") or "").strip()
    if not key:
        return 0.0, 0
    try:
        import stripe

        stripe.api_key = key
        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())
        total = 0.0
        n = 0
        for t in stripe.BalanceTransaction.list(
            created={"gte": since_ts, "lte": until_ts},
            limit=100,
        ).auto_paging_iter():
            ct = int(getattr(t, "created", 0) or 0)
            if ct > until_ts:
                continue
            fee = getattr(t, "fee", None) or 0
            total += float(fee) / 100.0
            n += 1
        return round(total, 4), n
    except Exception as e:
        logger.debug("stripe balance fees window: %s", e)
        return 0.0, 0


def _stripe_refunds_window_sync(since: datetime, until: datetime) -> Tuple[float, int]:
    key = (os.environ.get("STRIPE_SECRET_KEY") or "").strip()
    if not key:
        return 0.0, 0
    try:
        import stripe

        stripe.api_key = key
        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())
        total = 0.0
        n = 0
        for r in stripe.Refund.list(
            created={"gte": since_ts, "lte": until_ts},
            limit=100,
        ).auto_paging_iter():
            ct = int(getattr(r, "created", 0) or 0)
            if ct > until_ts:
                continue
            amt = getattr(r, "amount", None) or 0
            total += float(amt) / 100.0
            n += 1
        return round(total, 2), n
    except Exception as e:
        logger.debug("stripe refunds window: %s", e)
        return 0.0, 0


async def fetch_stripe_fees_window(since: datetime, until: datetime) -> Tuple[float, int]:
    return await asyncio.to_thread(_stripe_balance_fees_window_sync, since, until)


async def fetch_stripe_refunds_window(since: datetime, until: datetime) -> Tuple[float, int]:
    return await asyncio.to_thread(_stripe_refunds_window_sync, since, until)


async def _upstash_memory_mb() -> float:
    api_key = (os.environ.get("UPSTASH_API_KEY") or "").strip()
    db_id = (os.environ.get("UPSTASH_DB_ID") or "").strip()
    if not api_key:
        return 0.0
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://api.upstash.com/v2/database/{db_id}/stats" if db_id else "https://api.upstash.com/v2/databases"
            r = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code != 200:
                return 0.0
            data = r.json()
            mb = 0.0
            if isinstance(data, list):
                for db in data:
                    mb += float(db.get("memory_usage_mb", 0) or 0)
            elif isinstance(data, dict):
                mb = float(data.get("memory_usage_mb", 0) or 0)
            return mb
    except Exception as e:
        logger.debug("upstash memory: %s", e)
        return 0.0


async def _fetch_openai_usage_window(since: datetime, until: datetime) -> float:
    """Best-effort OpenAI org usage for the window (0 when Usage API unavailable)."""
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return 0.0
    try:
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                "https://api.openai.com/v1/organization/usage/completions",
                headers={"Authorization": f"Bearer {key}"},
                params={
                    "start_time": int(since.timestamp()),
                    "end_time": int(until.timestamp()),
                    "bucket_width": "1d",
                    "limit": 100,
                },
            )
            if r.status_code != 200:
                return 0.0
            data = r.json()
            total = 0.0
            for bucket in data.get("data", data.get("usage", [])) or []:
                if isinstance(bucket, dict):
                    total += float(bucket.get("cost_usd", bucket.get("cost", 0)) or 0)
            return round(total, 6)
    except Exception as e:
        logger.debug("openai usage window: %s", e)
        return 0.0


async def _estimate_pikzels_cost(conn, since: datetime, until: datetime) -> Tuple[float, int]:
    """Estimate Pikzels spend from studio_usage_events when no cost_tracking rows exist."""
    per_call = float(os.environ.get("KPI_PIKZELS_USD_PER_CALL", "0") or 0)
    if per_call <= 0:
        return 0.0, 0
    try:
        n = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM studio_usage_events
            WHERE created_at >= $1 AND created_at < $2
              AND (http_status IS NULL OR http_status < 400)
            """,
            since,
            until,
        )
        calls = int(n or 0)
        return round(calls * per_call, 4), calls
    except Exception as e:
        logger.debug("pikzels usage estimate: %s", e)
        return 0.0, 0


async def fetch_cost_categories_window(
    conn, since: datetime, until: datetime,
) -> Dict[str, Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT category,
               COALESCE(SUM(cost_usd), 0)::double precision AS sum_cost,
               COUNT(*)::int AS n_rows,
               COALESCE(SUM(COALESCE(tokens, 0)), 0)::bigint AS token_sum
        FROM cost_tracking
        WHERE created_at >= $1 AND created_at < $2
        GROUP BY category
        """,
        since,
        until,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cat = str(row["category"] or "").lower()
        out[cat] = {
            "sum_cost": float(row["sum_cost"] or 0),
            "n_rows": int(row["n_rows"] or 0),
            "token_sum": int(row["token_sum"] or 0),
        }
    return out


def _bucket_totals(by_cat: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, float] = {}
    for cat, meta in by_cat.items():
        b = provider_bucket_for_category(cat)
        buckets[b] = buckets.get(b, 0.0) + float(meta.get("sum_cost") or 0)
    return buckets


def _monthly_env_prorates(since: datetime, until: datetime) -> Dict[str, float]:
    """Prorated monthly vendor estimates from env when billing APIs are unavailable."""
    mapping = {
        "google_cloud": "KPI_GOOGLE_CLOUD_MONTHLY_COST",
        "pikzels": "KPI_PIKZELS_MONTHLY_COST",
        "twelvelabs": "KPI_TWELVELABS_MONTHLY_COST",
        "serpapi": "KPI_SERPAPI_MONTHLY_COST",
        "render_compute": "RENDER_MONTHLY_COST",
    }
    out: Dict[str, float] = {}
    for bucket, env_key in mapping.items():
        monthly = float(os.environ.get(env_key, "0") or 0)
        if monthly > 0:
            out[bucket] = _prorate_monthly_cost(monthly, since, until)
    return out


async def build_admin_costs_summary(
    conn,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    range_key: str | None = None,
) -> Dict[str, Any]:
    """
    Unified admin cost rollup: DB cost_tracking + live provider reads + env prorates.
    Used by GET /api/admin/kpis, /api/admin/kpi/costs, and /api/admin/kpi/provider-costs.
    """
    if since is None or until is None:
        win_since, win_until = window_from_range_key(range_key)
        since = since if since is not None else win_since
        until = until if until is not None else win_until

    provider_payload = await build_provider_costs_payload(conn, range_key=range_key or "30d")

    openai_cost = max(
        float(provider_payload.get("openai_cost_window") or 0),
        float(provider_payload.get("openai_cost_live") or 0),
    )
    google_cloud_cost = float(provider_payload.get("google_cloud_cost") or 0)
    pikzels_cost = float(provider_payload.get("pikzels_cost") or 0)
    twelvelabs_cost = float(provider_payload.get("twelvelabs_cost") or 0)
    serpapi_cost = float(provider_payload.get("serpapi_cost") or 0)
    other_provider_cost = float(provider_payload.get("other_provider_cost") or 0)

    storage_cost = float(provider_payload.get("storage_cost") or 0)
    bandwidth_cost = float(provider_payload.get("bandwidth_cost") or 0)
    compute_cost = float(provider_payload.get("render_cost") or 0)
    stripe_fees = float(provider_payload.get("stripe_fees") or 0)
    mailgun_cost = float(provider_payload.get("mailgun_cost") or 0)
    postgres_cost = float(provider_payload.get("postgres_cost") or 0)
    redis_cost = float(provider_payload.get("redis_cost") or 0)

    total_costs = round(
        openai_cost + google_cloud_cost + pikzels_cost + twelvelabs_cost + serpapi_cost
        + other_provider_cost + storage_cost + bandwidth_cost + compute_cost
        + stripe_fees + mailgun_cost + postgres_cost + redis_cost,
        4,
    )

    uploads = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM uploads
        WHERE status IN ('completed','succeeded') AND created_at >= $1 AND created_at < $2
        """,
        since,
        until,
    )
    uploads = int(uploads or 0)

    return {
        "openai_cost": round(openai_cost, 4),
        "openai_calls": int(provider_payload.get("openai_api_rows") or 0),
        "google_cloud_cost": round(google_cloud_cost, 4),
        "pikzels_cost": round(pikzels_cost, 4),
        "pikzels_calls": int(provider_payload.get("pikzels_calls") or 0),
        "twelvelabs_cost": round(twelvelabs_cost, 4),
        "serpapi_cost": round(serpapi_cost, 4),
        "other_provider_cost": round(other_provider_cost, 4),
        "storage_cost": round(storage_cost, 4),
        "storage_gb": float(provider_payload.get("storage_gb") or 0),
        "bandwidth_cost": round(bandwidth_cost, 4),
        "bandwidth_tb": float(provider_payload.get("bandwidth_tb") or 0),
        "compute_cost": round(compute_cost, 4),
        "stripe_fees": round(stripe_fees, 4),
        "stripe_fee_txns": int(provider_payload.get("stripe_fee_txns") or 0),
        "mailgun_cost": round(mailgun_cost, 4),
        "mailgun_emails_sent": int(provider_payload.get("mailgun_emails_sent") or 0),
        "postgres_cost": round(postgres_cost, 4),
        "postgres_size_gb": float(provider_payload.get("postgres_size_gb") or 0),
        "redis_cost": round(redis_cost, 4),
        "redis_memory_mb": float(provider_payload.get("redis_memory_mb") or 0),
        "total_costs": total_costs,
        "total_cogs": total_costs,
        "successful_uploads": uploads,
        "cost_per_upload": round(total_costs / max(uploads, 1), 4),
        "providers": provider_payload.get("providers") or [],
        "cost_provenance": provider_payload.get("cost_provenance") or {},
    }


async def build_provider_costs_payload(conn, *, range_key: str | None) -> Dict[str, Any]:
    cache_key = range_key or "30d"
    now_mono = time.monotonic()
    cached = _provider_costs_cache.get(cache_key)
    if cached and (now_mono - cached[0]) < _PROVIDER_COSTS_CACHE_TTL_S:
        return dict(cached[1])

    since, until = window_from_range_key(range_key)
    by_cat = await fetch_cost_categories_window(conn, since, until)
    buckets = _bucket_totals(by_cat)
    env_prorates = _monthly_env_prorates(since, until)

    def _sum_cats(cats: frozenset) -> float:
        return sum(float(by_cat.get(c, {}).get("sum_cost") or 0) for c in cats if c in by_cat)

    def _cat_sum(cat: str) -> float:
        return float(by_cat.get(cat, {}).get("sum_cost") or 0)

    def _cat_count(cat: str) -> int:
        return int(by_cat.get(cat, {}).get("n_rows") or 0)

    def _cat_tokens(cat: str) -> int:
        return int(by_cat.get(cat, {}).get("token_sum") or 0)

    db_openai = _sum_cats(_OPENAI_CATEGORIES) or _cat_sum("openai")
    openai_live = await _fetch_openai_usage_window(since, until)
    openai_cost = max(db_openai, openai_live)

    google_db = _sum_cats(_GOOGLE_CATEGORIES)
    google_cloud_cost = max(google_db, env_prorates.get("google_cloud", 0.0))

    pikzels_db = _sum_cats(_PIKZELS_CATEGORIES)
    pikzels_est, pikzels_calls = await _estimate_pikzels_cost(conn, since, until)
    pikzels_cost = max(pikzels_db, pikzels_est, env_prorates.get("pikzels", 0.0))

    twelvelabs_cost = max(_sum_cats(_TWELVELABS_CATEGORIES), env_prorates.get("twelvelabs", 0.0))
    serpapi_cost = max(_sum_cats(_SERPAPI_CATEGORIES), env_prorates.get("serpapi", 0.0))

    other_provider_cost = float(buckets.get("other", 0.0))

    db_storage = _cat_sum("storage")
    db_bandwidth = _cat_sum("bandwidth")
    db_compute = _cat_sum("compute")
    db_stripe = _cat_sum("stripe_fees")
    db_mailgun_cost = _cat_sum("mailgun")
    db_mailgun_emails = _cat_tokens("mailgun")
    db_postgres = _cat_sum("postgres")
    db_redis = _cat_sum("redis")

    stripe_fees_live, stripe_txn_live = await fetch_stripe_fees_window(since, until)
    if stripe_txn_live > 0 or stripe_fees_live > 0:
        stripe_fees = stripe_fees_live
        stripe_fee_txns = stripe_txn_live
        stripe_source = "stripe_api_live"
    else:
        stripe_fees = db_stripe
        stripe_fee_txns = _cat_count("stripe_fees")
        stripe_source = "cost_tracking_db"

    prorated_render = env_prorates.get("render_compute", 0.0)
    render_cost = max(db_compute, prorated_render)

    r2_storage_per_gb = float(os.environ.get("KPI_R2_STORAGE_COST_PER_GB", "0.015") or 0)
    r2_bw_per_tb = float(os.environ.get("KPI_R2_BANDWIDTH_COST_PER_TB", "0") or 0)

    storage_gb = 0.0
    bandwidth_tb = 0.0
    try:
        from stages.kpi_collector import _fetch_cloudflare_r2_usage

        storage_gb, bandwidth_tb = await _fetch_cloudflare_r2_usage(since)
    except Exception as e:
        logger.debug("cf r2 usage: %s", e)

    window_frac = max((until - since).total_seconds() / (30 * 24 * 3600), 1e-9)
    cf_storage_cost = storage_gb * r2_storage_per_gb * min(1.0, window_frac) if storage_gb > 0 else 0.0
    cf_bw_cost = bandwidth_tb * r2_bw_per_tb

    storage_cost = max(db_storage, round(cf_storage_cost, 4))
    bandwidth_cost = max(db_bandwidth, round(cf_bw_cost, 4))

    if storage_gb <= 0 and storage_cost > 0 and r2_storage_per_gb > 0:
        storage_gb = round(storage_cost / max(r2_storage_per_gb * window_frac, 1e-9), 3)

    if bandwidth_tb <= 0 and bandwidth_cost > 0 and r2_bw_per_tb > 0:
        bandwidth_tb = round(bandwidth_cost / max(r2_bw_per_tb, 1e-9), 4)

    redis_mb = await _upstash_memory_mb()
    upstash_per_gb = float(os.environ.get("KPI_UPSTASH_COST_PER_GB", "0.20") or 0)
    live_redis_cost = round((redis_mb / 1024.0) * upstash_per_gb, 4) if redis_mb > 0 else 0.0
    redis_cost = max(db_redis, live_redis_cost)

    pg_disk = float(os.environ.get("KPI_POSTGRES_DISK_GB", "0") or 0)

    mailgun_cost = db_mailgun_cost
    mailgun_emails_sent = db_mailgun_emails

    # Build provider list for UI / API transparency
    provider_amounts: Dict[str, float] = {
        "openai": openai_cost,
        "google_cloud": google_cloud_cost,
        "pikzels": pikzels_cost,
        "twelvelabs": twelvelabs_cost,
        "serpapi": serpapi_cost,
        "stripe": stripe_fees,
        "mailgun": mailgun_cost,
        "cloudflare_storage": storage_cost,
        "cloudflare_bandwidth": bandwidth_cost,
        "render_compute": render_cost,
        "postgres": db_postgres,
        "redis_upstash": redis_cost,
        "other": other_provider_cost,
    }
    providers: List[Dict[str, Any]] = []
    for key, label in _PROVIDER_LABELS.items():
        amt = round(float(provider_amounts.get(key, 0) or 0), 4)
        if amt <= 0 and key not in ("openai", "google_cloud", "stripe", "render_compute"):
            continue
        src = "cost_tracking_db"
        if key == "openai" and openai_live > db_openai:
            src = "openai_usage_api"
        elif key == "google_cloud" and env_prorates.get("google_cloud", 0) > google_db:
            src = "env_monthly_prorate"
        elif key == "pikzels" and (pikzels_est > pikzels_db or env_prorates.get("pikzels", 0) > pikzels_db):
            src = "studio_usage_estimate_or_env"
        elif key == "stripe" and stripe_source == "stripe_api_live":
            src = "stripe_api_live"
        elif key == "cloudflare_storage" and cf_storage_cost > db_storage:
            src = "cloudflare_api_live"
        elif key == "redis_upstash" and live_redis_cost > db_redis:
            src = "upstash_api_live"
        elif key == "render_compute" and prorated_render > db_compute:
            src = "env_monthly_prorate"
        providers.append({"provider": key, "label": label, "cost_usd": amt, "source": src})

    raw_categories = [
        {
            "category": cat,
            "cost_usd": round(float(meta.get("sum_cost") or 0), 4),
            "rows": int(meta.get("n_rows") or 0),
            "bucket": provider_bucket_for_category(cat),
        }
        for cat, meta in sorted(by_cat.items())
    ]

    result = {
        "range": range_key or "30d",
        "window_start": since.isoformat(),
        "window_end": until.isoformat(),
        "openai_cost_window": round(openai_cost, 4),
        "openai_cost_live": round(openai_live, 4),
        "openai_api_rows": sum(_cat_count(c) for c in _OPENAI_CATEGORIES if c in by_cat) or _cat_count("openai"),
        "google_cloud_cost": round(google_cloud_cost, 4),
        "pikzels_cost": round(pikzels_cost, 4),
        "pikzels_calls": pikzels_calls,
        "twelvelabs_cost": round(twelvelabs_cost, 4),
        "serpapi_cost": round(serpapi_cost, 4),
        "other_provider_cost": round(other_provider_cost, 4),
        "render_cost": round(render_cost, 4),
        "storage_gb": round(storage_gb, 4),
        "storage_cost": round(storage_cost, 4),
        "bandwidth_cost": round(bandwidth_cost, 4),
        "bandwidth_tb": round(bandwidth_tb, 6),
        "redis_cost": round(redis_cost, 4),
        "redis_memory_mb": round(redis_mb, 2),
        "postgres_cost": round(db_postgres, 4),
        "postgres_size_gb": round(pg_disk, 2) if pg_disk > 0 else 0.0,
        "mailgun_cost": round(mailgun_cost, 4),
        "mailgun_emails_sent": int(mailgun_emails_sent),
        "stripe_fees": float(stripe_fees),
        "stripe_fee_txns": int(stripe_fee_txns),
        "providers": providers,
        "categories_raw": raw_categories,
        "cost_provenance": {
            "revenue": "revenue_tracking DB + Stripe refunds API when STRIPE_SECRET_KEY set",
            "costs": "cost_tracking DB + live Stripe/Cloudflare/Upstash/OpenAI + env monthly prorates",
            "google_cloud_note": "Set KPI_GOOGLE_CLOUD_MONTHLY_COST or write vision_google/video_intelligence rows to cost_tracking",
            "pikzels_note": "Set KPI_PIKZELS_USD_PER_CALL or KPI_PIKZELS_MONTHLY_COST; usage from studio_usage_events",
        },
    }
    _provider_costs_cache[cache_key] = (time.monotonic(), result)
    return result


async def build_cost_tracker_payload(conn, *, range_key: str | None) -> Dict[str, Any]:
    since, until = window_from_range_key(range_key)
    uploads = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM uploads
        WHERE status IN ('completed','succeeded')
          AND created_at >= $1 AND created_at < $2
        """,
        since,
        until,
    )
    uploads = int(uploads or 0)
    per = float(os.environ.get("KPI_TOOL_ESTIMATE_PER_UPLOAD_USD", "0.025") or 0)
    if not math.isfinite(per) or per < 0:
        per = 0.0
    total = round(uploads * per, 4)
    return {
        "range": range_key or "30d",
        "estimated_total_window_usd": total,
        "estimated_total_per_upload_usd": per,
        "successful_uploads": uploads,
    }
