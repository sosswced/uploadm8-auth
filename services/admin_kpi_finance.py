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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

logger = logging.getLogger("uploadm8.admin_kpi_finance")

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


async def build_provider_costs_payload(conn, *, range_key: str | None) -> Dict[str, Any]:
    since, until = window_from_range_key(range_key)
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
    by_cat: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cat = str(row["category"] or "").lower()
        by_cat[cat] = {
            "sum_cost": float(row["sum_cost"] or 0),
            "n_rows": int(row["n_rows"] or 0),
            "token_sum": int(row["token_sum"] or 0),
        }

    def _sum(cat: str) -> float:
        return float(by_cat.get(cat, {}).get("sum_cost") or 0)

    def _count(cat: str) -> int:
        return int(by_cat.get(cat, {}).get("n_rows") or 0)

    def _tokens(cat: str) -> int:
        return int(by_cat.get(cat, {}).get("token_sum") or 0)

    db_openai = _sum("openai")
    db_storage = _sum("storage")
    db_bandwidth = _sum("bandwidth")
    db_compute = _sum("compute")
    db_stripe = _sum("stripe_fees")
    db_mailgun_cost = _sum("mailgun")
    db_mailgun_emails = _tokens("mailgun")
    db_postgres = _sum("postgres")
    db_redis = _sum("redis")

    stripe_fees_live, stripe_txn_live = await fetch_stripe_fees_window(since, until)
    if stripe_txn_live > 0 or stripe_fees_live > 0:
        stripe_fees = stripe_fees_live
        stripe_fee_txns = stripe_txn_live
    else:
        stripe_fees = db_stripe
        stripe_fee_txns = _count("stripe_fees")

    render_monthly = float(os.environ.get("RENDER_MONTHLY_COST", "0") or 0)
    prorated_render = _prorate_monthly_cost(render_monthly, since, until)
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

    return {
        "range": range_key or "30d",
        "window_start": since.isoformat(),
        "window_end": until.isoformat(),
        "openai_cost_window": round(db_openai, 4),
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
    }


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
