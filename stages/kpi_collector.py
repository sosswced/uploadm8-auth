"""
KPI Data Collector — Fetches cost and revenue data from external APIs every 30 minutes.

Uses env keys to authenticate and call:
  - Stripe API: balance transactions (fees)
  - OpenAI: usage/cost (if Usage API available)
  - Mailgun: stats (emails sent)
  - Cloudflare: R2 storage/bandwidth (if CF_API_TOKEN set)
  - Render: RENDER_MONTHLY_COST (prorated)
  - Upstash: Redis usage (if UPSTASH_* set)

Posts incremental costs to cost_tracking. Tracks last_sync per provider to avoid duplicates.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

logger = logging.getLogger("uploadm8-kpi")

# Cost estimates when APIs don't provide direct cost
MAILGUN_COST_PER_1000 = float(os.environ.get("KPI_MAILGUN_COST_PER_1000", "0.80"))
R2_STORAGE_COST_PER_GB_MONTH = float(os.environ.get("KPI_R2_STORAGE_COST_PER_GB", "0.015"))
R2_BANDWIDTH_COST_PER_TB = float(os.environ.get("KPI_R2_BANDWIDTH_COST_PER_TB", "0.00"))  # R2 egress free to CF
UPSTASH_COST_PER_GB = float(os.environ.get("KPI_UPSTASH_COST_PER_GB", "0.20"))

# Per-upload estimates for tools without reliable direct billing APIs.
# Values are intentionally explicit so admin cost cards stay deterministic.
TOOL_COST_ESTIMATES_USD_PER_UPLOAD = {
    "openai_whisper": 0.0030,
    "openai_gpt4o_mini": 0.0020,
    "acrcloud": 0.0010,
    "google_vision": 0.0045,
    "hume_ai": 0.0020,
    "fal_flux": 0.0030,
    "rembg": 0.0,
    "yamnet": 0.0,
    "playwright": 0.0,
}


async def _fetch_stripe_fees(since: datetime) -> float:
    """Sum Stripe fees from balance transactions since given time."""
    key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not key:
        return 0.0
    try:
        import stripe
        stripe.api_key = key
        since_ts = int(since.timestamp())
        total_fees = 0.0
        for t in stripe.BalanceTransaction.list(created={"gte": since_ts}, limit=100).auto_paging_iter():
            total_fees += (t.fee or 0) / 100.0  # cents to dollars
        return round(total_fees, 4)
    except Exception as e:
        logger.warning(f"Stripe fees fetch failed: {e}")
        return 0.0


async def _fetch_mailgun_stats(since: datetime) -> tuple[int, float]:
    """Get emails delivered since date. Returns (count, estimated_cost_usd)."""
    key = os.environ.get("MAILGUN_API_KEY", "")
    domain = os.environ.get("MAILGUN_DOMAIN", "")
    if not key or not domain:
        return 0, 0.0
    try:
        # Mailgun: GET /v3/{domain}/stats/total with event=delivered
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"https://api.mailgun.net/v3/{domain}/stats/total",
                auth=("api", key),
                params={
                    "event": "delivered",
                    "duration": "30m",
                },
            )
            if r.status_code != 200:
                logger.debug(f"Mailgun stats: {r.status_code}")
                return 0, 0.0
            data = r.json()
            total = 0
            for item in data.get("stats", []):
                if isinstance(item, dict):
                    total += item.get("delivered", {}).get("total", 0) if isinstance(item.get("delivered"), dict) else item.get("total", 0)
            cost = (total / 1000.0) * MAILGUN_COST_PER_1000
            return int(total), round(cost, 4)
    except Exception as e:
        logger.debug(f"Mailgun stats: {e}")
        return 0, 0.0


async def _fetch_openai_usage(since: datetime) -> float:
    """
    Fetch OpenAI usage/cost. Uses Usage API if available (org-level key).
    Falls back to 0 — per-request costs are already in cost_tracking from caption stage.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return 0.0
    try:
        # OpenAI Usage API: https://api.openai.com/v1/organization/usage/...
        # Requires organization context. Many keys don't have access.
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                "https://api.openai.com/v1/organization/usage/completions",
                headers={"Authorization": f"Bearer {key}"},
                params={
                    "start_time": int(since.timestamp()),
                    "end_time": int(datetime.now(timezone.utc).timestamp()),
                    "bucket_width": "1h",
                    "limit": 100,
                },
            )
            if r.status_code != 200:
                # 403/404 common when Usage API not enabled for key
                if r.status_code not in (403, 404):
                    logger.warning(f"OpenAI usage API: {r.status_code}")
                return 0.0
            data = r.json()
            # Sum cost from response if present
            total = 0.0
            for bucket in data.get("data", data.get("usage", [])) or []:
                if isinstance(bucket, dict):
                    total += float(bucket.get("cost_usd", bucket.get("cost", 0)) or 0)
            return round(total, 6)
    except Exception as e:
        logger.debug(f"OpenAI usage fetch: {e}")
        return 0.0


async def _fetch_cloudflare_r2_usage(since: datetime) -> tuple[float, float]:
    """
    Fetch R2 storage (GB) and bandwidth (TB) from Cloudflare Analytics.
    Requires CF_API_TOKEN with Analytics Read. Returns (0,0) if unavailable.
    """
    token = os.environ.get("CF_API_TOKEN", "")
    account_id = os.environ.get("R2_ACCOUNT_ID", "") or os.environ.get("CF_ACCOUNT_ID", "")
    if not token or not account_id:
        return 0.0, 0.0
    try:
        # Cloudflare Analytics API - R2 metrics (schema may vary by CF plan)
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/analytics/analytics",
                headers={"Authorization": f"Bearer {token}"},
                params={"since": since.strftime("%Y-%m-%d"), "until": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
            )
            if r.status_code != 200:
                return 0.0, 0.0
            data = r.json()
            storage_bytes = data.get("totals", {}).get("bytes", 0) or 0
            egress_bytes = data.get("totals", {}).get("bandwidth", 0) or 0
            return storage_bytes / (1024**3), egress_bytes / (1024**4)
    except Exception:
        return 0.0, 0.0


async def _fetch_upstash_usage() -> float:
    """Get Upstash Redis usage and estimate cost. Uses UPSTASH_API_KEY."""
    api_key = os.environ.get("UPSTASH_API_KEY", "")
    db_id = os.environ.get("UPSTASH_DB_ID", "")
    if not api_key:
        return 0.0
    try:
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
            gb = mb / 1024.0
            return round(gb * UPSTASH_COST_PER_GB, 4)
    except Exception:
        return 0.0


def _prorate_monthly_cost(monthly: float, since: datetime, until: datetime) -> float:
    """Prorate monthly cost to the given time window."""
    if monthly <= 0:
        return 0.0
    delta = (until - since).total_seconds()
    # ~30 days in seconds
    month_sec = 30 * 24 * 3600
    return round(monthly * (delta / month_sec), 4)


def estimate_tool_costs(successful_uploads: int) -> dict:
    """
    Convert per-upload tool assumptions into totals for a window.
    Returns payload safe for API responses and DB writes.
    """
    uploads = int(max(successful_uploads or 0, 0))
    per_tool = []
    total_per_upload = 0.0
    total_window = 0.0
    for tool_key, usd_per_upload in TOOL_COST_ESTIMATES_USD_PER_UPLOAD.items():
        unit = float(max(usd_per_upload, 0.0))
        window_cost = round(unit * uploads, 6)
        total_per_upload += unit
        total_window += window_cost
        per_tool.append(
            {
                "tool": tool_key,
                "usd_per_upload": round(unit, 6),
                "uploads": uploads,
                "window_cost_usd": window_cost,
                "source": "estimated",
            }
        )
    return {
        "uploads": uploads,
        "total_usd_per_upload": round(total_per_upload, 6),
        "total_window_cost_usd": round(total_window, 6),
        "tools": per_tool,
    }


async def run_kpi_collect(db_pool) -> dict:
    """
    Collect cost/revenue data from external APIs and insert into cost_tracking.
    Uses kpi_sync_state to track last sync per provider; only processes deltas.
    Returns summary of what was collected.
    """
    now = datetime.now(timezone.utc)
    window = timedelta(minutes=30)
    since = now - window

    summary = {
        "stripe_fees": 0.0,
        "mailgun_cost": 0.0,
        "mailgun_emails": 0,
        "openai_cost": 0.0,
        "storage_cost": 0.0,
        "bandwidth_cost": 0.0,
        "postgres_cost": 0.0,
        "redis_cost": 0.0,
        "rows_inserted": 0,
    }

    async with db_pool.acquire() as conn:
        # Ensure kpi_sync_state exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kpi_sync_state (
                id INT PRIMARY KEY DEFAULT 1,
                last_stripe_sync_at TIMESTAMPTZ,
                last_mailgun_sync_at TIMESTAMPTZ,
                last_openai_sync_at TIMESTAMPTZ,
                last_cf_sync_at TIMESTAMPTZ,
                last_upstash_sync_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute(
            "ALTER TABLE kpi_sync_state ADD COLUMN IF NOT EXISTS last_tool_est_sync_at TIMESTAMPTZ"
        )
        await conn.execute(
            "INSERT INTO kpi_sync_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING"
        )

        # Get last sync times
        row = await conn.fetchrow("SELECT * FROM kpi_sync_state WHERE id = 1")
        last_stripe = row.get("last_stripe_sync_at") if row else None
        last_mailgun = row.get("last_mailgun_sync_at") if row else None
        last_openai = row.get("last_openai_sync_at") if row else None
        last_cf = row.get("last_cf_sync_at") if row else None
        last_upstash = row.get("last_upstash_sync_at") if row else None
        last_tool_est = row.get("last_tool_est_sync_at") if row else None

        # Use last sync or window start
        stripe_since = last_stripe or since
        mailgun_since = last_mailgun or since
        openai_since = last_openai or since
        cf_since = last_cf or since

        # 1. Stripe fees
        fees = await _fetch_stripe_fees(stripe_since)
        if fees > 0:
            await conn.execute(
                """
                INSERT INTO cost_tracking (user_id, category, operation, cost_usd, created_at)
                VALUES (NULL, 'stripe_fees', 'kpi_sync', $1, NOW())
                """,
                fees,
            )
            summary["stripe_fees"] = fees
            summary["rows_inserted"] += 1
        await conn.execute(
            "UPDATE kpi_sync_state SET last_stripe_sync_at = $1, updated_at = NOW() WHERE id = 1",
            now,
        )

        # 2. Mailgun
        emails, mailgun_cost = await _fetch_mailgun_stats(mailgun_since)
        summary["mailgun_emails"] = emails
        if mailgun_cost > 0:
            await conn.execute(
                """
                INSERT INTO cost_tracking (user_id, category, operation, tokens, cost_usd, created_at)
                VALUES (NULL, 'mailgun', 'kpi_sync', $1, $2, NOW())
                """,
                emails,
                mailgun_cost,
            )
            summary["mailgun_cost"] = mailgun_cost
            summary["rows_inserted"] += 1
        await conn.execute(
            "UPDATE kpi_sync_state SET last_mailgun_sync_at = $1, updated_at = NOW() WHERE id = 1",
            now,
        )

        # 3. OpenAI (incremental from Usage API; per-request already in cost_tracking)
        openai_delta = await _fetch_openai_usage(openai_since)
        if openai_delta > 0:
            await conn.execute(
                """
                INSERT INTO cost_tracking (user_id, category, operation, cost_usd, created_at)
                VALUES (NULL, 'openai', 'kpi_usage_api', $1, NOW())
                """,
                openai_delta,
            )
            summary["openai_cost"] = openai_delta
            summary["rows_inserted"] += 1
        await conn.execute(
            "UPDATE kpi_sync_state SET last_openai_sync_at = $1, updated_at = NOW() WHERE id = 1",
            now,
        )

        # 4. Cloudflare R2 storage + bandwidth
        storage_gb, bandwidth_tb = await _fetch_cloudflare_r2_usage(cf_since)
        if storage_gb > 0 or bandwidth_tb > 0:
            storage_cost = storage_gb * R2_STORAGE_COST_PER_GB_MONTH * (window.total_seconds() / (30 * 24 * 3600))
            bandwidth_cost = bandwidth_tb * R2_BANDWIDTH_COST_PER_TB
            if storage_cost > 0:
                await conn.execute(
                    """
                    INSERT INTO cost_tracking (user_id, category, operation, cost_usd, created_at)
                    VALUES (NULL, 'storage', 'kpi_r2_sync', $1, NOW())
                    """,
                    round(storage_cost, 6),
                )
                summary["storage_cost"] = storage_cost
                summary["rows_inserted"] += 1
            if bandwidth_cost > 0:
                await conn.execute(
                    """
                    INSERT INTO cost_tracking (user_id, category, operation, cost_usd, created_at)
                    VALUES (NULL, 'bandwidth', 'kpi_r2_sync', $1, NOW())
                    """,
                    round(bandwidth_cost, 6),
                )
                summary["bandwidth_cost"] = bandwidth_cost
                summary["rows_inserted"] += 1
        await conn.execute(
            "UPDATE kpi_sync_state SET last_cf_sync_at = $1, updated_at = NOW() WHERE id = 1",
            now,
        )

        # 5. Render Postgres (prorated monthly)
        render_monthly = float(os.environ.get("RENDER_MONTHLY_COST", "0") or 0)
        if render_monthly > 0:
            postgres_cost = _prorate_monthly_cost(render_monthly, since, now)
            if postgres_cost > 0:
                await conn.execute(
                    """
                    INSERT INTO cost_tracking (user_id, category, operation, cost_usd, created_at)
                    VALUES (NULL, 'postgres', 'kpi_sync', $1, NOW())
                    """,
                    postgres_cost,
                )
                summary["postgres_cost"] = postgres_cost
                summary["rows_inserted"] += 1

        # 6. Upstash Redis
        redis_cost = await _fetch_upstash_usage()
        if redis_cost > 0:
            await conn.execute(
                """
                INSERT INTO cost_tracking (user_id, category, operation, cost_usd, created_at)
                VALUES (NULL, 'redis', 'kpi_sync', $1, NOW())
                """,
                redis_cost,
            )
            summary["redis_cost"] = redis_cost
            summary["rows_inserted"] += 1
        await conn.execute(
            "UPDATE kpi_sync_state SET last_upstash_sync_at = $1, updated_at = NOW() WHERE id = 1",
            now,
        )

        # 7. Per-upload tool estimates (for tools without direct cost APIs)
        tool_since = last_tool_est or since
        successful_uploads = await conn.fetchval(
            """
            SELECT COUNT(*)::int
            FROM uploads
            WHERE status IN ('completed', 'succeeded', 'partial')
              AND created_at >= $1
            """,
            tool_since,
        )
        tool_costs = estimate_tool_costs(int(successful_uploads or 0))
        for t in tool_costs["tools"]:
            if t["window_cost_usd"] <= 0:
                continue
            await conn.execute(
                """
                INSERT INTO cost_tracking (user_id, category, operation, tokens, cost_usd, created_at)
                VALUES (NULL, 'tool_estimate', $1, $2, $3, NOW())
                """,
                t["tool"],
                int(t["uploads"]),
                float(t["window_cost_usd"]),
            )
            summary["rows_inserted"] += 1
        summary["tool_estimated_total"] = float(tool_costs["total_window_cost_usd"])
        summary["tool_estimated_uploads"] = int(tool_costs["uploads"])
        await conn.execute(
            "UPDATE kpi_sync_state SET last_tool_est_sync_at = $1, updated_at = NOW() WHERE id = 1",
            now,
        )

    return summary


async def fetch_provider_costs_for_dashboard(window_minutes: int = 30 * 24 * 60) -> dict:
    """
    Read-only snapshot for GET /api/admin/kpi/provider-costs (admin KPI UI).
    Uses the same env + external helpers as run_kpi_collect without writing to cost_tracking.
    """
    now = datetime.now(timezone.utc)
    safe_minutes = int(max(window_minutes or 0, 1))
    since = now - timedelta(minutes=safe_minutes)
    storage_gb, bandwidth_tb = await _fetch_cloudflare_r2_usage(since)
    window = timedelta(minutes=safe_minutes)
    storage_cost = 0.0
    if storage_gb > 0:
        storage_cost = round(
            storage_gb * R2_STORAGE_COST_PER_GB_MONTH * (window.total_seconds() / (30 * 24 * 3600)),
            6,
        )
    bandwidth_cost = round(bandwidth_tb * R2_BANDWIDTH_COST_PER_TB, 6)
    render_monthly = float(os.environ.get("RENDER_MONTHLY_COST", "0") or 0)
    render_cost = _prorate_monthly_cost(render_monthly, since, now)
    redis_cost = await _fetch_upstash_usage()
    stripe_fees = await _fetch_stripe_fees(since)
    emails, mailgun_cost = await _fetch_mailgun_stats(since)
    openai_delta = await _fetch_openai_usage(since)
    redis_memory_mb = 0.0
    try:
        api_key = os.environ.get("UPSTASH_API_KEY", "")
        db_id = os.environ.get("UPSTASH_DB_ID", "")
        if api_key:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"https://api.upstash.com/v2/database/{db_id}/stats" if db_id else "https://api.upstash.com/v2/databases"
                r = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        for db in data:
                            redis_memory_mb += float(db.get("memory_usage_mb", 0) or 0)
                    elif isinstance(data, dict):
                        redis_memory_mb = float(data.get("memory_usage_mb", 0) or 0)
    except Exception:
        pass
    return {
        "render_cost": render_cost,
        "storage_gb": round(storage_gb, 4),
        "storage_cost": storage_cost,
        "bandwidth_cost": bandwidth_cost,
        "bandwidth_tb": round(bandwidth_tb, 6),
        "redis_cost": redis_cost,
        "redis_memory_mb": round(redis_memory_mb, 4),
        "postgres_cost": 0.0,
        "postgres_size_gb": 0.0,
        "mailgun_cost": mailgun_cost,
        "mailgun_emails_sent": int(emails),
        "stripe_fees": stripe_fees,
        "stripe_fee_txns": 0,
        "openai_usage_api": openai_delta,
    }
