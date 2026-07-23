"""Dashboard engagement KPIs use Analytics range=all (unbounded) when complete."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from services.canonical_engagement import engagement_time_window_for_analytics_range
from services.dashboard_user_stats import (
    _assemble_dashboard_stats,
    _dashboard_stats_on_conn,
    _engagement_scope_from_rollup,
)


def test_analytics_all_window_is_unbounded():
    start, end = engagement_time_window_for_analytics_range(
        "all", now=datetime(2026, 7, 22, tzinfo=timezone.utc)
    )
    assert start is None and end is None


def test_engagement_scope_labels_are_honest():
    assert _engagement_scope_from_rollup({"rollup_rule": "canonical"}) == "all_time"
    assert _engagement_scope_from_rollup({"rollup_rule": "light_metrics_cache"}) == "provisional_cache"
    assert (
        _engagement_scope_from_rollup({"rollup_rule": "fallback_upload_table_only"})
        == "upload_table_fallback"
    )


def _assemble_with_rule(rule: str) -> dict:
    return _assemble_dashboard_stats(
        {"role": "user", "subscription_tier": "free"},
        {"put_monthly": 60, "monthly_uploads": 60, "max_accounts": 1, "aic_monthly": 0},
        {"put_balance": 0, "put_reserved": 0, "aic_balance": 0, "aic_reserved": 0},
        {
            "total": 1,
            "completed": 1,
            "in_queue": 0,
            "successful_this_month": 0,
            "successful_last_month": 0,
        },
        0,
        0,
        0,
        0,
        [],
        {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []},
        {"views": 10, "likes": 2, "comments": 1, "shares": 0},
        {
            "views": 10,
            "likes": 2,
            "comments": 1,
            "shares": 0,
            "breakdown": {},
            "rollup_rule": rule,
            "catalog_tracked_videos": 1,
            "kpi_sources": {},
        },
        {"start": None, "end_exclusive": None},
    )


def test_assemble_marks_engagement_scope_all_time():
    body = _assemble_with_rule("canonical")
    assert body["engagement"]["scope"] == "all_time"
    assert body["uploads"]["total_scope"] == "all_time"


def test_assemble_light_scope_not_all_time():
    body = _assemble_with_rule("light_metrics_cache")
    assert body["engagement"]["scope"] == "provisional_cache"


def test_assemble_fallback_scope_not_all_time():
    body = _assemble_with_rule("fallback_upload_table_only")
    assert body["engagement"]["scope"] == "upload_table_fallback"


def test_dashboard_stats_requests_all_time_rollup():
    user = {"id": "u1", "role": "user", "subscription_tier": "free"}
    plan = {"put_monthly": 60, "monthly_uploads": 60, "max_accounts": 1, "aic_monthly": 0}
    wallet = {"put_balance": 0, "put_reserved": 0, "aic_balance": 0, "aic_reserved": 0}

    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "total": 0,
            "completed": 0,
            "in_queue": 0,
            "views": 0,
            "likes": 0,
            "successful_this_month": 0,
            "successful_last_month": 0,
        }
    )
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])

    rollup = AsyncMock(
        return_value={
            "views": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "breakdown": {},
            "rollup_rule": "canonical",
            "rollup_version": 1,
            "catalog_tracked_videos": 0,
            "kpi_sources": {},
        }
    )

    async def _run():
        with patch(
            "services.dashboard_user_stats._compute_upload_engagement_totals",
            new=AsyncMock(return_value={"views": 0, "likes": 0, "comments": 0, "shares": 0}),
        ) as totals, patch(
            "services.dashboard_user_stats.compute_canonical_engagement_rollup",
            new=rollup,
        ):
            out = await _dashboard_stats_on_conn(conn, user, plan, wallet, light=False, pool=None)
        return out, totals

    out, totals = asyncio.run(_run())
    assert out["engagement"]["scope"] == "all_time"
    assert totals.await_args.kwargs.get("since") is None
    assert totals.await_args.kwargs.get("until") is None
    assert rollup.await_args.kwargs.get("window_start") is None
    assert rollup.await_args.kwargs.get("window_end_exclusive") is None


def test_dashboard_stats_counts_billing_owner_uploads():
    """Monthly quota must use billing_user_id (same scope as shell uploads list)."""
    from services.dashboard_user_stats import _dashboard_stats_uid

    assert _dashboard_stats_uid({"id": "member", "billing_user_id": "owner"}) == "owner"
    assert _dashboard_stats_uid({"id": "solo"}) == "solo"

    user = {
        "id": "member-1",
        "billing_user_id": "owner-1",
        "role": "user",
        "subscription_tier": "agency",
    }
    plan = {"put_monthly": 20000, "monthly_uploads": 20000, "max_accounts": 10, "aic_monthly": 0}
    wallet = {"put_balance": 0, "put_reserved": 0, "aic_balance": 0, "aic_reserved": 0}

    seen_uids: list[str] = []

    async def _fetchrow(sql, *args):
        if args:
            seen_uids.append(str(args[0]))
        return {
            "total": 4,
            "completed": 3,
            "in_queue": 0,
            "views": 0,
            "likes": 0,
            "successful_this_month": 2,
            "successful_last_month": 1,
            "scheduled_count": 0,
            "uploads_used_month": 4,
        }

    async def _fetchval(sql, *args):
        if args:
            seen_uids.append(str(args[0]))
        return 4

    conn = MagicMock()
    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.fetchval = AsyncMock(side_effect=_fetchval)
    conn.fetch = AsyncMock(return_value=[])

    async def _run():
        return await _dashboard_stats_on_conn(conn, user, plan, wallet, light=True, pool=None)

    out = asyncio.run(_run())
    assert out["quota"]["uploads_used"] == 4
    assert seen_uids
    assert all(uid == "owner-1" for uid in seen_uids)


def test_rollup_timeout_closes_separate_connection():
    user = {"id": "u1", "role": "user", "subscription_tier": "free"}
    plan = {"put_monthly": 60, "monthly_uploads": 60, "max_accounts": 1, "aic_monthly": 0}
    wallet = {"put_balance": 0, "put_reserved": 0, "aic_balance": 0, "aic_reserved": 0}

    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "total": 0,
            "completed": 0,
            "in_queue": 0,
            "views": 0,
            "likes": 0,
            "successful_this_month": 0,
            "successful_last_month": 0,
        }
    )
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])

    rconn = MagicMock()
    rconn.close = AsyncMock()
    pool = MagicMock()
    pool.acquire = AsyncMock(return_value=rconn)
    pool.release = AsyncMock()

    async def _slow_rollup(*_a, **_k):
        await asyncio.sleep(60)
        return {}

    async def _run():
        with patch(
            "services.dashboard_user_stats._compute_upload_engagement_totals",
            new=AsyncMock(return_value={"views": 9, "likes": 1, "comments": 0, "shares": 0}),
        ), patch(
            "services.dashboard_user_stats.compute_canonical_engagement_rollup",
            new=_slow_rollup,
        ), patch(
            "services.dashboard_user_stats._ENGAGEMENT_ROLLUP_TIMEOUT_SEC",
            0.05,
        ):
            return await _dashboard_stats_on_conn(
                conn, user, plan, wallet, light=False, pool=pool
            )

    out = asyncio.run(_run())
    assert out["engagement"]["scope"] == "upload_table_fallback"
    assert out["engagement"]["views"] == 9
    rconn.close.assert_awaited()
    pool.release.assert_not_awaited()
