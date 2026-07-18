"""Unit tests for analytics range presets and Trill leaderboard response cache."""

from __future__ import annotations

from datetime import datetime, timezone

from routers.analytics import _RANGE_PRESETS_MINUTES
from routers.trill import _trill_since_dt, leaderboard_serve_from_cache
from services.canonical_engagement import (
    ALL_TIME_FLOOR_UTC,
    ANALYTICS_RANGE_MINUTES,
    engagement_time_window_for_analytics_range,
    sql_since_for_analytics_range,
)
from services.growth_intelligence import parse_range_since_until
from services.shell_bootstrap import _KPI_RANGE_DAYS


def test_365d_and_1y_range_minutes_match():
    assert ANALYTICS_RANGE_MINUTES["365d"] == ANALYTICS_RANGE_MINUTES["1y"]
    assert ANALYTICS_RANGE_MINUTES["365d"] == _RANGE_PRESETS_MINUTES["365d"]
    assert ANALYTICS_RANGE_MINUTES["1y"] == _RANGE_PRESETS_MINUTES["1y"]


def test_365d_and_1y_produce_same_window_span():
    now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    start_365d, end_365d = engagement_time_window_for_analytics_range("365d", now=now)
    start_1y, end_1y = engagement_time_window_for_analytics_range("1y", now=now)
    assert end_365d == end_1y
    assert start_365d == start_1y
    assert (end_365d - start_365d).days == 365


def test_all_range_engagement_unbounded_sql_uses_floor():
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    start, end = engagement_time_window_for_analytics_range("all", now=now)
    assert start is None and end is None
    since = sql_since_for_analytics_range("all", now=now)
    assert since == ALL_TIME_FLOOR_UTC


def test_all_range_aligned_across_pikzels_trill_kpi():
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    pikzels_since, pikzels_until = parse_range_since_until("all")
    assert pikzels_since == ALL_TIME_FLOOR_UTC
    assert pikzels_until.tzinfo is not None
    trill_since = _trill_since_dt("all")
    assert trill_since == ALL_TIME_FLOOR_UTC
    # KPI bootstrap uses range=all on analytics_overview (not a fixed 3650d map entry).
    assert "all" not in _KPI_RANGE_DAYS
    # 90d must be ~90 days (not the old ~182d typo); pin now= for deterministic span.
    trill_90d = sql_since_for_analytics_range("90d", now=now)
    assert (now - trill_90d).days == 90


def test_kpi_bootstrap_range_days_includes_1y_and_365d():
    assert _KPI_RANGE_DAYS["1y"] == 365
    assert _KPI_RANGE_DAYS["365d"] == 365
    assert "all" not in _KPI_RANGE_DAYS


def test_leaderboard_cache_strips_rival_alerts():
    cache = {}
    key = "user:30d:best_trill::50"
    payload = {
        "unlocked": True,
        "rows": [{"rank": 1, "driver_handle": "A"}],
        "rival_alerts": [{"rival_handle": "B", "rival_rank": 2, "your_rank": 5}],
    }
    cache[key] = (1000.0, payload)
    served = leaderboard_serve_from_cache(cache, key, now=1050.0, ttl_sec=120.0)
    assert served is not None
    assert served["rows"] == payload["rows"]
    assert served["rival_alerts"] == []


def test_leaderboard_cache_miss_when_expired():
    cache = {"k": (1000.0, {"rival_alerts": [{"x": 1}]})}
    assert leaderboard_serve_from_cache(cache, "k", now=1121.0, ttl_sec=120.0) is None


def test_leaderboard_cache_miss_when_missing_key():
    assert leaderboard_serve_from_cache({}, "missing", now=1000.0) is None


def test_live_platform_metrics_does_not_cache_empty_on_failed_refresh(monkeypatch):
    """Failed refresh + empty DB must not poison in-memory cache for full TTL."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    import core.state
    import routers.analytics as analytics

    analytics._platform_metrics_cache.clear()
    user = {"id": "user-cache-poison-test"}

    async def _refresh_fail(*_a, **_k):
        return False

    async def _db_get(*_a, **_k):
        return None

    class _Acquire:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *_exc):
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_Acquire())
    prev_pool = core.state.db_pool
    core.state.db_pool = pool
    monkeypatch.setattr(
        "services.platform_metrics_job.refresh_platform_metrics_for_user",
        _refresh_fail,
    )
    monkeypatch.setattr(analytics, "_platform_metrics_db_cache_get", _db_get)
    try:
        out = asyncio.run(analytics._compute_live_platform_metrics(user))
        assert out.get("refresh_failed") is True
        assert out["platforms"]["tiktok"]["status"] == "not_connected"
        assert "user-cache-poison-test" not in analytics._platform_metrics_cache
    finally:
        core.state.db_pool = prev_pool
        analytics._platform_metrics_cache.clear()


def test_live_platform_metrics_caches_successful_db_payload(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    import core.state
    import routers.analytics as analytics

    analytics._platform_metrics_cache.clear()
    user = {"id": "user-cache-ok-test"}
    payload = {
        "platforms": {
            "tiktok": {"status": "live", "views": 10, "accounts": []},
            "youtube": {"status": "not_connected", "accounts": []},
            "instagram": {"status": "not_connected", "accounts": []},
            "facebook": {"status": "not_connected", "accounts": []},
        },
        "aggregate": {"views": 10, "likes": 0, "comments": 0, "shares": 0, "platforms_included": ["tiktok"]},
    }

    async def _refresh_ok(*_a, **_k):
        return True

    async def _db_get(*_a, **_k):
        return dict(payload)

    class _Acquire:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *_exc):
            return False

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_Acquire())
    prev_pool = core.state.db_pool
    core.state.db_pool = pool
    monkeypatch.setattr(
        "services.platform_metrics_job.refresh_platform_metrics_for_user",
        _refresh_ok,
    )
    monkeypatch.setattr(analytics, "_platform_metrics_db_cache_get", _db_get)
    try:
        out = asyncio.run(analytics._compute_live_platform_metrics(user))
        assert out.get("cached") is False
        assert "refresh_failed" not in out
        assert "user-cache-ok-test" in analytics._platform_metrics_cache
        assert analytics._platform_metrics_cache["user-cache-ok-test"]["data"]["platforms"]["tiktok"]["views"] == 10
    finally:
        core.state.db_pool = prev_pool
        analytics._platform_metrics_cache.clear()
