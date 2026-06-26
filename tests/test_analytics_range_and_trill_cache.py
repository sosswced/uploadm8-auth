"""Unit tests for analytics range presets and Trill leaderboard response cache."""

from __future__ import annotations

from datetime import datetime, timezone

from routers.analytics import _RANGE_PRESETS_MINUTES
from routers.trill import leaderboard_serve_from_cache
from services.canonical_engagement import (
    ANALYTICS_RANGE_MINUTES,
    engagement_time_window_for_analytics_range,
)
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


def test_kpi_bootstrap_range_days_includes_1y_and_365d():
    assert _KPI_RANGE_DAYS["1y"] == 365
    assert _KPI_RANGE_DAYS["365d"] == 365


def test_leaderboard_cache_miss_when_missing_key():
    assert leaderboard_serve_from_cache({}, "missing", now=1000.0) is None
