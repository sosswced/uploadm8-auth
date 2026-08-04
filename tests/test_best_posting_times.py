"""Unit tests for best posting times compilers (no DB)."""

from __future__ import annotations

from services.best_posting_times import (
    _hour_label,
    _pinpoint_from_platforms,
    _serialize_hours,
    _top_hours,
    best_posting_times_fallback,
)
from core.scheduling import static_hour_prior_24


def test_hour_label_and_top_hours():
    assert _hour_label(0) == "12:00 AM"
    assert _hour_label(15) == "3:00 PM"
    w = static_hour_prior_24("tiktok")
    tops = _top_hours(w, n=3)
    assert len(tops) >= 1
    assert tops[0]["weight"] >= tops[-1]["weight"]
    hours = _serialize_hours(w, "tiktok")
    assert len(hours) == 24
    assert any(h["in_hot_window"] for h in hours)


def test_pinpoint_picks_strongest_platform_peak():
    platforms = {
        "tiktok": {"top_hours": [{"hour": 16, "weight": 0.2, "label": "4:00 PM"}]},
        "youtube": {"top_hours": [{"hour": 19, "weight": 0.35, "label": "7:00 PM"}]},
    }
    pin = _pinpoint_from_platforms(platforms)
    assert pin["best_overall"]["platform"] == "youtube"
    assert pin["by_platform"]["tiktok"]["hour_local"] == 16


def test_fallback_shape():
    fb = best_posting_times_fallback(scope="fleet", error="x")
    assert fb["ok"] is False
    assert fb["scope"] == "fleet"
    assert fb["best_combinations"] == []
