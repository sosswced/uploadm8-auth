"""Tests for thumbnail_stage Pikzels failure → skip_reason mapping."""

from __future__ import annotations

from stages.thumbnail_stage import _pikzels_failures_skip_reason


def test_pikzels_failures_skip_reason_credits():
    assert (
        _pikzels_failures_skip_reason(
            [
                {"platform": "youtube", "http_status": 402, "reason": "insufficient_credits"},
                {"platform": "tiktok", "http_status": 402, "reason": "insufficient_credits"},
            ]
        )
        == "pikzels_insufficient_credits"
    )


def test_pikzels_failures_skip_reason_generic_http():
    assert (
        _pikzels_failures_skip_reason(
            [{"platform": "youtube", "http_status": 500, "reason": "pikzels_http_error"}]
        )
        == "pikzels_http_error"
    )
