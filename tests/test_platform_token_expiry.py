"""Unit tests for absolute OAuth expiry + keepalive gating."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.platform_token_expiry import (
    normalize_connect_expires_at,
    parse_access_expires_at,
    should_refresh_access_token,
    stamp_token_expiry,
)


def test_stamp_token_expiry_sets_absolute_iso():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    out = stamp_token_expiry({"access_token": "x"}, expires_in=3600, now=now)
    assert out["expires_in"] == 3600
    assert out["expires_at"] == "2026-07-30T13:00:00+00:00"
    assert out["access_obtained_at"] == now.isoformat()
    assert "access_non_expiring" not in out


def test_stamp_non_expiring_clears_expires_at():
    out = stamp_token_expiry({"expires_at": "old"}, non_expiring=True)
    assert out["expires_at"] is None
    assert out["access_non_expiring"] is True


def test_parse_legacy_ttl_seconds_relative_to_obtained_at():
    obtained = datetime(2026, 7, 30, 10, 0, 0, tzinfo=timezone.utc)
    blob = {"expires_at": 3600, "access_obtained_at": obtained.isoformat()}
    exp = parse_access_expires_at(blob, now=obtained)
    assert exp == obtained + timedelta(hours=1)


def test_parse_unix_timestamp():
    ts = datetime(2026, 7, 30, 15, 0, 0, tzinfo=timezone.utc)
    blob = {"expires_at": int(ts.timestamp())}
    assert parse_access_expires_at(blob) == ts


def test_normalize_connect_expires_at_from_ttl():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    fields = normalize_connect_expires_at(7200, now=now)
    assert fields["expires_in"] == 7200
    assert fields["expires_at"] == "2026-07-30T14:00:00+00:00"


def test_youtube_refresh_when_within_lead():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    # Expires in 10 minutes; YouTube lead is 20 minutes → due
    blob = stamp_token_expiry({}, expires_in=600, now=now)
    assert should_refresh_access_token("youtube", blob, now=now) is True


def test_youtube_skip_when_outside_lead():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    blob = stamp_token_expiry({}, expires_in=3600, now=now)
    assert should_refresh_access_token("youtube", blob, now=now) is False


def test_tiktok_refresh_within_3h_lead():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    # 2h remaining; lead is 3h → due
    blob = stamp_token_expiry({}, expires_in=7200, now=now)
    assert should_refresh_access_token("tiktok", blob, now=now) is True


def test_meta_non_expiring_cadence():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    fresh = stamp_token_expiry({}, non_expiring=True, now=now)
    assert should_refresh_access_token("instagram", fresh, now=now) is False

    old_obtained = now - timedelta(days=45)
    aged = {
        "access_non_expiring": True,
        "expires_at": None,
        "access_obtained_at": old_obtained.isoformat(),
    }
    # 50d cadence minus 7d lead → due at ~43d
    assert should_refresh_access_token("facebook", aged, now=now) is True


def test_unknown_expiry_youtube_refreshes_tiktok_refreshes_meta_unknown_age():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    empty = {}
    assert should_refresh_access_token("youtube", empty, now=now) is True
    assert should_refresh_access_token("tiktok", empty, now=now) is True
    # Meta with no obtained_at → refresh once to stamp
    assert should_refresh_access_token("instagram", empty, now=now) is True


def test_force_always_refreshes():
    now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
    blob = stamp_token_expiry({}, expires_in=86400, now=now)
    assert should_refresh_access_token("youtube", blob, now=now, force=True) is True
