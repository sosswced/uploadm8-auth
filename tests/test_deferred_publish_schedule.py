"""Unit tests for per-platform deferred smart publish scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.deferred_publish_schedule import (
    handled_target_keys,
    hydrate_platform_results_into_ctx,
    next_due_scheduled_time,
    parse_schedule_metadata,
    platforms_due_for_publish,
    publish_target_already_done,
    publish_target_key,
    still_has_pending_publish_slots,
)
from stages.context import JobContext, PlatformResult


def _utc(y, m, d, h=12, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_parse_schedule_metadata():
    sm = parse_schedule_metadata({"TikTok": "2026-06-15T19:00:00+00:00", "youtube": "2026-06-17T14:00:00Z"})
    assert "tiktok" in sm
    assert "youtube" in sm
    assert sm["youtube"].day == 17


def test_platforms_due_smart_only_publishes_due_platform():
    now = _utc(2026, 6, 15, 20, 0)
    upload = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": {
            "tiktok": "2026-06-15T19:00:00+00:00",
            "youtube": "2026-06-17T14:00:00+00:00",
        },
        "scheduled_time": _utc(2026, 6, 15, 19, 0),
        "platform_results": None,
    }
    due = platforms_due_for_publish(upload, now)
    assert due == frozenset({"tiktok"})


def test_platforms_due_skips_already_handled_platform():
    now = _utc(2026, 6, 17, 15, 0)
    upload = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": {
            "tiktok": "2026-06-15T19:00:00+00:00",
            "youtube": "2026-06-17T14:00:00+00:00",
        },
        "scheduled_time": _utc(2026, 6, 15, 19, 0),
        "platform_results": [
            {"platform": "tiktok", "success": True, "token_row_id": None},
        ],
    }
    due = platforms_due_for_publish(upload, now)
    assert due == frozenset({"youtube"})


def test_still_has_pending_until_all_targets_handled():
    upload = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
    }
    assert still_has_pending_publish_slots(upload, [], publish_targets=[("tiktok", None), ("youtube", None)])
    partial = [{"platform": "tiktok", "success": True}]
    assert still_has_pending_publish_slots(upload, partial, publish_targets=[("tiktok", None), ("youtube", None)])
    done = [
        {"platform": "tiktok", "success": True},
        {"platform": "youtube", "success": True},
    ]
    assert not still_has_pending_publish_slots(upload, done, publish_targets=[("tiktok", None), ("youtube", None)])


def test_publish_target_already_done_respects_token():
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="user",
        platform_results=[
            PlatformResult(platform="tiktok", success=True, token_row_id="aaa-bbb"),
        ],
    )
    assert publish_target_already_done(ctx, "tiktok", "aaa-bbb")
    assert not publish_target_already_done(ctx, "tiktok", "ccc-ddd")


def test_non_smart_scheduled_all_at_once():
    now = _utc(2026, 6, 10, 12, 0)
    st = now - timedelta(hours=1)
    upload = {
        "schedule_mode": "scheduled",
        "platforms": ["tiktok", "youtube"],
        "scheduled_time": st,
        "platform_results": None,
    }
    due = platforms_due_for_publish(upload, now)
    assert due == frozenset({"tiktok", "youtube"})
    upload["platform_results"] = [{"platform": "tiktok", "success": True}]
    assert platforms_due_for_publish(upload, now) == frozenset()


def test_hydrate_platform_results_into_ctx():
    ctx = JobContext(job_id="j", upload_id="u", user_id="user")
    hydrate_platform_results_into_ctx(
        ctx,
        [{"platform": "youtube", "success": True, "platform_video_id": "vid1"}],
    )
    assert len(ctx.platform_results) == 1
    assert ctx.platform_results[0].platform_video_id == "vid1"


def test_handled_target_keys():
    keys = handled_target_keys(
        [{"platform": "tiktok", "success": True, "token_row_id": "t1"}]
    )
    assert publish_target_key("tiktok", "t1") in keys


def test_next_due_scheduled_time_advances_past_published_platform():
    upload = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube", "instagram"],
        "schedule_metadata": {
            "tiktok": "2026-06-15T19:00:00+00:00",
            "youtube": "2026-06-17T14:00:00+00:00",
            "instagram": "2026-06-19T18:00:00+00:00",
        },
        "scheduled_time": _utc(2026, 6, 15, 19, 0),
        "platform_results": [{"platform": "tiktok", "success": True}],
    }
    nxt = next_due_scheduled_time(upload)
    assert nxt is not None
    assert nxt == _utc(2026, 6, 17, 14, 0)


def test_next_due_scheduled_time_none_when_all_handled():
    upload = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": {
            "tiktok": "2026-06-15T19:00:00+00:00",
            "youtube": "2026-06-17T14:00:00+00:00",
        },
        "platform_results": [
            {"platform": "tiktok", "success": True},
            {"platform": "youtube", "success": True},
        ],
    }
    assert next_due_scheduled_time(upload) is None


def test_next_due_none_when_metadata_incomplete_and_no_scheduled_time():
    upload = {
        "schedule_mode": "smart",
        "platforms": ["tiktok", "youtube", "instagram", "facebook"],
        "schedule_metadata": {"tiktok": "2026-06-15T19:00:00+00:00"},
        "scheduled_time": None,
        "platform_results": [{"platform": "tiktok", "success": True}],
    }
    assert next_due_scheduled_time(upload) is None
    assert still_has_pending_publish_slots(upload, upload["platform_results"]) is True
