"""Tests for scheduling preview response helper."""

from __future__ import annotations

from datetime import datetime, timezone

from services.scheduling_preview import preview_response_payload, smart_schedule_explanation


def test_preview_response_payload_shape():
    smart = {
        "tiktok": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc),
        "youtube": datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc),
    }
    sm = {k: v.isoformat() for k, v in smart.items()}
    out = preview_response_payload(
        smart,
        sm,
        seed="test-seed",
        smart_schedule_days=14,
        user_timezone="America/Chicago",
    )
    assert out["seed"] == "test-seed"
    assert out["smart_schedule"] == sm
    assert out["schedule"] == sm
    assert "tiktok" in out["explanation"]
    assert out["scheduled_time"] == min(sm.values())
    assert "batch" not in out


def test_preview_response_payload_includes_batch():
    smart = {"tiktok": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc)}
    sm = {k: v.isoformat() for k, v in smart.items()}
    batch = [
        {"index": 0, "label": "a.mp4", "seed": "s:slot-0", "smart_schedule": sm, "schedule": sm},
        {"index": 1, "label": "b.mp4", "seed": "s:slot-1", "smart_schedule": sm, "schedule": sm},
    ]
    out = preview_response_payload(
        smart,
        sm,
        seed="s",
        smart_schedule_days=7,
        batch=batch,
    )
    assert out["batch_count"] == 2
    assert len(out["batch"]) == 2
    assert out["batch"][1]["label"] == "b.mp4"


def test_occupancy_from_schedule():
    from services.scheduling_preview import occupancy_from_schedule

    now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    smart = {
        "tiktok": datetime(2026, 6, 12, 19, 0, tzinfo=timezone.utc),
        "youtube": datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc),
    }
    occ = occupancy_from_schedule(smart, now=now)
    assert occ.get(2) == 2


def test_build_smart_schedule_uses_preloaded_weights_without_db_hour_fetch(monkeypatch):
    """Batch preview must not re-query hour scores for every video slot."""
    import asyncio
    from datetime import datetime, timezone

    from services.upload import schedule_guard as sg

    calls = {"occ": 0, "driven": 0}

    async def boom_occ(*_a, **_k):
        calls["occ"] += 1
        raise AssertionError("get_existing_scheduled_days should be skipped")

    async def boom_driven(*_a, **_k):
        calls["driven"] += 1
        raise AssertionError("calculate_smart_schedule_data_driven should be skipped")

    monkeypatch.setattr(sg, "get_existing_scheduled_days", boom_occ)
    monkeypatch.setattr(sg, "calculate_smart_schedule_data_driven", boom_driven)

    weights = {"tiktok": [1.0] * 24, "youtube": [1.0] * 24}

    async def _run():
        return await sg.build_smart_schedule_for_upload(
            None,
            "user-1",
            ["tiktok", "youtube"],
            num_days=7,
            random_seed="seed:slot-0",
            user_timezone="UTC",
            base_day_occupancy={2: 1},
            hour_weights_by_platform=weights,
            extra_day_occupancy={3: 2},
        )

    out = asyncio.run(_run())
    assert calls["occ"] == 0 and calls["driven"] == 0
    assert set(out.keys()) == {"tiktok", "youtube"}
    assert all(isinstance(v, datetime) and v.tzinfo == timezone.utc for v in out.values())


def test_preview_router_preloads_once():
    import inspect

    from routers import scheduling as scheduling_router

    src = inspect.getsource(scheduling_router.preview_smart_schedule)
    assert "build_hour_weights_for_platforms_batch" in src
    assert "base_day_occupancy=" in src
    assert "hour_weights_by_platform=" in src
    assert "get_existing_scheduled_days" in src
    assert "compact_preview_batch" in src
    assert "hour_weights: Dict[str, List[float]] = {}" in src


def test_build_smart_schedule_empty_weights_skips_data_driven_sql(monkeypatch):
    """Failed hour-weight batch must not re-query SQL per video slot."""
    import asyncio
    from datetime import datetime, timezone

    from services.upload import schedule_guard as sg

    calls = {"driven": 0}

    async def boom_driven(*_a, **_k):
        calls["driven"] += 1
        raise AssertionError("calculate_smart_schedule_data_driven should be skipped")

    monkeypatch.setattr(sg, "calculate_smart_schedule_data_driven", boom_driven)

    async def _run():
        return await sg.build_smart_schedule_for_upload(
            None,
            "user-1",
            ["tiktok"],
            num_days=7,
            random_seed="seed:slot-0",
            user_timezone="UTC",
            base_day_occupancy={},
            hour_weights_by_platform={},
        )

    out = asyncio.run(_run())
    assert calls["driven"] == 0
    assert "tiktok" in out
    assert isinstance(out["tiktok"], datetime)
    assert out["tiktok"].tzinfo == timezone.utc


def test_preview_accepts_five_thousand_batch_count():
    from core.scheduling import SMART_SCHEDULE_MAX_BATCH, clamp_smart_schedule_batch
    from routers.scheduling import SchedulePreviewRequest

    assert SMART_SCHEDULE_MAX_BATCH >= 5000
    body = SchedulePreviewRequest(
        platforms=["tiktok"],
        smart_schedule_days=50,
        batch_count=5000,
    )
    assert body.batch_count == 5000
    huge = SchedulePreviewRequest(
        platforms=["tiktok"],
        smart_schedule_days=30,
        batch_count=10000,
    )
    assert huge.batch_count == 10000
    assert clamp_smart_schedule_batch(10000) == SMART_SCHEDULE_MAX_BATCH


def test_one_video_over_one_day_and_n_over_w_never_reject_density():
    """Functional contract: any N over any W packs in-window. Plan limits are separate."""
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    from core.scheduling import calculate_smart_schedule
    import core.scheduling as sched
    from services.scheduling_preview import occupancy_from_schedule

    old_now = sched._now_utc
    sched._now_utc = lambda: now
    try:
        for videos, days in ((1, 1), (7, 7), (80, 1), (500, 30)):
            extra: dict[int, int] = {}
            for i in range(videos):
                smart = calculate_smart_schedule(
                    ["tiktok", "youtube"],
                    num_days=days,
                    user_timezone="UTC",
                    day_occupancy=dict(extra),
                    random_seed=f"any-n:{videos}:{days}:{i}",
                )
                assert len(smart) == 2
                for dt in smart.values():
                    offset = (dt.astimezone(timezone.utc).date() - now.date()).days
                    assert 1 <= offset <= days, (videos, days, i, offset)
                for offset, count in occupancy_from_schedule(smart, now=now).items():
                    extra[offset] = extra.get(offset, 0) + count
    finally:
        sched._now_utc = old_now


def test_compact_preview_batch_keeps_head_and_tail():
    from services.scheduling_preview import compact_preview_batch

    rows = [{"index": i} for i in range(500)]
    shown, truncated = compact_preview_batch(rows, cap=24)
    assert truncated is True
    assert len(shown) == 24
    assert shown[0]["index"] == 0
    assert shown[-1]["index"] == 499


def test_smart_schedule_explanation_has_reason():
    smart = {"tiktok": datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc)}
    exp = smart_schedule_explanation(smart, user_timezone="UTC")
    assert "reason" in exp["tiktok"]
