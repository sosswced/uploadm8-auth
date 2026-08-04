"""Unit tests for opt-in min-Trill publish skip gate."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from services.trill_min_gate import (
    should_skip_low_trill,
    trill_min_score,
    trill_skip_low_score_enabled,
)
from stages.context import JobContext, TrillScore
from stages.errors import CancelRequested


def test_skip_pref_defaults_off():
    assert trill_skip_low_score_enabled({}) is False
    assert trill_skip_low_score_enabled({"trillSkipLowScore": False}) is False
    assert trill_skip_low_score_enabled({"trill_skip_low_score": True}) is True
    assert trill_skip_low_score_enabled({"trillSkipLowScore": True}) is True


def test_min_score_clamped():
    assert trill_min_score({}) == 0
    assert trill_min_score({"trillMinScore": 60}) == 60
    assert trill_min_score({"trill_min_score": 150}) == 100
    assert trill_min_score({"trillMinScore": -5}) == 0


def test_no_skip_when_pref_off():
    skip, reason = should_skip_low_trill(
        {"trillMinScore": 60, "trillSkipLowScore": False},
        10.0,
    )
    assert skip is False
    assert reason == ""


def test_no_skip_without_score():
    skip, _ = should_skip_low_trill(
        {"trillMinScore": 60, "trillSkipLowScore": True},
        None,
    )
    assert skip is False


def test_skip_when_below_min():
    skip, reason = should_skip_low_trill(
        {"trillMinScore": 60, "trillSkipLowScore": True},
        42.0,
    )
    assert skip is True
    assert "42" in reason
    assert "60" in reason


def test_no_skip_when_at_or_above_min():
    skip, _ = should_skip_low_trill(
        {"trill_min_score": 60, "trill_skip_low_score": True},
        60.0,
    )
    assert skip is False
    skip2, _ = should_skip_low_trill(
        {"trillMinScore": 60, "trillSkipLowScore": True},
        80.0,
    )
    assert skip2 is False


def test_scenic_headroom_defers_borderline():
    # score 40 + max boost 28 = 68 >= min 60 → keep processing
    skip, _ = should_skip_low_trill(
        {"trillMinScore": 60, "trillSkipLowScore": True},
        40.0,
        allow_scenic_headroom=True,
        scenic_max_boost=28.0,
    )
    assert skip is False


def test_scenic_headroom_still_skips_hopeless():
    # score 20 + 28 = 48 < 60 → early abort
    skip, reason = should_skip_low_trill(
        {"trillMinScore": 60, "trillSkipLowScore": True},
        20.0,
        allow_scenic_headroom=True,
        scenic_max_boost=28.0,
    )
    assert skip is True
    assert "scenic" in reason.lower()


def _ctx_with_trill(score: int, *, skip: bool = True, min_score: int = 60) -> JobContext:
    ctx = JobContext(
        job_id="j-trill-gate",
        upload_id="u-trill-gate",
        user_id="user-trill",
        user_settings={
            "trillMinScore": min_score,
            "trillSkipLowScore": skip,
        },
    )
    tr = TrillScore(score=score, bucket="chill")
    ctx.trill = tr
    ctx.trill_score = tr
    return ctx


def test_worker_abort_raises_cancel_when_below_min():
    async def _run():
        import worker as worker_mod

        ctx = _ctx_with_trill(30, skip=True, min_score=60)
        mark = AsyncMock()
        save = AsyncMock()
        with patch.object(worker_mod, "db_stage") as db:
            db.mark_cancelled = mark
            db.save_trill_metadata = save
            with pytest.raises(CancelRequested):
                await worker_mod._abort_if_trill_below_min(ctx, allow_scenic_headroom=False)
        assert ctx.error_code == "TRILL_BELOW_MIN"
        assert "30" in (ctx.error_message or "")
        mark.assert_awaited_once()
        kwargs = mark.await_args.kwargs
        assert kwargs.get("error_code") == "TRILL_BELOW_MIN"
        save.assert_awaited_once()

    asyncio.run(_run())


def test_worker_abort_noop_when_pref_off_or_above_min():
    async def _run():
        import worker as worker_mod

        mark = AsyncMock()
        with patch.object(worker_mod, "db_stage") as db:
            db.mark_cancelled = mark
            db.save_trill_metadata = AsyncMock()
            await worker_mod._abort_if_trill_below_min(
                _ctx_with_trill(30, skip=False),
                allow_scenic_headroom=False,
            )
            await worker_mod._abort_if_trill_below_min(
                _ctx_with_trill(80, skip=True),
                allow_scenic_headroom=False,
            )
            # Early gate: 40 + scenic 28 can still reach 60
            await worker_mod._abort_if_trill_below_min(
                _ctx_with_trill(40, skip=True, min_score=60),
                allow_scenic_headroom=True,
            )
        mark.assert_not_awaited()

    asyncio.run(_run())


def test_apply_trill_caption_settings_strips_hype_below_min():
    import worker as worker_mod

    ctx = _ctx_with_trill(25, skip=False, min_score=60)
    ctx.trill.title_modifier = "INSANE"
    ctx.trill.hashtags = ["#trill"]
    worker_mod._apply_trill_caption_settings(ctx)
    assert ctx.trill.title_modifier == ""
    assert ctx.trill.hashtags == []
