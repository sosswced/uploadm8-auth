"""Unit tests for upload dupe + progress finish gate."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_insert_publish_attempt_fails_closed_on_generic_db_error():
    """Transient DB errors must skip the platform API, never risk a duplicate post."""
    from stages.db import insert_publish_attempt

    class _BoomPool:
        def acquire(self):
            raise RuntimeError("pool down")

    attempt_id, skip_api = asyncio.run(
        insert_publish_attempt(_BoomPool(), "u-1", "user-1", "tiktok", token_row_id=None)
    )
    assert attempt_id is None
    assert skip_api is True


def test_insert_publish_attempt_fails_open_only_when_table_missing():
    """Pre-migration bootstrap (no publish_attempts table) keeps publishing enabled."""
    import asyncpg

    from stages.db import insert_publish_attempt

    class _Conn:
        async def fetchrow(self, *_a, **_k):
            raise asyncpg.exceptions.UndefinedTableError("no table")

        async def execute(self, *_a, **_k):
            raise asyncpg.exceptions.UndefinedTableError("no table")

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_a):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    attempt_id, skip_api = asyncio.run(
        insert_publish_attempt(_Pool(), "u-1", "user-1", "tiktok", token_row_id=None)
    )
    assert attempt_id is None
    assert skip_api is False


def test_publish_stage_heartbeats_updated_at_in_fanout():
    """Long fan-outs must refresh uploads.updated_at so stale recovery leaves them alone."""
    import inspect

    from stages import publish_stage as ps

    src = inspect.getsource(ps)
    assert "UPDATE uploads SET updated_at = NOW() WHERE id = $1::uuid AND status = 'processing'" in src


def test_publish_stage_ledger_claim_exception_is_fail_closed():
    import inspect

    from stages import publish_stage as ps

    src = inspect.getsource(ps)
    marker = "Could not create publish_attempt row (fail-closed, skip)"
    assert marker in src
    after = src[src.index(marker):src.index(marker) + 400]
    assert "skip_api = True" in after


def _ledger_pool(pending_row, reclaim_result):
    """Pool stub: fetchrow returns the open slot, fetchval answers the reclaim UPDATE."""

    class _Conn:
        async def fetchrow(self, *_a, **_k):
            return pending_row

        async def fetchval(self, *_a, **_k):
            return reclaim_result

        async def execute(self, *_a, **_k):
            return "INSERT 0 1"

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_a):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    return _Pool()


def test_insert_publish_attempt_reclaims_aged_pending_slot():
    """Aged pending (owner died mid-call) must be reclaimable or retries deadlock:
    recovery redispatches past has_fresh_pending_attempts but the API would be
    skipped forever, terminalizing the upload as failed."""
    from stages.db import insert_publish_attempt

    row = {"id": "slot-1", "status": "pending", "updated_at": None, "created_at": None}
    attempt_id, skip_api = asyncio.run(
        insert_publish_attempt(
            _ledger_pool(row, "slot-1"), "u-1", "user-1", "tiktok", token_row_id=None
        )
    )
    assert attempt_id == "slot-1"
    assert skip_api is False


def test_insert_publish_attempt_skips_fresh_pending_slot():
    from stages.db import insert_publish_attempt

    row = {"id": "slot-1", "status": "pending", "updated_at": None, "created_at": None}
    attempt_id, skip_api = asyncio.run(
        insert_publish_attempt(
            _ledger_pool(row, None), "u-1", "user-1", "tiktok", token_row_id=None
        )
    )
    assert attempt_id == "slot-1"
    assert skip_api is True


def test_publish_stand_down_when_all_targets_inflight_elsewhere():
    """Empty batch + fresh pending slots owned elsewhere must NOT terminalize failed."""
    import inspect

    import worker

    src = inspect.getsource(worker.run_publish_and_notify)
    marker = "publish stand-down"
    assert marker in src
    # Stand-down must occur before the terminal-state decision block.
    assert src.index(marker) < src.index("# Decide terminal state")


def test_publish_stage_counts_inflight_skips():
    import inspect

    from stages import publish_stage as ps

    src = inspect.getsource(ps.run_publish_stage)
    assert "publish_inflight_skips" in src


def test_presign_duplicate_guard_covers_completed_status():
    import inspect

    from services.upload import presign

    src = inspect.getsource(presign.reject_recent_duplicate_source)
    assert "'completed'" in src


def test_filter_token_scoped_skip_matches_plan():
    from services.upload.publish_ledger_reconcile import (
        filter_pending_targets_against_accepted_ledger,
    )

    attempts = [
        {"id": "a1", "platform": "youtube", "status": "accepted", "token_row_id": "yt1"},
        {"id": "a2", "platform": "tiktok", "status": "accepted"},  # legacy null token
    ]
    pending = [("youtube", "yt1"), ("youtube", "yt2"), ("tiktok", "tt1")]
    still, synthetic = filter_pending_targets_against_accepted_ledger(
        pending, attempts, existing_platform_results=[]
    )
    assert ("youtube", "yt1") not in still
    assert ("youtube", "yt2") in still
    assert ("tiktok", "tt1") not in still  # legacy platform bucket
    assert len(synthetic) == 2


def test_hydrate_fail_closed_aborts_when_attempts_exist(tmp_path):
    """publish_stage must not continue fan-out when hydrate throws with ledger rows."""
    from pathlib import Path

    from stages import publish_stage as ps

    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")

    ctx = SimpleNamespace(
        upload_id="u-1",
        user_id="user-1",
        platforms=["tiktok"],
        target_accounts=[],
        deferred_publish_platform_filter=None,
        platform_results=[],
        platform_videos={"tiktok": video},
        processed_video_path=video,
        local_video_path=video,
        output_artifacts={},
        processed_r2_key=None,
        get_video_for_platform=lambda _p: video,
    )

    async def _boom(*_a, **_k):
        raise RuntimeError("hydrate exploded")

    async def _run():
        with patch.object(
            ps, "resolve_publish_targets", AsyncMock(return_value=[("tiktok", "t1")])
        ), patch.object(ps, "init_enc_keys", lambda: None), patch.object(
            ps, "coerce_processed_assets_map", return_value={}
        ), patch.object(ps, "logger"), patch(
            "services.deferred_publish_schedule.publish_target_already_done",
            return_value=False,
        ), patch(
            "services.upload.publish_ledger_reconcile.load_publish_attempts_for_upload",
            AsyncMock(return_value=[{"id": "a1", "status": "accepted", "platform": "tiktok"}]),
        ), patch(
            "services.upload.publish_ledger_reconcile.hydrate_ctx_from_accepted_ledger",
            _boom,
        ):
            await ps.run_publish_stage(ctx, db_pool=MagicMock())

    with pytest.raises(RuntimeError, match="hydrate exploded"):
        asyncio.run(_run())


def test_resolve_ready_publish_targets_uses_token_platforms():
    from services.upload import stuck_recovery as sr

    class Conn:
        async def fetch(self, sql, *args):
            assert "platform_tokens" in sql
            return [
                {"id": "tok-tt", "platform": "tiktok"},
                {"id": "tok-yt", "platform": "youtube"},
            ]

    upload = {
        "platforms": ["tiktok", "youtube"],
        "target_accounts": ["tok-tt", "tok-yt"],
    }
    pairs = asyncio.run(sr._resolve_ready_publish_targets(Conn(), upload))
    assert pairs == [("tiktok", "tok-tt"), ("youtube", "tok-yt")]


def test_ready_recovery_reconciles_when_ledger_complete(monkeypatch):
    from services.upload import stuck_recovery as sr

    now = datetime.now(timezone.utc)
    row = {
        "id": "u-led",
        "user_id": "user-1",
        "schedule_mode": "immediate",
        "platforms": ["tiktok", "youtube"],
        "schedule_metadata": None,
        "scheduled_time": None,
        "platform_results": None,
        "target_accounts": ["t1", "t2"],
        "processed_assets": None,
        "updated_at": now - timedelta(hours=2),
        "created_at": now - timedelta(hours=3),
    }
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[row])
    dispatch = AsyncMock()

    monkeypatch.setattr(
        "services.upload.publish_ledger_reconcile.load_publish_attempts_for_upload",
        AsyncMock(
            return_value=[
                {"status": "accepted", "platform": "tiktok"},
                {"status": "accepted", "platform": "youtube"},
            ]
        ),
    )
    monkeypatch.setattr(
        "services.upload.publish_ledger_reconcile.ledger_covers_expected_slots",
        lambda *_a, **_k: True,
    )
    reconcile = AsyncMock(return_value={"ok": True, "reason": "reconciled_from_ledger"})
    monkeypatch.setattr(
        "services.upload.publish_ledger_reconcile.reconcile_stuck_processing_from_ledger",
        reconcile,
    )

    stats = asyncio.run(
        sr.recover_stuck_ready_to_publish(
            conn, MagicMock(), dispatch_publish=dispatch, limit=5
        )
    )
    assert stats["reconciled"] == 1
    assert stats["redispatched"] == 0
    dispatch.assert_not_awaited()
    reconcile.assert_awaited()
