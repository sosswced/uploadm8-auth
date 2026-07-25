"""Ledger reconcile: accepted publish_attempts → terminalize without re-publish."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from services.upload.orphan_processing import classify_orphan_reclaim
from services.upload.publish_ledger_reconcile import (
    attempt_row_to_platform_result,
    expected_publish_slots,
    filter_pending_targets_against_accepted_ledger,
    has_accepted_publish_attempts,
    has_fresh_pending_attempts,
    ledger_covers_expected_slots,
    platform_results_from_attempts,
    reconcile_stuck_processing_from_ledger,
    terminal_state_from_attempt_rows,
)
from stages.db import mark_processing_completed


def test_attempt_row_to_platform_result_accepted():
    pr = attempt_row_to_platform_result(
        {
            "id": "a1",
            "platform": "TikTok",
            "status": "accepted",
            "platform_post_id": "vid123",
            "platform_url": "https://tiktok.com/@x/video/1",
            "publish_id": "pub1",
            "verify_status": "pending",
            "http_status": 200,
        }
    )
    assert pr["platform"] == "tiktok"
    assert pr["success"] is True
    assert pr["platform_video_id"] == "vid123"
    assert pr["error_code"] is None


def test_terminal_state_succeeded_and_partial():
    ok_rows = [
        {"status": "accepted", "platform": "youtube"},
        {"status": "accepted", "platform": "tiktok"},
    ]
    assert terminal_state_from_attempt_rows(ok_rows) == "succeeded"
    mixed = ok_rows + [{"status": "failed", "platform": "instagram"}]
    assert terminal_state_from_attempt_rows(mixed) == "partial"
    assert terminal_state_from_attempt_rows([{"status": "failed", "platform": "x"}]) is None
    assert terminal_state_from_attempt_rows([]) is None


def test_platform_results_from_attempts_skips_pending():
    rows = [
        {"id": "1", "status": "accepted", "platform": "youtube"},
        {"id": "2", "status": "pending", "platform": "tiktok"},
        {"id": "3", "status": "failed", "platform": "instagram", "error_code": "X"},
    ]
    out = platform_results_from_attempts(rows)
    assert len(out) == 2
    assert out[0]["success"] is True
    assert out[1]["success"] is False
    assert has_accepted_publish_attempts(rows) is True


def test_classify_orphan_reclaim_ledger_complete():
    assert (
        classify_orphan_reclaim(
            {"processed_assets": {"youtube": "y.mp4"}},
            has_accepted_ledger=True,
        )
        == "ledger_complete"
    )
    assert (
        classify_orphan_reclaim({"processed_assets": {"youtube": "y.mp4"}})
        == "publish"
    )


def test_filter_pending_targets_skips_accepted_ledger_slots():
    attempts = [
        {"id": "a1", "platform": "tiktok", "status": "accepted", "publish_id": "p1"},
        {"id": "a2", "platform": "youtube", "status": "accepted", "publish_id": "p2"},
    ]
    pending = [("tiktok", "tok-tt"), ("youtube", "tok-yt"), ("instagram", "tok-ig")]
    still, synthetic = filter_pending_targets_against_accepted_ledger(
        pending, attempts, existing_platform_results=[]
    )
    assert still == [("instagram", "tok-ig")]
    assert len(synthetic) == 2
    assert synthetic[0]["token_row_id"] == "tok-tt"
    assert synthetic[0]["success"] is True


def test_filter_pending_token_scoped_does_not_cross_accounts():
    """Two TikTok accounts: accepted for t1 must not skip t2."""
    attempts = [
        {
            "id": "a1",
            "platform": "tiktok",
            "status": "accepted",
            "token_row_id": "t1",
            "publish_id": "p1",
        },
    ]
    pending = [("tiktok", "t1"), ("tiktok", "t2")]
    still, synthetic = filter_pending_targets_against_accepted_ledger(
        pending, attempts, existing_platform_results=[]
    )
    assert still == [("tiktok", "t2")]
    assert len(synthetic) == 1
    assert synthetic[0]["token_row_id"] == "t1"


def test_filter_pending_respects_existing_platform_results():
    attempts = [
        {"id": "a1", "platform": "tiktok", "status": "accepted", "token_row_id": "t1"},
        {"id": "a2", "platform": "tiktok", "status": "accepted", "token_row_id": "t2"},
    ]
    # Real path: publish_target_already_done removes t1 before ledger hydrate.
    pending = [("tiktok", "t2")]
    still, synthetic = filter_pending_targets_against_accepted_ledger(
        pending,
        attempts,
        existing_platform_results=[
            {"platform": "tiktok", "success": True, "token_row_id": "t1"},
        ],
    )
    # t1 already in platform_results → consume a1; t2 covered by a2
    assert still == []
    assert len(synthetic) == 1
    assert synthetic[0]["token_row_id"] == "t2"


def test_ledger_covers_expected_slots():
    upload = {"platforms": ["tiktok", "youtube"], "target_accounts": ["a", "b"]}
    assert expected_publish_slots(upload) == 2
    rows = [
        {"status": "accepted", "platform": "tiktok"},
        {"status": "accepted", "platform": "youtube"},
    ]
    assert ledger_covers_expected_slots(rows, upload) is True
    assert ledger_covers_expected_slots(rows[:1], upload) is False


def test_expected_slots_ignores_revoked_tokens_when_live_ids_given():
    upload = {
        "platforms": ["tiktok", "youtube"],
        "target_accounts": ["live-a", "revoked-b", "live-c"],
    }
    assert expected_publish_slots(upload) == 3
    assert expected_publish_slots(upload, live_token_ids=["live-a", "live-c"]) == 2
    rows = [
        {"status": "accepted", "platform": "tiktok"},
        {"status": "accepted", "platform": "youtube"},
    ]
    assert (
        ledger_covers_expected_slots(
            rows, upload, live_token_ids=["live-a", "live-c"]
        )
        is True
    )


def test_reconcile_stuck_processing_from_ledger_writes_succeeded():
    executed = []
    attempts = [
        {
            "id": "a1",
            "platform": "youtube",
            "status": "accepted",
            "platform_post_id": "yt1",
            "platform_url": None,
            "publish_id": "p1",
            "http_status": 200,
            "error_code": None,
            "error_message": None,
            "verify_status": "pending",
        },
        {
            "id": "a2",
            "platform": "tiktok",
            "status": "accepted",
            "platform_post_id": None,
            "platform_url": None,
            "publish_id": "p2",
            "http_status": 200,
            "error_code": None,
            "error_message": None,
            "verify_status": "pending",
        },
    ]

    class FakeConn:
        async def fetch(self, sql, *args):
            if "FROM publish_attempts" in sql:
                return attempts
            return []

        async def fetchrow(self, sql, *args):
            return {
                "status": "processing",
                "platform_results": None,
                "platforms": ["youtube", "tiktok"],
                "target_accounts": None,
            }

        async def execute(self, sql, *args):
            executed.append((sql, args))
            return "UPDATE 1"

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    result = asyncio.run(
        reconcile_stuck_processing_from_ledger(
            FakePool(),
            "9e677d10-60a2-4fe2-96aa-e41d220c3552",
            user_id="user-1",
        )
    )
    assert result["ok"] is True
    assert result["state"] == "succeeded"
    assert result["reason"] == "reconciled_from_ledger"
    assert result["accepted_count"] == 2
    assert result["platform_results_count"] == 2
    assert executed, "expected UPDATE uploads"
    assert "failure_phase" in executed[0][0]
    assert executed[0][1][1] == "succeeded"


def test_reconcile_refuses_partial_ledger():
    attempts = [
        {
            "id": "a1",
            "platform": "youtube",
            "status": "accepted",
            "platform_post_id": "yt1",
            "platform_url": None,
            "publish_id": "p1",
            "http_status": 200,
            "error_code": None,
            "error_message": None,
            "verify_status": "pending",
        },
    ]

    class FakeConn:
        async def fetch(self, sql, *args):
            return attempts

        async def fetchrow(self, sql, *args):
            return {
                "status": "processing",
                "platform_results": None,
                "platforms": ["youtube", "tiktok", "instagram"],
                "target_accounts": ["t1", "t2", "t3"],
            }

        async def execute(self, sql, *args):
            raise AssertionError("must not terminalize partial ledger")

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    result = asyncio.run(
        reconcile_stuck_processing_from_ledger(FakePool(), "u-partial")
    )
    assert result["ok"] is False
    assert result["reason"] == "accepted_below_expected"


def test_reconcile_no_accepted_does_not_write():
    class FakeConn:
        async def fetch(self, sql, *args):
            return [{"id": "a1", "platform": "youtube", "status": "pending"}]

        async def fetchrow(self, sql, *args):
            raise AssertionError("should not load upload when no accepted")

        async def execute(self, sql, *args):
            raise AssertionError("should not update")

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    result = asyncio.run(
        reconcile_stuck_processing_from_ledger(FakePool(), "u-1")
    )
    assert result["ok"] is False
    assert result["reason"] == "no_accepted_attempts"


def test_mark_processing_completed_clears_failure_phase_flag():
    executed = []

    class FakeConn:
        async def execute(self, sql, *args):
            executed.append((sql, args))
            return "UPDATE 1"

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    ctx = MagicMock()
    ctx.upload_id = "62936293-4ad2-48f3-b372-3c2b50d6ae99"
    ctx.state = "succeeded"
    ctx.finished_at = datetime.now(timezone.utc)
    ctx.error_code = None
    ctx.error_message = None
    ctx.compute_seconds = 12
    ctx.thumbnail_r2_key = None
    ctx.platform_results = []

    mode = asyncio.run(mark_processing_completed(FakePool(), ctx))
    assert mode == "full"
    assert executed
    sql, args = executed[0]
    assert "failure_phase" in sql
    # clear_failure_phase is the last bind (True for succeeded)
    assert args[-1] is True


# --- has_fresh_pending_attempts: second-publisher dispatch gate --------------


def _pending_row(age_seconds: float):
    from datetime import timedelta

    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return {"status": "pending", "updated_at": ts, "created_at": ts}


def test_fresh_pending_blocks_redispatch():
    assert has_fresh_pending_attempts([_pending_row(60)]) is True
    assert has_fresh_pending_attempts([_pending_row(1799)]) is True


def test_aged_pending_allows_redispatch():
    assert has_fresh_pending_attempts([_pending_row(1801)]) is False


def test_pending_without_timestamp_counts_as_fresh():
    assert has_fresh_pending_attempts(
        [{"status": "pending", "updated_at": None, "created_at": None}]
    ) is True


def test_non_pending_rows_ignored():
    rows = [
        {"status": "accepted", "updated_at": datetime.now(timezone.utc)},
        {"status": "failed", "updated_at": datetime.now(timezone.utc)},
    ]
    assert has_fresh_pending_attempts(rows) is False
    assert has_fresh_pending_attempts([]) is False
    assert has_fresh_pending_attempts(None) is False


def test_naive_timestamp_treated_as_utc():
    naive = datetime.utcnow()  # naive but "now" in UTC → fresh
    assert has_fresh_pending_attempts(
        [{"status": "pending", "updated_at": naive, "created_at": None}]
    ) is True


def test_recovery_paths_use_fresh_pending_gate():
    """Orphan recovery and stuck-RTP recovery must never redispatch publish on
    top of an in-flight Step A (fresh pending ledger slot)."""
    import inspect

    import worker
    from services.upload import stuck_recovery

    orphan_src = inspect.getsource(worker.run_orphan_processing_recovery_loop)
    assert "has_fresh_pending_attempts" in orphan_src

    rtp_src = inspect.getsource(stuck_recovery.recover_stuck_ready_to_publish)
    assert "has_fresh_pending_attempts" in rtp_src


def test_stale_processing_reclaim_respects_heartbeat_ownership():
    """Stale recovery must never reclaim a processing row a live/stale worker
    still lists in its heartbeat — long FFmpeg encodes only touch updated_at at
    stage entry and can exceed STALE_PROCESSING_MINUTES while healthy."""
    import inspect

    import worker

    src = inspect.getsource(worker.run_stale_job_recovery_loop)
    assert "owning_worker_id" in src
    assert "fetch_fleet_workers" in src


def test_orphan_recovery_fail_closed_on_ledger_complete():
    """When the ledger already covers all slots, orphan recovery must not
    fall through to reset→queued→re-enqueue (double-post / re-encode risk)."""
    import inspect

    import worker

    src = inspect.getsource(worker.run_orphan_processing_recovery_loop)
    assert 'reclaim == "ledger_complete"' in src
    assert "reconcile failed" in src
    assert "leaving processing" in src


def test_publish_dispatch_tracks_ownership_before_semaphore():
    """Heartbeat ownership must start BEFORE the publish semaphore wait, or
    orphan recovery on another worker reclaims the row mid-wait and dispatches
    a duplicate publisher."""
    import inspect

    import worker

    for fn in (worker._run_deferred_publish_with_semaphore, worker._publish_one_job):
        src = inspect.getsource(fn)
        start = src.index("track_publish_start(")
        sem = src.index("async with _publish_semaphore", start)
        assert start < sem, f"{fn.__name__} must track before semaphore"
        assert "track_publish_end(" in src[sem:]


def test_recover_deferred_publish_false_failure_hydrates_and_terminalizes():
    """jsonb bind false-failure: empty ctx + accepted ledger → recover, no fail."""
    from services.upload.publish_ledger_reconcile import (
        recover_deferred_publish_false_failure,
    )

    attempts = [
        {
            "id": "a1",
            "platform": "tiktok",
            "status": "accepted",
            "platform_post_id": "vid1",
            "platform_url": None,
            "publish_id": "p1",
            "http_status": 200,
            "error_code": None,
            "error_message": None,
            "verify_status": "pending",
            "token_row_id": None,
        }
    ]
    executed = []

    class FakeConn:
        async def fetch(self, sql, *args):
            if "publish_attempts" in sql:
                return attempts
            return []

        async def fetchrow(self, sql, *args):
            return {
                "status": "processing",
                "platform_results": None,
                "platforms": ["tiktok"],
                "target_accounts": None,
            }

        async def execute(self, sql, *args):
            executed.append((sql, args))
            return "UPDATE 1"

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    class Ctx:
        def __init__(self):
            self.platform_results = []
            self.state = "processing"

        def is_success(self):
            return any(getattr(r, "success", False) for r in self.platform_results)

        def is_partial_success(self):
            return False

    ctx = Ctx()
    out = asyncio.run(
        recover_deferred_publish_false_failure(
            FakePool(),
            "41c86019-6f3a-4799-ab90-50394efdb0e6",
            user_id="0af99456-1002-49f8-8554-e4d4405e5884",
            ctx=ctx,
        )
    )
    assert out["recovered"] is True
    assert out["hydrated"] is True
    assert out["state"] == "succeeded"
    assert ctx.is_success() is True
    assert any("platform_results" in sql for sql, _ in executed)


def test_recover_deferred_publish_false_failure_no_ledger():
    from services.upload.publish_ledger_reconcile import (
        recover_deferred_publish_false_failure,
    )

    class FakeConn:
        async def fetch(self, sql, *args):
            return []

        async def fetchrow(self, sql, *args):
            return None

        async def execute(self, sql, *args):
            return "UPDATE 0"

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    out = asyncio.run(
        recover_deferred_publish_false_failure(
            FakePool(), "00000000-0000-0000-0000-000000000001"
        )
    )
    assert out["recovered"] is False
    assert out["reason"] == "no_accepted_attempts"


def test_deferred_publish_except_uses_ledger_false_failure_recovery():
    """Worker deferred-publish except must call ledger recover before upload_failed."""
    import inspect

    import worker

    src = inspect.getsource(worker.run_deferred_publish)
    assert "recover_deferred_publish_false_failure" in src
    assert src.index("recover_deferred_publish_false_failure") < src.index(
        "deferred_publish_failure"
    )
