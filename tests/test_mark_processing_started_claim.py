"""Processing claim must not revive failed uploads."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from stages.db import mark_processing_started


def test_mark_processing_started_only_claims_queued_or_staged():
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
    ctx.upload_id = "adb8a4f6-2ac1-4e26-a394-bb2a9859b458"
    ctx.started_at = datetime.now(timezone.utc)

    claimed = asyncio.run(mark_processing_started(FakePool(), ctx))
    assert claimed is True
    sql = executed[0][0].lower().replace("\n", " ")
    compact = sql.replace(" ", "")
    assert "statusin('queued','staged')" in compact
    # Orphan processing (stage never set) must also be claimable.
    assert "status='processing'" in compact
    assert "processing_stage" in sql
    # Single-winner: empty stage becomes 'claimed' so a second worker loses the race.
    assert "claimed" in sql


def test_mark_processing_started_returns_false_when_no_row_updated():
    class FakeConn:
        async def execute(self, *_a, **_k):
            return "UPDATE 0"

    class FakePool:
        def acquire(self):
            return self

        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *_a):
            return False

    ctx = MagicMock()
    ctx.upload_id = "adb8a4f6-2ac1-4e26-a394-bb2a9859b458"
    ctx.started_at = None

    claimed = asyncio.run(mark_processing_started(FakePool(), ctx))
    assert claimed is False
