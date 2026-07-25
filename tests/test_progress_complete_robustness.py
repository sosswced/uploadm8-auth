"""Progress heartbeats + terminal progress for accurate upload completion UX."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from stages.db import mark_processing_completed
from services.upload.stage_labels import stage_label_for


def test_mark_processing_completed_sets_done_progress_100():
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
    ctx.compute_seconds = 5
    ctx.thumbnail_r2_key = None
    ctx.platform_results = []

    mode = asyncio.run(mark_processing_completed(FakePool(), ctx))
    assert mode == "full"
    sql = executed[0][0]
    assert "processing_stage" in sql
    assert "processing_progress" in sql
    assert "done" in sql
    assert stage_label_for("done") == "Complete"
