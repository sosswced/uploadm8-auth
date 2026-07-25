"""mark_processing_completed must terminalize status even if jsonb bind fails."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from stages.db import mark_processing_completed


def test_mark_processing_completed_falls_back_to_status_first():
    executed = []

    class FakeConn:
        async def execute(self, sql, *args):
            executed.append((sql, args))
            # First call is the full write (includes platform_results) — fail it.
            if len(executed) == 1 and "platform_results" in sql:
                raise TypeError("expected str, got list")
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
    # Non-empty platform_results so _platform_results_payload returns a list.
    pr = MagicMock()
    pr.platform = "tiktok"
    pr.success = True
    pr.platform_video_id = None
    pr.platform_url = None
    pr.publish_id = "v_pub"
    pr.error_code = None
    pr.error_message = None
    pr.verify_status = None
    pr.http_status = None
    pr.views = None
    pr.likes = None
    ctx.platform_results = [pr]

    mode = asyncio.run(mark_processing_completed(FakePool(), ctx))
    assert mode == "degraded"
    assert len(executed) >= 2
    # Second statement terminalizes status without requiring jsonb success first.
    assert "status" in executed[1][0].lower()
    assert executed[1][1][1] == "succeeded"
    assert "processing_progress" in executed[1][0]
    assert "processing_stage" in executed[1][0]
