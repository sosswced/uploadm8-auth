"""UPLOADM8-82: refresh-all must batch account metric event inserts."""

import asyncio
from unittest.mock import AsyncMock


def test_store_account_metric_events_batch_one_executemany():
    from services import platform_metrics_job as pmj

    conn = AsyncMock()
    conn.executemany = AsyncMock()

    async def _run():
        await pmj._store_account_metric_events_batch(
            conn,
            user_id="user-1",
            rows=[
                ("tiktok", "tok-1", "acc-1", {"views": 1}),
                ("youtube", "tok-2", "acc-2", {"views": 2}),
            ],
        )

    asyncio.run(_run())
    assert conn.executemany.await_count == 1
    args = conn.executemany.await_args.args
    assert "platform_account_metrics_events" in args[0]
    assert len(args[1]) == 2
