"""Regression: smart-schedule batch must not concurrent-query one asyncpg conn."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch


class _ExclusiveConn:
    """Fails if two awaits touch the connection at once (asyncpg InterfaceError)."""

    def __init__(self):
        self._busy = False
        self.calls = 0

    async def _enter(self):
        if self._busy:
            raise RuntimeError("another operation is in progress")
        self._busy = True
        self.calls += 1
        await asyncio.sleep(0)
        self._busy = False


def test_gather_batch_is_sequential_on_single_conn():
    from services import smart_schedule_insights as ssi

    conn = _ExclusiveConn()
    now = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)

    async def _scores(*_a, **_k):
        await conn._enter()
        return {}

    async def _m8(*_a, **_k):
        await conn._enter()
        return {}

    async def _run():
        with patch.object(ssi, "_fetch_hour_scores_batch", new=AsyncMock(side_effect=_scores)), patch.object(
            ssi, "fetch_m8_hour_priors_batch", new=AsyncMock(side_effect=_m8)
        ):
            return await ssi._gather_batch(
                conn,
                "user-1",
                ["tiktok"],
                now,
                now,
                now,
                now,
                now,
                now,
            )

    out = asyncio.run(_run())
    assert len(out) == 5
    assert conn.calls == 5
