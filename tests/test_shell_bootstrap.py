"""Shell bootstrap helpers (unit-level, no DB)."""

import asyncio
from unittest.mock import AsyncMock, patch

from services.shell_bootstrap import _bootstrap_schedule_repair, _ok


def test_ok_coerces_exceptions_to_none():
    assert _ok({"uploads": []}) == {"uploads": []}
    assert _ok(RuntimeError("boom")) is None


def test_bootstrap_schedule_repair_timeout_returns_none():
    async def _wait_for_timeout(coro, *_a, **_k):
        if hasattr(coro, "close"):
            coro.close()
        raise asyncio.TimeoutError

    async def _run():
        with patch("services.shell_bootstrap.asyncio.wait_for", new=_wait_for_timeout):
            return await _bootstrap_schedule_repair(object(), "user-1")

    result = asyncio.run(_run())
    assert result is None
