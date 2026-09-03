"""HTTP IP rate-limit middleware — memory fallback + Retry-After."""

from __future__ import annotations

import asyncio

import core.state
from core.security import _json_429, rate_limit_allowed


def test_json_429_includes_retry_after():
    resp = _json_429("Rate limit exceeded (auth)", retry_after_sec=60)
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") == "60"
    body = resp.body
    assert b"rate_limited" in body


def test_rate_limit_memory_blocks_after_limit():
    core.state.redis_client = None
    core.state._RATE_BUCKETS.clear()
    key = "test:rl:ip:unit:auth"

    async def _run():
        allowed = [await rate_limit_allowed(key, 3, 60) for _ in range(3)]
        blocked = await rate_limit_allowed(key, 3, 60)
        return allowed, blocked

    allowed, blocked = asyncio.run(_run())
    assert allowed == [True, True, True]
    assert blocked is False
    core.state._RATE_BUCKETS.pop(key, None)
