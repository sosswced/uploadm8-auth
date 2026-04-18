"""Unit tests for services.ops_readiness."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import asyncpg

from services.ops_readiness import run_readiness_checks


def _mock_pool_ok():
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=cm)
    return pool


def test_readiness_db_ok_redis_not_configured():
    pool = _mock_pool_ok()

    async def run():
        return await run_readiness_checks(pool, None)

    ok, checks = asyncio.run(run())
    assert ok is True
    assert checks["database"] == "ok"
    assert checks["redis"] == "not_configured"


def test_readiness_db_failure():
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=asyncpg.exceptions.InterfaceError("pool closed"))

    async def run():
        return await run_readiness_checks(pool, None)

    ok, checks = asyncio.run(run())
    assert ok is False
    assert checks["database"].startswith("error:")
    assert checks["redis"] == "not_configured"


def test_readiness_redis_failure():
    pool = _mock_pool_ok()
    redis = AsyncMock()
    redis.ping = AsyncMock(side_effect=ConnectionError("refused"))

    async def run():
        return await run_readiness_checks(pool, redis)

    ok, checks = asyncio.run(run())
    assert ok is False
    assert checks["database"] == "ok"
    assert checks["redis"].startswith("error:")
