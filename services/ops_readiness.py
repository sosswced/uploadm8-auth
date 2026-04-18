"""Readiness probe logic for /ready (database + optional Redis)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

import asyncpg

try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover
    RedisError = Exception  # type: ignore[misc, assignment]


async def run_readiness_checks(
    db_pool: Any,
    redis_client: Any | None,
) -> Tuple[bool, Dict[str, str]]:
    """
    Run dependency checks. Returns (healthy, checks) where checks values are short status strings.
    """
    checks: Dict[str, str] = {}
    healthy = True

    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = "ok"
    except (
        asyncpg.exceptions.PostgresError,
        asyncpg.exceptions.InterfaceError,
        asyncio.TimeoutError,
        TimeoutError,
        OSError,
    ) as e:
        checks["database"] = f"error: {e}"
        healthy = False

    if redis_client:
        try:
            await redis_client.ping()
            checks["redis"] = "ok"
        except (RedisError, asyncio.TimeoutError, TimeoutError, OSError) as e:
            checks["redis"] = f"error: {e}"
            healthy = False
    else:
        checks["redis"] = "not_configured"

    return healthy, checks
