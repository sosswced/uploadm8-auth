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
    *,
    db_timeout_s: float = 8.0,
) -> Tuple[bool, Dict[str, str]]:
    """
    Run dependency checks. Returns (healthy, checks) where checks values are short status strings.

    DB acquire + ping are bounded so a saturated/wedged pool cannot hang ``/ready``
    forever while ``/health`` still returns 200.
    """
    checks: Dict[str, str] = {}
    healthy = True

    if db_pool is None:
        checks["database"] = "error: pool not ready"
        healthy = False
    else:
        conn = None
        try:
            conn = await asyncio.wait_for(db_pool.acquire(timeout=db_timeout_s), timeout=db_timeout_s)
            await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=db_timeout_s)
            checks["database"] = "ok"
        except (
            asyncpg.exceptions.PostgresError,
            asyncpg.exceptions.InterfaceError,
            asyncio.TimeoutError,
            TimeoutError,
            OSError,
        ) as e:
            checks["database"] = f"error: {type(e).__name__}"
            healthy = False
        finally:
            if conn is not None:
                try:
                    await db_pool.release(conn)
                except Exception:
                    pass

    if redis_client:
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=min(5.0, db_timeout_s))
            checks["redis"] = "ok"
        except (RedisError, asyncio.TimeoutError, TimeoutError, OSError) as e:
            checks["redis"] = f"error: {type(e).__name__}"
            healthy = False
    else:
        checks["redis"] = "not_configured"

    return healthy, checks
