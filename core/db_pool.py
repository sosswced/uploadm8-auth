"""Database pool helpers — transient connection retry on acquire."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

logger = logging.getLogger("uploadm8-api")

_TRANSIENT_DB_ERRORS = (
    asyncpg.CannotConnectNowError,
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.InterfaceError,
    asyncpg.PostgresConnectionError,
    asyncpg.PostgresError,  # recovery mode / cluster restart
    OSError,
)


def is_transient_db_error(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT_DB_ERRORS)


@asynccontextmanager
async def acquire_db(pool, *, attempts: int = 3) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pool connection; retry on stale socket / cluster recovery errors."""
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            async with pool.acquire() as conn:
                yield conn
                return
        except _TRANSIENT_DB_ERRORS as e:
            last_err = e
            if attempt < attempts:
                logger.warning(
                    "db acquire transient error (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    e,
                )
                await asyncio.sleep(0.15 * attempt)
                continue
            raise
    if last_err is not None:
        raise last_err
