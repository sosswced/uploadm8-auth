"""Database pool helpers — transient connection retry on acquire."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

logger = logging.getLogger("uploadm8-api")

_TRANSIENT_DB_ERRORS = (
    asyncpg.CannotConnectNowError,
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.InterfaceError,  # released mid-query — retry on fresh connection
    asyncpg.PostgresConnectionError,
    asyncpg.PostgresError,  # recovery mode / cluster restart
    OSError,
    TimeoutError,  # pool.acquire(timeout=...) / asyncio.timeout
    asyncio.TimeoutError,
)


class DatabaseUnavailableError(Exception):
    """Pool acquire/ping exhausted — map to HTTP 503 (handled), not unhandled 500."""

    def __init__(self, message: str = "Database temporarily unavailable", *, cause: BaseException | None = None):
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


def is_transient_db_error(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT_DB_ERRORS) or isinstance(exc, DatabaseUnavailableError)


_DEAD_CONNECTION_ERRORS = (
    asyncpg.CannotConnectNowError,
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.InterfaceError,
    asyncpg.PostgresConnectionError,
    OSError,
    TimeoutError,
    asyncio.TimeoutError,
)


def is_dead_connection_error(exc: BaseException) -> bool:
    """True when the pool connection is unusable — do not retry SQL on the same conn."""
    return isinstance(exc, _DEAD_CONNECTION_ERRORS) or isinstance(exc, DatabaseUnavailableError)


def _acquire_attempts() -> int:
    try:
        return max(3, int(os.environ.get("DB_ACQUIRE_ATTEMPTS", "8")))
    except ValueError:
        return 8


def _acquire_timeout_attempts() -> int:
    """Pool-queue timeouts mean the pool is saturated — fail faster than Neon wake retries."""
    try:
        return max(1, int(os.environ.get("DB_ACQUIRE_TIMEOUT_ATTEMPTS", "2")))
    except ValueError:
        return 2


def _acquire_timeout_s() -> float:
    """Fail fast on wedged Neon sockets instead of hanging request workers."""
    try:
        return max(2.0, float(os.environ.get("DB_ACQUIRE_TIMEOUT_S", "8")))
    except ValueError:
        return 8.0


def _backoff_s(attempt: int) -> float:
    """Exponential backoff for Neon wake / cluster recovery (caps at 8s)."""
    return min(8.0, 0.35 * (2 ** (attempt - 1)))


def _is_timeout_exc(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError))


async def _discard_bad_conn(pool, conn) -> None:
    """Drop a dead/half-open connection instead of returning it to the pool."""
    if conn is None:
        return
    try:
        terminate = getattr(conn, "terminate", None)
        if callable(terminate):
            terminate()
    except Exception:
        pass
    try:
        await pool.release(conn)
    except Exception:
        pass


@asynccontextmanager
async def acquire_db(pool, *, attempts: int | None = None) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pool connection; retry on stale socket / cluster recovery errors.

    After acquire, runs ``SELECT 1`` so dead Neon sockets fail before the handler
    starts work (avoids Slow DB spans that are really connection-wait).

    Retries only apply to acquire + ping — never after the caller has started work
    (retrying post-yield breaks ``@asynccontextmanager`` → RuntimeError generator
    didn't stop).

    Exhausted retries raise ``DatabaseUnavailableError`` (HTTP 503) so Starlette/Sentry
    do not treat pool saturation as an unhandled 500.
    """
    max_attempts = attempts if attempts is not None else _acquire_attempts()
    timeout_cap = min(max_attempts, _acquire_timeout_attempts())
    acquire_timeout = _acquire_timeout_s()
    last_err: Exception | None = None
    conn: asyncpg.Connection | None = None
    timeout_failures = 0
    for attempt in range(1, max_attempts + 1):
        try:
            conn = await pool.acquire(timeout=acquire_timeout)
            # Fail fast on half-open sockets left after Neon suspend/resume.
            await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=acquire_timeout)
            last_err = None
            break
        except _TRANSIENT_DB_ERRORS as e:
            last_err = e
            await _discard_bad_conn(pool, conn)
            conn = None
            if _is_timeout_exc(e):
                timeout_failures += 1
                # Saturated pool: short retry budget, then 503.
                if timeout_failures >= timeout_cap:
                    raise DatabaseUnavailableError(cause=e) from e
            if attempt < max_attempts:
                delay = _backoff_s(attempt)
                logger.warning(
                    "db acquire transient error (attempt %s/%s, sleep %.2fs): %s",
                    attempt,
                    max_attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            raise DatabaseUnavailableError(cause=e) from e
    if conn is None:
        if last_err is not None:
            raise DatabaseUnavailableError(cause=last_err) from last_err
        raise DatabaseUnavailableError()

    try:
        yield conn
    finally:
        try:
            await pool.release(conn)
        except Exception as e:
            logger.debug("db release after acquire_db: %s", e)
