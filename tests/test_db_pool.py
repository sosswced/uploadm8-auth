"""Tests for database pool acquire retry."""

import asyncio

import asyncpg
import pytest

from core.db_pool import (
    DatabaseUnavailableError,
    acquire_db,
    is_dead_connection_error,
    is_transient_db_error,
)


class _FakeConn:
    async def fetchval(self, *_a, **_k):
        return 1

    def terminate(self):
        pass


class _FakePool:
    def __init__(self, errors_before_success: int):
        self._remaining = errors_before_success
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire(self, timeout=None):
        self.acquire_calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise asyncpg.ConnectionDoesNotExistError("connection was closed")
        return _FakeConn()

    async def release(self, _conn):
        self.release_calls += 1


class _FakePoolPingFailsThenOk:
    """Acquire succeeds but SELECT 1 fails once (dead Neon socket)."""

    def __init__(self):
        self.acquire_calls = 0
        self.release_calls = 0
        self._ping_fails_left = 1

    async def acquire(self, timeout=None):
        self.acquire_calls += 1

        class _Conn:
            def terminate(self_inner):
                pass

            async def fetchval(self_inner, *_a, **_k):
                if self._ping_fails_left > 0:
                    self._ping_fails_left -= 1
                    raise asyncpg.CannotConnectNowError(
                        "the database system is not yet accepting connections"
                    )
                return 1

        return _Conn()

    async def release(self, _conn):
        self.release_calls += 1


class _FakePoolAlwaysTimeout:
    def __init__(self):
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire(self, timeout=None):
        self.acquire_calls += 1
        raise TimeoutError()

    async def release(self, _conn):
        self.release_calls += 1


def test_is_transient_db_error():
    assert is_transient_db_error(asyncpg.ConnectionDoesNotExistError("gone"))
    assert is_transient_db_error(asyncpg.InterfaceError("connection has been released"))


def test_is_dead_connection_error_excludes_schema_errors():
    assert is_dead_connection_error(asyncpg.ConnectionDoesNotExistError("gone"))
    assert is_dead_connection_error(asyncpg.InterfaceError("connection has been released"))
    uc = getattr(asyncpg, "UndefinedColumnError", None)
    if uc is not None:
        assert not is_dead_connection_error(uc("column revoked_at does not exist"))
    assert is_transient_db_error(
        asyncpg.CannotConnectNowError("the database system is not yet accepting connections")
    )
    assert is_transient_db_error(TimeoutError("acquire timed out"))
    assert is_transient_db_error(DatabaseUnavailableError(cause=TimeoutError()))
    assert not is_transient_db_error(ValueError("nope"))


def test_acquire_db_retries_transient_error():
    async def _run():
        pool = _FakePool(errors_before_success=2)
        async with acquire_db(pool, attempts=3) as conn:
            assert isinstance(conn, _FakeConn)
        assert pool.acquire_calls == 3
        assert pool.release_calls >= 1

    asyncio.run(_run())


def test_acquire_db_retries_failed_ping():
    async def _run():
        pool = _FakePoolPingFailsThenOk()
        async with acquire_db(pool, attempts=4) as conn:
            assert conn is not None
        assert pool.acquire_calls == 2
        assert pool.release_calls >= 2  # failed ping discard + success release

    asyncio.run(_run())


def test_acquire_db_timeout_raises_unavailable():
    async def _run():
        pool = _FakePoolAlwaysTimeout()
        with pytest.raises(DatabaseUnavailableError):
            async with acquire_db(pool, attempts=5):
                pass
        # Fail-fast: timeout attempts cap (default 2), not full attempts budget
        assert pool.acquire_calls == 2

    asyncio.run(_run())


def test_acquire_db_exhausted_connection_errors_raise_unavailable():
    async def _run():
        pool = _FakePool(errors_before_success=99)
        with pytest.raises(DatabaseUnavailableError):
            async with acquire_db(pool, attempts=3):
                pass
        assert pool.acquire_calls == 3

    asyncio.run(_run())
