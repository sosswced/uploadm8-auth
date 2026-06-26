"""Tests for database pool acquire retry."""

import asyncio

import asyncpg
import pytest

from core.db_pool import acquire_db, is_transient_db_error


class _FakeConn:
    pass


class _FakePool:
    def __init__(self, errors_before_success: int):
        self._remaining = errors_before_success
        self.acquire_calls = 0

    def acquire(self):
        pool = self

        class _CM:
            async def __aenter__(self_inner):
                pool.acquire_calls += 1
                if pool._remaining > 0:
                    pool._remaining -= 1
                    import asyncpg

                    raise asyncpg.ConnectionDoesNotExistError("connection was closed")
                return _FakeConn()

            async def __aexit__(self_inner, *args):
                return False

        return _CM()


def test_is_transient_db_error():
    assert is_transient_db_error(asyncpg.ConnectionDoesNotExistError("gone"))
    assert not is_transient_db_error(ValueError("nope"))


def test_acquire_db_retries_transient_error():
    async def _run():
        pool = _FakePool(errors_before_success=2)
        async with acquire_db(pool, attempts=3) as conn:
            assert isinstance(conn, _FakeConn)
        assert pool.acquire_calls == 3

    asyncio.run(_run())
