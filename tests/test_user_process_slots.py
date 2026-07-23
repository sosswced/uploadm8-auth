"""Unit tests for per-user process slot heal / leak recovery."""

from __future__ import annotations

import asyncio

from stages.redis_job_queue import (
    clear_all_user_process_slots,
    heal_user_process_slots,
    user_process_wait_acquire,
)


class _FakeRedis:
    def __init__(self, value: str | None):
        self.value = value
        self.deleted = False
        self.set_args = None
        self.keys = {}

    async def get(self, key):
        if key in self.keys:
            return self.keys[key]
        return self.value

    async def delete(self, *keys):
        n = 0
        for key in keys:
            if key in self.keys:
                del self.keys[key]
                n += 1
            elif self.value is not None and not self.keys:
                self.deleted = True
                self.value = None
                n += 1
        return n

    async def set(self, key, val, ex=None):
        self.set_args = (key, val, ex)
        self.value = val
        self.keys[key] = val
        return True

    async def scan(self, cursor=0, match=None, count=100):
        matched = [k for k in list(self.keys) if not match or k.startswith(match.rstrip("*"))]
        return 0, matched

    async def eval(self, script, numkeys, *args):
        # Acquire LUA: return 0 when at cap (simulate busy slot)
        key = args[0]
        cap = int(args[1])
        cur = int(self.keys.get(key) or self.value or 0)
        if cur >= cap:
            return 0
        self.keys[key] = str(cur + 1)
        return 1


def test_heal_clamps_redis_above_db_processing():
    r = _FakeRedis("4")

    async def _run():
        return await heal_user_process_slots(r, "user-1", db_processing=1)

    assert asyncio.run(_run()) == 1
    assert r.set_args is not None
    assert r.set_args[1] == "1"


def test_heal_clears_when_no_db_processing():
    r = _FakeRedis("3")

    async def _run():
        return await heal_user_process_slots(r, "user-1", db_processing=0)

    assert asyncio.run(_run()) == 0
    assert r.deleted is True


def test_heal_noop_when_redis_not_higher():
    r = _FakeRedis("1")

    async def _run():
        return await heal_user_process_slots(r, "user-1", db_processing=2)

    assert asyncio.run(_run()) is None
    assert r.set_args is None
    assert r.deleted is False


def test_clear_all_user_process_slots():
    r = _FakeRedis(None)
    r.keys = {
        "uploadm8:user_process_slots:aaa": "1",
        "uploadm8:user_process_slots:bbb": "2",
        "other:key": "9",
    }

    async def _run():
        return await clear_all_user_process_slots(r)

    assert asyncio.run(_run()) == 2
    assert "uploadm8:user_process_slots:aaa" not in r.keys
    assert "other:key" in r.keys


def test_wait_acquire_aborts_when_already_processing(monkeypatch):
    monkeypatch.setenv("USER_PROCESS_MAX_PARALLEL", "1")
    monkeypatch.setenv("USER_SLOT_MAX_WAIT_SEC", "30")
    monkeypatch.setenv("USER_SLOT_WAIT_INITIAL_SEC", "0.01")
    r = _FakeRedis("1")
    r.keys["uploadm8:user_process_slots:user-1"] = "1"
    calls = {"n": 0}

    async def _already():
        calls["n"] += 1
        return True

    async def _run():
        return await user_process_wait_acquire(
            r,
            "user-1",
            "p4",
            upload_id="up-1",
            already_processing_check=_already,
            max_wait_sec=5,
        )

    assert asyncio.run(_run()) is False
    assert calls["n"] >= 1
