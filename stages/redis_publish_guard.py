"""
Distributed publish throttling + circuit breaker (Redis), multi-worker safe.

Complements in-process outbound_rl.py. When ctx.redis_client is set by the worker,
_publish_one_target waits for a slot and records failures for circuit trips.

Env
---
PUBLISH_REDIS_RL_ENABLED       — default true
PUBLISH_PLATFORM_MIN_INTERVAL_MS — minimum gap between publish *starts* per bucket (default 350)
PUBLISH_CB_FAILURE_THRESHOLD   — consecutive weighted failures to open circuit (default 8)
PUBLISH_CB_OPEN_SEC           — seconds to short-circuit after trip (default 45)
PUBLISH_CB_WINDOW_SEC         — failure counter TTL (default 120)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("uploadm8-worker")

_RL_ENABLED = os.environ.get("PUBLISH_REDIS_RL_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MIN_INTERVAL_MS = max(0, int(os.environ.get("PUBLISH_PLATFORM_MIN_INTERVAL_MS", "350") or 0))
_CB_FAILS = max(1, int(os.environ.get("PUBLISH_CB_FAILURE_THRESHOLD", "8") or 8))
_CB_OPEN = max(5, int(os.environ.get("PUBLISH_CB_OPEN_SEC", "45") or 45))
_CB_WIN = max(30, int(os.environ.get("PUBLISH_CB_WINDOW_SEC", "120") or 120))

_LAST_KEY = "uploadm8:pub_rl:last:{bucket}"
_FAIL_KEY = "uploadm8:pub_cb:fails:{bucket}"
_OPEN_KEY = "uploadm8:pub_cb:open:{bucket}"

_RL_LUA = """
local lk = KEYS[1]
local min_ms = tonumber(ARGV[1])
local now_ms = tonumber(ARGV[2])
if min_ms <= 0 then return 1 end
local last = tonumber(redis.call('GET', lk) or '0')
local wait = last + min_ms - now_ms
if wait > 0 then return -wait end
redis.call('SET', lk, now_ms, 'PX', tonumber(ARGV[3]))
return 1
"""


def platform_bucket(platform: str) -> str:
    p = (platform or "unknown").strip().lower()
    if p in ("instagram", "facebook"):
        return "meta"
    if p in ("youtube", "tiktok", "twitter", "x"):
        return p
    return p.replace(" ", "_")[:32] or "unknown"


async def _redis(redis_client: Any):
    if redis_client is None:
        return None
    return redis_client


async def publish_circuit_open(redis_client: Any, platform: str) -> bool:
    if not _RL_ENABLED:
        return False
    r = await _redis(redis_client)
    if not r:
        return False
    b = platform_bucket(platform)
    try:
        return bool(await r.get(_OPEN_KEY.format(bucket=b)))
    except Exception:
        return False


async def publish_wait_slot(redis_client: Any, platform: str) -> None:
    """Block until min-interval spacing for this platform bucket (cluster-wide)."""
    if not _RL_ENABLED or _MIN_INTERVAL_MS <= 0:
        return
    r = await _redis(redis_client)
    if not r:
        return
    b = platform_bucket(platform)
    key = _LAST_KEY.format(bucket=b)
    now_ms = int(time.time() * 1000)
    ttl_ms = max(_MIN_INTERVAL_MS * 4, 60_000)
    try:
        rc = await r.eval(_RL_LUA, 1, key, str(_MIN_INTERVAL_MS), str(now_ms), str(ttl_ms))
        if isinstance(rc, bytes):
            rc = int(rc.decode())
        else:
            rc = int(rc)
        if rc < 0:
            await asyncio.sleep(min(5.0, (-rc) / 1000.0))
            # one retry
            now_ms = int(time.time() * 1000)
            rc2 = await r.eval(_RL_LUA, 1, key, str(_MIN_INTERVAL_MS), str(now_ms), str(ttl_ms))
            if isinstance(rc2, bytes):
                rc2 = int(rc2.decode())
            else:
                rc2 = int(rc2)
            if rc2 < 0:
                await asyncio.sleep(min(5.0, (-rc2) / 1000.0))
    except Exception as e:
        logger.debug("publish_wait_slot %s: %s", b, e)


async def publish_record_result(
    redis_client: Any,
    platform: str,
    success: bool,
    http_status: Optional[int] = None,
) -> None:
    if not _RL_ENABLED:
        return
    r = await _redis(redis_client)
    if not r:
        return
    b = platform_bucket(platform)
    fk = _FAIL_KEY.format(bucket=b)
    ok = _OPEN_KEY.format(bucket=b)
    try:
        if success:
            await r.delete(fk)
            return
        weight = 2 if http_status and int(http_status) >= 500 else 1
        if http_status and int(http_status) == 429:
            weight = 3
        n = await r.incrby(fk, weight)
        if n == 1:
            await r.expire(fk, _CB_WIN)
        if n >= _CB_FAILS:
            await r.set(ok, "1", ex=_CB_OPEN)
            await r.delete(fk)
            logger.warning(
                "Publish circuit OPEN for %ss (bucket=%s) after weighted_failures=%s",
                _CB_OPEN,
                b,
                n,
            )
    except Exception as e:
        logger.debug("publish_record_result: %s", e)
