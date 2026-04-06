"""
Redis job transport: list queues (legacy) and Redis Streams with consumer groups
(at-least-once). Stream keys are "{list_key}:stream" to avoid clashing with list keys.

Env
---
REDIS_JOB_USE_STREAMS     — default "true". If "false", use LPUSH/BRPOP only.
REDIS_JOB_STREAM_GROUP    — consumer group name (default uploadm8_process)
REDIS_JOB_STREAM_MAXLEN   — approximate max stream length (0 = disabled)
USER_PROCESS_MAX_PARALLEL — max concurrent process-lane jobs per user_id cluster-wide (0 = unlimited)
USER_PROCESS_MAX_PARALLEL_PRIORITY — cap for p0/p1 priority_class (default max(6, base))
USER_SLOT_TTL_SEC         — TTL on per-user slot keys (default 7200)
STREAM_RECLAIM_MIN_IDLE_MS — XAUTOCLAIM min idle (default 120000)
STREAM_RECLAIM_INTERVAL_SEC — reclaim loop sleep (default 25)
STREAM_RECLAIM_COUNT      — messages per stream per reclaim tick (default 8)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uploadm8-worker")

STREAM_FIELD = "payload"

DEFAULT_GROUP = os.environ.get("REDIS_JOB_STREAM_GROUP", "uploadm8_process")


def use_redis_streams() -> bool:
    return os.environ.get("REDIS_JOB_USE_STREAMS", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def stream_key_for_list(list_key: str) -> str:
    return f"{list_key}:stream"


def list_key_from_stream(stream_key: str) -> str:
    if stream_key.endswith(":stream"):
        return stream_key[: -len(":stream")]
    return stream_key


def process_stream_keys_ordered(
    process_priority: str,
    process_normal: str,
    priority_legacy: str,
    upload_legacy: str,
) -> List[str]:
    return [
        stream_key_for_list(process_priority),
        stream_key_for_list(process_normal),
        stream_key_for_list(priority_legacy),
        stream_key_for_list(upload_legacy),
    ]


def legacy_process_list_keys(
    process_priority: str,
    process_normal: str,
    priority_legacy: str,
    upload_legacy: str,
) -> List[str]:
    return [process_priority, process_normal, priority_legacy, upload_legacy]


async def ensure_stream_group(redis: Any, stream_key: str, group: str = DEFAULT_GROUP) -> None:
    """Create consumer group (MKSTREAM). BUSYGROUP is OK."""
    try:
        await redis.xgroup_create(stream_key, group, id="0", mkstream=True)
        logger.info("Created Redis stream group %s on %s", group, stream_key)
    except Exception as e:
        err = str(e).upper()
        if "BUSYGROUP" in err or "BUSY GROUP" in err:
            return
        logger.warning("xgroup_create %s %s: %s", stream_key, group, e)


async def xadd_process_job(redis: Any, list_key: str, job_data: dict) -> Optional[str]:
    """Append job to the stream backing list_key. Returns message id."""
    stream = stream_key_for_list(list_key)
    payload = json.dumps(job_data, default=str)
    maxlen = int(os.environ.get("REDIS_JOB_STREAM_MAXLEN", "0") or 0)
    kw: Dict[str, Any] = {STREAM_FIELD: payload}
    try:
        if maxlen > 0:
            mid = await redis.xadd(stream, kw, maxlen=maxlen, approximate=True)
        else:
            mid = await redis.xadd(stream, kw)
        logger.debug("XADD %s -> %s", stream, mid)
        return mid
    except Exception as e:
        logger.error("xadd failed stream=%s: %s", stream, e)
        return None


async def lpush_process_job(redis: Any, list_key: str, job_data: dict) -> bool:
    try:
        await redis.lpush(list_key, json.dumps(job_data, default=str))
        return True
    except Exception as e:
        logger.error("lpush failed queue=%s: %s", list_key, e)
        return False


async def enqueue_process_job(redis: Any, list_key: str, job_data: dict) -> bool:
    """
    Enqueue one process-lane job. Uses stream or list based on REDIS_JOB_USE_STREAMS.
    """
    if use_redis_streams():
        mid = await xadd_process_job(redis, list_key, job_data)
        return mid is not None
    return await lpush_process_job(redis, list_key, job_data)


def make_worker_consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


async def xreadgroup_one(
    redis: Any,
    stream_keys: List[str],
    group: str,
    consumer: str,
    block_ms: int = 2500,
) -> Optional[Tuple[str, str, str]]:
    """
    Read one new message (>) from the first available stream, priority = stream_keys order.
    Returns (stream_key, message_id, job_json) or None.
    """
    if not stream_keys:
        return None
    streams = {sk: ">" for sk in stream_keys}
    try:
        out = await redis.xreadgroup(group, consumer, streams, count=1, block=block_ms)
    except Exception as e:
        logger.warning("xreadgroup: %s", e)
        return None
    if not out:
        return None
    for sk, messages in out:
        if not messages:
            continue
        mid, fields = messages[0]
        raw = fields.get(STREAM_FIELD) if isinstance(fields, dict) else None
        if raw is None and isinstance(fields, (list, tuple)):
            # decode_responses=True usually gives dict
            for i in range(0, len(fields) - 1, 2):
                if fields[i] == STREAM_FIELD:
                    raw = fields[i + 1]
                    break
        if not raw:
            logger.error("Stream message %s %s missing payload field", sk, mid)
            try:
                await redis.xack(sk, group, mid)
            except Exception:
                pass
            continue
        return sk, mid, raw
    return None


async def xack_message(redis: Any, stream_key: str, group: str, message_id: str) -> None:
    try:
        await redis.xack(stream_key, group, message_id)
    except Exception as e:
        logger.warning("xack failed %s %s: %s", stream_key, message_id, e)


async def xautoclaim_batch(
    redis: Any,
    stream_key: str,
    group: str,
    consumer: str,
    min_idle_ms: int,
    count: int = 10,
) -> List[Tuple[str, str]]:
    """Return [(message_id, payload_json), ...] for stale pending entries."""
    try:
        # redis-py 5+: xautoclaim(stream, groupname, consumername, min_idle_time, start_id='0-0', count=...)
        out = await redis.xautoclaim(
            stream_key,
            group,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
    except Exception as e:
        logger.debug("xautoclaim %s: %s", stream_key, e)
        return []
    if not out or len(out) < 2:
        return []
    messages = out[1] or []
    parsed: List[Tuple[str, str]] = []
    for item in messages:
        if not item:
            continue
        mid, fields = item[0], item[1]
        raw = None
        if isinstance(fields, dict):
            raw = fields.get(STREAM_FIELD)
        elif isinstance(fields, (list, tuple)):
            for i in range(0, len(fields) - 1, 2):
                if fields[i] == STREAM_FIELD:
                    raw = fields[i + 1]
                    break
        if raw:
            parsed.append((mid, raw))
    return parsed


# --- Per-user process concurrency (cluster-wide) ---

_USER_SLOT_PREFIX = "uploadm8:user_process_slots:"


def user_process_cap(priority_class: str) -> int:
    raw = int(os.environ.get("USER_PROCESS_MAX_PARALLEL", "3") or 0)
    if raw <= 0:
        return 0  # unlimited
    pc = (priority_class or "p4").lower()
    if pc in ("p0", "p1"):
        hi = int(os.environ.get("USER_PROCESS_MAX_PARALLEL_PRIORITY", str(max(raw, 6))) or max(raw, 6))
        return max(raw, hi)
    return raw


_USER_ACQUIRE_LUA = """
local k = KEYS[1]
local cap = tonumber(ARGV[1])
if cap == nil or cap <= 0 then return 1 end
local cur = tonumber(redis.call('GET', k) or '0')
if cur >= cap then return 0 end
redis.call('INCR', k)
redis.call('EXPIRE', k, tonumber(ARGV[2]))
return 1
"""

_USER_RELEASE_LUA = """
local k = KEYS[1]
local cur = tonumber(redis.call('GET', k) or '0')
if cur <= 1 then redis.call('DEL', k) else redis.call('DECR', k) end
return 1
"""


async def user_process_try_acquire(redis: Any, user_id: str, priority_class: str) -> bool:
    cap = user_process_cap(priority_class)
    if cap <= 0 or not user_id:
        return True
    ttl = int(os.environ.get("USER_SLOT_TTL_SEC", "7200") or 7200)
    key = _USER_SLOT_PREFIX + str(user_id)
    try:
        n = await redis.eval(_USER_ACQUIRE_LUA, 1, key, str(cap), str(ttl))
        return int(n) == 1
    except Exception as e:
        logger.warning("user_process_try_acquire redis error: %s (allowing job)", e)
        return True


async def user_process_release(redis: Any, user_id: str) -> None:
    if not user_id:
        return
    key = _USER_SLOT_PREFIX + str(user_id)
    try:
        await redis.eval(_USER_RELEASE_LUA, 1, key)
    except Exception as e:
        logger.debug("user_process_release: %s", e)


async def stream_lengths(redis: Any, stream_keys: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for sk in stream_keys:
        try:
            out[sk] = int(await redis.xlen(sk))
        except Exception:
            out[sk] = -1
    return out


async def list_lengths(redis: Any, list_keys: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lk in list_keys:
        try:
            out[lk] = int(await redis.llen(lk))
        except Exception:
            out[lk] = -1
    return out
