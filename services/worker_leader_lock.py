"""
Redis leader locks for multi-instance workers (SET NX + compare-and-del release).

Used by worker.py periodic loops and verify_stage so only one replica runs a tick.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("uploadm8-worker")

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""


def leader_lock_enabled() -> bool:
    return os.environ.get("WORKER_LEADER_LOCK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def leader_lock_prefix() -> str:
    return os.environ.get("WORKER_LEADER_LOCK_PREFIX", "uploadm8:leader")


def _worker_leader_token() -> str:
    wid = os.environ.get("RENDER_INSTANCE_ID") or os.environ.get("HOSTNAME") or "local"
    return f"{wid}:{os.urandom(4).hex()}"


async def acquire_leader_lock(
    redis_client: Optional[Any],
    lock_name: str,
    ttl_sec: int,
) -> Optional[str]:
    """
    Returns a token if this instance should run the critical section.
    Returns None if a peer holds the lock (skip this cycle).
    Returns '' if locks are disabled or Redis is unavailable (fail-open).
    """
    if not leader_lock_enabled():
        return ""
    if redis_client is None:
        return ""
    token = _worker_leader_token()
    key = f"{leader_lock_prefix()}:{lock_name}"
    try:
        got = await redis_client.set(key, token, nx=True, ex=max(5, int(ttl_sec)))
        return token if got else None
    except Exception as e:
        logger.warning("leader lock %s acquire error: %s — proceeding without lock", lock_name, e)
        return ""


async def release_leader_lock(
    redis_client: Optional[Any],
    lock_name: str,
    token: Optional[str],
) -> None:
    if not token:
        return
    if redis_client is None:
        return
    key = f"{leader_lock_prefix()}:{lock_name}"
    try:
        await redis_client.eval(_RELEASE_LUA, 1, key, token)
    except Exception as e:
        logger.debug("leader lock %s release: %s", lock_name, e)
