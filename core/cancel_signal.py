"""
Upload cancel signaling
========================

Two-tier cancellation channel for in-flight upload jobs:

1. **Redis flag** (``cancel:{upload_id}``) — fast, in-memory, checked by the
   worker on every iteration of long-running stages (transcode loop, publish
   loop, anywhere we want sub-second cancellation latency). Auto-expires after
   ``CANCEL_FLAG_TTL`` so abandoned cancels don't pollute Redis forever.

2. **DB column** (``uploads.cancel_requested``) — durable source of truth.
   Survives Redis flushes / restarts. Set by the cancel API endpoint
   alongside the Redis flag and consumed by the worker at every stage
   boundary inside ``maybe_cancel``.

The Redis flag is a *cache* — DB is authoritative. If Redis is unavailable,
cancel still works (worker polls DB at stage boundaries); we just lose
sub-stage responsiveness.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("uploadm8-worker")

# How long the Redis cancel flag lives. One hour is plenty: any in-flight job
# that hasn't picked up the cancel within an hour is either dead or will hit
# the DB-backed check at its next stage boundary.
CANCEL_FLAG_TTL = 3600


def _key(upload_id: str) -> str:
    return f"cancel:{upload_id}"


async def signal_cancel(redis_client: Optional[Any], upload_id: str) -> None:
    """Set the Redis cancel flag for ``upload_id`` (best-effort).

    Called by the cancel API endpoint. Failing soft is fine — the DB column
    is the durable signal.
    """
    if not redis_client or not upload_id:
        return
    try:
        await redis_client.set(_key(upload_id), "1", ex=CANCEL_FLAG_TTL)
    except Exception as e:
        logger.debug(f"cancel signal Redis set failed for {upload_id}: {e}")


async def clear_cancel_signal(redis_client: Optional[Any], upload_id: str) -> None:
    """Drop the Redis cancel flag (after retry, or after we've finished
    handling the cancel in the worker)."""
    if not redis_client or not upload_id:
        return
    try:
        await redis_client.delete(_key(upload_id))
    except Exception as e:
        logger.debug(f"cancel signal Redis delete failed for {upload_id}: {e}")


async def is_cancelled_fast(redis_client: Optional[Any], upload_id: str) -> bool:
    """Cheap Redis-only check. Returns True only when the Redis flag is set.

    Use this inside hot loops (transcode group iteration, publish per-platform)
    where sub-second responsiveness matters and a missed cancel is fine because
    the next ``maybe_cancel`` stage boundary will pick it up from the DB.
    """
    if not redis_client or not upload_id:
        return False
    try:
        return bool(await redis_client.exists(_key(upload_id)))
    except Exception as e:
        logger.debug(f"cancel signal Redis check failed for {upload_id}: {e}")
        return False


async def is_cancelled(
    redis_client: Optional[Any],
    db_pool: Optional[Any],
    upload_id: str,
) -> bool:
    """Authoritative check: Redis first (fast), DB second (durable fallback).

    Used by the worker's ``maybe_cancel`` stage-boundary checkpoint so we
    pick up cancels even when Redis was unavailable when the API set the flag.
    """
    if await is_cancelled_fast(redis_client, upload_id):
        return True
    if db_pool is None or not upload_id:
        return False
    try:
        async with db_pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT cancel_requested FROM uploads WHERE id = $1", upload_id
            )
            return bool(val)
    except Exception as e:
        logger.debug(f"cancel signal DB check failed for {upload_id}: {e}")
        return False
