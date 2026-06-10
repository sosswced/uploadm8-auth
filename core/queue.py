"""
UploadM8 job queue — push jobs to Redis priority/normal lanes.
Extracted from app.py; uses core.state for Redis client.
"""

import asyncio
import json
import uuid
import logging

from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

import core.state
from core.config import (
    PROCESS_PRIORITY_QUEUE,
    PROCESS_NORMAL_QUEUE,
    PUBLISH_PRIORITY_QUEUE,
    PUBLISH_NORMAL_QUEUE,
)
from core.helpers import _now_utc
from core.upload_baseline_defaults import serialize_job_payload
from stages.entitlements import PRIORITY_QUEUE_CLASSES

logger = logging.getLogger("uploadm8-api")

# Errors that indicate a stale TCP socket (Windows WinError 10054, broken
# pipe, idle-reaped server side, etc). Worth one explicit retry — the redis-py
# client will reconnect on the next call.
_TRANSIENT_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, OSError)


async def enqueue_job(
    job_data: dict,
    lane: str = "process",
    priority_class: str = "p4",
) -> bool:
    """
    Push a job to the correct Redis lane based on job type and tier priority.

    Args:
        job_data:       Job payload dict. upload_id required.
        lane:           "process" (FFmpeg-heavy) or "publish" (API-light).
        priority_class: Tier priority class p0-p4. p0/p1/p2 -> priority queue.
                        p3/p4 -> normal queue.

    Queue routing:
        process + priority_class in {p0,p1,p2} -> PROCESS_PRIORITY_QUEUE
        process + priority_class in {p3,p4}    -> PROCESS_NORMAL_QUEUE
        publish + priority_class in {p0,p1,p2} -> PUBLISH_PRIORITY_QUEUE
        publish + priority_class in {p3,p4}    -> PUBLISH_NORMAL_QUEUE
    """
    if not core.state.redis_client:
        logger.warning("enqueue_job called but redis_client is None")
        return False

    is_priority = priority_class in PRIORITY_QUEUE_CLASSES

    if lane == "publish":
        queue = PUBLISH_PRIORITY_QUEUE if is_priority else PUBLISH_NORMAL_QUEUE
    else:
        queue = PROCESS_PRIORITY_QUEUE if is_priority else PROCESS_NORMAL_QUEUE

    job_data["enqueued_at"]    = _now_utc().isoformat()
    job_data["job_id"]         = str(uuid.uuid4())
    job_data["lane"]           = lane
    job_data["priority_class"] = priority_class

    # Coerce asyncpg UUID/datetime values before json.dumps (default=str alone
    # is not used on all worker enqueue paths).
    payload = serialize_job_payload(job_data)
    upload_id = job_data.get("upload_id", "?")

    # One extra retry on transient socket errors. The Retry policy on the
    # client itself handles in-flight reconnects, but if the connection was
    # already closed by the peer between requests (Windows WinError 10054,
    # idle Redis client kill) the FIRST write can still fail before the
    # client-level retry kicks in. A second attempt forces a fresh connection.
    for attempt in (1, 2):
        try:
            from stages.redis_job_queue import enqueue_process_job, use_redis_streams

            ok = await enqueue_process_job(core.state.redis_client, queue, job_data)
            if not ok:
                return False
            if not use_redis_streams():
                pass  # enqueue_process_job already LPUSH when streams off
            logger.debug(
                f"[{upload_id}] Enqueued -> {queue} "
                f"(lane={lane} priority_class={priority_class} streams={use_redis_streams()})"
            )
            return True
        except _TRANSIENT_REDIS_ERRORS as e:
            if attempt == 1:
                logger.warning(
                    f"[{upload_id}] enqueue_job transient redis error "
                    f"(attempt 1/2, will retry): {e!r}"
                )
                await asyncio.sleep(0.1)
                continue
            logger.error(f"[{upload_id}] enqueue_job failed after retry: {e!r}")
            return False
        except Exception as e:
            logger.error(f"[{upload_id}] enqueue_job failed: {e!r}")
            return False
    return False
