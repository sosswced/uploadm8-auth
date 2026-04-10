"""
UploadM8 job queue — push jobs to Redis priority/normal lanes.
Extracted from app.py; uses core.state for Redis client.
"""

import json
import uuid
import logging

import core.state
from core.config import (
    PROCESS_PRIORITY_QUEUE,
    PROCESS_NORMAL_QUEUE,
    PUBLISH_PRIORITY_QUEUE,
    PUBLISH_NORMAL_QUEUE,
)
from core.helpers import _now_utc
from stages.entitlements import PRIORITY_QUEUE_CLASSES

logger = logging.getLogger("uploadm8-api")


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

    try:
        await core.state.redis_client.lpush(queue, json.dumps(job_data))
        logger.debug(
            f"[{job_data.get('upload_id', '?')}] Enqueued -> {queue} "
            f"(lane={lane} priority_class={priority_class})"
        )
        return True
    except Exception as e:
        logger.error(f"enqueue_job failed: {e}")
        return False
