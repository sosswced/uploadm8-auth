"""Background loop for opt-in multi-channel marketing automation.

Mirrors the ML engine loop: runs ``run_touchpoint_cycle`` on an interval,
leader-lock safe so only one worker fires it, and fully self-gating on
``MARKETING_AUTOMATION_ENABLED`` so the task is always created but does nothing
until the operator opts in.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("uploadm8.marketing_automation_background")


def _enabled() -> bool:
    return os.environ.get("MARKETING_AUTOMATION_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _interval_seconds() -> int:
    try:
        return max(900, int(os.environ.get("MARKETING_AUTOMATION_INTERVAL_SEC", str(4 * 3600))))
    except (TypeError, ValueError):
        return 4 * 3600


async def run_marketing_automation_loop(pool, redis_client=None) -> None:
    """Periodic personalized-touchpoint cycle (email / Discord / in-app)."""
    from services.marketing_touchpoint_runner import run_touchpoint_cycle
    from services.worker_leader_lock import acquire_leader_lock, release_leader_lock

    await asyncio.sleep(60)
    while True:
        interval = _interval_seconds()
        if not _enabled():
            await asyncio.sleep(min(600, interval))
            continue

        token = await acquire_leader_lock(
            redis_client,
            "marketing_automation",
            ttl_sec=max(600, interval),
        )
        if token is None:
            await asyncio.sleep(120)
            continue

        try:
            result = await run_touchpoint_cycle(pool)
            logger.info(
                "marketing automation cycle | messaged=%s email=%s discord=%s in_app=%s",
                result.get("users_messaged"),
                result.get("email_sent"),
                result.get("discord_sent"),
                result.get("in_app_written"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("marketing automation loop: %s", e)
        finally:
            await release_leader_lock(redis_client, "marketing_automation", token)

        await asyncio.sleep(interval)
