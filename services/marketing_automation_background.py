"""Background loop for opt-in multi-channel marketing automation.

Mirrors the ML engine loop: runs ``run_touchpoint_cycle`` on an interval,
leader-lock safe so only one worker fires it, and fully self-gating on
``MARKETING_AUTOMATION_ENABLED`` so the task is always created but does nothing
until the operator opts in.

Restart safety: before each cycle we read ``marketing_automation_runs`` and
skip when the last completed touchpoint run is still inside the interval window
(uvicorn --reload / deploy bounce must not re-email everyone).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("uploadm8.marketing_automation_background")

_TOUCHPOINT_MODE = "touchpoints_v1"


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


def _startup_stagger_seconds() -> int:
    """Brief delay after process boot so migrations/session are ready — not a send trigger."""
    try:
        return max(15, int(os.environ.get("MARKETING_AUTOMATION_STARTUP_STAGGER_SEC", "90")))
    except (TypeError, ValueError):
        return 90


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def _touchpoint_cooldown_remaining_sec(pool, interval_sec: int) -> int:
    """
    Seconds until another touchpoint cycle may run (0 = allowed now).

    Uses last *completed* run in DB so API restarts do not bypass the interval.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE marketing_automation_runs
                SET status = 'stale',
                    finished_at = COALESCE(finished_at, NOW()),
                    error_detail = COALESCE(error_detail, 'interrupted by process restart')
                WHERE mode = $1 AND status = 'running'
                  AND started_at < NOW() - INTERVAL '45 minutes'
                """,
                _TOUCHPOINT_MODE,
            )
            row = await conn.fetchrow(
                """
                SELECT status, started_at, finished_at
                FROM marketing_automation_runs
                WHERE mode = $1
                ORDER BY COALESCE(finished_at, started_at) DESC
                LIMIT 1
                """,
                _TOUCHPOINT_MODE,
            )
        if not row:
            return 0

        st = str(row["status"] or "").lower()
        if st == "running":
            started = _as_utc(row["started_at"])
            if started is not None:
                age = (datetime.now(timezone.utc) - started).total_seconds()
                remaining_from_start = interval_sec - age
                if remaining_from_start > 0:
                    return max(1, int(min(remaining_from_start, interval_sec)))
                # Run exceeded interval but still marked running — brief pause until stale cleanup.
                return max(1, 300)

        finished = _as_utc(row["finished_at"])
        if finished is None:
            return 0
        elapsed = (datetime.now(timezone.utc) - finished).total_seconds()
        return max(0, int(interval_sec - elapsed))
    except Exception as e:
        logger.warning("marketing automation cooldown check failed: %s", e)
        return 0


async def run_marketing_automation_loop(pool, redis_client=None) -> None:
    """Periodic personalized-touchpoint cycle (email / Discord / in-app)."""
    from services.marketing_touchpoint_runner import run_touchpoint_cycle
    from services.worker_leader_lock import acquire_leader_lock, release_leader_lock

    await asyncio.sleep(_startup_stagger_seconds())
    while True:
        interval = _interval_seconds()
        if not _enabled():
            await asyncio.sleep(min(600, interval))
            continue

        remaining = await _touchpoint_cooldown_remaining_sec(pool, interval)
        if remaining > 0:
            logger.info(
                "marketing automation: deferring %ds — last touchpoint cycle still inside %ds interval "
                "(restart will not re-send)",
                remaining,
                interval,
            )
            await asyncio.sleep(min(remaining, interval))
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
                "marketing automation cycle | messaged=%s email=%s discord=%s in_app=%s dedupe_skip=%s",
                result.get("users_messaged"),
                result.get("email_sent"),
                result.get("discord_sent"),
                result.get("in_app_written"),
                result.get("skipped_dedupe"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("marketing automation loop: %s", e)
        finally:
            await release_leader_lock(redis_client, "marketing_automation", token)

        await asyncio.sleep(interval)
