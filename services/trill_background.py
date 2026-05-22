"""Background maintenance for Trill leaderboard (season archive)."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

TRILL_MAINTENANCE_INTERVAL_SEC = int(
    __import__("os").environ.get("TRILL_MAINTENANCE_INTERVAL_SEC", "3600")
)


async def run_trill_maintenance_loop(pool) -> None:
    """Hourly: archive ended seasons into hall of fame."""
    await asyncio.sleep(30)
    while True:
        try:
            from services.trill_engagement import archive_due_seasons

            async with pool.acquire() as conn:
                archived = await archive_due_seasons(conn)
            if archived:
                logger.info("trill maintenance: archived %s season(s)", archived)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("trill maintenance loop: %s", e)
        await asyncio.sleep(max(300, TRILL_MAINTENANCE_INTERVAL_SEC))
