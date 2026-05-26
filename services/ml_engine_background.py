"""Background loop for the UploadM8 ML / AI engine (leader-lock safe)."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("uploadm8.ml_engine_background")


async def run_ml_engine_loop(pool, redis_client=None) -> None:
    """Periodic ML engine cycle — dataset, train, eval, Trackio sync."""
    from services.ml_engine import run_ml_engine_cycle
    from services.ml_engine_config import get_ml_engine_config
    from services.worker_leader_lock import acquire_leader_lock, release_leader_lock

    await asyncio.sleep(45)
    while True:
        cfg = get_ml_engine_config()
        sleep_sec = max(3600, int(cfg.interval_seconds))
        if not cfg.enabled:
            await asyncio.sleep(min(600, sleep_sec))
            continue

        token = await acquire_leader_lock(
            redis_client,
            "ml_engine",
            ttl_sec=max(600, sleep_sec),
        )
        if token is None:
            await asyncio.sleep(120)
            continue

        try:
            result = await run_ml_engine_cycle(pool)
            if result.get("ok"):
                logger.info(
                    "ml engine cycle ok | mode=%s run_id=%s",
                    result.get("mode", "local"),
                    result.get("m8_model_run_id"),
                )
            elif not result.get("skipped"):
                logger.warning(
                    "ml engine cycle failed: %s",
                    result.get("error") or result.get("reason"),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("ml engine loop: %s", e)
        finally:
            await release_leader_lock(redis_client, "ml_engine", token)

        await asyncio.sleep(sleep_sec)
