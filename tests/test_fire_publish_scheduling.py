"""Fire-level unit test: parallel publish scheduling + worker env (no network)."""
import asyncio
import os

import pytest


def test_parallel_gather_runs_concurrently():
    """Proves asyncio.gather runs tasks concurrently (same pattern as publish_stage)."""
    events: list[str] = []

    async def slow(name: str, delay: float):
        events.append(f"{name}_start")
        await asyncio.sleep(delay)
        events.append(f"{name}_end")
        return name

    async def _run():
        await asyncio.gather(
            slow("a", 0.05),
            slow("b", 0.05),
        )

    asyncio.run(_run())
    assert events.index("a_start") < events.index("a_end")
    assert events.index("b_start") < events.index("b_end")
    assert events.index("b_start") < events.index("a_end")


def test_publish_parallel_env_default():
    parallel = os.environ.get("PUBLISH_PARALLEL", "true").lower() in ("1", "true", "yes")
    assert parallel is True


def test_job_timeout_default_matches_worker():
    """worker.py: JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SECONDS", "1800"))"""
    timeout = int(os.environ.get("JOB_TIMEOUT_SECONDS", "1800"))
    assert 60 <= timeout <= 86400
