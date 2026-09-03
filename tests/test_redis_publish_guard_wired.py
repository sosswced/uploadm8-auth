"""Publish fan-out must import the Redis spacing/circuit guard."""

from __future__ import annotations

import inspect

from stages import publish_stage
from stages.redis_publish_guard import platform_bucket, publish_circuit_open


def test_publish_stage_imports_redis_publish_guard():
    src = inspect.getsource(publish_stage.run_publish_stage)
    assert "publish_circuit_open" in src
    assert "publish_wait_slot" in src
    assert "publish_record_result" in src


def test_platform_bucket_groups_meta():
    assert platform_bucket("instagram") == "meta"
    assert platform_bucket("facebook") == "meta"
    assert platform_bucket("tiktok") == "tiktok"


def test_publish_circuit_open_false_without_redis():
    import asyncio

    assert asyncio.run(publish_circuit_open(None, "tiktok")) is False
