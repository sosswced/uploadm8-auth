"""Publish target resolution must fail closed when all target_accounts are dead."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from stages.context import JobContext
from stages.errors import PublishError
from stages.publish_stage import resolve_publish_targets


class _MagicPool:
    """Placeholder pool — load is patched."""


def test_resolve_publish_targets_fail_closed_when_all_dead():
    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="user-1",
        platforms=["tiktok", "youtube"],
        target_accounts=["dead-1", "dead-2"],
    )

    async def _run():
        with patch(
            "stages.publish_stage.db_stage.load_platform_token_by_id",
            new=AsyncMock(return_value=None),
        ):
            return await resolve_publish_targets(ctx, db_pool=_MagicPool())

    with pytest.raises(PublishError) as exc:
        asyncio.run(_run())
    assert "disconnected or revoked" in str(exc.value)
    assert exc.value.retryable is False


def test_resolve_publish_targets_keeps_live_partial():
    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="user-1",
        platforms=["tiktok", "youtube"],
        target_accounts=["dead-1", "live-tt"],
    )

    async def _load(_pool, token_id):
        if token_id == "live-tt":
            return {"_platform": "tiktok", "access_token": "x"}
        return None

    async def _run():
        with patch(
            "stages.publish_stage.db_stage.load_platform_token_by_id",
            new=AsyncMock(side_effect=_load),
        ):
            return await resolve_publish_targets(ctx, db_pool=_MagicPool())

    assert asyncio.run(_run()) == [("tiktok", "live-tt")]


def test_resolve_publish_targets_legacy_platform_fallback_without_targets():
    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="user-1",
        platforms=["youtube", "facebook"],
        target_accounts=[],
    )

    async def _run():
        return await resolve_publish_targets(ctx, db_pool=_MagicPool())

    assert asyncio.run(_run()) == [("youtube", None), ("facebook", None)]
