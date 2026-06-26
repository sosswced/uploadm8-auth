"""Deferred publish must probe post_publish checkpoint before publishing."""

import asyncio
from unittest.mock import AsyncMock, patch

from stages.context import JobContext


def _minimal_ctx() -> JobContext:
    ctx = JobContext(
        upload_id="u-deferred-cp",
        user_id="user-1",
        job_id="job-1",
        platforms=["youtube"],
    )
    ctx.state = "processing"
    return ctx


def test_deferred_publish_calls_checkpoint_resume():
    async def _run():
        upload_record = {
            "id": "u-deferred-cp",
            "platforms": ["youtube"],
            "schedule_mode": "smart",
            "schedule_metadata": {"youtube": "2030-01-01T12:00:00+00:00"},
            "platform_results": [],
            "processed_assets": {"default": "r2/key.mp4", "youtube": "r2/key.mp4"},
        }
        user_record = {"id": "user-1", "subscription_tier": "creator_pro"}
        resume = AsyncMock(return_value=("post_publish", {}))

        with patch("worker.db_stage.load_upload_record", new=AsyncMock(return_value=upload_record)), patch(
            "worker.db_stage.load_user", new=AsyncMock(return_value=user_record)
        ), patch("worker.db_stage.load_user_settings", new=AsyncMock(return_value={})), patch(
            "worker.db_stage.merge_pikzels_thumbnail_persona_id", new=AsyncMock()
        ), patch(
            "worker.db_stage.load_user_entitlement_overrides", new=AsyncMock(return_value={})
        ), patch(
            "worker.get_entitlements_from_user", return_value=type("E", (), {"max_thumbnails": 3, "max_caption_frames": 8, "can_ai": True})()
        ), patch(
            "worker.create_context", return_value=_minimal_ctx()
        ), patch(
            "worker.pipeline_checkpoint.try_resume_from_checkpoint", new=resume
        ), patch(
            "stages.publish_stage.resolve_publish_targets", new=AsyncMock(return_value=["youtube"])
        ), patch(
            "services.deferred_publish_schedule.platforms_due_for_publish", return_value=["youtube"]
        ), patch(
            "worker.db_stage.mark_processing_started", new=AsyncMock()
        ), patch(
            "worker.run_publish_and_notify", new=AsyncMock(return_value=True)
        ), patch(
            "worker.r2_stage.download_file", new=AsyncMock()
        ), patch(
            "worker.db_pool", new=object()
        ):
            from worker import run_deferred_publish

            ok = await run_deferred_publish("u-deferred-cp", "user-1")
            assert ok is True
            resume.assert_awaited_once()

    asyncio.run(_run())
