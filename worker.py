"""
UploadM8 Worker Service - Complete Pipeline

Pipeline order (critical):
  1. Download        - Fetch original video + telemetry from R2
  2. Telemetry       - Parse .map file, calculate Trill score
  3. Thumbnail       - Extract frame from original video
  4. Caption         - AI-generate title/caption/hashtags
  5. HUD             - Burn speed overlay onto video (if entitled)
  6. Watermark       - Burn text watermark onto video (if entitled)
  7. Transcode       - Create per-platform MP4s FROM the HUD+watermarked video
  8. Upload          - Upload EACH platform MP4 to its own R2 key
  9. Publish         - Send correct file to each platform API
  10. Notify         - Discord webhooks

WHY this order matters:
  - HUD + Watermark modify the video BEFORE platform-specific transcoding
  - Transcode then creates 4 separate files (tiktok.mp4, youtube.mp4, etc.)
    each with correct resolution, duration trim, bitrate, and frame rate
  - Each platform gets its OWN R2 key and its OWN API call
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import asyncpg
import redis.asyncio as redis

from stages.errors import StageError, SkipStage, CancelRequested
from stages.context import JobContext, create_context
from stages.entitlements import get_entitlements_from_user
from stages import db as db_stage
from stages import r2 as r2_stage
from stages.telemetry_stage import run_telemetry_stage
from stages.transcode_stage import run_transcode_stage
from stages.thumbnail_stage import run_thumbnail_stage
from stages.caption_stage import run_caption_stage
from stages.hud_stage import run_hud_stage
from stages.watermark_stage import run_watermark_stage
from stages.publish_stage import run_publish_stage
from stages.verify_stage import run_verification_loop
from stages.notify_stage import run_notify_stage, notify_admin_worker_start, notify_admin_worker_stop, notify_admin_error

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [worker] %(message)s")
logger = logging.getLogger("uploadm8-worker")

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")
PRIORITY_JOB_QUEUE = os.environ.get("PRIORITY_JOB_QUEUE", "uploadm8:priority")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))

# Redis resilience
REDIS_RETRY_DELAY = 5.0
REDIS_MAX_RETRIES = 10

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
shutdown_requested = False
shutdown_event = None


def handle_shutdown(signum, frame):
    global shutdown_requested, shutdown_event
    logger.info(f"Shutdown signal received ({signum})")
    shutdown_requested = True
    try:
        if shutdown_event is not None:
            shutdown_event.set()
    except Exception:
        pass


async def check_cancelled(ctx: JobContext) -> bool:
    if bool(getattr(ctx, "cancel_requested", False)):
        return True
    cancelled = await db_stage.check_cancel_requested(db_pool, ctx.upload_id)
    if cancelled:
        setattr(ctx, "cancel_requested", True)
    return cancelled


async def maybe_cancel(ctx: JobContext, stage: str):
    if await check_cancelled(ctx):
        logger.info(f"Cancel at {stage} for {ctx.upload_id}")
        await db_stage.mark_cancelled(db_pool, ctx.upload_id)
        raise CancelRequested(ctx.upload_id)


async def run_pipeline(job_data: dict) -> bool:
    upload_id = job_data.get("upload_id")
    user_id = job_data.get("user_id")
    job_id = job_data.get("job_id", "unknown")
    action = job_data.get("action") or "process"

    logger.info(f"Starting pipeline: upload={upload_id}, job={job_id}, action={action}")
    ctx = None
    temp_dir = None

    try:
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            logger.error(f"Records not found: upload={upload_id}, user={user_id}")
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.started_at = datetime.now(timezone.utc)
        ctx.state = "processing"
        await db_stage.mark_processing_started(db_pool, ctx)
        await maybe_cancel(ctx, "init")

        # ================================================================
        # STAGE 1: Download original video + telemetry from R2
        # ================================================================
        ctx.mark_stage("download")
        temp_dir = tempfile.TemporaryDirectory()
        ctx.temp_dir = Path(temp_dir.name)
        video_local = ctx.temp_dir / ctx.filename
        await r2_stage.download_file(ctx.source_r2_key, video_local)
        ctx.local_video_path = video_local

        if ctx.telemetry_r2_key:
            try:
                telem_local = ctx.temp_dir / "telemetry.map"
                await r2_stage.download_file(ctx.telemetry_r2_key, telem_local)
                ctx.local_telemetry_path = telem_local
            except Exception as e:
                logger.warning(f"Telemetry download failed: {e}")

        await maybe_cancel(ctx, "download")

        # ================================================================
        # STAGE 2: Telemetry - Parse .map, calculate Trill score
        # (Must run before HUD so speed data is available for overlay)
        # ================================================================
        try:
            ctx = await run_telemetry_stage(ctx)
        except SkipStage as e:
            logger.info(f"Telemetry skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Telemetry error: {e.message}")
        await maybe_cancel(ctx, "telemetry")

        # ================================================================
        # STAGE 3: Thumbnail - Extract frame from original video
        # (Run on original before HUD/watermark burns overlays)
        # ================================================================
        try:
            ctx = await run_thumbnail_stage(ctx)
        except SkipStage as e:
            logger.info(f"Thumbnail skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Thumbnail error: {e.message}")
        await maybe_cancel(ctx, "thumbnail")

        # ================================================================
        # STAGE 4: Caption - AI-generate title/caption/hashtags
        # ================================================================
        try:
            ctx = await run_caption_stage(ctx)
            await db_stage.save_generated_metadata(db_pool, ctx)
        except SkipStage as e:
            logger.info(f"Caption skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Caption error: {e.message}")
        await maybe_cancel(ctx, "caption")

        # ================================================================
        # STAGE 5: HUD - Burn speed overlay onto video
        # Sets ctx.processed_video_path if applied
        # ================================================================
        try:
            ctx = await run_hud_stage(ctx)
        except SkipStage as e:
            logger.info(f"HUD skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"HUD error: {e.message}")
        await maybe_cancel(ctx, "hud")

        # ================================================================
        # STAGE 6: Watermark - Burn text watermark onto video
        # Uses ctx.processed_video_path (from HUD) or ctx.local_video_path
        # Updates ctx.processed_video_path with watermarked output
        # ================================================================
        try:
            ctx = await run_watermark_stage(ctx)
        except SkipStage as e:
            logger.info(f"Watermark skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Watermark error: {e.message}")
        await maybe_cancel(ctx, "watermark")

        # ================================================================
        # STAGE 7: Transcode - Create per-platform MP4s
        # Input: ctx.processed_video_path (HUD+watermarked) or ctx.local_video_path
        # Output: ctx.platform_videos = {tiktok: Path, youtube: Path, ...}
        # Each file has platform-correct resolution, duration, fps, bitrate
        # ================================================================
        try:
            ctx = await run_transcode_stage(ctx)
        except SkipStage as e:
            logger.info(f"Transcode skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Transcode error: {e.message}")
        await maybe_cancel(ctx, "transcode")

        # ================================================================
        # STAGE 8: Upload - Upload EACH platform video to its own R2 key
        # Old: one processed/{user_id}/{upload_id}.mp4
        # New: processed/{user_id}/{upload_id}/tiktok.mp4
        #      processed/{user_id}/{upload_id}/youtube.mp4
        #      processed/{user_id}/{upload_id}/instagram.mp4
        #      processed/{user_id}/{upload_id}/facebook.mp4
        # ================================================================
        ctx.mark_stage("upload")
        processed_assets = {}

        if ctx.platform_videos:
            for platform, video_path in ctx.platform_videos.items():
                if video_path and video_path.exists():
                    r2_key = f"processed/{ctx.user_id}/{ctx.upload_id}/{platform}.mp4"
                    try:
                        await r2_stage.upload_file(video_path, r2_key, "video/mp4")
                        processed_assets[platform] = r2_key
                        logger.info(f"Uploaded {platform} asset: {r2_key} ({video_path.stat().st_size / 1024 / 1024:.1f}MB)")
                    except Exception as e:
                        logger.error(f"R2 upload failed for {platform}: {e}")
                else:
                    logger.warning(f"No video file for {platform}, skipping R2 upload")

        # Also upload a "default" processed video for backward compatibility
        # (admin preview, re-processing, etc.)
        fallback_video = ctx.processed_video_path or ctx.local_video_path
        if fallback_video and fallback_video.exists():
            default_key = f"processed/{ctx.user_id}/{ctx.upload_id}/default.mp4"
            try:
                await r2_stage.upload_file(fallback_video, default_key, "video/mp4")
                processed_assets["default"] = default_key
                ctx.processed_r2_key = default_key
            except Exception as e:
                logger.warning(f"Default R2 upload failed: {e}")

        # Store per-platform R2 keys in context (for publish stage + DB)
        ctx.output_artifacts["processed_assets"] = json.dumps(processed_assets)
        ctx.output_artifacts["processed_video"] = ctx.processed_r2_key or ""

        # Persist processed_assets to DB
        try:
            await db_stage.save_processed_assets(db_pool, ctx.upload_id, processed_assets)
        except Exception as e:
            logger.warning(f"Could not persist processed_assets to DB: {e}")

        logger.info(f"Upload summary: {', '.join(f'{p}={k}' for p, k in processed_assets.items())}")
        await maybe_cancel(ctx, "upload")

        # ================================================================
        # STAGE 9: Publish - Send correct file to each platform API
        # publish_stage uses ctx.get_video_for_platform(platform) which
        # checks ctx.platform_videos[platform] first (local temp files)
        # ================================================================
        try:
            ctx = await run_publish_stage(ctx, db_pool)
        except StageError as e:
            logger.error(f"Publish error: {e.message}")
            ctx.mark_error(e.code.value, e.message)
        await maybe_cancel(ctx, "publish")

        # ================================================================
        # STAGE 10: Notify - Discord webhooks
        # ================================================================
        try:
            ctx = await run_notify_stage(ctx)
        except Exception as e:
            logger.warning(f"Notify error: {e}")

        # ================================================================
        # Complete
        # ================================================================
        ctx.finished_at = datetime.now(timezone.utc)
        ctx.state = "succeeded" if ctx.is_success() else "failed"
        await db_stage.mark_processing_completed(db_pool, ctx)

        if ctx.is_success():
            await db_stage.increment_upload_count(db_pool, user_id)

        success_list = ctx.get_success_platforms()
        failed_list = ctx.get_failed_platforms()
        logger.info(
            f"Pipeline complete: {ctx.state}, "
            f"succeeded={success_list}, failed={failed_list}"
        )
        return ctx.is_success()

    except CancelRequested:
        logger.info(f"Pipeline cancelled: {upload_id}")
        return False
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        if ctx:
            ctx.mark_error("INTERNAL", str(e))
            await db_stage.mark_processing_failed(db_pool, ctx, "INTERNAL", str(e))
        await notify_admin_error("pipeline_failure", {"upload_id": upload_id, "error": str(e)}, db_pool)
        return False
    finally:
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


async def process_jobs():
    global shutdown_requested
    logger.info("Worker started, waiting for jobs...")

    consecutive_redis_errors = 0

    while not shutdown_requested:
        try:
            job_raw = await redis_client.brpop(
                [PRIORITY_JOB_QUEUE, UPLOAD_JOB_QUEUE],
                timeout=int(POLL_INTERVAL)
            )
            consecutive_redis_errors = 0  # Reset on success

            if not job_raw:
                continue

            _, job_json = job_raw
            job_data = json.loads(job_json)

            await run_pipeline(job_data)

        except redis.ReadOnlyError:
            # Redis upgrade in progress - back off and retry
            consecutive_redis_errors += 1
            wait_time = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(
                f"Redis read-only (upgrade in progress), "
                f"retrying in {wait_time:.0f}s (attempt {consecutive_redis_errors})"
            )
            await asyncio.sleep(wait_time)

        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            consecutive_redis_errors += 1
            wait_time = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(
                f"Redis connection error: {e}, "
                f"retrying in {wait_time:.0f}s (attempt {consecutive_redis_errors})"
            )
            await asyncio.sleep(wait_time)

            # If Redis is down for too long, try to reconnect
            if consecutive_redis_errors >= REDIS_MAX_RETRIES:
                logger.error("Redis unreachable after max retries, attempting reconnect...")
                try:
                    global redis_client
                    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                    await redis_client.ping()
                    logger.info("Redis reconnected successfully")
                    consecutive_redis_errors = 0
                except Exception as re_err:
                    logger.error(f"Redis reconnect failed: {re_err}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid job JSON: {e}")
        except Exception as e:
            logger.exception(f"Job processing error: {e}")
            await asyncio.sleep(1)

    logger.info("Worker shutting down...")


async def main():
    global db_pool, redis_client, shutdown_event

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    if not DATABASE_URL:
        logger.error("DATABASE_URL not set")
        sys.exit(1)
    if not REDIS_URL:
        logger.error("REDIS_URL not set")
        sys.exit(1)

    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    logger.info("Database connected")

    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info("Redis connected")

    shutdown_event = asyncio.Event()

    verify_task = asyncio.create_task(run_verification_loop(db_pool, shutdown_event))
    job_task = asyncio.create_task(process_jobs())

    try:
        await notify_admin_worker_start(db_pool)
        done, pending = await asyncio.wait(
            [job_task, verify_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    finally:
        try:
            shutdown_event.set()
        except Exception:
            pass
        try:
            await notify_admin_worker_stop(db_pool)
        except Exception:
            pass
        if db_pool:
            await db_pool.close()
        if redis_client:
            try:
                await redis_client.aclose()
            except AttributeError:
                await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
