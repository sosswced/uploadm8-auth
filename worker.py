"""
UploadM8 Worker Service
=======================
Pipeline orchestrator for video processing.

This worker:
1. Consumes jobs from Redis queue
2. Runs processing stages in order
3. Handles errors gracefully
4. Sends notifications

Run with: python worker.py
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import asyncpg
import redis.asyncio as redis

# Import stages
from stages.errors import StageError, SkipStage, ErrorCode
from stages.context import JobContext, create_context
from stages.entitlements import get_entitlements_from_user
from stages import db as db_stage
from stages import r2 as r2_stage
from stages.telemetry_stage import run_telemetry_stage
from stages.caption_stage import run_caption_stage
from stages.hud_stage import run_hud_stage
from stages.publish_stage import run_publish_stage
from stages.notify_stage import (
    run_notify_stage,
    notify_admin_worker_start,
    notify_admin_worker_stop,
    notify_admin_error
)


# ============================================================
# Configuration
# ============================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [worker] %(message)s"
)
logger = logging.getLogger("uploadm8-worker")

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))


# ============================================================
# Global State
# ============================================================

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None


# ============================================================
# Token Loader (for publish stage)
# ============================================================

async def load_platform_token(user_id: str, platform: str) -> Optional[str]:
    """Load platform token from database."""
    if not db_pool:
        return None
    return await db_stage.load_platform_token(db_pool, user_id, platform)


# ============================================================
# Pipeline Execution
# ============================================================

async def run_pipeline(job_data: dict) -> bool:
    """
    Execute the full processing pipeline.
    
    Stages:
    1. Load records from database
    2. Download files from R2
    3. Telemetry processing (if .map file)
    4. Caption generation (tier-gated)
    5. HUD overlay (tier-gated)
    6. Upload processed video to R2
    7. Publish to platforms
    8. Send notifications
    """
    upload_id = job_data.get("upload_id")
    user_id = job_data.get("user_id")
    job_id = job_data.get("job_id", "unknown")
    
    logger.info(f"Starting pipeline for upload {upload_id} (job {job_id})")
    
    ctx: Optional[JobContext] = None
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    
    try:
        # ============================================================
        # Stage 0: Load data from database
        # ============================================================
        
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        
        # Get entitlements
        entitlements = get_entitlements_from_user(user_record)
        
        # Create context
        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.started_at = datetime.now(timezone.utc)
        
        # Mark as processing
        await db_stage.mark_processing_started(db_pool, ctx)
        
        # ============================================================
        # Stage 1: Download files from R2
        # ============================================================
        
        temp_dir = tempfile.TemporaryDirectory()
        ctx.temp_dir = Path(temp_dir.name)
        
        # Download video
        video_local = ctx.temp_dir / ctx.filename
        await r2_stage.download_file(ctx.source_r2_key, video_local)
        ctx.local_video_path = video_local
        
        # Download telemetry if present
        if ctx.telemetry_r2_key:
            telemetry_local = ctx.temp_dir / "telemetry.map"
            try:
                await r2_stage.download_file(ctx.telemetry_r2_key, telemetry_local)
                ctx.local_telemetry_path = telemetry_local
            except Exception as e:
                logger.warning(f"Failed to download telemetry: {e}")
        
        # ============================================================
        # Stage 2: Telemetry processing
        # ============================================================
        
        try:
            ctx = await run_telemetry_stage(ctx)
        except SkipStage as e:
            logger.info(f"Telemetry stage skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Telemetry stage error: {e.message}")
            # Non-fatal - continue pipeline
        
        # ============================================================
        # Stage 3: Caption generation
        # ============================================================
        
        try:
            ctx = await run_caption_stage(ctx)
        except SkipStage as e:
            logger.info(f"Caption stage skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"Caption stage error: {e.message}")
        
        # ============================================================
        # Stage 4: HUD overlay
        # ============================================================
        
        try:
            ctx = await run_hud_stage(ctx)
        except SkipStage as e:
            logger.info(f"HUD stage skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"HUD stage error: {e.message}")
            # Non-fatal - publish without HUD
        
        # ============================================================
        # Stage 5: Upload processed video to R2
        # ============================================================
        
        if ctx.processed_video_path and ctx.processed_video_path.exists():
            processed_key = r2_stage.get_processed_key(ctx.source_r2_key)
            try:
                await r2_stage.upload_file(ctx.processed_video_path, processed_key)
                ctx.processed_r2_key = processed_key
            except Exception as e:
                logger.warning(f"Failed to upload processed video: {e}")
        
        # ============================================================
        # Stage 6: Publish to platforms
        # ============================================================
        
        ctx = await run_publish_stage(ctx, load_platform_token)
        
        # ============================================================
        # Stage 7: Finalize status
        # ============================================================
        
        if ctx.all_succeeded:
            await db_stage.mark_processing_complete(db_pool, ctx, "completed")
        elif ctx.any_succeeded:
            await db_stage.mark_partial_success(db_pool, ctx)
        else:
            await db_stage.mark_processing_failed(
                db_pool, ctx,
                ErrorCode.PUBLISH_ALL_FAILED.value,
                "All platform uploads failed"
            )
        
        ctx.finished_at = datetime.now(timezone.utc)
        
        # ============================================================
        # Stage 8: Notifications
        # ============================================================
        
        await run_notify_stage(ctx)
        
        logger.info(f"Pipeline complete for {upload_id}: {ctx.status}")
        return ctx.status != "failed"
        
    except StageError as e:
        logger.error(f"Pipeline failed for {upload_id}: {e.message}")
        if ctx and db_pool:
            await db_stage.mark_processing_failed(db_pool, ctx, e.code.value, e.detail or e.message)
            ctx.error_code = e.code.value
            ctx.error_detail = e.detail or e.message
            await run_notify_stage(ctx)
        return False
        
    except Exception as e:
        logger.exception(f"Pipeline error for {upload_id}: {e}")
        if ctx and db_pool:
            await db_stage.mark_processing_failed(db_pool, ctx, ErrorCode.UNKNOWN.value, str(e))
            ctx.error_code = ErrorCode.UNKNOWN.value
            ctx.error_detail = str(e)
            await run_notify_stage(ctx)
        return False
        
    finally:
        # Cleanup temp directory
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


# ============================================================
# Job Consumer
# ============================================================

async def process_job(job_json: str) -> bool:
    """Process a single job from the queue."""
    try:
        job = json.loads(job_json)
        job_type = job.get("type")
        
        if job_type == "process_upload":
            return await run_pipeline(job)
        else:
            logger.warning(f"Unknown job type: {job_type}")
            return False
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid job JSON: {e}")
        return False
    except Exception as e:
        logger.exception(f"Job processing failed: {e}")
        return False


async def worker_loop():
    """Main worker loop - consume jobs from Redis queue."""
    global db_pool, redis_client
    
    # Connect to database
    if DATABASE_URL:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        logger.info("Database connected")
    else:
        logger.error("DATABASE_URL not set")
        return
    
    # Connect to Redis
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected")
    else:
        logger.error("REDIS_URL not set")
        return
    
    logger.info(f"Worker started, listening on queue: {UPLOAD_JOB_QUEUE}")
    await notify_admin_worker_start()
    
    try:
        while True:
            try:
                # BRPOP blocks until a job is available
                result = await redis_client.brpop(
                    [UPLOAD_JOB_QUEUE],
                    timeout=int(POLL_INTERVAL_SECONDS * 10)
                )
                
                if result:
                    queue_name, job_json = result
                    success = await process_job(job_json)
                    if not success:
                        logger.warning(f"Job failed from queue {queue_name}")
                else:
                    # Timeout - no jobs available
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                await notify_admin_error(f"Redis connection lost: {e}")
                await asyncio.sleep(5)
                redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                
            except Exception as e:
                logger.exception(f"Worker loop error: {e}")
                await notify_admin_error(f"Worker loop error: {e}")
                await asyncio.sleep(5)
                
    finally:
        await notify_admin_worker_stop()
        if db_pool:
            await db_pool.close()
        if redis_client:
            await redis_client.close()


def main():
    """Entry point."""
    logger.info("Starting UploadM8 Worker...")
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
