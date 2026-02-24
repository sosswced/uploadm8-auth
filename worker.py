"""
UploadM8 Worker Service v3 — Deferred Processing + Concurrent Jobs

Pipeline order (critical — DO NOT REORDER):
  1.  Download        — Fetch original video + telemetry from R2
  2.  Telemetry       — Parse .map file, calculate Trill score, reverse-geocode
  3.  HUD             — Burn speed overlay onto raw video (before transcode!)
  4.  Watermark       — Burn text watermark (before transcode!)
  5.  Transcode       — Deduplicated per-platform MP4s
  6.  Thumbnail       — Extract frame from FINAL processed video
  7.  Caption         — AI title/caption/hashtags (vision + telemetry)
  8.  Upload          — Per-platform R2 keys
  9.  Publish         — Send to each platform API
  10. Verify          — Delivery verification loop (background)
  11. Notify          — Discord webhooks

NEW IN v3: DEFERRED PROCESSING (Staged Upload Flow)
═══════════════════════════════════════════════════
  PROBLEM (old):
    /complete → immediately enqueued → worker fires NOW
    Scheduled videos were getting processed days before they publish.
    500-pair batch would hammer the worker immediately.

  SOLUTION (new):
    Two status paths:

    A) IMMEDIATE uploads (schedule_mode = "immediate"):
       /complete → status=queued → worker fires immediately (unchanged)

    B) SCHEDULED uploads (schedule_mode = "scheduled" or "smart"):
       /complete → status=staged → worker NOT fired
                                ↓
                   run_scheduler_loop() polls DB every 60s
                   Finds: staged jobs where
                     scheduled_time - NOW ≤ processing_window_minutes
                   Fires worker job → full pipeline EXCEPT publish
                   Worker marks status=ready_to_publish
                                ↓
                   Scheduler finds ready_to_publish jobs where
                     NOW ≥ scheduled_time
                   Fires publish-only mini-pipeline
                   Status → completed/partial/failed

    C) REFRESH SAFETY:
       Files are in R2 the moment the browser PUT succeeds.
       DB row exists with status=staged or pending.
       Refreshing the browser loses the browser state but NOT the upload.
       queue.html shows the job. User can see it. It processes on schedule.

  STATUS FLOW:
    pending → staged → processing → ready_to_publish → completed
    pending → queued  → processing → completed           (immediate)
    any     → failed
    any     → cancelled

  PROCESSING WINDOW:
    Set PROCESSING_WINDOW_MINUTES env var (default: 15).
    Worker starts processing 15 min before scheduled_time.
    For a 500-pair batch scheduled over 30 days, this staggers
    processing naturally — only jobs in the next 15 min window
    fire at any given poll cycle.

CONCURRENCY:
  WORKER_CONCURRENCY=3 (default) — 3 jobs process simultaneously.
  Scheduler loop runs as a separate asyncio task, not a job slot.
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
import signal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import asyncpg
import redis.asyncio as redis

from stages.errors import StageError, SkipStage, CancelRequested
from stages.context import JobContext, create_context
from stages.entitlements import get_entitlements_from_user
from stages import db as db_stage
from stages import r2 as r2_stage
from stages.telemetry_stage import run_telemetry_stage
from stages.transcode_stage import (
    run_transcode_stage, PLATFORM_SPECS,
    get_video_info, needs_transcode, build_ffmpeg_command,
    resolve_reframe_action,
)
from stages.thumbnail_stage import run_thumbnail_stage
from stages.caption_stage import run_caption_stage
from stages.hud_stage import run_hud_stage
from stages.watermark_stage import run_watermark_stage
from stages.publish_stage import run_publish_stage
from stages.verify_stage import run_verification_loop
from stages.notify_stage import (
    run_notify_stage,
    notify_admin_worker_start,
    notify_admin_worker_stop,
    notify_admin_error,
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
)
logger = logging.getLogger("uploadm8-worker")

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "")

# ── 4-Lane Queue Names (must match app.py env vars) ──────────────
PROCESS_PRIORITY_QUEUE = os.environ.get("PROCESS_PRIORITY_QUEUE", "uploadm8:process:priority")
PROCESS_NORMAL_QUEUE   = os.environ.get("PROCESS_NORMAL_QUEUE",   "uploadm8:process:normal")
PUBLISH_PRIORITY_QUEUE = os.environ.get("PUBLISH_PRIORITY_QUEUE", "uploadm8:publish:priority")
PUBLISH_NORMAL_QUEUE   = os.environ.get("PUBLISH_NORMAL_QUEUE",   "uploadm8:publish:normal")
# Legacy queue names kept so old Redis entries dont get lost during transition
UPLOAD_JOB_QUEUE   = os.environ.get("UPLOAD_JOB_QUEUE",   "uploadm8:jobs")
PRIORITY_JOB_QUEUE = os.environ.get("PRIORITY_JOB_QUEUE", "uploadm8:priority")

POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))

# ── Concurrency ───────────────────────────────────────────────────
# WORKER_CONCURRENCY  = FFmpeg-heavy process jobs (CPU-bound, default 3)
# PUBLISH_CONCURRENCY = API-light publish jobs   (I/O-bound, default 5)
WORKER_CONCURRENCY  = int(os.environ.get("WORKER_CONCURRENCY",  "3"))
PUBLISH_CONCURRENCY = int(os.environ.get("PUBLISH_CONCURRENCY", "5"))

# Scheduler: how far in advance to start processing a scheduled upload (minutes)
PROCESSING_WINDOW_MINUTES = int(os.environ.get("PROCESSING_WINDOW_MINUTES", "15"))

# How often the scheduler polls the DB for staged/ready jobs (seconds)
SCHEDULER_POLL_INTERVAL = int(os.environ.get("SCHEDULER_POLL_INTERVAL", "60"))

# Redis resilience
REDIS_RETRY_DELAY = 5.0
REDIS_MAX_RETRIES = 10

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
shutdown_requested = False
shutdown_event: Optional[asyncio.Event] = None

# ── Separate semaphores for each lane ────────────────────────────
# Process semaphore: guards FFmpeg transcode slots (CPU-bound)
# Publish semaphore: guards platform API call slots (I/O-bound)
# Keeping them separate means a 10-minute transcode CANNOT block
# a 10-second TikTok API publish call.
_process_semaphore: Optional[asyncio.Semaphore] = None
_publish_semaphore: Optional[asyncio.Semaphore] = None
# Legacy alias — some internal helpers still reference _job_semaphore
_job_semaphore: Optional[asyncio.Semaphore] = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


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


STAGE_PROGRESS = {
    "init":       5,
    "download":   10,
    "telemetry":  18,
    "hud":        26,
    "watermark":  32,
    "transcode":  58,
    "thumbnail":  65,
    "caption":    75,
    "upload":     87,
    "publish":    96,
    "notify":     99,
}


async def maybe_cancel(ctx: JobContext, stage: str):
    progress = STAGE_PROGRESS.get(stage, 0)
    await db_stage.update_stage_progress(db_pool, ctx.upload_id, stage, progress)
    if await check_cancelled(ctx):
        logger.info(f"[{ctx.upload_id}] Cancel at {stage}")
        await db_stage.mark_cancelled(db_pool, ctx.upload_id)
        raise CancelRequested(ctx.upload_id)


# ---------------------------------------------------------------------------
# Transcode Deduplication
# ---------------------------------------------------------------------------

def _platform_spec_fingerprint(platform: str) -> str:
    spec = PLATFORM_SPECS.get(platform, {})
    return (
        f"{spec.get('video_codec', 'h264')}"
        f"_{spec.get('max_width', 1080)}"
        f"x{spec.get('max_height', 1920)}"
        f"_{spec.get('max_fps', 30)}fps"
        f"_{spec.get('max_duration', 9999)}s"
        f"_{spec.get('sample_rate', 44100)}hz"
    )


def _group_platforms_by_spec(platforms: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in platforms:
        fp = _platform_spec_fingerprint(p)
        groups.setdefault(fp, []).append(p)
    return groups


async def _run_deduplicated_transcode(ctx: JobContext) -> JobContext:
    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No video file to transcode")

    platforms = ctx.platforms or []
    if not platforms:
        raise SkipStage("No target platforms specified")

    source_video = ctx.processed_video_path or ctx.local_video_path
    if not Path(source_video).exists():
        source_video = ctx.local_video_path

    logger.info(f"[{ctx.upload_id}] Dedup transcode: source={source_video.name}, platforms={platforms}")

    try:
        info = await get_video_info(source_video)
        ctx.video_info = {
            "width": info.width, "height": info.height,
            "duration": info.duration, "fps": info.fps,
            "video_codec": info.video_codec, "audio_codec": info.audio_codec,
        }
    except Exception as e:
        logger.warning(f"[{ctx.upload_id}] ffprobe failed: {e} — using standard transcode")
        return await run_transcode_stage(ctx)

    groups = _group_platforms_by_spec(platforms)
    reframe_mode = getattr(ctx, "reframe_mode", "auto") or "auto"
    ctx.platform_videos = {}

    for fingerprint, group_platforms in groups.items():
        canonical = group_platforms[0]
        reframe_action = resolve_reframe_action(info, reframe_mode, canonical)
        needs_tc, reasons = needs_transcode(info, canonical, reframe_action)

        if needs_tc:
            logger.info(f"[{ctx.upload_id}] Transcode [{canonical}] shared by {group_platforms}: {', '.join(reasons)}")
            output_path = ctx.temp_dir / f"transcoded_{canonical}.mp4"
            try:
                cmd = build_ffmpeg_command(source_video, output_path, info, canonical, reframe_action)
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                if proc.returncode != 0 or not output_path.exists():
                    logger.error(f"[{ctx.upload_id}] FFmpeg failed [{canonical}]: {stderr.decode()[-300:]}")
                    for p in group_platforms:
                        ctx.platform_videos[p] = source_video
                    continue

                sz_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"[{ctx.upload_id}] Transcode done [{canonical}]: {sz_mb:.1f}MB → {group_platforms}")
                for p in group_platforms:
                    ctx.platform_videos[p] = output_path

            except Exception as e:
                logger.error(f"[{ctx.upload_id}] Transcode error [{canonical}]: {e}")
                for p in group_platforms:
                    ctx.platform_videos[p] = source_video
        else:
            logger.info(f"[{ctx.upload_id}] Platforms {group_platforms} already compatible")
            for p in group_platforms:
                ctx.platform_videos[p] = source_video

    if ctx.platform_videos:
        first = ctx.platform_videos.get(platforms[0])
        if first and Path(first).exists() and first != source_video:
            ctx.processed_video_path = first
        elif not ctx.processed_video_path:
            ctx.processed_video_path = source_video

    return ctx


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def _should_run_trill(ctx: JobContext) -> bool:
    has_file = (
        ctx.local_telemetry_path is not None
        and Path(ctx.local_telemetry_path).exists()
    )
    if not has_file:
        return False
    setting = ctx.user_settings.get("telemetry_enabled", True)
    if isinstance(setting, str):
        setting = setting.lower() not in ("false", "0", "no", "off")
    return bool(setting)


def _should_run_hud(ctx: JobContext) -> bool:
    if not _should_run_trill(ctx):
        return False
    hud = ctx.user_settings.get("hud_enabled", True)
    if isinstance(hud, str):
        hud = hud.lower() not in ("false", "0", "no", "off")
    if not bool(hud):
        return False
    if ctx.entitlements and not ctx.entitlements.can_burn_hud:
        return False
    return True


def _trill_min_score(ctx: JobContext) -> int:
    raw = ctx.user_settings.get("trill_min_score") or ctx.user_settings.get("trillMinScore") or 0
    try:
        return max(0, min(100, int(raw)))
    except (TypeError, ValueError):
        return 0


def _apply_trill_caption_settings(ctx: JobContext) -> None:
    trill = ctx.trill_score or getattr(ctx, "trill", None)
    if not trill:
        return
    min_score = _trill_min_score(ctx)
    if trill.score < min_score:
        logger.info(f"[{ctx.upload_id}] Trill score {trill.score} < threshold {min_score} — suppressing Trill content")
        ctx.trill_score = None
        ctx.trill = None


# ---------------------------------------------------------------------------
# PROCESSING PIPELINE (all stages except publish)
# ---------------------------------------------------------------------------

async def run_processing_pipeline(job_data: dict) -> bool:
    """
    Run all stages EXCEPT publish.
    Used for both immediate uploads and deferred staged uploads.

    For staged uploads, this sets status=ready_to_publish when done.
    For immediate uploads, this flows directly into publish.
    """
    upload_id = job_data.get("upload_id")
    user_id = job_data.get("user_id")
    job_id = job_data.get("job_id", "unknown")
    is_deferred = job_data.get("deferred", False)  # True = staged, publish later

    logger.info(f"[{upload_id}] Processing start | job={job_id} | deferred={is_deferred}")
    ctx = None
    temp_dir = None

    try:
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            logger.error(f"[{upload_id}] Records not found")
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.started_at = _now_utc()
        ctx.state = "processing"
        ctx.user_settings["_openai_model_override"] = (
            ctx.user_settings.get("trillOpenaiModel")
            or ctx.user_settings.get("trill_openai_model")
            or "gpt-4o"
        )

        await db_stage.mark_processing_started(db_pool, ctx)
        # Record processing_started_at for staged jobs
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE uploads SET processing_started_at = NOW() WHERE id = $1",
                    upload_id,
                )
        except Exception:
            pass

        await maybe_cancel(ctx, "init")

        # ============================================================
        # STAGE 1: Download
        # ============================================================
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
                logger.info(f"[{upload_id}] Telemetry downloaded")
            except Exception as e:
                logger.warning(f"[{upload_id}] Telemetry download failed: {e}")
                ctx.local_telemetry_path = None

        trill_active = _should_run_trill(ctx)
        hud_active = _should_run_hud(ctx)
        logger.info(f"[{upload_id}] Flags: trill={trill_active} hud={hud_active} platforms={ctx.platforms}")
        await maybe_cancel(ctx, "download")

        # ============================================================
        # STAGE 2: Telemetry
        # ============================================================
        if trill_active:
            try:
                ctx = await run_telemetry_stage(ctx)
                _apply_trill_caption_settings(ctx)
            except SkipStage as e:
                logger.info(f"[{upload_id}] Telemetry skipped: {e.reason}")
                trill_active = False
                hud_active = False
            except StageError as e:
                logger.warning(f"[{upload_id}] Telemetry error: {e.message}")
                trill_active = False
                hud_active = False
        else:
            ctx.telemetry = None
            ctx.telemetry_data = None
            ctx.trill = None
            ctx.trill_score = None
        await maybe_cancel(ctx, "telemetry")

        # ============================================================
        # STAGE 3: HUD
        # ============================================================
        if hud_active:
            try:
                ctx = await run_hud_stage(ctx)
            except SkipStage as e:
                logger.info(f"[{upload_id}] HUD skipped: {e.reason}")
            except StageError as e:
                logger.warning(f"[{upload_id}] HUD error: {e.message}")
        await maybe_cancel(ctx, "hud")

        # ============================================================
        # STAGE 4: Watermark
        # ============================================================
        try:
            ctx = await run_watermark_stage(ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Watermark skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Watermark error: {e.message}")
        await maybe_cancel(ctx, "watermark")

        # ============================================================
        # STAGE 5: Transcode (deduplicated)
        # ============================================================
        try:
            ctx = await _run_deduplicated_transcode(ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Transcode skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Transcode error: {e.message}")
            source = ctx.processed_video_path or ctx.local_video_path
            for p in (ctx.platforms or []):
                ctx.platform_videos[p] = source
        await maybe_cancel(ctx, "transcode")

        # ============================================================
        # STAGE 6: Thumbnail
        # ============================================================
        try:
            ctx = await run_thumbnail_stage(ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Thumbnail skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Thumbnail error: {e.message}")
        await maybe_cancel(ctx, "thumbnail")

        # ============================================================
        # STAGE 7: Caption
        # ============================================================
        try:
            ctx = await run_caption_stage(ctx)
            await db_stage.save_generated_metadata(db_pool, ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Caption skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Caption error: {e.message}")
        await maybe_cancel(ctx, "caption")

        # ============================================================
        # STAGE 8: Upload processed files to R2
        # ============================================================
        ctx.mark_stage("upload")
        processed_assets: Dict[str, str] = {}
        uploaded_paths: Dict[str, str] = {}

        if ctx.platform_videos:
            for platform, video_path in ctx.platform_videos.items():
                if not video_path or not Path(video_path).exists():
                    continue
                path_str = str(video_path)
                if path_str in uploaded_paths:
                    processed_assets[platform] = uploaded_paths[path_str]
                    logger.info(f"[{upload_id}] {platform} reuses {uploaded_paths[path_str]}")
                    continue
                r2_key = f"processed/{ctx.user_id}/{upload_id}/{platform}.mp4"
                try:
                    await r2_stage.upload_file(Path(video_path), r2_key, "video/mp4")
                    processed_assets[platform] = r2_key
                    uploaded_paths[path_str] = r2_key
                    sz = Path(video_path).stat().st_size / 1024 / 1024
                    logger.info(f"[{upload_id}] R2 uploaded {platform}: {r2_key} ({sz:.1f}MB)")
                except Exception as e:
                    logger.error(f"[{upload_id}] R2 upload failed [{platform}]: {e}")

        fallback = ctx.processed_video_path or ctx.local_video_path
        if fallback and Path(fallback).exists():
            default_key = f"processed/{ctx.user_id}/{upload_id}/default.mp4"
            try:
                await r2_stage.upload_file(Path(fallback), default_key, "video/mp4")
                processed_assets["default"] = default_key
                ctx.processed_r2_key = default_key
            except Exception as e:
                logger.warning(f"[{upload_id}] Default R2 upload failed: {e}")

        ctx.output_artifacts["processed_assets"] = json.dumps(processed_assets)
        ctx.output_artifacts["processed_video"] = ctx.processed_r2_key or ""

        try:
            await db_stage.save_processed_assets(db_pool, ctx.upload_id, processed_assets)
        except Exception as e:
            logger.warning(f"[{upload_id}] Could not persist processed_assets: {e}")

        await maybe_cancel(ctx, "upload")

        # ============================================================
        # DEFERRED PATH: mark ready_to_publish and STOP
        # The scheduler will fire publish when scheduled_time arrives.
        # ============================================================
        if is_deferred:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status = 'ready_to_publish',
                            ready_to_publish_at = NOW(),
                            processed_assets = $2,
                            updated_at = NOW()
                        WHERE id = $1
                        """,
                        upload_id,
                        json.dumps(processed_assets),
                    )
                logger.info(f"[{upload_id}] Processing complete → ready_to_publish (waiting for scheduled_time)")
            except Exception as e:
                logger.error(f"[{upload_id}] Failed to set ready_to_publish: {e}")
            return True

        # ============================================================
        # IMMEDIATE PATH: continue to publish
        # ============================================================
        return await run_publish_and_notify(ctx, upload_id, user_id)

    except CancelRequested:
        logger.info(f"[{upload_id}] Pipeline cancelled")
        return False
    except Exception as e:
        logger.exception(f"[{upload_id}] Processing failed: {e}")
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


async def run_publish_and_notify(
    ctx: JobContext,
    upload_id: str,
    user_id: str,
) -> bool:
    """
    Run publish + notify stages and finalize upload status.
    Called by both the immediate pipeline and the deferred publish path.
    """
    try:
        ctx = await run_publish_stage(ctx, db_pool)
    except StageError as e:
        logger.error(f"[{upload_id}] Publish error: {e.message}")
        ctx.mark_error(e.code.value, e.message)

    try:
        ctx = await run_notify_stage(ctx)
    except Exception as e:
        logger.warning(f"[{upload_id}] Notify error: {e}")

    ctx.finished_at = _now_utc()
    if ctx.is_partial_success():
        ctx.state = "partial"
    elif ctx.is_success():
        ctx.state = "succeeded"
    else:
        ctx.state = "failed"

    await db_stage.mark_processing_completed(db_pool, ctx)

    if ctx.is_success():
        await db_stage.increment_upload_count(db_pool, user_id)

    elapsed = (ctx.finished_at - ctx.started_at).total_seconds() if ctx.started_at else 0
    logger.info(
        f"[{upload_id}] Pipeline {ctx.state} in {elapsed:.1f}s | "
        f"ok={ctx.get_success_platforms()} fail={ctx.get_failed_platforms()}"
    )
    return ctx.is_success()


# ---------------------------------------------------------------------------
# PUBLISH-ONLY pipeline for ready_to_publish jobs
# ---------------------------------------------------------------------------

async def run_deferred_publish(upload_id: str, user_id: str) -> bool:
    """
    Called by the scheduler when a ready_to_publish job hits its scheduled_time.
    Loads the already-processed assets from DB and runs publish + notify.
    Does NOT re-download or re-transcode anything.
    """
    logger.info(f"[{upload_id}] Deferred publish firing")
    ctx = None

    try:
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            logger.error(f"[{upload_id}] Records not found for deferred publish")
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        # Reconstruct minimal job_data
        job_data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "job_id": f"deferred-publish-{upload_id}",
        }
        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.started_at = _now_utc()
        ctx.state = "processing"

        # Mark processing started again (publish phase)
        await db_stage.mark_processing_started(db_pool, ctx)

        # Restore platform_videos from processed_assets stored in DB
        processed_assets_json = upload_record.get("processed_assets") or "{}"
        try:
            processed_assets: Dict[str, str] = json.loads(processed_assets_json)
        except Exception:
            processed_assets = {}

        if not processed_assets:
            logger.error(f"[{upload_id}] No processed_assets found — cannot publish")
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE uploads SET status='failed', error_code='NO_PROCESSED_ASSETS', updated_at=NOW() WHERE id=$1",
                    upload_id,
                )
            return False

        # platform_videos aren't local paths anymore — they're R2 keys.
        # publish_stage needs to download them temporarily.
        # We store them in ctx.output_artifacts for publish_stage to read.
        ctx.output_artifacts["processed_assets"] = json.dumps(processed_assets)
        ctx.processed_r2_key = processed_assets.get("default") or next(iter(processed_assets.values()), "")

        # For publish_stage to get the right file per platform, we need temp downloads.
        # Create a minimal temp dir and download each unique asset.
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        ctx.temp_dir = temp_dir

        downloaded: Dict[str, Path] = {}
        for platform, r2_key in processed_assets.items():
            if platform == "default":
                continue
            if r2_key in downloaded:
                ctx.platform_videos[platform] = downloaded[r2_key]
                continue
            local_path = temp_dir / f"{platform}.mp4"
            try:
                await r2_stage.download_file(r2_key, local_path)
                ctx.platform_videos[platform] = local_path
                downloaded[r2_key] = local_path
                logger.info(f"[{upload_id}] Downloaded {platform} for publish: {r2_key}")
            except Exception as e:
                logger.error(f"[{upload_id}] Failed to download {platform} asset: {e}")

        if not ctx.platform_videos:
            logger.error(f"[{upload_id}] No platform videos available after download")
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE uploads SET status='failed', error_code='ASSET_DOWNLOAD_FAILED', updated_at=NOW() WHERE id=$1",
                    upload_id,
                )
            temp_dir_obj.cleanup()
            return False

        # Also set processed_video_path for fallback
        default_key = processed_assets.get("default")
        if default_key:
            default_local = temp_dir / "default.mp4"
            try:
                await r2_stage.download_file(default_key, default_local)
                ctx.processed_video_path = default_local
            except Exception:
                pass

        try:
            result = await run_publish_and_notify(ctx, upload_id, user_id)
            return result
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass

    except Exception as e:
        logger.exception(f"[{upload_id}] Deferred publish failed: {e}")
        if ctx:
            ctx.mark_error("INTERNAL", str(e))
            await db_stage.mark_processing_failed(db_pool, ctx, "INTERNAL", str(e))
        await notify_admin_error("deferred_publish_failure", {"upload_id": upload_id, "error": str(e)}, db_pool)
        return False


# ---------------------------------------------------------------------------
# SCHEDULER LOOP
# ---------------------------------------------------------------------------

async def run_scheduler_loop() -> None:
    """
    Background loop that manages deferred (staged/ready_to_publish) uploads.

    Runs every SCHEDULER_POLL_INTERVAL seconds.

    Two checks per cycle:

    1. STAGED JOBS — files uploaded, not yet processed:
       Finds jobs where:
         status = 'staged'
         AND scheduled_time - NOW() <= processing_window_minutes
       Fires processing pipeline with deferred=True flag.
       Pipeline runs transcode/caption/etc., marks ready_to_publish.

    2. READY_TO_PUBLISH JOBS — processed, waiting for publish time:
       Finds jobs where:
         status = 'ready_to_publish'
         AND scheduled_time <= NOW()
       Fires deferred publish pipeline.

    Both dispatch as asyncio tasks governed by _job_semaphore.
    """
    global shutdown_requested, _job_semaphore

    logger.info(
        f"Scheduler loop started | "
        f"poll_interval={SCHEDULER_POLL_INTERVAL}s | "
        f"processing_window={PROCESSING_WINDOW_MINUTES}min"
    )

    while not shutdown_requested:
        try:
            now = _now_utc()
            process_cutoff = now + timedelta(minutes=PROCESSING_WINDOW_MINUTES)

            async with db_pool.acquire() as conn:

                # --------------------------------------------------------
                # CHECK 1: staged jobs entering processing window
                # --------------------------------------------------------
                # ── Capacity-aware dispatch ────────────────────────────────
                # Only pull as many staged jobs as there are free process slots.
                # This prevents dispatching 500-job batches into memory when
                # workers are already saturated.
                free_slots = WORKER_CONCURRENCY - (
                    WORKER_CONCURRENCY - _process_semaphore._value
                    if _process_semaphore else WORKER_CONCURRENCY
                )
                dispatch_limit = max(1, free_slots)

                staged_jobs = await conn.fetch(
                    """
                    SELECT u.id AS upload_id, u.user_id
                    FROM uploads u
                    WHERE u.status = 'staged'
                      AND u.scheduled_time IS NOT NULL
                      AND u.scheduled_time <= $1
                    ORDER BY u.scheduled_time ASC
                    LIMIT $2
                    """,
                    process_cutoff,
                    dispatch_limit,
                )

                for row in staged_jobs:
                    upload_id = str(row["upload_id"])
                    user_id = str(row["user_id"])

                    # Atomically claim the job (prevent duplicate dispatch)
                    updated = await conn.fetchval(
                        """
                        UPDATE uploads
                        SET status = 'queued', updated_at = NOW()
                        WHERE id = $1 AND status = 'staged'
                        RETURNING id
                        """,
                        row["upload_id"],
                    )
                    if not updated:
                        continue  # Already claimed by another worker instance

                    logger.info(
                        f"[{upload_id}] Scheduler: staging → queued for processing "
                        f"(scheduled at {row.get('scheduled_time')})"
                    )

                    job_data = {
                        "upload_id": upload_id,
                        "user_id": user_id,
                        "job_id": f"scheduled-{upload_id}",
                        "deferred": True,  # <-- tells pipeline to stop before publish
                    }

                    task = asyncio.create_task(_run_job_with_semaphore(job_data))
                    logger.debug(f"[{upload_id}] Processing task dispatched")

                # --------------------------------------------------------
                # CHECK 2: ready_to_publish jobs past scheduled_time
                # --------------------------------------------------------
                ready_jobs = await conn.fetch(
                    """
                    SELECT u.id AS upload_id, u.user_id, u.scheduled_time
                    FROM uploads u
                    WHERE u.status = 'ready_to_publish'
                      AND u.scheduled_time IS NOT NULL
                      AND u.scheduled_time <= $1
                    ORDER BY u.scheduled_time ASC
                    LIMIT 50
                    """,
                    now,
                )

                for row in ready_jobs:
                    upload_id = str(row["upload_id"])
                    user_id = str(row["user_id"])

                    # Atomically claim
                    updated = await conn.fetchval(
                        """
                        UPDATE uploads
                        SET status = 'processing', updated_at = NOW()
                        WHERE id = $1 AND status = 'ready_to_publish'
                        RETURNING id
                        """,
                        row["upload_id"],
                    )
                    if not updated:
                        continue

                    logger.info(
                        f"[{upload_id}] Scheduler: ready_to_publish → publishing NOW "
                        f"(was scheduled for {row.get('scheduled_time')})"
                    )

                    asyncio.create_task(
                        _run_deferred_publish_with_semaphore(upload_id, user_id)
                    )

        except asyncpg.PostgresError as e:
            logger.warning(f"Scheduler DB error: {e}")
        except Exception as e:
            logger.exception(f"Scheduler loop error: {e}")

        # Wait for next poll cycle
        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=SCHEDULER_POLL_INTERVAL,
            )
            break  # shutdown was signalled
        except asyncio.TimeoutError:
            pass  # normal — continue loop

    logger.info("Scheduler loop stopped")


async def _run_job_with_semaphore(job_data: dict) -> None:
    """
    Wrapper: run processing pipeline (FFmpeg-heavy) inside the PROCESS semaphore.
    Uses _process_semaphore (WORKER_CONCURRENCY=3 slots by default).
    """
    async with _process_semaphore:
        try:
            await run_processing_pipeline(job_data)
        except Exception as e:
            logger.exception(f"[{job_data.get('upload_id')}] Unhandled pipeline error: {e}")


async def _run_deferred_publish_with_semaphore(upload_id: str, user_id: str) -> None:
    """
    Wrapper: run deferred publish (API-light) inside the PUBLISH semaphore.
    Uses _publish_semaphore (PUBLISH_CONCURRENCY=5 slots by default).
    This NEVER blocks on FFmpeg transcode slots — separate lane entirely.
    """
    async with _publish_semaphore:
        try:
            await run_deferred_publish(upload_id, user_id)
        except Exception as e:
            logger.exception(f"[{upload_id}] Unhandled deferred publish error: {e}")


# ---------------------------------------------------------------------------
# Redis job consumer (immediate uploads only)
# ---------------------------------------------------------------------------

async def _process_one_job(job_json: str) -> None:
    try:
        job_data = json.loads(job_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid job JSON: {e}")
        return
    async with _process_semaphore:
        try:
            await run_processing_pipeline(job_data)
        except Exception as e:
            logger.exception(f"Unhandled pipeline exception: {e}")


async def process_jobs() -> None:
    """
    Consume process-lane jobs from Redis (FFmpeg-heavy).
    Reads from [process:priority, process:normal] + legacy queues.
    Scheduled/staged jobs are handled by run_scheduler_loop() instead.
    """
    global shutdown_requested, redis_client, _process_semaphore, _publish_semaphore, _job_semaphore

    _process_semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)
    _publish_semaphore = asyncio.Semaphore(PUBLISH_CONCURRENCY)
    _job_semaphore     = _process_semaphore  # legacy alias

    # All queues worker should drain — priority first, then normal, then legacy
    all_process_queues = [
        PROCESS_PRIORITY_QUEUE,
        PROCESS_NORMAL_QUEUE,
        PRIORITY_JOB_QUEUE,   # legacy
        UPLOAD_JOB_QUEUE,     # legacy
    ]

    logger.info(
        f"Job consumer started | "        f"process_concurrency={WORKER_CONCURRENCY} | "        f"publish_concurrency={PUBLISH_CONCURRENCY} | "        f"process_queues={PROCESS_PRIORITY_QUEUE}, {PROCESS_NORMAL_QUEUE}"
    )

    consecutive_redis_errors = 0
    active_tasks: List[asyncio.Task] = []

    while not shutdown_requested:
        active_tasks = [t for t in active_tasks if not t.done()]

        try:
            job_raw = await redis_client.brpop(
                all_process_queues,
                timeout=int(POLL_INTERVAL),
            )
            consecutive_redis_errors = 0

            if not job_raw:
                continue

            _, job_json = job_raw
            task = asyncio.create_task(_process_one_job(job_json))
            active_tasks.append(task)

        except redis.ReadOnlyError:
            consecutive_redis_errors += 1
            wait = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(f"Redis read-only, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)

        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            consecutive_redis_errors += 1
            wait = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(f"Redis error: {e}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
            if consecutive_redis_errors >= REDIS_MAX_RETRIES:
                try:
                    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                    await redis_client.ping()
                    logger.info("Redis reconnected")
                    consecutive_redis_errors = 0
                except Exception as re_err:
                    logger.error(f"Redis reconnect failed: {re_err}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid job JSON: {e}")
        except Exception as e:
            logger.exception(f"Job consumer error: {e}")
            await asyncio.sleep(1)

    if active_tasks:
        logger.info(f"Shutdown: waiting for {len(active_tasks)} in-flight jobs...")
        await asyncio.gather(*active_tasks, return_exceptions=True)

    logger.info("Job consumer stopped")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    global db_pool, redis_client, shutdown_event

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    if not DATABASE_URL:
        logger.error("DATABASE_URL not set")
        sys.exit(1)
    if not REDIS_URL:
        logger.error("REDIS_URL not set")
        sys.exit(1)

    total_concurrency = WORKER_CONCURRENCY + PUBLISH_CONCURRENCY
    db_min = max(2, total_concurrency)
    db_max = max(15, total_concurrency * 3)
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=db_min, max_size=db_max)
    logger.info(f"Database connected | pool={db_min}-{db_max}")

    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info("Redis connected")

    shutdown_event = asyncio.Event()

    # Four concurrent loops:
    # 1. process_jobs     — Redis consumer for immediate uploads
    # 2. run_scheduler_loop — polls DB for staged/ready_to_publish jobs
    # 3. run_verification_loop — delivery verification polling
    tasks = [
        asyncio.create_task(process_jobs()),
        asyncio.create_task(run_scheduler_loop()),
        asyncio.create_task(run_verification_loop(db_pool, shutdown_event)),
    ]

    try:
        await notify_admin_worker_start(db_pool)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
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
