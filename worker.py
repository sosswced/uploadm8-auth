"""
UploadM8 Worker Service - Upgraded Pipeline v2

Pipeline order (critical — DO NOT REORDER):
  1.  Download        — Fetch original video + telemetry from R2
  2.  Telemetry       — Parse .map file, calculate Trill score, reverse-geocode
  3.  HUD             — Burn speed overlay onto raw video (before transcode!)
  4.  Watermark       — Burn text watermark (before transcode!)
  5.  Transcode       — Smart per-platform MP4s with DEDUPLICATION
                        (Instagram + Facebook share one transcode, etc.)
  6.  Thumbnail       — Extract frame from the FINAL processed video
  7.  Caption         — AI-generate title/caption/hashtags (vision + telemetry)
  8.  Upload          — Upload EACH platform MP4 to its own R2 key
  9.  Publish         — Send correct file to each platform API
  10. Verify          — Delivery verification loop (background)
  11. Notify          — Discord webhooks

UPGRADE SUMMARY v2:
  - CONCURRENT JOB PROCESSING: WORKER_CONCURRENCY env var (default 3)
    Multiple jobs run simultaneously using an asyncio Semaphore.
    Each job gets its own temp dir, context, and DB connection slot.

  - SMART STAGE SKIPPING:
    If NO telemetry (.map file) → skip Telemetry, HUD, and Trill-dependent
    caption content automatically. Only run transcode, thumbnail, caption
    (visual-only), upload, publish, notify.
    If trill_enabled=False in user settings → skip HUD regardless of entitlements.
    If hud_enabled=False → skip HUD regardless of telemetry presence.
    If trill score < user's trillMinScore threshold → suppress Trill content
    from captions but continue pipeline normally.

  - TRANSCODE DEDUPLICATION:
    Instagram and Facebook have IDENTICAL specs (H.264, AAC, 1080x1920, 30fps).
    TikTok and YouTube have identical resolution/codec but different max durations.
    Rather than running FFmpeg 4 times, the worker builds a spec fingerprint per
    platform, groups platforms by fingerprint, transcodes ONCE per unique spec,
    and assigns the same output Path to all matching platforms.
    Result: 2-4 FFmpeg passes instead of up to 4, cutting transcode time ~40%.

  - SETTINGS WIRED CORRECTLY:
    telemetry_enabled  → gates Trill/HUD pipeline
    hud_enabled        → gates HUD burn
    speeding_mph       → passed to telemetry_stage for Trill scoring
    euphoria_mph       → passed to telemetry_stage
    trillMinScore      → gates whether AI caption uses Trill content
    trillAiEnhance     → enables/disables AI caption generation for Trill videos
    trillOpenaiModel   → model selection passed to caption stage
    auto_generate_captions / auto_generate_hashtags → caption stage honours these

  - HUD BURN POSITION FIXED:
    HUD burns on raw local_video_path BEFORE transcode.
    Transcode then encodes the HUD-burned video into platform specs.
    This ensures HUD is pixel-sharp at final platform resolution, not
    rescaled/recompressed after burning.

  - 500-PAIR THROUGHPUT ESTIMATE:
    Single worker (WORKER_CONCURRENCY=1): ~1.5-2 min/job = 12-17h for 500
    Three concurrent (WORKER_CONCURRENCY=3, default): ~6-7h for 500
    Five concurrent (WORKER_CONCURRENCY=5): ~3-4h for 500
    Bottleneck is FFmpeg CPU. Render's starter instance (2 vCPU) can safely
    run 3 concurrent FFmpeg processes. Upgrade to 4 vCPU for CONCURRENCY=5.
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
from typing import Optional, Dict, List, Tuple

import asyncpg
import redis.asyncio as redis

from stages.errors import StageError, SkipStage, CancelRequested
from stages.context import JobContext, create_context
from stages.entitlements import get_entitlements_from_user
from stages import db as db_stage
from stages import r2 as r2_stage
from stages.telemetry_stage import run_telemetry_stage
from stages.transcode_stage import run_transcode_stage, PLATFORM_SPECS, get_video_info, needs_transcode, build_ffmpeg_command
from stages.thumbnail_stage import run_thumbnail_stage
from stages.caption_stage import run_caption_stage
from stages.hud_stage import run_hud_stage
from stages.watermark_stage import run_watermark_stage
from stages.publish_stage import run_publish_stage
from stages.verify_stage import run_verification_loop
from stages.notify_stage import run_notify_stage, notify_admin_worker_start, notify_admin_worker_stop, notify_admin_error

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
)
logger = logging.getLogger("uploadm8-worker")

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")
PRIORITY_JOB_QUEUE = os.environ.get("PRIORITY_JOB_QUEUE", "uploadm8:priority")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))

# CONCURRENCY: How many jobs run in parallel on this worker process.
# Render starter (2 vCPU): use 3.  Standard (4 vCPU): use 5.
# Set WORKER_CONCURRENCY=1 to revert to sequential for debugging.
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "3"))

# Redis resilience
REDIS_RETRY_DELAY = 5.0
REDIS_MAX_RETRIES = 10

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
shutdown_requested = False
shutdown_event: Optional[asyncio.Event] = None

# Semaphore limits concurrent job execution
_job_semaphore: Optional[asyncio.Semaphore] = None


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


# Maps each pipeline checkpoint to an approximate % complete value.
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
    """Write progress to DB and check for cancellation at each stage boundary."""
    progress = STAGE_PROGRESS.get(stage, 0)
    await db_stage.update_stage_progress(db_pool, ctx.upload_id, stage, progress)

    if await check_cancelled(ctx):
        logger.info(f"Cancel at {stage} for {ctx.upload_id}")
        await db_stage.mark_cancelled(db_pool, ctx.upload_id)
        raise CancelRequested(ctx.upload_id)


# ---------------------------------------------------------------------------
# Transcode Deduplication
# ---------------------------------------------------------------------------

def _platform_spec_fingerprint(platform: str) -> str:
    """
    Compute a compact fingerprint for a platform's transcode spec.
    Platforms sharing the same fingerprint can share one FFmpeg output.

    Key dimensions: codec, max_width, max_height, max_fps, max_duration, sample_rate
    """
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
    """
    Group platforms by their transcode spec fingerprint.

    Returns a dict of fingerprint → [platform1, platform2, ...]
    The first platform in each group is the "canonical" platform
    whose spec will be used for the single FFmpeg pass.

    Example output given ["tiktok","youtube","instagram","facebook"]:
    {
      "h264_1080x1920_60fps_600s_44100hz": ["tiktok"],
      "h264_1080x1920_60fps_60s_48000hz":  ["youtube"],
      "h264_1080x1920_30fps_90s_44100hz":  ["instagram", "facebook"],
    }
    """
    groups: Dict[str, List[str]] = {}
    for p in platforms:
        fp = _platform_spec_fingerprint(p)
        groups.setdefault(fp, []).append(p)
    return groups


async def _run_deduplicated_transcode(ctx: JobContext) -> JobContext:
    """
    Transcode with deduplication — run one FFmpeg pass per unique spec,
    then assign the resulting file to all platforms with that spec.

    Instagram + Facebook typically share a single pass.
    TikTok and YouTube each get their own pass (different durations/sample rates).

    Falls back to the standard run_transcode_stage if anything unexpected occurs.
    """
    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No video file to transcode")

    platforms = ctx.platforms or []
    if not platforms:
        raise SkipStage("No target platforms specified")

    # Source video: HUD+watermarked if available, else original
    source_video = ctx.processed_video_path or ctx.local_video_path
    if not Path(source_video).exists():
        source_video = ctx.local_video_path

    logger.info(f"Dedup transcode: source={source_video.name}, platforms={platforms}")

    try:
        info = await get_video_info(source_video)
        logger.info(
            f"Video info: {info.width}x{info.height} "
            f"{info.fps:.1f}fps {info.duration:.1f}s "
            f"codec={info.video_codec}"
        )
        ctx.video_info = {
            "width": info.width,
            "height": info.height,
            "duration": info.duration,
            "fps": info.fps,
            "video_codec": info.video_codec,
            "audio_codec": info.audio_codec,
        }
    except Exception as e:
        logger.warning(f"ffprobe failed: {e} — falling back to standard transcode")
        return await run_transcode_stage(ctx)

    groups = _group_platforms_by_spec(platforms)
    ctx.platform_videos = {}

    for fingerprint, group_platforms in groups.items():
        canonical = group_platforms[0]
        needs_tc, reasons = needs_transcode(info, canonical)

        if needs_tc:
            logger.info(
                f"Transcode group [{canonical}] (shared by {group_platforms}): "
                f"{', '.join(reasons)}"
            )
            output_path = ctx.temp_dir / f"transcoded_{canonical}.mp4"
            try:
                cmd = build_ffmpeg_command(source_video, output_path, info, canonical)

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                if proc.returncode != 0 or not output_path.exists():
                    error_detail = stderr.decode()[-400:]
                    logger.error(f"FFmpeg failed for {canonical}: {error_detail}")
                    # Fall back: all platforms in group use source video
                    for p in group_platforms:
                        ctx.platform_videos[p] = source_video
                    continue

                sz_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(
                    f"Transcode complete [{canonical}]: {sz_mb:.1f}MB → "
                    f"shared by {group_platforms}"
                )

                # Assign same output path to ALL platforms in this spec group
                for p in group_platforms:
                    ctx.platform_videos[p] = output_path

            except Exception as e:
                logger.error(f"Transcode error [{canonical}]: {e}")
                for p in group_platforms:
                    ctx.platform_videos[p] = source_video
        else:
            logger.info(f"Platforms {group_platforms} already compatible — no transcode needed")
            for p in group_platforms:
                ctx.platform_videos[p] = source_video

    # Update processed_video_path to the first transcoded output
    # for backward compatibility (thumbnail, caption use this)
    if ctx.platform_videos:
        first_platform = platforms[0]
        candidate = ctx.platform_videos.get(first_platform)
        if candidate and Path(candidate).exists() and candidate != source_video:
            ctx.processed_video_path = candidate
        elif not ctx.processed_video_path:
            ctx.processed_video_path = source_video

    logger.info(
        f"Dedup transcode summary: "
        f"{len(groups)} unique spec(s) for {len(platforms)} platform(s)"
    )
    return ctx


# ---------------------------------------------------------------------------
# Settings helpers — read user_settings from DB into pipeline decisions
# ---------------------------------------------------------------------------

def _should_run_trill(ctx: JobContext) -> bool:
    """
    Determine whether Trill-specific processing should run.

    Rules (all must pass):
    1. A .map telemetry file must be available (physical presence)
    2. User must have telemetry_enabled=True in settings
    3. User must have entitlement can_burn_hud OR trill is just for captions
    """
    has_telemetry_file = (
        ctx.local_telemetry_path is not None
        and Path(ctx.local_telemetry_path).exists()
    )
    if not has_telemetry_file:
        return False

    trill_setting = ctx.user_settings.get("telemetry_enabled", True)
    # Convert from various truthy formats the DB might store
    if isinstance(trill_setting, str):
        trill_setting = trill_setting.lower() not in ("false", "0", "no", "off")
    return bool(trill_setting)


def _should_run_hud(ctx: JobContext) -> bool:
    """
    Determine whether HUD burn should run.

    Rules (all must pass):
    1. Trill pipeline is running (_should_run_trill returned True)
    2. hud_enabled=True in user settings
    3. User tier has can_burn_hud entitlement
    """
    if not _should_run_trill(ctx):
        return False

    hud_setting = ctx.user_settings.get("hud_enabled", True)
    if isinstance(hud_setting, str):
        hud_setting = hud_setting.lower() not in ("false", "0", "no", "off")
    if not bool(hud_setting):
        return False

    if ctx.entitlements and not ctx.entitlements.can_burn_hud:
        return False

    return True


def _trill_min_score(ctx: JobContext) -> int:
    """
    Read the user's minimum Trill score threshold for content generation.
    If the Trill score is BELOW this value, the caption stage will not
    inject Trill content but will still run visual-only generation.
    """
    raw = ctx.user_settings.get("trill_min_score") or ctx.user_settings.get("trillMinScore") or 0
    try:
        return max(0, min(100, int(raw)))
    except (TypeError, ValueError):
        return 0


def _apply_trill_caption_settings(ctx: JobContext) -> None:
    """
    If Trill score is below user's minimum threshold, strip Trill data
    from context so caption stage won't reference it — but keep running.
    """
    trill = ctx.trill_score or getattr(ctx, "trill", None)
    if not trill:
        return

    min_score = _trill_min_score(ctx)
    if trill.score < min_score:
        logger.info(
            f"Trill score {trill.score} < threshold {min_score} — "
            "suppressing Trill content from captions"
        )
        # Null out Trill data so caption_stage skips Trill-specific prompts
        ctx.trill_score = None
        ctx.trill = None
        # Keep telemetry for location/speed context — just drop the Trill score


def _ai_model_for_user(ctx: JobContext) -> str:
    """Read the user's preferred OpenAI model from settings."""
    return ctx.user_settings.get("trillOpenaiModel") or ctx.user_settings.get("trill_openai_model") or "gpt-4o"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(job_data: dict) -> bool:
    """
    Execute the full processing pipeline for one upload job.

    This function is designed to be called concurrently from process_jobs().
    Each invocation is isolated: own temp dir, own JobContext, own DB operations.
    """
    upload_id = job_data.get("upload_id")
    user_id = job_data.get("user_id")
    job_id = job_data.get("job_id", "unknown")

    logger.info(f"[{upload_id}] Pipeline start | job={job_id}")
    ctx = None
    temp_dir = None

    try:
        # ------------------------------------------------------------------ #
        # Load records                                                         #
        # ------------------------------------------------------------------ #
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            logger.error(f"[{upload_id}] Records not found")
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.started_at = datetime.now(timezone.utc)
        ctx.state = "processing"

        # Inject model preference into context for caption stage to read
        ctx.user_settings["_openai_model_override"] = _ai_model_for_user(ctx)

        await db_stage.mark_processing_started(db_pool, ctx)
        await maybe_cancel(ctx, "init")

        # ================================================================
        # STAGE 1: Download
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
                logger.info(f"[{upload_id}] Telemetry file downloaded")
            except Exception as e:
                logger.warning(f"[{upload_id}] Telemetry download failed: {e}")
                ctx.local_telemetry_path = None

        # Determine at download time whether Trill pipeline will run
        trill_active = _should_run_trill(ctx)
        hud_active = _should_run_hud(ctx)
        logger.info(
            f"[{upload_id}] Pipeline flags: "
            f"trill={trill_active} | hud={hud_active} | "
            f"platforms={ctx.platforms}"
        )
        await maybe_cancel(ctx, "download")

        # ================================================================
        # STAGE 2: Telemetry — ONLY runs if trill_active
        # ================================================================
        if trill_active:
            try:
                ctx = await run_telemetry_stage(ctx)
                logger.info(f"[{upload_id}] Telemetry complete")

                # Check Trill min score threshold — may suppress Trill content
                _apply_trill_caption_settings(ctx)

            except SkipStage as e:
                logger.info(f"[{upload_id}] Telemetry skipped: {e.reason}")
                trill_active = False  # Cascade — HUD also won't run
                hud_active = False
            except StageError as e:
                logger.warning(f"[{upload_id}] Telemetry error: {e.message}")
                trill_active = False
                hud_active = False
        else:
            logger.info(f"[{upload_id}] Telemetry skipped — no .map file or trill disabled in settings")
            ctx.telemetry = None
            ctx.telemetry_data = None
            ctx.trill = None
            ctx.trill_score = None

        await maybe_cancel(ctx, "telemetry")

        # ================================================================
        # STAGE 3: HUD — ONLY runs if hud_active
        # Burns onto raw local_video_path BEFORE transcode
        # This preserves HUD sharpness at final platform resolution
        # ================================================================
        if hud_active:
            try:
                ctx = await run_hud_stage(ctx)
                logger.info(f"[{upload_id}] HUD burned")
            except SkipStage as e:
                logger.info(f"[{upload_id}] HUD skipped: {e.reason}")
            except StageError as e:
                logger.warning(f"[{upload_id}] HUD error (non-fatal): {e.message}")
        else:
            logger.info(f"[{upload_id}] HUD skipped — not active for this job")

        await maybe_cancel(ctx, "hud")

        # ================================================================
        # STAGE 4: Watermark
        # Burns onto ctx.processed_video_path (HUD output) or local_video_path
        # ================================================================
        try:
            ctx = await run_watermark_stage(ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Watermark skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Watermark error: {e.message}")

        await maybe_cancel(ctx, "watermark")

        # ================================================================
        # STAGE 5: Transcode — Deduplicated per-platform MP4 generation
        # Input: ctx.processed_video_path (HUD+watermarked) or local_video_path
        # Output: ctx.platform_videos = {platform: Path}
        # Instagram + Facebook share one transcode pass if same spec.
        # ================================================================
        try:
            ctx = await _run_deduplicated_transcode(ctx)
            logger.info(
                f"[{upload_id}] Transcode complete: "
                f"{list(ctx.platform_videos.keys())}"
            )
        except SkipStage as e:
            logger.info(f"[{upload_id}] Transcode skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Transcode error: {e.message}")
            # Fallback: point all platforms at the source video
            source = ctx.processed_video_path or ctx.local_video_path
            for p in (ctx.platforms or []):
                ctx.platform_videos[p] = source

        await maybe_cancel(ctx, "transcode")

        # ================================================================
        # STAGE 6: Thumbnail
        # Runs AFTER transcode so the frame is from the final processed video.
        # Entitlement + user preference gated inside the stage.
        # ================================================================
        try:
            ctx = await run_thumbnail_stage(ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Thumbnail skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Thumbnail error: {e.message}")

        await maybe_cancel(ctx, "thumbnail")

        # ================================================================
        # STAGE 7: Caption — AI title / caption / hashtags
        # Runs AFTER thumbnail (GPT-4o vision needs the JPEG).
        # Trill/telemetry data already on ctx if available.
        # Stage is smart enough to skip if no entitlement or API key.
        # ================================================================
        try:
            ctx = await run_caption_stage(ctx)
            await db_stage.save_generated_metadata(db_pool, ctx)
        except SkipStage as e:
            logger.info(f"[{upload_id}] Caption skipped: {e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Caption error: {e.message}")

        await maybe_cancel(ctx, "caption")

        # ================================================================
        # STAGE 8: Upload — per-platform R2 keys
        # Uploaded files: processed/{user_id}/{upload_id}/{platform}.mp4
        # Also uploads a default.mp4 for backward compatibility / admin preview
        # ================================================================
        ctx.mark_stage("upload")
        processed_assets: Dict[str, str] = {}

        if ctx.platform_videos:
            # Deduplicate R2 uploads — if two platforms point at the same
            # local file path, upload ONCE and store the same R2 key for both.
            uploaded_paths: Dict[str, str] = {}  # local path str → r2 key

            for platform, video_path in ctx.platform_videos.items():
                if not video_path or not Path(video_path).exists():
                    logger.warning(f"[{upload_id}] No video file for {platform}")
                    continue

                path_str = str(video_path)

                if path_str in uploaded_paths:
                    # Reuse the already-uploaded R2 key (same file, same spec)
                    processed_assets[platform] = uploaded_paths[path_str]
                    logger.info(
                        f"[{upload_id}] {platform} → reuses R2 key from "
                        f"{uploaded_paths[path_str]} (shared transcode)"
                    )
                    continue

                r2_key = f"processed/{ctx.user_id}/{ctx.upload_id}/{platform}.mp4"
                try:
                    await r2_stage.upload_file(Path(video_path), r2_key, "video/mp4")
                    processed_assets[platform] = r2_key
                    uploaded_paths[path_str] = r2_key
                    sz_mb = Path(video_path).stat().st_size / 1024 / 1024
                    logger.info(f"[{upload_id}] Uploaded {platform}: {r2_key} ({sz_mb:.1f}MB)")
                except Exception as e:
                    logger.error(f"[{upload_id}] R2 upload failed for {platform}: {e}")

        # Default fallback video (admin preview / re-processing)
        fallback_video = ctx.processed_video_path or ctx.local_video_path
        if fallback_video and Path(fallback_video).exists():
            default_key = f"processed/{ctx.user_id}/{ctx.upload_id}/default.mp4"
            try:
                await r2_stage.upload_file(Path(fallback_video), default_key, "video/mp4")
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

        logger.info(
            f"[{upload_id}] Upload summary: "
            f"{', '.join(f'{p}={k}' for p, k in processed_assets.items())}"
        )
        await maybe_cancel(ctx, "upload")

        # ================================================================
        # STAGE 9: Publish
        # ================================================================
        try:
            ctx = await run_publish_stage(ctx, db_pool)
        except StageError as e:
            logger.error(f"[{upload_id}] Publish error: {e.message}")
            ctx.mark_error(e.code.value, e.message)

        await maybe_cancel(ctx, "publish")

        # ================================================================
        # STAGE 10: Notify
        # ================================================================
        try:
            ctx = await run_notify_stage(ctx)
        except Exception as e:
            logger.warning(f"[{upload_id}] Notify error: {e}")

        # ================================================================
        # Complete
        # ================================================================
        ctx.finished_at = datetime.now(timezone.utc)

        if ctx.is_partial_success():
            ctx.state = "partial"
        elif ctx.is_success():
            ctx.state = "succeeded"
        else:
            ctx.state = "failed"

        await db_stage.mark_processing_completed(db_pool, ctx)

        if ctx.is_success():
            await db_stage.increment_upload_count(db_pool, user_id)

        elapsed = (ctx.finished_at - ctx.started_at).total_seconds()
        logger.info(
            f"[{upload_id}] Pipeline {ctx.state} in {elapsed:.1f}s | "
            f"succeeded={ctx.get_success_platforms()} | "
            f"failed={ctx.get_failed_platforms()}"
        )
        return ctx.is_success()

    except CancelRequested:
        logger.info(f"[{upload_id}] Pipeline cancelled")
        return False
    except Exception as e:
        logger.exception(f"[{upload_id}] Pipeline failed: {e}")
        if ctx:
            ctx.mark_error("INTERNAL", str(e))
            await db_stage.mark_processing_failed(db_pool, ctx, "INTERNAL", str(e))
        await notify_admin_error(
            "pipeline_failure",
            {"upload_id": upload_id, "error": str(e)},
            db_pool,
        )
        return False
    finally:
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Concurrent job processor
# ---------------------------------------------------------------------------

async def _process_one_job(job_json: str) -> None:
    """
    Parse one job from Redis and run it inside the concurrency semaphore.
    Designed to be fire-and-forget via asyncio.create_task().
    """
    global _job_semaphore
    try:
        job_data = json.loads(job_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid job JSON: {e} — raw: {job_json[:200]}")
        return

    async with _job_semaphore:
        try:
            await run_pipeline(job_data)
        except Exception as e:
            logger.exception(f"Unhandled pipeline exception: {e}")


async def process_jobs() -> None:
    """
    Main job consumption loop.

    Pops jobs from Redis and dispatches them as concurrent asyncio tasks,
    bounded by WORKER_CONCURRENCY via a Semaphore. Priority queue is checked
    first (brpop respects list order).
    """
    global shutdown_requested, redis_client, _job_semaphore

    _job_semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)

    logger.info(
        f"Worker started | concurrency={WORKER_CONCURRENCY} | "
        f"queue={UPLOAD_JOB_QUEUE} | priority={PRIORITY_JOB_QUEUE}"
    )
    await notify_admin_worker_start(db_pool)

    consecutive_redis_errors = 0
    active_tasks: List[asyncio.Task] = []

    while not shutdown_requested:
        # Prune completed tasks from tracking list
        active_tasks = [t for t in active_tasks if not t.done()]

        try:
            job_raw = await redis_client.brpop(
                [PRIORITY_JOB_QUEUE, UPLOAD_JOB_QUEUE],
                timeout=int(POLL_INTERVAL),
            )
            consecutive_redis_errors = 0

            if not job_raw:
                continue

            _, job_json = job_raw

            # Dispatch as a new concurrent task — does not block the loop
            task = asyncio.create_task(_process_one_job(job_json))
            active_tasks.append(task)

            logger.debug(
                f"Job dispatched | active_tasks={len(active_tasks)} | "
                f"semaphore_value={_job_semaphore._value}"
            )

        except redis.ReadOnlyError:
            consecutive_redis_errors += 1
            wait_time = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(
                f"Redis read-only (upgrade in progress), "
                f"retrying in {wait_time:.0f}s"
            )
            await asyncio.sleep(wait_time)

        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            consecutive_redis_errors += 1
            wait_time = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(f"Redis connection error: {e}, retrying in {wait_time:.0f}s")
            await asyncio.sleep(wait_time)

            if consecutive_redis_errors >= REDIS_MAX_RETRIES:
                logger.error("Redis unreachable, attempting reconnect...")
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
            logger.exception(f"Job processing error: {e}")
            await asyncio.sleep(1)

    # Graceful shutdown — wait for all in-flight jobs to complete
    if active_tasks:
        logger.info(f"Shutdown: waiting for {len(active_tasks)} in-flight jobs...")
        await asyncio.gather(*active_tasks, return_exceptions=True)

    logger.info("Worker shutdown complete")
    await notify_admin_worker_stop(db_pool)


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

    # DB pool: min_size should be >= WORKER_CONCURRENCY to avoid connection starvation
    db_min = max(2, WORKER_CONCURRENCY)
    db_max = max(10, WORKER_CONCURRENCY * 3)
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=db_min, max_size=db_max)
    logger.info(f"Database connected | pool={db_min}-{db_max}")

    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info("Redis connected")

    shutdown_event = asyncio.Event()

    verify_task = asyncio.create_task(run_verification_loop(db_pool, shutdown_event))
    job_task = asyncio.create_task(process_jobs())

    try:
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
        if db_pool:
            await db_pool.close()
        if redis_client:
            try:
                await redis_client.aclose()
            except AttributeError:
                await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
