"""
UploadM8 Worker Service v3 — Deferred Processing + Concurrent Jobs

Pipeline order (critical — DO NOT REORDER):
  1.  Download        — Fetch original video + telemetry from R2
  2.  Telemetry       — Parse .map file, calculate Trill score, reverse-geocode + PADUS/gazetteer
  3.  Watermark       — Burn text watermark (before transcode!)
  4.  Transcode       — Deduplicated per-platform MP4s (+ ffprobe → ctx.video_info)
  5.  Audio context   — Whisper / ACR / YAMNet / GPT summary (feeds M8)
  6.  Vision          — Google Cloud Vision on sampled frames (labels/OCR/landmarks)
  7.  Video Intel     — GCP Video Intelligence (before Twelve Labs so TL skip gate works)
  8.  Twelve Labs     — Optional full-video scene understanding (skipped when VI is rich)
  9.  Dashcam OSD     — OCR of on-screen dashcam text (pref on); backfill telemetry when no .map
  10. Thumbnail       — Extract frame from FINAL processed video
  11. Caption         — M8 / LLM title+caption+hashtags (uses steps 2–9)
  12. Upload          — Per-platform R2 keys
  13. Publish         — Send to each platform API
  14. Verify          — Delivery verification loop (background)
  15. Notify          — Discord webhooks

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
  WORKER_LANE=full|process|publish — optional split: process lane skips publish consumer; publish lane runs publish_jobs + scheduler (staged jobs enqueue to Redis for process fleet).
  ASYNC_PUBLISH_QUEUE — when true, immediate jobs enqueue publish to Redis after encode (same or separate worker).
  Scheduler loop runs as a separate asyncio task, not a job slot.

WATERMARK (free tier — full-file FFmpeg re-encode before transcode):
  Admin settings: watermark_burn_text, watermark_size_scale (50–200%),
  watermark_opacity, watermark_position. Font size auto-scales to video resolution.
  Env overrides: WATERMARK_TEXT, WATERMARK_SIZE_SCALE, WATERMARK_OPACITY,
  WATERMARK_POSITION. WATERMARK_X264_PRESET (default: veryfast) and
  WATERMARK_X264_CRF (default: 23) tune CPU time; intermediate file is
  re-encoded again in transcode.

TRANSCODE quality (hybrid speed + dashcam detail on scaled workers):
  TRANSCODE_X264_PRESET (default: fast), TRANSCODE_X264_CRF (default: 19)
  Tier overrides: TRANSCODE_X264_CRF_1080=19, TRANSCODE_X264_CRF_4K=18, PRESET_STANDARD=faster
  TRANSCODE_DASHCAM_TUNE_FILM (default: true) — film tune on 1080p/4K sources
  TRANSCODE_PRESERVE_HD_TIER (default: true) — 1080p/4K targets + higher max bitrates for HD sources
  TRANSCODE_YOUTUBE_4K (default: auto), TRANSCODE_VERTICAL_4K_ALL_PLATFORMS (default: true)
  Video stream-copy when only audio/trim needs work (preserves original pixels)
"""

import os
import re
import sys
import json
import asyncio
import logging
import tempfile
import signal
import socket
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Awaitable, Any

import asyncpg
import redis.asyncio as redis

from core.helpers import coerce_jsonb_dict, coerce_jsonb_list, merge_platform_hashtag_overlay
from stages.errors import StageError, SkipStage, CancelRequested
from stages.context import (
    JobContext,
    build_hydration_story_text,
    build_video_story_timeline,
    create_context,
)
from stages.entitlements import get_entitlements_from_user, PRIORITY_QUEUE_CLASSES
from stages import db as db_stage
from stages import r2 as r2_stage
from stages import pipeline_checkpoint
from stages.telemetry_stage import run_telemetry_stage
from stages.transcode_stage import (
    run_transcode_stage, PLATFORM_SPECS,
    get_video_info, needs_transcode, build_ffmpeg_command,
    resolve_reframe_action,
)
from stages.audio_stage import run_audio_context_stage
from stages.vision_stage import run_vision_stage, VISION_STAGE_ENABLED
from stages.twelvelabs_stage import run_twelvelabs_stage
from stages.video_intelligence_stage import (
    run_video_intelligence_stage,
    VIDEO_INTELLIGENCE_STAGE_ENABLED,
)
from stages.dashcam_osd_stage import (
    backfill_telemetry_from_vision_osd,
    run_dashcam_osd_stage,
    DASHCAM_OSD_STAGE_ENABLED,
)
from stages.hydration_payload_stage import run_hydration_payload_stage
from stages.thumbnail_stage import run_thumbnail_stage
from stages.caption_stage import run_caption_stage
from stages.watermark_stage import run_watermark_stage
from stages.publish_stage import run_publish_stage
from stages.verify_stage import run_verification_loop
from stages.notify_stage import (
    run_notify_stage,
    notify_admin_worker_start,
    notify_admin_worker_stop,
    notify_admin_error,
    notify_upload_terminal,
)
from stages.emails import (
    send_low_token_warning_email,
)
from services.pipeline_ai_trace import emit_ai_pipeline_summary, record_ai_pipeline_trace
from stages.pipeline_manifest import init_pipeline_diag, diag_step, finalize_pipeline_diag
from stages import pipeline_stage_budgets as stage_budgets
from stages.pipeline_runner import persist_pipeline_manifest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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
# WORKER_CONCURRENCY  = FFmpeg-heavy process jobs (CPU-bound)
# PUBLISH_CONCURRENCY = API-light publish jobs   (I/O-bound)
# On Render background workers, default to 1 process slot so two HD/4K
# encodes cannot OOM a 2 GB instance. Override explicitly if on Pro/4GB+.
def _env_concurrency(name: str, *, render_default: int, local_default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return render_default if os.environ.get("RENDER") else local_default


WORKER_CONCURRENCY = _env_concurrency("WORKER_CONCURRENCY", render_default=1, local_default=2)
# Publish still pulls full platform MP4s from R2 — keep at 1 on 2 GB Render.
PUBLISH_CONCURRENCY = _env_concurrency("PUBLISH_CONCURRENCY", render_default=1, local_default=5)

# Scheduler: how far in advance to start processing a scheduled upload (minutes)
PROCESSING_WINDOW_MINUTES = int(os.environ.get("PROCESSING_WINDOW_MINUTES", "15"))

# How often the scheduler polls the DB for staged/ready jobs (seconds)
SCHEDULER_POLL_INTERVAL = int(os.environ.get("SCHEDULER_POLL_INTERVAL", "60"))

# ── Analytics auto-sync ──────────────────────────────────────────
# How often the analytics sync loop runs (seconds). Default: 6 hours.
ANALYTICS_SYNC_INTERVAL = int(os.environ.get("ANALYTICS_SYNC_INTERVAL_SECONDS", str(6 * 3600)))

# ── KPI collector ─────────────────────────────────────────────────
# How often to collect cost/revenue from Stripe, OpenAI, Mailgun, etc. (seconds). Default: 30 min.
KPI_COLLECTOR_INTERVAL = int(os.environ.get("KPI_COLLECTOR_INTERVAL_SECONDS", str(30 * 60)))
# How many uploads to sync per cycle (avoids platform API rate-limit hammering)
ANALYTICS_SYNC_BATCH    = int(os.environ.get("ANALYTICS_SYNC_BATCH_SIZE", "20"))
# Only re-sync uploads whose analytics_synced_at is older than this many hours
ANALYTICS_RESYNC_HOURS  = int(os.environ.get("ANALYTICS_RESYNC_HOURS", "6"))

# ── Per-stage timeouts (seconds) — see stages/pipeline_stage_budgets.py ──
STAGE_TIMEOUT_WATERMARK = int(stage_budgets.stage_timeout_watermark())
STAGE_TIMEOUT_TRANSCODE = int(stage_budgets.stage_timeout_transcode()) or 1800
STAGE_TIMEOUT_THUMBNAIL = int(stage_budgets.stage_timeout_thumbnail())
STAGE_TIMEOUT_PUBLISH = int(stage_budgets.stage_timeout_publish())
STAGE_TIMEOUT_AUDIO = int(stage_budgets.stage_timeout_audio())
STAGE_TIMEOUT_VISION = int(stage_budgets.stage_timeout_vision())
STAGE_TIMEOUT_VI = int(stage_budgets.stage_timeout_video_intelligence())
STAGE_TIMEOUT_TWELVELABS = int(stage_budgets.stage_timeout_twelvelabs())
STAGE_TIMEOUT_CAPTION = int(stage_budgets.stage_timeout_caption())

from core.pipeline_env_defaults import env_bool, env_int

WATERMARK_SINGLE_PASS = env_bool("WATERMARK_SINGLE_PASS", default=True)
TWELVE_LABS_PARALLEL = env_bool("TWELVE_LABS_PARALLEL", default=True)
WORKER_HEAVY_PIPELINE_SLOTS = max(1, env_int("WORKER_HEAVY_PIPELINE_SLOTS", default=1))
REDIS_JOB_LEGACY_DRAIN = env_bool("REDIS_JOB_LEGACY_DRAIN", default=True)

# ── Worker heartbeat ─────────────────────────────────────────────
# Writes a liveness row every N seconds so admin can see if the worker
# is actually doing work vs silently zombied. Self-creating schema.
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "10"))
WORKER_ID = (
    os.environ.get("WORKER_ID")
    or os.environ.get("RENDER_INSTANCE_ID")
    or socket.gethostname()
    or "worker"
)

# Worker deployment lane: full (default) | process | publish
# Use separate Render services: process workers run FFmpeg + consume process Redis
# queues; publish workers consume publish Redis queues (when ASYNC_PUBLISH_QUEUE).
WORKER_LANE = (os.environ.get("WORKER_LANE") or "full").strip().lower()

# When true, immediate uploads enqueue a publish-lane job after process pipeline
# instead of calling publish inline (requires publish consumer on this or another instance).
ASYNC_PUBLISH_QUEUE = (os.environ.get("ASYNC_PUBLISH_QUEUE", "false").lower() in ("1", "true", "yes", "on"))

# Stale job recovery (best-effort re-enqueue / visibility)
STALE_JOB_RECOVERY_ENABLED = (os.environ.get("STALE_JOB_RECOVERY_ENABLED", "true").lower() in ("1", "true", "yes", "on"))
STALE_JOB_RECOVERY_INTERVAL_SEC = max(60, int(os.environ.get("STALE_JOB_RECOVERY_INTERVAL_SEC", "300")))
STALE_QUEUED_MINUTES = max(15, int(os.environ.get("STALE_QUEUED_MINUTES", "45")))
STALE_PROCESSING_MINUTES = int(os.environ.get("STALE_PROCESSING_MINUTES", "20"))  # 0 = disable processing recovery
STALE_PROCESSING_RECOVER_CHECKPOINT = (
    os.environ.get("STALE_PROCESSING_RECOVER_CHECKPOINT", "true").lower() in ("1", "true", "yes", "on")
)

# Redis resilience
REDIS_RETRY_DELAY = 5.0
REDIS_MAX_RETRIES = 10

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None


def _google_multimodal_strict_mode() -> str:
    v = (os.environ.get("GOOGLE_MULTIMODAL_STRICT") or "").strip().lower()
    if not v or v in ("0", "false", "no", "off"):
        return "off"
    if v in ("halt", "fail", "block", "abort", "hard"):
        return "halt"
    return "degrade"


def _google_multimodal_strict_enabled() -> bool:
    return _google_multimodal_strict_mode() != "off"


def _strict_multimodal_gap_reason(reason: str) -> bool:
    r = (reason or "").lower()
    if "disabled via env" in r:
        return False
    return "credential" in r or "not configured" in r


def _merge_telemetry_ingest_artifact(ctx: JobContext, patch: Dict[str, Any]) -> None:
    """Merge diagnostics into ``output_artifacts.telemetry_ingest`` (JSON object)."""
    if not isinstance(getattr(ctx, "output_artifacts", None), dict):
        return
    cur: Dict[str, Any] = {}
    try:
        raw = ctx.output_artifacts.get("telemetry_ingest")
        if isinstance(raw, str) and raw.strip().startswith("{"):
            cur = json.loads(raw)
            if not isinstance(cur, dict):
                cur = {}
    except Exception:
        cur = {}
    for k, v in patch.items():
        if v is not None:
            cur[k] = v
    try:
        ctx.output_artifacts["telemetry_ingest"] = json.dumps(cur, default=str)[:8000]
    except Exception:
        pass


def _bootstrap_policy_flags_from_upload_artifacts(ctx: JobContext) -> None:
    """Rehydrate policy flags saved in ``output_artifacts`` for deferred publish-only runs."""

    arts = getattr(ctx, "output_artifacts", None) or {}
    if not isinstance(arts, dict):
        return
    viol = getattr(ctx, "google_multimodal_strict_violations", None)
    if not viol:
        raw_g = arts.get("google_multimodal_gaps")
        if isinstance(raw_g, str) and raw_g.strip():
            try:
                blob = json.loads(raw_g)
                gaps = blob.get("gaps") if isinstance(blob, dict) else None
                if isinstance(gaps, list) and gaps:
                    setattr(ctx, "google_multimodal_strict_violations", list(gaps))
            except (json.JSONDecodeError, TypeError):
                pass
    mq_v = getattr(ctx, "metadata_quality_violations", None)
    mq_ok_known = getattr(ctx, "metadata_quality_ok", None) is not None
    if mq_v is None or not mq_ok_known:
        raw_m = arts.get("metadata_quality_report")
        if isinstance(raw_m, str) and raw_m.strip():
            try:
                j = json.loads(raw_m)
                if isinstance(j, dict):
                    if isinstance(j.get("violations"), list):
                        setattr(ctx, "metadata_quality_violations", list(j["violations"]))
                    if "ok" in j:
                        setattr(ctx, "metadata_quality_ok", bool(j["ok"]))
            except (json.JSONDecodeError, TypeError):
                pass


def _publish_policy_block_reason(ctx: JobContext) -> str:
    """Human-readable blocker string, or ``\"\"`` when publishing is allowed."""

    _bootstrap_policy_flags_from_upload_artifacts(ctx)
    mode_g = _google_multimodal_strict_mode()
    if mode_g == "halt":
        v = getattr(ctx, "google_multimodal_strict_violations", None) or []
        if v:
            tail = "; ".join(str(x) for x in v[:24])
            return f"MULTIMODAL_STRICT_HALT: {tail}"[:4000]
    try:
        from services.metadata_quality import metadata_quality_strict_mode

        if metadata_quality_strict_mode() == "halt":
            if getattr(ctx, "metadata_quality_ok", True) is False:
                mv = getattr(ctx, "metadata_quality_violations", None) or []
                tail = "; ".join(str(x) for x in mv[:40])
                suf = tail if tail else "metadata_quality_fail"
                return f"METADATA_QUALITY_HALT: {suf}"[:4000]
    except Exception:
        pass
    return ""


# ── Wallet helpers (worker-side) ─────────────────────────────────────────────
async def _capture_tokens(upload_id: str, user_id: str, put_cost: int, aic_cost: int):
    """
    Confirm a hold: move reserved → actual spend.
    Called on successful job completion.
    """
    if not db_pool:
        return
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT u.platforms, u.compute_seconds, u.put_cost AS u_put, u.aic_cost AS u_aic,
                   u.billing_breakdown,
                   usr.subscription_tier
            FROM uploads u
            JOIN users usr ON usr.id = u.user_id
            WHERE u.id = $1 AND u.user_id = $2::uuid
            """,
            upload_id,
            user_id,
        )
        platforms = list(row["platforms"] or []) if row else []
        n_pf = len(platforms)
        tier = str(row["subscription_tier"] or "free") if row else "free"
        compute_seconds = float(row["compute_seconds"] or 0) if row else 0.0
        ai_services = []
        if aic_cost > 0:
            ai_services = ["captions", "hashtags", "thumbnails"]
        bd = row.get("billing_breakdown") if row else None
        if isinstance(bd, str):
            try:
                bd = json.loads(bd)
            except Exception:
                bd = None
        meta_obj = {
            "ai_services": ai_services,
            "num_platforms": n_pf,
            "subscription_tier": tier,
            "compute_seconds": compute_seconds,
            "put_cost_applied": put_cost,
            "aic_cost_applied": aic_cost,
            "billing_breakdown": bd,
        }
        # Debit balance and clear reservation simultaneously
        await conn.execute("""
            UPDATE wallets SET
                put_balance  = put_balance  - $1,
                aic_balance  = aic_balance  - $2,
                put_reserved = put_reserved - $1,
                aic_reserved = aic_reserved - $2,
                updated_at   = NOW()
            WHERE user_id = $3
              AND put_reserved >= $1
              AND aic_reserved >= $2
        """, put_cost, aic_cost, user_id)

        # Ledger entries
        if put_cost > 0:
            await conn.execute("""
                INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type, meta)
                VALUES ($1, 'put', $2, 'upload_debit', $3, 'upload', $4::jsonb)
            """, user_id, -put_cost, upload_id, json.dumps(meta_obj))
        if aic_cost > 0:
            await conn.execute("""
                INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type, meta)
                VALUES ($1, 'aic', $2, 'upload_debit', $3, 'upload', $4::jsonb)
            """, user_id, -aic_cost, upload_id, json.dumps(meta_obj))

        # Update wallet_holds record
        await conn.execute("""
            UPDATE wallet_holds SET status = 'captured', resolved_at = NOW()
            WHERE upload_id = $1 AND status = 'held'
        """, upload_id)

        # Mark upload hold_status
        await conn.execute("""
            UPDATE uploads SET hold_status = 'captured' WHERE id = $1
        """, upload_id)

        # Low token warning: if balance dropped to/below threshold, email user
        LOW_THRESHOLD = 5
        wallet = await conn.fetchrow(
            "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1", user_id
        )
        if wallet:
            prefs = await conn.fetchrow(
                "SELECT email_notifications FROM user_preferences WHERE user_id = $1",
                user_id,
            )
            user_row = await conn.fetchrow(
                "SELECT email, name FROM users WHERE id = $1", user_id
            )
            wants_email = True
            if prefs and prefs.get("email_notifications") is not None:
                wants_email = bool(prefs["email_notifications"])
            if wants_email and user_row and user_row.get("email"):
                put_bal = int(wallet.get("put_balance") or 0)
                aic_bal = int(wallet.get("aic_balance") or 0)
                if put_bal <= LOW_THRESHOLD and put_cost > 0:
                    await send_low_token_warning_email(
                        user_row["email"],
                        user_row.get("name") or "there",
                        "put",
                        put_bal,
                        LOW_THRESHOLD,
                    )
                elif aic_bal <= LOW_THRESHOLD and aic_cost > 0:
                    await send_low_token_warning_email(
                        user_row["email"],
                        user_row.get("name") or "there",
                        "aic",
                        aic_bal,
                        LOW_THRESHOLD,
                    )


async def _release_tokens(upload_id: str, user_id: str, put_cost: int, aic_cost: int, reason: str = "release"):
    """
    Release a hold without spending: restore reserved tokens back to available.
    Called on job failure or cancel.
    """
    if not db_pool:
        return
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE wallets SET
                put_reserved = GREATEST(0, put_reserved - $1),
                aic_reserved = GREATEST(0, aic_reserved - $2),
                updated_at   = NOW()
            WHERE user_id = $3
        """, put_cost, aic_cost, user_id)

        rel_meta = json.dumps({"hold_release": True, "detail": reason}, default=str)
        if put_cost > 0:
            await conn.execute("""
                INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type, meta)
                VALUES ($1, 'put', $2, $3, $4, 'upload', $5::jsonb)
            """, user_id, put_cost, reason, upload_id, rel_meta)
        if aic_cost > 0:
            await conn.execute("""
                INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type, meta)
                VALUES ($1, 'aic', $2, $3, $4, 'upload', $5::jsonb)
            """, user_id, aic_cost, reason, upload_id, rel_meta)

        await conn.execute("""
            UPDATE wallet_holds SET status = 'released', resolved_at = NOW()
            WHERE upload_id = $1 AND status = 'held'
        """, upload_id)

        await conn.execute("""
            UPDATE uploads SET hold_status = 'released' WHERE id = $1
        """, upload_id)


async def _get_upload_costs(upload_id: str):
    """Fetch the stored put_cost / aic_cost for an upload."""
    if not db_pool:
        return 0, 0
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT put_cost, aic_cost, put_reserved, aic_reserved, user_id FROM uploads WHERE id = $1",
            upload_id
        )
        if not row:
            return 0, 0
        # Use put_cost column if populated, else fall back to put_reserved
        put = row["put_cost"] or row["put_reserved"] or 0
        aic = row["aic_cost"] or row["aic_reserved"] or 0
        return int(put), int(aic)


async def partial_refund_tokens(
    conn,
    user_id: str,
    upload_id: str,
    succeeded_platforms: list,
    failed_platforms: list,
    original_put_cost: int,
):
    """
    Pro-rate a PUT refund for platforms that failed in a partial upload.

    PUT cost formula (mirrors entitlements.py compute_put_cost):
      base = 10 (covers the first platform)
      +2 per extra platform beyond the first

    For partial failure we refund:
      2 tokens × number_of_failed_platforms
      (base-10 is always kept because work was done)

    Note: In the worker flow we capture the full hold first (reserved → spent),
    so this refund credits *put_balance* back (no reserved adjustment).
    """
    n_failed = len(failed_platforms or [])
    if n_failed == 0 or not (succeeded_platforms or []):
        return

    put_refund = n_failed * 2
    put_refund = min(int(put_refund), max(0, int(original_put_cost) - 10))  # never refund the base-10

    if put_refund <= 0:
        return

    await conn.execute(
        """
        UPDATE wallets
        SET
            put_balance = put_balance + $1,
            updated_at  = NOW()
        WHERE user_id = $2
        """,
        put_refund,
        user_id,
    )

    # Ledger entry — positive delta = credit back
    pr_meta = json.dumps(
        {
            "failed_platforms": [str(x) for x in (failed_platforms or [])],
            "succeeded_platforms": [str(x) for x in (succeeded_platforms or [])],
            "original_put_cost": int(original_put_cost),
        },
        default=str,
    )
    await conn.execute(
        """
        INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type, meta)
        VALUES ($1, 'put', $2, 'partial_platform_refund', $3, 'upload', $4::jsonb)
        """,
        user_id,
        put_refund,
        upload_id,
        pr_meta,
    )


shutdown_requested = False
shutdown_event: Optional[asyncio.Event] = None

# ── Separate semaphores for each lane ────────────────────────────
# Process semaphore: guards FFmpeg transcode slots (CPU-bound)
# Publish semaphore: guards platform API call slots (I/O-bound)
# Keeping them separate means a 10-minute transcode CANNOT block
# a 10-second TikTok API publish call.
_process_semaphore: Optional[asyncio.Semaphore] = None
_heavy_semaphore: Optional[asyncio.Semaphore] = None
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
    "watermark":  28,
    "transcode":  48,
    "audio":      52,
    "vision":     55,
    "twelvelabs": 57,
    "video_intelligence": 58,
    "dashcam_osd": 60,
    "thumbnail":  65,
    "caption":    75,
    "upload":     87,
    "publish":    96,
    "notify":     99,
}


def _pipeline_failure_code_and_detail(exc: Exception) -> tuple[str, str]:
    from services.upload.r2_storage_guard import classify_r2_head_not_found

    classified = classify_r2_head_not_found(exc)
    if classified:
        return classified
    return "INTERNAL", str(exc)


async def maybe_cancel(ctx: JobContext, stage: str):
    progress = STAGE_PROGRESS.get(stage, 0)
    await db_stage.update_stage_progress(db_pool, ctx.upload_id, stage, progress)
    try:
        from services.worker_runtime_state import track_process_stage

        await track_process_stage(str(ctx.upload_id), stage)
    except Exception:
        pass
    if await check_cancelled(ctx):
        logger.info(f"[{ctx.upload_id}] Cancel at {stage}")
        await db_stage.mark_cancelled(db_pool, ctx.upload_id)
        raise CancelRequested(ctx.upload_id)


# ---------------------------------------------------------------------------
# Transcode Deduplication
# ---------------------------------------------------------------------------

def _platform_spec_fingerprint(platform: str, info=None) -> str:
    from stages.transcode_stage import get_platform_spec

    spec = get_platform_spec(platform, info)
    return (
        f"{spec.get('video_codec', 'h264')}"
        f"_{spec.get('target_width', 1080)}"
        f"x{spec.get('target_height', 1920)}"
        f"_{spec.get('max_fps', 30)}fps"
        f"_{spec.get('max_duration', 9999)}s"
        f"_{spec.get('sample_rate', 44100)}hz"
        f"_{spec.get('max_bitrate_video', '12M')}"
        f"_{spec.get('profile', 'high')}"
    )


def _group_platforms_by_spec(platforms: List[str], info=None) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in platforms:
        fp = _platform_spec_fingerprint(p, info)
        groups.setdefault(fp, []).append(p)
    return groups


async def _maybe_burn_tiktok_styled_cover(ctx: JobContext, upload_id: str) -> None:
    """Composite TikTok styled thumbnail into platform MP4 at cover frame time."""
    from pathlib import Path as _Path

    from stages.tiktok_cover_burn import (
        burn_tiktok_styled_cover,
        resolve_keyframe_hint,
        resolve_tiktok_cover_offset_sec,
        tiktok_burn_enabled,
        tiktok_cover_burn_mode,
    )

    platforms = [str(p).lower() for p in (ctx.platforms or [])]
    if "tiktok" not in platforms:
        return
    if not tiktok_burn_enabled(
        getattr(ctx, "user_settings", None) or {},
        getattr(ctx, "entitlements", None),
    ):
        return

    pm_json = (ctx.output_artifacts or {}).get("platform_thumbnail_map", "{}")
    try:
        platform_map = json.loads(pm_json) if isinstance(pm_json, str) else (pm_json or {})
    except Exception:
        platform_map = {}
    thumb_raw = platform_map.get("tiktok") if isinstance(platform_map, dict) else None
    if not thumb_raw:
        return

    render_method = str((ctx.output_artifacts or {}).get("thumbnail_render_method") or "").strip()
    if render_method not in ("studio_renderer", "template", "ai_edit"):
        logger.info(
            "[%s] TikTok cover burn skipped — no styled tiktok thumb (render_method=%r)",
            upload_id,
            render_method or None,
        )
        return

    thumb_path = _Path(str(thumb_raw))
    if not thumb_path.exists():
        return

    video_raw = (ctx.platform_videos or {}).get("tiktok")
    if not video_raw:
        return
    video_path = _Path(str(video_raw))
    if not video_path.exists():
        return

    offset = resolve_tiktok_cover_offset_sec(ctx.output_artifacts)
    out_path = ctx.temp_dir / f"tiktok_burn_{upload_id}.mp4"
    vi = getattr(ctx, "video_info", None) or {}
    keyframe_hint = resolve_keyframe_hint(ctx.output_artifacts)

    logger.info(
        "[%s] TikTok styled cover burn starting at %.2fs mode=%s tier=%s",
        upload_id,
        offset,
        tiktok_cover_burn_mode(),
        getattr(getattr(ctx, "entitlements", None), "tier", "?"),
    )
    result = await burn_tiktok_styled_cover(
        video_path,
        thumb_path,
        out_path,
        offset,
        work_dir=ctx.temp_dir / f"tiktok_burn_work_{upload_id}",
        video_fps=float(vi.get("fps") or 0) or None,
        video_duration=float(vi.get("duration") or 0) or None,
        keyframe_hint=keyframe_hint,
    )
    if not result.ok:
        if result.error:
            logger.warning("[%s] TikTok cover burn failed mode=%s: %s", upload_id, result.mode, result.error[:200])
        return

    ctx.platform_videos["tiktok"] = out_path
    arts = ctx.output_artifacts
    arts["tiktok_cover_burned"] = "true"
    arts["tiktok_cover_burn_offset_seconds"] = str(offset)
    arts["tiktok_cover_burn_mode"] = result.mode
    arts["tiktok_cover_burn_elapsed_sec"] = f"{result.elapsed_sec:.3f}"
    if result.window:
        arts["tiktok_cover_keyframe_window"] = json.dumps(result.window)
    logger.info(
        "[%s] TikTok styled cover burned at %.2fs mode=%s elapsed=%.2fs → %s window=%s",
        upload_id,
        offset,
        result.mode,
        result.elapsed_sec,
        out_path.name,
        result.window,
    )


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

    groups = _group_platforms_by_spec(platforms, info)
    reframe_mode = getattr(ctx, "reframe_mode", "auto") or "auto"
    ctx.platform_videos = {}

    async def _transcode_one_group(group_platforms: List[str]) -> Dict[str, Path]:
        canonical = group_platforms[0]
        reframe_action = resolve_reframe_action(info, reframe_mode, canonical)
        needs_tc, reasons = needs_transcode(info, canonical, reframe_action)
        out: Dict[str, Path] = {}

        if needs_tc:
            logger.info(
                f"[{ctx.upload_id}] Transcode [{canonical}] shared by {group_platforms}: {', '.join(reasons)}"
            )
            output_path = ctx.temp_dir / f"transcoded_{canonical}.mp4"
            try:
                tiktok_cover_off = None
                if canonical == "tiktok":
                    from stages.tiktok_cover_burn import resolve_tiktok_cover_offset_sec

                    tiktok_cover_off = resolve_tiktok_cover_offset_sec(ctx.output_artifacts)
                cmd = build_ffmpeg_command(
                    source_video,
                    output_path,
                    info,
                    canonical,
                    reframe_action,
                    tiktok_cover_offset_sec=tiktok_cover_off,
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                if proc.returncode != 0 or not output_path.exists():
                    logger.error(
                        f"[{ctx.upload_id}] FFmpeg failed [{canonical}]: {stderr.decode()[-300:]}"
                    )
                    for p in group_platforms:
                        out[p] = source_video
                    return out

                sz_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(
                    f"[{ctx.upload_id}] Transcode done [{canonical}]: {sz_mb:.1f}MB → {group_platforms}"
                )
                if canonical == "tiktok" and tiktok_cover_off is not None:
                    from stages.tiktok_cover_burn import store_tiktok_transcode_keyframe_artifacts

                    store_tiktok_transcode_keyframe_artifacts(
                        ctx.output_artifacts,
                        offset_sec=tiktok_cover_off,
                        fps=float(info.fps or 30.0),
                        duration_sec=float(info.duration or 0.0),
                    )
                for p in group_platforms:
                    out[p] = output_path
            except Exception as e:
                logger.error(f"[{ctx.upload_id}] Transcode error [{canonical}]: {e}")
                for p in group_platforms:
                    out[p] = source_video
        else:
            logger.info(f"[{ctx.upload_id}] Platforms {group_platforms} already compatible")
            for p in group_platforms:
                out[p] = source_video
        return out

    group_lists = list(groups.values())
    if len(group_lists) > 1:
        group_results = await asyncio.gather(
            *[_transcode_one_group(gp) for gp in group_lists],
            return_exceptions=True,
        )
        for gp, gr in zip(group_lists, group_results):
            if isinstance(gr, Exception):
                logger.error(f"[{ctx.upload_id}] Transcode group failed: {gr}")
                for p in gp:
                    ctx.platform_videos[p] = source_video
            else:
                ctx.platform_videos.update(gr)
    else:
        for group_platforms in group_lists:
            ctx.platform_videos.update(await _transcode_one_group(group_platforms))

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

def _settings_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
    return bool(value)


def _merge_job_preferences(ctx: JobContext, job_data: dict) -> None:
    """Overlay per-job upload preference snapshot without wiping richer DB prefs."""
    raw = (job_data or {}).get("preferences")
    if not raw:
        return
    prefs = coerce_jsonb_dict(raw, default={})
    if not prefs:
        return
    for key, val in prefs.items():
        if val is None:
            continue
        if key in ("platformHashtags", "platform_hashtags"):
            merged = merge_platform_hashtag_overlay(
                ctx.user_settings.get("platformHashtags") or ctx.user_settings.get("platform_hashtags"),
                val,
            )
            ctx.user_settings["platformHashtags"] = merged
            ctx.user_settings["platform_hashtags"] = merged
            continue
        if key in ("alwaysHashtags", "always_hashtags", "blockedHashtags", "blocked_hashtags"):
            val = coerce_jsonb_list(val)
        ctx.user_settings[key] = val


def _should_run_trill(ctx: JobContext) -> bool:
    has_file = (
        ctx.local_telemetry_path is not None
        and Path(ctx.local_telemetry_path).exists()
    )
    if not has_file:
        return False
    us = ctx.user_settings or {}
    telemetry_on = _settings_bool(us.get("telemetry_enabled"), True)
    trill_on = _settings_bool(us.get("trill_enabled", us.get("trillEnabled")), telemetry_on)
    return bool(telemetry_on and trill_on)


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
        logger.info(
            f"[{ctx.upload_id}] Trill score {trill.score} < threshold {min_score} — suppressing Trill hype tags only"
        )
        trill.title_modifier = ""
        trill.hashtags = []
        ctx.trill_score = trill
        ctx.trill = trill


def _log_multimodal_pipeline_survey(upload_id: str, ctx: JobContext) -> None:
    """Single INFO line after vision/audio/Twelve Labs/VI so operators can spot
    skipped credentials, tier gates, or prefs that leave captions thin.
    """
    vis = getattr(ctx, "vision_context", None) or {}
    n_vis = len(vis.get("label_names") or []) if isinstance(vis, dict) else 0
    ac = getattr(ctx, "audio_context", None) or {}
    whisper_chars = 0
    if isinstance(ac, dict):
        whisper_chars = int(ac.get("transcript_chars") or 0)
    whisper_chars = max(whisper_chars, len((getattr(ctx, "ai_transcript", None) or "").strip()))
    vit = getattr(ctx, "video_intelligence", None) or {}
    vic = getattr(ctx, "video_intelligence_context", None) or {}
    n_vi_obj = len(vit.get("object_tracks") or []) if isinstance(vit, dict) else 0
    if not n_vi_obj and isinstance(vic, dict):
        n_vi_obj = len(vic.get("object_tracks") or [])
    vu = getattr(ctx, "video_understanding", None) or {}
    tl_chars = len(str(vu.get("scene_description") or "")) if isinstance(vu, dict) else 0
    vi_err = bool(isinstance(vic, dict) and vic.get("error"))
    logger.info(
        "[%s] multimodal survey: vision_labels=%d whisper_chars=%d vi_objects=%d "
        "twelve_labs_scene_chars=%d vi_ctx_error=%s",
        upload_id,
        n_vis,
        whisper_chars,
        n_vi_obj,
        tl_chars,
        vi_err,
    )


def _ai_trace(ctx: Optional[JobContext], upload_id: str, stage: str, payload: Dict[str, object]) -> None:
    """Structured AI diagnostics: accumulates under ctx + optional per-line `[ai-trace]`."""
    record_ai_pipeline_trace(ctx, upload_id, stage, payload, log=logger)


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
    resume_stage: Optional[str] = None

    try:
        from services.worker_runtime_state import track_process_start

        await track_process_start(str(upload_id), stage="init")
        async def _persist_diag_artifacts_now(*keys: str) -> None:
            """Persist selected in-memory artifacts immediately (best-effort)."""
            if not ctx or not isinstance(getattr(ctx, "output_artifacts", None), dict):
                return
            patch: Dict[str, str] = {}
            for k in keys:
                v = ctx.output_artifacts.get(k)
                if v is None:
                    continue
                if isinstance(v, str):
                    if v.strip():
                        patch[k] = v
                    continue
                try:
                    patch[k] = json.dumps(v, default=str)
                except Exception:
                    continue
            if not patch:
                return
            try:
                await db_stage.merge_job_output_artifacts_strings(db_pool, str(upload_id), patch)
            except Exception as _e:
                logger.debug(f"[{upload_id}] immediate artifact persist skipped: {_e}")

        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            logger.error(f"[{upload_id}] Records not found")
            await notify_admin_error(
                "upload_records_missing",
                {"upload_id": upload_id, "user_id": user_id},
                db_pool,
            )
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        await db_stage.merge_pikzels_thumbnail_persona_id(db_pool, user_id, user_settings)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        from core.upload_baseline_defaults import apply_free_tier_processing_defaults

        if getattr(entitlements, "tier", "") == "free":
            apply_free_tier_processing_defaults(user_settings)

        from services.upload.r2_storage_guard import (
            ERROR_SOURCE_NOT_IN_R2,
            SOURCE_NOT_IN_R2_MESSAGE,
            upload_source_head_status,
        )

        r2_status = upload_source_head_status(upload_record)
        if r2_status == "unknown":
            logger.warning(
                "[%s] R2 head check inconclusive for %r — deferring job",
                upload_id,
                upload_record.get("r2_key"),
            )
            return
        if r2_status == "missing":
            logger.error(
                f"[{upload_id}] source object missing in R2 at {upload_record.get('r2_key')!r} — aborting pipeline"
            )
            ctx = create_context(job_data, upload_record, user_settings, entitlements)
            ctx.upload_id = upload_id
            ctx.mark_error(ERROR_SOURCE_NOT_IN_R2, SOURCE_NOT_IN_R2_MESSAGE)
            ctx.state = "failed"
            ctx.finished_at = _now_utc()
            ctx.mark_stage("download")
            await db_stage.mark_processing_failed(
                db_pool, ctx, ERROR_SOURCE_NOT_IN_R2, SOURCE_NOT_IN_R2_MESSAGE
            )
            # No Discord/ops page: missing source is a precondition, not a pipeline fault.
            # Retry API now blocks these before enqueue; this path is a race/legacy job only.
            return False

        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        _merge_job_preferences(ctx, job_data)
        init_pipeline_diag(ctx, upload_record, is_deferred=is_deferred)
        if WATERMARK_SINGLE_PASS:
            setattr(ctx, "watermark_single_pass", True)
        ctx.user_record = user_record
        ctx.started_at = _now_utc()
        ctx.state = "processing"
        # Stash db_pool on ctx so producer stages can self-persist diag artifacts
        # (hydration_payload / studio_render_report / thumbnail_trace / hydration_report
        # / pikzels_prompt_by_platform). Without this they only land in DB if the
        # orchestrator finishes save_generated_metadata, which silently drops them
        # whenever caption_stage exits early or an older worker is deployed.
        try:
            setattr(ctx, "_db_pool", db_pool)
        except Exception:
            pass
        try:
            if getattr(ctx, "vehicle_make_id", None) or getattr(ctx, "vehicle_model_id", None):
                from services.vehicle_catalog import fetch_vehicle_labels

                async with db_pool.acquire() as _c:
                    _lab = await fetch_vehicle_labels(_c, ctx.vehicle_make_id, ctx.vehicle_model_id)
                ctx.vehicle_make_name = _lab.get("make_name")
                ctx.vehicle_model_name = _lab.get("model_name")
        except Exception as _ve:
            logger.debug("[%s] vehicle label hydrate skipped: %s", upload_id, _ve)
        ctx.user_settings["_openai_model_override"] = (
            ctx.user_settings.get("trillOpenaiModel")
            or ctx.user_settings.get("trill_openai_model")
            or "gpt-4o"
        )

        # ── Server-side entitlement cap enforcement ───────────────────────
        # Re-resolve entitlements from DB (authoritative — not from job_data).
        # This prevents any client or stale job_data from bypassing tier limits.
        if ctx.user_record and ctx.entitlements:
            ent = ctx.entitlements

            # Cap thumbnails to tier max
            if hasattr(ctx, "num_thumbnails") and ctx.num_thumbnails > ent.max_thumbnails:
                logger.info(
                    f"[{ctx.upload_id}] Clamping thumbnails {ctx.num_thumbnails} → {ent.max_thumbnails} (tier cap)"
                )
                ctx.num_thumbnails = ent.max_thumbnails

            # Cap caption frames
            if hasattr(ctx, "caption_frames") and ctx.caption_frames > ent.max_caption_frames:
                ctx.caption_frames = ent.max_caption_frames

            # Free tier: burn-in settings from admin_settings (master-editable), DB each job.
            if ent.can_watermark:
                wm_settings = await db_stage.load_watermark_settings(db_pool)
                ctx.watermark_settings = wm_settings
                ctx.watermark_text = wm_settings["text"]

            # Enforce AI depth
            if not ent.can_ai and hasattr(ctx, "use_ai"):
                ctx.use_ai = False

        if not await db_stage.mark_processing_started(db_pool, ctx):
            logger.warning(
                f"[{upload_id}] processing claim lost (status not queued/staged) — skipping duplicate/stale job"
            )
            return False
        try:
            from services.upload_funnel import emit_upload_funnel_event

            emit_upload_funnel_event(str(upload_id), "worker_started", {})
        except Exception:
            pass
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

        resume_holder = None
        try:
            resume_stage, resume_holder = await pipeline_checkpoint.try_resume_from_checkpoint(
                db_pool, job_data, upload_record, ctx
            )
        except Exception as _cp_e:
            logger.warning(f"[{upload_id}] checkpoint resume probe failed (non-fatal): {_cp_e}")
            resume_stage, resume_holder = None, None

        if resume_stage:
            logger.info(f"[{upload_id}] Pipeline checkpoint resume | stage={resume_stage}")

        # ============================================================
        # STAGE 1: Download
        # ============================================================
        if resume_holder is None:
            ctx.mark_stage("download")
            temp_dir = tempfile.TemporaryDirectory()
            ctx.temp_dir = Path(temp_dir.name)

            video_local = ctx.temp_dir / ctx.filename

            async def _download_video() -> None:
                await r2_stage.download_file(ctx.source_r2_key, video_local)

            async def _download_telemetry() -> None:
                if not ctx.telemetry_r2_key:
                    return
                try:
                    telem_local = ctx.temp_dir / "telemetry.map"
                    await r2_stage.download_file(ctx.telemetry_r2_key, telem_local)
                    ctx.local_telemetry_path = telem_local
                    logger.info(f"[{upload_id}] Telemetry downloaded")
                except Exception as e:
                    logger.warning(f"[{upload_id}] Telemetry download failed: {e}")
                    ctx.local_telemetry_path = None
                    _merge_telemetry_ingest_artifact(
                        ctx,
                        {
                            "telemetry_r2_key_configured": True,
                            "telemetry_downloaded": False,
                            "telemetry_download_error": str(e)[:500],
                        },
                    )

            await asyncio.gather(_download_video(), _download_telemetry())
            ctx.local_video_path = video_local

            if ctx.telemetry_r2_key and ctx.local_telemetry_path:
                _merge_telemetry_ingest_artifact(
                    ctx,
                    {
                        "telemetry_r2_key_configured": True,
                        "telemetry_downloaded": True,
                    },
                )
            elif ctx.telemetry_r2_key and not getattr(ctx, "local_telemetry_path", None):
                _merge_telemetry_ingest_artifact(
                    ctx,
                    {"telemetry_r2_key_configured": True, "telemetry_downloaded": False},
                )
            elif not ctx.telemetry_r2_key:
                _merge_telemetry_ingest_artifact(
                    ctx,
                    {"telemetry_r2_key_configured": False, "telemetry_downloaded": False},
                )
        else:
            temp_dir = resume_holder
            _merge_telemetry_ingest_artifact(
                ctx,
                {
                    "checkpoint_resume_stage": str(resume_stage or ""),
                    "telemetry_r2_key_configured": bool(ctx.telemetry_r2_key),
                    "telemetry_downloaded": bool(getattr(ctx, "local_telemetry_path", None)),
                },
            )

        trill_requested = _should_run_trill(ctx)
        trill_active = bool(trill_requested and not resume_stage)
        logger.info(f"[{upload_id}] Flags: trill={trill_active} platforms={ctx.platforms}")
        await maybe_cancel(ctx, "download")

        # ============================================================
        # STAGE 2: Telemetry
        # ============================================================
        if trill_active:
            try:
                ctx = await run_telemetry_stage(ctx)
                _apply_trill_caption_settings(ctx)
                tel = ctx.telemetry or ctx.telemetry_data
                _ai_trace(ctx, upload_id, "telemetry", {
                    "status": "ok",
                    "points": len((getattr(tel, "points", None) or [])) if tel else 0,
                    "location": getattr(tel, "location_display", None) if tel else None,
                    "trill_bucket": getattr((ctx.trill or ctx.trill_score), "bucket", None),
                })
            except SkipStage as e:
                logger.info(f"[{upload_id}] Telemetry skipped: {e.reason}")
                _ai_trace(ctx, upload_id, "telemetry", {"status": "skipped", "reason": e.reason})
                trill_active = False
            except StageError as e:
                logger.warning(f"[{upload_id}] Telemetry error: {e.message}")
                _ai_trace(ctx, upload_id, "telemetry", {"status": "error", "reason": e.message})
                trill_active = False
        elif not resume_stage:
            ctx.telemetry = None
            ctx.telemetry_data = None
            ctx.trill = None
            ctx.trill_score = None

        try:
            _tel_sum = ctx.telemetry or ctx.telemetry_data
            _npts = len(getattr(_tel_sum, "points", None) or []) if _tel_sum else 0
            _merge_telemetry_ingest_artifact(
                ctx,
                {
                    "telemetry_points": _npts,
                    "telemetry_trill_requested": bool(trill_requested),
                    "telemetry_trill_active_after_stage": bool(trill_active),
                },
            )
            await _persist_diag_artifacts_now("telemetry_ingest")
        except Exception as _ti_e:
            logger.debug(f"[{upload_id}] telemetry_ingest artifact skipped: {_ti_e}")

        # ── Trill trace (single admin-trace line for ops visibility) ───────
        try:
            _tr_obj = ctx.trill or ctx.trill_score
            _tel_obj = ctx.telemetry or ctx.telemetry_data
            if _tr_obj is not None and (getattr(_tr_obj, "score", None) is not None):
                logger.info(
                    "[%s] Trill OK: score=%s bucket=%s max_speed=%s points=%s geo=%r padus=%r",
                    upload_id,
                    getattr(_tr_obj, "score", "?"),
                    getattr(_tr_obj, "bucket", "?"),
                    getattr(_tel_obj, "max_speed_mph", None) if _tel_obj else None,
                    len(getattr(_tel_obj, "points", None) or []) if _tel_obj else 0,
                    getattr(_tel_obj, "location_display", None) if _tel_obj else None,
                    getattr(_tel_obj, "padus_unit_name", None) if _tel_obj else None,
                )
            else:
                _reason = "telemetry_disabled" if not trill_active else (
                    "no_telemetry_points" if (not _tel_obj or not getattr(_tel_obj, "points", None))
                    else "trill_score_not_computed"
                )
                logger.warning(
                    "[%s] Trill MISSING: reason=%s telemetry_present=%s",
                    upload_id, _reason, bool(_tel_obj),
                )
        except Exception as _tre:
            logger.debug(f"[{upload_id}] trill trace log skipped: {_tre}")

        await maybe_cancel(ctx, "telemetry")

        if not resume_stage:
            try:
                await pipeline_checkpoint.save_post_telemetry_checkpoint(db_pool, ctx)
            except Exception as _cp1:
                logger.debug(f"[{upload_id}] post_telemetry checkpoint skipped: {_cp1}")

        skip_encode_stages = resume_stage in ("post_transcode", "post_audio", "post_caption")

        # ============================================================
        # STAGE 3: Watermark (skipped when WATERMARK_SINGLE_PASS — burn in transcode)
        # ============================================================
        if not skip_encode_stages:
            try:
                await db_stage.save_trill_metadata(db_pool, ctx)
            except Exception as e:
                logger.debug(f"[{upload_id}] Trill metadata persist skipped: {e}")

            if not WATERMARK_SINGLE_PASS:
                diag_step(ctx, stage="watermark", status="started", provider="ffmpeg")
                try:
                    ctx = await asyncio.wait_for(run_watermark_stage(ctx), timeout=STAGE_TIMEOUT_WATERMARK)
                    diag_step(ctx, stage="watermark", status="ok", provider="ffmpeg")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{upload_id}] Watermark timed out after {STAGE_TIMEOUT_WATERMARK}s — continuing without watermark"
                    )
                    diag_step(ctx, stage="watermark", status="failed", provider="ffmpeg", reason="timeout")
                except SkipStage as e:
                    logger.info(f"[{upload_id}] Watermark skipped: {e.reason}")
                    diag_step(ctx, stage="watermark", status="skipped", provider="ffmpeg", reason=e.reason)
                except StageError as e:
                    logger.warning(f"[{upload_id}] Watermark error: {e.message}")
                    diag_step(ctx, stage="watermark", status="failed", provider="ffmpeg", reason=e.message)
            else:
                diag_step(
                    ctx,
                    stage="watermark",
                    status="skipped",
                    provider="ffmpeg",
                    reason="WATERMARK_SINGLE_PASS — burn in transcode",
                )
            await maybe_cancel(ctx, "watermark")

            # ============================================================
            # STAGE 5: Transcode (deduplicated)
            # ============================================================
            try:
                ctx = await asyncio.wait_for(_run_deduplicated_transcode(ctx), timeout=STAGE_TIMEOUT_TRANSCODE)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{upload_id}] Transcode timed out after {STAGE_TIMEOUT_TRANSCODE}s — falling back to source video"
                )
                source = ctx.processed_video_path or ctx.local_video_path
                for p in (ctx.platforms or []):
                    ctx.platform_videos[p] = source
            except SkipStage as e:
                logger.info(f"[{upload_id}] Transcode skipped: {e.reason}")
            except StageError as e:
                logger.warning(f"[{upload_id}] Transcode error: {e.message}")
                source = ctx.processed_video_path or ctx.local_video_path
                for p in (ctx.platforms or []):
                    ctx.platform_videos[p] = source
            await maybe_cancel(ctx, "transcode")
        else:
            try:
                await db_stage.save_trill_metadata(db_pool, ctx)
            except Exception as e:
                logger.debug(f"[{upload_id}] Trill metadata persist skipped: {e}")
            await maybe_cancel(ctx, "watermark")
            await maybe_cancel(ctx, "transcode")

        if resume_stage in (None, "post_telemetry"):
            try:
                await pipeline_checkpoint.save_post_transcode_checkpoint(db_pool, ctx)
            except Exception as _cp2:
                logger.debug(f"[{upload_id}] post_transcode checkpoint skipped: {_cp2}")

        # ============================================================
        # STAGE 6–10: Multimodal context (audio, vision, TL, VI, dashcam OSD — feeds M8/captions)
        # ============================================================
        _multimodal_strict_gaps: List[str] = []
        _multimodal_parallel = (
            os.environ.get("MULTIMODAL_PARALLEL", "true").lower() not in ("0", "false", "no", "off")
        )

        async def _run_audio_multimodal() -> None:
            nonlocal ctx
            diag_step(ctx, stage="audio", status="started", provider="openai/whisper")
            if resume_stage == "post_audio":
                ac = ctx.audio_context or {}
                _ai_trace(ctx, upload_id, "audio", {
                    "status": "ok",
                    "resume": True,
                    "transcript_chars": len((ctx.ai_transcript or ac.get("transcript") or "").strip()),
                    "music_detected": bool(ac.get("music_detected")),
                    "yamnet_events": len(ac.get("yamnet_events") or []),
                })
                return
            try:
                ctx = await asyncio.wait_for(
                    run_audio_context_stage(ctx),
                    timeout=STAGE_TIMEOUT_AUDIO or None,
                )
                ac = ctx.audio_context or {}
                _ai_trace(ctx, upload_id, "audio", {
                    "status": "ok",
                    "transcript_chars": len((ctx.ai_transcript or ac.get("transcript") or "").strip()),
                    "music_detected": bool(ac.get("music_detected")),
                    "yamnet_events": len(ac.get("yamnet_events") or []),
                })
            except SkipStage as e:
                logger.info(f"[{upload_id}] Audio context skipped: {e.reason}")
                _ai_trace(ctx, upload_id, "audio", {"status": "skipped", "reason": e.reason})
            except StageError as e:
                logger.warning(f"[{upload_id}] Audio context error: {e.message}")
                _ai_trace(ctx, upload_id, "audio", {"status": "error", "reason": e.message})

        async def _run_vision_multimodal() -> None:
            nonlocal ctx
            diag_step(ctx, stage="vision", status="started", provider="google_vision")
            try:
                ctx = await asyncio.wait_for(
                    run_vision_stage(ctx),
                    timeout=STAGE_TIMEOUT_VISION or None,
                )
                vis = ctx.vision_context or {}
                _ai_trace(ctx, upload_id, "vision", {
                    "status": "ok",
                    "labels": len(vis.get("label_names") or []) if isinstance(vis, dict) else 0,
                    "ocr_chars": len((vis.get("ocr_text") or "").strip()) if isinstance(vis, dict) else 0,
                    "landmarks": len(vis.get("landmark_names") or []) if isinstance(vis, dict) else 0,
                })
            except SkipStage as e:
                logger.info(f"[{upload_id}] Vision skipped: {e.reason}")
                _ai_trace(ctx, upload_id, "vision", {"status": "skipped", "reason": e.reason})
                if (
                    _google_multimodal_strict_enabled()
                    and VISION_STAGE_ENABLED
                    and _strict_multimodal_gap_reason(str(e.reason or ""))
                ):
                    _multimodal_strict_gaps.append(f"vision:{e.reason}")
            except StageError as e:
                logger.warning(f"[{upload_id}] Vision error: {e.message}")
                _ai_trace(ctx, upload_id, "vision", {"status": "error", "reason": e.message})

        async def _run_vi_multimodal() -> None:
            nonlocal ctx
            _ensure_worker_semaphores()
            try:
                async with _heavy_semaphore:
                    ctx = await asyncio.wait_for(
                        run_video_intelligence_stage(ctx),
                        timeout=STAGE_TIMEOUT_VI or None,
                    )
                vi = getattr(ctx, "video_intelligence", None) or {}
                vic = getattr(ctx, "video_intelligence_context", None) or {}
                verr = ""
                if isinstance(vic, dict):
                    verr = str(vic.get("error") or "").strip()
                    if not verr:
                        verr = str(vic.get("parse_error") or "").strip()
                if not verr and isinstance(vi, dict):
                    verr = str(vi.get("error") or "").strip()
                vi_trace_status = "failed" if verr else "ok"
                _ai_trace(ctx, upload_id, "video_intelligence", {
                    "status": vi_trace_status,
                    "objects": len(vi.get("object_tracks") or []) if isinstance(vi, dict) else 0,
                    "logos": len(vi.get("logos") or []) if isinstance(vi, dict) else 0,
                    "text": len(vi.get("on_screen_text") or []) if isinstance(vi, dict) else 0,
                    "error": verr[:500] if verr else None,
                })
                if isinstance(getattr(ctx, "output_artifacts", None), dict):
                    try:
                        ctx.output_artifacts["video_intelligence_status"] = json.dumps(
                            {
                                "status": vi_trace_status,
                                "error": verr[:2000] if verr else None,
                            },
                            default=str,
                        )[:4000]
                    except Exception:
                        pass
                await _persist_diag_artifacts_now("video_intelligence_status")
            except SkipStage as e:
                logger.info(f"[{upload_id}] Video intelligence skipped: {e.reason}")
                _ai_trace(ctx, upload_id, "video_intelligence", {"status": "skipped", "reason": e.reason})
                if (
                    _google_multimodal_strict_enabled()
                    and VIDEO_INTELLIGENCE_STAGE_ENABLED
                    and _strict_multimodal_gap_reason(str(e.reason or ""))
                ):
                    _multimodal_strict_gaps.append(f"video_intelligence:{e.reason}")
            except StageError as e:
                logger.warning(f"[{upload_id}] Video intelligence error: {e.message}")
                _ai_trace(ctx, upload_id, "video_intelligence", {"status": "error", "reason": e.message})

            if _google_multimodal_strict_enabled() and VIDEO_INTELLIGENCE_STAGE_ENABLED:
                vic = getattr(ctx, "video_intelligence_context", None) or {}
                if isinstance(vic, dict):
                    verr2 = str(vic.get("error") or "").strip()
                    if not verr2:
                        verr2 = str(vic.get("parse_error") or "").strip()
                    if verr2:
                        vl = verr2.lower()
                        gap_on_any_error = (
                            (os.environ.get("VIDEO_INTELLIGENCE_GAP_ON_ANY_ERROR") or "")
                            .strip()
                            .lower()
                            in ("1", "true", "yes", "on")
                        )
                        hard_gap = (
                            gap_on_any_error
                            or "403" in vl
                            or "401" in vl
                            or "permission denied" in vl
                            or ("permission" in vl and "denied" in vl)
                            or "disabled" in vl
                            or "not been used" in vl
                            or "has not been enabled" in vl
                            or "not enabled" in vl
                            or "billing" in vl
                            or "quota" in vl
                            or "invalid credential" in vl
                            or "unauthenticated" in vl
                            or ("api" in vl and "not" in vl and "enable" in vl)
                        )
                        if hard_gap:
                            _multimodal_strict_gaps.append(f"video_intelligence:{verr2[:400]}")

            try:
                from services.recognition_engine import persist_recognition

                vi_payload = getattr(ctx, "video_intelligence", None) or {}
                if vi_payload and not vi_payload.get("error"):
                    summary = await persist_recognition(
                        db_pool,
                        upload_id=str(ctx.upload_id),
                        user_id=str(ctx.user_id),
                        vi_payload=vi_payload,
                    )
                    if summary:
                        setattr(ctx, "recognition_summary", summary)
            except Exception as rec_e:
                logger.warning(f"[{upload_id}] recognition persist non-fatal: {rec_e}")

        async def _run_twelvelabs_multimodal() -> None:
            nonlocal ctx
            diag_step(ctx, stage="twelvelabs", status="started", provider="twelve_labs")
            try:
                ctx = await asyncio.wait_for(
                    run_twelvelabs_stage(ctx),
                    timeout=STAGE_TIMEOUT_TWELVELABS or None,
                )
                vu = ctx.video_understanding or {}
                diag_step(ctx, stage="twelvelabs", status="ok", provider="twelve_labs")
                _ai_trace(ctx, upload_id, "twelvelabs", {
                    "status": "ok",
                    "scene_chars": len((vu.get("scene_description") or "").strip()) if isinstance(vu, dict) else 0,
                    "title_suggestion": (vu.get("title_suggestion") or "") if isinstance(vu, dict) else "",
                })
            except asyncio.TimeoutError:
                logger.warning(f"[{upload_id}] Twelve Labs timed out after {STAGE_TIMEOUT_TWELVELABS}s")
                diag_step(ctx, stage="twelvelabs", status="failed", provider="twelve_labs", reason="timeout")
                _ai_trace(ctx, upload_id, "twelvelabs", {"status": "error", "reason": "timeout"})
            except SkipStage as e:
                logger.info(f"[{upload_id}] Twelve Labs skipped: {e.reason}")
                diag_step(ctx, stage="twelvelabs", status="skipped", provider="twelve_labs", reason=e.reason)
                _ai_trace(ctx, upload_id, "twelvelabs", {"status": "skipped", "reason": e.reason})
            except StageError as e:
                logger.warning(f"[{upload_id}] Twelve Labs error: {e.message}")
                diag_step(ctx, stage="twelvelabs", status="failed", provider="twelve_labs", reason=e.message)
                _ai_trace(ctx, upload_id, "twelvelabs", {"status": "error", "reason": e.message})

        _mm_tasks = [_run_audio_multimodal(), _run_vision_multimodal(), _run_vi_multimodal()]
        if _multimodal_parallel and TWELVE_LABS_PARALLEL and resume_stage != "post_audio":
            _mm_tasks.append(_run_twelvelabs_multimodal())
        if _multimodal_parallel:
            await asyncio.gather(*_mm_tasks)
        else:
            for _t in _mm_tasks:
                await _t

        await maybe_cancel(ctx, "audio")
        await maybe_cancel(ctx, "vision")

        if resume_stage in (None, "post_telemetry", "post_transcode"):
            try:
                await pipeline_checkpoint.save_post_audio_checkpoint(db_pool, ctx)
            except Exception as _cp3:
                logger.debug(f"[{upload_id}] post_audio checkpoint skipped: {_cp3}")

        await maybe_cancel(ctx, "video_intelligence")

        if not (_multimodal_parallel and TWELVE_LABS_PARALLEL and resume_stage != "post_audio"):
            await _run_twelvelabs_multimodal()
        await maybe_cancel(ctx, "twelvelabs")

        # Depth router: after Vision/VI/TL race, force a TL retry when Vision is
        # generic and scene understanding never landed (accuracy over cost).
        try:
            from services.multimodal_depth_router import apply_depth_route_to_ctx

            _depth = apply_depth_route_to_ctx(ctx)
            _ai_trace(ctx, upload_id, "multimodal_depth", {
                "clip_kind": _depth.get("clip_kind"),
                "force_twelvelabs": _depth.get("force_twelvelabs"),
                "vision_weak": _depth.get("vision_weak"),
                "reason": _depth.get("reason"),
            })
            vu = getattr(ctx, "video_understanding", None) or {}
            has_scene = isinstance(vu, dict) and bool(
                (vu.get("scene_description") or vu.get("description") or "").strip()
            )
            if _depth.get("force_twelvelabs") and not has_scene:
                logger.info(
                    "[%s] depth router forcing Twelve Labs retry (%s)",
                    upload_id,
                    _depth.get("reason"),
                )
                await _run_twelvelabs_multimodal()
                await maybe_cancel(ctx, "twelvelabs_depth_retry")
        except Exception as _depth_e:
            logger.debug(f"[{upload_id}] multimodal depth router skipped: {_depth_e}")

        try:
            await db_stage.save_pipeline_manifest(
                db_pool,
                str(upload_id),
                finalize_pipeline_diag(ctx, terminal_status="processing"),
            )
        except Exception:
            pass

        _log_multimodal_pipeline_survey(upload_id, ctx)

        try:
            ctx = await run_dashcam_osd_stage(ctx)
            osd = ctx.dashcam_osd_context or {}
            _ai_trace(ctx, upload_id, "dashcam_osd", {
                "status": "ok",
                "frames_sampled": osd.get("frames_sampled") if isinstance(osd, dict) else None,
                "gps_fix_count": len(osd.get("gps_path") or []) if isinstance(osd, dict) else 0,
                "telemetry_backfilled": bool(osd.get("telemetry_backfilled")) if isinstance(osd, dict) else False,
            })
        except SkipStage as e:
            logger.info(f"[{upload_id}] Dashcam OSD skipped: {e.reason}")
            _ai_trace(ctx, upload_id, "dashcam_osd", {"status": "skipped", "reason": e.reason})
            if (
                _google_multimodal_strict_enabled()
                and DASHCAM_OSD_STAGE_ENABLED
                and _strict_multimodal_gap_reason(str(e.reason or ""))
            ):
                _multimodal_strict_gaps.append(f"dashcam_osd:{e.reason}")
        except StageError as e:
            logger.warning(f"[{upload_id}] Dashcam OSD error: {e.message}")
            _ai_trace(ctx, upload_id, "dashcam_osd", {"status": "error", "reason": e.message})
        try:
            await backfill_telemetry_from_vision_osd(ctx)
        except Exception as e:
            logger.warning(f"[{upload_id}] Vision OSD fallback skipped: {e}")

        # Place evidence: landmarks / OCR / transcript → Nominatim when no .map
        try:
            from services.place_evidence import backfill_place_from_vision

            _pe = await backfill_place_from_vision(ctx)
            _ai_trace(ctx, upload_id, "place_evidence", {
                "sources": (_pe or {}).get("sources"),
                "places": len((_pe or {}).get("places") or []),
                "landmarks": len((_pe or {}).get("landmarks") or []),
                "geocode": bool(((_pe or {}).get("geocode_from_landmark") or {}).get("ok")),
                "teams": len((_pe or {}).get("sports_teams") or []),
                "plates": len((_pe or {}).get("license_plates") or []),
            })
        except Exception as _pe_e:
            logger.debug(f"[{upload_id}] place_evidence skipped: {_pe_e}")

        if _multimodal_strict_gaps and isinstance(getattr(ctx, "output_artifacts", None), dict):
            ctx.output_artifacts["google_multimodal_gaps"] = json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "gaps": list(_multimodal_strict_gaps),
                },
                default=str,
            )[:16000]
        if _google_multimodal_strict_enabled() and _multimodal_strict_gaps:
            setattr(ctx, "google_multimodal_strict_violations", list(_multimodal_strict_gaps))
            logger.warning(
                "[%s] GOOGLE_MULTIMODAL_STRICT gaps recorded: %s",
                upload_id,
                _multimodal_strict_gaps,
            )

        try:
            from services.google_visual_recognition import attach_visual_recognition

            attach_visual_recognition(ctx)
            try:
                from services.visual_entity_memory import upsert_catalog_entities

                vr_flat = (getattr(ctx, "visual_recognition", None) or {}).get("flat") or {}
                niche = (getattr(ctx, "visual_recognition", None) or {}).get("niche") or "general"
                if vr_flat:
                    await upsert_catalog_entities(
                        db_pool,
                        user_id=str(ctx.user_id),
                        upload_id=str(ctx.upload_id),
                        catalog_flat=vr_flat,
                        category=str(niche),
                    )
                    try:
                        from services.ml_entity_hub_sync import sync_visual_entities_for_user

                        hf_sync = await sync_visual_entities_for_user(
                            db_pool,
                            user_id=str(ctx.user_id),
                            category=str(niche),
                        )
                        if hf_sync.get("ok") and hf_sync.get("rows"):
                            logger.info(
                                "[%s] HF visual entities synced rows=%s",
                                upload_id,
                                hf_sync.get("rows"),
                            )
                    except Exception as _hf_sync_e:
                        logger.debug(f"[{upload_id}] HF entity sync skipped: {_hf_sync_e}")
            except Exception as _mem_e:
                logger.debug(f"[{upload_id}] visual entity catalog upsert skipped: {_mem_e}")
        except Exception as _gvr_e:
            logger.debug(f"[{upload_id}] visual recognition finalize skipped: {_gvr_e}")
        try:
            from services.trill_scenic_boost import apply_scenic_trill_boost

            apply_scenic_trill_boost(ctx)
        except Exception as _tsb_e:
            logger.debug(f"[{upload_id}] Trill scenic boost skipped: {_tsb_e}")
        try:
            await db_stage.save_trill_metadata(db_pool, ctx)
        except Exception as e:
            logger.debug(f"[{upload_id}] OSD Trill metadata persist skipped: {e}")
        await maybe_cancel(ctx, "dashcam_osd")

        # ── Scene-story + timeline fusion ───────────────────────────────────
        # Build the ordered VI+Vision+OSD+telemetry+audio timeline AFTER the
        # dashcam OSD stage so OSD backfill is incorporated. Persist into
        # output_artifacts so M8 (next), the API, the queue UI, the upload
        # email, and the user/admin Discord embeds can all surface it.
        try:
            _scene_story = build_hydration_story_text(ctx, max_chars=900) or ""
            _timeline_events = build_video_story_timeline(ctx, max_events=80) or []
            # First-class shot list for M8 prompt spine (temporal story, not bag-of-labels).
            _shot_list = []
            for ev in _timeline_events:
                if not isinstance(ev, dict):
                    continue
                kind = str(ev.get("kind") or "").lower()
                if kind in ("shot", "segment", "scene", "vi_label", "vision", "osd", "speech", "music"):
                    _shot_list.append({
                        "t_seconds": ev.get("t_seconds"),
                        "kind": kind,
                        "text": str(ev.get("text") or "")[:240],
                    })
                if len(_shot_list) >= 40:
                    break
            if isinstance(ctx.output_artifacts, dict):
                ctx.output_artifacts["scene_story"] = _scene_story[:1600]
                ctx.output_artifacts["timeline_story"] = json.dumps(_timeline_events)[:48_000]
                ctx.output_artifacts["shot_list_v1"] = _shot_list
            await _persist_diag_artifacts_now("timeline_story", "scene_story", "shot_list_v1")
            logger.info(
                f"[{upload_id}] scene_story built ({len(_scene_story)} chars), "
                f"timeline events={len(_timeline_events)}, shot_list={len(_shot_list)}"
            )
        except Exception as _scts_e:
            logger.debug(f"[{upload_id}] scene_story/timeline build skipped: {_scts_e}")

        # Canonical hydration snapshot (thumb + caption + `output_artifacts`).
        try:
            ctx = await run_hydration_payload_stage(ctx)
            await _persist_diag_artifacts_now("hydration_payload")
        except Exception as hp_e:
            logger.warning(f"[{upload_id}] hydration_payload_stage failed (non-fatal): {hp_e}")

        # ============================================================
        # STAGE 11: Thumbnail — extract frame then immediately upload to R2
        # ============================================================
        try:
            ctx = await asyncio.wait_for(run_thumbnail_stage(ctx), timeout=STAGE_TIMEOUT_THUMBNAIL)
            _ai_trace(ctx, upload_id, "thumbnail", {
                "status": "ok",
                "thumbnail_path": str(ctx.thumbnail_path or ""),
                "candidate_count": len(ctx.thumbnail_paths or []),
            })
        except asyncio.TimeoutError:
            logger.warning(f"[{upload_id}] Thumbnail timed out after {STAGE_TIMEOUT_THUMBNAIL}s")
            _ai_trace(ctx, upload_id, "thumbnail", {"status": "error", "reason": "timeout"})
        except SkipStage as e:
            logger.info(f"[{upload_id}] Thumbnail skipped: {e.reason}")
            _ai_trace(ctx, upload_id, "thumbnail", {"status": "skipped", "reason": e.reason})
        except StageError as e:
            logger.warning(f"[{upload_id}] Thumbnail error: {e.message}")
            _ai_trace(ctx, upload_id, "thumbnail", {"status": "error", "reason": e.message})

        from services.thumbnail_ops import record_pikzels_render_failures_incident

        await record_pikzels_render_failures_incident(
            db_pool,
            upload_id=str(upload_id),
            user_id=str(user_id) if user_id else None,
            output_artifacts=ctx.output_artifacts,
        )

        if not ctx.thumbnail_path or not Path(ctx.thumbnail_path).exists():
            logger.warning(
                f"[{upload_id}] No thumbnail produced — "
                "caption frames and publish previews will lack visual context. "
                "Check thumbnail_stage for ffmpeg/entitlement errors."
            )

        # ── Upload the best thumbnail to R2 and record its key ─────────────
        # thumbnail_stage sets ctx.thumbnail_path but never uploads it.
        # We do it here so thumbnail_r2_key is persisted to the DB and the
        # API can return a presigned URL for the dashboard to display.
        if ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
            from stages.image_format import ensure_jpeg_file as _ensure_jpeg_file

            thumb_r2_key = f"thumbnails/{ctx.user_id}/{upload_id}/thumbnail.jpg"
            try:
                thumb_local = _ensure_jpeg_file(Path(ctx.thumbnail_path))
                await r2_stage.upload_file(
                    thumb_local,
                    thumb_r2_key,
                    "image/jpeg",
                )
                ctx.thumbnail_r2_key = thumb_r2_key
                # Persist immediately so the API can serve it even before the
                # pipeline finishes (user sees the thumbnail as soon as it's ready).
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE uploads SET thumbnail_r2_key = $1, updated_at = NOW() WHERE id = $2",
                        thumb_r2_key,
                        upload_id,
                    )
                logger.info(f"[{upload_id}] Thumbnail uploaded to R2: {thumb_r2_key}")
            except Exception as e:
                logger.warning(f"[{upload_id}] Thumbnail R2 upload failed (non-fatal): {e}")

        # ── Upload platform-specific styled thumbnails to R2 (for publish + deferred + UI) ──
        # platform_thumbnail_map has local paths per platform. Upload each so publish_stage
        # can use supported custom covers and queue/detail UI can display previews for all
        # selected platforms, including TikTok even when publish falls back to thumb_offset.
        platform_thumb_r2: Dict[str, str] = {}
        pm_json = ctx.output_artifacts.get("platform_thumbnail_map", "{}")
        try:
            platform_map = json.loads(pm_json) if isinstance(pm_json, str) else (pm_json or {})
        except Exception:
            platform_map = {}
        for platform, local_path in platform_map.items():
            if not local_path or not Path(local_path).exists():
                continue
            r2_key = f"thumbnails/{ctx.user_id}/{upload_id}/{platform}.jpg"
            try:
                thumb_local = _ensure_jpeg_file(Path(local_path))
                await r2_stage.upload_file(thumb_local, r2_key, "image/jpeg")
                platform_thumb_r2[platform] = r2_key
                logger.debug(f"[{upload_id}] Platform thumb {platform} → {r2_key}")
            except Exception as e:
                logger.debug(f"[{upload_id}] Platform thumb {platform} upload failed: {e}")
        if platform_thumb_r2:
            ctx.output_artifacts["platform_thumbnail_r2_keys"] = json.dumps(platform_thumb_r2)

        # TikTok: burn styled cover into video pixels (API only supports frame timestamp).
        try:
            await _maybe_burn_tiktok_styled_cover(ctx, upload_id)
        except Exception as _ttb_e:
            logger.warning(f"[{upload_id}] TikTok cover burn skipped: {_ttb_e}")

        # Also upload any additional candidate thumbnails (for tier users who can pick)
        if ctx.thumbnail_paths and len(ctx.thumbnail_paths) > 1:
            candidate_keys = []
            for i, cpath in enumerate(ctx.thumbnail_paths):
                if not cpath or not Path(cpath).exists():
                    continue
                if str(cpath) == str(ctx.thumbnail_path):
                    candidate_keys.append(ctx.thumbnail_r2_key or "")
                    continue
                ckey = f"thumbnails/{ctx.user_id}/{upload_id}/candidate_{i:02d}.jpg"
                try:
                    await r2_stage.upload_file(Path(cpath), ckey, "image/jpeg")
                    candidate_keys.append(ckey)
                    logger.debug(f"[{upload_id}] Candidate thumb {i} → {ckey}")
                except Exception as e:
                    logger.debug(f"[{upload_id}] Candidate thumb {i} upload failed: {e}")
            if candidate_keys:
                ctx.output_artifacts["thumbnail_r2_candidates"] = json.dumps(candidate_keys)

        # Persist thumbnail diagnostics immediately so admin trace and rescue checks
        # can see persona/Pikzels evidence even if later stages are skipped/fail.
        await _persist_diag_artifacts_now(
            "thumbnail_trace",
            "pikzels_prompt_by_platform",
            "platform_thumbnail_map",
            "platform_thumbnail_r2_keys",
            "thumbnail_render_method",
            "thumbnail_selection_method",
            "thumbnail_brief_json",
            "studio_render_report",
            "thumbnail_r2_candidates",
        )

        # ML / ops: log styled thumbnail renders from the upload pipeline (Pikzels v2 or GPT image edit)
        # so training data is not limited to the standalone Thumbnail Studio tool.
        _thumb_render = str(ctx.output_artifacts.get("thumbnail_render_method") or "").strip()
        if _thumb_render in ("studio_renderer", "ai_edit") and getattr(ctx, "user_id", None):
            try:
                from services.growth_intelligence import m8_engine_identity_payload, record_studio_usage_event
                from services.ml_marketing import record_thumbnail_studio_engine_ml_batch

                _engine_mode = (
                    "uploadm8_pikzels_v2_pipeline"
                    if _thumb_render == "studio_renderer"
                    else "uploadm8_gpt_image_edit_pipeline"
                )
                _report = {}
                try:
                    import json as _json_rep

                    _raw_rep = ctx.output_artifacts.get("studio_render_report")
                    if isinstance(_raw_rep, dict):
                        _report = _raw_rep
                    elif isinstance(_raw_rep, str) and _raw_rep.strip():
                        _parsed = _json_rep.loads(_raw_rep)
                        if isinstance(_parsed, dict):
                            _report = _parsed
                except Exception:
                    _report = {}
                _plat = _report.get("platform_render_methods") if isinstance(_report.get("platform_render_methods"), dict) else {}
                _first_plat = next(iter(_plat.values()), {}) if _plat else {}
                if not isinstance(_first_plat, dict):
                    _first_plat = {}
                _variants = [
                    {
                        "index": 1,
                        "ctr_score": _first_plat.get("ctr_score") or _report.get("ctr_score"),
                        "engine_status": "ok" if not _report.get("skip_reason") else "skipped",
                        "pikzels_main_score": _first_plat.get("pikzels_main_score") or _report.get("pikzels_main_score"),
                        "pikzels_recreate_http_status": _first_plat.get("http_status")
                        or _report.get("pikzels_recreate_http_status"),
                        "persona_kind": _report.get("persona_kind"),
                        "persona_uuid": _report.get("persona_uuid"),
                        "render_steps": _report.get("render_steps"),
                    }
                ]
                async with db_pool.acquire() as conn:
                    await record_thumbnail_studio_engine_ml_batch(
                        conn,
                        user_id=str(ctx.user_id),
                        job_id=str(upload_id),
                        engine_mode=_engine_mode,
                        variants=_variants,
                        youtube_video_id=None,
                        upload_id=str(upload_id),
                    )
                    _m8 = m8_engine_identity_payload()
                    await record_studio_usage_event(
                        conn,
                        str(ctx.user_id),
                        "thumbnail_upload_pipeline_render",
                        200,
                        {
                            "upload_id": str(upload_id),
                            "thumbnail_render_method": _thumb_render,
                            "m8_engine_ai_slug": _m8.get("ai_slug"),
                        },
                    )
            except Exception:
                logger.debug(
                    "[%s] upload thumbnail studio ML telemetry failed",
                    upload_id,
                    exc_info=True,
                )

        try:
            from services.thumbnail_ops import (
                record_pikzels_studio_ineligible_incident,
                record_pikzels_template_render_incident,
            )

            _srr_raw = (ctx.output_artifacts or {}).get("studio_render_report")
            _srr: dict = {}
            if isinstance(_srr_raw, str) and _srr_raw.strip():
                _srr = json.loads(_srr_raw) if _srr_raw.strip().startswith("{") else {}
            elif isinstance(_srr_raw, dict):
                _srr = _srr_raw
            await record_pikzels_template_render_incident(
                db_pool,
                upload_id=str(upload_id),
                user_id=str(ctx.user_id) if ctx.user_id else None,
                render_method=_thumb_render,
                studio_report=_srr if isinstance(_srr, dict) else None,
            )
            if isinstance(_srr, dict) and _srr:
                await record_pikzels_studio_ineligible_incident(
                    db_pool,
                    upload_id=str(upload_id),
                    user_id=str(ctx.user_id) if ctx.user_id else None,
                    studio_report=_srr,
                )
        except Exception as _pikz_ops_e:
            logger.debug("[%s] thumbnail ops incidents skipped: %s", upload_id, _pikz_ops_e)

        await maybe_cancel(ctx, "thumbnail")

        # ============================================================
        # STAGE 12: Caption + deterministic signal hashtags (geo / vision / ACR)
        # ============================================================
        diag_step(ctx, stage="caption", status="started", provider="m8/openai")
        try:
            ctx = await asyncio.wait_for(
                run_caption_stage(ctx, db_pool),
                timeout=STAGE_TIMEOUT_CAPTION or None,
            )
            diag_step(ctx, stage="caption", status="ok", provider="m8/openai")
            _ai_trace(ctx, upload_id, "caption", {
                "status": "ok",
                "ai_title": str(ctx.ai_title or ""),
                "ai_caption": str(ctx.ai_caption or ""),
                "ai_hashtag_count": len(ctx.ai_hashtags or []),
                "m8_platforms": list((ctx.m8_platform_captions or {}).keys()),
            })
        except asyncio.TimeoutError:
            logger.warning(f"[{upload_id}] Caption timed out after {STAGE_TIMEOUT_CAPTION}s")
            diag_step(ctx, stage="caption", status="failed", provider="m8/openai", reason="timeout")
            _ai_trace(ctx, upload_id, "caption", {"status": "error", "reason": "timeout"})
        except SkipStage as e:
            logger.info(f"[{upload_id}] Caption skipped: {e.reason}")
            diag_step(ctx, stage="caption", status="skipped", provider="m8/openai", reason=e.reason)
            _ai_trace(ctx, upload_id, "caption", {"status": "skipped", "reason": e.reason})
        except StageError as e:
            logger.warning(f"[{upload_id}] Caption error: {e.message}")
            diag_step(ctx, stage="caption", status="failed", provider="m8/openai", reason=e.message)
            _ai_trace(ctx, upload_id, "caption", {"status": "error", "reason": e.message})

        # Signal hashtags and hydration always run after caption — even when the
        # caption stage was skipped — so geo/vision signals appear in the final
        # hashtag list regardless of the autoCaptions setting.
        try:
            from services.hydration_payload import sync_hydration_payload_signal_hashtags
            from services.signal_hashtags import merge_signal_hashtags_into_ctx

            _sig_extras = merge_signal_hashtags_into_ctx(ctx)
            sync_hydration_payload_signal_hashtags(ctx, _sig_extras)
        except Exception as sig_e:
            logger.warning(f"[{upload_id}] signal_hashtags merge failed (non-fatal): {sig_e}")
        try:
            from services.hydration_enforcer import enforce_hydration

            enforce_hydration(ctx)
            await _persist_diag_artifacts_now("hydration_report")
            try:
                from services.metadata_quality import validate_metadata_quality

                validate_metadata_quality(ctx)
                await _persist_diag_artifacts_now("metadata_quality_report", "hydration_report")
            except Exception as mq_e:
                logger.debug(f"[{upload_id}] metadata_quality validation skipped: {mq_e}")
        except Exception as hy_e:
            logger.warning(f"[{upload_id}] hydration_enforcer failed (non-fatal): {hy_e}")
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                ctx.output_artifacts.setdefault(
                    "hydration_report",
                    {
                        "ok": False,
                        "error": str(hy_e)[:500],
                        "source": "worker_fallback",
                    },
                )
            await _persist_diag_artifacts_now("hydration_report")
        try:
            await db_stage.save_generated_metadata(db_pool, ctx)
        except Exception as save_e:
            logger.warning(f"[{upload_id}] save_generated_metadata failed (non-fatal): {save_e}")

        # ── Evidence-usage probes (geo + speed) ─────────────────────────────
        # These fire ops_incidents when present-but-unused signals reveal that
        # the LLM/ranker ignored hydration evidence. Per user policy, the post
        # still ships — these are alerts, not blockers.
        try:
            hints = await _check_evidence_usage_and_alert(ctx, upload_id, db_pool)
            if hints and isinstance(getattr(ctx, "output_artifacts", None), dict):
                import json as _json

                ctx.output_artifacts["coach_hints"] = _json.dumps(hints, default=str)[:8000]
        except Exception as _eu_e:
            logger.debug(f"[{upload_id}] evidence-usage probe failed: {_eu_e}")

        try:
            from services.content_success_model import score_upload_context

            hot_score, hot_model, hot_band = score_upload_context(ctx)
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                import json as _json2

                ctx.output_artifacts["content_hotness"] = _json2.dumps(
                    {"score": round(hot_score, 4), "model": hot_model, "band": hot_band},
                    default=str,
                )[:2000]
        except Exception as _hot_e:
            logger.debug(f"[{upload_id}] content hotness score skipped: {_hot_e}")

        await maybe_cancel(ctx, "caption")

        if resume_stage in (None, "post_telemetry", "post_transcode", "post_audio"):
            try:
                await pipeline_checkpoint.save_post_caption_checkpoint(db_pool, ctx)
            except Exception as _cp4:
                logger.debug(f"[{upload_id}] post_caption checkpoint skipped: {_cp4}")

        # ============================================================
        # STAGE 13: Upload processed files to R2
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

        # Merge platform thumbnail R2 keys into processed_assets for deferred publish
        if platform_thumb_r2:
            for k, v in platform_thumb_r2.items():
                processed_assets[f"thumb_{k}"] = v

        ctx.output_artifacts["processed_assets"] = json.dumps(processed_assets)
        ctx.output_artifacts["processed_video"] = ctx.processed_r2_key or ""

        try:
            await db_stage.save_processed_assets(db_pool, ctx.upload_id, processed_assets)
        except Exception as e:
            logger.warning(f"[{upload_id}] Could not persist processed_assets: {e}")

        await maybe_cancel(ctx, "upload")

        try:
            await pipeline_checkpoint.clear_checkpoint(db_pool, str(upload_id))
        except Exception:
            pass

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
            except Exception as e:
                logger.error(f"[{upload_id}] Failed to set ready_to_publish: {e}")
                if ctx:
                    try:
                        ctx.mark_error("INTERNAL", f"ready_to_publish DB update failed: {e}")
                        ctx.state = "failed"
                        ctx.finished_at = _now_utc()
                        await db_stage.mark_processing_failed(db_pool, ctx, "INTERNAL", str(e)[:2000])
                    except Exception as _dbf:
                        logger.warning(f"[{upload_id}] mark_processing_failed after ready_to_publish error: {_dbf}")
                    try:
                        _scene_fail = ""
                        if isinstance(getattr(ctx, "output_artifacts", None), dict):
                            _scene_fail = str(ctx.output_artifacts.get("scene_story") or "")
                        await notify_upload_terminal(
                            db_pool,
                            ctx,
                            str(upload_id),
                            status="failed",
                            scene_story=_scene_fail,
                        )
                    except Exception as _nf:
                        logger.warning(f"[{upload_id}] ready_to_publish failure comms: {_nf}")
                return False
            logger.info(
                f"[{upload_id}] Processing complete → ready_to_publish (waiting for scheduled_time)"
            )
            ctx.finished_at = _now_utc()
            _scene_stg = ""
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                _scene_stg = str(ctx.output_artifacts.get("scene_story") or "")
            try:
                await notify_upload_terminal(
                    db_pool,
                    ctx,
                    str(upload_id),
                    status="staged",
                    scene_story=_scene_stg,
                )
            except Exception as _stg_e:
                logger.warning(f"[{upload_id}] staged terminal comms error: {_stg_e}")
            return True

        # ============================================================
        # IMMEDIATE PATH: publish (inline or async Redis lane)
        # ============================================================
        if ASYNC_PUBLISH_QUEUE:
            pclass = (
                getattr(ctx.entitlements, "priority_class", None)
                or job_data.get("priority_class")
                or "p4"
            )
            prefs = ctx.user_settings if isinstance(ctx.user_settings, dict) else (job_data.get("preferences") or {})
            pub_payload = {
                "upload_id": str(upload_id),
                "user_id": str(user_id),
                "job_id": f"publish-{upload_id}",
                "priority_class": pclass,
                "preferences": prefs,
                "plan_features": job_data.get("plan_features") or {},
            }
            if await enqueue_publish_lane_job(pub_payload):
                logger.info(f"[{upload_id}] Immediate processing complete → publish lane queued")
                return True
            logger.error(f"[{upload_id}] Publish lane enqueue failed — falling back to inline publish")
        return await run_publish_and_notify(ctx, upload_id, user_id)

    except CancelRequested:
        logger.info(f"[{upload_id}] Pipeline cancelled")
        # ── Release wallet hold on failure ───────────────────────────────────
        try:
            if ctx:
                put_cost, aic_cost = await _get_upload_costs(ctx.upload_id)
                user_id_str = str(ctx.user_id) if ctx.user_id else None
                if user_id_str and (put_cost > 0 or aic_cost > 0):
                    await _release_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost, reason="upload_failed_refund")
        except Exception as wallet_err:
            logger.error(f"[{upload_id}] Wallet release failed: {wallet_err}")
        # User comms — cancellation is a terminal state (publish never ran).
        if ctx:
            try:
                ctx.state = "cancelled"
                ctx.finished_at = _now_utc()
                if not (getattr(ctx, "error_message", None) or "").strip():
                    ctx.error_message = "Upload cancelled before publish."
                _scene = ""
                if isinstance(getattr(ctx, "output_artifacts", None), dict):
                    _scene = str(ctx.output_artifacts.get("scene_story") or "")
                await notify_upload_terminal(
                    db_pool, ctx, str(upload_id), status="cancelled", scene_story=_scene
                )
            except Exception as _cce:
                logger.warning(f"[{upload_id}] cancel comms error: {_cce}")
            try:
                from services.upload_funnel import emit_funnel_terminal_if_needed

                emit_funnel_terminal_if_needed(str(upload_id), ctx)
            except Exception:
                pass
        return False
    except Exception as e:
        logger.exception(f"[{upload_id}] Processing failed: {e}")
        if ctx:
            err_code, err_detail = _pipeline_failure_code_and_detail(e)
            ctx.mark_error(err_code, err_detail)
            ctx.state = "failed"
            ctx.finished_at = _now_utc()
            await db_stage.mark_processing_failed(db_pool, ctx, err_code, err_detail)
        # ── Release wallet hold on failure ───────────────────────────────────
        try:
            if ctx:
                put_cost, aic_cost = await _get_upload_costs(ctx.upload_id)
                user_id_str = str(ctx.user_id) if ctx.user_id else None
                if user_id_str and (put_cost > 0 or aic_cost > 0):
                    await _release_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost, reason="upload_failed_refund")
        except Exception as wallet_err:
            logger.error(f"[{upload_id}] Wallet release failed: {wallet_err}")
        await notify_admin_error("pipeline_failure", {"upload_id": upload_id, "error": str(e)}, db_pool)
        # User comms + admin upload-status incident on full-pipeline crash
        if ctx:
            try:
                _scene = ""
                if isinstance(getattr(ctx, "output_artifacts", None), dict):
                    _scene = str(ctx.output_artifacts.get("scene_story") or "")
                await notify_upload_terminal(
                    db_pool, ctx, str(upload_id), status="failed", scene_story=_scene
                )
            except Exception as _cce:
                logger.warning(f"[{upload_id}] failure comms (notify) error: {_cce}")
        return False
    finally:
        uid = upload_id or (str(ctx.upload_id) if ctx else "")
        if uid:
            try:
                from services.worker_runtime_state import track_process_end

                await track_process_end(str(uid))
            except Exception:
                pass
        if ctx:
            try:
                await emit_ai_pipeline_summary(ctx, uid, logger, db_pool)
            except Exception:
                logger.debug("[%s] ai-pipeline-summary emission failed", uid, exc_info=True)
            try:
                term = str(getattr(ctx, "state", None) or "processing")
                await persist_pipeline_manifest(db_pool, uid, ctx, term)
            except Exception:
                logger.debug("[%s] pipeline_manifest finalize failed", uid, exc_info=True)
            try:
                from services.upload_funnel import emit_funnel_terminal_if_needed

                emit_funnel_terminal_if_needed(uid, ctx)
            except Exception:
                logger.debug("[%s] funnel terminal emit failed", uid, exc_info=True)
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


_MPH_TOKEN_RE = re.compile(r"\b\d{1,3}\s?mph\b", re.IGNORECASE)


def _contains_any_token(haystack: str, needles: List[str]) -> List[str]:
    """Return the subset of ``needles`` that appear in ``haystack`` (case-insensitive)."""
    if not haystack or not needles:
        return []
    h = haystack.lower()
    hits: List[str] = []
    for n in needles:
        nn = str(n or "").strip().lower()
        if nn and nn in h:
            hits.append(n)
    return hits


async def _check_evidence_usage_and_alert(ctx: JobContext, upload_id: str, db_pool_) -> List[Dict[str, str]]:
    """Fire ops_incidents for resolved-but-unused PAD-US/gazetteer/speed evidence.

    Two incident types (both loud-log + admin Discord/email; post still ships):
      - ``geo_signals_unused``    — gazetteer/protected-area resolved but final
        title/caption/hashtags do not mention any of them.
      - ``evidence_unused``       — telemetry/OSD speed >= 25 mph OR osd
        coverage >= 30% but no MPH/geo/Trill token in final outputs.

    Returns user-facing coach hints for growth_intelligence / queue UI.
    """
    hints: List[Dict[str, str]] = []
    if not db_pool_:
        return hints
    try:
        from services.ops_incidents import record_operational_incident
    except Exception:
        return hints

    ai_title = (ctx.ai_title or "").strip()
    ai_caption = (ctx.ai_caption or "").strip()
    tags_list = ctx.ai_hashtags or []
    ai_tags = " ".join(str(t) for t in tags_list)
    blob = f"{ai_title}\n{ai_caption}\n{ai_tags}"

    tel = ctx.telemetry or ctx.telemetry_data
    osd = ctx.dashcam_osd_context or {}

    geo_tokens: List[str] = []
    if tel is not None:
        for attr in (
            "location_road",
            "location_city",
            "location_state",
            "gazetteer_place_name",
            "padus_unit_name",
        ):
            v = getattr(tel, attr, None)
            if isinstance(v, str) and v.strip():
                geo_tokens.append(v.strip())
    near_protected = bool(getattr(tel, "near_padus", False)) if tel else False

    if geo_tokens or near_protected:
        hits = _contains_any_token(blob, geo_tokens)
        if not hits:
            try:
                await record_operational_incident(
                    db_pool_,
                    source="caption",
                    incident_type="geo_signals_unused",
                    subject=f"Geo signals unused in upload {upload_id}",
                    body=(
                        f"Resolved geo: {geo_tokens or ['(near protected land)']}\n"
                        f"near_protected: {near_protected}\n"
                        f"ai_title: {ai_title[:240]}\n"
                        f"ai_caption: {ai_caption[:600]}\n"
                        f"ai_hashtags: {ai_tags[:400]}"
                    ),
                    details={
                        "upload_id": str(upload_id),
                        "user_id": str(ctx.user_id) if ctx.user_id else None,
                        "geo_tokens": geo_tokens,
                        "near_protected_land": near_protected,
                        "ai_title": ai_title[:240],
                        "ai_caption": ai_caption[:600],
                    },
                    user_id=str(ctx.user_id) if ctx.user_id else None,
                    upload_id=str(upload_id),
                    alert_email=True,
                    alert_discord=True,
                )
                logger.warning(
                    "[%s] geo_signals_unused fired: tokens=%s near_protected=%s",
                    upload_id, geo_tokens, near_protected,
                )
                hints.append({
                    "id": "geo_signals_unused",
                    "message": (
                        "Your Trill geo signals were not reflected in the caption. "
                        "Try scenic boost or regenerate captions."
                    ),
                })
            except Exception as e:
                logger.debug(f"[{upload_id}] geo_signals_unused record failed: {e}")

    max_mph_tel = 0.0
    try:
        max_mph_tel = float(getattr(tel, "max_speed_mph", 0) or 0)
    except (TypeError, ValueError):
        max_mph_tel = 0.0
    max_mph_osd = 0.0
    coverage_pct = 0.0
    if isinstance(osd, dict):
        try:
            max_mph_osd = float(osd.get("max_speed_mph") or 0)
        except (TypeError, ValueError):
            max_mph_osd = 0.0
        try:
            coverage_pct = float(osd.get("coverage_pct") or 0.0)
        except (TypeError, ValueError):
            coverage_pct = 0.0

    fast_enough = (max_mph_tel >= 25 or max_mph_osd >= 25)
    coverage_strong = coverage_pct >= 0.30

    if fast_enough or coverage_strong:
        trill_bucket = ""
        try:
            tr_obj = ctx.trill or ctx.trill_score
            trill_bucket = str(getattr(tr_obj, "bucket", "") or "").strip()
        except Exception:
            trill_bucket = ""
        trill_hit = bool(trill_bucket and trill_bucket.lower() in blob.lower())
        mph_hit = bool(_MPH_TOKEN_RE.search(blob))
        geo_hit = bool(_contains_any_token(blob, geo_tokens))
        if not (mph_hit or geo_hit or trill_hit):
            try:
                await record_operational_incident(
                    db_pool_,
                    source="caption",
                    incident_type="evidence_unused",
                    subject=f"Evidence unused in upload {upload_id}",
                    body=(
                        f"max_speed_tel={max_mph_tel} max_speed_osd={max_mph_osd} "
                        f"coverage_pct={coverage_pct}\n"
                        f"geo_tokens={geo_tokens}\n"
                        f"trill_bucket={trill_bucket}\n"
                        f"ai_title: {ai_title[:240]}\n"
                        f"ai_caption: {ai_caption[:600]}\n"
                        f"ai_hashtags: {ai_tags[:400]}"
                    ),
                    details={
                        "upload_id": str(upload_id),
                        "user_id": str(ctx.user_id) if ctx.user_id else None,
                        "max_speed_mph_telemetry": max_mph_tel,
                        "max_speed_mph_osd": max_mph_osd,
                        "osd_coverage_pct": coverage_pct,
                        "geo_tokens": geo_tokens,
                        "trill_bucket": trill_bucket,
                        "ai_title": ai_title[:240],
                        "ai_caption": ai_caption[:600],
                    },
                    user_id=str(ctx.user_id) if ctx.user_id else None,
                    upload_id=str(upload_id),
                    alert_email=True,
                    alert_discord=True,
                )
                logger.warning(
                    "[%s] evidence_unused fired: speed_tel=%.1f speed_osd=%.1f coverage=%.2f geo=%s trill=%r",
                    upload_id, max_mph_tel, max_mph_osd, coverage_pct, geo_tokens, trill_bucket,
                )
                hints.append({
                    "id": "evidence_unused",
                    "message": (
                        "Speed or telemetry evidence was available but not used in your title/caption. "
                        "Regenerate with Trill-focused settings for a stronger automotive hook."
                    ),
                })
            except Exception as e:
                logger.debug(f"[{upload_id}] evidence_unused record failed: {e}")

    return hints


async def run_publish_and_notify(
    ctx: JobContext,
    upload_id: str,
    user_id: str,
    *,
    upload_record: Optional[dict] = None,
    publish_targets: Optional[list] = None,
) -> bool:
    """
    Run publish + notify stages and finalize upload status.
    Called by both the immediate pipeline and the deferred publish path.

    For smart deferred uploads, may persist a partial batch and return early
    while later platform slots remain (``upload_record`` + ``publish_targets``).
    """
    from services.deferred_publish_schedule import still_has_pending_publish_slots

    block_reason = _publish_policy_block_reason(ctx)
    if block_reason:
        logger.error(f"[{upload_id}] Publish blocked by policy ({block_reason[:200]!r})")
        ctx.mark_stage("publish_blocked")
        ctx.mark_error("POLICY_BLOCKED", block_reason[:2000])
        ctx.finished_at = _now_utc()
        ctx.state = "failed"
        await db_stage.mark_processing_failed(db_pool, ctx, "POLICY_BLOCKED", block_reason[:4000])
        try:
            put_cost, aic_cost = await _get_upload_costs(ctx.upload_id)
            user_id_str = str(ctx.user_id) if ctx.user_id else None
            if user_id_str and (put_cost > 0 or aic_cost > 0):
                await _release_tokens(
                    ctx.upload_id,
                    user_id_str,
                    put_cost,
                    aic_cost,
                    reason="policy_blocked_refund",
                )
        except Exception as rel_e:
            logger.error(f"[{ctx.upload_id}] Policy-block wallet release failed: {rel_e}")
        scene_story_pb = ""
        try:
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                scene_story_pb = str(ctx.output_artifacts.get("scene_story") or "")
        except Exception:
            scene_story_pb = ""
        try:
            await notify_upload_terminal(
                db_pool, ctx, upload_id, status="failed", scene_story=scene_story_pb
            )
        except Exception as ne:
            logger.warning(f"[{upload_id}] notify after policy-block (non-fatal): {ne}")
        try:
            from services.upload_funnel import emit_funnel_terminal_if_needed

            emit_funnel_terminal_if_needed(str(upload_id), ctx)
        except Exception:
            pass
        return False

    try:
        ctx = await asyncio.wait_for(
            run_publish_stage(ctx, db_pool), timeout=STAGE_TIMEOUT_PUBLISH
        )
        try:
            await pipeline_checkpoint.save_post_publish_checkpoint(db_pool, ctx)
        except Exception as _pp_cp:
            logger.debug(f"[{upload_id}] post_publish checkpoint skipped: {_pp_cp}")
    except asyncio.TimeoutError:
        logger.error(f"[{upload_id}] Publish timed out after {STAGE_TIMEOUT_PUBLISH}s")
        ctx.mark_error("TIMEOUT", f"Publish stage exceeded {STAGE_TIMEOUT_PUBLISH}s")
    except StageError as e:
        logger.error(f"[{upload_id}] Publish error: {e.message}")
        try:
            meta = getattr(e, "meta", None)
            if isinstance(meta, dict) and isinstance(getattr(ctx, "output_artifacts", None), dict):
                pmg = meta.get("publish_metadata_gate")
                if isinstance(pmg, dict):
                    ctx.output_artifacts["publish_metadata_gate"] = json.dumps(
                        pmg, default=str
                    )[:12000]
        except Exception:
            pass
        ctx.mark_error(e.code.value, e.message)

    # Smart schedule only: more platform slots remain — save batch and stay ready.
    # Immediate/scheduled must never take this path (false "pending" used to mark
    # successful publishes as PUBLISH_SLOT_MISSING).
    mode = str((upload_record or {}).get("schedule_mode") or "").strip().lower()
    if (
        upload_record
        and mode == "smart"
        and still_has_pending_publish_slots(
            upload_record,
            ctx.platform_results,
            publish_targets=publish_targets,
        )
    ):
        await db_stage.mark_deferred_publish_batch(db_pool, ctx)
        logger.info(
            f"[{upload_id}] Smart deferred partial publish saved — "
            f"waiting for remaining platform slot(s)"
        )
        return True

    try:
        await run_notify_stage(ctx, db_pool)
    except Exception as e:
        logger.warning(f"[{upload_id}] Notify error: {e}")

    ctx.finished_at = _now_utc()
    if ctx.is_partial_success():
        ctx.state = "partial"
    elif ctx.is_success():
        ctx.state = "succeeded"
    else:
        ctx.state = "failed"

    viol = list(getattr(ctx, "google_multimodal_strict_violations", None) or [])
    if viol and _google_multimodal_strict_mode() == "degrade":
        try:
            blob = {"strict": True, "violations": viol[:40], "mode": "degrade"}
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                ctx.output_artifacts["google_multimodal_strict_report"] = json.dumps(
                    blob,
                    default=str,
                )[:16000]
        except Exception:
            pass

        if ctx.state == "succeeded":
            ctx.state = "partial"
            setattr(ctx, "_notify_terminal_as_degraded", True)
            suffix = (
                "Multimodal requirements not met (GOOGLE_MULTIMODAL_STRICT): "
                + "; ".join(viol[:12])
            )
            em_old = getattr(ctx, "error_message", None) or ""
            ctx.error_message = (((em_old + " | ").strip(" | ") + suffix) if em_old else suffix)[
                :4800
            ]
        logger.warning("[%s] GOOGLE_MULTIMODAL_STRICT(degrade): %s", upload_id, viol)

    try:
        from services.metadata_quality import metadata_quality_strict_mode as _mq_mode

        if _mq_mode() == "degrade" and getattr(ctx, "metadata_quality_ok", True) is False:
            if ctx.state == "succeeded":
                ctx.state = "partial"
                setattr(ctx, "_notify_terminal_metadata_degraded", True)
            mv_list = getattr(ctx, "metadata_quality_violations", None) or []
            suffix_m = (
                "Metadata quality: " + "; ".join(str(x) for x in mv_list[:10])
            ).strip()
            if suffix_m:
                em_old2 = getattr(ctx, "error_message", None) or ""
                ctx.error_message = (((em_old2 + " | " + suffix_m).strip(" | ")) if em_old2 else suffix_m)[
                    :4800
                ]
            logger.warning("[%s] METADATA_QUALITY_STRICT(degrade): %s", upload_id, mv_list[:12])
            if isinstance(getattr(ctx, "output_artifacts", None), dict):
                ctx.output_artifacts["publish_quality_notice"] = (
                    "Publish shipped with degraded metadata: "
                    + "; ".join(str(x) for x in mv_list[:6])
                )[:2000]
    except Exception:
        pass

    await db_stage.mark_processing_completed(db_pool, ctx)

    # ── Post-publish Trill persistence check ────────────────────────────
    # Trill metadata may be saved during telemetry and dashcam OSD stages
    # backfill). If it still isn't on disk after publish — and we *had* a
    # score in memory — that's a silent persistence regression worth paging
    # admin so the score doesn't quietly disappear from analytics.
    try:
        _trill_in_mem = ctx.trill or ctx.trill_score
        if _trill_in_mem and getattr(_trill_in_mem, "score", None) is not None:
            async with db_pool.acquire() as _tconn:
                _trow = await _tconn.fetchrow(
                    "SELECT trill_score, trill_metadata FROM uploads WHERE id = $1",
                    ctx.upload_id,
                )
            _persisted_score = _trow["trill_score"] if _trow else None
            _persisted_meta = _trow["trill_metadata"] if _trow else None
            if _persisted_score is None and not _persisted_meta:
                logger.warning(
                    "[%s] Trill MISSING after publish: in_memory_score=%s bucket=%s "
                    "but DB trill_score & trill_metadata are both NULL",
                    upload_id,
                    getattr(_trill_in_mem, "score", "?"),
                    getattr(_trill_in_mem, "bucket", "?"),
                )
                try:
                    from services.ops_incidents import record_operational_incident

                    await record_operational_incident(
                        db_pool,
                        source="worker",
                        incident_type="trill_persist_missing",
                        subject=f"Trill not persisted for upload {upload_id}",
                        body=(
                            f"In-memory Trill score={getattr(_trill_in_mem, 'score', '?')} "
                            f"bucket={getattr(_trill_in_mem, 'bucket', '?')} but "
                            "uploads.trill_score and trill_metadata are both NULL."
                        ),
                        details={
                            "upload_id": str(upload_id),
                            "user_id": str(ctx.user_id) if ctx.user_id else None,
                            "in_memory_score": getattr(_trill_in_mem, "score", None),
                            "in_memory_bucket": getattr(_trill_in_mem, "bucket", None),
                        },
                        user_id=str(ctx.user_id) if ctx.user_id else None,
                        upload_id=str(upload_id),
                        alert_email=True,
                        alert_discord=True,
                    )
                except Exception as _tpe:
                    logger.debug(f"[{upload_id}] trill_persist_missing record failed: {_tpe}")
            else:
                logger.info(
                    "[%s] Trill OK persisted: db_score=%s",
                    upload_id, _persisted_score,
                )
    except Exception as _tce:
        logger.debug(f"[{upload_id}] post-publish Trill persistence check skipped: {_tce}")

    # ── Finalize wallet hold (capture on success/partial, release on failure) ──
    try:
        put_cost, aic_cost = await _get_upload_costs(ctx.upload_id)
        user_id_str = str(ctx.user_id) if ctx.user_id else None
        if user_id_str and (put_cost > 0 or aic_cost > 0):
            if ctx.is_success():
                await _capture_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost)

            elif ctx.is_partial_success():
                # Capture full cost first, then refund failed-platform slots (PUT only)
                await _capture_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost)
                async with db_pool.acquire() as _wconn:
                    await partial_refund_tokens(
                        _wconn,
                        user_id=user_id_str,
                        upload_id=ctx.upload_id,
                        succeeded_platforms=ctx.get_success_platforms(),
                        failed_platforms=ctx.get_failed_platforms(),
                        original_put_cost=put_cost,
                    )

            else:  # full failure
                await _release_tokens(
                    ctx.upload_id,
                    user_id_str,
                    put_cost,
                    aic_cost,
                    reason="upload_failed_refund",
                )

    except Exception as wallet_err:
        # Never let wallet accounting crash the job finalization
        logger.error(f"[{ctx.upload_id}] Wallet finalize failed (non-fatal): {wallet_err}")

    # ── Unified upload comms (user when enabled, admin always for partial/failed) ─
    scene_story = ""
    try:
        if isinstance(getattr(ctx, "output_artifacts", None), dict):
            scene_story = str(ctx.output_artifacts.get("scene_story") or "")
    except Exception:
        scene_story = ""
    term_status = str(getattr(ctx, "state", "") or "").strip().lower()
    if getattr(ctx, "_notify_terminal_as_degraded", False) or getattr(
        ctx, "_notify_terminal_metadata_degraded", False
    ):
        term_status = "degraded"
    try:
        await notify_upload_terminal(
            db_pool, ctx, upload_id, status=term_status, scene_story=scene_story
        )
    except Exception as _comms_err:
        logger.warning(f"[{upload_id}] Upload comms (notify) failed (non-fatal): {_comms_err}")

    if ctx.is_success():
        await db_stage.increment_upload_count(db_pool, user_id)

    try:
        from services.upload_funnel import emit_funnel_terminal_if_needed

        emit_funnel_terminal_if_needed(str(upload_id), ctx)
    except Exception:
        pass

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
    Called by the scheduler when a ready_to_publish job has platform slot(s) due.
    Loads processed assets and publishes only the due platform batch (smart mode)
    or all platforms at once (single scheduled_time mode).
    """
    from services.deferred_publish_schedule import (
        hydrate_platform_results_into_ctx,
        platforms_due_for_publish,
    )
    from stages.publish_stage import resolve_publish_targets

    logger.info(f"[{upload_id}] Deferred publish firing")
    ctx = None

    try:
        from services.worker_runtime_state import track_publish_start

        await track_publish_start(str(upload_id), stage="publish")
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            logger.error(f"[{upload_id}] Records not found for deferred publish")
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET status = 'failed',
                        error_code = 'INTERNAL',
                        error_detail = 'Deferred publish could not load upload or user record.',
                        updated_at = NOW()
                    WHERE id = $1 AND status = 'processing'
                    """,
                    upload_id,
                )
            try:
                await notify_admin_error(
                    "deferred_publish_records_missing",
                    {"upload_id": upload_id, "user_id": user_id},
                    db_pool,
                )
            except Exception:
                pass
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        await db_stage.merge_pikzels_thumbnail_persona_id(db_pool, user_id, user_settings)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        # Reconstruct minimal job_data
        job_data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "job_id": f"deferred-publish-{upload_id}",
        }
        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        _merge_job_preferences(ctx, job_data)
        ctx.user_record = user_record
        ctx.started_at = _now_utc()
        ctx.state = "processing"
        try:
            setattr(ctx, "_db_pool", db_pool)
        except Exception:
            pass

        try:
            resume_stage, _resume_holder = await pipeline_checkpoint.try_resume_from_checkpoint(
                db_pool, job_data, upload_record, ctx
            )
            if resume_stage:
                logger.info(f"[{upload_id}] Deferred publish checkpoint resume | stage={resume_stage}")
        except Exception as _cp_e:
            logger.debug(f"[{upload_id}] deferred publish checkpoint probe skipped: {_cp_e}")

        publish_targets = await resolve_publish_targets(ctx, db_pool)
        now = _now_utc()
        due_platforms = platforms_due_for_publish(upload_record, now, publish_targets)
        if not due_platforms:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET status = 'ready_to_publish', updated_at = NOW()
                    WHERE id = $1 AND status = 'processing'
                    """,
                    upload_id,
                )
            logger.debug(
                f"[{upload_id}] Deferred publish: no platform slots due this cycle"
            )
            return True

        hydrate_platform_results_into_ctx(ctx, upload_record.get("platform_results"))
        ctx.deferred_publish_platform_filter = set(due_platforms)
        logger.info(
            f"[{upload_id}] Deferred publish batch platforms={sorted(due_platforms)}"
        )
        try:
            setattr(ctx, "_db_pool", db_pool)
        except Exception:
            pass
        try:
            if getattr(ctx, "vehicle_make_id", None) or getattr(ctx, "vehicle_model_id", None):
                from services.vehicle_catalog import fetch_vehicle_labels

                async with db_pool.acquire() as _c:
                    _lab = await fetch_vehicle_labels(_c, ctx.vehicle_make_id, ctx.vehicle_model_id)
                ctx.vehicle_make_name = _lab.get("make_name")
                ctx.vehicle_model_name = _lab.get("model_name")
        except Exception as _ve:
            logger.debug("[%s] vehicle label hydrate skipped (deferred): %s", upload_id, _ve)

        # ── Server-side entitlement cap enforcement ───────────────────────
        # Re-resolve entitlements from DB (authoritative — not from job_data).
        # This prevents any client or stale job_data from bypassing tier limits.
        if ctx.user_record and ctx.entitlements:
            ent = ctx.entitlements

            # Cap thumbnails to tier max
            if hasattr(ctx, "num_thumbnails") and ctx.num_thumbnails > ent.max_thumbnails:
                logger.info(
                    f"[{ctx.upload_id}] Clamping thumbnails {ctx.num_thumbnails} → {ent.max_thumbnails} (tier cap)"
                )
                ctx.num_thumbnails = ent.max_thumbnails

            # Cap caption frames
            if hasattr(ctx, "caption_frames") and ctx.caption_frames > ent.max_caption_frames:
                ctx.caption_frames = ent.max_caption_frames

            # Enforce AI depth
            if not ent.can_ai and hasattr(ctx, "use_ai"):
                ctx.use_ai = False

        # Mark processing started again (publish phase)
        await db_stage.mark_processing_started(db_pool, ctx)

        # Restore platform_videos from processed_assets stored in DB
        from core.helpers import coerce_processed_assets_map

        processed_assets = coerce_processed_assets_map(upload_record.get("processed_assets"))

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
        due_filter = set(due_platforms) if due_platforms else None
        for platform, r2_key in processed_assets.items():
            if platform == "default" or platform.startswith("thumb_"):
                continue
            if due_filter is not None and str(platform).strip().lower() not in due_filter:
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

        # Download platform-specific thumbnails for publish (thumb_youtube, thumb_instagram, thumb_facebook)
        platform_thumb_map: Dict[str, str] = {}
        platform_thumb_r2_keys: Dict[str, str] = {}
        for key, r2_key in processed_assets.items():
            if not key.startswith("thumb_") or not r2_key:
                continue
            plat = key.replace("thumb_", "")
            if due_filter is not None and str(plat).strip().lower() not in due_filter:
                continue
            platform_thumb_r2_keys[plat] = r2_key
            local_thumb = temp_dir / f"thumb_{plat}.jpg"
            try:
                await r2_stage.download_file(r2_key, local_thumb)
                if local_thumb.exists():
                    platform_thumb_map[plat] = str(local_thumb)
                    logger.debug(f"[{upload_id}] Downloaded thumb_{plat} for publish")
            except Exception as e:
                logger.debug(f"[{upload_id}] Thumb {plat} download failed: {e}")
        if platform_thumb_map:
            ctx.output_artifacts["platform_thumbnail_map"] = json.dumps(platform_thumb_map)
        if platform_thumb_r2_keys:
            ctx.output_artifacts["platform_thumbnail_r2_keys"] = json.dumps(platform_thumb_r2_keys)

        try:
            result = await run_publish_and_notify(
                ctx,
                upload_id,
                user_id,
                upload_record=upload_record,
                publish_targets=publish_targets,
            )
            return result
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass

    except Exception as e:
        logger.exception(f"[{upload_id}] Deferred publish failed: {e}")
        if ctx:
            err_code, err_detail = _pipeline_failure_code_and_detail(e)
            ctx.mark_error(err_code, err_detail)
            ctx.state = "failed"
            ctx.finished_at = _now_utc()
            await db_stage.mark_processing_failed(db_pool, ctx, err_code, err_detail)
        await notify_admin_error("deferred_publish_failure", {"upload_id": upload_id, "error": str(e)}, db_pool)
        if ctx:
            try:
                _scene_dp = ""
                if isinstance(getattr(ctx, "output_artifacts", None), dict):
                    _scene_dp = str(ctx.output_artifacts.get("scene_story") or "")
                await notify_upload_terminal(
                    db_pool, ctx, str(upload_id), status="failed", scene_story=_scene_dp
                )
            except Exception as _cce:
                logger.warning(f"[{upload_id}] deferred-publish comms error: {_cce}")
            try:
                from services.upload_funnel import emit_funnel_terminal_if_needed

                emit_funnel_terminal_if_needed(str(upload_id), ctx)
            except Exception:
                pass
        return False
    finally:
        try:
            from services.worker_runtime_state import track_publish_end

            await track_publish_end(str(upload_id))
        except Exception:
            pass



# ---------------------------------------------------------------------------
# ANALYTICS SYNC LOOP
# ---------------------------------------------------------------------------

async def _sync_one_upload_analytics(
    conn: asyncpg.Connection,
    upload_id: str,
    user_id: str,
    pr_list: list,
    token_map: dict,
) -> dict:
    """
    Pull engagement stats for one completed upload from each platform API.
    Returns totals dict and writes them + analytics_synced_at to DB.
    """
    import httpx as _httpx
    from stages.publish_stage import decrypt_token

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats = {}

    async with _httpx.AsyncClient(timeout=15) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            ok = (
                pr.get("success") is True
                or str(pr.get("status", "")).lower() in ("published", "succeeded", "success")
            )
            if not ok:
                continue

            tok = token_map.get(plat, {})
            access_token = tok.get("access_token", "")
            if not access_token:
                continue

            video_id = (
                pr.get("platform_video_id")
                or pr.get("video_id") or pr.get("videoId") or pr.get("id")
                or pr.get("media_id") or pr.get("post_id") or pr.get("share_id")
            )

            try:
                if plat == "tiktok" and video_id:
                    resp = await client.post(
                        "https://open.tiktokapis.com/v2/video/query/",
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "application/json",
                        },
                        params={"fields": "id,view_count,like_count,comment_count,share_count"},
                        json={"filters": {"video_ids": [str(video_id)]}},
                    )
                    if resp.status_code == 200:
                        vids = resp.json().get("data", {}).get("videos", []) or []
                        if vids:
                            v = vids[0]
                            s = {
                                "views":    int(v.get("view_count")    or 0),
                                "likes":    int(v.get("like_count")     or 0),
                                "comments": int(v.get("comment_count") or 0),
                                "shares":   int(v.get("share_count")   or 0),
                            }
                            platform_stats["tiktok"] = s
                            total_views    += s["views"];    total_likes    += s["likes"]
                            total_comments += s["comments"]; total_shares   += s["shares"]
                    else:
                        logger.debug(f"[analytics-sync] TikTok HTTP {resp.status_code} for {upload_id}")

                elif plat == "youtube" and video_id:
                    resp = await client.get(
                        "https://www.googleapis.com/youtube/v3/videos",
                        params={"part": "statistics", "id": str(video_id)},
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    if resp.status_code == 200:
                        items = resp.json().get("items", []) or []
                        if items:
                            st = items[0].get("statistics", {})
                            s = {
                                "views":    int(st.get("viewCount")    or 0),
                                "likes":    int(st.get("likeCount")    or 0),
                                "comments": int(st.get("commentCount") or 0),
                                "shares":   0,
                            }
                            platform_stats["youtube"] = s
                            total_views    += s["views"]; total_likes    += s["likes"]
                            total_comments += s["comments"]
                    else:
                        logger.debug(f"[analytics-sync] YouTube HTTP {resp.status_code} for {upload_id}")

                elif plat == "instagram" and video_id:
                    # Instagram Insights API requires numeric media_id (not shortcode)
                    media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{media_id}/insights",
                        params={
                            "access_token": access_token,
                            "metric": "views,plays,likes,comments,saved,shares,reach",
                        },
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        ig_views = ig_plays = 0
                        for m in resp.json().get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [])
                            val  = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                            if name == "views":      ig_views     = val
                            elif name == "plays":    ig_plays     = val  # deprecated fallback
                            elif name == "likes":    s["likes"]   += val
                            elif name == "comments": s["comments"] += val
                            elif name == "shares":   s["shares"]  += val
                        s["views"] = ig_views or ig_plays  # prefer views over deprecated plays
                        platform_stats["instagram"] = s
                        total_views    += s["views"];    total_likes    += s["likes"]
                        total_comments += s["comments"]; total_shares   += s["shares"]
                    else:
                        logger.debug(f"[analytics-sync] Instagram HTTP {resp.status_code} for {upload_id}")

                elif plat == "facebook" and video_id:
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{video_id}",
                        params={
                            "access_token": access_token,
                            "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)",
                        },
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        for m in (resp.json().get("insights", {}) or {}).get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [{}])
                            val  = vals[-1].get("value", 0) if vals else 0
                            if isinstance(val, dict):
                                val = sum(val.values())
                            val = int(val or 0)
                            if name == "total_video_views":                     s["views"]    += val
                            elif name == "total_video_reactions_by_type_total": s["likes"]    += val
                            elif name == "total_video_comments":                 s["comments"] += val
                            elif name == "total_video_shares":                   s["shares"]   += val
                        platform_stats["facebook"] = s
                        total_views    += s["views"];    total_likes    += s["likes"]
                        total_comments += s["comments"]; total_shares   += s["shares"]
                    else:
                        logger.debug(f"[analytics-sync] Facebook HTTP {resp.status_code} for {upload_id}")

            except Exception as e:
                logger.warning(f"[analytics-sync] {plat}/{upload_id}: {e}")
                continue

    # Persist results + stamp analytics_synced_at regardless (even if all zeros)
    # so we dont endlessly retry uploads whose scopes arent approved yet.
    await conn.execute(
        """
        UPDATE uploads
           SET views = $1,
               likes = $2,
               comments = $3,
               shares = $4,
               analytics_synced_at = NOW(),
               updated_at = NOW()
         WHERE id = $5
           AND user_id = $6
        """,
        total_views,
        total_likes,
        total_comments,
        total_shares,
        upload_id,
        user_id,
    )

    return {
        "views": total_views, "likes": total_likes,
        "comments": total_comments, "shares": total_shares,
        "platform_stats": platform_stats,
    }


async def run_analytics_sync_loop() -> None:
    """
    Background loop that periodically fetches engagement stats from platform APIs
    for all completed uploads and writes them back to the DB.

    Config (env vars):
      ANALYTICS_SYNC_INTERVAL_SECONDS  How often to run (default: 21600 = 6h)
      ANALYTICS_SYNC_BATCH_SIZE        Uploads per cycle   (default: 20)
      ANALYTICS_RESYNC_HOURS           Re-sync every N hrs (default: 6)

    Only syncs uploads where:
      • status IN (completed, succeeded, partial)
      • analytics_synced_at IS NULL  OR  analytics_synced_at < NOW() - resync_interval
      • created_at within the last 90 days (older content rarely changes)

    Intentionally sequential — no concurrency within the batch to stay well
    inside platform rate limits.
    """
    from stages.publish_stage import decrypt_token

    global shutdown_requested

    logger.info(
        f"[analytics-sync] loop started | "        f"interval={ANALYTICS_SYNC_INTERVAL}s | "        f"batch={ANALYTICS_SYNC_BATCH} | "        f"resync_every={ANALYTICS_RESYNC_HOURS}h"
    )

    # Stagger startup — let the worker settle before hitting platform APIs
    await asyncio.sleep(60)

    while not shutdown_requested:
        try:
            resync_cutoff = _now_utc() - timedelta(hours=ANALYTICS_RESYNC_HOURS)
            window_start  = _now_utc() - timedelta(days=90)

            async with db_pool.acquire() as conn:
                uploads = await conn.fetch(
                    """
                    SELECT u.id AS upload_id, u.user_id, u.platform_results, u.platforms
                      FROM uploads u
                     WHERE u.status = 'completed'
                       AND u.created_at >= $1
                       AND (u.analytics_synced_at IS NULL OR u.analytics_synced_at < $2)
                     ORDER BY u.analytics_synced_at ASC NULLS FIRST, u.created_at DESC
                     LIMIT $3
                    """,
                    window_start,
                    resync_cutoff,
                    ANALYTICS_SYNC_BATCH,
                )

            if not uploads:
                logger.debug("[analytics-sync] nothing to sync this cycle")
            else:
                logger.info(f"[analytics-sync] syncing {len(uploads)} upload(s)")
                synced = errors = 0

                for row in uploads:
                    if shutdown_requested:
                        break

                    upload_id = str(row["upload_id"])
                    user_id   = str(row["user_id"])

                    # Parse platform_results
                    try:
                        raw_pr = row["platform_results"]
                        if isinstance(raw_pr, str):
                            raw_pr = json.loads(raw_pr)
                    except Exception:
                        raw_pr = []

                    pr_list = []
                    if isinstance(raw_pr, list):
                        pr_list = [x for x in raw_pr if isinstance(x, dict)]
                    elif isinstance(raw_pr, dict):
                        pr_list = [
                            {"platform": k, **(v if isinstance(v, dict) else {})}
                            for k, v in raw_pr.items()
                        ]
                    # Normalise canonical field aliases
                    for pr in pr_list:
                        if pr.get("platform_video_id") and not pr.get("video_id"):
                            pr["video_id"] = pr["platform_video_id"]

                    if not pr_list:
                        # No platform results to query — stamp and skip
                        async with db_pool.acquire() as conn:
                            await conn.execute(
                                """
                                UPDATE uploads
                                   SET analytics_synced_at = NOW(),
                                       updated_at = NOW()
                                 WHERE id = $1
                                """,
                                row["upload_id"],
                            )
                        continue

                    # Fetch active tokens for this user
                    async with db_pool.acquire() as conn:
                        token_rows = await conn.fetch(
                            """
                            SELECT platform, token_blob, account_id
                              FROM platform_tokens
                             WHERE user_id = $1
                               AND revoked_at IS NULL
                            """,
                            row["user_id"],
                        )

                    token_map = {}
                    for tr in token_rows:
                        try:
                            blob = tr["token_blob"]
                            if isinstance(blob, str):
                                blob = json.loads(blob)
                            dec = decrypt_token(blob)
                            if dec:
                                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                                    dec["ig_user_id"] = str(tr["account_id"])
                                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                                    dec["page_id"] = str(tr["account_id"])
                                token_map[tr["platform"]] = dec
                        except Exception:
                            pass

                    if not token_map:
                        # User has no active tokens — stamp and skip
                        async with db_pool.acquire() as conn:
                            await conn.execute(
                                """
                                UPDATE uploads
                                   SET analytics_synced_at = NOW(),
                                       updated_at = NOW()
                                 WHERE id = $1
                                """,
                                row["upload_id"],
                            )
                        continue

                    try:
                        async with db_pool.acquire() as conn:
                            result = await _sync_one_upload_analytics(
                                conn, upload_id, user_id, pr_list, token_map
                            )
                        synced += 1
                        logger.debug(
                            f"[analytics-sync] {upload_id}: "                            f"views={result['views']} likes={result['likes']} "                            f"comments={result['comments']} shares={result['shares']}"                        )
                    except Exception as e:
                        errors += 1
                        logger.warning(f"[analytics-sync] {upload_id} failed: {e}")

                    # Small delay between API calls — be a good citizen
                    await asyncio.sleep(1)

                logger.info(
                    f"[analytics-sync] cycle complete | synced={synced} errors={errors}"
                )

        except asyncpg.PostgresError as e:
            logger.warning(f"[analytics-sync] DB error: {e}")
        except Exception as e:
            logger.exception(f"[analytics-sync] unexpected error: {e}")

        # Sleep until next cycle, but wake early on shutdown
        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=ANALYTICS_SYNC_INTERVAL,
            )
            break  # shutdown signalled
        except asyncio.TimeoutError:
            pass  # normal — continue loop

    logger.info("[analytics-sync] loop stopped")


# ---------------------------------------------------------------------------
# KPI COLLECTOR LOOP
# ---------------------------------------------------------------------------

async def run_kpi_collector_loop() -> None:
    """
    Every KPI_COLLECTOR_INTERVAL (default 30 min), fetch cost/revenue from
    Stripe, OpenAI, Mailgun, Cloudflare, etc. and post to cost_tracking.
    Uses env keys: STRIPE_SECRET_KEY, OPENAI_API_KEY, MAILGUN_API_KEY, etc.
    """
    global shutdown_requested, shutdown_event

    from stages.kpi_collector import run_kpi_collect

    logger.info(f"[kpi-collector] loop started | interval={KPI_COLLECTOR_INTERVAL}s")

    await asyncio.sleep(120)  # Stagger after analytics sync

    while not shutdown_requested:
        try:
            summary = await run_kpi_collect(db_pool)
            if summary.get("rows_inserted", 0) > 0:
                logger.info(
                    f"[kpi-collector] synced | stripe=${summary.get('stripe_fees',0):.2f} "
                    f"mailgun=${summary.get('mailgun_cost',0):.2f} openai=${summary.get('openai_cost',0):.4f} "
                    f"rows={summary.get('rows_inserted',0)}"
                )
        except Exception as e:
            logger.warning(f"[kpi-collector] error: {e}")

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=KPI_COLLECTOR_INTERVAL,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("[kpi-collector] loop stopped")


# ---------------------------------------------------------------------------
# PLATFORM METRICS CACHE LOOP
# ---------------------------------------------------------------------------

async def run_platform_metrics_cache_loop() -> None:
    """Refresh platform_metrics_cache for all OAuth-connected users (worker background)."""
    from services.platform_metrics_job import run_platform_metrics_cache_loop as _service_loop

    logger.info("[platform-metrics-cache] delegating to services.platform_metrics_job")
    await _service_loop(db_pool)


async def run_admin_email_jobs_loop() -> None:
    """Daily cron for trial reminders, digests, and scheduled publish alerts."""
    global shutdown_event
    from services.admin_email_jobs import run_admin_email_jobs_loop as _service_loop

    logger.info("[admin-email-jobs] delegating to services.admin_email_jobs")
    await _service_loop(db_pool, shutdown_event)


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
       Non-smart: scheduled_time <= NOW() → publish all platforms.
       Smart: each platform slot in schedule_metadata is published when its
       time arrives; partial results merge into platform_results until all
       targets are handled, then status → completed/partial/failed.
       Fires deferred publish pipeline per due batch.

    Both dispatch as asyncio tasks governed by _job_semaphore.
    """
    global shutdown_requested, _job_semaphore

    logger.info(
        f"Scheduler loop started | "
        f"poll_interval={SCHEDULER_POLL_INTERVAL}s | "
        f"processing_window={PROCESSING_WINDOW_MINUTES}min"
    )

    while not shutdown_requested:
        _ensure_worker_semaphores()
        try:
            now = _now_utc()
            process_cutoff = now + timedelta(minutes=PROCESSING_WINDOW_MINUTES)

            async with db_pool.acquire() as conn:
                from services.upload.stuck_recovery import recover_staged_without_schedule

                await recover_staged_without_schedule(conn, db_pool, limit=20)

                # --------------------------------------------------------
                # CHECK 1: staged jobs entering processing window
                # --------------------------------------------------------
                # ── Capacity-aware dispatch ────────────────────────────────
                # Only pull as many staged jobs as there are free process slots.
                # This prevents dispatching 500-job batches into memory when
                # workers are already saturated.
                free_slots = _process_slots_free()
                from core.process_stats import blocks_new_process_job
                from services.worker_admission import process_dispatch_limit

                fleet_summary = None
                try:
                    from services.worker_fleet_snapshot import (
                        fetch_worker_heartbeat_rows,
                        summarize_fleet,
                    )

                    fleet_summary = summarize_fleet(await fetch_worker_heartbeat_rows(db_pool))
                except Exception:
                    fleet_summary = None

                dispatch_limit = process_dispatch_limit(
                    local_free_slots=free_slots,
                    memory_blocks=blocks_new_process_job(),
                    fleet=fleet_summary,
                )

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
                    max(dispatch_limit, 0),
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

                    if WORKER_LANE == "publish":
                        payload = await _build_process_job_payload(
                            upload_id,
                            user_id,
                            deferred=True,
                            job_id=f"scheduled-{upload_id}",
                        )
                        if payload and await enqueue_process_lane_job(payload):
                            logger.info(
                                f"[{upload_id}] Scheduler: staged → Redis process queue "
                                f"(publish worker)"
                            )
                        else:
                            logger.error(
                                f"[{upload_id}] Scheduler: Redis enqueue failed — reverting to staged"
                            )
                            await conn.execute(
                                """
                                UPDATE uploads
                                   SET status = 'staged', updated_at = NOW()
                                 WHERE id = $1 AND status = 'queued'
                                """,
                                row["upload_id"],
                            )
                    else:
                        asyncio.create_task(_run_job_with_semaphore(job_data))
                        logger.debug(f"[{upload_id}] Processing task dispatched")

                # --------------------------------------------------------
                # CHECK 2: ready_to_publish jobs with due platform slot(s)
                # --------------------------------------------------------
                ready_jobs = await conn.fetch(
                    """
                    SELECT u.id AS upload_id, u.user_id, u.scheduled_time,
                           u.schedule_mode, u.schedule_metadata, u.platforms,
                           u.platform_results, u.target_accounts
                    FROM uploads u
                    WHERE u.status = 'ready_to_publish'
                      AND u.scheduled_time IS NOT NULL
                    ORDER BY u.scheduled_time ASC
                    LIMIT 100
                    """,
                )

                from services.deferred_publish_schedule import platforms_due_for_publish

                for row in ready_jobs:
                    upload_id = str(row["upload_id"])
                    user_id = str(row["user_id"])
                    upload_row = dict(row)

                    # Require at least one due platform (smart and non-smart).
                    # Claiming with an empty due set caused deferred publish to
                    # download R2 assets then no-op back to ready_to_publish.
                    mode = str(upload_row.get("schedule_mode") or "scheduled").lower()
                    if mode != "smart":
                        st = upload_row.get("scheduled_time")
                        if st is None or st > now:
                            continue
                    due = platforms_due_for_publish(upload_row, now)
                    if not due:
                        continue

                    # Atomically claim — refresh processing_started_at so stale
                    # recovery does not treat deferred publish as a zombie transcode.
                    updated = await conn.fetchval(
                        """
                        UPDATE uploads
                        SET status = 'processing',
                            processing_started_at = NOW(),
                            updated_at = NOW()
                        WHERE id = $1 AND status = 'ready_to_publish'
                        RETURNING id
                        """,
                        row["upload_id"],
                    )
                    if not updated:
                        continue

                    logger.info(
                        f"[{upload_id}] Scheduler: ready_to_publish → publishing batch "
                        f"(platforms={sorted(due)}, schedule_mode={mode})"
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


def upload_row_indicates_active_pipeline(
    *,
    status: Any,
    processing_stage: Any,
    updated_at: Any = None,
    stale_after_minutes: int | None = None,
) -> bool:
    """
    True when another worker has *entered* the pipeline for this upload.

    ``status='processing'`` alone is NOT enough — ``/complete`` and enqueue set
    that before the consumer runs. Skipping on status deadlocks the job
    (stage stays null forever). ``claimed`` (set by mark_processing_started)
    counts as active until a real stage overwrites it.
    """
    if str(status or "").strip().lower() != "processing":
        return False
    stage = str(processing_stage or "").strip()
    if not stage:
        return False
    if stale_after_minutes is None:
        try:
            from services.worker_admission import active_pipeline_stale_minutes

            stale_after_minutes = active_pipeline_stale_minutes()
        except Exception:
            stale_after_minutes = 90
    if updated_at is not None and stale_after_minutes > 0:
        try:
            from datetime import datetime, timedelta, timezone

            ts = updated_at
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - ts > timedelta(minutes=int(stale_after_minutes)):
                return False
        except Exception:
            pass
    return True


async def _upload_already_processing(upload_id: str) -> bool:
    if not upload_id or db_pool is None:
        return False
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status, processing_stage, updated_at
                  FROM uploads
                 WHERE id = $1::uuid
                """,
                upload_id,
            )
        if not row:
            return False
        return upload_row_indicates_active_pipeline(
            status=row["status"],
            processing_stage=row["processing_stage"],
            updated_at=row["updated_at"],
        )
    except Exception:
        return False


async def _run_job_with_semaphore(job_data: dict) -> None:
    """
    Wrapper: run processing pipeline (FFmpeg-heavy) inside the PROCESS semaphore.
    Uses _process_semaphore (WORKER_CONCURRENCY=3 slots by default).
    """
    uid = str(job_data.get("user_id") or "")
    pc = str(job_data.get("priority_class") or "p4")
    upload_id = str(job_data.get("upload_id") or "")
    from stages.redis_job_queue import user_process_release, user_process_wait_acquire

    # Duplicate reclaim/scheduler race: don't wait on our own processing row.
    if upload_id and await _upload_already_processing(upload_id):
        logger.info(
            "[%s] scheduler skip — upload already processing",
            upload_id,
        )
        return

    slot_held = False
    try:
        if uid and not await user_process_wait_acquire(
            redis_client,
            uid,
            pc,
            upload_id=upload_id,
            shutdown_check=lambda: shutdown_requested,
            already_processing_check=(
                (lambda: _upload_already_processing(upload_id)) if upload_id else None
            ),
        ):
            logger.warning(
                "[%s] scheduler job skipped — user process slot unavailable; reverting queued → staged",
                upload_id or "?",
            )
            # Avoid parking forever in queued when the user process slot is busy.
            if upload_id and db_pool is not None:
                try:
                    async with db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            UPDATE uploads
                            SET status = 'staged', updated_at = NOW()
                            WHERE id = $1 AND status = 'queued'
                              AND COALESCE(schedule_mode, 'immediate') IN ('scheduled', 'smart')
                            """,
                            upload_id,
                        )
                except Exception as rev_e:
                    logger.warning("[%s] queued→staged revert failed: %s", upload_id, rev_e)
            return
        slot_held = bool(uid)
        if not await _wait_for_process_memory_headroom(upload_id):
            logger.warning(
                "[%s] scheduler process deferred — memory pressure; leaving job for later reclaim",
                upload_id or "?",
            )
            if upload_id and db_pool is not None:
                try:
                    async with db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            UPDATE uploads
                            SET status = 'staged', updated_at = NOW()
                            WHERE id = $1 AND status = 'queued'
                              AND COALESCE(schedule_mode, 'immediate') IN ('scheduled', 'smart')
                            """,
                            upload_id,
                        )
                except Exception as rev_e:
                    logger.warning("[%s] memory-defer queued→staged failed: %s", upload_id, rev_e)
            return
        async with _process_semaphore:
            try:
                await run_processing_pipeline(job_data)
            except Exception as e:
                logger.exception(f"[{job_data.get('upload_id')}] Unhandled pipeline error: {e}")
            finally:
                try:
                    from core.process_stats import memory_pressure_level

                    if memory_pressure_level() != "ok":
                        import gc

                        gc.collect()
                except Exception:
                    pass
    finally:
        if slot_held and uid:
            await user_process_release(redis_client, uid)


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


def _ensure_worker_semaphores() -> None:
    """Initialize lane semaphores once (safe for publish-only or process-only workers)."""
    global _process_semaphore, _publish_semaphore, _job_semaphore, _heavy_semaphore
    if _process_semaphore is None:
        _process_semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)
    if _publish_semaphore is None:
        _publish_semaphore = asyncio.Semaphore(PUBLISH_CONCURRENCY)
    if _heavy_semaphore is None:
        _heavy_semaphore = asyncio.Semaphore(WORKER_HEAVY_PIPELINE_SLOTS)
    _job_semaphore = _process_semaphore


def _process_slots_free() -> int:
    _ensure_worker_semaphores()
    if _process_semaphore is None:
        return WORKER_CONCURRENCY
    try:
        return max(0, int(getattr(_process_semaphore, "_value", 0) or 0))
    except (TypeError, ValueError):
        return 0


async def _wait_for_process_memory_headroom(upload_id: str = "") -> bool:
    """Block briefly under soft pressure; return False under hard pressure (caller should requeue)."""
    from core.process_stats import blocks_new_process_job, memory_pressure_level, sample_memory_mb

    # Fast path
    mem = sample_memory_mb()
    level = memory_pressure_level(mem)
    if level == "ok":
        return True

    max_wait = float(os.environ.get("MEMORY_ADMIT_WAIT_SEC", "45") or 45)
    step = float(os.environ.get("MEMORY_ADMIT_POLL_SEC", "3") or 3)
    waited = 0.0
    while waited < max_wait and not shutdown_requested:
        mem = sample_memory_mb()
        level = memory_pressure_level(mem)
        if level == "ok":
            if waited > 0:
                logger.info(
                    "[%s] memory headroom restored after %.0fs | rss=%sMB pct=%s%%",
                    upload_id or "?",
                    waited,
                    mem.get("rss_mb"),
                    mem.get("pct_of_limit"),
                )
            return True
        if level == "hard":
            logger.warning(
                "[%s] memory HARD pressure — deferring process job | rss=%sMB pct=%s%% limit=%sMB",
                upload_id or "?",
                mem.get("rss_mb"),
                mem.get("pct_of_limit"),
                mem.get("limit_mb"),
            )
            return False
        logger.warning(
            "[%s] memory soft pressure — waiting before encode | rss=%sMB pct=%s%% (%.0fs/%.0fs)",
            upload_id or "?",
            mem.get("rss_mb"),
            mem.get("pct_of_limit"),
            waited,
            max_wait,
        )
        await asyncio.sleep(step)
        waited += step
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    return not blocks_new_process_job()


def _process_queue_name(priority_class: str) -> str:
    return PROCESS_PRIORITY_QUEUE if priority_class in PRIORITY_QUEUE_CLASSES else PROCESS_NORMAL_QUEUE


def _publish_queue_name(priority_class: str) -> str:
    return PUBLISH_PRIORITY_QUEUE if priority_class in PRIORITY_QUEUE_CLASSES else PUBLISH_NORMAL_QUEUE


async def _build_process_job_payload(
    upload_id: str,
    user_id: str,
    *,
    deferred: bool,
    job_id: str,
    resume_from_checkpoint: bool = False,
) -> Optional[dict]:
    """Shape matches /complete enqueue for Redis process lane."""
    try:
        upload_record = await db_stage.load_upload_record(db_pool, upload_id)
        user_record = await db_stage.load_user(db_pool, user_id)
        if not upload_record or not user_record:
            return None
        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        await db_stage.merge_pikzels_thumbnail_persona_id(db_pool, user_id, user_settings)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        ent = get_entitlements_from_user(user_record, overrides)
        from core.upload_baseline_defaults import (
            apply_upload_baseline_defaults,
            sanitize_settings_for_job_payload,
        )

        apply_upload_baseline_defaults(user_settings, tier=ent.tier)
        payload = {
            "upload_id": str(upload_id),
            "user_id": str(user_id),
            "job_id": job_id,
            "deferred": deferred,
            "preferences": sanitize_settings_for_job_payload(user_settings),
            "plan_features": {
                "ai": ent.can_ai,
                "priority": ent.can_priority,
                "watermark": ent.can_watermark,
                "ai_depth": ent.ai_depth,
                "caption_frames": ent.max_caption_frames,
            },
            "priority_class": ent.priority_class,
        }
        if resume_from_checkpoint:
            payload["resume_from_checkpoint"] = True
        return payload
    except Exception as e:
        logger.warning(f"[{upload_id}] _build_process_job_payload failed: {e}")
        return None


async def enqueue_process_lane_job(payload: dict) -> bool:
    """LPUSH a job to the process Redis lane (worker has its own redis client)."""
    global redis_client
    if not redis_client:
        return False
    pc = str(payload.get("priority_class") or "p4")
    q = _process_queue_name(pc)
    body = {
        **payload,
        "job_id": payload.get("job_id") or str(uuid.uuid4()),
        "lane": "process",
        "enqueued_at": _now_utc().isoformat(),
    }
    try:
        from stages.redis_job_queue import enqueue_process_job

        ok = await enqueue_process_job(redis_client, q, body)
        if ok:
            logger.info(f"[{payload.get('upload_id')}] Enqueued process lane → {q}")
        return ok
    except Exception as e:
        logger.error(f"[{payload.get('upload_id')}] enqueue process lane failed: {e}")
        return False


async def enqueue_publish_lane_job(payload: dict) -> bool:
    """LPUSH a job to the publish Redis lane."""
    global redis_client
    if not redis_client:
        return False
    pc = str(payload.get("priority_class") or "p4")
    q = _publish_queue_name(pc)
    body = {
        **payload,
        "job_id": payload.get("job_id") or str(uuid.uuid4()),
        "lane": "publish",
        "enqueued_at": _now_utc().isoformat(),
    }
    try:
        from stages.redis_job_queue import enqueue_process_job

        ok = await enqueue_process_job(redis_client, q, body)
        if ok:
            logger.info(f"[{payload.get('upload_id')}] Enqueued publish lane → {q}")
        return ok
    except Exception as e:
        logger.error(f"[{payload.get('upload_id')}] enqueue publish lane failed: {e}")
        return False


async def _publish_one_job(job_json: str) -> None:
    try:
        job_data = json.loads(job_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid publish job JSON: {e}")
        return
    upload_id = job_data.get("upload_id")
    user_id = job_data.get("user_id")
    if not upload_id or not user_id:
        logger.error("Publish job missing upload_id or user_id")
        return
    async with _publish_semaphore:
        try:
            await run_deferred_publish(str(upload_id), str(user_id))
        except Exception as e:
            logger.exception(f"[{upload_id}] publish lane job error: {e}")


async def publish_jobs() -> None:
    """
    Consume publish-lane Redis jobs (API-light). Used when ASYNC_PUBLISH_QUEUE is on
    or on dedicated publish workers (WORKER_LANE=publish).
    """
    global shutdown_requested, redis_client, _publish_semaphore

    _ensure_worker_semaphores()
    from stages.redis_job_queue import (
        DEFAULT_GROUP,
        ensure_stream_group,
        make_worker_consumer_name,
        stream_key_for_list,
        use_redis_streams,
        xack_message,
        xreadgroup_one,
    )

    queues = [PUBLISH_PRIORITY_QUEUE, PUBLISH_NORMAL_QUEUE]
    stream_keys = [stream_key_for_list(q) for q in queues]
    stream_consumer = make_worker_consumer_name()
    if use_redis_streams():
        for sk in stream_keys:
            try:
                await ensure_stream_group(redis_client, sk, DEFAULT_GROUP)
            except Exception as e:
                logger.warning("publish ensure_stream_group %s: %s", sk, e)

    logger.info(
        f"Publish consumer started | publish_concurrency={PUBLISH_CONCURRENCY} | "
        f"queues={PUBLISH_PRIORITY_QUEUE}, {PUBLISH_NORMAL_QUEUE} | streams={use_redis_streams()}"
    )
    consecutive_redis_errors = 0
    active_tasks: List[asyncio.Task] = []

    while not shutdown_requested:
        active_tasks = [t for t in active_tasks if not t.done()]
        try:
            job_json = None
            stream_ack: Optional[Tuple[str, str]] = None
            if use_redis_streams():
                got = await xreadgroup_one(
                    redis_client,
                    stream_keys,
                    DEFAULT_GROUP,
                    stream_consumer,
                    block_ms=int(POLL_INTERVAL * 1000),
                )
                if got:
                    sk, mid, job_json = got
                    stream_ack = (sk, mid)
            if job_json is None and REDIS_JOB_LEGACY_DRAIN:
                job_raw = await redis_client.brpop(
                    queues,
                    timeout=1 if use_redis_streams() else int(POLL_INTERVAL),
                )
                if job_raw:
                    _, job_json = job_raw
            consecutive_redis_errors = 0
            if not job_json:
                continue

            async def _run_publish_and_ack(payload: str, ack: Optional[Tuple[str, str]]) -> None:
                try:
                    await _publish_one_job(payload)
                finally:
                    if ack and use_redis_streams():
                        try:
                            await xack_message(redis_client, ack[0], DEFAULT_GROUP, ack[1])
                        except Exception as ack_e:
                            logger.warning("publish xack failed: %s", ack_e)

            task = asyncio.create_task(_run_publish_and_ack(job_json, stream_ack))
            active_tasks.append(task)
        except redis.ReadOnlyError:
            consecutive_redis_errors += 1
            wait = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(f"Redis read-only (publish), retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            consecutive_redis_errors += 1
            wait = min(REDIS_RETRY_DELAY * consecutive_redis_errors, 60.0)
            logger.warning(f"Redis error (publish): {e}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            logger.exception(f"Publish consumer error: {e}")
            await asyncio.sleep(1)

    if active_tasks:
        logger.info(f"Shutdown: waiting for {len(active_tasks)} in-flight publish jobs...")
        await asyncio.gather(*active_tasks, return_exceptions=True)
    logger.info("Publish consumer stopped")


async def run_stream_reclaim_loop() -> None:
    """
    Reclaim stale pending Redis Stream messages (worker crash mid-job).
    """
    global shutdown_requested, redis_client

    from stages.redis_job_queue import (
        DEFAULT_GROUP,
        make_worker_consumer_name,
        process_stream_keys_ordered,
        stream_key_for_list,
        use_redis_streams,
        xack_message,
        xautoclaim_batch,
    )

    if not use_redis_streams():
        while not shutdown_requested:
            await asyncio.sleep(30)
        return

    interval = int(os.environ.get("STREAM_RECLAIM_INTERVAL_SEC", "25") or 25)
    min_idle = int(os.environ.get("STREAM_RECLAIM_MIN_IDLE_MS", "120000") or 120000)
    count = int(os.environ.get("STREAM_RECLAIM_COUNT", "8") or 8)
    consumer = make_worker_consumer_name() + "-reclaim"
    publish_streams = [
        stream_key_for_list(PUBLISH_PRIORITY_QUEUE),
        stream_key_for_list(PUBLISH_NORMAL_QUEUE),
    ]
    process_streams = process_stream_keys_ordered(
        PROCESS_PRIORITY_QUEUE,
        PROCESS_NORMAL_QUEUE,
        PRIORITY_JOB_QUEUE,
        UPLOAD_JOB_QUEUE,
    )
    publish_set = set(publish_streams)

    logger.info(
        "Stream reclaim loop started | interval=%ss min_idle_ms=%s streams=%s",
        interval,
        min_idle,
        len(process_streams) + len(publish_streams),
    )

    while not shutdown_requested:
        try:
            for sk in process_streams + publish_streams:
                batch = await xautoclaim_batch(
                    redis_client, sk, DEFAULT_GROUP, consumer, min_idle, count
                )
                for mid, job_json in batch:
                    is_publish = sk in publish_set

                    async def _reclaim_one(
                        payload: str,
                        ack_sk: str,
                        ack_mid: str,
                        publish_lane: bool,
                    ) -> None:
                        try:
                            if publish_lane:
                                await _publish_one_job(payload)
                            else:
                                await _process_one_job(payload)
                        finally:
                            try:
                                await xack_message(redis_client, ack_sk, DEFAULT_GROUP, ack_mid)
                            except Exception as ack_e:
                                logger.warning("reclaim xack failed: %s", ack_e)

                    asyncio.create_task(_reclaim_one(job_json, sk, mid, is_publish))
            await asyncio.sleep(interval)
        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            logger.warning("stream reclaim redis error: %s", e)
            await asyncio.sleep(min(interval, 10))
        except Exception as e:
            logger.exception("stream reclaim loop error: %s", e)
            await asyncio.sleep(interval)

    logger.info("Stream reclaim loop stopped")


async def run_stale_job_recovery_loop() -> None:
    """
    Re-enqueue long-stuck `queued` uploads (Redis consumer died / lost message).

    When STALE_PROCESSING_MINUTES > 0, also handle stuck `processing` rows:
    if STALE_PROCESSING_RECOVER_CHECKPOINT is on and ``pipeline_resume`` is
    at ``post_telemetry`` / ``post_transcode`` / ``post_audio`` / ``post_caption``
    with a valid checkpoint (R2 objects verified when past transcode),
    reset the row to ``queued`` and enqueue with ``resume_from_checkpoint`` so
    the worker skips finished stages (e.g. mid-transcode crash resumes after
    telemetry without re-scoring Trill). Otherwise full re-enqueue, then
    STALE_PROCESSING fail if enqueue fails.
    """
    global shutdown_requested, shutdown_event, redis_client

    if not STALE_JOB_RECOVERY_ENABLED:
        logger.info("Stale job recovery disabled (STALE_JOB_RECOVERY_ENABLED=false)")
        await shutdown_event.wait()
        return

    logger.info(
        f"Stale job recovery started | interval={STALE_JOB_RECOVERY_INTERVAL_SEC}s | "
        f"queued>{STALE_QUEUED_MINUTES}m | processing>{STALE_PROCESSING_MINUTES}m (0=off) | "
        f"processing_checkpoint_recover={STALE_PROCESSING_RECOVER_CHECKPOINT}"
    )

    while not shutdown_requested:
        try:
            if db_pool and redis_client:
                async with db_pool.acquire() as conn:
                    stale = await conn.fetch(
                        """
                        SELECT id, user_id
                          FROM uploads
                         WHERE status = 'queued'
                           AND processing_started_at IS NULL
                           AND updated_at < NOW() - ($1::int * INTERVAL '1 minute')
                         ORDER BY updated_at ASC
                         LIMIT 15
                        """,
                        STALE_QUEUED_MINUTES,
                    )
                for row in stale:
                    uid = str(row["user_id"])
                    up = str(row["id"])
                    lock_key = f"stale_recover:{up}"
                    try:
                        got = await redis_client.set(lock_key, "1", nx=True, ex=180)
                    except Exception:
                        got = True
                    if not got:
                        continue
                    async with db_pool.acquire() as conn2:
                        st = await conn2.fetchval(
                            "SELECT status FROM uploads WHERE id = $1",
                            up,
                        )
                    if str(st or "").lower() != "queued":
                        continue
                    payload = await _build_process_job_payload(
                        up,
                        uid,
                        deferred=False,
                        job_id=f"stale-recover-{up}",
                        resume_from_checkpoint=True,
                    )
                    if not payload:
                        continue
                    ok = await enqueue_process_lane_job(payload)
                    if ok:
                        async with db_pool.acquire() as conn3:
                            await conn3.execute(
                                "UPDATE uploads SET updated_at = NOW() WHERE id = $1",
                                up,
                            )
                        logger.warning(
                            f"[{up}] stale recovery: re-enqueued process job "
                            f"(queued {STALE_QUEUED_MINUTES}+ min, resume_checkpoint=1)"
                        )

                if STALE_PROCESSING_MINUTES > 0:
                    async with db_pool.acquire() as conn:
                        # Include rows with null processing_started_at (never claimed cleanly).
                        zomb = await conn.fetch(
                            """
                            SELECT id, user_id
                              FROM uploads
                             WHERE status = 'processing'
                               AND COALESCE(updated_at, processing_started_at, created_at)
                                   < NOW() - ($1::int * INTERVAL '1 minute')
                               AND (
                                    processing_started_at IS NULL
                                    OR processing_started_at
                                       < NOW() - ($1::int * INTERVAL '1 minute')
                               )
                             ORDER BY COALESCE(updated_at, created_at) ASC
                             LIMIT 10
                            """,
                            STALE_PROCESSING_MINUTES,
                        )
                    for row in zomb:
                        up = str(row["id"])
                        uid = str(row["user_id"])
                        lock_key = f"stale_recover:{up}"
                        try:
                            got = await redis_client.set(lock_key, "1", nx=True, ex=180)
                        except Exception:
                            got = True
                        if not got:
                            continue

                        recover_ok = False
                        ur: Optional[dict] = None
                        if STALE_PROCESSING_RECOVER_CHECKPOINT:
                            ur = await db_stage.load_upload_record(db_pool, up)
                            if ur:
                                cp = pipeline_checkpoint.load_resume(ur)
                                stg = str((cp or {}).get("stage") or "").lower()
                                if (
                                    cp
                                    and stg
                                    in (
                                        "post_telemetry",
                                        "post_transcode",
                                        "post_audio",
                                        "post_caption",
                                    )
                                    and pipeline_checkpoint.checkpoint_matches_upload(cp, ur)
                                ):
                                    try:
                                        if stg == "post_telemetry":
                                            # Mid-transcode / pre-transcode crash: source still in R2;
                                            # resume re-downloads video and skips telemetry/trill.
                                            src = (
                                                (cp.get("source_r2_key") or ur.get("r2_key") or "")
                                            ).strip()
                                            recover_ok = bool(src)
                                        else:
                                            transcode_ok = await pipeline_checkpoint.verify_transcode_r2_keys(
                                                dict(cp.get("transcode_r2") or {})
                                            )
                                            if stg == "post_caption":
                                                thumb_ok = bool(
                                                    (
                                                        cp.get("thumbnail_r2_key")
                                                        or ur.get("thumbnail_r2_key")
                                                        or ""
                                                    ).strip()
                                                )
                                                recover_ok = transcode_ok and thumb_ok
                                            else:
                                                recover_ok = transcode_ok
                                    except Exception:
                                        recover_ok = False

                        if recover_ok and ur:
                            deferred = str(ur.get("schedule_mode") or "immediate").lower() in (
                                "scheduled",
                                "smart",
                            )
                            payload = await _build_process_job_payload(
                                up,
                                uid,
                                deferred=deferred,
                                job_id=f"stale-recover-proc-{up}",
                                resume_from_checkpoint=True,
                            )
                            if payload and await enqueue_process_lane_job(payload):
                                async with db_pool.acquire() as uconn:
                                    await uconn.execute(
                                        """
                                        UPDATE uploads
                                           SET status = 'queued',
                                               processing_started_at = NULL,
                                               error_code = NULL,
                                               error_detail = NULL,
                                               updated_at = NOW()
                                         WHERE id = $1 AND status = 'processing'
                                        """,
                                        up,
                                    )
                                logger.warning(
                                    f"[{up}] stale recovery: re-queued from pipeline checkpoint "
                                    f"(processing>{STALE_PROCESSING_MINUTES}m)"
                                )
                                continue
                            if recover_ok:
                                logger.error(
                                    f"[{up}] stale checkpoint recovery: enqueue failed — leaving status=processing"
                                )
                                continue

                        # No checkpoint (or verify failed): full re-process beats leaving a zombie row.
                        ur = ur or await db_stage.load_upload_record(db_pool, up)
                        if ur:
                            from services.upload.r2_storage_guard import (
                                mark_source_not_in_r2_failed,
                                SOURCE_NOT_IN_R2_MESSAGE,
                                upload_source_head_status,
                            )

                            r2_status = upload_source_head_status(ur)
                            if r2_status == "missing":
                                async with db_pool.acquire() as uconn:
                                    await mark_source_not_in_r2_failed(
                                        uconn,
                                        up,
                                        detail=SOURCE_NOT_IN_R2_MESSAGE,
                                    )
                                logger.error(
                                    f"[{up}] stale recovery: source missing in R2 — marked SOURCE_NOT_IN_R2"
                                )
                                continue

                            # Publish-phase zombies already have encoded assets — do NOT
                            # re-enqueue FFmpeg (stale-recover-full → claim lost + OOM).
                            _assets = ur.get("processed_assets")
                            if isinstance(_assets, str):
                                try:
                                    _assets = json.loads(_assets) if _assets.strip() else {}
                                except Exception:
                                    _assets = {}
                            _has_assets = isinstance(_assets, dict) and any(
                                k and not str(k).startswith("thumb_") and k != "default"
                                for k in _assets
                                if _assets.get(k)
                            )
                            _cp_pub = pipeline_checkpoint.load_resume(ur)
                            _stg_pub = str((_cp_pub or {}).get("stage") or "").lower()
                            if _has_assets or _stg_pub in ("post_caption", "post_publish"):
                                async with db_pool.acquire() as uconn:
                                    await uconn.execute(
                                        """
                                        UPDATE uploads
                                           SET status = 'ready_to_publish',
                                               processing_started_at = NULL,
                                               updated_at = NOW()
                                         WHERE id = $1 AND status = 'processing'
                                        """,
                                        up,
                                    )
                                asyncio.create_task(
                                    _run_deferred_publish_with_semaphore(up, uid)
                                )
                                logger.warning(
                                    f"[{up}] stale recovery: publish-phase → ready_to_publish "
                                    f"(skip full re-encode; processing>{STALE_PROCESSING_MINUTES}m)"
                                )
                                continue
                        deferred = False
                        resume_cp = False
                        if ur:
                            deferred = str(ur.get("schedule_mode") or "immediate").lower() in (
                                "scheduled",
                                "smart",
                            )
                            _cp_full = pipeline_checkpoint.load_resume(ur)
                            _stg_full = str((_cp_full or {}).get("stage") or "").lower()
                            resume_cp = bool(
                                _cp_full
                                and _stg_full
                                in (
                                    "post_telemetry",
                                    "post_transcode",
                                    "post_audio",
                                    "post_caption",
                                )
                                and pipeline_checkpoint.checkpoint_matches_upload(_cp_full, ur)
                            )
                        full_payload = await _build_process_job_payload(
                            up,
                            uid,
                            deferred=deferred,
                            job_id=f"stale-recover-full-{up}",
                            resume_from_checkpoint=resume_cp,
                        )
                        if full_payload and await enqueue_process_lane_job(full_payload):
                            async with db_pool.acquire() as uconn:
                                await uconn.execute(
                                    """
                                    UPDATE uploads
                                       SET status = 'queued',
                                           processing_started_at = NULL,
                                           error_code = NULL,
                                           error_detail = NULL,
                                           updated_at = NOW()
                                     WHERE id = $1 AND status = 'processing'
                                    """,
                                    up,
                                )
                            logger.warning(
                                f"[{up}] stale recovery: full re-enqueue (processing>{STALE_PROCESSING_MINUTES}m)"
                            )
                            continue

                        async with db_pool.acquire() as uconn:
                            tag = await uconn.execute(
                                """
                                UPDATE uploads
                                   SET status = 'failed',
                                       error_code = 'STALE_PROCESSING',
                                       error_detail = 'Worker marked this upload stale after extended processing with no progress.',
                                       updated_at = NOW()
                                 WHERE id = $1 AND status = 'processing'
                                """,
                                up,
                            )
                        if tag and str(tag) != "UPDATE 0":
                            logger.error(f"[{up}] stale recovery: marked failed STALE_PROCESSING")
                            try:
                                await notify_admin_error(
                                    "stale_processing_marked_failed",
                                    {"upload_id": up, "user_id": uid},
                                    db_pool,
                                )
                            except Exception:
                                pass

                from services.upload.stuck_recovery import (
                    recover_abandoned_pending,
                    recover_auto_retry_failed_immediate,
                    recover_enqueue_failed_pending,
                    recover_stuck_ready_to_publish,
                    recover_stuck_staged_past_window,
                )

                async def _dispatch_stuck_publish(uid_up: str, uid_user: str) -> None:
                    asyncio.create_task(
                        _run_deferred_publish_with_semaphore(uid_up, uid_user)
                    )

                async with db_pool.acquire() as conn:
                    n_enqueue = await recover_enqueue_failed_pending(
                        conn,
                        build_payload=_build_process_job_payload,
                        enqueue=enqueue_process_lane_job,
                    )
                    if n_enqueue:
                        logger.warning("stale recovery: re-enqueued %s ENQUEUE_FAILED pending uploads", n_enqueue)

                    n_auto = await recover_auto_retry_failed_immediate(
                        conn,
                        build_payload=_build_process_job_payload,
                        enqueue=enqueue_process_lane_job,
                    )
                    if n_auto:
                        logger.warning("stale recovery: auto-retried %s failed immediate uploads", n_auto)

                    n_staged = await recover_stuck_staged_past_window(
                        conn,
                        build_payload=_build_process_job_payload,
                        enqueue=enqueue_process_lane_job,
                    )
                    if n_staged:
                        logger.warning("stale recovery: re-dispatched %s overdue staged uploads", n_staged)

                    n_abandoned = await recover_abandoned_pending(conn, db_pool)
                    if n_abandoned:
                        logger.warning(
                            "stale recovery: failed %s abandoned pending uploads",
                            n_abandoned,
                        )

                    ready_stats = await recover_stuck_ready_to_publish(
                        conn,
                        db_pool,
                        dispatch_publish=_dispatch_stuck_publish,
                    )
                    if any(ready_stats.get(k, 0) for k in ("failed", "redispatched", "repaired")):
                        logger.warning(
                            "stale recovery: ready_to_publish repaired=%s redispatched=%s failed=%s",
                            ready_stats.get("repaired", 0),
                            ready_stats.get("redispatched", 0),
                            ready_stats.get("failed", 0),
                        )
        except Exception as e:
            logger.warning(f"Stale job recovery cycle error: {e}")

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=float(STALE_JOB_RECOVERY_INTERVAL_SEC),
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("Stale job recovery stopped")


# ---------------------------------------------------------------------------
# Redis job consumer (immediate uploads only)
# ---------------------------------------------------------------------------

async def _process_one_job(job_json: str) -> None:
    try:
        job_data = json.loads(job_json)
        while isinstance(job_data, str):
            job_data = json.loads(job_data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid job JSON: {e}")
        return
    if not isinstance(job_data, dict):
        logger.error(f"Invalid job payload type: {type(job_data).__name__}")
        return
    uid = str(job_data.get("user_id") or "")
    pc = str(job_data.get("priority_class") or "p4")
    upload_id = str(job_data.get("upload_id") or "")
    from stages.redis_job_queue import (
        heal_user_process_slots,
        user_process_release,
        user_process_wait_acquire,
    )

    # Slot must be released on EVERY exit after acquire — including memory-pressure
    # requeue. Leaking here filled the per-user cap after OOM/memory storms.
    slot_held = False
    try:
        # Duplicate stream reclaim / dual consumer: never wait on our own row.
        if upload_id and await _upload_already_processing(upload_id):
            logger.info(
                "[%s] skip — upload already processing (no slot wait)",
                upload_id,
            )
            return

        if uid:
            # Heal counter vs DB before waiting (OOM leaves Redis INCR without DECR).
            try:
                if db_pool is not None:
                    async with db_pool.acquire() as conn:
                        db_n = int(
                            await conn.fetchval(
                                """
                                SELECT COUNT(*)::int FROM uploads
                                 WHERE user_id = $1::uuid AND status = 'processing'
                                """,
                                uid,
                            )
                            or 0
                        )
                    await heal_user_process_slots(redis_client, uid, db_processing=db_n)
            except Exception as _heal_e:
                logger.debug("[%s] user slot heal skipped: %s", upload_id or "?", _heal_e)

            got = await user_process_wait_acquire(
                redis_client,
                uid,
                pc,
                upload_id=upload_id,
                shutdown_check=lambda: shutdown_requested,
                already_processing_check=(
                    (lambda: _upload_already_processing(upload_id)) if upload_id else None
                ),
            )
            if not got:
                # Another consumer may already be processing this upload — don't stack
                # duplicate requeues (logs showed attempt 1/5 forever via stale recovery).
                if upload_id and await _upload_already_processing(upload_id):
                    logger.info(
                        "[%s] user slot unavailable but status=processing — skip requeue",
                        upload_id,
                    )
                    return
                max_requeue = int(os.environ.get("USER_SLOT_MAX_REQUEUE", "5") or 5)
                gen = int(job_data.get("_user_slot_wait_gen") or 0) + 1
                if gen <= max_requeue and not shutdown_requested:
                    job_data["_user_slot_wait_gen"] = gen
                    delay = min(5.0 * gen, 60.0)
                    logger.warning(
                        "[%s] user slot unavailable — requeue attempt %s/%s in %.0fs",
                        upload_id or "?",
                        gen,
                        max_requeue,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    await enqueue_process_lane_job(job_data)
                else:
                    logger.error(
                        "[%s] user slot wait exhausted — job not requeued (gen=%s shutdown=%s)",
                        upload_id or "?",
                        gen,
                        shutdown_requested,
                    )
                return
            slot_held = True

        if not await _wait_for_process_memory_headroom(upload_id):
            max_requeue = int(os.environ.get("MEMORY_PRESSURE_MAX_REQUEUE", "8") or 8)
            gen = int(job_data.get("_memory_wait_gen") or 0) + 1
            if gen <= max_requeue and not shutdown_requested:
                job_data["_memory_wait_gen"] = gen
                delay = min(8.0 * gen, 90.0)
                logger.warning(
                    "[%s] memory pressure — requeue attempt %s/%s in %.0fs",
                    upload_id or "?",
                    gen,
                    max_requeue,
                    delay,
                )
                await asyncio.sleep(delay)
                await enqueue_process_lane_job(job_data)
            else:
                logger.error(
                    "[%s] memory pressure requeue exhausted (gen=%s) — job left for stale recovery",
                    upload_id or "?",
                    gen,
                )
            return

        async with _process_semaphore:
            try:
                await run_processing_pipeline(job_data)
            except Exception as e:
                logger.exception(f"Unhandled pipeline exception: {e}")
            finally:
                try:
                    from core.process_stats import memory_pressure_level

                    if memory_pressure_level() != "ok":
                        import gc

                        gc.collect()
                except Exception:
                    pass
    finally:
        if slot_held and uid:
            await user_process_release(redis_client, uid)


async def process_jobs() -> None:
    """
    Consume process-lane jobs from Redis (FFmpeg-heavy).
    Reads from [process:priority, process:normal] + legacy queues.
    Scheduled/staged jobs are handled by run_scheduler_loop() instead.
    """
    global shutdown_requested, redis_client, _process_semaphore, _publish_semaphore, _job_semaphore

    _ensure_worker_semaphores()

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

    from stages.redis_job_queue import (
        DEFAULT_GROUP,
        ensure_stream_group,
        legacy_process_list_keys,
        make_worker_consumer_name,
        process_stream_keys_ordered,
        use_redis_streams,
        xack_message,
        xreadgroup_one,
    )

    consecutive_redis_errors = 0
    active_tasks: List[asyncio.Task] = []
    stream_keys = process_stream_keys_ordered(
        PROCESS_PRIORITY_QUEUE,
        PROCESS_NORMAL_QUEUE,
        PRIORITY_JOB_QUEUE,
        UPLOAD_JOB_QUEUE,
    )
    stream_consumer = make_worker_consumer_name()
    if use_redis_streams():
        for sk in stream_keys:
            try:
                await ensure_stream_group(redis_client, sk, DEFAULT_GROUP)
            except Exception as e:
                logger.warning("ensure_stream_group %s: %s", sk, e)

    while not shutdown_requested:
        active_tasks = [t for t in active_tasks if not t.done()]

        try:
            # Backpressure: never pile waiting process tasks in RAM (FFmpeg + buffers).
            if len(active_tasks) >= WORKER_CONCURRENCY or _process_slots_free() <= 0:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            from core.process_stats import blocks_new_process_job, sample_memory_mb

            if blocks_new_process_job():
                mem = sample_memory_mb()
                logger.warning(
                    "process consumer paused — memory pressure | rss=%sMB pct=%s%% active=%s",
                    mem.get("rss_mb"),
                    mem.get("pct_of_limit"),
                    len(active_tasks),
                )
                await asyncio.sleep(max(POLL_INTERVAL, 2.0))
                continue

            job_json = None
            stream_ack: Optional[Tuple[str, str]] = None

            if use_redis_streams():
                got = await xreadgroup_one(
                    redis_client,
                    stream_keys,
                    DEFAULT_GROUP,
                    stream_consumer,
                    block_ms=int(POLL_INTERVAL * 1000),
                )
                if got:
                    sk, mid, job_json = got
                    stream_ack = (sk, mid)

            if job_json is None and REDIS_JOB_LEGACY_DRAIN:
                job_raw = await redis_client.brpop(
                    all_process_queues,
                    timeout=1 if use_redis_streams() else int(POLL_INTERVAL),
                )
                if job_raw:
                    _, job_json = job_raw

            consecutive_redis_errors = 0

            if not job_json:
                continue

            async def _run_and_ack(payload: str, ack: Optional[Tuple[str, str]]) -> None:
                try:
                    await _process_one_job(payload)
                finally:
                    if ack and use_redis_streams():
                        try:
                            await xack_message(redis_client, ack[0], DEFAULT_GROUP, ack[1])
                        except Exception as ack_e:
                            logger.warning("xack failed: %s", ack_e)

            task = asyncio.create_task(_run_and_ack(job_json, stream_ack))
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

async def _ensure_worker_heartbeat_schema() -> None:
    """Create worker_heartbeat table on first run. Idempotent."""
    if not db_pool:
        return
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_heartbeat (
                worker_id           TEXT PRIMARY KEY,
                last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                worker_concurrency  INT NOT NULL DEFAULT 0,
                publish_concurrency INT NOT NULL DEFAULT 0,
                version             TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_worker_heartbeat_last_seen
                ON worker_heartbeat (last_seen_at DESC)
        """)
        for ddl in (
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS worker_lane TEXT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS service_name TEXT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS region TEXT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS memory_rss_mb REAL",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS memory_peak_mb REAL",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS memory_limit_mb REAL",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS heavy_pipeline_slots INT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS process_slots_in_use INT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS publish_slots_in_use INT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS active_process_jobs JSONB",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS active_publish_jobs JSONB",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS hostname TEXT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS git_commit TEXT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS memory_pct REAL",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS memory_pressure TEXT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS heavy_slots_in_use INT",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS load_1m REAL",
            "ALTER TABLE worker_heartbeat ADD COLUMN IF NOT EXISTS admission_blocked BOOLEAN",
        ):
            await conn.execute(ddl)


def _heartbeat_runtime_snapshot() -> dict:
    """Collect memory, slot usage, and active jobs for this worker instance."""
    from core.process_stats import (
        format_semaphore_slots,
        observability_sample,
        render_instance_context,
        worker_config_snapshot,
    )
    from services.worker_runtime_state import snapshot as runtime_snapshot

    obs = observability_sample()
    cfg = worker_config_snapshot()
    inst = render_instance_context()
    jobs = runtime_snapshot()
    proc_slots = format_semaphore_slots(WORKER_CONCURRENCY, _process_semaphore)
    pub_slots = format_semaphore_slots(PUBLISH_CONCURRENCY, _publish_semaphore)
    heavy_slots = format_semaphore_slots(WORKER_HEAVY_PIPELINE_SLOTS, _heavy_semaphore)
    return {
        "mem": obs,
        "obs": obs,
        "cfg": cfg,
        "inst": inst,
        "jobs": jobs,
        "proc_slots": proc_slots,
        "pub_slots": pub_slots,
        "heavy_slots": heavy_slots,
    }


async def run_heartbeat_loop() -> None:
    """
    Background loop: writes WORKER_ID + last_seen_at + concurrency snapshot
    every HEARTBEAT_INTERVAL seconds. Admin can SELECT worker_id, last_seen_at
    FROM worker_heartbeat to see liveness (stale = NOW() - last_seen_at > 30s).
    """
    global shutdown_event
    await _ensure_worker_heartbeat_schema()

    snap = _heartbeat_runtime_snapshot()
    version = snap["inst"].get("git_commit") or (
        os.environ.get("RENDER_GIT_COMMIT", "")[:12] or None
    )
    mem_warn_pct = float(os.environ.get("MEMORY_WARN_PCT", "85") or 85)
    last_mem_warn_at = 0.0

    # Upsert started_at on first beat so admin can see worker uptime
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO worker_heartbeat
                        (worker_id, last_seen_at, started_at,
                         worker_concurrency, publish_concurrency, version,
                         worker_lane, service_name, region, hostname, git_commit,
                         heavy_pipeline_slots, memory_limit_mb)
                    VALUES ($1, NOW(), NOW(), $2, $3, $4,
                            $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (worker_id) DO UPDATE SET
                        started_at          = NOW(),
                        worker_concurrency  = EXCLUDED.worker_concurrency,
                        publish_concurrency = EXCLUDED.publish_concurrency,
                        version             = EXCLUDED.version,
                        worker_lane         = EXCLUDED.worker_lane,
                        service_name        = EXCLUDED.service_name,
                        region              = EXCLUDED.region,
                        hostname            = EXCLUDED.hostname,
                        git_commit          = EXCLUDED.git_commit,
                        heavy_pipeline_slots = EXCLUDED.heavy_pipeline_slots,
                        memory_limit_mb     = EXCLUDED.memory_limit_mb,
                        last_seen_at        = NOW()
                    """,
                    WORKER_ID,
                    WORKER_CONCURRENCY,
                    PUBLISH_CONCURRENCY,
                    version,
                    snap["cfg"]["worker_lane"],
                    snap["inst"].get("service_name"),
                    snap["inst"].get("region"),
                    snap["inst"].get("hostname"),
                    version,
                    snap["cfg"]["heavy_pipeline_slots"],
                    snap["mem"].get("limit_mb"),
                )
        except Exception as e:
            logger.warning(f"Heartbeat init failed: {e}")

    logger.info(
        "Heartbeat loop started | worker_id=%s | interval=%ss | lane=%s | "
        "process=%s/%s publish=%s/%s heavy=%s/%s | rss=%sMB limit=%sMB",
        WORKER_ID,
        HEARTBEAT_INTERVAL,
        snap["cfg"]["worker_lane"],
        snap["proc_slots"]["in_use"],
        snap["proc_slots"]["total"],
        snap["pub_slots"]["in_use"],
        snap["pub_slots"]["total"],
        snap["heavy_slots"]["in_use"],
        snap["heavy_slots"]["total"],
        snap["mem"].get("rss_mb"),
        snap["mem"].get("limit_mb"),
    )

    import time as _time

    while not shutdown_requested:
        try:
            snap = _heartbeat_runtime_snapshot()
            mem = snap["mem"]
            jobs = snap["jobs"]
            if db_pool:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE worker_heartbeat SET
                            last_seen_at = NOW(),
                            memory_rss_mb = $2,
                            memory_peak_mb = $3,
                            process_slots_in_use = $4,
                            publish_slots_in_use = $5,
                            active_process_jobs = $6::jsonb,
                            active_publish_jobs = $7::jsonb,
                            memory_pct = $8,
                            memory_pressure = $9,
                            heavy_slots_in_use = $10,
                            load_1m = $11,
                            admission_blocked = $12,
                            memory_limit_mb = COALESCE($13, memory_limit_mb)
                        WHERE worker_id = $1
                        """,
                        WORKER_ID,
                        mem.get("rss_mb"),
                        mem.get("peak_rss_mb"),
                        snap["proc_slots"]["in_use"],
                        snap["pub_slots"]["in_use"],
                        json.dumps(jobs.get("active_process_jobs") or []),
                        json.dumps(jobs.get("active_publish_jobs") or []),
                        mem.get("pct_of_limit"),
                        mem.get("memory_pressure"),
                        snap["heavy_slots"]["in_use"],
                        mem.get("load_1m"),
                        bool(mem.get("admission_blocked")),
                        mem.get("limit_mb"),
                    )
            pct = mem.get("pct_of_limit")
            if pct is not None and pct >= mem_warn_pct:
                now = _time.monotonic()
                if now - last_mem_warn_at >= 60:
                    last_mem_warn_at = now
                    logger.warning(
                        "Memory pressure | rss=%sMB peak=%sMB limit=%sMB pct=%s%% pressure=%s load1=%s | "
                        "process_slots=%s/%s heavy=%s/%s | active_process=%s active_publish=%s | jobs=%s",
                        mem.get("rss_mb"),
                        mem.get("peak_rss_mb"),
                        mem.get("limit_mb"),
                        pct,
                        mem.get("memory_pressure"),
                        mem.get("load_1m"),
                        snap["proc_slots"]["in_use"],
                        snap["proc_slots"]["total"],
                        snap["heavy_slots"]["in_use"],
                        snap["heavy_slots"]["total"],
                        jobs.get("process_count"),
                        jobs.get("publish_count"),
                        [j.get("upload_id") for j in (jobs.get("active_process_jobs") or [])[:5]],
                    )
                    if mem.get("memory_pressure") == "hard":
                        try:
                            import sentry_sdk

                            sentry_sdk.capture_message(
                                f"Worker hard memory pressure {pct}% "
                                f"(rss={mem.get('rss_mb')} limit={mem.get('limit_mb')})",
                                level="warning",
                            )
                        except Exception:
                            pass
        except Exception as e:
            # Don't crash the loop on transient DB issues — supervisor would
            # restart us, but heartbeat misses are themselves the signal admin
            # cares about, so just log and keep trying.
            logger.warning(f"Heartbeat write failed: {e}")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=HEARTBEAT_INTERVAL)
            break  # shutdown signaled
        except asyncio.TimeoutError:
            pass  # normal — beat again

    logger.info("Heartbeat loop stopped")


async def _supervise(name: str, coro_factory: Callable[[], Awaitable[Any]]) -> None:
    """
    Run a background loop forever, restarting it with exponential backoff
    if it crashes. Exits cleanly when shutdown_event is set.
    """
    backoff = 1.0
    max_backoff = 60.0
    while not shutdown_requested:
        try:
            logger.info(f"[supervisor] starting {name}")
            await coro_factory()
            # Coroutine returned cleanly (e.g. drained on shutdown). Exit.
            logger.info(f"[supervisor] {name} returned cleanly")
            return
        except asyncio.CancelledError:
            logger.info(f"[supervisor] {name} cancelled")
            raise
        except Exception as e:
            logger.exception(f"[supervisor] {name} crashed: {e}")
            try:
                await notify_admin_error(
                    "supervisor_loop_crash",
                    {"loop": name, "error": str(e)},
                    db_pool,
                )
            except Exception:
                pass
            # Sleep with shutdown sensitivity
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=backoff)
                return  # shutdown set during backoff
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, max_backoff)


async def main() -> None:
    global db_pool, redis_client, shutdown_event

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        from core.sentry_init import init_sentry_for_worker

        init_sentry_for_worker()
        try:
            import sentry_sdk

            from core.process_stats import render_instance_context, worker_config_snapshot

            inst = render_instance_context()
            cfg = worker_config_snapshot()
            sentry_sdk.set_tag("component", "worker")
            sentry_sdk.set_tag("worker_lane", str(cfg.get("worker_lane") or "full"))
            if inst.get("instance_id"):
                sentry_sdk.set_tag("render_instance_id", str(inst["instance_id"])[:64])
            if inst.get("service_id"):
                sentry_sdk.set_tag("render_service_id", str(inst["service_id"])[:64])
            if inst.get("git_commit"):
                sentry_sdk.set_tag("git_commit", str(inst["git_commit"])[:12])
        except Exception:
            pass
    except Exception as _sentry_e:
        logger.warning("Sentry worker init skipped: %s", _sentry_e)

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

    try:
        from services.upload_funnel import set_funnel_db_pool

        set_funnel_db_pool(db_pool)
    except Exception:
        pass

    try:
        from services.ml_model_sync import sync_ml_models_from_hub
        from services.promo_targeting_model import reload_model as reload_promo_model
        from services.content_success_model import reload_model as reload_content_model

        sync_ml_models_from_hub()
        reload_promo_model()
        reload_content_model()
    except Exception:
        pass

    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info("Redis connected")

    # After OOM/SIGKILL the process dies holding Redis INCR; DB may still say
    # processing so heal_user_process_slots cannot clamp to 0. Single-worker
    # Render boots clear orphans so the next job is not stuck in_use=1/1.
    # Opt out with USER_SLOT_CLEAR_ON_START=0 if multiple process workers share Redis.
    _clear_slots = (os.environ.get("USER_SLOT_CLEAR_ON_START") or "1").strip().lower()
    if _clear_slots in ("1", "true", "yes", "on"):
        try:
            from stages.redis_job_queue import clear_all_user_process_slots

            await clear_all_user_process_slots(redis_client)
        except Exception as _clr_e:
            logger.warning("USER_SLOT_CLEAR_ON_START failed: %s", _clr_e)

    shutdown_event = asyncio.Event()
    _ensure_worker_semaphores()

    lane = WORKER_LANE if WORKER_LANE in ("full", "process", "publish") else "full"
    from core.process_stats import render_instance_context, sample_memory_mb, worker_config_snapshot

    inst = render_instance_context()
    mem = sample_memory_mb()
    cfg = worker_config_snapshot()
    logger.info(
        "Worker starting | instance=%s service=%s region=%s commit=%s | "
        "lane=%s ASYNC_PUBLISH_QUEUE=%s | WORKER_CONCURRENCY=%s PUBLISH_CONCURRENCY=%s "
        "HEAVY_PIPELINE_SLOTS=%s | rss=%sMB limit=%sMB",
        inst.get("instance_id"),
        inst.get("service_name"),
        inst.get("region"),
        inst.get("git_commit"),
        lane,
        ASYNC_PUBLISH_QUEUE,
        WORKER_CONCURRENCY,
        PUBLISH_CONCURRENCY,
        WORKER_HEAVY_PIPELINE_SLOTS,
        mem.get("rss_mb"),
        mem.get("limit_mb"),
    )
    logger.info(
        "Worker env snapshot | profile=%s streams=%s legacy_drain=%s "
        "poll=%ss heartbeat=%ss stale_recovery=%s",
        cfg.get("worker_pipeline_profile"),
        os.environ.get("REDIS_JOB_USE_STREAMS", "false"),
        REDIS_JOB_LEGACY_DRAIN,
        POLL_INTERVAL,
        HEARTBEAT_INTERVAL,
        STALE_JOB_RECOVERY_ENABLED,
    )

    background_loops: List[Tuple[str, Any]] = [
        ("heartbeat", run_heartbeat_loop),
    ]
    if lane != "publish":
        background_loops.append(("process_jobs", process_jobs))
    if ASYNC_PUBLISH_QUEUE or lane == "publish":
        background_loops.append(("publish_jobs", publish_jobs))
    background_loops.append(("scheduler_loop", run_scheduler_loop))
    from stages.redis_job_queue import use_redis_streams

    if use_redis_streams():
        background_loops.append(("stream_reclaim", run_stream_reclaim_loop))
    if STALE_JOB_RECOVERY_ENABLED:
        background_loops.append(("stale_recovery", run_stale_job_recovery_loop))
    def _loop_enabled(name: str, default: bool = True) -> bool:
        raw = (os.environ.get(name) or "").strip().lower()
        if not raw:
            return default
        return raw in ("1", "true", "yes", "on")

    background_loops.append(
        ("verification_loop", lambda: run_verification_loop(db_pool, shutdown_event))
    )
    if _loop_enabled("WORKER_ENABLE_ANALYTICS_SYNC", True):
        background_loops.append(("analytics_sync", run_analytics_sync_loop))
    if _loop_enabled("WORKER_ENABLE_KPI_COLLECTOR", True):
        background_loops.append(("kpi_collector", run_kpi_collector_loop))
    if _loop_enabled("WORKER_ENABLE_PLATFORM_METRICS", True):
        background_loops.append(("platform_metrics_cache", run_platform_metrics_cache_loop))
    if _loop_enabled("WORKER_ENABLE_ADMIN_EMAIL_JOBS", True):
        background_loops.append(("admin_email_jobs", run_admin_email_jobs_loop))

    tasks = [
        asyncio.create_task(_supervise(name, factory))
        for name, factory in background_loops
    ]

    try:
        await notify_admin_worker_start(db_pool)
        # Wait until shutdown is signaled (SIGTERM/SIGINT). Supervised loops
        # restart themselves on failure; only an explicit shutdown ends them.
        await shutdown_event.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
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