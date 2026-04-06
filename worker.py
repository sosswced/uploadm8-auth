"""
UploadM8 Worker Service v3 — Deferred Processing + Concurrent Jobs (Full Stack)

Pipeline order (critical — DO NOT REORDER):
  1.  Download        — Fetch original video + telemetry from R2
  2.  Telemetry       — Parse .map file, calculate Trill score, reverse-geocode
  3.  HUD             — Burn speed overlay onto raw video (before transcode!)
  4.  Watermark       — Burn text watermark (before transcode!)
  5.  Transcode       — Deduplicated per-platform MP4s
  5.5 Audio Context   — Whisper + ACRCloud + YAMNet + Hume AI context
  5.6 Vision          — Google Cloud Vision face/OCR on best frame
  5.7 Twelve Labs     — Deep video understanding (if enabled)
  5.8 Video Intelligence — Google Cloud full-clip labels + shots (if enabled)
  6.  Thumbnail       — Playwright HTML + AI backgrounds + rembg + Pillow fallback
  7.  Caption         — Transcript + Hume emotion + audio keywords (context feeds thumbnail/caption/hashtag)
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
  WORKER_CONCURRENCY (default 2) — simultaneous FFmpeg-heavy process jobs.
  WORKER_HEAVY_PIPELINE_SLOTS (default 1) — jobs sharing the memory-heavy tail
    (audio/YAMNet, Vision, Twelve Labs, thumbnails/ONNX/rembg). Keeps peak RAM
    predictable on small Render instances even when WORKER_CONCURRENCY > 1.
  WORKER_PIPELINE_PROFILE — optional preset: full_safe | minimal_ram | throughput
    (applies setdefault for concurrency keys so explicit env still wins).
  Scheduler loop runs as a separate asyncio task, not a job slot.
"""

import os
import sys
import json
import asyncio
import logging
import random
import tempfile
import signal
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import asyncpg
import redis.asyncio as redis

from stages.asyncpg_json_codecs import apply_asyncpg_json_codecs as _init_asyncpg_codecs

from stages.errors import StageError, SkipStage, CancelRequested, log_stage_skip
from stages.context import JobContext, create_context
from stages.entitlements import get_entitlements_from_user, get_entitlements_for_tier
from services.wallet import partial_refund_upload_partial_success
from stages import db as db_stage
from stages import r2 as r2_stage
from stages.telemetry_stage import run_telemetry_stage
from stages.transcode_stage import (
    run_transcode_stage, PLATFORM_SPECS,
    get_video_info, needs_transcode, build_ffmpeg_command,
    resolve_reframe_action,
)
from stages.audio_stage import run_audio_context_stage
from stages.vision_stage import run_vision_stage
from stages.twelvelabs_stage import run_twelvelabs_stage
from stages.video_intelligence_stage import run_video_intelligence_stage
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
from stages.pipeline_manifest import init_pipeline_diag, diag_step, finalize_pipeline_diag
from stages.pipeline_stage_budgets import (
    stage_timeout_audio,
    stage_timeout_vision,
    stage_timeout_twelvelabs,
    stage_timeout_video_intelligence,
    stage_timeout_thumbnail,
    stage_timeout_caption,
    stage_timeout_transcode,
)
from stages.redis_job_queue import (
    DEFAULT_GROUP,
    enqueue_process_job,
    ensure_stream_group,
    list_key_from_stream,
    make_worker_consumer_name,
    process_stream_keys_ordered,
    use_redis_streams,
    user_process_release,
    user_process_try_acquire,
    xack_message,
    xautoclaim_batch,
    xreadgroup_one,
)
from stages.emails import (
    send_upload_completed_email,
    send_upload_failed_email,
    send_low_token_warning_email,
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_JSON_LOGS = os.environ.get("JSON_LOGS", "1").strip().lower() in ("1", "true", "yes")

class _JsonFormatter(logging.Formatter):
    def format(self, record):
        import json as _j
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exc"] = self.formatException(record.exc_info)
        return _j.dumps(entry, default=str, ensure_ascii=False)

_handler = logging.StreamHandler()
if _JSON_LOGS:
    _handler.setFormatter(_JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [worker] %(message)s"))
logging.root.handlers = [_handler]
logging.root.setLevel(LOG_LEVEL)
logger = logging.getLogger("uploadm8-worker")

_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.asyncpg import AsyncPGIntegration
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            environment=os.environ.get("SENTRY_ENV", "production"),
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_RATE", "0.05")),
            send_default_pii=False,
            integrations=[AsyncPGIntegration()],
        )
        logger.info("Sentry initialised (worker, env=%s)", os.environ.get("SENTRY_ENV", "production"))
    except ImportError:
        logger.warning("SENTRY_DSN set but sentry-sdk not installed — skipping")

try:
    from stages.outbound_rl import startup_log_line as _outbound_rl_startup_log_line

    logger.info(_outbound_rl_startup_log_line())
except Exception as _e:
    logger.warning("outbound_rl startup log skipped: %s", _e)

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "")

async def _acquire_cron_lock(lock_name: str, ttl_seconds: int = 300) -> bool:
    """Distributed Redis lock for cron leadership. Only one worker instance
    runs a given cron tick. Falls back to always-True when Redis is missing."""
    if not redis_client:
        return True
    try:
        acquired = await redis_client.set(f"cron_lock:{lock_name}", "1", nx=True, ex=ttl_seconds)
        return bool(acquired)
    except Exception as e:
        logger.debug("cron lock Redis error lock=%s (proceeding as leader): %s", lock_name, e)
        return True

# ── 4-Lane Queue Names (must match app.py env vars) ──────────────
PROCESS_PRIORITY_QUEUE = os.environ.get("PROCESS_PRIORITY_QUEUE", "uploadm8:process:priority")
PROCESS_NORMAL_QUEUE   = os.environ.get("PROCESS_NORMAL_QUEUE",   "uploadm8:process:normal")
PUBLISH_PRIORITY_QUEUE = os.environ.get("PUBLISH_PRIORITY_QUEUE", "uploadm8:publish:priority")
PUBLISH_NORMAL_QUEUE   = os.environ.get("PUBLISH_NORMAL_QUEUE",   "uploadm8:publish:normal")
# Legacy queue names kept so old Redis entries dont get lost during transition
UPLOAD_JOB_QUEUE   = os.environ.get("UPLOAD_JOB_QUEUE",   "uploadm8:jobs")
PRIORITY_JOB_QUEUE = os.environ.get("PRIORITY_JOB_QUEUE", "uploadm8:priority")

POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))

# ── Pipeline profile (optional) ─────────────────────────────────────
# Ship all stages (FFmpeg + audio + Vision + VI + Twelve Labs + thumbnail + caption)
# without blowing RAM: cap *parallel* jobs and *parallel* heavy tails.
# Explicit WORKER_* / PUBLISH_* env vars in Render always win (setdefault).
_PIPELINE_PROFILE = (os.environ.get("WORKER_PIPELINE_PROFILE") or "").strip().lower()
if _PIPELINE_PROFILE == "full_safe":
    os.environ.setdefault("WORKER_CONCURRENCY", "2")
    os.environ.setdefault("WORKER_HEAVY_PIPELINE_SLOTS", "1")
    os.environ.setdefault("PUBLISH_CONCURRENCY", "4")
elif _PIPELINE_PROFILE == "minimal_ram":
    os.environ.setdefault("WORKER_CONCURRENCY", "1")
    os.environ.setdefault("WORKER_HEAVY_PIPELINE_SLOTS", "1")
    os.environ.setdefault("PUBLISH_CONCURRENCY", "3")
elif _PIPELINE_PROFILE == "throughput":
    os.environ.setdefault("WORKER_CONCURRENCY", "3")
    os.environ.setdefault("WORKER_HEAVY_PIPELINE_SLOTS", "2")
    os.environ.setdefault("PUBLISH_CONCURRENCY", "6")
elif _PIPELINE_PROFILE:
    logger.warning("Unknown WORKER_PIPELINE_PROFILE=%r — ignoring", _PIPELINE_PROFILE)

# ── Concurrency ───────────────────────────────────────────────────
# WORKER_CONCURRENCY  = FFmpeg-heavy process jobs (CPU-bound; default 2 for small hosts)
# PUBLISH_CONCURRENCY = API-light publish jobs   (I/O-bound, default 5)
WORKER_CONCURRENCY  = int(os.environ.get("WORKER_CONCURRENCY",  "2"))
PUBLISH_CONCURRENCY = int(os.environ.get("PUBLISH_CONCURRENCY", "5"))
# Caps concurrent "ML + thumbnail" tails (TensorFlow YAMNet, ONNX, rembg, large API buffers).
WORKER_HEAVY_PIPELINE_SLOTS = max(1, int(os.environ.get("WORKER_HEAVY_PIPELINE_SLOTS", "1")))
if _PIPELINE_PROFILE and _PIPELINE_PROFILE in ("full_safe", "minimal_ram", "throughput"):
    logger.info(
        "WORKER_PIPELINE_PROFILE=%s → concurrency=%s heavy_slots=%s publish=%s",
        _PIPELINE_PROFILE,
        WORKER_CONCURRENCY,
        WORKER_HEAVY_PIPELINE_SLOTS,
        PUBLISH_CONCURRENCY,
    )

# Scheduler: how far in advance to start processing a scheduled upload (minutes)
PROCESSING_WINDOW_MINUTES = int(os.environ.get("PROCESSING_WINDOW_MINUTES", "15"))

# How often the scheduler polls the DB for staged/ready jobs (seconds)
SCHEDULER_POLL_INTERVAL = int(os.environ.get("SCHEDULER_POLL_INTERVAL", "60"))

# ── Analytics auto-sync ──────────────────────────────────────────
# How often the analytics sync *loop cycle* wakes (seconds). Default: 10800 = 3 hours ≈ 8×/day.
# Runs inside the worker process only: no browser, no "logged in" session — 24/7 while the worker is up.
# Per-upload work is additionally gated by ANALYTICS_RESYNC_HOURS (stale rows only).
ANALYTICS_SYNC_INTERVAL = int(os.environ.get("ANALYTICS_SYNC_INTERVAL_SECONDS", str(3 * 3600)))

# ── KPI collector ─────────────────────────────────────────────────
# How often to collect cost/revenue from Stripe, OpenAI, Mailgun, etc. (seconds).
# Default aligns with platform metrics cache cadence unless explicitly overridden.
KPI_COLLECTOR_INTERVAL = int(
    os.environ.get("KPI_COLLECTOR_INTERVAL_SECONDS", os.environ.get("PLATFORM_METRICS_CACHE_INTERVAL_SECONDS", str(3 * 3600)))
)
# How many uploads to sync per cycle (avoids platform API rate-limit hammering)
ANALYTICS_SYNC_BATCH    = int(os.environ.get("ANALYTICS_SYNC_BATCH_SIZE", "20"))
# Only re-sync uploads whose analytics_synced_at is older than this many hours
ANALYTICS_RESYNC_HOURS  = int(os.environ.get("ANALYTICS_RESYNC_HOURS", "3"))

# Per-user cooldown for TikTok/YouTube OAuth refresh inside analytics batch (avoids refresh storms
# when many upload rows share the same user_id). Default 3600s; set 0 to refresh every row (old behavior).
ANALYTICS_OAUTH_REFRESH_COOLDOWN_SEC = float(
    os.environ.get("ANALYTICS_OAUTH_REFRESH_COOLDOWN_SECONDS", "3600")
)
_analytics_oauth_last_refresh: dict[str, float] = {}

# Account-level platform_metrics_cache refresh (TikTok/YouTube/IG/FB live rollup in DB).
# Default: 10800s = 3 hours ≈ 8×/day. Server-side worker only — not tied to user login.
PLATFORM_METRICS_CACHE_INTERVAL = int(
    os.environ.get("PLATFORM_METRICS_CACHE_INTERVAL_SECONDS", str(3 * 3600))
)
ML_SCORING_INTERVAL = int(
    os.environ.get("ML_SCORING_INTERVAL_SECONDS", str(6 * 3600))
)
ML_SCORING_LOOKBACK_DAYS = int(
    os.environ.get("ML_SCORING_LOOKBACK_DAYS", "180")
)

# Redis resilience
REDIS_RETRY_DELAY = 5.0
REDIS_MAX_RETRIES = 10

# Dead-letter queue: max retries before moving to DLQ
DLQ_MAX_RETRIES = int(os.environ.get("DLQ_MAX_RETRIES", "3"))

# Overall pipeline timeout — kills a stuck job so it doesn't hog a semaphore slot forever.
# Default 30 min covers even large videos (10 min transcode + 5 min AI + 10 min publish).
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SECONDS", "1800"))

db_pool: Optional[asyncpg.Pool] = None


def _merge_upload_prefs_into_settings(user_settings: dict, upload_record: dict) -> dict:
    """
    Merge upload's user_preferences (captured at presign) into user_settings.
    Fills in platform_hashtags, always_hashtags, etc. when DB sources are empty.

    This fixes platform-specific hashtags not applying when users.preferences
    and user_preferences are empty — the presign snapshot (get_user_prefs_for_upload)
    is stored in the upload and used here.
    """
    out = dict(user_settings or {})
    raw = upload_record.get("user_preferences")
    if not raw:
        return out
    try:
        prefs = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.debug("_merge_upload_prefs_into_settings: skip invalid user_preferences: %s", e)
        return out
    if not isinstance(prefs, dict):
        return out

    def _has_content(v, is_platform_map: bool = False) -> bool:
        if v is None:
            return False
        if is_platform_map and isinstance(v, dict):
            return any(
                (isinstance(x, list) and len(x) > 0) or (isinstance(x, str) and x.strip())
                for x in (v.values() or [])
            )
        if isinstance(v, (list, tuple)):
            return len(v) > 0
        if isinstance(v, dict):
            return len(v) > 0
        return bool(v)

    # Fill in hashtag fields from presign when current is empty
    ph_up = prefs.get("platformHashtags") or prefs.get("platform_hashtags")
    ph_out = out.get("platformHashtags") or out.get("platform_hashtags")
    if _has_content(ph_up, is_platform_map=True) and not _has_content(ph_out, is_platform_map=True):
        out["platformHashtags"] = ph_up
        out["platform_hashtags"] = ph_up

    for camel, snake in [("alwaysHashtags", "always_hashtags"), ("blockedHashtags", "blocked_hashtags")]:
        val = prefs.get(camel) or prefs.get(snake)
        cur = out.get(camel) or out.get(snake)
        if _has_content(val) and not _has_content(cur):
            out[camel] = val
            out[snake] = val

    # Thumbnail Studio / Pikzels scalar controls should follow per-upload snapshot.
    for camel, snake in [
        ("thumbnailStudioEnabled", "thumbnail_studio_enabled"),
        ("thumbnailStudioEngineEnabled", "thumbnail_studio_engine_enabled"),
        ("thumbnailPikzelsEnabled", "thumbnail_pikzels_enabled"),
        ("thumbnailPersonaEnabled", "thumbnail_persona_enabled"),
        ("thumbnailDefaultPersonaId", "thumbnail_default_persona_id"),
        ("thumbnailPersonaStrength", "thumbnail_persona_strength"),
        ("thumbnailUseStudioEngine", "thumbnail_use_studio_engine"),
        ("thumbnailUsePikzels", "thumbnail_use_pikzels"),
        ("thumbnailUsePersona", "thumbnail_use_persona"),
        ("thumbnailPersonaId", "thumbnail_persona_id"),
    ]:
        val = prefs.get(camel) if prefs.get(camel) is not None else prefs.get(snake)
        if val is not None:
            out[camel] = val
            out[snake] = val

    return out

redis_client: Optional[redis.Redis] = None
# Unique per worker process — Redis Streams consumer + reclaim identity
WORKER_STREAM_CONSUMER: str = ""

# ── Wallet helpers (worker-side) ─────────────────────────────────────────────
async def _capture_tokens(upload_id: str, user_id: str, put_cost: int, aic_cost: int):
    """
    Confirm a hold: move reserved → actual spend.
    Called on successful job completion.
    """
    if not db_pool:
        return
    async with db_pool.acquire() as conn:
        urow = await conn.fetchrow("SELECT subscription_tier FROM users WHERE id = $1::uuid", user_id)
        tier = (urow.get("subscription_tier") if urow else "free") or "free"
        ent = get_entitlements_for_tier(str(tier))
        if getattr(ent, "is_internal", False):
            await conn.execute(
                """
                UPDATE wallet_holds SET status = 'captured', resolved_at = NOW()
                WHERE upload_id = $1 AND status = 'held'
                """,
                upload_id,
            )
            await conn.execute("UPDATE uploads SET hold_status = 'captured' WHERE id = $1", upload_id)
            return

        before_wallet = await conn.fetchrow(
            "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1",
            user_id,
        )
        before_put = int((before_wallet or {}).get("put_balance") or 0)
        before_aic = int((before_wallet or {}).get("aic_balance") or 0)

        # Debit balance and clear reservation simultaneously; only log ledger if update applied.
        row = await conn.fetchrow(
            """
            UPDATE wallets SET
                put_balance  = put_balance  - $1,
                aic_balance  = aic_balance  - $2,
                put_reserved = put_reserved - $1,
                aic_reserved = aic_reserved - $2,
                updated_at   = NOW()
            WHERE user_id = $3::uuid
              AND put_reserved >= $1
              AND aic_reserved >= $2
            RETURNING put_balance
            """,
            put_cost,
            aic_cost,
            user_id,
        )

        if row:
            if put_cost > 0:
                await conn.execute(
                    """
                    INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type)
                    VALUES ($1, 'put', $2, 'upload_debit', $3, 'upload')
                    """,
                    user_id,
                    -put_cost,
                    upload_id,
                )
            if aic_cost > 0:
                await conn.execute(
                    """
                    INSERT INTO token_ledger (user_id, token_type, delta, reason, upload_id, ref_type)
                    VALUES ($1, 'aic', $2, 'upload_debit', $3, 'upload')
                    """,
                    user_id,
                    -aic_cost,
                    upload_id,
                )
        else:
            logger.warning(
                "capture_tokens: wallet update skipped (no hold?) upload_id=%s user=%s put=%s aic=%s",
                upload_id,
                user_id,
                put_cost,
                aic_cost,
            )

        if row:
            await conn.execute(
                """
                UPDATE wallet_holds SET status = 'captured', resolved_at = NOW()
                WHERE upload_id = $1 AND status = 'held'
                """,
                upload_id,
            )

            await conn.execute(
                """
                UPDATE uploads SET hold_status = 'captured' WHERE id = $1
                """,
                upload_id,
            )

        # Low token warning: if balance dropped to/below threshold, email user
        LOW_THRESHOLD = 5
        wallet = await conn.fetchrow(
            "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1", user_id
        )
        if wallet and row:
            prefs = await conn.fetchrow(
                "SELECT email_notifications FROM user_preferences WHERE user_id = $1",
                user_id,
            )
            user_row = await conn.fetchrow(
                "SELECT email, name FROM users WHERE id = $1", user_id
            )
            wants_email = prefs["email_notifications"] if prefs else True
            if wants_email and user_row and user_row.get("email"):
                put_bal = int(wallet.get("put_balance") or 0)
                aic_bal = int(wallet.get("aic_balance") or 0)
                put_crossed_threshold = (put_cost > 0 and before_put > LOW_THRESHOLD and put_bal <= LOW_THRESHOLD)
                aic_crossed_threshold = (aic_cost > 0 and before_aic > LOW_THRESHOLD and aic_bal <= LOW_THRESHOLD)
                if put_crossed_threshold:
                    import asyncio as _aio
                    _aio.ensure_future(send_low_token_warning_email(
                        user_row["email"],
                        user_row.get("name") or "there",
                        "put",
                        put_bal,
                        LOW_THRESHOLD,
                    ))
                elif aic_crossed_threshold:
                    import asyncio as _aio
                    _aio.ensure_future(send_low_token_warning_email(
                        user_row["email"],
                        user_row.get("name") or "there",
                        "aic",
                        aic_bal,
                        LOW_THRESHOLD,
                    ))


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

        # No ledger: releasing a hold does not change balance (reserve had no ledger row).

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


shutdown_requested = False
shutdown_event: Optional[asyncio.Event] = None


def _build_worker_failure_diag(
    category: str,
    *,
    ctx: Optional[JobContext] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured JSON for uploads.failure_diag (support + user-visible copy)."""
    out: Dict[str, Any] = {
        "category": category,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "render_service_id": os.environ.get("RENDER_SERVICE_ID") or None,
        "render_instance_id": os.environ.get("RENDER_INSTANCE_ID") or None,
        "git_commit": os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT") or None,
    }
    if ctx is not None:
        out["pipeline_stage"] = getattr(ctx, "stage", None) or None
        out["platforms"] = list(getattr(ctx, "platforms", None) or [])
        out["filename"] = getattr(ctx, "filename", None) or None
        out["upload_id"] = str(getattr(ctx, "upload_id", "") or "")
    if extra:
        out.update(extra)
    return {k: v for k, v in out.items() if v is not None}


async def _send_to_dead_letter(
    upload_id: str,
    user_id: str,
    job_data: dict,
    error_code: str,
    error_message: str,
    retry_count: int = 0,
):
    """Move a failed job to the dead-letter queue for admin review."""
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO dead_letter_queue
                       (upload_id, user_id, job_data, error_code, error_message,
                        retry_count, max_retries, last_attempt_at)
                   VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, NOW())
                   ON CONFLICT DO NOTHING""",
                upload_id, user_id, json.dumps(job_data),
                error_code, error_message[:2000],
                retry_count, DLQ_MAX_RETRIES,
            )
        logger.warning(f"[{upload_id}] Sent to dead-letter queue: {error_code}")
    except Exception as e:
        logger.error(f"[{upload_id}] Failed to write to DLQ: {e}")


async def _mark_pipeline_uncaught_failure(upload_id: str, user_id: str, job_data: dict, exc: Exception) -> None:
    """Avoid leaving uploads stuck in processing when the worker escapes the inner pipeline try/except."""
    detail = f"{type(exc).__name__}: {exc}"[:2000]
    diag = _build_worker_failure_diag(
        "PIPELINE_EXCEPTION",
        ctx=None,
        extra={
            "upload_id": str(upload_id),
            "exception_type": type(exc).__name__,
            "hint": "Uncaught error in worker job wrapper — often memory pressure or a bug. Retry or contact support with failure_diag.",
        },
    )
    if db_pool:
        try:
            await db_stage.mark_upload_failed_diagnostic(
                db_pool,
                str(upload_id),
                "PIPELINE_EXCEPTION",
                detail,
                failure_diag=diag,
                only_if_status="processing",
            )
        except Exception as e:
            logger.warning("[%s] mark failed after PIPELINE_EXCEPTION: %s", upload_id, e)
    await _send_to_dead_letter(
        upload_id,
        user_id,
        job_data,
        "PIPELINE_EXCEPTION",
        detail,
        retry_count=job_data.get("_retry_count", 0),
    )


# ── Separate semaphores for each lane ────────────────────────────
# Process semaphore: guards FFmpeg transcode slots (CPU-bound)
# Publish semaphore: guards platform API call slots (I/O-bound)
# Keeping them separate means a 10-minute transcode CANNOT block
# a 10-second TikTok API publish call.
_process_semaphore: Optional[asyncio.Semaphore] = None
_publish_semaphore: Optional[asyncio.Semaphore] = None
# Legacy alias — some internal helpers still reference _job_semaphore
_job_semaphore: Optional[asyncio.Semaphore] = None
# Limits parallel memory-heavy stages (audio context → thumbnail), independent of transcode slots.
_heavy_pipeline_sem: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def _heavy_pipeline_slot():
    """Limit concurrent memory-heavy stages (audio/ML/thumbnail)."""
    sem = _heavy_pipeline_sem
    if sem is None:
        yield
        return
    async with sem:
        yield


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def handle_shutdown(signum, frame):
    global shutdown_requested, shutdown_event
    logger.warning(
        "Shutdown signal received (%s) — draining in-flight work. "
        "Uploads still in status=processing may be marked WORKER_ORPHANED after a heartbeat timeout "
        "when a new worker starts (see WORKER_ORPHAN_STALE_MINUTES).",
        signum,
    )
    shutdown_requested = True
    try:
        if shutdown_event is not None:
            shutdown_event.set()
    except Exception as e:
        logger.debug("handle_shutdown: could not set shutdown_event: %s", e)


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
    "transcode":  55,
    "audio":      62,
    "vision":     64,
    "twelvelabs": 65,
    "video_intelligence": 66,
    "thumbnail":  68,
    "caption":    78,
    "upload":     87,
    "publish":    96,
    "notify":     99,
}

# Shown when a stage *starts* so the queue/upload UI is not stuck on the previous
# milestone for the whole duration of FFmpeg-heavy work (watermark + transcode).
STAGE_LIVE_PROGRESS = {
    "watermark": 27,
    "transcode": 33,
    "audio":     56,
    "vision":    63,
    "twelvelabs": 64,
    "video_intelligence": 65,
    "thumbnail": 67,
    "caption":   69,
}


async def report_stage_started(ctx: JobContext, stage: str):
    """Write current stage to DB before long work; complements maybe_cancel at stage end."""
    pct = STAGE_LIVE_PROGRESS.get(stage)
    if pct is None:
        return
    await db_stage.update_stage_progress(db_pool, ctx.upload_id, stage, pct)


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

def _has_telemetry_file(ctx: JobContext) -> bool:
    return (
        ctx.local_telemetry_path is not None
        and Path(ctx.local_telemetry_path).exists()
    )


def _as_bool_pref(val, default: bool = True) -> bool:
    """Coerce DB/UI string or missing values to bool."""
    if val is None:
        return default
    if isinstance(val, str):
        return val.lower() not in ("false", "0", "no", "off", "")
    return bool(val)


def _should_run_trill(ctx: JobContext) -> bool:
    """
    Drive / .map telemetry for captions: needs a map file plus user consent.
    Aligns user_settings.telemetry_enabled (worker) with user_preferences.trill_enabled
    (POST /api/settings/preferences); either may be set — both false disables.
    """
    if not _has_telemetry_file(ctx):
        return False
    us = ctx.user_settings or {}
    te = _as_bool_pref(us.get("telemetry_enabled", True), True)
    tr = us.get("trill_enabled", us.get("trillEnabled"))
    if tr is None:
        return te
    return te or _as_bool_pref(tr, False)


def _should_run_hud(ctx: JobContext) -> bool:
    """Speed HUD overlay: needs a .map file + user HUD on + tier allows burn."""
    if not _has_telemetry_file(ctx):
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
        if (upload_record.get("status") or "").lower() == "cancelled":
            logger.info(f"[{upload_id}] Skipping — already cancelled")
            return False

        user_settings = await db_stage.load_user_settings(db_pool, user_id)
        # Merge upload's user_preferences (captured at presign) into user_settings — fixes platform_hashtags
        # when users.preferences / user_preferences are empty, so platform-specific hashtags always apply
        user_settings = _merge_upload_prefs_into_settings(user_settings, upload_record)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.user_record = user_record
        ctx.started_at = _now_utc()
        ctx.state = "processing"
        ctx.user_settings["_openai_model_override"] = (
            ctx.user_settings.get("trillOpenaiModel")
            or ctx.user_settings.get("trill_openai_model")
            or "gpt-4o"
        )

        def _svc_enabled(camel_key: str, default: bool = True) -> bool:
            """Read per-service toggles from ctx.user_settings (camel + snake aliases)."""
            us = ctx.user_settings or {}
            snake = camel_key[0].lower()
            for ch in camel_key[1:]:
                snake += ("_" + ch.lower()) if ch.isupper() else ch
            raw = us.get(camel_key, us.get(snake, default))
            return bool(raw)

        # ── Server-side entitlement cap enforcement ───────────────────────
        if ctx.user_record and ctx.entitlements:
            ent = ctx.entitlements

            if hasattr(ctx, "num_thumbnails") and ctx.num_thumbnails > ent.max_thumbnails:
                logger.info(
                    f"[{ctx.upload_id}] Clamping thumbnails {ctx.num_thumbnails} → {ent.max_thumbnails} (tier cap)"
                )
                ctx.num_thumbnails = ent.max_thumbnails

            if hasattr(ctx, "caption_frames") and ctx.caption_frames > ent.max_caption_frames:
                ctx.caption_frames = ent.max_caption_frames

            if not ent.can_burn_hud and hasattr(ctx, "hud_enabled"):
                ctx.hud_enabled = False

            if ent.can_watermark and hasattr(ctx, "watermark_text"):
                if not ctx.watermark_text:
                    ctx.watermark_text = "UploadM8"

            if not ent.can_ai and hasattr(ctx, "use_ai"):
                ctx.use_ai = False

        ctx.redis_client = redis_client

        transitioned = await db_stage.mark_processing_started_if_status_in(
            db_pool, ctx, ("queued", "pending")
        )
        if not transitioned:
            ur = await db_stage.load_upload_record(db_pool, upload_id)
            st = ((ur or {}).get("status") or "").lower()
            if st in ("completed", "succeeded", "partial", "failed", "cancelled", "ready_to_publish"):
                logger.info("[%s] Skip job — already terminal (%s)", upload_id, st)
                return True
            if st == "processing":
                logger.info("[%s] Skip job — already processing (at-least-once dedupe)", upload_id, st)
                return True
            logger.warning("[%s] Skip job — unexpected status %s (not queued/pending)", upload_id, st)
            return True

        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE uploads SET processing_started_at = NOW() WHERE id = $1",
                    upload_id,
                )
        except Exception as e:
            logger.debug("[%s] processing_started_at timestamp update skipped: %s", upload_id, e)

        init_pipeline_diag(ctx, upload_record, is_deferred=is_deferred)
        diag_step(ctx, stage="pipeline", status="started", provider="uploadm8_worker")

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

        trill_want = _should_run_trill(ctx)
        hud_active = _should_run_hud(ctx)
        svc_telemetry = _svc_enabled("aiServiceTelemetry", True)
        # HUD still needs map parse; caption-side Trill only when drive + Telemetry Insights on.
        run_telemetry_stage_flag = hud_active or (trill_want and svc_telemetry)
        trill_for_captions = bool(trill_want and svc_telemetry)
        setattr(ctx, "skip_trill_caption_injection", not trill_for_captions)

        logger.info(
            f"[{upload_id}] Flags: trill_for_captions={trill_for_captions} "
            f"trill_want={trill_want} hud={hud_active} ai_svc_telemetry={svc_telemetry} "
            f"platforms={ctx.platforms}"
        )
        diag_step(ctx, stage="download", status="ok", provider="cloudflare_r2", extra={"telemetry_map": bool(ctx.local_telemetry_path)})
        await maybe_cancel(ctx, "download")

        # ============================================================
        # STAGE 2: Telemetry
        # ============================================================
        if run_telemetry_stage_flag:
            try:
                ctx = await run_telemetry_stage(ctx)
                if trill_for_captions:
                    _apply_trill_caption_settings(ctx)
            except SkipStage as e:
                log_stage_skip(logger, "Telemetry", e.reason, upload_id=upload_id)
                diag_step(ctx, stage="telemetry", status="skipped", provider="internal", reason=str(e.reason or e))
                hud_active = False
            except StageError as e:
                logger.warning(f"[{upload_id}] Telemetry error: {e.message}")
                diag_step(ctx, stage="telemetry", status="failed", provider="internal", reason=str(e.message)[:200])
                hud_active = False
            else:
                diag_step(
                    ctx,
                    stage="telemetry",
                    status="ok",
                    provider="internal",
                    extra={"trill_captions": trill_for_captions, "hud": hud_active},
                )
        else:
            ctx.telemetry = None
            ctx.telemetry_data = None
            ctx.trill = None
            ctx.trill_score = None
            diag_step(
                ctx,
                stage="telemetry",
                status="skipped",
                provider="internal",
                reason="no_telemetry_or_disabled",
            )
        await maybe_cancel(ctx, "telemetry")

        # ============================================================
        # STAGE 3: HUD
        # ============================================================
        if hud_active:
            try:
                ctx = await run_hud_stage(ctx)
            except SkipStage as e:
                log_stage_skip(logger, "HUD", e.reason, upload_id=upload_id)
                diag_step(ctx, stage="hud", status="skipped", provider="ffmpeg", reason=str(e.reason or e))
            except StageError as e:
                logger.warning(f"[{upload_id}] HUD error: {e.message}")
                diag_step(ctx, stage="hud", status="failed", provider="ffmpeg", reason=str(e.message)[:200])
            else:
                diag_step(ctx, stage="hud", status="ok", provider="ffmpeg")
        else:
            diag_step(ctx, stage="hud", status="skipped", provider="ffmpeg", reason="no_telemetry_or_disabled")
        await maybe_cancel(ctx, "hud")

        # ============================================================
        # STAGE 4: Watermark
        # ============================================================
        if not (ctx.entitlements and not ctx.entitlements.can_watermark):
            await report_stage_started(ctx, "watermark")
        try:
            ctx = await run_watermark_stage(ctx)
        except SkipStage as e:
            log_stage_skip(logger, "Watermark", e.reason, upload_id=upload_id)
            diag_step(ctx, stage="watermark", status="skipped", provider="ffmpeg", reason=str(e.reason or e))
        except StageError as e:
            logger.warning(f"[{upload_id}] Watermark error: {e.message}")
            diag_step(ctx, stage="watermark", status="failed", provider="ffmpeg", reason=str(e.message)[:200])
        else:
            if ctx.entitlements and not ctx.entitlements.can_watermark:
                diag_step(ctx, stage="watermark", status="skipped", provider="ffmpeg", reason="tier_no_watermark")
            else:
                diag_step(ctx, stage="watermark", status="ok", provider="ffmpeg")
        await maybe_cancel(ctx, "watermark")

        # ============================================================
        # STAGE 5: Transcode (deduplicated)
        # ============================================================
        await report_stage_started(ctx, "transcode")
        _tt_budget = int(stage_timeout_transcode())
        try:
            if _tt_budget > 0:
                ctx = await asyncio.wait_for(
                    _run_deduplicated_transcode(ctx),
                    timeout=float(_tt_budget),
                )
            else:
                ctx = await _run_deduplicated_transcode(ctx)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Transcode exceeded %ss — using source video for platforms",
                upload_id,
                _tt_budget,
            )
            source = ctx.processed_video_path or ctx.local_video_path
            for p in (ctx.platforms or []):
                ctx.platform_videos[p] = source
            diag_step(
                ctx,
                stage="transcode",
                status="skipped",
                provider="ffmpeg",
                reason=f"stage_timeout_{_tt_budget}s",
            )
        except SkipStage as e:
            log_stage_skip(logger, "Transcode", e.reason, upload_id=upload_id)
            diag_step(ctx, stage="transcode", status="skipped", provider="ffmpeg", reason=str(e.reason or e))
        except StageError as e:
            logger.warning(f"[{upload_id}] Transcode error: {e.message}")
            source = ctx.processed_video_path or ctx.local_video_path
            for p in (ctx.platforms or []):
                ctx.platform_videos[p] = source
            diag_step(ctx, stage="transcode", status="failed", provider="ffmpeg", reason=str(e.message)[:200])
        else:
            nv = len(getattr(ctx, "platform_videos", None) or {})
            diag_step(
                ctx,
                stage="transcode",
                status="ok" if nv else "partial",
                provider="ffmpeg",
                extra={"platform_outputs": nv},
            )
        await maybe_cancel(ctx, "transcode")

        async with _heavy_pipeline_slot():
            # ============================================================
            # STAGE 5.5: Audio Context — Whisper + ACRCloud + YAMNet + Hume
            # Runs AFTER transcode so ctx.video_info.audio_codec is populated.
            # Runs BEFORE thumbnail so ctx.audio_context feeds thumbnail/caption.
            # ============================================================
            await report_stage_started(ctx, "audio")
            _aud_b = stage_timeout_audio()
            try:
                if _aud_b > 0:
                    ctx = await asyncio.wait_for(run_audio_context_stage(ctx), timeout=_aud_b)
                else:
                    ctx = await run_audio_context_stage(ctx)
                logger.info(
                    f"[{upload_id}] Audio  — category={ctx.get_audio_category()} "
                    f"emotion={ctx.get_audio_emotion()} mood={ctx.get_thumbnail_mood()}"
                )
            except asyncio.TimeoutError:
                ctx.audio_context = {}
                diag_step(
                    ctx,
                    stage="audio",
                    status="skipped",
                    provider="audio_context",
                    reason=f"stage_timeout_{int(_aud_b)}s",
                )
                logger.warning("[%s] Audio context timed out after %ss — continuing", upload_id, int(_aud_b))
            except SkipStage as e:
                log_stage_skip(logger, "Audio context", e.reason, upload_id=upload_id)
                ctx.audio_context = {}
                diag_step(ctx, stage="audio", status="skipped", provider="audio_context", reason=str(e.reason or e))
            except StageError as e:
                logger.warning(f"[{upload_id}] Audio context error: {e.message}")
                ctx.audio_context = {}
                diag_step(ctx, stage="audio", status="failed", provider="audio_context", reason=str(e.message)[:200])
            except Exception as e:
                logger.warning(f"[{upload_id}] Audio error (non-fatal): {e}")
                ctx.audio_context = {}
                diag_step(ctx, stage="audio", status="failed", provider="audio_context", reason=str(e)[:200])
            else:
                ac = getattr(ctx, "audio_context", None) or {}
                keys = list(ac.keys())[:20] if isinstance(ac, dict) else []
                diag_step(
                    ctx,
                    stage="audio",
                    status="ok" if keys else "partial",
                    provider="audio_context",
                    extra={"context_keys": keys},
                )
            await maybe_cancel(ctx, "audio")

            # ============================================================
            # STAGE 5.6: Vision — Google Cloud Vision face/OCR on best frame
            # Context feeds thumbnail (face-priority crop) + caption (OCR).
            # ============================================================
            await report_stage_started(ctx, "vision")
            _vis_b = stage_timeout_vision()
            try:
                if _svc_enabled("aiServiceFrameInspector", True):
                    if _vis_b > 0:
                        ctx = await asyncio.wait_for(run_vision_stage(ctx), timeout=_vis_b)
                    else:
                        ctx = await run_vision_stage(ctx)
                else:
                    raise SkipStage("Frame inspector disabled by user preference")
            except asyncio.TimeoutError:
                ctx.vision_context = {}
                diag_step(
                    ctx,
                    stage="vision",
                    status="skipped",
                    provider="google_cloud_vision",
                    reason=f"stage_timeout_{int(_vis_b)}s",
                )
                logger.warning("[%s] Vision timed out after %ss — continuing", upload_id, int(_vis_b))
            except SkipStage as e:
                log_stage_skip(logger, "Vision", e.reason, upload_id=upload_id)
                ctx.vision_context = {}
                diag_step(ctx, stage="vision", status="skipped", provider="google_cloud_vision", reason=str(e.reason or e))
            except Exception as e:
                logger.warning(f"[{upload_id}] Vision error (non-fatal): {e}")
                ctx.vision_context = {}
                diag_step(ctx, stage="vision", status="failed", provider="google_cloud_vision", reason=str(e)[:200])
            else:
                vc = getattr(ctx, "vision_context", None) or {}
                has_data = bool(vc) and not (isinstance(vc, dict) and vc.get("skipped"))
                diag_step(
                    ctx,
                    stage="vision",
                    status="ok" if has_data else "partial",
                    provider="google_cloud_vision",
                )
            await maybe_cancel(ctx, "vision")

            # ============================================================
            # STAGE 5.7 + 5.8: Twelve Labs + Google Video Intelligence (parallel)
            # Independent cloud jobs — running together cuts wall time when both run.
            # ============================================================
            async def _twelvelabs_branch() -> None:
                _tl_b = stage_timeout_twelvelabs()
                try:
                    if _svc_enabled("aiServiceSceneUnderstanding", True):
                        if _tl_b > 0:
                            await asyncio.wait_for(run_twelvelabs_stage(ctx), timeout=_tl_b)
                        else:
                            await run_twelvelabs_stage(ctx)
                    else:
                        raise SkipStage("Scene understanding disabled by user preference")
                except asyncio.TimeoutError:
                    ctx.video_understanding = {}
                    diag_step(
                        ctx,
                        stage="twelvelabs",
                        status="skipped",
                        provider="twelve_labs_api",
                        reason=f"stage_timeout_{int(_tl_b)}s",
                    )
                    logger.warning("[%s] Twelve Labs timed out after %ss", upload_id, int(_tl_b))
                except SkipStage as e:
                    log_stage_skip(logger, "Twelve Labs", e.reason, upload_id=upload_id)
                    ctx.video_understanding = {}
                except Exception as e:
                    logger.warning(f"[{upload_id}] Twelve Labs error (non-fatal): {e}")
                    ctx.video_understanding = {}

            async def _video_intelligence_branch() -> None:
                _vi_b = stage_timeout_video_intelligence()
                try:
                    if _svc_enabled("aiServiceVideoAnalyzer", True):
                        if _vi_b > 0:
                            await asyncio.wait_for(run_video_intelligence_stage(ctx), timeout=_vi_b)
                        else:
                            await run_video_intelligence_stage(ctx)
                    else:
                        raise SkipStage("Video analyzer disabled by user preference")
                except asyncio.TimeoutError:
                    ctx.video_intelligence_context = ctx.video_intelligence_context or {}
                    diag_step(
                        ctx,
                        stage="video_intelligence",
                        status="skipped",
                        provider="google_video_intelligence",
                        reason=f"stage_timeout_{int(_vi_b)}s",
                    )
                    logger.warning("[%s] Video Intelligence timed out after %ss", upload_id, int(_vi_b))
                except SkipStage as e:
                    log_stage_skip(logger, "Video Intelligence", e.reason, upload_id=upload_id)
                    ctx.video_intelligence_context = ctx.video_intelligence_context or {}
                except Exception as e:
                    logger.warning(f"[{upload_id}] Video Intelligence error (non-fatal): {e}")
                    ctx.video_intelligence_context = ctx.video_intelligence_context or {}

            _tl_on = _svc_enabled("aiServiceSceneUnderstanding", True)
            _vi_on = _svc_enabled("aiServiceVideoAnalyzer", True)
            if _vi_on:
                await report_stage_started(ctx, "video_intelligence")
            elif _tl_on:
                await report_stage_started(ctx, "twelvelabs")
            await asyncio.gather(_twelvelabs_branch(), _video_intelligence_branch())

            vu = getattr(ctx, "video_understanding", None) or {}
            if not _tl_on:
                diag_step(ctx, stage="twelvelabs", status="skipped", provider="twelve_labs_api", reason="user_pref_off")
            elif isinstance(vu, dict) and vu.get("error"):
                diag_step(
                    ctx,
                    stage="twelvelabs",
                    status="failed",
                    provider="twelve_labs_api",
                    reason=str(vu.get("error"))[:200],
                )
            elif vu and (vu.get("scene_description") or vu.get("video_id") or vu.get("summary")):
                diag_step(ctx, stage="twelvelabs", status="ok", provider="twelve_labs_api")
            else:
                diag_step(
                    ctx,
                    stage="twelvelabs",
                    status="partial",
                    provider="twelve_labs_api",
                    reason="empty_or_unknown",
                )

            vi_ctx = getattr(ctx, "video_intelligence_context", None) or {}
            if not _vi_on:
                diag_step(
                    ctx,
                    stage="video_intelligence",
                    status="skipped",
                    provider="google_video_intelligence",
                    reason="user_pref_off",
                )
            elif isinstance(vi_ctx, dict) and vi_ctx.get("error"):
                diag_step(
                    ctx,
                    stage="video_intelligence",
                    status="failed",
                    provider="google_video_intelligence",
                    reason=str(vi_ctx.get("error"))[:200],
                )
            elif vi_ctx and (vi_ctx.get("top_labels") or vi_ctx.get("segment_labels")):
                diag_step(ctx, stage="video_intelligence", status="ok", provider="google_video_intelligence")
            else:
                diag_step(
                    ctx,
                    stage="video_intelligence",
                    status="partial",
                    provider="google_video_intelligence",
                    reason="empty_or_unknown",
                )

            await maybe_cancel(ctx, "twelvelabs")
            await maybe_cancel(ctx, "video_intelligence")
    
            # ============================================================
            # STAGE 6: Thumbnail — extract frame then immediately upload to R2
            # ============================================================
            try:
                recent_style_sigs: Dict[str, List[str]] = {}
                recent_style_packs: Dict[str, List[str]] = {}
                for pl in ("youtube", "instagram", "facebook"):
                    if pl not in [str(x).lower() for x in (ctx.platforms or [])]:
                        continue
                    recent_style_sigs[pl] = await db_stage.fetch_recent_thumbnail_style_signatures(
                        db_pool,
                        str(ctx.user_id),
                        pl,
                        limit=30,
                    )
                    hist = await db_stage.fetch_recent_thumbnail_style_history(
                        db_pool,
                        str(ctx.user_id),
                        pl,
                        limit=30,
                    )
                    recent_style_packs[pl] = [
                        str(x.get("style_pack") or "").lower()
                        for x in (hist or [])
                        if isinstance(x, dict) and x.get("style_pack")
                    ]
                if recent_style_sigs:
                    ctx.output_artifacts["_recent_thumbnail_style_signatures"] = json.dumps(recent_style_sigs)
                if recent_style_packs:
                    ctx.output_artifacts["_recent_thumbnail_style_packs"] = json.dumps(recent_style_packs)
            except Exception as e:
                logger.debug(f"[{upload_id}] Could not load recent thumbnail style signatures: {e}")
    
            await report_stage_started(ctx, "thumbnail")
            _th_b = stage_timeout_thumbnail()
            try:
                if _svc_enabled("aiServiceThumbnailDesigner", True):
                    if _th_b > 0:
                        ctx = await asyncio.wait_for(
                            run_thumbnail_stage(ctx, db_pool=db_pool),
                            timeout=_th_b,
                        )
                    else:
                        ctx = await run_thumbnail_stage(ctx, db_pool=db_pool)
                else:
                    raise SkipStage("Thumbnail designer disabled by user preference")
            except asyncio.TimeoutError:
                logger.warning("[%s] Thumbnail stage timed out after %ss — continuing", upload_id, int(_th_b))
                diag_step(
                    ctx,
                    stage="thumbnail",
                    status="skipped",
                    provider="thumbnail_pipeline",
                    reason=f"stage_timeout_{int(_th_b)}s",
                )
            except SkipStage as e:
                log_stage_skip(logger, "Thumbnail", e.reason, upload_id=upload_id)
            except StageError as e:
                logger.warning(f"[{upload_id}] Thumbnail error: {e.message}")
            except Exception as e:
                logger.warning(f"[{upload_id}] Thumbnail error (non-fatal): {e}")

            # ── Upload the best thumbnail to R2 and record its key ─────────────
            if ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
                thumb_r2_key = f"thumbnails/{ctx.user_id}/{upload_id}/thumbnail.jpg"
                try:
                    await r2_stage.upload_file(
                        Path(ctx.thumbnail_path),
                        thumb_r2_key,
                        "image/jpeg",
                    )
                    ctx.thumbnail_r2_key = thumb_r2_key
                    async with db_pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE uploads SET thumbnail_r2_key = $1, updated_at = NOW() WHERE id = $2",
                            thumb_r2_key,
                            upload_id,
                        )
                    logger.info(f"[{upload_id}] Thumbnail uploaded to R2: {thumb_r2_key}")
                except Exception as e:
                    logger.warning(f"[{upload_id}] Thumbnail R2 upload failed (non-fatal): {e}")
    
            # ── Upload platform-specific styled thumbnails to R2 ──────────────
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
                    await r2_stage.upload_file(Path(local_path), r2_key, "image/jpeg")
                    platform_thumb_r2[platform] = r2_key
                    logger.debug(f"[{upload_id}] Platform thumb {platform} → {r2_key}")
                except Exception as e:
                    logger.debug(f"[{upload_id}] Platform thumb {platform} upload failed: {e}")
            if platform_thumb_r2:
                ctx.output_artifacts["platform_thumbnail_r2_keys"] = json.dumps(platform_thumb_r2)
                # Persist style signatures for anti-repeat memory.
                try:
                    raw_style = ctx.output_artifacts.get("thumbnail_style_signatures", "{}")
                    style_map = json.loads(raw_style) if isinstance(raw_style, str) else (raw_style or {})
                    for pl, md in (style_map or {}).items():
                        if pl not in platform_thumb_r2:
                            continue
                        if not isinstance(md, dict):
                            continue
                        await db_stage.insert_thumbnail_style_signature(
                            db_pool,
                            user_id=str(ctx.user_id),
                            upload_id=str(upload_id),
                            platform=str(pl).lower(),
                            style_signature=str(md.get("signature") or ""),
                            style_pack=str(md.get("style_pack") or ""),
                            score=float(md.get("score") or 0.0),
                        )
                except Exception as e:
                    logger.debug(f"[{upload_id}] Could not persist thumbnail style memory: {e}")
    
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
    
            # YouTube Test & Compare — extra JPEGs for Studio (public API has no experiment resource)
            ab_raw = ctx.output_artifacts.get("youtube_thumbnail_ab_candidates", "")
            try:
                ab_list = json.loads(ab_raw) if isinstance(ab_raw, str) else (ab_raw or [])
            except Exception:
                ab_list = []
            if ab_list:
                ab_r2: Dict[str, str] = {}
                for i, row in enumerate(ab_list):
                    if not isinstance(row, dict):
                        continue
                    p = row.get("path")
                    if not p or not Path(p).exists():
                        continue
                    ab_key = f"thumbnails/{ctx.user_id}/{upload_id}/youtube_ab_{i}.jpg"
                    try:
                        await r2_stage.upload_file(Path(p), ab_key, "image/jpeg")
                        ab_r2[str(row.get("label") or f"B{i + 1}")] = ab_key
                        logger.info(f"[{upload_id}] YouTube AB variant {i} → {ab_key}")
                    except Exception as e:
                        logger.debug(f"[{upload_id}] YouTube AB variant {i} upload failed: {e}")
                if ab_r2:
                    ctx.output_artifacts["youtube_thumbnail_ab_r2_keys"] = json.dumps(ab_r2)

            _thumb_svc = _svc_enabled("aiServiceThumbnailDesigner", True)
            if not _thumb_svc:
                diag_step(ctx, stage="thumbnail", status="skipped", provider="thumbnail_pipeline", reason="user_pref_off")
            elif ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
                diag_step(
                    ctx,
                    stage="thumbnail",
                    status="ok",
                    provider="thumbnail_pipeline",
                    extra={"r2": bool(ctx.thumbnail_r2_key)},
                )
            else:
                diag_step(
                    ctx,
                    stage="thumbnail",
                    status="skipped",
                    provider="thumbnail_pipeline",
                    reason="no_thumbnail_generated",
                )

            await maybe_cancel(ctx, "thumbnail")

        # ============================================================
        # STAGE 7: Caption
        # ============================================================
        await report_stage_started(ctx, "caption")
        _cap_b = stage_timeout_caption()
        try:
            if _svc_enabled("aiServiceCaptionWriter", True):
                if _cap_b > 0:
                    ctx = await asyncio.wait_for(run_caption_stage(ctx, db_pool), timeout=_cap_b)
                else:
                    ctx = await run_caption_stage(ctx, db_pool)
                await db_stage.save_generated_metadata(db_pool, ctx)
            else:
                raise SkipStage("Caption writer disabled by user preference")
        except asyncio.TimeoutError:
            logger.warning("[%s] Caption stage timed out after %ss — continuing", upload_id, int(_cap_b))
            diag_step(
                ctx,
                stage="caption",
                status="skipped",
                provider="openai",
                reason=f"stage_timeout_{int(_cap_b)}s",
            )
        except SkipStage as e:
            log_stage_skip(logger, "Caption", e.reason, upload_id=upload_id)
            diag_step(ctx, stage="caption", status="skipped", provider="openai", reason=str(e.reason or e))
        except StageError as e:
            logger.warning(f"[{upload_id}] Caption error: {e.message}")
            diag_step(ctx, stage="caption", status="failed", provider="openai", reason=str(e.message)[:200])
        except Exception as e:
            logger.warning(f"[{upload_id}] Caption error (non-fatal): {e}")
            diag_step(ctx, stage="caption", status="failed", provider="openai", reason=str(e)[:200])
        else:
            diag_step(ctx, stage="caption", status="ok", provider="openai")
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

        if platform_thumb_r2:
            for k, v in platform_thumb_r2.items():
                processed_assets[f"thumb_{k}"] = v

        ctx.output_artifacts["processed_assets"] = json.dumps(processed_assets)
        ctx.output_artifacts["processed_video"] = ctx.processed_r2_key or ""

        try:
            await db_stage.save_processed_assets(db_pool, ctx.upload_id, processed_assets)
        except Exception as e:
            logger.warning(f"[{upload_id}] Could not persist processed_assets: {e}")

        diag_step(
            ctx,
            stage="processed_r2_upload",
            status="ok" if processed_assets else "partial",
            provider="cloudflare_r2",
            extra={"keys": list(processed_assets.keys())[:24]},
        )

        await maybe_cancel(ctx, "upload")

        # ============================================================
        # DEFERRED PATH: mark ready_to_publish and STOP
        # ============================================================
        if is_deferred:
            try:
                pm = finalize_pipeline_diag(ctx, terminal_status="ready_to_publish")
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status = 'ready_to_publish',
                            ready_to_publish_at = NOW(),
                            processed_assets = $2,
                            pipeline_manifest = $3::jsonb,
                            updated_at = NOW()
                        WHERE id = $1
                        """,
                        upload_id,
                        json.dumps(processed_assets),
                        json.dumps(pm),
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
        try:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key FROM uploads WHERE id = $1",
                    upload_id,
                )
                if row:
                    for col in ("r2_key", "telemetry_r2_key", "processed_r2_key", "thumbnail_r2_key"):
                        k = row.get(col)
                        if k:
                            try:
                                await r2_stage.delete_file(k)
                            except Exception as del_err:
                                logger.warning(f"[{upload_id}] R2 delete {col} failed: {del_err}")
        except Exception as r2_err:
            logger.warning(f"[{upload_id}] R2 cleanup on cancel failed: {r2_err}")
        try:
            put_cost, aic_cost = await _get_upload_costs(ctx.upload_id)
            user_id_str = str(ctx.user_id) if ctx.user_id else None
            if user_id_str and (put_cost > 0 or aic_cost > 0):
                await _release_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost, reason="upload_failed_refund")
        except Exception as wallet_err:
            logger.error(f"[{ctx.upload_id}] Wallet release failed: {wallet_err}")
        return False
    except Exception as e:
        logger.exception(f"[{upload_id}] Processing failed: {e}")
        if ctx:
            ctx.mark_error("INTERNAL", str(e))
            ctx.failure_diag = _build_worker_failure_diag(
                "PIPELINE_INTERNAL",
                ctx=ctx,
                extra={
                    "exception_type": type(e).__name__,
                    "hint": "Processing raised before publish — see error_detail. Retry the upload or share failure_diag with support.",
                },
            )
            await db_stage.mark_processing_failed(db_pool, ctx, "INTERNAL", str(e))
        _uid_for_release = str(ctx.user_id) if ctx else user_id
        _upid_for_release = ctx.upload_id if ctx else upload_id
        try:
            put_cost, aic_cost = await _get_upload_costs(_upid_for_release)
            if _uid_for_release and (put_cost > 0 or aic_cost > 0):
                await _release_tokens(_upid_for_release, _uid_for_release, put_cost, aic_cost, reason="upload_failed_refund")
        except Exception as wallet_err:
            logger.error(f"[{upload_id}] Wallet release failed: {wallet_err}")
        await notify_admin_error("pipeline_failure", {"upload_id": upload_id, "error": str(e)}, db_pool)
        await _send_to_dead_letter(
            upload_id, user_id, job_data, "INTERNAL", str(e)[:1000],
            retry_count=job_data.get("_retry_count", 0),
        )
        return False
    finally:
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.debug("[%s] temp_dir cleanup after pipeline: %s", upload_id, e)


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
        for pr in ctx.platform_results or []:
            reason = (getattr(pr, "error_code", None) or getattr(pr, "error_message", None) or "")[:200]
            diag_step(
                ctx,
                stage="publish",
                status="ok" if pr.success else "failed",
                provider=str(pr.platform or "unknown"),
                reason=reason,
                extra={"account": getattr(pr, "account_username", None)},
            )
    except StageError as e:
        logger.error(f"[{upload_id}] Publish error: {e.message}")
        ctx.mark_error(e.code.value, e.message)
        ec = e.code.value if hasattr(e.code, "value") else str(e.code)
        ctx.failure_diag = _build_worker_failure_diag(
            "PUBLISH_STAGE_ERROR",
            ctx=ctx,
            extra={"publish_error_code": ec, "publish_stage_detail": (e.detail or "")[:500] or None},
        )
        diag_step(ctx, stage="publish", status="failed", provider="platform_apis", reason=str(e.message)[:200])

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

    if ctx.state == "failed" and not ctx.platform_results:
        if not (ctx.error_message or "").strip():
            ctx.error_message = (
                "Publish did not produce any per-platform results (nothing was posted). "
                "Confirm destinations and connected accounts, or retry if the worker restarted during processing."
            )
        base = _build_worker_failure_diag(
            "PUBLISH_EMPTY_RESULTS",
            ctx=ctx,
            extra={
                "target_accounts": list(ctx.target_accounts or []),
                "hint": "No PlatformResult rows — often worker restart before publish, missing video file, or no valid OAuth targets.",
            },
        )
        prev = ctx.failure_diag if isinstance(ctx.failure_diag, dict) else {}
        ctx.failure_diag = {**base, **prev}

    ctx.pipeline_manifest_final = finalize_pipeline_diag(ctx, terminal_status=ctx.state)
    await db_stage.mark_processing_completed(db_pool, ctx)

    # ── Upload email notification ──────────────────────────────────────────
    try:
        _u = ctx.user_record
        _user_email = _u.get("email") if _u else None
        _user_name  = (_u.get("name") if _u else None) or "there"
        if _user_email:
            async with db_pool.acquire() as _pconn:
                _prefs = await _pconn.fetchrow(
                    "SELECT email_notifications FROM user_preferences WHERE user_id = $1",
                    ctx.user_id,
                )
            _wants_email = _prefs["email_notifications"] if _prefs else True
            if _wants_email:
                import asyncio as _aio
                _platforms = ctx.platforms or []
                _put_cost, _aic_cost = await _get_upload_costs(ctx.upload_id)
                if ctx.state in ("succeeded", "partial"):
                    _effective_title = (
                        ctx.get_effective_title("") if hasattr(ctx, "get_effective_title") else None
                    )
                    _effective_caption = (
                        ctx.get_effective_caption("") if hasattr(ctx, "get_effective_caption") else None
                    )
                    _video_title = (
                        _effective_title
                        or getattr(ctx, "title", None)
                        or getattr(ctx, "ai_title", None)
                        or getattr(ctx, "title", None)
                        or ctx.filename
                        or upload_id
                        or "Untitled"
                    )
                    _video_caption = (
                        _effective_caption
                        or getattr(ctx, "caption", None)
                        or getattr(ctx, "ai_caption", None)
                        or getattr(ctx, "caption", None)
                        or ""
                    )
                    _video_hashtags = (
                        getattr(ctx, "ai_hashtags", None)
                        or getattr(ctx, "hashtags", None)
                        or []
                    )
                    _platform_results = getattr(ctx, "platform_results", None) or []
                    _aio.ensure_future(send_upload_completed_email(
                        _user_email,
                        _user_name,
                        ctx.filename or upload_id,
                        ctx.get_success_platforms() or _platforms,
                        int(_put_cost or 0),
                        int(_aic_cost or 0),
                        str(upload_id),
                        video_title=_video_title,
                        video_caption=_video_caption,
                        video_hashtags=_video_hashtags,
                        platform_results=_platform_results,
                    ))
                elif ctx.state == "failed":
                    _err_reason = getattr(ctx, "error_message", "") or ""
                    _err_stage  = getattr(ctx, "current_stage", "") or ""
                    _effective_title = (
                        ctx.get_effective_title("") if hasattr(ctx, "get_effective_title") else None
                    )
                    _effective_caption = (
                        ctx.get_effective_caption("") if hasattr(ctx, "get_effective_caption") else None
                    )
                    _video_title = (
                        _effective_title
                        or getattr(ctx, "title", None)
                        or getattr(ctx, "ai_title", None)
                        or getattr(ctx, "title", None)
                        or ctx.filename
                        or upload_id
                        or "Untitled"
                    )
                    _video_caption = (
                        _effective_caption
                        or getattr(ctx, "caption", None)
                        or getattr(ctx, "ai_caption", None)
                        or getattr(ctx, "caption", None)
                        or ""
                    )
                    _video_hashtags = (
                        getattr(ctx, "ai_hashtags", None)
                        or getattr(ctx, "hashtags", None)
                        or []
                    )
                    _platform_results = getattr(ctx, "platform_results", None) or []
                    _aio.ensure_future(send_upload_failed_email(
                        _user_email,
                        _user_name,
                        ctx.filename or upload_id,
                        _platforms,
                        _err_reason,
                        str(upload_id),
                        _err_stage,
                        video_title=_video_title,
                        video_caption=_video_caption,
                        video_hashtags=_video_hashtags,
                        platform_results=_platform_results,
                    ))
    except Exception as _email_err:
        logger.warning(f"[{upload_id}] Upload email notification failed (non-fatal): {_email_err}")

    # ── Finalize wallet hold ───────────────────────────────────────────────
    try:
        put_cost, aic_cost = await _get_upload_costs(ctx.upload_id)
        user_id_str = str(ctx.user_id) if ctx.user_id else None
        if user_id_str and (put_cost > 0 or aic_cost > 0):
            if ctx.is_success():
                await _capture_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost)

            elif ctx.is_partial_success():
                await _capture_tokens(ctx.upload_id, user_id_str, put_cost, aic_cost)
                async with db_pool.acquire() as _wconn:
                    await partial_refund_upload_partial_success(
                        _wconn,
                        user_id=user_id_str,
                        upload_id=ctx.upload_id,
                        succeeded_platforms=ctx.get_success_platforms(),
                        failed_platforms=ctx.get_failed_platforms(),
                        original_put_cost=put_cost,
                        original_aic_cost=aic_cost,
                    )

            else:
                await _release_tokens(
                    ctx.upload_id,
                    user_id_str,
                    put_cost,
                    aic_cost,
                    reason="upload_failed_refund",
                )

    except Exception as wallet_err:
        logger.error(f"[{ctx.upload_id}] Wallet finalize failed (non-fatal): {wallet_err}")

    if ctx.is_success():
        await db_stage.increment_upload_count(db_pool, user_id)

        elapsed = (ctx.finished_at - ctx.started_at).total_seconds() if ctx.started_at else 0
        logger.info(
            f"[{upload_id}] Pipeline {ctx.state} in {elapsed:.1f}s | "
            f"ok={ctx.get_success_platforms()} fail={ctx.get_failed_platforms()} | "
            f"category={ctx.get_audio_category()} mood={ctx.get_thumbnail_mood()}"
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
        user_settings = _merge_upload_prefs_into_settings(user_settings, upload_record)
        overrides = await db_stage.load_user_entitlement_overrides(db_pool, user_id)
        entitlements = get_entitlements_from_user(user_record, overrides)

        job_data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "job_id": f"deferred-publish-{upload_id}",
        }
        ctx = create_context(job_data, upload_record, user_settings, entitlements)
        ctx.user_record = user_record
        ctx.started_at = _now_utc()
        ctx.state = "processing"
        ctx.redis_client = redis_client

        if ctx.user_record and ctx.entitlements:
            ent = ctx.entitlements

            if hasattr(ctx, "num_thumbnails") and ctx.num_thumbnails > ent.max_thumbnails:
                ctx.num_thumbnails = ent.max_thumbnails
            if hasattr(ctx, "caption_frames") and ctx.caption_frames > ent.max_caption_frames:
                ctx.caption_frames = ent.max_caption_frames
            if not ent.can_burn_hud and hasattr(ctx, "hud_enabled"):
                ctx.hud_enabled = False
            if ent.can_watermark and hasattr(ctx, "watermark_text"):
                if not ctx.watermark_text:
                    ctx.watermark_text = "UploadM8"
            if not ent.can_ai and hasattr(ctx, "use_ai"):
                ctx.use_ai = False

        await db_stage.mark_processing_started(db_pool, ctx)

        processed_assets_json = upload_record.get("processed_assets") or "{}"
        try:
            processed_assets: Dict[str, str] = json.loads(processed_assets_json)
        except Exception:
            processed_assets = {}

        if not processed_assets:
            logger.error(f"[{upload_id}] No processed_assets found — cannot publish")
            await db_stage.mark_upload_failed_diagnostic(
                db_pool,
                str(upload_id),
                "NO_PROCESSED_ASSETS",
                "Scheduled publish could not find processed video assets — upstream processing may not have completed.",
                failure_diag=_build_worker_failure_diag(
                    "NO_PROCESSED_ASSETS",
                    ctx=None,
                    extra={
                        "upload_id": str(upload_id),
                        "hint": "Worker restart or failed processing before R2 processed_assets were saved. Retry from upload or re-queue.",
                    },
                ),
                only_if_status="processing",
            )
            return False

        ctx.output_artifacts["processed_assets"] = json.dumps(processed_assets)
        ctx.processed_r2_key = processed_assets.get("default") or next(iter(processed_assets.values()), "")

        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        ctx.temp_dir = temp_dir

        downloaded: Dict[str, Path] = {}
        for platform, r2_key in processed_assets.items():
            if platform == "default" or platform.startswith("thumb_"):
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
            await db_stage.mark_upload_failed_diagnostic(
                db_pool,
                str(upload_id),
                "ASSET_DOWNLOAD_FAILED",
                "Could not download processed MP4s from storage for publish.",
                failure_diag=_build_worker_failure_diag(
                    "ASSET_DOWNLOAD_FAILED",
                    ctx=None,
                    extra={
                        "upload_id": str(upload_id),
                        "hint": "R2 keys in processed_assets may be invalid or storage temporarily unavailable.",
                    },
                ),
                only_if_status="processing",
            )
            temp_dir_obj.cleanup()
            return False

        default_key = processed_assets.get("default")
        if default_key:
            default_local = temp_dir / "default.mp4"
            try:
                await r2_stage.download_file(default_key, default_local)
                ctx.processed_video_path = default_local
            except Exception as e:
                logger.debug("[%s] default processed video download skipped: %s", upload_id, e)

        platform_thumb_map: Dict[str, str] = {}
        platform_thumb_r2_keys: Dict[str, str] = {}
        for key, r2_key in processed_assets.items():
            if not key.startswith("thumb_") or not r2_key:
                continue
            plat = key.replace("thumb_", "")
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
            result = await run_publish_and_notify(ctx, upload_id, user_id)
            return result
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception as e:
                logger.debug("[%s] deferred publish temp cleanup: %s", upload_id, e)

    except Exception as e:
        logger.exception(f"[{upload_id}] Deferred publish failed: {e}")
        if ctx:
            ctx.mark_error("INTERNAL", str(e))
            ctx.failure_diag = _build_worker_failure_diag(
                "DEFERRED_PUBLISH_INTERNAL",
                ctx=ctx,
                extra={"exception_type": type(e).__name__},
            )
            await db_stage.mark_processing_failed(db_pool, ctx, "INTERNAL", str(e))
        await notify_admin_error("deferred_publish_failure", {"upload_id": upload_id, "error": str(e)}, db_pool)
        return False



# ---------------------------------------------------------------------------
# ANALYTICS SYNC LOOP
# ---------------------------------------------------------------------------

def _analytics_oauth_refresh_allowed(user_id: str) -> bool:
    """True if TikTok/YouTube OAuth refresh may run for this user (per-worker-process cooldown)."""
    if ANALYTICS_OAUTH_REFRESH_COOLDOWN_SEC <= 0:
        return True
    uid = str(user_id)
    now = time.monotonic()
    last = _analytics_oauth_last_refresh.get(uid)
    return last is None or (now - last) >= ANALYTICS_OAUTH_REFRESH_COOLDOWN_SEC


def _analytics_oauth_mark_refreshed(user_id: str) -> None:
    _analytics_oauth_last_refresh[str(user_id)] = time.monotonic()


async def _sync_one_upload_analytics(
    conn: asyncpg.Connection,
    upload_id: str,
    user_id: str,
    pr_list: list,
    token_map: dict,
    token_map_by_platform: dict = None,
    token_map_by_plat_account: dict = None,
) -> dict:
    """
    Pull engagement stats for one completed upload from each platform API.
    Returns totals dict and writes them + analytics_synced_at to DB.

    token_map: keyed by platform_tokens.id (UUID). token_map_by_plat_account: (platform, account_id) → token.
    """
    import json as _json
    import httpx as _httpx
    from stages.publish_stage import decrypt_token
    from services.sync_analytics_helpers import resolve_token_candidates_for_platform_result
    from services.meta_graph_metrics import (
        facebook_per_video_engagement_fallback,
        instagram_per_media_engagement_fallback,
    )

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats = {}
    platform_fallback = token_map_by_platform or {}
    plat_acct = token_map_by_plat_account or {}

    def _accum_platform_stats(pstat: dict, p: str, s: dict) -> None:
        if p not in pstat:
            pstat[p] = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
        for k in ("views", "likes", "comments", "shares"):
            pstat[p][k] = int(pstat[p].get(k, 0)) + int(s.get(k, 0) or 0)

    async with _httpx.AsyncClient(timeout=15) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            ok = (
                pr.get("success") is True
                or str(pr.get("status", "")).lower() in ("published", "succeeded", "success")
            )
            if not ok:
                continue

            candidates = resolve_token_candidates_for_platform_result(
                pr, token_map, plat_acct, platform_fallback
            )
            if not candidates:
                continue

            video_id = (
                pr.get("platform_video_id")
                or pr.get("video_id") or pr.get("videoId") or pr.get("id")
                or pr.get("media_id") or pr.get("post_id") or pr.get("share_id")
            )

            try:
                resolved = False
                for tok in candidates:
                    access_token = tok.get("access_token", "")
                    if not access_token:
                        continue

                    if plat == "tiktok" and video_id:
                        from services.tiktok_api import tiktok_envelope_error, tiktok_video_query_url

                        qurl = tiktok_video_query_url()
                        resp = await client.post(
                            qurl,
                            headers={
                                "Authorization": f"Bearer {access_token}",
                                "Content-Type": "application/json",
                            },
                            json={"filters": {"video_ids": [str(video_id)]}},
                        )
                        try:
                            body = resp.json()
                        except Exception:
                            body = {}
                        if tiktok_envelope_error(body):
                            continue
                        if resp.status_code == 200:
                            vids = body.get("data", {}).get("videos", []) or []
                            if vids:
                                v = vids[0]
                                s = {
                                    "views":    int(v.get("view_count")    or 0),
                                    "likes":    int(v.get("like_count")     or 0),
                                    "comments": int(v.get("comment_count") or 0),
                                    "shares":   int(v.get("share_count")   or 0),
                                }
                                _accum_platform_stats(platform_stats, "tiktok", s)
                                total_views    += s["views"];    total_likes    += s["likes"]
                                total_comments += s["comments"]; total_shares   += s["shares"]
                                resolved = True
                                break

                    elif plat == "youtube" and video_id:
                        resp = await client.get(
                            "https://www.googleapis.com/youtube/v3/videos",
                            params={"part": "statistics,snippet,status", "id": str(video_id)},
                            headers={"Authorization": f"Bearer {access_token}"},
                        )
                        if resp.status_code == 200:
                            items = resp.json().get("items", []) or []
                            if not items:
                                pr["platform_presence"] = "not_found"
                                pr["visibility"] = None
                                resolved = True
                                break
                            st = items[0].get("statistics", {}) or {}
                            status = items[0].get("status") or {}
                            vis = str(status.get("privacyStatus") or "").lower() or None
                            pr["visibility"] = vis
                            pr["platform_presence"] = "ok"
                            s = {
                                "views":    int(st.get("viewCount")    or 0),
                                "likes":    int(st.get("likeCount")    or 0),
                                "comments": int(st.get("commentCount") or 0),
                                "shares":   0,
                            }
                            _accum_platform_stats(platform_stats, "youtube", s)
                            total_views    += s["views"]; total_likes    += s["likes"]
                            total_comments += s["comments"]
                            resolved = True
                            break
                        elif resp.status_code == 401:
                            continue

                    elif plat == "instagram" and video_id:
                        media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
                        resp = await client.get(
                            f"https://graph.facebook.com/v21.0/{media_id}/insights",
                            params={
                                "access_token": access_token,
                                "metric": "views,plays,likes,comments,saved,shares,reach",
                            },
                        )
                        if resp.status_code in (401, 190):
                            continue
                        if resp.status_code == 200:
                            data = resp.json().get("data", []) or []
                            if not data:
                                pass
                            else:
                                s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                                ig_views = ig_plays = 0
                                for m in data:
                                    name = m.get("name", "")
                                    vals = m.get("values", [])
                                    val  = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                                    if name == "views":      ig_views     = val
                                    elif name == "plays":    ig_plays     = val
                                    elif name == "likes":    s["likes"]   += val
                                    elif name == "comments": s["comments"] += val
                                    elif name == "shares":   s["shares"]  += val
                                s["views"] = ig_views or ig_plays
                                _accum_platform_stats(platform_stats, "instagram", s)
                                total_views    += s["views"];    total_likes    += s["likes"]
                                total_comments += s["comments"]; total_shares   += s["shares"]
                                resolved = True
                                break
                        if not resolved:
                            fb_ig = await instagram_per_media_engagement_fallback(client, access_token, str(media_id))
                            if fb_ig:
                                _accum_platform_stats(platform_stats, "instagram", fb_ig)
                                total_views    += fb_ig["views"]
                                total_likes    += fb_ig["likes"]
                                total_comments += fb_ig["comments"]
                                total_shares   += fb_ig["shares"]
                                resolved = True
                                break

                    elif plat == "facebook" and video_id:
                        resp = await client.get(
                            f"https://graph.facebook.com/v21.0/{video_id}",
                            params={
                                "access_token": access_token,
                                "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)",
                            },
                        )
                        if resp.status_code in (401, 190):
                            continue
                        if resp.status_code == 200:
                            data = (resp.json().get("insights", {}) or {}).get("data", []) or []
                            if not data:
                                pass
                            else:
                                s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                                for m in data:
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
                                _accum_platform_stats(platform_stats, "facebook", s)
                                total_views    += s["views"];    total_likes    += s["likes"]
                                total_comments += s["comments"]; total_shares   += s["shares"]
                                resolved = True
                                break
                        if not resolved:
                            fb_fb = await facebook_per_video_engagement_fallback(client, access_token, str(video_id))
                            if fb_fb:
                                s = {
                                    "views": fb_fb["views"],
                                    "likes": fb_fb["likes"],
                                    "comments": fb_fb["comments"],
                                    "shares": fb_fb["shares"],
                                }
                                _accum_platform_stats(platform_stats, "facebook", s)
                                total_views    += s["views"]
                                total_likes    += s["likes"]
                                total_comments += s["comments"]
                                total_shares   += s["shares"]
                                resolved = True
                                break

            except Exception as e:
                logger.warning(f"[analytics-sync] {plat}/{upload_id}: {e}")
                continue

    await conn.execute(
        """
        UPDATE uploads
           SET views = $1,
               likes = $2,
               comments = $3,
               shares = $4,
               platform_results = $5::jsonb,
               analytics_synced_at = NOW(),
               updated_at = NOW()
         WHERE id = $6
           AND user_id = $7
        """,
        total_views,
        total_likes,
        total_comments,
        total_shares,
        _json.dumps(pr_list),
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
    for completed uploads and writes them to `uploads` (views/likes/...).

    Schedule: every ANALYTICS_SYNC_INTERVAL_SECONDS (default 3h ≈ 8×/day). Runs on the
    worker host — users do not need to be logged in or have the app open.
    Manual: POST /api/uploads/{id}/sync-analytics or queue/dashboard sync actions.
    """
    from stages.publish_stage import decrypt_token

    global shutdown_requested

    logger.info(
        f"[analytics-sync] loop started | "
        f"interval={ANALYTICS_SYNC_INTERVAL}s | "
        f"batch={ANALYTICS_SYNC_BATCH} | "
        f"resync_every={ANALYTICS_RESYNC_HOURS}h"
    )

    await asyncio.sleep(60)

    while not shutdown_requested:
        if not await _acquire_cron_lock("analytics_sync", ttl_seconds=max(ANALYTICS_SYNC_INTERVAL - 30, 120)):
            await asyncio.sleep(ANALYTICS_SYNC_INTERVAL)
            continue
        try:
            resync_cutoff = _now_utc() - timedelta(hours=ANALYTICS_RESYNC_HOURS)
            window_start  = _now_utc() - timedelta(days=90)

            async with db_pool.acquire() as conn:
                uploads = await conn.fetch(
                    """
                    SELECT u.id AS upload_id, u.user_id, u.platform_results, u.platforms
                      FROM uploads u
                     WHERE u.status IN ('completed','succeeded','partial')
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
                    for pr in pr_list:
                        if pr.get("platform_video_id") and not pr.get("video_id"):
                            pr["video_id"] = pr["platform_video_id"]

                    if not pr_list:
                        async with db_pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE uploads SET analytics_synced_at=NOW() WHERE id = $1",
                                row["upload_id"],
                            )
                        continue

                    async with db_pool.acquire() as conn:
                        token_rows = await conn.fetch(
                            """SELECT id, platform, token_blob, account_id
                                  FROM platform_tokens
                                 WHERE user_id = $1 AND revoked_at IS NULL""",
                            row["user_id"],
                        )

                    token_map = {}
                    token_map_by_platform = {}

                    def _dec_blob_for_map(blob):
                        if isinstance(blob, str):
                            blob = json.loads(blob)
                        return decrypt_token(blob)

                    for tr in token_rows:
                        try:
                            blob = tr["token_blob"]
                            dec = _dec_blob_for_map(blob)
                            if dec:
                                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                                    dec["ig_user_id"] = str(tr["account_id"])
                                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                                    dec["page_id"] = str(tr["account_id"])
                                token_id = str(tr["id"])
                                token_map[token_id] = dec
                                token_map_by_platform.setdefault(tr["platform"], []).append(dec)
                        except Exception as e:
                            logger.debug(
                                "[analytics-sync] token row skip id=%s: %s", tr.get("id"), e
                            )

                    from services.sync_analytics_helpers import build_plat_account_token_map

                    token_map_by_plat_account = build_plat_account_token_map(token_rows, _dec_blob_for_map)

                    if not token_map:
                        async with db_pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE uploads SET analytics_synced_at=NOW() WHERE id = $1",
                                row["upload_id"],
                            )
                        continue

                    # Refresh all platform tokens at most once per user per cooldown.
                    if _analytics_oauth_refresh_allowed(user_id):
                        try:
                            from stages.publish_stage import _refresh_tiktok_token, _refresh_youtube_token, _refresh_meta_token

                            for tr in token_rows:
                                if tr["platform"] != "tiktok":
                                    continue
                                try:
                                    dec = _dec_blob_for_map(tr["token_blob"])
                                    if dec:
                                        await _refresh_tiktok_token(
                                            dict(dec), db_pool=db_pool, user_id=str(user_id)
                                        )
                                except Exception as e:
                                    logger.debug(
                                        "[analytics-sync] TikTok refresh skip id=%s: %s",
                                        tr.get("id"),
                                        e,
                                    )
                            for tr in token_rows:
                                if tr["platform"] != "youtube":
                                    continue
                                try:
                                    dec = _dec_blob_for_map(tr["token_blob"])
                                    if dec:
                                        await _refresh_youtube_token(
                                            dict(dec), db_pool=db_pool, user_id=str(user_id)
                                        )
                                except Exception as e:
                                    logger.debug(
                                        "[analytics-sync] YouTube refresh skip id=%s: %s",
                                        tr.get("id"),
                                        e,
                                    )
                            for tr in token_rows:
                                if tr["platform"] not in ("instagram", "facebook"):
                                    continue
                                try:
                                    dec = _dec_blob_for_map(tr["token_blob"])
                                    if dec:
                                        await _refresh_meta_token(
                                            dict(dec), platform=tr["platform"],
                                            db_pool=db_pool, user_id=str(user_id)
                                        )
                                except Exception as e:
                                    logger.debug(
                                        "[analytics-sync] Meta refresh skip id=%s: %s",
                                        tr.get("id"),
                                        e,
                                    )
                            async with db_pool.acquire() as conn:
                                trs = await conn.fetch(
                                    """SELECT id, platform, token_blob, account_id
                                         FROM platform_tokens
                                        WHERE user_id = $1 AND revoked_at IS NULL""",
                                    row["user_id"],
                                )
                            token_map = {}
                            token_map_by_platform = {}
                            for tr in trs:
                                try:
                                    blob = tr["token_blob"]
                                    dec = _dec_blob_for_map(blob)
                                    if dec:
                                        if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                                            dec["ig_user_id"] = str(tr["account_id"])
                                        if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                                            dec["page_id"] = str(tr["account_id"])
                                        token_map[str(tr["id"])] = dec
                                        token_map_by_platform.setdefault(tr["platform"], []).append(dec)
                                except Exception as e:
                                    logger.debug(
                                        "[analytics-sync] token remap skip id=%s: %s",
                                        tr.get("id"),
                                        e,
                                    )
                            token_map_by_plat_account = build_plat_account_token_map(trs, _dec_blob_for_map)
                            _analytics_oauth_mark_refreshed(user_id)
                        except Exception as _wr:
                            logger.debug(f"[analytics-sync] oauth refresh skipped: {_wr}")

                    try:
                        async with db_pool.acquire() as conn:
                            result = await _sync_one_upload_analytics(
                                conn,
                                upload_id,
                                user_id,
                                pr_list,
                                token_map,
                                token_map_by_platform,
                                token_map_by_plat_account,
                            )
                        synced += 1
                        logger.debug(
                            f"[analytics-sync] {upload_id}: "
                            f"views={result['views']} likes={result['likes']} "
                            f"comments={result['comments']} shares={result['shares']}"
                        )
                    except Exception as e:
                        errors += 1
                        logger.warning(f"[analytics-sync] {upload_id} failed: {e}")

                    await asyncio.sleep(1)

                logger.info(
                    f"[analytics-sync] cycle complete | synced={synced} errors={errors}"
                )

        except asyncpg.PostgresError as e:
            logger.warning(f"[analytics-sync] DB error: {e}")
        except Exception as e:
            logger.exception(f"[analytics-sync] unexpected error: {e}")

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=ANALYTICS_SYNC_INTERVAL,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("[analytics-sync] loop stopped")


# ---------------------------------------------------------------------------
# PLATFORM METRICS CACHE (account-level live stats → platform_metrics_cache)
# ---------------------------------------------------------------------------

async def run_platform_metrics_cache_loop() -> None:
    """
    Periodically refresh `platform_metrics_cache` for every user with connected accounts
    (account-level TikTok/YouTube/IG/FB rollups).

    Schedule: every PLATFORM_METRICS_CACHE_INTERVAL_SECONDS (default 3h ≈ 8×/day), worker
    only — independent of login. Same data as GET /api/analytics/platform-metrics, persisted
    for dashboards/analytics when the API process reads DB cache. Manual: ?force=true on that endpoint.
    """
    global shutdown_requested, shutdown_event

    logger.info(
        f"[platform-metrics-cache] loop started | interval={PLATFORM_METRICS_CACHE_INTERVAL}s"
    )
    await asyncio.sleep(120)

    while not shutdown_requested:
        if not await _acquire_cron_lock("platform_metrics_cache", ttl_seconds=max(PLATFORM_METRICS_CACHE_INTERVAL - 30, 120)):
            await asyncio.sleep(PLATFORM_METRICS_CACHE_INTERVAL)
            continue
        try:
            from services.platform_metrics_job import refresh_all_users_platform_metrics_cache

            n = await refresh_all_users_platform_metrics_cache(db_pool)
            logger.info(f"[platform-metrics-cache] cycle complete | users_refreshed={n}")
        except Exception as e:
            logger.exception(f"[platform-metrics-cache] cycle error: {e}")

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=PLATFORM_METRICS_CACHE_INTERVAL,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("[platform-metrics-cache] loop stopped")


# ---------------------------------------------------------------------------
# CATALOG SYNC LOOP  — discover external videos + link to uploads
# ---------------------------------------------------------------------------

CATALOG_SYNC_INTERVAL = int(os.environ.get("CATALOG_SYNC_INTERVAL_SECONDS", str(6 * 3600)))


async def run_catalog_sync_loop() -> None:
    """
    Periodically syncs the video catalog for every user with connected accounts.

    Schedule: every CATALOG_SYNC_INTERVAL_SECONDS (default 6 h).
    Each run is incremental — pagination cursors are stored in
    `platform_content_sync_state` so we only fetch new videos from each platform.
    After catalog fetch, the linker matches platform_results from uploads and
    sets upload_id on any external row that maps to an UploadM8 upload.
    """
    global shutdown_requested, shutdown_event

    logger.info(f"[catalog-sync] loop started | interval={CATALOG_SYNC_INTERVAL}s")
    await asyncio.sleep(180)  # stagger startup behind other loops

    while not shutdown_requested:
        if not await _acquire_cron_lock("catalog_sync", ttl_seconds=max(CATALOG_SYNC_INTERVAL - 60, 300)):
            await asyncio.sleep(CATALOG_SYNC_INTERVAL)
            continue
        try:
            from services.catalog_sync import refresh_catalog_for_all_users
            n = await refresh_catalog_for_all_users(db_pool)
            logger.info(f"[catalog-sync] cycle complete | users_synced={n}")
        except Exception as e:
            logger.exception(f"[catalog-sync] cycle error: {e}")

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=CATALOG_SYNC_INTERVAL,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("[catalog-sync] loop stopped")


# ---------------------------------------------------------------------------
# KPI COLLECTOR LOOP
# ---------------------------------------------------------------------------

async def run_kpi_collector_loop() -> None:
    """
    Every KPI_COLLECTOR_INTERVAL (default 30 min), fetch cost/revenue from
    Stripe, OpenAI, Mailgun, Cloudflare, etc. and post to cost_tracking.
    """
    global shutdown_requested, shutdown_event

    from stages.kpi_collector import run_kpi_collect

    logger.info(f"[kpi-collector] loop started | interval={KPI_COLLECTOR_INTERVAL}s")

    await asyncio.sleep(120)

    while not shutdown_requested:
        if not await _acquire_cron_lock("kpi_collector", ttl_seconds=max(KPI_COLLECTOR_INTERVAL - 30, 120)):
            await asyncio.sleep(KPI_COLLECTOR_INTERVAL)
            continue
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


async def run_ml_scoring_loop() -> None:
    """
    Periodically recompute strategy quality score rows used by the
    generation engine for long-horizon learnings.
    """
    global shutdown_requested, shutdown_event
    from services.ml_scoring_job import run_ml_scoring_cycle

    logger.info(
        f"[ml-scoring] loop started | interval={ML_SCORING_INTERVAL}s lookback_days={ML_SCORING_LOOKBACK_DAYS}"
    )
    await asyncio.sleep(120)

    while not shutdown_requested:
        if not await _acquire_cron_lock("ml_scoring", ttl_seconds=max(ML_SCORING_INTERVAL - 30, 120)):
            await asyncio.sleep(ML_SCORING_INTERVAL)
            continue
        try:
            await run_ml_scoring_cycle(db_pool, lookback_days=ML_SCORING_LOOKBACK_DAYS)
        except Exception as e:
            logger.warning(f"[ml-scoring] cycle error: {e}")

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=ML_SCORING_INTERVAL,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("[ml-scoring] loop stopped")


# ---------------------------------------------------------------------------
# SCHEDULER LOOP
# ---------------------------------------------------------------------------

async def run_scheduler_loop() -> None:
    """
    Background loop that manages deferred (staged/ready_to_publish) uploads.
    """
    global shutdown_requested, _job_semaphore

    logger.info(
        f"Scheduler loop started | "
        f"poll_interval={SCHEDULER_POLL_INTERVAL}s | "
        f"processing_window={PROCESSING_WINDOW_MINUTES}min"
    )

    while not shutdown_requested:
        if not await _acquire_cron_lock("scheduler", ttl_seconds=max(SCHEDULER_POLL_INTERVAL - 5, 10)):
            await asyncio.sleep(SCHEDULER_POLL_INTERVAL)
            continue
        try:
            now = _now_utc()
            try:
                _orph_stale = int(os.environ.get("WORKER_ORPHAN_STALE_MINUTES", "25"))
                _orph_min_age = int(os.environ.get("WORKER_ORPHAN_MIN_JOB_MINUTES", "10"))
                if _orph_stale > 0:
                    await db_stage.reconcile_stale_processing_uploads(
                        db_pool,
                        stale_minutes=_orph_stale,
                        min_job_age_minutes=_orph_min_age,
                        base_diag=_build_worker_failure_diag("WORKER_ORPHANED", ctx=None, extra={}),
                    )
            except Exception as _sched_recon_err:
                logger.debug("scheduler stale reconcile: %s", _sched_recon_err)

            process_cutoff = now + timedelta(minutes=PROCESSING_WINDOW_MINUTES)

            async with db_pool.acquire() as conn:

                if _process_semaphore is not None:
                    try:
                        free_slots = _process_semaphore._value
                    except AttributeError:
                        free_slots = WORKER_CONCURRENCY
                else:
                    free_slots = 0
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
                        continue

                    logger.info(
                        f"[{upload_id}] Scheduler: staging → queued for processing "
                        f"(scheduled at {row.get('scheduled_time')})"
                    )

                    job_data = {
                        "upload_id": upload_id,
                        "user_id": user_id,
                        "job_id": f"scheduled-{upload_id}",
                        "deferred": True,
                    }

                    task = asyncio.create_task(_run_job_with_semaphore(job_data))
                    logger.debug(f"[{upload_id}] Processing task dispatched")

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

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=SCHEDULER_POLL_INTERVAL,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("Scheduler loop stopped")


async def _run_job_with_semaphore(job_data: dict) -> None:
    upload_id = str(job_data.get("upload_id", "?"))
    user_id = str(job_data.get("user_id", ""))
    async with _process_semaphore:
        try:
            await asyncio.wait_for(
                run_processing_pipeline(job_data),
                timeout=JOB_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{upload_id}] Scheduled pipeline timed out after {JOB_TIMEOUT}s")
            try:
                await db_stage.mark_upload_failed_diagnostic(
                    db_pool,
                    str(upload_id),
                    "JOB_TIMEOUT",
                    f"Pipeline exceeded {JOB_TIMEOUT}s timeout (host may be undersized or clip very long).",
                    failure_diag=_build_worker_failure_diag(
                        "JOB_TIMEOUT",
                        ctx=None,
                        extra={
                            "upload_id": str(upload_id),
                            "job_timeout_sec": int(JOB_TIMEOUT),
                            "hint": "Increase JOB_TIMEOUT_SECONDS or reduce WORKER_CONCURRENCY / heavy AI stages on small instances.",
                        },
                    ),
                    only_if_status="processing",
                )
            except Exception as e:
                logger.warning("[%s] timeout: failed to mark upload failed in DB: %s", upload_id, e)
            await _send_to_dead_letter(
                upload_id, user_id, job_data,
                "JOB_TIMEOUT", f"Pipeline timed out after {JOB_TIMEOUT}s",
                retry_count=job_data.get("_retry_count", 0),
            )
        except Exception as e:
            logger.exception(f"[{upload_id}] Unhandled pipeline error: {e}")
            await _mark_pipeline_uncaught_failure(upload_id, user_id, job_data, e)


async def _run_deferred_publish_with_semaphore(upload_id: str, user_id: str) -> None:
    async with _publish_semaphore:
        try:
            await asyncio.wait_for(
                run_deferred_publish(upload_id, user_id),
                timeout=JOB_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{upload_id}] Deferred publish timed out after {JOB_TIMEOUT}s")
            try:
                await db_stage.mark_upload_failed_diagnostic(
                    db_pool,
                    str(upload_id),
                    "PUBLISH_TIMEOUT",
                    f"Deferred publish exceeded {JOB_TIMEOUT}s timeout.",
                    failure_diag=_build_worker_failure_diag(
                        "PUBLISH_TIMEOUT",
                        ctx=None,
                        extra={
                            "upload_id": str(upload_id),
                            "job_timeout_sec": int(JOB_TIMEOUT),
                            "hint": "Platform APIs were too slow or blocked; retry scheduled publish or increase JOB_TIMEOUT_SECONDS.",
                        },
                    ),
                    only_if_status="processing",
                )
            except Exception as e:
                logger.warning("[%s] deferred publish timeout: DB update failed: %s", upload_id, e)
        except Exception as e:
            logger.exception(f"[{upload_id}] Unhandled deferred publish error: {e}")


# ---------------------------------------------------------------------------
# Redis job consumer (immediate uploads only)
# ---------------------------------------------------------------------------

async def _process_one_job_body(job_json: str) -> None:
    try:
        job_data = json.loads(job_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid job JSON: {e}")
        return
    upload_id = job_data.get("upload_id", "?")
    async with _process_semaphore:
        try:
            await asyncio.wait_for(
                run_processing_pipeline(job_data),
                timeout=JOB_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{upload_id}] Job timed out after {JOB_TIMEOUT}s — releasing slot")
            try:
                await db_stage.mark_upload_failed_diagnostic(
                    db_pool,
                    str(upload_id),
                    "JOB_TIMEOUT",
                    f"Pipeline exceeded {JOB_TIMEOUT}s timeout.",
                    failure_diag=_build_worker_failure_diag(
                        "JOB_TIMEOUT",
                        ctx=None,
                        extra={
                            "upload_id": str(upload_id),
                            "job_timeout_sec": int(JOB_TIMEOUT),
                            "hint": "Worker job slot limit or slow FFmpeg/ML — reduce concurrency or disable optional AI stages.",
                        },
                    ),
                    only_if_status="processing",
                )
            except Exception as e:
                logger.warning("[%s] job timeout: failed to mark upload failed: %s", upload_id, e)
            await _send_to_dead_letter(
                upload_id, job_data.get("user_id", ""), job_data,
                "JOB_TIMEOUT", f"Pipeline timed out after {JOB_TIMEOUT}s",
                retry_count=job_data.get("_retry_count", 0),
            )
        except Exception as e:
            logger.exception(f"[{upload_id}] Unhandled pipeline exception: {e}")
            await _mark_pipeline_uncaught_failure(
                upload_id, str(job_data.get("user_id", "")), job_data, e
            )


async def _process_one_job_tracked(
    job_json: str,
    stream_key: Optional[str] = None,
    message_id: Optional[str] = None,
    list_key: Optional[str] = None,
) -> None:
    """
    Per-user cluster cap, then pipeline. XACK stream messages after work (at-least-once).
    """
    global redis_client
    manual_xack = False
    user_got = False
    user_id = ""
    should_final_ack = False
    try:
        try:
            job_data = json.loads(job_json)
        except json.JSONDecodeError as e:
            logger.error("Invalid job JSON (will ACK stream entry): %s", e)
            should_final_ack = True
            return

        user_id = str(job_data.get("user_id") or "")
        pc = str(job_data.get("priority_class") or "p4")
        upload_id = job_data.get("upload_id", "?")

        if redis_client:
            if not await user_process_try_acquire(redis_client, user_id, pc):
                if stream_key and message_id:
                    ok = await enqueue_process_job(
                        redis_client, list_key_from_stream(stream_key), job_data
                    )
                    if ok:
                        await xack_message(redis_client, stream_key, DEFAULT_GROUP, message_id)
                        manual_xack = True
                        logger.info("[%s] Re-queued (per-user process cap)", upload_id)
                    else:
                        should_final_ack = False
                elif list_key:
                    await asyncio.sleep(0.2 + random.random() * 0.5)
                    try:
                        await redis_client.rpush(list_key, job_json)
                        logger.info("[%s] Re-pushed list queue (per-user cap)", upload_id)
                    except Exception as ex:
                        logger.error("[%s] requeue rpush failed: %s", upload_id, ex)
                return
            user_got = True

        await _process_one_job_body(job_json)
        should_final_ack = True
    except Exception as e:
        logger.exception("Job tracked wrapper error: %s", e)
        should_final_ack = bool(stream_key and message_id)
    finally:
        if user_got and redis_client and user_id:
            await user_process_release(redis_client, user_id)
        if (
            stream_key
            and message_id
            and redis_client
            and should_final_ack
            and not manual_xack
        ):
            await xack_message(redis_client, stream_key, DEFAULT_GROUP, message_id)


async def stream_reclaim_loop() -> None:
    """XAUTOCLAIM stale pending stream messages (worker crash before XACK)."""
    global redis_client, shutdown_requested
    reclaim_as = f"{WORKER_STREAM_CONSUMER}:reclaim"
    min_idle = int(os.environ.get("STREAM_RECLAIM_MIN_IDLE_MS", "120000"))
    interval = max(5, int(os.environ.get("STREAM_RECLAIM_INTERVAL_SEC", "25")))
    batch = max(1, int(os.environ.get("STREAM_RECLAIM_COUNT", "8")))
    stream_keys = process_stream_keys_ordered(
        PROCESS_PRIORITY_QUEUE,
        PROCESS_NORMAL_QUEUE,
        PRIORITY_JOB_QUEUE,
        UPLOAD_JOB_QUEUE,
    )
    logger.info(
        "Stream reclaim loop | min_idle_ms=%s interval=%s batch=%s consumer=%s",
        min_idle,
        interval,
        batch,
        reclaim_as,
    )
    while not shutdown_requested:
        await asyncio.sleep(interval)
        if shutdown_requested or not use_redis_streams() or not redis_client:
            continue
        for sk in stream_keys:
            try:
                claimed = await xautoclaim_batch(
                    redis_client, sk, DEFAULT_GROUP, reclaim_as, min_idle, batch
                )
            except Exception as e:
                logger.debug("xautoclaim %s: %s", sk, e)
                continue
            for mid, payload in claimed:
                asyncio.create_task(
                    _process_one_job_tracked(payload, stream_key=sk, message_id=mid, list_key=None)
                )


async def process_jobs() -> None:
    """
    Consume process-lane jobs from Redis (FFmpeg-heavy).
    Streams (at-least-once) + legacy list drain (BRPOP).
    """
    global shutdown_requested, redis_client, _process_semaphore, _publish_semaphore, _job_semaphore

    _process_semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)
    _publish_semaphore = asyncio.Semaphore(PUBLISH_CONCURRENCY)
    _job_semaphore     = _process_semaphore

    all_process_queues = [
        PROCESS_PRIORITY_QUEUE,
        PROCESS_NORMAL_QUEUE,
        PRIORITY_JOB_QUEUE,
        UPLOAD_JOB_QUEUE,
    ]
    stream_keys = process_stream_keys_ordered(
        PROCESS_PRIORITY_QUEUE,
        PROCESS_NORMAL_QUEUE,
        PRIORITY_JOB_QUEUE,
        UPLOAD_JOB_QUEUE,
    )

    logger.info(
        f"Job consumer started | "
        f"streams={use_redis_streams()} | "
        f"consumer={WORKER_STREAM_CONSUMER} | "
        f"process_concurrency={WORKER_CONCURRENCY} | "
        f"heavy_pipeline_slots={WORKER_HEAVY_PIPELINE_SLOTS} | "
        f"publish_concurrency={PUBLISH_CONCURRENCY} | "
        f"process_queues={PROCESS_PRIORITY_QUEUE}, {PROCESS_NORMAL_QUEUE}"
    )

    consecutive_redis_errors = 0
    active_tasks: List[asyncio.Task] = []

    while not shutdown_requested:
        active_tasks = [t for t in active_tasks if not t.done()]

        try:
            job_json = None
            sk: Optional[str] = None
            mid: Optional[str] = None
            lk: Optional[str] = None

            if use_redis_streams():
                got = await xreadgroup_one(
                    redis_client,
                    stream_keys,
                    DEFAULT_GROUP,
                    WORKER_STREAM_CONSUMER,
                    block_ms=max(500, int(float(POLL_INTERVAL) * 1000)),
                )
                if got:
                    sk, mid, job_json = got

            if not job_json:
                job_raw = await redis_client.brpop(
                    all_process_queues,
                    timeout=int(max(1, float(POLL_INTERVAL))),
                )
                if job_raw:
                    lk, job_json = job_raw

            consecutive_redis_errors = 0

            if job_json:
                task = asyncio.create_task(
                    _process_one_job_tracked(job_json, stream_key=sk, message_id=mid, list_key=lk)
                )
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
    global db_pool, redis_client, shutdown_event, _heavy_pipeline_sem, WORKER_STREAM_CONSUMER

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    if not DATABASE_URL:
        logger.error("DATABASE_URL not set")
        sys.exit(1)
    if not REDIS_URL:
        logger.error("REDIS_URL not set")
        sys.exit(1)

    total_concurrency = WORKER_CONCURRENCY + PUBLISH_CONCURRENCY
    db_min = int(os.environ.get("WORKER_DB_POOL_MIN", str(max(3, total_concurrency))))
    db_max = int(os.environ.get("WORKER_DB_POOL_MAX", str(max(20, total_concurrency * 3))))
    db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=db_min,
        max_size=db_max,
        command_timeout=60,
        init=_init_asyncpg_codecs,
    )
    logger.info(f"Database connected | pool={db_min}-{db_max}")

    try:
        _orph_stale = int(os.environ.get("WORKER_ORPHAN_STALE_MINUTES", "25"))
        _orph_min_age = int(os.environ.get("WORKER_ORPHAN_MIN_JOB_MINUTES", "10"))
        if _orph_stale > 0 and _orph_min_age >= 0:
            _n = await db_stage.reconcile_stale_processing_uploads(
                db_pool,
                stale_minutes=_orph_stale,
                min_job_age_minutes=_orph_min_age,
                base_diag=_build_worker_failure_diag("WORKER_ORPHANED", ctx=None, extra={}),
            )
            if _n:
                logger.info("Reconciled %s stale processing upload(s) as WORKER_ORPHANED", _n)
    except Exception as _recon_err:
        logger.warning("Stale processing reconciliation skipped: %s", _recon_err)

    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()
    logger.info("Redis connected")

    WORKER_STREAM_CONSUMER = make_worker_consumer_name()
    if use_redis_streams():
        _skeys = process_stream_keys_ordered(
            PROCESS_PRIORITY_QUEUE,
            PROCESS_NORMAL_QUEUE,
            PRIORITY_JOB_QUEUE,
            UPLOAD_JOB_QUEUE,
        )
        for _sk in _skeys:
            await ensure_stream_group(redis_client, _sk, DEFAULT_GROUP)
        logger.info(
            "Redis Streams ready | group=%s consumer=%s (%s streams)",
            DEFAULT_GROUP,
            WORKER_STREAM_CONSUMER,
            len(_skeys),
        )

    _heavy_pipeline_sem = asyncio.Semaphore(WORKER_HEAVY_PIPELINE_SLOTS)
    logger.info(
        f"Heavy pipeline slots={WORKER_HEAVY_PIPELINE_SLOTS} "
        f"(parallel audio/ML/thumbnail tails capped; process concurrency={WORKER_CONCURRENCY})"
    )

    shutdown_event = asyncio.Event()

    tasks = [
        asyncio.create_task(process_jobs()),
    ]
    if use_redis_streams():
        tasks.append(asyncio.create_task(stream_reclaim_loop()))
    tasks.extend(
        [
        asyncio.create_task(run_scheduler_loop()),
        asyncio.create_task(run_verification_loop(db_pool, shutdown_event)),
        asyncio.create_task(run_analytics_sync_loop()),
        asyncio.create_task(run_platform_metrics_cache_loop()),
        asyncio.create_task(run_catalog_sync_loop()),
        asyncio.create_task(run_kpi_collector_loop()),
        asyncio.create_task(run_ml_scoring_loop()),
        ]
    )

    try:
        await notify_admin_worker_start(db_pool)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    finally:
        try:
            shutdown_event.set()
        except Exception as e:
            logger.debug("main shutdown: shutdown_event.set failed: %s", e)
        try:
            await notify_admin_worker_stop(db_pool)
        except Exception as e:
            logger.debug("main shutdown: notify_admin_worker_stop failed: %s", e)
        # Close Playwright browser on shutdown
        try:
            from stages.playwright_stage import close_browser
            await close_browser()
        except Exception as e:
            logger.debug("main shutdown: close_browser failed: %s", e)
        if db_pool:
            await db_pool.close()
        if redis_client:
            try:
                await redis_client.aclose()
            except AttributeError:
                await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
