"""
UploadM8 Upload routes -- extracted from app.py.

Handles upload lifecycle: presign, complete, cancel, retry, list, update,
thumbnail generation, analytics sync, and single-upload detail.
"""

import json
import logging
import pathlib
import re
import uuid
from datetime import datetime
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request

import core.state
from core.audit import log_system_event
from core.auth import decrypt_blob
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user
from core.helpers import (
    _load_uploads_columns,
    _now_utc,
    _pick_cols,
    _safe_col,
    _safe_json,
    get_plan,
)
from core.models import CompleteUploadBody, UploadInit, UploadUpdate
from core.queue import enqueue_job
from core.r2 import (
    _delete_r2_objects,
    _normalize_r2_key,
    generate_presigned_download_url,
    generate_presigned_upload_url,
    get_s3_client,
)
from core.scheduling import calculate_smart_schedule, get_existing_scheduled_days
from core.wallet import (
    atomic_reserve_tokens,
    get_wallet,
    partial_refund_tokens,
    refund_tokens,
    spend_tokens,
)
from routers.preferences import get_user_prefs_for_upload
from stages.entitlements import (
    compute_upload_cost,
    get_entitlements_for_tier,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

# ── Status view groupings for queue/dashboard (simplified UX) ──
# pending: waiting to process (includes smart + scheduled)
# processing: actively being processed
# completed: done (succeeded, partial, or legacy completed)
# failed: publish failed
_UPLOAD_VIEW_STATUS = {
    "pending": ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
    "processing": ("processing",),
    "completed": ("completed", "succeeded", "partial"),
    "failed": ("failed",),
    "staged": ("pending", "staged", "queued", "scheduled", "ready_to_publish"),  # alias for pending
    "smart_schedule": None,  # special: schedule_mode='smart' + pending statuses
}
_STATUS_LABEL = {
    "pending": "Pending",
    "staged": "Scheduled",
    "queued": "Queued",
    "scheduled": "Scheduled",
    "ready_to_publish": "Ready to publish",
    "processing": "Processing",
    "completed": "Completed",
    "succeeded": "Succeeded",
    "partial": "Partial",
    "failed": "Failed",
    "cancelled": "Cancelled",
}

_ALLOWED_VIDEO_TYPES = frozenset({
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/webm", "video/x-matroska",
})


# ── Helper: enrich platform_results with account identity ──

async def _enrich_platform_results(conn, upload_row: dict, user_id: str) -> list:
    """
    Return platform_results as a flat list. Each entry enriched with
    account_name/username/avatar.

    Priority:
      1. Fields already stored IN the entry (set by worker after Fix 2/3)
      2. target_accounts UUID -> platform_tokens JOIN (for uploads before Fix 3)
      3. Primary account per platform (last resort for very old uploads)
    """
    raw = upload_row.get("platform_results") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []

    if isinstance(raw, dict):
        items = [{"platform": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    elif isinstance(raw, list):
        items = list(raw)
    else:
        items = []

    if not items:
        return items

    # If ALL successful entries already have identity (set by new worker), skip DB join
    successful = [e for e in items if e.get("success") is not False]
    already_enriched = successful and all(
        e.get("account_username") or e.get("account_name") or e.get("account_id")
        for e in successful
    )
    if already_enriched:
        return items

    # Build token_row_id -> identity map from target_accounts
    token_map = {}
    target_ids = [str(t) for t in (upload_row.get("target_accounts") or []) if t]

    if target_ids:
        try:
            rows = await conn.fetch(
                """SELECT id, platform, account_id, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND id = ANY($2::uuid[]) AND revoked_at IS NULL""",
                user_id, target_ids
            )
            for r in rows:
                token_map[str(r["id"])] = {
                    "token_row_id":     str(r["id"]),
                    "account_id":       r["account_id"]       or "",
                    "account_name":     r["account_name"]     or "",
                    "account_username": r["account_username"] or "",
                    "account_avatar":   r["account_avatar"]   or "",
                    "platform":         r["platform"],
                }
        except Exception as e:
            logger.warning(f"_enrich_platform_results target lookup failed: {e}")

    # Fallback: primary token per platform for old uploads
    platform_fallback = {}
    if not token_map:
        try:
            platforms_needed = list({(e.get("platform") or "").lower() for e in items if e.get("platform")})
            if platforms_needed:
                rows = await conn.fetch(
                    """SELECT DISTINCT ON (platform)
                              id, platform, account_id, account_name, account_username, account_avatar
                       FROM platform_tokens
                       WHERE user_id = $1 AND platform = ANY($2::text[]) AND revoked_at IS NULL
                       ORDER BY platform, is_primary DESC NULLS LAST, updated_at DESC""",
                    user_id, platforms_needed
                )
                for r in rows:
                    platform_fallback[r["platform"]] = {
                        "token_row_id":     str(r["id"]),
                        "account_id":       r["account_id"]       or "",
                        "account_name":     r["account_name"]     or "",
                        "account_username": r["account_username"] or "",
                        "account_avatar":   r["account_avatar"]   or "",
                    }
        except Exception as e:
            logger.warning(f"_enrich_platform_results fallback lookup failed: {e}")

    enriched = []
    for entry in items:
        p = (entry.get("platform") or "").lower()

        stored_token_id = entry.get("token_row_id") or ""
        if stored_token_id and stored_token_id in token_map:
            acct = token_map[stored_token_id]
        elif token_map:
            acct = next((v for v in token_map.values() if v.get("platform") == p), {})
        else:
            acct = platform_fallback.get(p, {})

        merged = {**entry}
        for field in ("token_row_id", "account_id", "account_name", "account_username", "account_avatar"):
            if not merged.get(field) and acct.get(field):
                merged[field] = acct[field]

        enriched.append(merged)

    return enriched


# ── Helper: parse smart_schedule dict ──

def _parse_smart_schedule(sm: dict, upload_platforms: list) -> tuple:
    """
    Parse smart_schedule (platform -> ISO datetime string).
    Returns (schedule_metadata_json, scheduled_time_dt).
    Same logic as create flow: validate platforms, parse ISO, set scheduled_time = min.
    """
    if not isinstance(sm, dict):
        raise HTTPException(400, "smart_schedule must be a dict of platform -> ISO datetime string")
    if not sm:
        raise HTTPException(400, "smart_schedule requires per-platform times (non-empty object)")
    platforms = list(upload_platforms or [])
    for k in sm:
        if k not in platforms:
            raise HTTPException(400, f"smart_schedule platform '{k}' not in upload platforms")
    dts = []
    for v in sm.values():
        if not v:
            continue
        s = str(v).replace("Z", "+00:00").replace("z", "+00:00")
        try:
            dts.append(datetime.fromisoformat(s))
        except ValueError:
            raise HTTPException(400, "smart_schedule values must be valid ISO datetime strings")
    metadata = {k: v for k, v in sm.items() if v}
    scheduled_dt = min(dts) if dts else None
    return metadata, scheduled_dt


# ── Helper: update upload metadata (used by PATCH) ──

async def _update_upload_metadata(conn, upload_id: str, user_id: str, update_data: UploadUpdate) -> None:
    """PATCH /api/uploads - title, caption, hashtags, scheduled_time, smart_schedule."""
    upload = await conn.fetchrow(
        "SELECT id, status, platforms FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id, user_id
    )
    if not upload:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if upload["status"] not in editable:
        raise HTTPException(400, "Cannot edit upload that is already processing or published")

    _UPLOAD_META_COLS = frozenset({"title", "caption", "hashtags", "scheduled_time", "schedule_metadata", "schedule_mode", "updated_at"})
    updates = []
    params: list = [upload_id, user_id]
    param_count = 2

    if update_data.title is not None:
        param_count += 1
        updates.append(f"{_safe_col('title', _UPLOAD_META_COLS)} = ${param_count}")
        params.append(update_data.title)

    if update_data.caption is not None:
        param_count += 1
        updates.append(f"{_safe_col('caption', _UPLOAD_META_COLS)} = ${param_count}")
        params.append(update_data.caption)

    if update_data.hashtags is not None:
        param_count += 1
        updates.append(f"{_safe_col('hashtags', _UPLOAD_META_COLS)} = ${param_count}")
        params.append(update_data.hashtags)

    if update_data.scheduled_time is not None:
        param_count += 1
        updates.append(f"{_safe_col('scheduled_time', _UPLOAD_META_COLS)} = ${param_count}")
        params.append(update_data.scheduled_time)

    if update_data.smart_schedule is not None:
        metadata, scheduled_dt = _parse_smart_schedule(update_data.smart_schedule, upload["platforms"])
        param_count += 1
        updates.append(f"{_safe_col('schedule_metadata', _UPLOAD_META_COLS)} = ${param_count}::jsonb")
        params.append(json.dumps(metadata))
        if scheduled_dt is not None:
            param_count += 1
            updates.append(f"{_safe_col('scheduled_time', _UPLOAD_META_COLS)} = ${param_count}")
            params.append(scheduled_dt)
        param_count += 1
        updates.append(f"{_safe_col('schedule_mode', _UPLOAD_META_COLS)} = ${param_count}")
        params.append("smart")

    if not updates:
        raise HTTPException(400, "No updates provided")

    param_count += 1
    updates.append(f"{_safe_col('updated_at', _UPLOAD_META_COLS)} = ${param_count}")
    params.append(_now_utc())

    await conn.execute(f"UPDATE uploads SET {', '.join(updates)} WHERE id = $1 AND user_id = $2", *params)


# ============================================================
# Routes
# ============================================================

@router.post("/presign")
async def presign_upload(data: UploadInit, request: Request, user: dict = Depends(get_current_user)):
    """Create upload with user preferences applied"""
    plan = get_plan(user.get("subscription_tier", "free"))
    wallet = user.get("wallet", {})

    # Normalize optional fields coming from the client
    if getattr(data, "hashtags", None) is None:
        data.hashtags = []
    if getattr(data, "platforms", None) is None:
        data.platforms = []

    async with core.state.db_pool.acquire() as conn:
        # Fetch user preferences to apply defaults
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])

        # Apply preference defaults if user didn't specify
        if not getattr(data, "privacy", None):
            data.privacy = user_prefs["default_privacy"]

        # --- Hashtag assembly ---
        # Step 1: Merge form hashtags + always_hashtags into upload record.
        #         Platform-specific hashtags are applied per-platform at publish time
        #         (context.get_effective_hashtags(platform)), not here.
        def _split_tags(v):
            if v is None:
                return []
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return []
                try:
                    maybe = json.loads(s)
                    if isinstance(maybe, list):
                        v = maybe
                    else:
                        v = s
                except Exception:
                    v = s
            if isinstance(v, str):
                return [p for p in re.split(r"[\s,]+", v.strip()) if p]
            if isinstance(v, (list, tuple, set)):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            return [s] if s else []

        def _to_hash_tags(v):
            out = []
            for t in _split_tags(v):
                t = str(t).strip()
                if not t:
                    continue
                t = t.lower().lstrip("#")[:50]
                if t:
                    out.append(f"#{t}")
            return out

        # Store ONLY form/manual hashtags here. always_hashtags and platform_hashtags
        # are applied per-platform at publish via context.get_effective_hashtags(platform).
        combined = _to_hash_tags(getattr(data, "hashtags", []) or [])

        # Step 2: AI-generated hashtag injection -- gated on plan + user toggle.
        #         Actual AI generation happens later in caption_stage.
        if user_prefs.get("ai_hashtags_enabled") and plan.get("ai"):
            pass  # ai_hashtags merged into ctx later by caption_stage

        # Step 3: Deduplicate, strip blocked, enforce limit -- always runs.
        blocked = set(_to_hash_tags(user_prefs.get("blocked_hashtags", []) or user_prefs.get("blockedHashtags", [])))
        combined = [h for h in combined if h and h not in blocked]
        data.hashtags = list(dict.fromkeys(combined))[: int(user_prefs.get("max_hashtags", 30))]

        # -- Compute PUT/AIC cost -- canonical formula from entitlements --
        ent_cost = get_entitlements_for_tier(user.get("subscription_tier", "free"))
        use_ai  = bool(getattr(data, "use_ai", False)) and ent_cost.can_ai
        use_hud = bool(user_prefs.get("hud_enabled", False)) and ent_cost.can_burn_hud

        # Each target account counts as a separate publish (costs +2 PUT per extra beyond 1)
        # When target_accounts provided: user selected specific accounts.
        # When empty: legacy one-per-platform.
        num_publish_targets = len(data.target_accounts) if data.target_accounts else len(data.platforms)
        put_cost, aic_cost = compute_upload_cost(
            entitlements=ent_cost,
            num_platforms=num_publish_targets,
            use_ai=use_ai,
            use_hud=use_hud,
            num_thumbnails=ent_cost.max_thumbnails,
        )

        # Balance is checked atomically during reserve_tokens below to prevent
        # race conditions where concurrent presign calls both pass a stale balance check.

        # -- Queue depth guard --
        pending_count = await conn.fetchval(
            """SELECT COUNT(*) FROM uploads
               WHERE user_id = $1
               AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
            user["id"]
        )
        ent_check = get_entitlements_for_tier(user.get("subscription_tier", "free"))
        if pending_count >= ent_check.queue_depth:
            raise HTTPException(
                429,
                f"Queue limit reached ({pending_count}/{ent_check.queue_depth} uploads pending). "
                "Wait for existing uploads to complete or upgrade your plan."
            )

        upload_id = str(uuid.uuid4())
        r2_key = f"uploads/{user['id']}/{upload_id}/{data.filename}"

        # Smart scheduling logic
        smart_schedule = None
        if getattr(data, "schedule_mode", None) == "smart":
            smart_schedule = calculate_smart_schedule(
                data.platforms,
                num_days=getattr(data, "smart_schedule_days", 7)
            )
            existing_days = await get_existing_scheduled_days(conn, user["id"], getattr(data, "smart_schedule_days", 7))
            if existing_days:
                smart_schedule = calculate_smart_schedule(
                    data.platforms,
                    num_days=getattr(data, "smart_schedule_days", 7)
                )

        scheduled_time = getattr(data, "scheduled_time", None)
        schedule_metadata = None

        if getattr(data, "schedule_mode", None) == "smart" and smart_schedule:
            schedule_metadata = {p: dt.isoformat() for p, dt in smart_schedule.items()}
            scheduled_time = min(smart_schedule.values())

        # Store upload with preferences metadata
        await conn.execute("""
            INSERT INTO uploads (
                id, user_id, r2_key, filename, file_size, platforms,
                title, caption, hashtags, privacy, status, scheduled_time,
                schedule_mode, put_reserved, aic_reserved, schedule_metadata,
                user_preferences, target_accounts
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'pending', $11, $12, $13, $14, $15, $16, $17)
        """,
            upload_id, user["id"], r2_key, data.filename, data.file_size,
            data.platforms, data.title, data.caption, data.hashtags,
            data.privacy, scheduled_time, data.schedule_mode, put_cost,
            aic_cost, json.dumps(schedule_metadata) if schedule_metadata else None,
            json.dumps(user_prefs), data.target_accounts or []
        )

        # Reserve tokens atomically -- enforces balance floor at DB level
        reserved = await atomic_reserve_tokens(conn, user["id"], put_cost, aic_cost, upload_id)
        if not reserved:
            # Roll back the upload record since we can't reserve tokens
            await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
            # Read fresh wallet to determine which token type is insufficient
            fresh_wallet = await get_wallet(conn, user["id"])
            put_avail = fresh_wallet["put_balance"] - fresh_wallet["put_reserved"]
            aic_avail = fresh_wallet["aic_balance"] - fresh_wallet["aic_reserved"]
            if put_avail < put_cost:
                raise HTTPException(429, {
                    "code": "insufficient_put",
                    "message": f"Insufficient PUT tokens ({put_avail} available, {put_cost} needed).",
                    "topup_url": "/settings.html#billing",
                })
            raise HTTPException(429, {
                "code": "insufficient_aic",
                "message": f"Insufficient AIC credits ({aic_avail} available, {aic_cost} needed).",
                "topup_url": "/settings.html#billing",
            })

    if data.content_type not in _ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, detail=f"Unsupported file type: {data.content_type}")

    presigned_url = generate_presigned_upload_url(r2_key, data.content_type)
    result = {
        "upload_id": upload_id,
        "presigned_url": presigned_url,
        "r2_key": r2_key,
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "schedule_mode": data.schedule_mode,
        "target_accounts": data.target_accounts or [],
        "preferences_applied": {
            "auto_captions": bool(user_prefs.get("auto_captions")),
            "auto_thumbnails": bool(user_prefs.get("auto_thumbnails")),
            "ai_hashtags": bool(user_prefs.get("ai_hashtags_enabled"))
        }
    }

    if smart_schedule:
        result["smart_schedule"] = {p: dt.isoformat() for p, dt in smart_schedule.items()}

    if getattr(data, "has_telemetry", False):
        telem_key = f"uploads/{user['id']}/{upload_id}/telemetry.map"
        result["telemetry_presigned_url"] = generate_presigned_upload_url(telem_key, "application/octet-stream")
        result["telemetry_r2_key"] = telem_key

    # Fire-and-forget audit -- does not affect upload flow
    async with core.state.db_pool.acquire() as _ac:
        await log_system_event(_ac, user_id=str(user["id"]), action="UPLOAD_INITIATED",
                               event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                               details={"filename": data.filename, "platforms": list(data.platforms or []),
                                        "schedule_mode": data.schedule_mode, "put_cost": put_cost,
                                        "aic_cost": aic_cost, "file_size": data.file_size},
                               request=request)

    return result


@router.post("/smart-schedule/preview")
async def preview_smart_schedule(platforms: List[str] = Query(...), days: int = Query(7), user: dict = Depends(get_current_user)):
    """Preview what the smart schedule would look like for given platforms"""
    if not platforms:
        raise HTTPException(400, "At least one platform required")

    schedule = calculate_smart_schedule(platforms, num_days=days)

    return {
        "schedule": {p: dt.isoformat() for p, dt in schedule.items()},
        "explanation": {
            p: {
                "date": dt.strftime("%A, %B %d"),
                "time": dt.strftime("%I:%M %p"),
                "reason": f"Optimal posting time for {p.title()}"
            }
            for p, dt in schedule.items()
        }
    }


@router.post("/{upload_id}/complete")
async def complete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    """
    Complete upload and either enqueue immediately (immediate mode) or stage
    for deferred processing (scheduled / smart mode).

    IMMEDIATE  -> status=queued, pushed to Redis -> worker fires NOW
    SCHEDULED  -> status=staged, NOT pushed to Redis -> scheduler fires at scheduled_time - processing_window
    SMART      -> status=staged, NOT pushed to Redis -> scheduler fires at first scheduled_time - processing_window

    Request body may include title, caption, hashtags from the upload page (manual metadata for single-file uploads).
    These override presign defaults and are stored before enqueue.
    """
    body = {}
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw) or {}
    except Exception:
        pass

    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        # Fetch preferences (fresher than what presign stored)
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])

        schedule_mode = upload.get("schedule_mode") or "immediate"

        # -- Apply manual metadata from upload page (single-file) --
        _COMPLETE_COLS = frozenset({"title", "caption", "hashtags"})
        updates = []
        params = []
        idx = 1
        if body.get("title") is not None:
            updates.append(f"{_safe_col('title', _COMPLETE_COLS)} = ${idx}")
            params.append(str(body["title"])[:512])
            idx += 1
        if body.get("caption") is not None:
            updates.append(f"{_safe_col('caption', _COMPLETE_COLS)} = ${idx}")
            params.append(str(body["caption"])[:10000])
            idx += 1
        if body.get("hashtags") is not None:
            raw_tags = body["hashtags"]
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in re.split(r"[\s,]+", str(raw_tags)) if t.strip()]
            tags = []
            for t in (raw_tags if isinstance(raw_tags, (list, tuple)) else []):
                t = str(t).strip().lstrip("#")[:50]
                if t:
                    tags.append(f"#{t}" if not t.startswith("#") else t)
            blocked = set(
                str(x).strip().lstrip("#").lower()
                for x in (user_prefs.get("blocked_hashtags") or user_prefs.get("blockedHashtags") or [])
            )
            tags = [t for t in tags if t and t.lstrip("#").lower() not in blocked]
            tags = list(dict.fromkeys(tags))[: int(user_prefs.get("max_hashtags", 30))]
            updates.append(f"{_safe_col('hashtags', _COMPLETE_COLS)} = ${idx}")
            params.append(tags)  # TEXT[] expects list
            idx += 1

        if updates:
            params.append(upload_id)
            await conn.execute(
                f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = ${idx}",
                *params
            )

        # -- Determine status and whether to enqueue --
        if schedule_mode in ("scheduled", "smart"):
            # DO NOT touch the Redis queue -- scheduler loop handles this.
            # Mark as 'staged' so the scheduler knows files are ready.
            new_status = "staged"
            await conn.execute(
                "UPDATE uploads SET status = 'staged', updated_at = NOW() WHERE id = $1",
                upload_id
            )
        else:
            # Immediate publish -- enqueue to Redis right now.
            new_status = "queued"
            await conn.execute(
                "UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1",
                upload_id
            )

    # Resolve full entitlements -- drives queue routing, AI depth, priority class
    ent = get_entitlements_for_tier(user.get("subscription_tier", "free"))

    if schedule_mode not in ("scheduled", "smart"):
        job_data = {
            "upload_id": upload_id,
            "user_id": str(user["id"]),
            "preferences": user_prefs,
            "plan_features": {
                "ai":           ent.can_ai,
                "priority":     ent.can_priority,
                "watermark":    ent.can_watermark,
                "ai_depth":     ent.ai_depth,
                "caption_frames": ent.max_caption_frames,
            },
            "priority_class": ent.priority_class,
        }
        await enqueue_job(job_data, lane="process", priority_class=ent.priority_class)

    # Compute scheduled_time display for smart schedules
    schedule_metadata = upload.get("schedule_metadata")
    smart_schedule_display = None
    if schedule_mode == "smart" and schedule_metadata:
        try:
            sm = schedule_metadata if isinstance(schedule_metadata, dict) else json.loads(schedule_metadata)
            smart_schedule_display = {p: v for p, v in sm.items()}
        except Exception:
            pass

    # Audit: upload submitted to pipeline
    await log_system_event(user_id=str(user["id"]), action="UPLOAD_SUBMITTED",
                           event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                           details={"schedule_mode": schedule_mode, "new_status": new_status,
                                    "platforms": list(upload.get("platforms") or [])},
                           request=request)

    return {
        "status": new_status,
        "upload_id": upload_id,
        "schedule_mode": schedule_mode,
        "scheduled_time": upload["scheduled_time"].isoformat() if upload.get("scheduled_time") else None,
        "smart_schedule": smart_schedule_display,
        "processing_features": {
            # `plan` is not defined in this scope -- use ent (resolved above)
            "auto_captions":  bool(user_prefs.get("auto_captions"))        if ent.can_ai else False,
            "auto_thumbnails": bool(user_prefs.get("auto_thumbnails"))     if ent.can_ai else False,
            "ai_hashtags":    bool(user_prefs.get("ai_hashtags_enabled"))  if ent.can_ai else False,
        }
    }


@router.post("/{upload_id}/reprepare")
async def reprepare_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Generate a fresh presigned R2 URL for an upload stuck in pending state.
    Used when the browser refreshed mid-transfer before /complete was called.
    The DB record exists -- we just issue new PUT URLs so the client can retry.
    """
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT id, r2_key, filename, status, telemetry_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")
        if upload["status"] not in ("pending",):
            raise HTTPException(400, f"Upload is not resumable (status: {upload['status']}). Use /retry for failed uploads.")

    r2_key = upload["r2_key"]
    filename = upload["filename"] or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    ct_map = {"mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo", "webm": "video/webm"}
    content_type = ct_map.get(ext, "video/mp4")

    result = {
        "upload_id": upload_id,
        "presigned_url": generate_presigned_upload_url(r2_key, content_type),
        "r2_key": r2_key,
        "filename": filename,
        "status": upload["status"],
    }
    if upload["telemetry_r2_key"]:
        result["telemetry_presigned_url"] = generate_presigned_upload_url(
            upload["telemetry_r2_key"], "application/octet-stream"
        )
        result["telemetry_r2_key"] = upload["telemetry_r2_key"]

    return result


@router.post("/{upload_id}/cancel")
async def cancel_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT put_reserved, aic_reserved, status, r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("completed", "succeeded", "cancelled", "failed"):
            raise HTTPException(400, "Cannot cancel this upload")

        current_status = upload["status"]

        if current_status == "processing":
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCEL_REQUESTED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request)
            return {"status": "cancel_requested", "message": "Cancel signal sent -- job will stop at next checkpoint"}
        else:
            await conn.execute(
                "UPDATE uploads SET cancel_requested = TRUE, status = 'cancelled', updated_at = NOW() WHERE id = $1",
                upload_id
            )
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
            await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_CANCELLED",
                                   event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                                   details={"status_at_cancel": current_status}, request=request, severity="WARNING")
            # Remove video and related assets from R2 so they don't persist
            r2_keys = [k for k in (
                upload.get("r2_key"),
                upload.get("telemetry_r2_key"),
                upload.get("processed_r2_key"),
                upload.get("thumbnail_r2_key"),
            ) if k]
            if r2_keys:
                await _delete_r2_objects(r2_keys)
            return {"status": "cancelled"}


@router.get("")
async def get_uploads(
    status: Optional[str] = None,
    view: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    trill_only: bool = False,
    meta: bool = False,
    user: dict = Depends(get_current_user),
):
    """
    Upload queue list for current user.

    Filter by status (exact) or view (semantic group):
      view=pending   -> status IN (pending, staged, queued, scheduled, ready_to_publish) -- waiting uploads incl. smart/scheduled
      view=processing -> status = processing
      view=completed -> status IN (completed, succeeded, partial)
      view=failed   -> status = failed
      view=staged   -> same as pending (alias for Staged/Pending filter)
      view=smart_schedule -> schedule_mode='smart' AND status IN pending group

    Contract (frontend-safe):
      - status_label: human-readable label for display (fixes ? succeeded, ? staged)
      - thumbnail_url, platform_results, hashtags, etc.
    """
    cols = await _load_uploads_columns(core.state.db_pool)

    wanted = [
        "id","filename","platforms","status","privacy",
        "title","caption","hashtags",
        "scheduled_time","created_at","completed_at",
        "put_reserved","aic_reserved",
        "error_code","error_detail",
        "thumbnail_r2_key","platform_results","file_size",
        "processing_started_at","processing_finished_at",
        "processing_stage","processing_progress",
        "views","likes","comments","shares",
        "schedule_mode","schedule_metadata",
        "video_url",
        "ai_title","ai_caption",
        "ai_generated_title","ai_generated_caption","ai_generated_hashtags",
        "target_accounts",
    ]
    select_cols = _pick_cols(wanted, cols) or ["id","filename","platforms","status","created_at"]
    select_sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE user_id = $1"
    count_sql = "SELECT COUNT(*) FROM uploads WHERE user_id = $1"
    params = [user["id"]]
    count_params = [user["id"]]

    # view takes precedence over status for semantic filtering
    # view=all: no status filter -- show all uploads
    if view == "all":
        pass  # no status filter
    elif view and view in _UPLOAD_VIEW_STATUS:
        statuses = _UPLOAD_VIEW_STATUS[view]
        if statuses is not None:
            placeholders = ", ".join(f"${i}" for i in range(len(params) + 1, len(params) + 1 + len(statuses)))
            params.extend(statuses)
            count_params.extend(statuses)
            select_sql += f" AND status IN ({placeholders})"
            count_sql += f" AND status IN ({placeholders})"
        else:
            # smart_schedule: schedule_mode='smart' AND status IN pending group
            pending = _UPLOAD_VIEW_STATUS["pending"]
            ph = ", ".join(f"${i}" for i in range(len(params) + 1, len(params) + 1 + len(pending)))
            params.extend(pending)
            count_params.extend(pending)
            select_sql += f" AND schedule_mode = 'smart' AND status IN ({ph})"
            count_sql += f" AND schedule_mode = 'smart' AND status IN ({ph})"
    elif status:
        params.append(status)
        count_params.append(status)
        select_sql += f" AND status = ${len(params)}"
        count_sql += f" AND status = ${len(count_params)}"

    if trill_only and "trill_score" in cols:
        select_sql += " AND trill_score IS NOT NULL"
        count_sql += " AND trill_score IS NOT NULL"

    params.extend([limit, offset])
    select_sql += f" ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}"

    def _normalize_hashtags(raw):
        tags = _safe_json(raw, [])
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        if isinstance(tags, str) and tags.strip():
            return [tags.strip()]
        return []

    s3 = None
    out = []
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(select_sql, *params)
        total = await conn.fetchval(count_sql, *count_params) if meta else None

        for r in rows:
            d = dict(r)

            ai_title = (d.get("ai_title") or d.get("ai_generated_title") or "") or ""
            ai_caption = (d.get("ai_caption") or d.get("ai_generated_caption") or "") or ""
            ai_hashtags = _normalize_hashtags(d.get("ai_generated_hashtags"))

            title = (d.get("title") or "").strip() or ai_title
            caption = (d.get("caption") or "").strip() or ai_caption

            hashtags = _normalize_hashtags(d.get("hashtags"))
            platform_results = await _enrich_platform_results(conn, d, str(user["id"]))

            thumbnail_url = None
            thumb_key = d.get("thumbnail_r2_key")
            if thumb_key:
                try:
                    s3 = s3 or get_s3_client()
                    thumbnail_url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(thumb_key)},
                        ExpiresIn=3600,
                    )
                except Exception:
                    thumbnail_url = None

            raw_status = d.get("status") or ""
            item = {
                "id": str(d.get("id")),
                "filename": d.get("filename"),
                "platforms": list(d.get("platforms") or []),
                "status": raw_status,
                "status_label": _STATUS_LABEL.get(raw_status, raw_status.replace("_", " ").title() if raw_status else "Unknown"),
                "privacy": d.get("privacy", "public"),
                "title": title,
                "caption": caption,
                "hashtags": hashtags,

                "ai_title": ai_title,
                "ai_caption": ai_caption,
                "ai_hashtags": ai_hashtags,

                "scheduled_time": d.get("scheduled_time").isoformat() if d.get("scheduled_time") else None,
                "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
                "completed_at": d.get("completed_at").isoformat() if d.get("completed_at") else None,

                "put_cost": int(d.get("put_reserved") or 0),
                "aic_cost": int(d.get("aic_reserved") or 0),

                "error_code": d.get("error_code"),
                "error": d.get("error_detail") or d.get("error_code") or None,

                "thumbnail_url": thumbnail_url,
                "platform_results": platform_results,

                "file_size": d.get("file_size"),
                "views":    int(d.get("views")    or 0),
                "likes":    int(d.get("likes")    or 0),
                "comments": int(d.get("comments") or 0),
                "shares":   int(d.get("shares")   or 0),

                "progress": int(d.get("processing_progress") or 0),
                "current_stage": d.get("processing_stage"),

                "schedule_mode":     d.get("schedule_mode") or "immediate",
                "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
                "smart_schedule":    _safe_json(d.get("schedule_metadata"), None),

                "is_editable": d.get("status") in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),

                "video_url": d.get("video_url"),
            }
            out.append(item)

    if not meta:
        return out

    return {"uploads": out, "total": int(total or 0), "limit": limit, "offset": offset}


@router.get("/queue-stats")
async def get_uploads_queue_stats(user: dict = Depends(get_current_user)):
    """
    Queue summary counts for queue.html and dashboard.html.
    Use these counts for Pending, Processing, Completed, Failed cards.
    Pending includes staged, queued, scheduled, ready_to_publish (smart + scheduled).
    """
    pending_statuses = _UPLOAD_VIEW_STATUS["pending"]
    completed_statuses = _UPLOAD_VIEW_STATUS["completed"]
    n_p, n_c = len(pending_statuses), len(completed_statuses)
    ph_p = ", ".join(f"${i}" for i in range(2, 2 + n_p))
    ph_c = ", ".join(f"${i}" for i in range(2 + n_p, 2 + n_p + n_c))
    params = [user["id"]] + list(pending_statuses) + list(completed_statuses)

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT
                SUM(CASE WHEN status IN ({ph_p}) THEN 1 ELSE 0 END)::int AS pending,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END)::int AS processing,
                SUM(CASE WHEN status IN ({ph_c}) THEN 1 ELSE 0 END)::int AS completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)::int AS failed
            FROM uploads WHERE user_id = $1
            """,
            *params,
        )
    return {
        "pending": row["pending"] or 0,
        "processing": row["processing"] or 0,
        "completed": row["completed"] or 0,
        "failed": row["failed"] or 0,
    }


@router.post("/{upload_id}/generate-thumbnail")
async def generate_thumbnail_for_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Backfill / regenerate the thumbnail for an existing upload.

    Workflow:
      1. Fetch the video from R2 to a temp file
      2. Run FFmpeg to extract a frame at 30% into the video
      3. Upload the JPEG to R2 at thumbnails/{user_id}/{upload_id}/thumbnail.jpg
      4. Update thumbnail_r2_key in the uploads row
      5. Return a fresh presigned URL

    This fixes the gap where uploads processed before the worker fix
    have thumbnail_r2_key = NULL in the database.
    """
    import tempfile, subprocess
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, r2_key, thumbnail_r2_key, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    # If thumbnail already exists, just return the presigned URL
    if row.get("thumbnail_r2_key"):
        try:
            s3 = get_s3_client()
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(row["thumbnail_r2_key"])},
                ExpiresIn=3600,
            )
            return {"thumbnail_url": url, "r2_key": row["thumbnail_r2_key"], "generated": False}
        except Exception:
            pass  # fall through and regenerate

    r2_key = row.get("r2_key")
    if not r2_key:
        raise HTTPException(400, "No video file key found for this upload")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        video_path = tmp_path / "video.mp4"
        thumb_path = tmp_path / "thumbnail.jpg"

        # 1. Download video from R2
        try:
            s3 = get_s3_client()
            s3.download_file(R2_BUCKET_NAME, _normalize_r2_key(r2_key), str(video_path))
        except Exception as e:
            raise HTTPException(500, f"Could not download video from storage: {e}")

        # 2. Get duration then extract frame at 30%
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            duration = 10.0
            if probe.returncode == 0:
                import json as _json
                for stream in _json.loads(probe.stdout).get("streams", []):
                    if stream.get("codec_type") == "video":
                        duration = float(stream.get("duration", 10) or 10)
                        break
            offset = max(0.5, duration * 0.30)
        except Exception:
            offset = 5.0

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{offset:.3f}",
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-q:v", "2",
                    "-vf", "scale=1080:-2",
                    str(thumb_path),
                ],
                capture_output=True, timeout=60
            )
            if result.returncode != 0 or not thumb_path.exists():
                # Fallback: try at 1 second
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", "1", "-i", str(video_path),
                     "-vframes", "1", "-q:v", "2", "-vf", "scale=1080:-2", str(thumb_path)],
                    capture_output=True, timeout=30
                )
        except Exception as e:
            raise HTTPException(500, f"FFmpeg thumbnail extraction failed: {e}")

        if not thumb_path.exists():
            raise HTTPException(500, "Thumbnail extraction produced no output")

        # 3. Upload to R2
        thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/thumbnail.jpg"
        try:
            s3.upload_file(
                str(thumb_path), R2_BUCKET_NAME, thumb_r2_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to upload thumbnail to storage: {e}")

        # 4. Update DB
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE uploads SET thumbnail_r2_key = $1, updated_at = NOW() WHERE id = $2",
                thumb_r2_key, upload_id
            )

        # 5. Return presigned URL
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": thumb_r2_key},
                ExpiresIn=3600,
            )
        except Exception:
            url = None

        return {
            "thumbnail_url": url,
            "r2_key": thumb_r2_key,
            "generated": True,
            "offset_seconds": offset,
        }


@router.post("/{upload_id}/thumbnail-presign")
async def presign_thumbnail_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Get a presigned URL for uploading a custom thumbnail.
    After uploading, call PATCH /api/uploads/{upload_id} with thumbnail_r2_key in the body (if supported).
    """
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
    if not row:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if row["status"] not in editable:
        raise HTTPException(400, "Cannot change thumbnail after upload is processing or published")

    thumb_r2_key = f"thumbnails/{user['id']}/{upload_id}/custom.jpg"
    presigned_url = generate_presigned_upload_url(thumb_r2_key, "image/jpeg")
    return {"presigned_url": presigned_url, "r2_key": thumb_r2_key}


@router.post("/{upload_id}/sync-analytics")
async def sync_upload_analytics(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Fetch latest engagement stats for a single completed upload from platform APIs.
    Uses the video IDs stored in platform_results to query per-video metrics.
    Updates the uploads table (views, likes, comments, shares) and returns fresh data.
    """
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platforms, platform_results, status FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"],
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if row["status"] not in ("completed", "succeeded", "partial"):
        return {"synced": False, "reason": "not_completed", "views": 0, "likes": 0, "comments": 0, "shares": 0}

    # Parse platform_results to get per-platform video IDs
    raw_pr = _safe_json(row["platform_results"], [])
    pr_list = []
    if isinstance(raw_pr, list):
        pr_list = [x for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, dict):
        pr_list = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw_pr.items()]
    # Alias canonical worker fields so lookups below find them
    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]
        if pr.get("platform_url") and not pr.get("url"):
            pr["url"] = pr["platform_url"]

    # Get tokens for all connected platforms (include id for multi-account lookup)
    async with core.state.db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )

    token_map_by_id = {}
    token_map_by_platform = {}
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if dec:
                if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                    dec["ig_user_id"] = str(tr["account_id"])
                if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                    dec["page_id"] = str(tr["account_id"])
                token_id = str(tr["id"])
                token_map_by_id[token_id] = dec
                token_map_by_platform[tr["platform"]] = dec
        except Exception:
            pass

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats = {}

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            ok   = pr.get("success") == True or str(pr.get("status","")).lower() in ("published","succeeded","success")
            if not ok:
                continue

            # Multi-account: use token for this specific account; fallback to platform
            account_id = pr.get("account_id")
            tok = token_map_by_id.get(str(account_id), {}) if account_id else {}
            if not tok:
                tok = token_map_by_platform.get(plat, {})
            access_token = tok.get("access_token", "")
            if not access_token:
                continue

            # platform_video_id is the canonical field written by db.py/mark_processing_completed
            # video_id / media_id / share_id etc. are legacy / webhook-written variants
            video_id = (
                pr.get("platform_video_id")  # canonical (worker pipeline)
                or pr.get("video_id") or pr.get("videoId") or pr.get("id")
                or pr.get("media_id") or pr.get("post_id") or pr.get("share_id")
            )

            try:
                if plat == "tiktok" and video_id:
                    resp = await client.post(
                        "https://open.tiktokapis.com/v2/video/query/",
                        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                        params={"fields": "id,view_count,like_count,comment_count,share_count"},
                        json={"filters": {"video_ids": [str(video_id)]}},
                    )
                    if resp.status_code == 200:
                        vids = resp.json().get("data", {}).get("videos", []) or []
                        if vids:
                            v = vids[0]
                            s = {"views": int(v.get("view_count") or 0), "likes": int(v.get("like_count") or 0),
                                 "comments": int(v.get("comment_count") or 0), "shares": int(v.get("share_count") or 0)}
                            platform_stats["tiktok"] = s
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]; total_shares  += s["shares"]

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
                            s = {"views": int(st.get("viewCount") or 0), "likes": int(st.get("likeCount") or 0),
                                 "comments": int(st.get("commentCount") or 0), "shares": 0}
                            platform_stats["youtube"] = s
                            total_views    += s["views"];    total_likes   += s["likes"]
                            total_comments += s["comments"]

                elif plat == "instagram" and video_id:
                    # Instagram Insights API requires numeric media_id (not shortcode)
                    media_id = pr.get("platform_video_id") or pr.get("media_id") or video_id
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{media_id}/insights",
                        params={"access_token": access_token,
                                "metric": "views,plays,likes,comments,saved,shares,reach"},
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        ig_views = ig_plays = 0
                        for m in resp.json().get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [])
                            val  = int(vals[-1].get("value", 0) if vals else m.get("value", 0) or 0)
                            if name == "views":       ig_views     = val
                            elif name == "plays":     ig_plays     = val  # deprecated fallback
                            elif name == "likes":     s["likes"]   += val
                            elif name == "comments":  s["comments"] += val
                            elif name == "shares":    s["shares"]  += val
                        s["views"] = ig_views or ig_plays  # prefer views over deprecated plays
                        platform_stats["instagram"] = s
                        total_views    += s["views"];    total_likes   += s["likes"]
                        total_comments += s["comments"]; total_shares  += s["shares"]

                elif plat == "facebook" and video_id:
                    page_id = tok.get("page_id", "")
                    resp = await client.get(
                        f"https://graph.facebook.com/v21.0/{video_id}",
                        params={"access_token": access_token,
                                "fields": "insights.metric(total_video_views,total_video_reactions_by_type_total,total_video_comments,total_video_shares)"},
                    )
                    if resp.status_code == 200:
                        s = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
                        for m in resp.json().get("insights", {}).get("data", []) or []:
                            name = m.get("name", "")
                            vals = m.get("values", [{}])
                            val  = vals[-1].get("value", 0) if vals else 0
                            if isinstance(val, dict): val = sum(val.values())
                            val = int(val or 0)
                            if name == "total_video_views":                      s["views"]    += val
                            elif name == "total_video_reactions_by_type_total":  s["likes"]    += val
                            elif name == "total_video_comments":                  s["comments"] += val
                            elif name == "total_video_shares":                    s["shares"]   += val
                        platform_stats["facebook"] = s
                        total_views    += s["views"];    total_likes   += s["likes"]
                        total_comments += s["comments"]; total_shares  += s["shares"]

            except Exception as e:
                logger.warning(f"sync-analytics error for {plat}/{video_id}: {e}")
                continue

    # Persist to DB
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE uploads SET views=$1, likes=$2, comments=$3, shares=$4, updated_at=NOW()
               WHERE id=$5 AND user_id=$6""",
            total_views, total_likes, total_comments, total_shares,
            upload_id, user["id"],
        )

    return {
        "synced": True,
        "views": total_views, "likes": total_likes,
        "comments": total_comments, "shares": total_shares,
        "platform_stats": platform_stats,
    }


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str, request: Request, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT put_reserved, aic_reserved, status, title, platforms FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user["id"])
        if not upload: raise HTTPException(404, "Upload not found")
        if upload["status"] in ("pending", "queued"):
            await refund_tokens(conn, user["id"], upload["put_reserved"], upload["aic_reserved"], upload_id)
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        await log_system_event(conn, user_id=str(user["id"]), action="UPLOAD_DELETED",
                               event_category="UPLOAD", resource_type="upload", resource_id=upload_id,
                               details={"title": upload["title"], "status_at_delete": upload["status"],
                                        "platforms": list(upload["platforms"] or [])},
                               request=request, severity="WARNING")
    return {"status": "deleted"}


@router.patch("/{upload_id}")
async def update_upload(
    upload_id: str,
    update_data: UploadUpdate,
    user: dict = Depends(get_current_user),
):
    """Update an upload's metadata: title, caption, hashtags, scheduled_time, smart_schedule."""
    async with core.state.db_pool.acquire() as conn:
        await _update_upload_metadata(conn, upload_id, user["id"], update_data)
    return {"status": "updated", "id": upload_id}


@router.post("/{upload_id}/retry")
async def retry_upload(upload_id: str, user: dict = Depends(get_current_user)):
    """Reset a failed/cancelled upload and re-queue it for processing."""
    async with core.state.db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        if not upload:
            raise HTTPException(404, "Upload not found")

        # Only allow retry for terminal states
        if upload["status"] not in ("failed", "cancelled"):
            raise HTTPException(400, "Only failed or cancelled uploads can be retried")

        # Reset processing state (keep engagement + cost fields intact)
        await conn.execute(
            """
            UPDATE uploads
            SET status = 'pending',
                error_code = NULL,
                error_detail = NULL,
                processing_started_at = NULL,
                processing_finished_at = NULL,
                completed_at = NULL,
                cancel_requested = FALSE,
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            """,
            upload_id, user["id"]
        )

        # Pull latest preferences (and respect plan entitlements)
        user_prefs = await get_user_prefs_for_upload(conn, user["id"])
        plan = get_plan(user.get("subscription_tier", "free"))

    job_data = {
        "job_id": str(uuid.uuid4()),
        "upload_id": upload_id,
        "user_id": str(user["id"]),
        "preferences": user_prefs,
        "plan_features": {
            "ai": plan.get("ai", False),
            "priority": plan.get("priority", False),
            "watermark": plan.get("watermark", True),
        },
        "action": "retry",
    }

    await enqueue_job(job_data, priority=plan.get("priority", False))
    return {"status": "requeued", "upload_id": upload_id}


@router.get("/{upload_id}")
async def get_upload_details(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Upload detail for current user.

    Contract (frontend-safe):
      - thumbnail_url: presigned R2 URL (if thumbnail_r2_key exists)
      - platform_results: always list
      - hashtags: always list[str]
      - title/caption: falls back to AI values when empty
      - ai_title/ai_caption/ai_hashtags always present
      - duration_seconds computed from processing timestamps when available
    """
    cols = await _load_uploads_columns(core.state.db_pool)
    wanted = [
        "id","user_id","r2_key","filename","file_size","platforms",
        "title","caption","hashtags","privacy","status",
        "scheduled_time","created_at","completed_at",
        "put_reserved","aic_reserved",
        "error_code","error_detail",
        "thumbnail_r2_key","platform_results",
        "processing_started_at","processing_finished_at",
        "processing_stage","processing_progress",
        "views","likes",
        "schedule_mode","schedule_metadata","timezone",
        # AI fields (older/newer schema variants)
        "ai_title","ai_caption",
        "ai_generated_title","ai_generated_caption","ai_generated_hashtags",
    ]
    select_cols = _pick_cols(wanted, cols) or ["id","user_id","r2_key","filename","platforms","status","created_at"]
    sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE id = $1 AND user_id = $2"

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(sql, upload_id, user["id"])

    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")

    d = dict(row)

    def _normalize_platform_results(raw):
        pr = _safe_json(raw, [])
        if isinstance(pr, list):
            items = [x for x in pr if isinstance(x, dict)]
        elif isinstance(pr, dict):
            items = []
            for k, v in pr.items():
                if isinstance(v, dict):
                    items.append({"platform": k, **v})
                else:
                    items.append({"platform": k, "value": v})
        else:
            return []
        out = []
        for item in items:
            d = dict(item)
            if d.get("platform_video_id") and not d.get("video_id"):
                d["video_id"] = d["platform_video_id"]
            if d.get("platform_url") and not d.get("url"):
                d["url"] = d["platform_url"]
            if d.get("account_id") and not d.get("token_id"):
                d["token_id"] = d["account_id"]
            out.append(d)
        return out

    def _normalize_hashtags(raw):
        tags = _safe_json(raw, [])
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        if isinstance(tags, str) and tags.strip():
            return [tags.strip()]
        return []

    ai_title = (d.get("ai_title") or d.get("ai_generated_title") or "") or ""
    ai_caption = (d.get("ai_caption") or d.get("ai_generated_caption") or "") or ""
    ai_hashtags = _normalize_hashtags(d.get("ai_generated_hashtags"))

    title = (d.get("title") or "").strip() or ai_title
    caption = (d.get("caption") or "").strip() or ai_caption
    hashtags = _normalize_hashtags(d.get("hashtags"))
    platform_results = _normalize_platform_results(d.get("platform_results"))

    thumbnail_url = None
    thumb_key = d.get("thumbnail_r2_key")
    if thumb_key:
        try:
            s3 = get_s3_client()
            thumbnail_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(thumb_key)},
                ExpiresIn=3600,
            )
        except Exception:
            thumbnail_url = None

    duration_seconds = None
    ps = d.get("processing_started_at")
    pf = d.get("processing_finished_at")
    if ps and pf:
        try:
            duration_seconds = int((pf - ps).total_seconds())
        except Exception:
            duration_seconds = None

    return {
        "id": str(d.get("id")),
        "filename": d.get("filename"),
        "r2_key": d.get("r2_key"),
        "platforms": list(d.get("platforms") or []),
        "status": d.get("status"),
        "privacy": d.get("privacy", "public"),

        "title": title,
        "caption": caption,
        "hashtags": hashtags,

        "ai_title": ai_title,
        "ai_caption": ai_caption,
        "ai_hashtags": ai_hashtags,

        "scheduled_time": d.get("scheduled_time").isoformat() if d.get("scheduled_time") else None,
        "schedule_mode": d.get("schedule_mode") or "immediate",
        "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
        "smart_schedule": _safe_json(d.get("schedule_metadata"), None),  # alias for queue.html edit modal
        "timezone": d.get("timezone") or "UTC",
        "is_editable": d.get("status") in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
        "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
        "completed_at": d.get("completed_at").isoformat() if d.get("completed_at") else None,

        "put_cost": int(d.get("put_reserved") or 0),
        "aic_cost": int(d.get("aic_reserved") or 0),

        "error_code": d.get("error_code"),
        "error": d.get("error_detail") or d.get("error_code") or None,

        "thumbnail_url": thumbnail_url,
        "platform_results": platform_results,

        "file_size": d.get("file_size"),
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),

        "progress": int(d.get("processing_progress") or 0),
        "current_stage": d.get("processing_stage"),
        "duration_seconds": duration_seconds,

        # camelCase aliases required by upload.html pollUpload() tick loop
        # The poller keys on processingStartedAt to flip from Queued -> Processing
        "processingStartedAt":  d.get("processing_started_at").isoformat() if d.get("processing_started_at") else None,
        "processingFinishedAt": d.get("processing_finished_at").isoformat() if d.get("processing_finished_at") else None,
        "processingProgress":   int(d.get("processing_progress") or 0),
        "processingStage":      d.get("processing_stage"),
    }
