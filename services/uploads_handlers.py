"""
Upload route business logic: presign insert, complete transaction, list/detail shaping, queue stats.

Routers keep HTTP wiring (cookies, Request, presigned URL generation, enqueue, audit).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException

from core.config import R2_BUCKET_NAME
from core.helpers import (
    _load_uploads_columns,
    _pick_cols,
    _safe_col,
    _safe_json,
    ensure_uploads_columns_loaded,
    sanitize_hashtag_body,
)
from core.models import UploadInit
from core.r2 import _normalize_r2_key, get_s3_client, r2_object_exists
from core.scheduling import get_existing_scheduled_days
from services.smart_schedule_insights import calculate_smart_schedule_data_driven
from core.sql_allowlist import UPLOADS_COMPLETE_BODY_COLUMNS, assert_set_fragments_columns
from core.wallet import atomic_reserve_tokens, get_wallet
from routers.preferences import get_user_prefs_for_upload
from services.thumbnail_regenerate import ensure_upload_thumbnail_resident
from services.uploads_api import enrich_platform_results_batch
from stages.ai_service_costs import compute_presign_put_aic_costs
from stages.context import is_placeholder_upload_caption, is_placeholder_upload_title
from stages.entitlements import entitlements_to_dict, get_entitlements_from_user

# Parallel R2 HEAD checks for list payloads — sequential per-row checks dominated bootstrap latency.
_THUMBNAIL_R2_EXISTS_MAX_CONCURRENCY = 32


async def _batch_r2_object_exists(keys: List[str], *, concurrency: int = _THUMBNAIL_R2_EXISTS_MAX_CONCURRENCY) -> List[bool]:
    if not keys:
        return []
    sem = asyncio.Semaphore(concurrency)

    async def one(k: str) -> bool:
        async with sem:
            try:
                return await asyncio.to_thread(r2_object_exists, k)
            except Exception:
                return False

    return list(await asyncio.gather(*(one(k) for k in keys)))


def merge_upload_init_thumbnail_preferences(user_prefs: Dict[str, Any], data: Any) -> None:
    """Overlay presign-body thumbnail toggles onto the snapshot stored on ``uploads.user_preferences``."""
    use_eng = getattr(data, "thumbnail_use_studio_engine", None)
    if use_eng is not None:
        v = bool(use_eng)
        user_prefs["thumbnail_studio_engine_enabled"] = v
        user_prefs["thumbnailStudioEngineEnabled"] = v
        # Worker thumbnail stage treats Pikzels v2 as the studio engine; keep legacy
        # keys aligned when the uploader opts into Aurora / studio for this job.
        if v:
            user_prefs["thumbnail_pikzels_enabled"] = True
            user_prefs["thumbnailPikzelsEnabled"] = True
    use_pkz = getattr(data, "thumbnail_use_pikzels", None)
    if use_pkz is not None:
        v = bool(use_pkz)
        user_prefs["thumbnail_pikzels_enabled"] = v
        user_prefs["thumbnailPikzelsEnabled"] = v

    use_per = getattr(data, "thumbnail_use_persona", None)
    if use_per is True:
        user_prefs["thumbnail_persona_enabled"] = True
        user_prefs["thumbnailPersonaEnabled"] = True
    elif use_per is False:
        user_prefs["thumbnail_persona_enabled"] = False
        user_prefs["thumbnailPersonaEnabled"] = False
        user_prefs.pop("thumbnail_default_persona_id", None)
        user_prefs.pop("thumbnailDefaultPersonaId", None)

    pid = getattr(data, "thumbnail_persona_id", None)
    if pid and str(pid).strip():
        s = str(pid).strip()
        user_prefs["thumbnail_default_persona_id"] = s
        user_prefs["thumbnailDefaultPersonaId"] = s

    pst = getattr(data, "thumbnail_persona_strength", None)
    if pst is not None:
        try:
            v = max(0, min(100, int(pst)))
        except (TypeError, ValueError):
            v = 70
        user_prefs["thumbnail_persona_strength"] = v
        user_prefs["thumbnailPersonaStrength"] = v


def _json_for_upload_row(obj: Any) -> str:
    """DB-backed dicts may contain datetime/UUID/decimals — asyncpg returns native types."""
    return json.dumps(obj, default=str)


def telemetry_r2_key_for_upload(user_id: str, upload_id: str, has_telemetry: bool) -> Optional[str]:
    """Companion .map object key saved on uploads so the worker can fetch it."""
    if not has_telemetry:
        return None
    return f"uploads/{user_id}/{upload_id}/telemetry.map"


def _schedule_slot_iso(v: Any) -> str:
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


def youtube_copyright_shorts_notice_from_artifacts(raw: Any) -> Optional[dict]:
    """Parse ``youtube_copyright_shorts`` blob from uploads.output_artifacts (jsonb)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    v = raw.get("youtube_copyright_shorts")
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def platform_thumbnail_urls_from_artifacts(raw: Any, expires_in: int = 3600) -> dict:
    """Return presigned per-platform styled thumbnail URLs from upload artifacts."""
    artifacts = _safe_json(raw, {})
    if not isinstance(artifacts, dict):
        return {}
    raw_keys = artifacts.get("platform_thumbnail_r2_keys") or {}
    if isinstance(raw_keys, str):
        raw_keys = _safe_json(raw_keys, {})
    if not isinstance(raw_keys, dict) or not raw_keys:
        return {}
    try:
        s3 = get_s3_client()
    except Exception:
        return {}
    out = {}
    for platform, key in raw_keys.items():
        plat = str(platform or "").strip().lower()
        k = str(key or "").strip()
        if not plat or not k:
            continue
        try:
            out[plat] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(k)},
                ExpiresIn=expires_in,
            )
        except Exception:
            continue
    return out

# Status view groupings for queue/dashboard.
#
# UI contract (see frontend queue.html / dashboard.html):
#   processing — anything not yet finalised: pending, scheduled, smart-scheduled,
#                future uploads, staged, queued, ready_to_publish, currently processing.
#   completed  — fully successful uploads only (every platform succeeded).
#   partial    — at least one platform succeeded AND at least one failed.
#   failed     — every platform failed (or upload errored before publish).
#
# `pending` / `staged` are kept as aliases so legacy callers (older clients,
# admin tooling, scheduled.html stats) keep working.
# ── Canonical status buckets (single source of truth) ──────────────────────────
#
# These tuples are imported by routers, services, and serialized into shared
# constants for the frontend (frontend/js/scheduled-status.js) so dashboard,
# queue, and scheduled pages all agree on what counts as "scheduled" / etc.
#
# DO NOT inline these literals in SQL elsewhere. Import from here.

# "Scheduled" / "in the pipeline, not yet started publishing" — the canonical
# definition shared by dashboard.scheduled, scheduled.html list+stats, queue
# pending tab, and edit-permission checks.
SCHEDULED_PIPELINE_STATUSES: tuple[str, ...] = (
    "pending",
    "scheduled",
    "queued",
    "staged",
    "ready_to_publish",
)

# Currently doing publish work (one bucket past scheduled).
PROCESSING_STATUSES: tuple[str, ...] = ("processing",)

# "In the queue / processing tab" = scheduled + actively processing.
QUEUE_VIEW_STATUSES: tuple[str, ...] = SCHEDULED_PIPELINE_STATUSES + PROCESSING_STATUSES

# Terminal-success and terminal-non-success buckets.
COMPLETED_STATUSES: tuple[str, ...] = ("completed", "succeeded")
PARTIAL_STATUSES: tuple[str, ...] = ("partial",)
FAILED_STATUSES: tuple[str, ...] = ("failed",)


def scheduled_in_clause(start_param_idx: int) -> tuple[str, list[str]]:
    """Return ``("$2,$3,...", [statuses])`` for inlining the canonical
    SCHEDULED_PIPELINE_STATUSES into a parametrized SQL ``IN (...)`` clause.

    ``start_param_idx`` is the asyncpg ``$N`` index of the first status param.
    """
    statuses = list(SCHEDULED_PIPELINE_STATUSES)
    placeholders = ", ".join(f"${i}" for i in range(start_param_idx, start_param_idx + len(statuses)))
    return placeholders, statuses


UPLOAD_VIEW_STATUS: Dict[str, Any] = {
    "processing": QUEUE_VIEW_STATUSES,
    "completed": COMPLETED_STATUSES,
    "partial": PARTIAL_STATUSES,
    "failed": FAILED_STATUSES,
    # Legacy aliases — same canonical scheduled bucket. Do not remove without
    # sweeping callers (front-end and admin tooling still read these keys).
    "pending": SCHEDULED_PIPELINE_STATUSES,
    "staged": SCHEDULED_PIPELINE_STATUSES,
    "scheduled": SCHEDULED_PIPELINE_STATUSES,
    "smart_schedule": None,
}

UPLOAD_STATUS_LABEL = {
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

ALLOWED_VIDEO_TYPES = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
        "video/x-matroska",
    }
)


def _split_tags(v: Any) -> List[str]:
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


def _to_hash_tags(v: Any) -> List[str]:
    out: List[str] = []
    for t in _split_tags(v):
        body = sanitize_hashtag_body(t)
        if body:
            out.append(f"#{body}")
    return out


def _normalize_hashtags_list(raw: Any) -> List[str]:
    tags = _safe_json(raw, [])
    if isinstance(tags, list):
        out: List[str] = []
        for t in tags:
            if not t:
                continue
            body = sanitize_hashtag_body(str(t))
            if body:
                out.append(f"#{body}")
        return out
    if isinstance(tags, str) and tags.strip():
        body = sanitize_hashtag_body(tags)
        return [f"#{body}"] if body else []
    return []


def _normalize_platform_results_detail(raw: Any) -> List[dict]:
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
    out: List[dict] = []
    for item in items:
        row = dict(item)
        if row.get("platform_video_id") and not row.get("video_id"):
            row["video_id"] = row["platform_video_id"]
        if row.get("platform_url") and not row.get("url"):
            row["url"] = row["platform_url"]
        if row.get("account_id") and not row.get("token_id"):
            row["token_id"] = row["account_id"]
        out.append(row)
    return out


def compute_smart_schedule_display(schedule_mode: str, schedule_metadata: Any) -> Optional[dict]:
    if schedule_mode != "smart" or not schedule_metadata:
        return None
    try:
        sm = schedule_metadata if isinstance(schedule_metadata, dict) else json.loads(schedule_metadata)
        return {p: v for p, v in sm.items()}
    except Exception:
        return None


async def presign_create_upload(conn, data: UploadInit, user: dict) -> dict:
    """
    Insert upload row and reserve tokens. Mutates ``data`` (hashtags, privacy, defaults).

    Returns dict with upload_id, r2_key, put_cost, aic_cost, user_prefs, smart_schedule (or None).
    """
    db_ent = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user["id"],
    )
    user_for_ent = dict(user)
    if db_ent:
        for _k in ("subscription_tier", "role", "flex_enabled"):
            _v = db_ent.get(_k)
            if _v is not None:
                user_for_ent[_k] = _v
    ent_cost = get_entitlements_from_user(user_for_ent)
    plan = entitlements_to_dict(ent_cost)

    if getattr(data, "hashtags", None) is None:
        data.hashtags = []
    if getattr(data, "platforms", None) is None:
        data.platforms = []

    user_prefs = await get_user_prefs_for_upload(conn, user["id"])
    merge_upload_init_thumbnail_preferences(user_prefs, data)

    if not getattr(data, "privacy", None):
        data.privacy = user_prefs["default_privacy"]

    combined = _to_hash_tags(getattr(data, "hashtags", []) or [])

    if user_prefs.get("ai_hashtags_enabled") and plan.get("ai"):
        pass

    blocked = set(
        _to_hash_tags(user_prefs.get("blocked_hashtags", []) or user_prefs.get("blockedHashtags", []))
    )
    combined = [h for h in combined if h and h not in blocked]
    data.hashtags = list(dict.fromkeys(combined))[: int(user_prefs.get("max_hashtags", 30))]

    use_ai_checkbox = bool(getattr(data, "use_ai", False))
    use_hud = bool(user_prefs.get("hud_enabled", False)) and ent_cost.can_burn_hud

    num_publish_targets = len(data.target_accounts) if data.target_accounts else len(data.platforms)
    put_cost, aic_cost = compute_presign_put_aic_costs(
        ent_cost,
        num_publish_targets=num_publish_targets,
        file_size=getattr(data, "file_size", None),
        duration_hint=None,
        has_telemetry=bool(getattr(data, "has_telemetry", False)),
        use_ai_checkbox=use_ai_checkbox,
        hud_enabled_effective=use_hud,
        user_prefs=user_prefs,
        num_thumbnails_override=None,
    )

    pending_count = await conn.fetchval(
        """SELECT COUNT(*) FROM uploads
           WHERE user_id = $1
           AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
        user["id"],
    )
    if pending_count >= ent_cost.queue_depth:
        raise HTTPException(
            429,
            f"Queue limit reached ({pending_count}/{ent_cost.queue_depth} uploads pending). "
            "Wait for existing uploads to complete or upgrade your plan.",
        )

    upload_id = str(uuid.uuid4())
    r2_key = f"uploads/{user['id']}/{upload_id}/{data.filename}"
    telemetry_r2_key = telemetry_r2_key_for_upload(
        str(user["id"]),
        upload_id,
        bool(getattr(data, "has_telemetry", False)),
    )

    smart_schedule = None
    if getattr(data, "schedule_mode", None) == "smart":
        days = getattr(data, "smart_schedule_days", 7)
        blocked = await get_existing_scheduled_days(conn, user["id"], days)
        smart_schedule = await calculate_smart_schedule_data_driven(
            conn,
            str(user["id"]),
            data.platforms,
            num_days=days,
            blocked_day_offsets=blocked or None,
        )

    scheduled_time = getattr(data, "scheduled_time", None)
    schedule_metadata = None

    if getattr(data, "schedule_mode", None) == "smart" and smart_schedule:
        schedule_metadata = {p: _schedule_slot_iso(dt) for p, dt in smart_schedule.items()}
        scheduled_time = min(smart_schedule.values())

    await conn.execute(
        """
            INSERT INTO uploads (
                id, user_id, r2_key, telemetry_r2_key, filename, file_size, platforms,
                title, caption, hashtags, privacy, status, scheduled_time,
                schedule_mode, put_reserved, aic_reserved, schedule_metadata,
                user_preferences, target_accounts
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending', $12, $13, $14, $15, $16, $17, $18)
        """,
        upload_id,
        user["id"],
        r2_key,
        telemetry_r2_key,
        data.filename,
        data.file_size,
        data.platforms,
        data.title,
        data.caption,
        data.hashtags,
        data.privacy,
        scheduled_time,
        data.schedule_mode,
        put_cost,
        aic_cost,
        _json_for_upload_row(schedule_metadata) if schedule_metadata else None,
        _json_for_upload_row(user_prefs),
        data.target_accounts or [],
    )

    reserved = await atomic_reserve_tokens(conn, user["id"], put_cost, aic_cost, upload_id)
    if not reserved:
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        fresh_wallet = await get_wallet(conn, user["id"])
        put_avail = fresh_wallet["put_balance"] - fresh_wallet["put_reserved"]
        aic_avail = fresh_wallet["aic_balance"] - fresh_wallet["aic_reserved"]
        if put_avail < put_cost:
            raise HTTPException(
                429,
                {
                    "code": "insufficient_put",
                    "message": f"Insufficient PUT tokens ({put_avail} available, {put_cost} needed).",
                    "topup_url": "/settings.html#billing",
                },
            )
        raise HTTPException(
            429,
            {
                "code": "insufficient_aic",
                "message": f"Insufficient AIC credits ({aic_avail} available, {aic_cost} needed).",
                "topup_url": "/settings.html#billing",
            },
        )

    return {
        "upload_id": upload_id,
        "r2_key": r2_key,
        "telemetry_r2_key": telemetry_r2_key,
        "put_cost": put_cost,
        "aic_cost": aic_cost,
        "user_prefs": user_prefs,
        "smart_schedule": smart_schedule,
    }


async def complete_upload_transaction(conn, upload_id: str, user_id: str, body: dict) -> dict:
    """
    Apply optional title/caption/hashtags, set staged or queued. Returns:
    new_status, schedule_mode, upload (dict snapshot from initial fetch), user_prefs.
    """
    upload = await conn.fetchrow(
        "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id,
        user_id,
    )
    if not upload:
        raise HTTPException(404, "Upload not found")

    user_prefs = await get_user_prefs_for_upload(conn, user_id)

    schedule_mode = upload.get("schedule_mode") or "immediate"

    _COMPLETE_COLS = UPLOADS_COMPLETE_BODY_COLUMNS
    updates: List[str] = []
    params: List[Any] = []
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
        tags: List[str] = []
        for t in (raw_tags if isinstance(raw_tags, (list, tuple)) else []):
            tag_body = sanitize_hashtag_body(str(t))
            if tag_body:
                tags.append(f"#{tag_body}")
        blocked = set(
            str(x).strip().lstrip("#").lower()
            for x in (user_prefs.get("blocked_hashtags") or user_prefs.get("blockedHashtags") or [])
        )
        tags = [t for t in tags if t and t.lstrip("#").lower() not in blocked]
        tags = list(dict.fromkeys(tags))[: int(user_prefs.get("max_hashtags", 30))]
        updates.append(f"{_safe_col('hashtags', _COMPLETE_COLS)} = ${idx}")
        params.append(tags)
        idx += 1

    if updates:
        assert_set_fragments_columns(updates, UPLOADS_COMPLETE_BODY_COLUMNS)
        params.append(upload_id)
        await conn.execute(
            f"UPDATE uploads SET {', '.join(updates)}, updated_at = NOW() WHERE id = ${idx}",
            *params,
        )

    if schedule_mode in ("scheduled", "smart"):
        new_status = "staged"
        await conn.execute(
            "UPDATE uploads SET status = 'staged', updated_at = NOW() WHERE id = $1",
            upload_id,
        )
    else:
        new_status = "queued"
        await conn.execute(
            "UPDATE uploads SET status = 'queued', updated_at = NOW() WHERE id = $1",
            upload_id,
        )

    return {
        "new_status": new_status,
        "schedule_mode": schedule_mode,
        "upload": dict(upload),
        "user_prefs": user_prefs,
    }


async def fetch_user_uploads_list(
    pool,
    user_id: str,
    *,
    status: Optional[str],
    view: Optional[str],
    limit: int,
    offset: int,
    trill_only: bool,
    meta: bool,
) -> Union[list, dict]:
    cols = await _load_uploads_columns(pool)

    wanted = [
        "id",
        "filename",
        "platforms",
        "status",
        "privacy",
        "title",
        "caption",
        "hashtags",
        "scheduled_time",
        "created_at",
        "completed_at",
        "put_reserved",
        "aic_reserved",
        "error_code",
        "error_detail",
        "thumbnail_r2_key",
        "r2_key",
        "processed_r2_key",
        "user_preferences",
        "platform_results",
        "file_size",
        "processing_started_at",
        "processing_finished_at",
        "updated_at",
        "processing_stage",
        "processing_progress",
        "views",
        "likes",
        "comments",
        "shares",
        "schedule_mode",
        "schedule_metadata",
        "video_url",
        "ai_title",
        "ai_caption",
        "ai_generated_title",
        "ai_generated_caption",
        "ai_generated_hashtags",
        "target_accounts",
        "trill_score",
        "speed_bucket",
        "trill_metadata",
        "output_artifacts",
    ]
    select_cols = _pick_cols(wanted, cols) or ["id", "filename", "platforms", "status", "created_at"]
    select_sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE user_id = $1"
    count_sql = "SELECT COUNT(*) FROM uploads WHERE user_id = $1"
    params: List[Any] = [user_id]
    count_params: List[Any] = [user_id]

    if view == "all":
        pass
    elif view and view in UPLOAD_VIEW_STATUS:
        statuses = UPLOAD_VIEW_STATUS[view]
        if statuses is not None:
            placeholders = ", ".join(f"${i}" for i in range(len(params) + 1, len(params) + 1 + len(statuses)))
            params.extend(statuses)
            count_params.extend(statuses)
            select_sql += f" AND status IN ({placeholders})"
            count_sql += f" AND status IN ({placeholders})"
        else:
            pending = UPLOAD_VIEW_STATUS["pending"]
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
    select_sql += f" ORDER BY created_at DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}"

    s3 = None
    out: List[dict] = []
    async with pool.acquire() as conn:
        rows = await conn.fetch(select_sql, *params)
        total = await conn.fetchval(count_sql, *count_params) if meta else None

        row_dicts = [dict(r) for r in rows]
        enriched_pr = await enrich_platform_results_batch(conn, row_dicts, str(user_id))

        for d, platform_results in zip(row_dicts, enriched_pr):
            ai_title = (d.get("ai_generated_title") or d.get("ai_title") or "") or ""
            ai_caption = (d.get("ai_generated_caption") or d.get("ai_caption") or "") or ""
            ai_hashtags = _normalize_hashtags_list(d.get("ai_generated_hashtags"))

            raw_title = (d.get("title") or "").strip()
            title = ai_title if ai_title and is_placeholder_upload_title(raw_title, d.get("filename") or "") else (raw_title or ai_title)
            raw_caption = (d.get("caption") or "").strip()
            caption = ai_caption if ai_caption and is_placeholder_upload_caption(raw_caption) else (raw_caption or ai_caption)

            hashtags = _normalize_hashtags_list(d.get("hashtags"))

            # Generate presigned URL optimistically — no R2 HEAD check here; any missing object
            # is surfaced by an image onerror on the client, which then triggers backfill.
            thumbnail_url = None
            thumb_key = d.get("thumbnail_r2_key")
            sk = str(thumb_key).strip() if thumb_key else ""
            if sk:
                try:
                    s3 = s3 or get_s3_client()
                    thumbnail_url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(sk)},
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
                "status_label": UPLOAD_STATUS_LABEL.get(
                    raw_status, raw_status.replace("_", " ").title() if raw_status else "Unknown"
                ),
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
                "thumbnail_storage_missing": not bool(thumbnail_url) and bool(sk),
                "platform_thumbnail_urls": platform_thumbnail_urls_from_artifacts(d.get("output_artifacts")),
                "platform_results": platform_results,
                "file_size": d.get("file_size"),
                "views": int(d.get("views") or 0),
                "likes": int(d.get("likes") or 0),
                "comments": int(d.get("comments") or 0),
                "shares": int(d.get("shares") or 0),
                "progress": int(d.get("processing_progress") or 0),
                "current_stage": d.get("processing_stage"),
                "processing_started_at": d.get("processing_started_at").isoformat()
                if d.get("processing_started_at")
                else None,
                "processingStartedAt": d.get("processing_started_at").isoformat()
                if d.get("processing_started_at")
                else None,
                "updated_at": d.get("updated_at").isoformat()
                if d.get("updated_at")
                else None,
                "updatedAt": d.get("updated_at").isoformat()
                if d.get("updated_at")
                else None,
                "schedule_mode": d.get("schedule_mode") or "immediate",
                "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
                "smart_schedule": _safe_json(d.get("schedule_metadata"), None),
                "is_editable": d.get("status")
                in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
                "video_url": d.get("video_url"),
                "trill_score": float(d["trill_score"]) if d.get("trill_score") is not None else None,
                "speed_bucket": d.get("speed_bucket"),
                "trill_metadata": _safe_json(d.get("trill_metadata"), None),
                "youtubeCopyrightShorts": youtube_copyright_shorts_notice_from_artifacts(
                    d.get("output_artifacts")
                ),
            }
            out.append(item)

    if not meta:
        return out

    return {"uploads": out, "total": int(total or 0), "limit": limit, "offset": offset}


async def fetch_upload_queue_stats(pool, user_id: str) -> dict:
    """Counts for the 4 queue/dashboard buckets.

    Buckets mirror UPLOAD_VIEW_STATUS:
        processing — pending + scheduled + queued + staged + ready_to_publish + processing
        completed  — completed / succeeded only (no partial)
        partial    — partial
        failed     — failed
    The ``pending`` key is kept (mirrors processing) so older clients that still
    read ``pending`` keep working until they migrate.
    """
    processing_statuses = UPLOAD_VIEW_STATUS["processing"]
    completed_statuses = UPLOAD_VIEW_STATUS["completed"]
    n_proc, n_comp = len(processing_statuses), len(completed_statuses)
    ph_proc = ", ".join(f"${i}" for i in range(2, 2 + n_proc))
    ph_comp = ", ".join(
        f"${i}" for i in range(2 + n_proc, 2 + n_proc + n_comp)
    )
    params = [user_id] + list(processing_statuses) + list(completed_statuses)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT
                SUM(CASE WHEN status IN ({ph_proc}) THEN 1 ELSE 0 END)::int AS processing,
                SUM(CASE WHEN status IN ({ph_comp}) THEN 1 ELSE 0 END)::int AS completed,
                SUM(CASE WHEN status = 'partial'  THEN 1 ELSE 0 END)::int AS partial,
                SUM(CASE WHEN status = 'failed'   THEN 1 ELSE 0 END)::int AS failed
            FROM uploads WHERE user_id = $1
            """,
            *params,
        )
    processing = int(row["processing"] or 0)
    return {
        "processing": processing,
        "completed":  int(row["completed"] or 0),
        "partial":    int(row["partial"]   or 0),
        "failed":     int(row["failed"]    or 0),
        # Back-compat: legacy clients still read "pending"; mirror processing
        # so the old card never goes blank during a frontend rollout window.
        "pending":    processing,
    }


def build_upload_detail_payload(d: dict) -> dict:
    """Shape one uploads row dict for GET /api/uploads/{id} (no DB)."""
    ai_title = (d.get("ai_generated_title") or d.get("ai_title") or "") or ""
    ai_caption = (d.get("ai_generated_caption") or d.get("ai_caption") or "") or ""
    ai_hashtags = _normalize_hashtags_list(d.get("ai_generated_hashtags"))

    raw_title = (d.get("title") or "").strip()
    title = ai_title if ai_title and is_placeholder_upload_title(raw_title, d.get("filename") or "") else (raw_title or ai_title)
    raw_caption = (d.get("caption") or "").strip()
    caption = ai_caption if ai_caption and is_placeholder_upload_caption(raw_caption) else (raw_caption or ai_caption)
    hashtags = _normalize_hashtags_list(d.get("hashtags"))
    platform_results = _normalize_platform_results_detail(d.get("platform_results"))

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
        "smart_schedule": _safe_json(d.get("schedule_metadata"), None),
        "timezone": d.get("timezone") or "UTC",
        "is_editable": d.get("status")
        in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
        "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
        "completed_at": d.get("completed_at").isoformat() if d.get("completed_at") else None,
        "put_cost": int(d.get("put_reserved") or 0),
        "aic_cost": int(d.get("aic_reserved") or 0),
        "error_code": d.get("error_code"),
        "error": d.get("error_detail") or d.get("error_code") or None,
        "thumbnail_url": thumbnail_url,
        "platform_thumbnail_urls": platform_thumbnail_urls_from_artifacts(d.get("output_artifacts")),
        "platform_results": platform_results,
        "file_size": d.get("file_size"),
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),
        "progress": int(d.get("processing_progress") or 0),
        "current_stage": d.get("processing_stage"),
        "duration_seconds": duration_seconds,
        "processingStartedAt": d.get("processing_started_at").isoformat()
        if d.get("processing_started_at")
        else None,
        "processingFinishedAt": d.get("processing_finished_at").isoformat()
        if d.get("processing_finished_at")
        else None,
        "processingProgress": int(d.get("processing_progress") or 0),
        "processingStage": d.get("processing_stage"),
        "trill_score": float(d["trill_score"]) if d.get("trill_score") is not None else None,
        "speed_bucket": d.get("speed_bucket"),
        "trill_metadata": _safe_json(d.get("trill_metadata"), None),
        "youtubeCopyrightShorts": youtube_copyright_shorts_notice_from_artifacts(
            d.get("output_artifacts")
        ),
    }


async def fetch_upload_detail(pool, upload_id: str, user_id: str) -> dict:
    """
    Load upload detail for ``user_id`` after the same auth gates as ``get_current_user_readonly``,
    using a single pooled connection (upload row + optional recognition + cold schema probe).
    Caller should pass a JWT-verified ``user_id`` (e.g. from ``get_verified_user_id``).
    """
    wanted = [
        "id",
        "user_id",
        "r2_key",
        "filename",
        "file_size",
        "platforms",
        "title",
        "caption",
        "hashtags",
        "privacy",
        "status",
        "scheduled_time",
        "created_at",
        "completed_at",
        "put_reserved",
        "aic_reserved",
        "error_code",
        "error_detail",
        "thumbnail_r2_key",
        "r2_key",
        "processed_r2_key",
        "user_preferences",
        "platform_results",
        "processing_started_at",
        "processing_finished_at",
        "processing_stage",
        "processing_progress",
        "views",
        "likes",
        "schedule_mode",
        "schedule_metadata",
        "timezone",
        "ai_title",
        "ai_caption",
        "ai_generated_title",
        "ai_generated_caption",
        "ai_generated_hashtags",
        "trill_score",
        "speed_bucket",
        "trill_metadata",
        "output_artifacts",
    ]
    thumb_user: Optional[Dict[str, Any]] = None
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, status, email_verified, subscription_tier, role, flex_enabled FROM users WHERE id = $1",
            user_id,
        )
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        if user["status"] == "banned":
            raise HTTPException(status_code=403, detail="Account suspended")
        if user.get("email_verified") is False:
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Please verify your email to use the app.",
                    "code": "email_not_verified",
                },
            )

        thumb_user = {
            "subscription_tier": user.get("subscription_tier"),
            "role": user.get("role"),
            "flex_enabled": user.get("flex_enabled"),
        }

        cols = await ensure_uploads_columns_loaded(conn)
        select_cols = _pick_cols(wanted, cols) or [
            "id",
            "user_id",
            "r2_key",
            "filename",
            "platforms",
            "status",
            "created_at",
        ]
        sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE id = $1 AND user_id = $2"

        row = await conn.fetchrow(sql, upload_id, user_id)
        recognition_row = None
        try:
            recognition_row = await conn.fetchrow(
                """
                SELECT object_track_count, person_segment_count,
                       text_detection_count, logo_count,
                       top_objects, top_logos, top_text,
                       has_people, coverage_seconds, summary_text,
                       hydration_score, updated_at
                  FROM upload_recognition_summary
                 WHERE upload_id = $1
                """,
                upload_id,
            )
        except Exception:
            recognition_row = None

    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")

    row_d = dict(row)
    payload = build_upload_detail_payload(row_d)
    if thumb_user:
        try:
            thumb_url, _ = await ensure_upload_thumbnail_resident(
                db_pool=pool,
                user_id=user_id,
                upload_row=row_d,
                user_row=thumb_user,
            )
            if thumb_url:
                payload["thumbnail_url"] = thumb_url
        except Exception:
            pass
    if recognition_row is not None:
        payload["recognition"] = {
            "object_track_count": int(recognition_row["object_track_count"] or 0),
            "person_segment_count": int(recognition_row["person_segment_count"] or 0),
            "text_detection_count": int(recognition_row["text_detection_count"] or 0),
            "logo_count": int(recognition_row["logo_count"] or 0),
            "top_objects": list(recognition_row["top_objects"] or []),
            "top_logos": list(recognition_row["top_logos"] or []),
            "top_text": list(recognition_row["top_text"] or []),
            "has_people": bool(recognition_row["has_people"]),
            "coverage_seconds": float(recognition_row["coverage_seconds"] or 0),
            "summary_text": recognition_row["summary_text"] or "",
            "hydration_score": float(recognition_row["hydration_score"] or 0),
        }
    return payload
