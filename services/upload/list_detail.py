"""Upload list/detail shaping, queue stats, and artifact geo/scene helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException

from core.helpers import (
    _load_uploads_columns,
    _pick_cols,
    _safe_json,
    coerce_output_artifacts_dict,
    ensure_uploads_columns_loaded,
)
from services.uploads_api import enrich_platform_results_batch
from stages.context import is_placeholder_upload_caption, is_placeholder_upload_title

from services.upload.hashtags import _normalize_hashtags_list
from services.upload.schedule_guard import UPLOAD_ERROR_MESSAGES
from services.upload.stage_labels import stage_label_for
from services.upload.status import (
    CANCELLABLE_STATUSES,
    UPLOAD_STATUS_LABEL,
    UPLOAD_VIEW_STATUS,
    is_requeueable_upload,
    is_retryable_upload,
)
from services.upload.thumbnails import (
    _normalize_platform_results_detail,
    card_thumbnail_url,
    enrich_posted_thumbnail_urls,
    merged_platform_thumbnail_urls,
    pikzels_template_thumbnail_warning,
    thumbnail_render_method_from_artifacts,
    studio_thumb_diagnostics_from_artifacts,
    thumbnail_storage_missing_flag,
)

logger = logging.getLogger(__name__)


def _upload_error_message(d: dict) -> Optional[str]:
    code = (d.get("error_code") or "").strip()
    detail = (d.get("error_detail") or "").strip()
    if code and code in UPLOAD_ERROR_MESSAGES:
        return UPLOAD_ERROR_MESSAGES[code]
    return detail or code or None


def _dt_iso(v: Any) -> Optional[str]:
    """Serialize DB timestamps for API JSON (datetime, date, or legacy strings)."""
    if v is None or v == "":
        return None
    if hasattr(v, "isoformat") and callable(getattr(v, "isoformat", None)):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


def _json_safe_for_api(v: Any) -> Any:
    """Normalize values for Starlette JSONResponse (json.dumps without default=)."""
    if v is None:
        return None
    if isinstance(v, dict):
        return {str(k): _json_safe_for_api(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe_for_api(x) for x in v]
    if hasattr(v, "isoformat") and callable(getattr(v, "isoformat", None)):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return v


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


def failure_phase_from_artifacts(raw: Any) -> Optional[str]:
    """Pipeline phase where a terminal failure occurred (``output_artifacts.failure_phase``)."""
    artifacts = _safe_json(raw, {})
    if not isinstance(artifacts, dict):
        return None
    v = artifacts.get("failure_phase")
    return str(v).strip() if v else None


_ARTIFACT_UI_KEYS = (
    "publish_quality_notice",
    "content_hotness",
    "coach_hints",
    "failure_phase",
    "failure_diag",
)


def output_artifacts_dict(raw: Any) -> Dict[str, Any]:
    """Parse uploads.output_artifacts jsonb for API consumers."""
    return coerce_output_artifacts_dict(raw)


def _normalize_coach_hints_artifact(val: Any) -> Any:
    """coach_hints may be a JSON string (or double-encoded) inside jsonb."""
    if val is None:
        return None
    if isinstance(val, list):
        return _json_safe_for_api(val)
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
        except Exception:
            return val
        if isinstance(parsed, list):
            return _json_safe_for_api(parsed)
        if isinstance(parsed, str) and parsed.strip():
            try:
                inner = json.loads(parsed)
                if isinstance(inner, list):
                    return _json_safe_for_api(inner)
            except Exception:
                pass
        return _json_safe_for_api(parsed)
    return _json_safe_for_api(val)


def slim_output_artifacts_for_ui(raw: Any) -> Dict[str, Any]:
    """Queue/detail UI fields only — avoids shipping multi‑MB hydration blobs on list."""
    arts = output_artifacts_dict(raw)
    if not arts:
        return {}
    out: Dict[str, Any] = {}
    for key in _ARTIFACT_UI_KEYS:
        if key in arts and arts[key] is not None:
            if key == "coach_hints":
                out[key] = _normalize_coach_hints_artifact(arts[key])
            elif key == "content_hotness" and isinstance(arts[key], str):
                try:
                    parsed = json.loads(arts[key])
                    out[key] = _json_safe_for_api(parsed) if isinstance(parsed, dict) else arts[key]
                except Exception:
                    out[key] = _json_safe_for_api(arts[key])
            else:
                out[key] = _json_safe_for_api(arts[key])
    return out


def failure_diag_from_upload_row(d: dict) -> Optional[Dict[str, Any]]:
    """Support bundle for queue diagnostics (explicit artifact or derived from manifest)."""
    arts = output_artifacts_dict(d.get("output_artifacts"))
    fd = arts.get("failure_diag")
    if isinstance(fd, dict):
        return _json_safe_for_api(fd)
    if isinstance(fd, str) and fd.strip():
        try:
            parsed = json.loads(fd)
            if isinstance(parsed, dict):
                return _json_safe_for_api(parsed)
        except Exception:
            pass

    pm = _safe_json(d.get("pipeline_manifest"), {})
    if isinstance(pm, dict):
        steps = pm.get("steps") or []
        failed = [
            s
            for s in steps
            if isinstance(s, dict) and str(s.get("status") or "").lower() in ("failed", "error", "timeout")
        ]
        if failed or pm.get("terminal"):
            return _json_safe_for_api(
                {
                    "terminal": pm.get("terminal"),
                    "pipeline_failed_steps": failed[-8:],
                }
            )

    fp = failure_phase_from_artifacts(d.get("output_artifacts"))
    if fp or d.get("error_code") or d.get("error_detail"):
        return {
            "failure_phase": fp,
            "error_code": d.get("error_code"),
            "error": _upload_error_message(d),
        }
    return None


def _pipeline_manifest_dict(raw: Any) -> Optional[Dict[str, Any]]:
    pm = _safe_json(raw, None)
    if isinstance(pm, dict) and pm:
        return _json_safe_for_api(pm)
    return None


def scene_story_from_artifacts(raw: Any) -> str:
    """Pull the human-readable scene-story paragraph from output_artifacts."""
    artifacts = _safe_json(raw, {})
    if not isinstance(artifacts, dict):
        return ""
    v = artifacts.get("scene_story")
    if isinstance(v, str):
        return v.strip()
    return ""


def timeline_story_from_artifacts(raw: Any) -> list:
    """Pull the ordered [{t_seconds, kind, text}] timeline from output_artifacts."""
    artifacts = _safe_json(raw, {})
    if not isinstance(artifacts, dict):
        return []
    v = artifacts.get("timeline_story")
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def geo_location_hint_for_upload(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    When driving/telemetry content has no resolved place, suggest .map or dashcam HUD.
    """
    tel = None
    tm = _safe_json(row.get("trill_metadata"), {}) or {}
    if isinstance(tm, dict) and isinstance(tm.get("telemetry"), dict):
        tel = tm["telemetry"]
    has_gps = False
    has_place = False
    if isinstance(tel, dict):
        for k in (
            "location_display",
            "location_city",
            "location_road",
            "gazetteer_place_name",
            "padus_unit_name",
        ):
            if str(tel.get(k) or "").strip():
                has_place = True
                break
        if tel.get("near_padus") in (True, "true", "1"):
            has_place = True
        for k in ("mid_lat", "mid_lon", "start_lat", "start_lon"):
            try:
                v = float(tel.get(k))
                if abs(v) > 1e-6:
                    has_gps = True
                    break
            except (TypeError, ValueError):
                pass
    if has_place:
        return None
    platforms = list(row.get("platforms") or [])
    filename = str(row.get("filename") or "").lower()
    drivingish = bool(row.get("telemetry_r2_key")) or filename.endswith(".map")
    if not drivingish and not row.get("trill_score"):
        return None
    if has_gps and not has_place:
        msg = (
            "GPS was detected but place names did not resolve. Confirm gazetteer/PAD-US "
            "tables are loaded on the server, or retry after processing finishes."
        )
    else:
        msg = (
            "No route GPS for this upload. Add a companion .map file (same basename as the video) "
            "or use dashcam footage with a burned-in GPS HUD so captions and hashtags can name roads and places."
        )
    return {
        "code": "geo_signals_missing",
        "message": msg,
        "settings_path": "settings.html",
    }


def compute_smart_schedule_display(schedule_mode: str, schedule_metadata: Any) -> Optional[dict]:
    if schedule_mode != "smart" or not schedule_metadata:
        return None
    try:
        sm = schedule_metadata if isinstance(schedule_metadata, dict) else json.loads(schedule_metadata)
        if not isinstance(sm, dict):
            return None
        out = _json_safe_for_api(sm)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def _build_slim_upload_item(d: dict) -> dict:
    """Minimal projection for analytics top-content (no enrichment / presign)."""
    ai_title = (d.get("ai_generated_title") or d.get("ai_title") or "") or ""
    raw_title = (d.get("title") or "").strip()
    title = (
        ai_title
        if ai_title and is_placeholder_upload_title(raw_title, d.get("filename") or "")
        else (raw_title or ai_title)
    )
    return {
        "id": str(d.get("id")),
        "title": title,
        "filename": d.get("filename"),
        "platforms": list(d.get("platforms") or []),
        "status": d.get("status") or "",
        "created_at": _dt_iso(d.get("created_at")),
        "completed_at": _dt_iso(d.get("completed_at")),
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),
        "comments": int(d.get("comments") or 0),
        "shares": int(d.get("shares") or 0),
    }


def build_upload_list_item(
    d: dict,
    platform_results: Any,
    *,
    creator_map: Dict[str, str],
    presign_r2_thumbnails: bool,
) -> dict:
    ai_title = (d.get("ai_generated_title") or d.get("ai_title") or "") or ""
    ai_caption = (d.get("ai_generated_caption") or d.get("ai_caption") or "") or ""
    ai_hashtags = _normalize_hashtags_list(d.get("ai_generated_hashtags"))

    raw_title = (d.get("title") or "").strip()
    title = (
        ai_title
        if ai_title and is_placeholder_upload_title(raw_title, d.get("filename") or "")
        else (raw_title or ai_title)
    )
    raw_caption = (d.get("caption") or "").strip()
    caption = (
        ai_caption
        if ai_caption and is_placeholder_upload_caption(raw_caption)
        else (raw_caption or ai_caption)
    )

    hashtags = _normalize_hashtags_list(d.get("hashtags"))

    thumb_key = d.get("thumbnail_r2_key")
    sk = str(thumb_key).strip() if thumb_key else ""

    plat_thumb_urls = merged_platform_thumbnail_urls(d.get("output_artifacts"), platform_results)
    posted_urls = enrich_posted_thumbnail_urls(platform_results)
    upload_id_str = str(d.get("id") or "")
    thumbnail_url = card_thumbnail_url(
        upload_id_str,
        thumbnail_r2_key=thumb_key,
        output_artifacts=d.get("output_artifacts"),
        platform_results=platform_results,
        upload_platforms=list(d.get("platforms") or []),
        presign_r2_thumbnails=presign_r2_thumbnails,
    )
    storage_missing = thumbnail_storage_missing_flag(
        primary_sk=sk,
        upload_id=upload_id_str,
        thumbnail_url=thumbnail_url,
        output_artifacts=d.get("output_artifacts"),
        platform_results=platform_results,
        upload_platforms=list(d.get("platforms") or []),
    )

    raw_status = str(d.get("status") or "")
    trill_raw = d.get("trill_score")
    trill_score = None
    if trill_raw is not None:
        try:
            trill_score = float(trill_raw)
        except (TypeError, ValueError):
            trill_score = None

    return {
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
        "scheduled_time": _dt_iso(d.get("scheduled_time")),
        "created_at": _dt_iso(d.get("created_at")),
        "completed_at": _dt_iso(d.get("completed_at")),
        "put_cost": int(d.get("put_reserved") or 0),
        "aic_cost": int(d.get("aic_reserved") or 0),
        "error_code": d.get("error_code"),
        "error": _upload_error_message(d),
        "thumbnail_url": thumbnail_url,
        "thumbnail_storage_missing": storage_missing,
        "posted_platform_thumbnail_urls": posted_urls,
        "platform_thumbnail_urls": plat_thumb_urls,
        "scene_story": scene_story_from_artifacts(d.get("output_artifacts")),
        "timeline_story": timeline_story_from_artifacts(d.get("output_artifacts")),
        "failure_phase": failure_phase_from_artifacts(d.get("output_artifacts")),
        "platform_results": platform_results,
        "file_size": d.get("file_size"),
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),
        "comments": int(d.get("comments") or 0),
        "shares": int(d.get("shares") or 0),
        "progress": int(d.get("processing_progress") or 0),
        "current_stage": d.get("processing_stage"),
        "stage_label": stage_label_for(d.get("processing_stage")),
        "processing_started_at": _dt_iso(d.get("processing_started_at")),
        "processingStartedAt": _dt_iso(d.get("processing_started_at")),
        "updated_at": _dt_iso(d.get("updated_at")),
        "updatedAt": _dt_iso(d.get("updated_at")),
        "schedule_mode": d.get("schedule_mode") or "immediate",
        "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
        "smart_schedule": _safe_json(d.get("schedule_metadata"), None),
        "is_editable": raw_status
        in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
        "is_cancellable": raw_status in CANCELLABLE_STATUSES or raw_status == "processing",
        "is_requeueable": is_requeueable_upload(raw_status, d.get("error_code")),
        "is_retryable": is_retryable_upload(raw_status, error_code=d.get("error_code")),
        "video_url": d.get("video_url"),
        "trill_score": trill_score,
        "speed_bucket": d.get("speed_bucket"),
        "created_by_user_id": str(d.get("created_by_user_id")) if d.get("created_by_user_id") else None,
        "created_by_email": creator_map.get(str(d.get("created_by_user_id")))
        if d.get("created_by_user_id")
        else None,
        "trill_metadata": _safe_json(d.get("trill_metadata"), None),
        "youtubeCopyrightShorts": youtube_copyright_shorts_notice_from_artifacts(
            d.get("output_artifacts")
        ),
        "thumbnail_render_method": thumbnail_render_method_from_artifacts(
            d.get("output_artifacts")
        ),
        "pikzels_thumbnail_warning": pikzels_template_thumbnail_warning(d.get("output_artifacts")),
        "studio_thumb_diagnostics": studio_thumb_diagnostics_from_artifacts(
            d.get("output_artifacts")
        ),
        "geo_location_hint": geo_location_hint_for_upload(d),
        "output_artifacts": slim_output_artifacts_for_ui(d.get("output_artifacts")),
        "pipeline_manifest": _pipeline_manifest_dict(d.get("pipeline_manifest")),
        "failure_diag": failure_diag_from_upload_row(d),
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
    workspace_id: Optional[str] = None,
    presign_r2_thumbnails: bool = False,
    presign_platform_avatars: bool = False,
    sort: Optional[str] = None,
    order: str = "desc",
    since: Any = None,
    slim: bool = False,
) -> Union[list, dict]:
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    cols = await _load_uploads_columns(pool)

    if slim:
        wanted = [
            "id",
            "filename",
            "title",
            "ai_title",
            "ai_generated_title",
            "platforms",
            "status",
            "created_at",
            "completed_at",
            "views",
            "likes",
            "comments",
            "shares",
        ]
    else:
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
            "pipeline_manifest",
            "created_by_user_id",
            "workspace_id",
        ]
    select_cols = _pick_cols(wanted, cols) or ["id", "filename", "platforms", "status", "created_at"]
    if workspace_id:
        select_sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE workspace_id = $1::uuid"
        count_sql = "SELECT COUNT(*) FROM uploads WHERE workspace_id = $1::uuid"
        params: List[Any] = [workspace_id]
        count_params: List[Any] = [workspace_id]
    else:
        select_sql = f"SELECT {', '.join(select_cols)} FROM uploads WHERE user_id = $1"
        count_sql = "SELECT COUNT(*) FROM uploads WHERE user_id = $1"
        params = [user_id]
        count_params = [user_id]

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

    if since is not None and "created_at" in cols:
        params.append(since)
        count_params.append(since)
        select_sql += f" AND created_at >= ${len(params)}"
        count_sql += f" AND created_at >= ${len(count_params)}"

    params.extend([limit, offset])
    limit_ph = len(params) - 1
    offset_ph = len(params)
    if sort in ("views", "engagement") and "views" in cols:
        order_dir = "ASC" if str(order).lower() == "asc" else "DESC"
        if sort == "views":
            order_col = "views"
        else:
            order_col = (
                "((COALESCE(likes,0)+COALESCE(comments,0)+COALESCE(shares,0))::float "
                "/ NULLIF(views,0))"
            )
        select_sql += (
            f" ORDER BY {order_col} {order_dir} NULLS LAST, created_at DESC "
            f"LIMIT ${limit_ph} OFFSET ${offset_ph}"
        )
    else:
        select_sql += f" ORDER BY created_at DESC LIMIT ${limit_ph} OFFSET ${offset_ph}"

    out: List[dict] = []
    async with pool.acquire() as conn:
        rows = await conn.fetch(select_sql, *params)
        total = await conn.fetchval(count_sql, *count_params) if meta else None

        row_dicts = [dict(r) for r in rows]

        if slim:
            slim_out = [_build_slim_upload_item(d) for d in row_dicts]
            if not meta:
                return slim_out
            return {"uploads": slim_out, "total": int(total or 0), "limit": limit, "offset": offset}

        enriched_pr = await enrich_platform_results_batch(
            conn, row_dicts, str(user_id), presign_avatars=presign_platform_avatars
        )
        creator_ids = list({
            str(d["created_by_user_id"])
            for d in row_dicts
            if d.get("created_by_user_id")
        })
        creator_map: Dict[str, str] = {}
        if creator_ids:
            crows = await conn.fetch(
                "SELECT id::text AS id, email FROM users WHERE id = ANY($1::uuid[])",
                creator_ids,
            )
            creator_map = {str(r["id"]): r["email"] for r in crows}

    for d, platform_results in zip(row_dicts, enriched_pr):
        try:
            item = build_upload_list_item(
                d,
                platform_results,
                creator_map=creator_map,
                presign_r2_thumbnails=presign_r2_thumbnails,
            )
        except Exception as exc:
            logger.warning(
                "upload list row skipped upload_id=%s user_id=%s: %s",
                d.get("id"),
                user_id,
                exc,
                exc_info=True,
            )
            continue
        out.append(item)

    if not meta:
        return out

    return {"uploads": out, "total": int(total or 0), "limit": limit, "offset": offset}


async def fetch_upload_queue_stats(pool, user_id: str) -> dict:
    """Counts for the 4 queue/dashboard buckets."""
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
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
        "completed": int(row["completed"] or 0),
        "partial": int(row["partial"] or 0),
        "failed": int(row["failed"] or 0),
        "pending": processing,
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

    thumb_key = d.get("thumbnail_r2_key")
    sk = str(thumb_key).strip() if thumb_key else ""

    plat_thumb_urls = merged_platform_thumbnail_urls(d.get("output_artifacts"), platform_results)
    posted_urls = enrich_posted_thumbnail_urls(platform_results)
    upload_id_str = str(d.get("id") or "")
    thumbnail_url = card_thumbnail_url(
        upload_id_str,
        thumbnail_r2_key=thumb_key,
        output_artifacts=d.get("output_artifacts"),
        platform_results=platform_results,
        upload_platforms=list(d.get("platforms") or []),
        presign_r2_thumbnails=True,
    )
    storage_missing = thumbnail_storage_missing_flag(
        primary_sk=sk,
        upload_id=upload_id_str,
        thumbnail_url=thumbnail_url,
        output_artifacts=d.get("output_artifacts"),
        platform_results=platform_results,
        upload_platforms=list(d.get("platforms") or []),
    )

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
        "scheduled_time": _dt_iso(d.get("scheduled_time")),
        "schedule_mode": d.get("schedule_mode") or "immediate",
        "schedule_metadata": _safe_json(d.get("schedule_metadata"), None),
        "smart_schedule": _safe_json(d.get("schedule_metadata"), None),
        "timezone": d.get("timezone") or "UTC",
        "is_editable": d.get("status")
        in ("pending", "staged", "queued", "scheduled", "ready_to_publish"),
        "is_cancellable": (d.get("status") or "") in CANCELLABLE_STATUSES
        or (d.get("status") or "") == "processing",
        "is_requeueable": is_requeueable_upload(
            str(d.get("status") or ""), d.get("error_code")
        ),
        "is_retryable": is_retryable_upload(
            str(d.get("status") or ""), error_code=d.get("error_code")
        ),
        "created_at": _dt_iso(d.get("created_at")),
        "completed_at": _dt_iso(d.get("completed_at")),
        "put_cost": int(d.get("put_reserved") or 0),
        "aic_cost": int(d.get("aic_reserved") or 0),
        "error_code": d.get("error_code"),
        "error": _upload_error_message(d),
        "thumbnail_url": thumbnail_url,
        "thumbnail_storage_missing": storage_missing,
        "posted_platform_thumbnail_urls": posted_urls,
        "platform_thumbnail_urls": plat_thumb_urls,
        "scene_story": scene_story_from_artifacts(d.get("output_artifacts")),
        "timeline_story": timeline_story_from_artifacts(d.get("output_artifacts")),
        "failure_phase": failure_phase_from_artifacts(d.get("output_artifacts")),
        "platform_results": platform_results,
        "file_size": d.get("file_size"),
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),
        "progress": int(d.get("processing_progress") or 0),
        "current_stage": d.get("processing_stage"),
        "stage_label": stage_label_for(d.get("processing_stage")),
        "duration_seconds": duration_seconds,
        "processingStartedAt": _dt_iso(d.get("processing_started_at")),
        "processingFinishedAt": _dt_iso(d.get("processing_finished_at")),
        "processingProgress": int(d.get("processing_progress") or 0),
        "processingStage": d.get("processing_stage"),
        "trill_score": float(d["trill_score"]) if d.get("trill_score") is not None else None,
        "speed_bucket": d.get("speed_bucket"),
        "trill_metadata": _safe_json(d.get("trill_metadata"), None),
        "youtubeCopyrightShorts": youtube_copyright_shorts_notice_from_artifacts(
            d.get("output_artifacts")
        ),
        "thumbnail_render_method": thumbnail_render_method_from_artifacts(d.get("output_artifacts")),
        "pikzels_thumbnail_warning": pikzels_template_thumbnail_warning(d.get("output_artifacts")),
        "studio_thumb_diagnostics": studio_thumb_diagnostics_from_artifacts(
            d.get("output_artifacts")
        ),
        "geo_location_hint": geo_location_hint_for_upload(d),
        "output_artifacts": slim_output_artifacts_for_ui(d.get("output_artifacts")),
        "pipeline_manifest": _pipeline_manifest_dict(d.get("pipeline_manifest")),
        "failure_diag": failure_diag_from_upload_row(d),
    }


async def fetch_upload_detail(pool, upload_id: str, user_id: str) -> dict:
    """
    Load upload detail for ``user_id`` after the same auth gates as ``get_current_user_readonly``,
    using a single pooled connection (upload row + optional recognition + cold schema probe).
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
        "pipeline_manifest",
    ]
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, status, email_verified FROM users WHERE id = $1",
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
    try:
        from services.upload_funnel import get_upload_funnel_events_async

        payload["funnel_events"] = await get_upload_funnel_events_async(str(upload_id))
    except Exception:
        payload["funnel_events"] = []
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


async def poll_upload_thumbnails_payload(
    pool: Any,
    user_id: str,
    upload_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    DB-only thumbnail snapshot for client polling (no platform API calls).

    Returns upload_id -> {thumbnail_url, posted_platform_thumbnail_urls, platform_thumbnail_urls}.
    """
    if not upload_ids:
        return {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, thumbnail_r2_key, platforms, output_artifacts, platform_results
            FROM uploads
            WHERE user_id = $1 AND id = ANY($2::uuid[])
            """,
            user_id,
            upload_ids,
        )
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        uid = str(d.get("id") or "")
        if not uid:
            continue
        platform_results = _normalize_platform_results_detail(d.get("platform_results"))
        upload_id_str = str(d.get("id") or "")
        thumb = card_thumbnail_url(
            upload_id_str,
            thumbnail_r2_key=d.get("thumbnail_r2_key"),
            output_artifacts=d.get("output_artifacts"),
            platform_results=platform_results,
            upload_platforms=list(d.get("platforms") or []),
            presign_r2_thumbnails=False,
        )
        if not thumb:
            continue
        sk = str(d.get("thumbnail_r2_key") or "").strip()
        plat_thumb_urls = merged_platform_thumbnail_urls(d.get("output_artifacts"), platform_results)
        posted_urls = enrich_posted_thumbnail_urls(platform_results)
        storage_missing = thumbnail_storage_missing_flag(
            primary_sk=sk,
            upload_id=upload_id_str,
            thumbnail_url=thumb,
            output_artifacts=d.get("output_artifacts"),
            platform_results=platform_results,
            upload_platforms=list(d.get("platforms") or []),
        )
        out[uid] = {
            "thumbnail_url": thumb,
            "thumbnail_storage_missing": storage_missing,
            "posted_platform_thumbnail_urls": posted_urls,
            "platform_thumbnail_urls": plat_thumb_urls,
        }
    return out
