"""
Upload route business logic: presign insert, complete transaction, list/detail shaping, queue stats.

Routers keep HTTP wiring (cookies, Request, presigned URL generation, enqueue, audit).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException

from core.config import R2_BUCKET_NAME
from core.media_mirror import is_hotlink_blocked_image_url
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
from services.billing_service_weights import fetch_service_weights_map
from services.account_groups import resolve_group_ids_to_target_accounts
from services.smart_schedule_insights import calculate_smart_schedule_data_driven
from services.workspace import billing_user_id, can_upload_in_workspace, get_workspace_for_user
from core.sql_allowlist import UPLOADS_COMPLETE_BODY_COLUMNS, assert_set_fragments_columns
from core.wallet import atomic_reserve_tokens, get_wallet
from routers.preferences import get_user_prefs_for_upload
from services.platform_posted_thumbnails import (
    PLATFORM_THUMB_PRIORITY,
    pick_primary_thumbnail_url,
    posted_platform_thumbnail_urls_from_results,
)
from services.uploads_api import enrich_platform_results_batch
from stages.ai_service_costs import compute_presign_put_aic_costs
from stages.context import is_placeholder_upload_caption, is_placeholder_upload_title
from stages.entitlements import entitlements_to_dict, get_entitlements_from_user

logger = logging.getLogger(__name__)


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
    elif use_eng is None:
        # Presign default: when the server has Pikzels configured, opt uploads into
        # studio unless the client explicitly disabled engine/pikzels on the body.
        try:
            from stages.pikzels_api import studio_renderer_enabled

            if studio_renderer_enabled():
                user_prefs["thumbnail_pikzels_enabled"] = True
                user_prefs["thumbnailPikzelsEnabled"] = True
                user_prefs.setdefault("thumbnail_studio_engine_enabled", True)
                user_prefs.setdefault("thumbnailStudioEngineEnabled", True)
        except Exception:
            pass

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
        user_prefs["thumbnail_persona_enabled"] = True
        user_prefs["thumbnailPersonaEnabled"] = True

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


def thumbnail_render_method_from_artifacts(raw: Any) -> str:
    """Worker-persisted thumbnail pipeline method (studio_renderer, template, none, …)."""
    artifacts = _safe_json(raw, {})
    if not isinstance(artifacts, dict):
        return ""
    return str(artifacts.get("thumbnail_render_method") or "").strip().lower()


def pikzels_template_thumbnail_warning(raw_artifacts: Any) -> Optional[Dict[str, str]]:
    """
    When the server has PIKZELS_API_KEY but this upload used PIL template render,
    return a short warning for queue/upload UI.
    """
    method = thumbnail_render_method_from_artifacts(raw_artifacts)
    if method not in ("template", "none", ""):
        return None
    try:
        from services.pikzels_v2 import resolve_public_api_key

        if not (resolve_public_api_key() or "").strip():
            return None
    except Exception:
        return None
    artifacts = _safe_json(raw_artifacts, {}) or {}
    skip_reason = ""
    raw_report = artifacts.get("studio_render_report")
    if isinstance(raw_report, str) and raw_report.strip():
        try:
            rep = json.loads(raw_report)
            if isinstance(rep, dict):
                skip_reason = str(rep.get("skip_reason") or "").strip()
        except Exception:
            pass
    elif isinstance(raw_report, dict):
        skip_reason = str(raw_report.get("skip_reason") or "").strip()
    return {
        "code": "pikzels_template_fallback",
        "message": (
            "This upload did not use Pikzels Studio (template or raw frame only). "
            "Turn on auto-thumbnails and Thumbnail Studio in Settings, and set render pipeline to Auto."
        ),
        "settings_path": "settings.html#thumbnail-studio",
        "skip_reason": skip_reason,
    }


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


def presign_upload_thumbnail_r2_key(key: str, *, expires_in: int = 3600) -> Optional[str]:
    k = str(key or "").strip()
    if not k:
        return None
    try:
        s3 = get_s3_client()
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET_NAME, "Key": _normalize_r2_key(k)},
            ExpiresIn=expires_in,
        )
    except Exception:
        return None


def browser_safe_thumbnail_url(url: Optional[str]) -> Optional[str]:
    u = str(url or "").strip()
    if not u.startswith("http"):
        return None
    if is_hotlink_blocked_image_url(u):
        return None
    return u


def upload_card_thumbnail_href(upload_id: str) -> str:
    """Stable first-party card image URL (no presign expiry)."""
    return f"/api/uploads/{upload_id}/thumbnail"


def artifact_platform_thumbnail_r2_keys(output_artifacts: Any) -> Dict[str, str]:
    artifacts = _safe_json(output_artifacts, {})
    if not isinstance(artifacts, dict):
        return {}
    raw_keys = artifacts.get("platform_thumbnail_r2_keys") or {}
    if isinstance(raw_keys, str):
        raw_keys = _safe_json(raw_keys, {})
    if not isinstance(raw_keys, dict):
        return {}
    out: Dict[str, str] = {}
    for platform, key in raw_keys.items():
        plat = str(platform or "").strip().lower()
        k = str(key or "").strip()
        if plat and k:
            out[plat] = k
    return out


def mirrored_platform_thumbnail_r2_keys(platform_results: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pr in _normalize_platform_results_detail(platform_results):
        if pr.get("success") is False:
            continue
        plat = str(pr.get("platform") or "").lower().strip()
        k = str(pr.get("platform_thumbnail_r2_key") or "").strip()
        if plat and k:
            out[plat] = k
    return out


def resolve_upload_thumbnail_r2_key(
    *,
    thumbnail_r2_key: Optional[str],
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
) -> Optional[str]:
    """Best R2 object key for card/detail thumbnail streaming."""
    artifacts = artifact_platform_thumbnail_r2_keys(output_artifacts)
    platforms = [str(p).lower() for p in (upload_platforms or []) if p]
    order = platforms + [p for p in PLATFORM_THUMB_PRIORITY if p not in platforms]
    for plat in order:
        k = artifacts.get(plat)
        if k:
            return k
    for k in artifacts.values():
        if k:
            return k
    sk = str(thumbnail_r2_key or "").strip()
    if sk:
        return sk
    mirrored = mirrored_platform_thumbnail_r2_keys(platform_results)
    for plat in order:
        k = mirrored.get(plat)
        if k:
            return k
    for k in mirrored.values():
        if k:
            return k
    return None


def card_thumbnail_url(
    upload_id: str,
    *,
    thumbnail_r2_key: Optional[str],
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
    presign_r2_thumbnails: bool = False,
) -> Optional[str]:
    """
    URL for queue/dashboard <img src>.

    Prefer first-party proxy when any R2 key exists; otherwise browser-safe CDN URL.
    """
    uid = str(upload_id or "").strip()
    if resolve_upload_thumbnail_r2_key(
        thumbnail_r2_key=thumbnail_r2_key,
        output_artifacts=output_artifacts,
        platform_results=platform_results,
        upload_platforms=upload_platforms,
    ):
        return upload_card_thumbnail_href(uid) if uid else None

    platform_results_norm = _normalize_platform_results_detail(platform_results)
    sk = str(thumbnail_r2_key or "").strip()
    r2_thumb_url = None
    if sk and presign_r2_thumbnails:
        r2_thumb_url = presign_upload_thumbnail_r2_key(sk, expires_in=3600)

    plat_thumb_urls = merged_platform_thumbnail_urls(output_artifacts, platform_results_norm)
    posted_urls = posted_platform_thumbnail_urls_from_results(platform_results_norm)
    return browser_safe_thumbnail_url(
        pick_primary_thumbnail_url(
            posted=posted_urls,
            artifact_platform_urls=plat_thumb_urls,
            r2_presigned=r2_thumb_url,
            upload_platforms=list(upload_platforms or []),
        )
    )


THUMBNAIL_REPAIR_STATUSES = frozenset({"completed", "succeeded", "partial"})


def thumbnail_storage_missing_flag(
    *,
    primary_sk: str,
    upload_id: str,
    thumbnail_url: Optional[str],
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
) -> bool:
    """
    True when the primary ``thumbnail_r2_key`` may be absent from R2 and should be
    verified/regenerated by ``repair_upload_thumbnails_batch``.

    Proxy card URLs (``/api/uploads/{id}/thumbnail``) backed by the primary key — not
    artifacts or mirrored platform thumbs — are flagged so bootstrap repair runs even
    when ``thumbnail_url`` is present.
    """
    sk = str(primary_sk or "").strip()
    if not sk:
        return False
    if not thumbnail_url:
        return True
    uid = str(upload_id or "").strip()
    if not uid:
        return False
    resolved = resolve_upload_thumbnail_r2_key(
        thumbnail_r2_key=sk,
        output_artifacts=output_artifacts,
        platform_results=platform_results,
        upload_platforms=upload_platforms,
    )
    if resolved != sk:
        return False
    return thumbnail_url == upload_card_thumbnail_href(uid)


def collect_thumbnail_repair_ids(items: List[dict], *, limit: int = 5) -> List[str]:
    """Upload ids needing primary thumbnail repair (bounded; scans full list)."""
    ids: List[str] = []
    for item in items:
        if not item.get("thumbnail_storage_missing"):
            continue
        st = str(item.get("status") or "").lower()
        if st not in THUMBNAIL_REPAIR_STATUSES:
            continue
        uid = str(item.get("id") or "").strip()
        if uid and uid not in ids:
            ids.append(uid)
        if len(ids) >= limit:
            break
    return ids


async def repair_upload_thumbnails_batch(
    pool: Any,
    user_id: str,
    upload_ids: List[str],
) -> None:
    """Background: verify R2 object for primary thumb; regenerate when missing (max 5)."""
    from services.thumbnail_regenerate import ensure_upload_thumbnail_resident

    uids = [str(x).strip() for x in (upload_ids or []) if str(x).strip()][:5]
    if not uids:
        return
    try:
        async with pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
                user_id,
            )
    except Exception as e:
        logger.warning("repair_upload_thumbnails_batch user load failed: %s", e)
        return
    if not user_row:
        return
    user_dict = {
        "subscription_tier": user_row["subscription_tier"],
        "role": user_row["role"],
        "flex_enabled": user_row["flex_enabled"],
    }
    repaired = 0
    for upload_id in uids:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, status, thumbnail_r2_key, r2_key, processed_r2_key,
                           platforms, title, caption, user_preferences, output_artifacts,
                           platform_results
                    FROM uploads
                    WHERE id = $1 AND user_id = $2
                    """,
                    upload_id,
                    user_id,
                )
            if not row:
                continue
            d = dict(row)
            sk = str(d.get("thumbnail_r2_key") or "").strip()
            if not sk:
                continue
            st = str(d.get("status") or "").lower()
            if st not in THUMBNAIL_REPAIR_STATUSES:
                continue
            had_object = await asyncio.to_thread(r2_object_exists, _normalize_r2_key(sk))
            url, _rk = await ensure_upload_thumbnail_resident(
                db_pool=pool,
                user_id=user_id,
                upload_row=d,
                user_row=user_dict,
            )
            if not had_object and url:
                repaired += 1
                logger.info(
                    "repair_upload_thumbnails_batch regenerated upload=%s key=%s",
                    upload_id,
                    sk[:80],
                )
        except Exception as e:
            logger.warning(
                "repair_upload_thumbnails_batch upload=%s: %s", upload_id, e
            )
        await asyncio.sleep(0.08)
    if repaired:
        logger.info(
            "repair_upload_thumbnails_batch user=%s repaired=%s queued=%s",
            user_id,
            repaired,
            len(uids),
        )


async def stream_upload_thumbnail_bytes(
    pool: Any,
    user_id: str,
    upload_id: str,
) -> tuple[bytes, str, str]:
    """
    Load thumbnail bytes for GET /api/uploads/{id}/thumbnail.

    Returns (body, content_type, etag_key). Raises HTTPException 404 when missing.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, thumbnail_r2_key, output_artifacts, platform_results, platforms
            FROM uploads
            WHERE id = $1 AND user_id = $2
            """,
            upload_id,
            user_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    d = dict(row)
    r2_key = resolve_upload_thumbnail_r2_key(
        thumbnail_r2_key=d.get("thumbnail_r2_key"),
        output_artifacts=d.get("output_artifacts"),
        platform_results=d.get("platform_results"),
        upload_platforms=list(d.get("platforms") or []),
    )
    if not r2_key:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    norm_key = _normalize_r2_key(r2_key)

    def _read() -> Optional[tuple[bytes, str]]:
        if not r2_object_exists(norm_key):
            return None
        obj = get_s3_client().get_object(Bucket=R2_BUCKET_NAME, Key=norm_key)
        body_obj = obj.get("Body")
        raw = body_obj.read() if body_obj else b""
        if not raw:
            return None
        ct = str(obj.get("ContentType") or "image/jpeg").strip().lower()
        if not ct.startswith("image/"):
            ct = "image/jpeg"
        return raw, ct

    read_out = await asyncio.to_thread(_read)
    if not read_out:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    raw, ct = read_out
    return raw, ct, norm_key


def merged_platform_thumbnail_urls(
    output_artifacts: Any,
    platform_results: Any,
    *,
    expires_in: int = 3600,
) -> dict:
    """UploadM8-generated R2 previews; live posted covers fill gaps only."""
    artifact_urls = platform_thumbnail_urls_from_artifacts(output_artifacts, expires_in=expires_in)
    posted_urls = posted_platform_thumbnail_urls_from_results(
        platform_results if isinstance(platform_results, list) else []
    )
    merged = dict(artifact_urls)
    for plat, url in posted_urls.items():
        existing = str(merged.get(plat) or "").strip()
        if not existing.startswith("http"):
            merged[plat] = url
    return merged


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


async def presign_create_upload(conn, data: UploadInit, user: dict) -> dict:
    """
    Insert upload row and reserve tokens. Mutates ``data`` (hashtags, privacy, defaults).

    Returns dict with upload_id, r2_key, put_cost, aic_cost, user_prefs, smart_schedule (or None).
    """
    member_id = str(user["id"])
    ws_ctx = await get_workspace_for_user(conn, member_id)
    if ws_ctx and not can_upload_in_workspace(ws_ctx):
        raise HTTPException(403, "Viewers cannot create uploads")
    billing_user = ws_ctx.owner_row if ws_ctx else user
    bill_id = billing_user_id(ws_ctx, member_id)

    db_ent = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        bill_id,
    )
    user_for_ent = dict(billing_user)
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

    group_ids_raw = getattr(data, "group_ids", None) or []
    resolved_group_ids: List[str] = []
    if group_ids_raw:
        resolved_accounts, resolved_group_ids = await resolve_group_ids_to_target_accounts(
            conn,
            bill_id,
            group_ids_raw,
            data.platforms,
        )
        data.target_accounts = resolved_accounts

    user_prefs = await get_user_prefs_for_upload(conn, bill_id)
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

    num_publish_targets = len(data.target_accounts) if data.target_accounts else len(data.platforms)
    db_weights = await fetch_service_weights_map(conn)
    put_cost, aic_cost, billing_breakdown = compute_presign_put_aic_costs(
        ent_cost,
        num_publish_targets=num_publish_targets,
        file_size=getattr(data, "file_size", None),
        duration_hint=None,
        has_telemetry=bool(getattr(data, "has_telemetry", False)),
        use_ai_checkbox=use_ai_checkbox,
        user_prefs=user_prefs,
        num_thumbnails_override=None,
        service_weights_map=db_weights,
    )

    pending_count = await conn.fetchval(
        """SELECT COUNT(*) FROM uploads
           WHERE user_id = $1
           AND status IN ('pending','staged','queued','processing','ready_to_publish')""",
        bill_id,
    )
    if pending_count >= ent_cost.queue_depth:
        raise HTTPException(
            429,
            f"Queue limit reached ({pending_count}/{ent_cost.queue_depth} uploads pending). "
            "Wait for existing uploads to complete or upgrade your plan.",
        )

    upload_id = str(uuid.uuid4())
    r2_key = f"uploads/{bill_id}/{upload_id}/{data.filename}"
    telemetry_r2_key = telemetry_r2_key_for_upload(
        bill_id,
        upload_id,
        bool(getattr(data, "has_telemetry", False)),
    )

    ws_id = ws_ctx.workspace_id if ws_ctx else None

    smart_schedule = None
    if getattr(data, "schedule_mode", None) == "smart":
        days = getattr(data, "smart_schedule_days", 7)
        blocked = await get_existing_scheduled_days(conn, bill_id, days)
        smart_schedule = await calculate_smart_schedule_data_driven(
            conn,
            bill_id,
            data.platforms,
            num_days=days,
            blocked_day_offsets=blocked or None,
        )

    scheduled_time = getattr(data, "scheduled_time", None)
    schedule_metadata = None

    if getattr(data, "schedule_mode", None) == "smart" and smart_schedule:
        schedule_metadata = {p: _schedule_slot_iso(dt) for p, dt in smart_schedule.items()}
        scheduled_time = min(smart_schedule.values())

    vm_id = getattr(data, "vehicle_make_id", None)
    vmd_id = getattr(data, "vehicle_model_id", None)
    if vm_id is None:
        vm_id = user_prefs.get("default_vehicle_make_id")
    if vmd_id is None:
        vmd_id = user_prefs.get("default_vehicle_model_id")
    if vm_id is not None and vmd_id is not None:
        ok = await conn.fetchrow(
            "SELECT 1 FROM vehicle_models WHERE id = $1 AND make_id = $2",
            vmd_id,
            vm_id,
        )
        if not ok:
            raise HTTPException(400, "Invalid vehicle model for selected make")
    elif vmd_id is not None and vm_id is None:
        raise HTTPException(400, "vehicle_make_id required when vehicle_model_id is set")

    await conn.execute(
        """
            INSERT INTO uploads (
                id, user_id, r2_key, telemetry_r2_key, filename, file_size, platforms,
                title, caption, hashtags, privacy, status, scheduled_time,
                schedule_mode, put_reserved, aic_reserved, billing_breakdown, schedule_metadata,
                user_preferences, target_accounts, vehicle_make_id, vehicle_model_id, group_ids,
                workspace_id, created_by_user_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending', $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
        """,
        upload_id,
        bill_id,
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
        json.dumps(billing_breakdown, default=str),
        _json_for_upload_row(schedule_metadata) if schedule_metadata else None,
        _json_for_upload_row(user_prefs),
        data.target_accounts or [],
        vm_id,
        vmd_id,
        resolved_group_ids or [],
        ws_id,
        member_id,
    )

    ledger_meta = {"billing_breakdown": billing_breakdown}
    if ws_ctx:
        ledger_meta["actor_user_id"] = member_id
        ledger_meta["workspace_id"] = ws_id
    reserved = await atomic_reserve_tokens(
        conn, bill_id, put_cost, aic_cost, upload_id, ledger_meta=ledger_meta
    )
    if not reserved:
        await conn.execute("DELETE FROM uploads WHERE id = $1", upload_id)
        fresh_wallet = await get_wallet(conn, bill_id)
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
        "billing_breakdown": billing_breakdown,
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

    if any(
        k in body
        for k in ("vehicle_make_id", "vehicleMakeId", "vehicle_model_id", "vehicleModelId")
    ):
        vm_raw = body.get("vehicle_make_id", body.get("vehicleMakeId"))
        vmd_raw = body.get("vehicle_model_id", body.get("vehicleModelId"))
        try:
            vm_id = int(vm_raw) if vm_raw is not None and str(vm_raw).strip() != "" else None
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid vehicle_make_id") from None
        try:
            vmd_id = int(vmd_raw) if vmd_raw is not None and str(vmd_raw).strip() != "" else None
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid vehicle_model_id") from None
        if vm_id is not None and vmd_id is not None:
            ok = await conn.fetchrow(
                "SELECT 1 FROM vehicle_models WHERE id = $1 AND make_id = $2",
                vmd_id,
                vm_id,
            )
            if not ok:
                raise HTTPException(400, "Invalid vehicle model for selected make")
        elif vmd_id is not None and vm_id is None:
            raise HTTPException(400, "vehicle_make_id required when vehicle_model_id is set")
        updates.append(f"{_safe_col('vehicle_make_id', _COMPLETE_COLS)} = ${idx}")
        params.append(vm_id)
        idx += 1
        updates.append(f"{_safe_col('vehicle_model_id', _COMPLETE_COLS)} = ${idx}")
        params.append(vmd_id)
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
        "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
        "completed_at": d.get("completed_at").isoformat() if d.get("completed_at") else None,
        "views": int(d.get("views") or 0),
        "likes": int(d.get("likes") or 0),
        "comments": int(d.get("comments") or 0),
        "shares": int(d.get("shares") or 0),
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

    if since is not None and "created_at" in cols:
        params.append(since)
        count_params.append(since)
        select_sql += f" AND created_at >= ${len(params)}"
        count_sql += f" AND created_at >= ${len(count_params)}"

    params.extend([limit, offset])
    limit_ph = len(params) - 1
    offset_ph = len(params)
    # Default path stays byte-for-byte identical (created_at DESC). Alternate sorts
    # are opt-in via sort= and require the metric columns to exist.
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

    s3 = None
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
        urow = await conn.fetchrow(
            "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
            user_id,
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
        ai_title = (d.get("ai_generated_title") or d.get("ai_title") or "") or ""
        ai_caption = (d.get("ai_generated_caption") or d.get("ai_caption") or "") or ""
        ai_hashtags = _normalize_hashtags_list(d.get("ai_generated_hashtags"))

        raw_title = (d.get("title") or "").strip()
        title = ai_title if ai_title and is_placeholder_upload_title(raw_title, d.get("filename") or "") else (raw_title or ai_title)
        raw_caption = (d.get("caption") or "").strip()
        caption = ai_caption if ai_caption and is_placeholder_upload_caption(raw_caption) else (raw_caption or ai_caption)

        hashtags = _normalize_hashtags_list(d.get("hashtags"))

        thumb_key = d.get("thumbnail_r2_key")
        sk = str(thumb_key).strip() if thumb_key else ""

        plat_thumb_urls = merged_platform_thumbnail_urls(
            d.get("output_artifacts"), platform_results
        )
        posted_urls = posted_platform_thumbnail_urls_from_results(platform_results)
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
            "thumbnail_storage_missing": storage_missing,
            "posted_platform_thumbnail_urls": posted_urls,
            "platform_thumbnail_urls": plat_thumb_urls,
            "scene_story": scene_story_from_artifacts(d.get("output_artifacts")),
            "timeline_story": timeline_story_from_artifacts(d.get("output_artifacts")),
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
            "created_by_user_id": str(d.get("created_by_user_id")) if d.get("created_by_user_id") else None,
            "created_by_email": creator_map.get(str(d.get("created_by_user_id"))) if d.get("created_by_user_id") else None,
            "trill_metadata": _safe_json(d.get("trill_metadata"), None),
            "youtubeCopyrightShorts": youtube_copyright_shorts_notice_from_artifacts(
                d.get("output_artifacts")
            ),
            "thumbnail_render_method": thumbnail_render_method_from_artifacts(
                d.get("output_artifacts")
            ),
            "pikzels_thumbnail_warning": pikzels_template_thumbnail_warning(
                d.get("output_artifacts")
            ),
            "geo_location_hint": geo_location_hint_for_upload(d),
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

    thumb_key = d.get("thumbnail_r2_key")
    sk = str(thumb_key).strip() if thumb_key else ""

    plat_thumb_urls = merged_platform_thumbnail_urls(d.get("output_artifacts"), platform_results)
    posted_urls = posted_platform_thumbnail_urls_from_results(platform_results)
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
        "thumbnail_storage_missing": storage_missing,
        "posted_platform_thumbnail_urls": posted_urls,
        "platform_thumbnail_urls": plat_thumb_urls,
        "scene_story": scene_story_from_artifacts(d.get("output_artifacts")),
        "timeline_story": timeline_story_from_artifacts(d.get("output_artifacts")),
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
        "thumbnail_render_method": thumbnail_render_method_from_artifacts(d.get("output_artifacts")),
        "pikzels_thumbnail_warning": pikzels_template_thumbnail_warning(d.get("output_artifacts")),
        "geo_location_hint": geo_location_hint_for_upload(d),
    }


async def fetch_upload_detail(pool, upload_id: str, user_id: str) -> dict:
    """
    Load upload detail for ``user_id`` after the same auth gates as ``get_current_user_readonly``,
    using a single pooled connection (upload row + optional recognition + cold schema probe).

    Thumbnail URLs are presigned optimistically in ``build_upload_detail_payload`` (same as the
    uploads list). Missing R2 objects are handled client-side via image ``onerror`` backfill —
    do not call ``ensure_upload_thumbnail_resident`` here; it opens extra pool checkouts and
    can synchronously regenerate on GET (Sentry UPLOADM8-2H).

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
        plat_thumb_urls = merged_platform_thumbnail_urls(d.get("output_artifacts"), platform_results)
        posted_urls = posted_platform_thumbnail_urls_from_results(platform_results)
        out[uid] = {
            "thumbnail_url": thumb,
            "posted_platform_thumbnail_urls": posted_urls,
            "platform_thumbnail_urls": plat_thumb_urls,
        }
    return out
