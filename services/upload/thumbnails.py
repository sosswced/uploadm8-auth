"""Thumbnail URL resolution, R2 streaming, and repair helpers."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from core.config import R2_BUCKET_NAME
from core.helpers import _safe_json, coerce_output_artifacts_dict
from core.media_mirror import is_hotlink_blocked_image_url
from core.r2 import _normalize_r2_key, get_s3_client, r2_object_exists
from services.platform_posted_thumbnails import (
    PLATFORM_THUMB_PRIORITY,
    pick_primary_thumbnail_url,
    platform_video_id_from_result,
    posted_platform_thumbnail_urls_from_results,
)

logger = logging.getLogger(__name__)


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


def thumbnail_render_method_from_artifacts(raw: Any) -> str:
    """Worker-persisted thumbnail pipeline method (studio_renderer, template, none, …)."""
    artifacts = coerce_output_artifacts_dict(raw)
    return str(artifacts.get("thumbnail_render_method") or "").strip().lower()


def _parse_studio_render_report(raw_artifacts: Any) -> Dict[str, Any]:
    artifacts = coerce_output_artifacts_dict(raw_artifacts)
    raw_report = artifacts.get("studio_render_report")
    if isinstance(raw_report, str) and raw_report.strip():
        try:
            rep = json.loads(raw_report)
            return rep if isinstance(rep, dict) else {}
        except Exception:
            return {}
    if isinstance(raw_report, dict):
        return dict(raw_report)
    return {}


def studio_thumb_diagnostics_from_artifacts(raw_artifacts: Any) -> Dict[str, Any]:
    """Queue/detail payload: render methods + cover push status + skip reason."""
    artifacts = coerce_output_artifacts_dict(raw_artifacts)
    report = _parse_studio_render_report(raw_artifacts)
    methods = report.get("platform_render_methods")
    if not isinstance(methods, dict):
        raw_m = artifacts.get("platform_render_methods")
        if isinstance(raw_m, str) and raw_m.strip():
            try:
                methods = json.loads(raw_m)
            except Exception:
                methods = {}
        elif isinstance(raw_m, dict):
            methods = raw_m
        else:
            methods = {}
    push = artifacts.get("platform_thumb_push_status")
    if isinstance(push, str) and push.strip():
        try:
            push = json.loads(push)
        except Exception:
            push = {}
    if not isinstance(push, dict):
        push = {}
    return {
        "thumbnail_render_method": thumbnail_render_method_from_artifacts(raw_artifacts),
        "skip_reason": str(report.get("skip_reason") or "").strip() or None,
        "pikzels_requested_but_skipped": bool(
            report.get("pikzels_requested_but_skipped")
            or str(artifacts.get("pikzels_requested_but_skipped") or "") in ("1", "true", "yes")
        ),
        "platform_render_methods": methods,
        "platform_thumb_push_status": push,
        "instagram_cover_url_set": report.get("instagram_cover_url_set"),
        "studio_winner_apply_mode": report.get("studio_winner_apply_mode"),
    }


def pikzels_template_thumbnail_warning(raw_artifacts: Any) -> Optional[Dict[str, str]]:
    """
    When the server has PIKZELS_API_KEY but this upload used PIL template render,
    return a short warning for queue/upload UI.
    """
    method = thumbnail_render_method_from_artifacts(raw_artifacts)
    report = _parse_studio_render_report(raw_artifacts)
    requested = bool(
        report.get("pikzels_requested_but_skipped")
        or str(coerce_output_artifacts_dict(raw_artifacts).get("pikzels_requested_but_skipped") or "")
        in ("1", "true", "yes")
    )
    if method in ("studio_renderer", "studio_winner_cover_direct") and not requested:
        return None
    if method not in ("template", "none", "", "sticker_composite") and not requested:
        return None
    try:
        from services.pikzels_v2 import resolve_public_api_key

        if not (resolve_public_api_key() or "").strip() and not requested:
            return None
    except Exception:
        if not requested:
            return None
    skip_reason = str(report.get("skip_reason") or "").strip()
    msg = (
        "Pikzels was requested for this upload but was not used "
        f"({method or 'no render'})."
        if requested
        else (
            "This upload did not use Pikzels Studio (template or raw frame only). "
            "Turn on auto-thumbnails and Thumbnail Studio in Settings, and set render pipeline to Auto."
        )
    )
    if skip_reason == "tier_lacks_ai_thumbnail_styling":
        msg = (
            "Your plan can store custom thumbnails but Pikzels AI styling needs Creator Pro or higher. "
            "Upgrade to generate Pikzels covers on upload."
        )
    elif skip_reason == "persona_not_linked":
        msg = (
            "A persona was selected but is not linked to Pikzels. "
            "Open Thumbnail Studio → Personas → Link to Pikzels."
        )
    return {
        "code": "pikzels_requested_skipped" if requested else "pikzels_template_fallback",
        "message": msg,
        "settings_path": "settings.html#thumbnail-studio",
        "skip_reason": skip_reason,
        "upgrade_path": "billing.html" if skip_reason == "tier_lacks_ai_thumbnail_styling" else None,
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


def resolve_upload_thumbnail_r2_keys_ordered(
    *,
    thumbnail_r2_key: Optional[str],
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
) -> List[str]:
    """All candidate R2 keys for thumbnail streaming, best-first."""
    artifacts = artifact_platform_thumbnail_r2_keys(output_artifacts)
    platforms = [str(p).lower() for p in (upload_platforms or []) if p]
    order = platforms + [p for p in PLATFORM_THUMB_PRIORITY if p not in platforms]
    seen: set[str] = set()
    keys: List[str] = []

    def _add(k: Optional[str]) -> None:
        s = str(k or "").strip()
        if s and s not in seen:
            seen.add(s)
            keys.append(s)

    for plat in order:
        _add(artifacts.get(plat))
    for k in artifacts.values():
        _add(k)
    _add(thumbnail_r2_key)
    mirrored = mirrored_platform_thumbnail_r2_keys(platform_results)
    for plat in order:
        _add(mirrored.get(plat))
    for k in mirrored.values():
        _add(k)
    return keys


def resolve_upload_thumbnail_r2_key(
    *,
    thumbnail_r2_key: Optional[str],
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
) -> Optional[str]:
    """Best R2 object key for card/detail thumbnail streaming."""
    keys = resolve_upload_thumbnail_r2_keys_ordered(
        thumbnail_r2_key=thumbnail_r2_key,
        output_artifacts=output_artifacts,
        platform_results=platform_results,
        upload_platforms=upload_platforms,
    )
    return keys[0] if keys else None


def first_verified_thumbnail_r2_key(
    *,
    thumbnail_r2_key: Optional[str],
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
) -> Optional[str]:
    """First candidate R2 key that actually exists (avoids proxy 404s)."""
    for key in resolve_upload_thumbnail_r2_keys_ordered(
        thumbnail_r2_key=thumbnail_r2_key,
        output_artifacts=output_artifacts,
        platform_results=platform_results,
        upload_platforms=upload_platforms,
    ):
        norm = _normalize_r2_key(key)
        try:
            if r2_object_exists(norm):
                return key
        except Exception:
            continue
    return None


def enrich_posted_thumbnail_urls(
    platform_results: Any,
    posted_urls: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Add cheap offline fallbacks (e.g. YouTube CDN from video id) when sync has not run."""
    from services.thumbnail_studio import youtube_reference_thumbnail_url

    platform_results_norm = _normalize_platform_results_detail(platform_results)
    out = dict(posted_urls or {})
    out = dict(
        posted_platform_thumbnail_urls_from_results(platform_results_norm),
        **out,
    )
    for pr in platform_results_norm:
        if not isinstance(pr, dict) or pr.get("success") is False:
            continue
        plat = str(pr.get("platform") or "").lower().strip()
        if not plat or str(out.get(plat) or "").strip().startswith("http"):
            continue
        if plat == "youtube":
            vid = platform_video_id_from_result(pr)
            if vid:
                url = youtube_reference_thumbnail_url(vid)
                if url:
                    out[plat] = url
    return out


def _enrich_posted_thumbnail_urls(
    platform_results: List[dict],
    posted_urls: Dict[str, str],
) -> Dict[str, str]:
    return enrich_posted_thumbnail_urls(platform_results, posted_urls)


def posted_thumbnail_browser_url(
    *,
    output_artifacts: Any,
    platform_results: Any,
    upload_platforms: Optional[List[str]] = None,
) -> Optional[str]:
    """Browser-safe live platform cover URL (YouTube/TikTok CDN, etc.)."""
    platform_results_norm = _normalize_platform_results_detail(platform_results)
    posted_urls = _enrich_posted_thumbnail_urls(
        platform_results_norm,
        posted_platform_thumbnail_urls_from_results(platform_results_norm),
    )
    return browser_safe_thumbnail_url(
        pick_primary_thumbnail_url(
            posted=posted_urls,
            artifact_platform_urls={},
            r2_presigned=None,
            upload_platforms=list(upload_platforms or []),
        )
    )


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

    Resolution order (first match wins):
      1. Platform CDN URLs in platform_results (posted_platform_thumbnail_urls)
      2. YouTube CDN synthesized from video_id when sync has not run
      3. Presigned R2 artifact URLs (only keys verified present in R2)
      4. First-party proxy /api/uploads/{id}/thumbnail (only when R2 verified)
    """
    uid = str(upload_id or "").strip()
    platforms_list = list(upload_platforms or [])
    platform_results_norm = _normalize_platform_results_detail(platform_results)
    posted_urls = _enrich_posted_thumbnail_urls(
        platform_results_norm,
        posted_platform_thumbnail_urls_from_results(platform_results_norm),
    )

    posted_direct = browser_safe_thumbnail_url(
        pick_primary_thumbnail_url(
            posted=posted_urls,
            artifact_platform_urls={},
            r2_presigned=None,
            upload_platforms=platforms_list,
        )
    )
    if posted_direct:
        return posted_direct

    sk = str(thumbnail_r2_key or "").strip()
    r2_thumb_url = None
    if sk and presign_r2_thumbnails:
        r2_thumb_url = presign_upload_thumbnail_r2_key(sk, expires_in=3600)

    plat_thumb_urls = merged_platform_thumbnail_urls(output_artifacts, platform_results_norm)
    fallback = browser_safe_thumbnail_url(
        pick_primary_thumbnail_url(
            posted=posted_urls,
            artifact_platform_urls=plat_thumb_urls,
            r2_presigned=r2_thumb_url,
            upload_platforms=platforms_list,
        )
    )
    if fallback:
        return fallback

    if first_verified_thumbnail_r2_key(
        thumbnail_r2_key=thumbnail_r2_key,
        output_artifacts=output_artifacts,
        platform_results=platform_results_norm,
        upload_platforms=platforms_list,
    ):
        return upload_card_thumbnail_href(uid) if uid else None
    return None


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
    when ``thumbnail_url`` is present. Also flags when we fell back to a platform CDN
    cover but the primary ``thumbnail_r2_key`` still needs regeneration in R2.
    """
    uid = str(upload_id or "").strip()
    if not uid:
        return False
    proxy_href = upload_card_thumbnail_href(uid)
    if thumbnail_url == proxy_href:
        return False
    sk = str(primary_sk or "").strip()
    if not sk:
        return False
    resolved = resolve_upload_thumbnail_r2_key(
        thumbnail_r2_key=sk,
        output_artifacts=output_artifacts,
        platform_results=platform_results,
        upload_platforms=upload_platforms,
    )
    if resolved != sk:
        return False
    if not thumbnail_url:
        return True
    # Showing a platform CDN fallback while primary R2 key is still on record.
    return bool(posted_thumbnail_browser_url(
        output_artifacts=output_artifacts,
        platform_results=platform_results,
        upload_platforms=upload_platforms,
    ))


def collect_thumbnail_repair_ids(items: List[dict], *, limit: int = 15) -> List[str]:
    """Upload ids needing thumbnail repair or first-time R2 backfill (bounded)."""
    ids: List[str] = []
    for item in items:
        st = str(item.get("status") or "").lower()
        if st not in THUMBNAIL_REPAIR_STATUSES:
            continue
        uid = str(item.get("id") or "").strip()
        if not uid or uid in ids:
            continue
        needs = bool(item.get("thumbnail_storage_missing"))
        if not needs and not item.get("thumbnail_url"):
            pr = item.get("platform_results") or []
            has_publish = False
            if isinstance(pr, list):
                for row in pr:
                    if isinstance(row, dict) and row.get("success") is not False:
                        if platform_video_id_from_result(row):
                            has_publish = True
                            break
            needs = has_publish
        if not needs:
            continue
        ids.append(uid)
        if len(ids) >= limit:
            break
    return ids


async def repair_upload_thumbnails_batch(
    pool: Any,
    user_id: str,
    upload_ids: List[str],
) -> None:
    """Background: verify R2 object for primary thumb; regenerate when missing (max 15)."""
    from services.thumbnail_regenerate import ensure_upload_thumbnail_resident

    uids = [str(x).strip() for x in (upload_ids or []) if str(x).strip()][:15]
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
    # One SELECT for all ids — avoids shell/bootstrap N+1 (UPLOADM8-89).
    rows_by_id: Dict[str, dict] = {}
    try:
        async with pool.acquire() as conn:
            fetched = await conn.fetch(
                """
                SELECT id, status, thumbnail_r2_key, r2_key, processed_r2_key,
                       platforms, title, caption, user_preferences, output_artifacts,
                       platform_results
                FROM uploads
                WHERE user_id = $1 AND id = ANY($2::uuid[])
                """,
                user_id,
                uids,
            )
        for row in fetched or []:
            rows_by_id[str(row["id"])] = dict(row)
    except Exception as e:
        logger.warning("repair_upload_thumbnails_batch batch load failed: %s", e)
        return

    repaired = 0
    for upload_id in uids:
        try:
            d = rows_by_id.get(upload_id)
            if not d:
                continue
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


async def posted_thumbnail_fallback_for_upload(
    pool: Any,
    user_id: str,
    upload_id: str,
) -> Optional[str]:
    """Browser-safe platform cover URL when R2 streaming is unavailable."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT output_artifacts, platform_results, platforms
            FROM uploads
            WHERE id = $1 AND user_id = $2
            """,
            upload_id,
            user_id,
        )
    if not row:
        return None
    d = dict(row)
    return posted_thumbnail_browser_url(
        output_artifacts=d.get("output_artifacts"),
        platform_results=d.get("platform_results"),
        upload_platforms=list(d.get("platforms") or []),
    )


async def stream_upload_thumbnail_bytes(
    pool: Any,
    user_id: str,
    upload_id: str,
) -> tuple[bytes, str, str]:
    """
    Load thumbnail bytes for GET /api/uploads/{id}/thumbnail.

    Returns (body, content_type, etag_key). Raises HTTPException 404 when missing.
    Tries all candidate R2 keys (platform artifacts, primary, mirrored) before 404.
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
    candidate_keys = resolve_upload_thumbnail_r2_keys_ordered(
        thumbnail_r2_key=d.get("thumbnail_r2_key"),
        output_artifacts=d.get("output_artifacts"),
        platform_results=d.get("platform_results"),
        upload_platforms=list(d.get("platforms") or []),
    )
    if not candidate_keys:
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    def _read_key(norm_key: str) -> Optional[tuple[bytes, str]]:
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

    for r2_key in candidate_keys:
        norm_key = _normalize_r2_key(r2_key)
        read_out = await asyncio.to_thread(_read_key, norm_key)
        if read_out:
            raw, ct = read_out
            return raw, ct, norm_key

    raise HTTPException(status_code=404, detail="Thumbnail not available")


def merged_platform_thumbnail_urls(
    output_artifacts: Any,
    platform_results: Any,
    *,
    expires_in: int = 3600,
) -> dict:
    """UploadM8-generated R2 previews; live posted covers fill gaps only."""
    platform_results_norm = _normalize_platform_results_detail(platform_results)
    artifact_urls = platform_thumbnail_urls_from_artifacts(output_artifacts, expires_in=expires_in)
    posted_urls = enrich_posted_thumbnail_urls(platform_results_norm)
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
        norm = _normalize_r2_key(k)
        try:
            if not r2_object_exists(norm):
                continue
        except Exception:
            continue
        try:
            out[plat] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": norm},
                ExpiresIn=expires_in,
            )
        except Exception:
            continue
    return out
