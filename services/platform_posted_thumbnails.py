"""
Live thumbnail URLs from published posts (YouTube, TikTok, Instagram, Facebook).

Used by uploads list/detail and sync-analytics so dashboard/queue cards show what is
actually on each platform, not only locally generated R2 frames.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.auth import decrypt_blob
from services.platform_oauth_refresh import refresh_decrypted_token_for_row
from services.sync_analytics_helpers import resolve_token_candidates_for_platform_result
from services.thumbnail_studio import youtube_reference_thumbnail_url

logger = logging.getLogger(__name__)

PLATFORM_THUMB_PRIORITY = ("youtube", "instagram", "tiktok", "facebook")


def _plat_token_resolution_maps(
    token_rows: list,
    token_map_by_id: Dict[str, dict],
    token_map_by_platform: Dict[str, dict],
) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], Tuple[str, dict]], Dict[str, List[Tuple[str, dict]]]]:
    token_map_by_plat_account: Dict[Tuple[str, str], dict] = {}
    plat_account_row_map: Dict[Tuple[str, str], Tuple[str, dict]] = {}
    platform_token_rows: Dict[str, List[Tuple[str, dict]]] = {}
    for tr in token_rows:
        tid = str(tr["id"])
        dec = token_map_by_id.get(tid)
        if not dec:
            continue
        plat = str(tr.get("platform") or "").lower()
        aid = tr.get("account_id")
        if aid is not None and str(aid).strip() != "":
            a = str(aid).strip()
            token_map_by_plat_account[(plat, a)] = dec
            plat_account_row_map[(plat, a)] = (tid, dec)
        platform_token_rows.setdefault(plat, []).append((tid, dec))
    return token_map_by_plat_account, plat_account_row_map, platform_token_rows


def platform_video_id_from_result(pr: Dict[str, Any]) -> Optional[str]:
    vid = (
        pr.get("platform_video_id")
        or pr.get("video_id")
        or pr.get("videoId")
        or pr.get("media_id")
        or pr.get("post_id")
        or pr.get("share_id")
    )
    s = str(vid or "").strip()
    return s or None


def posted_platform_thumbnail_urls_from_results(pr_list: Any) -> Dict[str, str]:
    """Read cached live thumbnail URLs already stored on platform_results rows."""
    out: Dict[str, str] = {}
    if not isinstance(pr_list, list):
        return out
    for pr in pr_list:
        if not isinstance(pr, dict) or pr.get("success") is False:
            continue
        plat = str(pr.get("platform") or "").lower().strip()
        url = str(
            pr.get("platform_thumbnail_url")
            or pr.get("thumbnail_url")
            or pr.get("cover_image_url")
            or ""
        ).strip()
        if plat and url.startswith("http"):
            out[plat] = url
    return out


def pick_primary_thumbnail_url(
    *,
    posted: Dict[str, str],
    artifact_platform_urls: Optional[Dict[str, str]] = None,
    r2_presigned: Optional[str] = None,
    upload_platforms: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Choose the best thumbnail for a list card.

    Prefer UploadM8-generated covers (Pikzels / persona / styled per platform), then the
    primary R2 frame, then live platform CDN URLs fetched after publish.
    """
    artifacts = artifact_platform_urls or {}
    platforms = [str(p).lower() for p in (upload_platforms or []) if p]
    order = platforms + [p for p in PLATFORM_THUMB_PRIORITY if p not in platforms]
    for plat in order:
        u = (artifacts.get(plat) or "").strip()
        if u.startswith("http"):
            return u
    if artifacts:
        for u in artifacts.values():
            if isinstance(u, str) and u.startswith("http"):
                return u
    r2 = (r2_presigned or "").strip()
    if r2.startswith("http"):
        return r2
    for plat in order:
        u = (posted.get(plat) or "").strip()
        if u.startswith("http"):
            return u
    for u in posted.values():
        if isinstance(u, str) and u.startswith("http"):
            return u
    return None


async def load_user_token_maps(pool: Any, user_id: str) -> Tuple[
    Dict[str, dict],
    Dict[str, dict],
    Dict[Tuple[str, str], dict],
    Dict[Tuple[str, str], Tuple[str, dict]],
    Dict[str, List[Tuple[str, dict]]],
]:
    """Build the same token lookup maps used by sync-analytics."""
    async with pool.acquire() as conn:
        token_rows = await conn.fetch(
            """
            SELECT id, platform, token_blob, account_id
            FROM platform_tokens
            WHERE user_id = $1 AND revoked_at IS NULL
            """,
            user_id,
        )

    token_map_by_id: Dict[str, dict] = {}
    token_map_by_platform: Dict[str, dict] = {}
    uid = str(user_id)
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if not dec:
                continue
            if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                dec["ig_user_id"] = str(tr["account_id"])
            if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                dec["page_id"] = str(tr["account_id"])
            token_id = str(tr["id"])
            dec = await refresh_decrypted_token_for_row(
                tr["platform"],
                dec,
                db_pool=pool,
                user_id=uid,
                token_row_id=token_id,
            )
            token_map_by_id[token_id] = dec
            plat_norm = str(tr.get("platform") or "").lower()
            if plat_norm:
                token_map_by_platform[plat_norm] = dec
        except Exception:
            continue

    plat_account, plat_row_map, platform_rows = _plat_token_resolution_maps(
        list(token_rows), token_map_by_id, token_map_by_platform
    )
    return token_map_by_id, token_map_by_platform, plat_account, plat_row_map, platform_rows


def _youtube_thumb_from_snippet(thumbs: Any) -> Optional[str]:
    if not isinstance(thumbs, dict):
        return None
    for name in ("maxres", "standard", "high", "medium", "default"):
        row = thumbs.get(name)
        if isinstance(row, dict) and row.get("url"):
            return str(row["url"])
    return None


async def fetch_posted_thumbnail_for_platform(
    client: httpx.AsyncClient,
    plat: str,
    video_id: str,
    pr: Dict[str, Any],
    access_token: str,
) -> Optional[str]:
    """Fetch the current cover/thumbnail URL for one published video."""
    plat = str(plat or "").lower().strip()
    vid = str(video_id or "").strip()
    if not plat or not vid:
        return None

    try:
        if plat == "youtube":
            if access_token:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/videos",
                    params={"part": "snippet", "id": vid},
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if resp.status_code == 200:
                    items = resp.json().get("items") or []
                    if items:
                        thumbs = (items[0].get("snippet") or {}).get("thumbnails")
                        url = _youtube_thumb_from_snippet(thumbs)
                        if url:
                            return url
            pub = youtube_reference_thumbnail_url(vid)
            return pub or None

        if plat == "tiktok" and access_token:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/video/query/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                params={"fields": "id,cover_image_url"},
                json={"filters": {"video_ids": [vid]}},
            )
            if resp.status_code == 200:
                vids = resp.json().get("data", {}).get("videos", []) or []
                if vids:
                    u = vids[0].get("cover_image_url")
                    if u:
                        return str(u)

        if plat == "instagram" and access_token:
            media_id = pr.get("platform_video_id") or pr.get("media_id") or vid
            resp = await client.get(
                f"https://graph.facebook.com/v21.0/{media_id}",
                params={"fields": "thumbnail_url", "access_token": access_token},
            )
            if resp.status_code == 200:
                u = resp.json().get("thumbnail_url")
                if u:
                    return str(u)

        if plat == "facebook" and access_token:
            resp = await client.get(
                f"https://graph.facebook.com/v21.0/{vid}",
                params={"fields": "picture", "access_token": access_token},
            )
            if resp.status_code == 200:
                body = resp.json()
                pic = body.get("picture")
                if isinstance(pic, str) and pic.startswith("http"):
                    return pic
                if isinstance(pic, dict) and pic.get("data", {}).get("url"):
                    return str(pic["data"]["url"])
    except Exception as e:
        logger.debug("posted thumbnail fetch %s/%s: %s", plat, vid[:24], e)
    return None


async def sync_posted_thumbnails_for_platform_results(
    pr_list: List[Dict[str, Any]],
    *,
    token_map_by_id: Dict[str, dict],
    token_map_by_plat_account: Dict[Tuple[str, str], dict],
    token_map_by_platform: Dict[str, dict],
    plat_account_row_map: Dict[Tuple[str, str], Tuple[str, dict]],
    platform_token_rows: Dict[str, List[Tuple[str, dict]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], bool]:
    """
    Fill ``platform_thumbnail_url`` on each successful platform_results entry.

    Returns (updated pr_list, posted map, changed).
    """
    if not pr_list:
        return pr_list, {}, False

    changed = False
    posted: Dict[str, str] = posted_platform_thumbnail_urls_from_results(pr_list)

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            if not isinstance(pr, dict) or pr.get("success") is False:
                continue
            plat = str(pr.get("platform") or "").lower()
            video_id = platform_video_id_from_result(pr)
            if not plat or not video_id:
                continue
            existing = str(pr.get("platform_thumbnail_url") or "").strip()
            if existing.startswith("http"):
                posted.setdefault(plat, existing)
                continue

            candidates = resolve_token_candidates_for_platform_result(
                pr,
                token_map_by_id,
                token_map_by_plat_account,
                token_map_by_platform,
                plat_account_row_map=plat_account_row_map,
                platform_token_rows=platform_token_rows,
            )
            url: Optional[str] = None
            for tok in candidates:
                at = (tok or {}).get("access_token", "")
                url = await fetch_posted_thumbnail_for_platform(
                    client, plat, video_id, pr, at
                )
                if url:
                    break
            if not url and plat == "youtube":
                url = youtube_reference_thumbnail_url(video_id) or None

            if url:
                pr["platform_thumbnail_url"] = url
                posted[plat] = url
                changed = True

    return pr_list, posted, changed


async def sync_posted_thumbnails_for_upload(
    pool: Any,
    user_id: str,
    upload_row: Dict[str, Any],
    *,
    token_bundle: Optional[
        Tuple[
            Dict[str, dict],
            Dict[str, dict],
            Dict[Tuple[str, str], dict],
            Dict[Tuple[str, str], Tuple[str, dict]],
            Dict[str, List[Tuple[str, dict]]],
        ]
    ] = None,
) -> Dict[str, str]:
    """
    Fetch live platform covers for one upload and persist platform_results when changed.

    Returns platform -> thumbnail URL map.
    """
    raw_pr = upload_row.get("platform_results")
    pr_list: List[Dict[str, Any]] = []
    if isinstance(raw_pr, list):
        pr_list = [dict(x) for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, str) and raw_pr.strip():
        try:
            j = json.loads(raw_pr)
            if isinstance(j, list):
                pr_list = [dict(x) for x in j if isinstance(x, dict)]
        except Exception:
            pr_list = []
    elif isinstance(raw_pr, dict):
        pr_list = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw_pr.items()]

    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]

    if not pr_list:
        return {}

    if token_bundle is None:
        token_bundle = await load_user_token_maps(pool, user_id)

    (
        token_map_by_id,
        token_map_by_platform,
        token_map_by_plat_account,
        plat_account_row_map,
        platform_token_rows,
    ) = token_bundle

    pr_list, posted, changed = await sync_posted_thumbnails_for_platform_results(
        pr_list,
        token_map_by_id=token_map_by_id,
        token_map_by_plat_account=token_map_by_plat_account,
        token_map_by_platform=token_map_by_platform,
        plat_account_row_map=plat_account_row_map,
        platform_token_rows=platform_token_rows,
    )

    if changed:
        upload_id = str(upload_row.get("id") or "").strip()
        if upload_id:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET platform_results = $1::jsonb, updated_at = NOW()
                    WHERE id = $2 AND user_id = $3
                    """,
                    json.dumps(pr_list),
                    upload_id,
                    user_id,
                )
        upload_row["platform_results"] = pr_list

    return posted


def parse_platform_results_list(raw: Any) -> List[Dict[str, Any]]:
    pr_list: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        pr_list = [dict(x) for x in raw if isinstance(x, dict)]
    elif isinstance(raw, str) and raw.strip():
        try:
            j = json.loads(raw)
            if isinstance(j, list):
                pr_list = [dict(x) for x in j if isinstance(x, dict)]
        except Exception:
            pr_list = []
    elif isinstance(raw, dict):
        pr_list = [
            {"platform": k, **v} if isinstance(v, dict) else {"platform": k}
            for k, v in raw.items()
        ]
    return pr_list


def upload_needs_posted_thumbnail_sync(upload_row: Dict[str, Any]) -> bool:
    """True when a published row has platform video ids but no cached live cover yet."""
    st = str(upload_row.get("status") or "").lower()
    if st not in ("completed", "succeeded", "partial"):
        return False
    pr_list = parse_platform_results_list(upload_row.get("platform_results"))
    if not pr_list:
        return False
    needs = False
    has_success = False
    for pr in pr_list:
        if pr.get("success") is False:
            continue
        if not platform_video_id_from_result(pr):
            continue
        has_success = True
        url = str(pr.get("platform_thumbnail_url") or "").strip()
        if not url.startswith("http"):
            needs = True
    return has_success and needs


def upload_ids_needing_posted_thumbnail_sync(upload_row: Dict[str, Any]) -> bool:
    return upload_needs_posted_thumbnail_sync(upload_row)


async def background_sync_posted_thumbnails(
    pool: Any,
    user_id: str,
    upload_ids: List[str],
) -> int:
    """Background worker: fetch live platform covers and persist platform_results."""
    if not upload_ids:
        return 0
    synced = 0
    try:
        token_bundle = await load_user_token_maps(pool, user_id)
    except Exception as e:
        logger.warning("background_sync_posted_thumbnails token load failed user=%s: %s", user_id, e)
        return 0

    for upload_id in upload_ids:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, status, platform_results, thumbnail_r2_key,
                           output_artifacts, platforms, r2_key, processed_r2_key
                    FROM uploads
                    WHERE id = $1 AND user_id = $2
                    """,
                    upload_id,
                    user_id,
                )
            if not row:
                continue
            d = dict(row)
            if not upload_needs_posted_thumbnail_sync(d):
                continue
            await sync_posted_thumbnails_for_upload(
                pool,
                user_id,
                d,
                token_bundle=token_bundle,
            )
            synced += 1
        except Exception as e:
            logger.warning(
                "background_sync_posted_thumbnails upload=%s: %s", upload_id, e
            )
        await asyncio.sleep(0.28)
    return synced
