"""Per-upload engagement sync writers (uploads columns, platform_results, PCI).

Moved out of routers/uploads_analytics.py so the HTTP module stays under
router-lint line caps. Shared Meta fetchers: services.meta_graph_metrics.
Catalog list ingest: services.catalog_sync. Headline KPIs: canonical_engagement.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx
from fastapi import HTTPException

import core.state
from core.auth import decrypt_blob
from core.helpers import _safe_json
from services.platform_oauth_refresh import refresh_decrypted_token_for_row
from services.sync_analytics_helpers import resolve_token_candidates_for_platform_result
from services.workspace import resolve_billing_user_id

logger = logging.getLogger("uploadm8.upload_analytics_sync")

_sync_analytics_running: set[str] = set()
_sync_analytics_pending: Dict[str, List[str]] = {}


async def mirror_pci_metrics_into_uploads(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    platform: Optional[str] = None,
    account_id: Optional[str] = None,
    limit: int = 500,
) -> int:
    """
    Copy catalog (PCI) engagement onto uploads so dashboard chips update
    even when live Graph insights fail or sync-analytics is still queued.

    Matches by pci.upload_id and/or platform_video_id inside platform_results;
    GREATEST so we never clobber higher live stats.
    """
    uid = str(user_id)
    plat = str(platform or "").strip().lower() or None
    aid = str(account_id or "").strip() or None
    async with pool.acquire() as conn:
        pci_rows = await conn.fetch(
            """
            SELECT pci.upload_id, pci.platform, pci.platform_video_id,
                   pci.views, pci.likes, pci.comments, pci.shares, pci.platform_url
              FROM platform_content_items pci
             WHERE pci.user_id = $1::uuid
               AND pci.platform_video_id IS NOT NULL AND pci.platform_video_id != ''
               AND ($2::text IS NULL OR pci.platform = $2)
               AND ($3::text IS NULL OR pci.account_id = $3)
               AND (
                    COALESCE(pci.views, 0) > 0
                 OR COALESCE(pci.likes, 0) > 0
                 OR COALESCE(pci.comments, 0) > 0
                 OR COALESCE(pci.shares, 0) > 0
               )
             ORDER BY pci.updated_at DESC NULLS LAST
             LIMIT $4
            """,
            uid,
            plat,
            aid,
            int(limit),
        )
        if not pci_rows:
            return 0

        pci_by_key: Dict[Tuple[str, str], Any] = {}
        pci_by_upload: Dict[str, List[Any]] = {}
        for r in pci_rows:
            key = (
                str(r["platform"] or "").lower(),
                str(r["platform_video_id"] or "").strip(),
            )
            if key[1]:
                prev = pci_by_key.get(key)
                if prev is None or int(r["views"] or 0) >= int(prev["views"] or 0):
                    pci_by_key[key] = r
            if r["upload_id"]:
                pci_by_upload.setdefault(str(r["upload_id"]), []).append(r)

        upload_rows = await conn.fetch(
            """
            SELECT id, platform_results, views, likes, comments, shares
              FROM uploads
             WHERE user_id = $1::uuid
               AND platform_results IS NOT NULL
             ORDER BY updated_at DESC NULLS LAST
             LIMIT 800
            """,
            uid,
        )

        mirrored = 0
        for urow in upload_rows:
            upload_id = str(urow["id"])
            raw = _safe_json(urow["platform_results"], [])
            pr_list: List[dict] = []
            if isinstance(raw, list):
                pr_list = [x for x in raw if isinstance(x, dict)]
            elif isinstance(raw, dict):
                pr_list = [
                    {"platform": k, **v} if isinstance(v, dict) else {"platform": k}
                    for k, v in raw.items()
                ]
            if not pr_list:
                continue

            linked_pci = pci_by_upload.get(upload_id) or []
            changed = False
            total_v = total_l = total_c = total_s = 0
            for pr in pr_list:
                plat_k = str(pr.get("platform") or "").lower()
                if plat and plat_k and plat_k != plat:
                    total_v += int(pr.get("views") or pr.get("view_count") or 0)
                    total_l += int(pr.get("likes") or pr.get("like_count") or 0)
                    total_c += int(pr.get("comments") or pr.get("comment_count") or 0)
                    total_s += int(pr.get("shares") or pr.get("share_count") or 0)
                    continue
                vid = str(
                    pr.get("platform_video_id")
                    or pr.get("video_id")
                    or pr.get("videoId")
                    or pr.get("media_id")
                    or ""
                ).strip()
                hit = pci_by_key.get((plat_k, vid)) if vid else None
                if hit is None and len(linked_pci) == 1:
                    only = linked_pci[0]
                    if str(only["platform"] or "").lower() == plat_k:
                        hit = only
                        if not vid and only["platform_video_id"]:
                            pr["platform_video_id"] = str(only["platform_video_id"])
                            pr["video_id"] = pr["platform_video_id"]
                            changed = True
                if hit is not None:
                    before = (
                        int(pr.get("views") or pr.get("view_count") or 0),
                        int(pr.get("likes") or pr.get("like_count") or 0),
                        int(pr.get("comments") or pr.get("comment_count") or 0),
                        int(pr.get("shares") or pr.get("share_count") or 0),
                    )
                    _merge_stats_into_platform_result(
                        pr,
                        {
                            "views": int(hit["views"] or 0),
                            "likes": int(hit["likes"] or 0),
                            "comments": int(hit["comments"] or 0),
                            "shares": int(hit["shares"] or 0),
                            "platform_url": hit.get("platform_url"),
                        },
                    )
                    after = (
                        int(pr.get("views") or 0),
                        int(pr.get("likes") or 0),
                        int(pr.get("comments") or 0),
                        int(pr.get("shares") or 0),
                    )
                    if after != before:
                        changed = True
                total_v += int(pr.get("views") or pr.get("view_count") or 0)
                total_l += int(pr.get("likes") or pr.get("like_count") or 0)
                total_c += int(pr.get("comments") or pr.get("comment_count") or 0)
                total_s += int(pr.get("shares") or pr.get("share_count") or 0)

            if not changed:
                continue
            await conn.execute(
                """
                UPDATE uploads SET
                    platform_results = $1::jsonb,
                    views = GREATEST(COALESCE(views, 0), $2),
                    likes = GREATEST(COALESCE(likes, 0), $3),
                    comments = GREATEST(COALESCE(comments, 0), $4),
                    shares = GREATEST(COALESCE(shares, 0), $5),
                    updated_at = NOW()
                 WHERE id = $6::uuid AND user_id = $7::uuid
                """,
                json.dumps(pr_list),
                total_v,
                total_l,
                total_c,
                total_s,
                upload_id,
                uid,
            )
            mirrored += 1
        return mirrored


async def _upsert_pci_metrics_from_platform_results(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    upload_id: str,
    pr_list: List[dict],
    published_at=None,
) -> int:
    """
    Write sync-analytics engagement into platform_content_items (GREATEST).

    Catalog cards / Top Performing (catalog toggle) read PCI; without this bridge
    LIVE API and uploads columns can look healthy while catalog stays at 0.
    """
    from services.content_success_features import entry_metrics, entry_successful

    n = 0
    for pr in pr_list or []:
        if not isinstance(pr, dict) or not entry_successful(pr):
            continue
        plat = str(pr.get("platform") or "").strip().lower()
        if plat not in ("tiktok", "youtube", "instagram", "facebook"):
            continue
        vid = str(
            pr.get("platform_video_id")
            or pr.get("video_id")
            or pr.get("videoId")
            or pr.get("media_id")
            or pr.get("post_id")
            or ""
        ).strip()
        if not vid:
            continue
        account_id = str(
            pr.get("account_id")
            or pr.get("open_id")
            or pr.get("page_id")
            or pr.get("ig_user_id")
            or ""
        ).strip()
        if not account_id:
            # Prefer linked token row account when present.
            account_id = str(pr.get("token_account_id") or "").strip()
        if not account_id:
            continue
        m = entry_metrics(pr, plat)
        if (m["views"] + m["likes"] + m["comments"] + m["shares"]) <= 0:
            continue
        try:
            await conn.execute(
                """
                INSERT INTO platform_content_items
                    (user_id, platform, account_id, platform_video_id,
                     upload_id, source, published_at,
                     views, likes, comments, shares, metrics_synced_at, updated_at)
                VALUES (
                    $1::uuid, $2, $3, $4,
                    $5::uuid, 'uploadm8', COALESCE($6::timestamptz, NOW()),
                    $7, $8, $9, $10, NOW(), NOW()
                )
                ON CONFLICT (user_id, platform, account_id, platform_video_id) DO UPDATE SET
                    upload_id = COALESCE(EXCLUDED.upload_id, platform_content_items.upload_id),
                    source = CASE
                        WHEN platform_content_items.source = 'external' THEN 'linked'
                        ELSE COALESCE(platform_content_items.source, 'uploadm8')
                    END,
                    views = GREATEST(COALESCE(platform_content_items.views, 0), EXCLUDED.views),
                    likes = GREATEST(COALESCE(platform_content_items.likes, 0), EXCLUDED.likes),
                    comments = GREATEST(COALESCE(platform_content_items.comments, 0), EXCLUDED.comments),
                    shares = GREATEST(COALESCE(platform_content_items.shares, 0), EXCLUDED.shares),
                    metrics_synced_at = NOW(),
                    updated_at = NOW()
                """,
                user_id,
                plat,
                account_id,
                vid,
                upload_id,
                published_at,
                int(m["views"]),
                int(m["likes"]),
                int(m["comments"]),
                int(m["shares"]),
            )
            n += 1
        except Exception as e:
            logger.debug(
                "pci metrics upsert skipped upload=%s plat=%s: %s",
                upload_id[:8] if upload_id else "",
                plat,
                e,
            )
    return n


async def _fetch_platform_video_engagement(
    client: httpx.AsyncClient,
    plat: str,
    video_id: str,
    pr: dict,
    access_token: str,
) -> Optional[Dict[str, int]]:
    """Call the platform metrics API for one video/reel/post."""
    if not access_token:
        return None
    try:
        if plat == "tiktok" and video_id:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/video/query/",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                params={"fields": "id,view_count,like_count,comment_count,share_count"},
                json={"filters": {"video_ids": [str(video_id)]}},
            )
            if resp.status_code != 200:
                return None
            vids = resp.json().get("data", {}).get("videos", []) or []
            if not vids:
                return None
            v = vids[0]
            return {
                "views": int(v.get("view_count") or 0),
                "likes": int(v.get("like_count") or 0),
                "comments": int(v.get("comment_count") or 0),
                "shares": int(v.get("share_count") or 0),
            }

        if plat == "youtube" and video_id:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "statistics", "id": str(video_id)},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                return None
            items = resp.json().get("items", []) or []
            if not items:
                return None
            st = items[0].get("statistics", {})
            return {
                "views": int(st.get("viewCount") or 0),
                "likes": int(st.get("likeCount") or 0),
                "comments": int(st.get("commentCount") or 0),
                "shares": 0,
            }

        if plat == "instagram" and video_id:
            from services.meta_graph_metrics import (
                extract_instagram_shortcode,
                fetch_instagram_media_engagement_result,
                resolve_instagram_media_id_by_shortcode,
            )

            media_id = str(
                pr.get("platform_video_id") or pr.get("media_id") or video_id or ""
            ).strip()
            pr.pop("_sync_fail_reason", None)
            detailed = await fetch_instagram_media_engagement_result(
                client, access_token, media_id
            )
            s = detailed.get("metrics")
            fail_reason = str(detailed.get("reason") or "")
            if s is not None:
                return s
            if fail_reason == "rate_limited":
                pr["_sync_fail_reason"] = "rate_limited"
                return None
            # Dead/container ids: resolve live media id from reel shortcode, then retry.
            sc = extract_instagram_shortcode(
                pr.get("shortcode"),
                pr.get("platform_url"),
                pr.get("url"),
            )
            ig_user = str(
                (pr.get("_ig_user_id") or pr.get("ig_user_id") or pr.get("account_id") or "")
            ).strip()
            if not sc or not ig_user:
                # Still try caption-only resolve when shortcode missing.
                if ig_user and (pr.get("_caption_hint") or pr.get("title")):
                    resolved = await resolve_instagram_media_id_by_shortcode(
                        client,
                        access_token,
                        ig_user,
                        "",
                        caption_hint=str(pr.get("_caption_hint") or pr.get("title") or ""),
                    )
                    if resolved and resolved != media_id:
                        pr["platform_video_id"] = resolved
                        pr["video_id"] = resolved
                        retry = await fetch_instagram_media_engagement_result(
                            client, access_token, resolved
                        )
                        if retry.get("metrics") is not None:
                            return retry.get("metrics")
                if fail_reason == "not_found":
                    pr["_sync_fail_reason"] = "instagram_media_missing"
                return None
            resolved = await resolve_instagram_media_id_by_shortcode(
                client,
                access_token,
                ig_user,
                sc or "",
                caption_hint=str(pr.get("_caption_hint") or pr.get("title") or ""),
            )
            if not resolved or resolved == media_id:
                # Shortcode not on connected account media list → deleted / wrong account.
                pr["_sync_fail_reason"] = "instagram_media_missing"
                return None
            pr["platform_video_id"] = resolved
            pr["video_id"] = resolved
            pr["shortcode"] = sc
            retry = await fetch_instagram_media_engagement_result(
                client, access_token, resolved
            )
            if retry.get("reason") == "rate_limited":
                pr["_sync_fail_reason"] = "rate_limited"
                return None
            if retry.get("metrics") is None:
                pr["_sync_fail_reason"] = "instagram_media_missing"
            return retry.get("metrics")

        if plat == "facebook" and video_id:
            from services.meta_graph_metrics import fetch_facebook_video_engagement
            return await fetch_facebook_video_engagement(
                client, access_token, str(video_id)
            )
    except Exception as e:
        logger.warning("sync-analytics fetch %s/%s: %s", plat, video_id, e)
        return None
    return None


def _merge_stats_into_platform_result(pr: dict, s: Dict[str, int]) -> None:
    """Max-merge live stats into PR — never clobber non-zero with a transient zero."""
    def _max_metric(*keys: str, incoming: int) -> int:
        prev = 0
        for k in keys:
            try:
                prev = max(prev, int(pr.get(k) or 0))
            except (TypeError, ValueError):
                continue
        return max(prev, int(incoming or 0))

    views = _max_metric(
        "views",
        "view_count",
        "video_view_count",
        "play_count",
        "plays",
        "total_views",
        "fb_reels_total_plays",
        "blue_reels_play_count",
        incoming=s.get("views", 0),
    )
    likes = _max_metric("likes", "like_count", "reactions", incoming=s.get("likes", 0))
    comments = _max_metric("comments", "comment_count", incoming=s.get("comments", 0))
    shares = _max_metric("shares", "share_count", incoming=s.get("shares", 0))
    pr["views"] = views
    pr["view_count"] = views
    pr["likes"] = likes
    pr["like_count"] = likes
    pr["comments"] = comments
    pr["comment_count"] = comments
    pr["shares"] = shares
    pr["share_count"] = shares
    url = str(s.get("platform_url") or "").strip()
    if url and not (pr.get("platform_url") or pr.get("url")):
        pr["platform_url"] = url
        pr["url"] = url
    sc = str(s.get("shortcode") or "").strip()
    if sc and not pr.get("shortcode"):
        pr["shortcode"] = sc


def _pr_publish_id(pr: dict) -> str:
    return str(pr.get("publish_id") or pr.get("publishId") or "").strip()


def _pr_raw_video_id(pr: dict) -> str:
    return str(
        pr.get("platform_video_id")
        or pr.get("video_id")
        or pr.get("videoId")
        or pr.get("id")
        or pr.get("media_id")
        or pr.get("post_id")
        or pr.get("share_id")
        or ""
    ).strip()


def _pr_queryable_video_id(pr: dict) -> Optional[str]:
    """Video id usable for live metrics APIs (not a TikTok publish_id placeholder)."""
    vid = _pr_raw_video_id(pr)
    if not vid or vid.lower() == "null":
        return None
    pub = _pr_publish_id(pr)
    if pub and vid == pub:
        return None
    return vid


def _tiktok_needs_video_id_backfill(pr_list: List[dict]) -> bool:
    for pr in pr_list:
        if not isinstance(pr, dict):
            continue
        if str(pr.get("platform") or "").lower() != "tiktok":
            continue
        if _pr_queryable_video_id(pr) is None:
            return True
    return False


async def _reload_platform_results_list(
    conn: asyncpg.Connection, *, upload_id: str, user_id: str
) -> List[dict]:
    row = await conn.fetchrow(
        "SELECT platform_results FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id,
        user_id,
    )
    if not row:
        return []
    raw_pr = _safe_json(row["platform_results"], [])
    pr_list: List[dict] = []
    if isinstance(raw_pr, list):
        pr_list = [x for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, dict):
        pr_list = [
            {"platform": k, **v} if isinstance(v, dict) else {"platform": k}
            for k, v in raw_pr.items()
        ]
    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]
        if pr.get("platform_url") and not pr.get("url"):
            pr["url"] = pr["platform_url"]
    return pr_list


async def _backfill_tiktok_ids_before_metrics(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    pr_list: List[dict],
) -> None:
    """Use catalog titles to replace TikTok publish_id placeholders when PCI exists."""
    if not _tiktok_needs_video_id_backfill(pr_list):
        return
    from services.catalog_sync import _backfill_tiktok_video_ids

    account_ids = {
        str(pr.get("account_id") or "").strip()
        for pr in pr_list
        if isinstance(pr, dict)
        and str(pr.get("platform") or "").lower() == "tiktok"
        and str(pr.get("account_id") or "").strip()
    }
    if not account_ids:
        rows = await conn.fetch(
            """
            SELECT DISTINCT account_id FROM platform_tokens
             WHERE user_id = $1::uuid AND platform = 'tiktok' AND revoked_at IS NULL
               AND account_id IS NOT NULL AND account_id != ''
            """,
            user_id,
        )
        account_ids = {str(r["account_id"]).strip() for r in rows if r["account_id"]}
    for aid in account_ids:
        try:
            await _backfill_tiktok_video_ids(conn, user_id, aid)
        except Exception as e:
            logger.warning("[sync-analytics] TikTok video-id backfill failed for %s: %s", aid[:12], e)


def _apply_pci_row_to_pr(pr: dict, chosen: Any) -> None:
    """Copy catalog video id / URL / metrics onto a platform_results entry."""
    vid = str(chosen["platform_video_id"] or "").strip()
    if not vid:
        return
    cur = str(
        pr.get("platform_video_id")
        or pr.get("video_id")
        or pr.get("videoId")
        or ""
    ).strip()
    publish_id = str(pr.get("publish_id") or pr.get("publishId") or "").strip()
    plat = str(pr.get("platform") or "").lower()
    # Prefer PCI id when missing, or when PR still holds publish_id as the video id.
    should_replace = (not cur) or (publish_id and cur == publish_id)
    if not should_replace and plat == "tiktok" and not publish_id and cur != vid:
        should_replace = True
    if should_replace:
        pr["platform_video_id"] = vid
        pr["video_id"] = vid
    pci_url = str(chosen.get("platform_url") or "").strip()
    cur_url = str(pr.get("platform_url") or pr.get("url") or "").strip().lower()
    pci_is_watch = any(p in pci_url.lower() for p in ("/reel/", "/p/", "/videos/", "/watch"))
    cur_is_watch = any(p in cur_url for p in ("/reel/", "/p/", "/videos/", "/watch"))
    if pci_url and (not cur_url or (pci_is_watch and not cur_is_watch)):
        pr["platform_url"] = pci_url
        pr["url"] = pci_url
    # Stash catalog metrics for fallback if live APIs return empty.
    pr["_pci_metrics"] = {
        "views": int(chosen["views"] or 0),
        "likes": int(chosen["likes"] or 0),
        "comments": int(chosen["comments"] or 0),
        "shares": int(chosen["shares"] or 0),
    }


async def _hydrate_platform_results_from_pci(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    upload_id: str,
    pr_list: List[dict],
) -> None:
    """
    After catalog sync, PCI may hold the real TikTok ``platform_video_id`` (and metrics)
    while ``platform_results`` still has only a publish_id. Patch PR entries in-place so
    live API sync and token matching use the catalog id.

    Also matches by ``platform_video_id`` when upload_id is not linked yet — catalog
    likes/views can still fill the row if Graph returns empty.
    """
    if not pr_list:
        return
    rows = await conn.fetch(
        """
        SELECT platform, account_id, platform_video_id, views, likes, comments, shares, platform_url
          FROM platform_content_items
         WHERE user_id = $1::uuid AND upload_id = $2::uuid
           AND platform_video_id IS NOT NULL AND platform_video_id != ''
        """,
        user_id,
        upload_id,
    )
    by_plat: Dict[str, list] = {}
    for r in rows:
        plat = str(r["platform"] or "").lower()
        by_plat.setdefault(plat, []).append(r)

    for pr in pr_list:
        if not isinstance(pr, dict):
            continue
        plat = str(pr.get("platform") or "").lower()
        matches = by_plat.get(plat) or []
        aid = str(pr.get("account_id") or "").strip()
        chosen = None
        if matches:
            if aid:
                for r in matches:
                    if str(r["account_id"] or "").strip() == aid:
                        chosen = r
                        break
            if chosen is None:
                # upload_id already scopes PCI rows. One row for this platform is safe
                # even when PR account_id is stale after reconnect.
                if len(matches) == 1:
                    chosen = matches[0]
        if chosen is None:
            # Catalog row may exist as external (no upload_id) after Sync Catalog —
            # match exact Graph media/video id so metrics fallback still works.
            cur_vid = str(
                pr.get("platform_video_id")
                or pr.get("video_id")
                or pr.get("videoId")
                or pr.get("media_id")
                or ""
            ).strip()
            publish_id = str(pr.get("publish_id") or pr.get("publishId") or "").strip()
            if cur_vid and cur_vid != publish_id and plat:
                by_id_rows = await conn.fetch(
                    """
                    SELECT platform, account_id, platform_video_id, views, likes,
                           comments, shares, platform_url
                      FROM platform_content_items
                     WHERE user_id = $1::uuid
                       AND platform = $2
                       AND platform_video_id = $3
                     ORDER BY updated_at DESC NULLS LAST
                     LIMIT 1
                    """,
                    user_id,
                    plat,
                    cur_vid,
                )
                if by_id_rows:
                    chosen = by_id_rows[0]
        if chosen is None:
            continue
        _apply_pci_row_to_pr(pr, chosen)


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


async def _warm_user_platform_oauth_tokens(user_id: str) -> None:
    """Refresh each connected platform token once per batch (cached in platform_oauth_refresh)."""
    uid = str(user_id)
    async with core.state.db_pool.acquire() as conn:
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            uid,
        )
    for tr in token_rows:
        try:
            dec = decrypt_blob(tr["token_blob"])
            if not dec:
                continue
            if tr["platform"] == "instagram" and not dec.get("ig_user_id") and tr["account_id"]:
                dec["ig_user_id"] = str(tr["account_id"])
            if tr["platform"] == "facebook" and not dec.get("page_id") and tr["account_id"]:
                dec["page_id"] = str(tr["account_id"])
            await refresh_decrypted_token_for_row(
                tr["platform"],
                dec,
                db_pool=core.state.db_pool,
                user_id=uid,
                token_row_id=str(tr["id"]),
            )
        except Exception:
            pass


async def _sync_upload_analytics_core(
    user: dict,
    upload_id: str,
    *,
    skip_token_refresh: bool = False,
) -> dict:
    """Shared implementation for per-upload analytics sync."""
    uid = resolve_billing_user_id(user)
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platforms, platform_results, status, title, ai_title FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            uid,
        )
    if not row:
        raise HTTPException(404, "Upload not found")

    if row["status"] not in ("completed", "succeeded", "partial"):
        return {"synced": False, "reason": "not_completed", "views": 0, "likes": 0, "comments": 0, "shares": 0}

    raw_pr = _safe_json(row["platform_results"], [])
    pr_list = []
    if isinstance(raw_pr, list):
        pr_list = [x for x in raw_pr if isinstance(x, dict)]
    elif isinstance(raw_pr, dict):
        pr_list = [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw_pr.items()]
    for pr in pr_list:
        if pr.get("platform_video_id") and not pr.get("video_id"):
            pr["video_id"] = pr["platform_video_id"]
        if pr.get("platform_url") and not pr.get("url"):
            pr["url"] = pr["platform_url"]
        hint = str(row["title"] or row["ai_title"] or "").strip()
        if hint:
            pr["_caption_hint"] = hint

    async with core.state.db_pool.acquire() as conn:
        # Catalog may already have real TikTok ids while PR still holds publish_id.
        needs_tt_backfill = _tiktok_needs_video_id_backfill(pr_list)
        if needs_tt_backfill:
            await _backfill_tiktok_ids_before_metrics(
                conn, user_id=uid, pr_list=pr_list
            )
            reloaded = await _reload_platform_results_list(
                conn, upload_id=str(upload_id), user_id=uid
            )
            if reloaded:
                pr_list = reloaded
        await _hydrate_platform_results_from_pci(
            conn, user_id=uid, upload_id=str(upload_id), pr_list=pr_list
        )
        token_rows = await conn.fetch(
            "SELECT id, platform, token_blob, account_id FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
            uid,
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
                if not skip_token_refresh:
                    dec = await refresh_decrypted_token_for_row(
                        tr["platform"],
                        dec,
                        db_pool=core.state.db_pool,
                        user_id=uid,
                        token_row_id=token_id,
                    )
                token_map_by_id[token_id] = dec
                plat_norm = str(tr.get("platform") or "").lower()
                if plat_norm:
                    token_map_by_platform[plat_norm] = dec
        except Exception:
            pass

    token_map_by_plat_account, plat_account_row_map, platform_token_rows = _plat_token_resolution_maps(
        list(token_rows), token_map_by_id, token_map_by_platform
    )

    total_views = total_likes = total_comments = total_shares = 0
    platform_stats: Dict[str, Dict[str, int]] = {}
    rows_with_video_id = 0
    fetched_any = False
    had_token_candidates = False
    tried_platform_fetch = False

    async with httpx.AsyncClient(timeout=20) as client:
        for pr in pr_list:
            plat = str(pr.get("platform") or "").lower()
            video_id = _pr_queryable_video_id(pr)
            if not video_id:
                continue
            rows_with_video_id += 1

            candidates = resolve_token_candidates_for_platform_result(
                pr,
                token_map_by_id,
                token_map_by_plat_account,
                token_map_by_platform,
                plat_account_row_map=plat_account_row_map,
                platform_token_rows=platform_token_rows,
            )
            if not candidates:
                continue
            had_token_candidates = True

            s: Optional[Dict[str, int]] = None
            for tok in candidates:
                at = (tok or {}).get("access_token", "")
                if not at:
                    continue
                tried_platform_fetch = True
                if plat == "instagram":
                    ig_uid = str(
                        (tok or {}).get("ig_user_id")
                        or (tok or {}).get("user_id")
                        or pr.get("account_id")
                        or ""
                    ).strip()
                    if ig_uid:
                        pr["_ig_user_id"] = ig_uid
                s = await _fetch_platform_video_engagement(client, plat, str(video_id), pr, at)
                if s is not None:
                    break
            pr.pop("_ig_user_id", None)

            if not s:
                pci = pr.pop("_pci_metrics", None)
                if isinstance(pci, dict) and (
                    int(pci.get("views") or 0)
                    + int(pci.get("likes") or 0)
                    + int(pci.get("comments") or 0)
                    + int(pci.get("shares") or 0)
                ) > 0:
                    s = {
                        "views": int(pci.get("views") or 0),
                        "likes": int(pci.get("likes") or 0),
                        "comments": int(pci.get("comments") or 0),
                        "shares": int(pci.get("shares") or 0),
                    }
                else:
                    pr.pop("_pci_metrics", None)
                    continue
            else:
                pr.pop("_pci_metrics", None)

            _merge_stats_into_platform_result(pr, s)
            fetched_any = True
            total_views += s["views"]
            total_likes += s["likes"]
            total_comments += s["comments"]
            total_shares += s["shares"]
            prev = platform_stats.get(plat)
            if prev:
                platform_stats[plat] = {
                    "views": prev["views"] + s["views"],
                    "likes": prev["likes"] + s["likes"],
                    "comments": prev["comments"] + s["comments"],
                    "shares": prev["shares"] + s["shares"],
                }
            else:
                platform_stats[plat] = dict(s)

    async with core.state.db_pool.acquire() as conn:
        if pr_list:
            pr_json = json.dumps(pr_list)
            await conn.execute(
                """UPDATE uploads SET views=GREATEST(COALESCE(views,0),$1),
                       likes=GREATEST(COALESCE(likes,0),$2),
                       comments=GREATEST(COALESCE(comments,0),$3),
                       shares=GREATEST(COALESCE(shares,0),$4),
                       platform_results = $7::jsonb,
                       analytics_synced_at=NOW(), updated_at=NOW()
                   WHERE id=$5 AND user_id=$6""",
                total_views,
                total_likes,
                total_comments,
                total_shares,
                upload_id,
                uid,
                pr_json,
            )
        else:
            await conn.execute(
                """UPDATE uploads SET views=GREATEST(COALESCE(views,0),$1),
                       likes=GREATEST(COALESCE(likes,0),$2),
                       comments=GREATEST(COALESCE(comments,0),$3),
                       shares=GREATEST(COALESCE(shares,0),$4),
                       analytics_synced_at=NOW(), updated_at=NOW()
                   WHERE id=$5 AND user_id=$6""",
                total_views,
                total_likes,
                total_comments,
                total_shares,
                upload_id,
                uid,
            )

        # Mirror per-platform stats into platform_content_items so catalog /
        # Top Performing (catalog mode) / aggregate cards leave zero when PR has data.
        await _upsert_pci_metrics_from_platform_results(
            conn,
            user_id=uid,
            upload_id=str(upload_id),
            pr_list=pr_list,
            published_at=None,
        )

    if not rows_with_video_id:
        return {
            "synced": False,
            "reason": "no_platform_video_ids",
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "shares": total_shares,
            "platform_stats": platform_stats,
        }
    if not fetched_any:
        fail_reasons = {
            str(pr.get("_sync_fail_reason") or "")
            for pr in pr_list
            if isinstance(pr, dict) and pr.get("_sync_fail_reason")
        }
        for pr in pr_list:
            if isinstance(pr, dict):
                pr.pop("_sync_fail_reason", None)
        if not had_token_candidates:
            reason = "no_matching_token"
            message = (
                "No connected OAuth token matched this upload. "
                "Reconnect the account on Platforms, then Sync again."
            )
        elif "rate_limited" in fail_reasons:
            reason = "platform_rate_limited"
            message = (
                "Meta Graph rate limit hit while fetching Instagram metrics. "
                "Wait a few minutes, then Sync again (avoid hammering Sync catalog)."
            )
        elif "instagram_media_missing" in fail_reasons:
            reason = "instagram_media_missing"
            message = (
                "Instagram Graph does not list this reel on the connected account’s /media "
                "API (saved id looks like a container id; shortcode/ig_id not in the media "
                "list). If it’s still visible in the Instagram app, use Share → Copy link and "
                "confirm the shortcode, or paste the Insights media id from Meta Business "
                "Suite so we can repair the stored ID."
            )
        elif tried_platform_fetch:
            reason = "platform_api_no_data"
            message = (
                "Connected tokens were tried but platforms returned no metrics yet. "
                "Run Sync catalog first (to refresh video IDs), wait for it to finish, then Sync again."
            )
        else:
            reason = "no_tokens_or_metrics"
            message = "No working OAuth token matched this upload, or platforms returned no data."
        return {
            "synced": False,
            "reason": reason,
            "message": message,
            "views": total_views,
            "likes": total_likes,
            "comments": total_comments,
            "shares": total_shares,
            "platform_stats": platform_stats,
        }

    return {
        "synced": True,
        "views": total_views,
        "likes": total_likes,
        "comments": total_comments,
        "shares": total_shares,
        "platform_stats": platform_stats,
    }


async def _background_sync_uploads_analytics(user_id: str, upload_ids: list[str]) -> None:
    uid = str(user_id)
    incoming = [str(x) for x in (upload_ids or []) if x]
    if uid in _sync_analytics_running:
        pending = _sync_analytics_pending.setdefault(uid, [])
        for i in incoming:
            if i not in pending:
                pending.append(i)
        logger.info(
            "sync-analytics/all: queued %s ids for user %s (batch already running)",
            len(incoming),
            uid[:8],
        )
        return
    _sync_analytics_running.add(uid)
    try:
        try:
            await _warm_user_platform_oauth_tokens(uid)
        except Exception as e:
            logger.warning("sync-analytics/all token warm user=%s: %s", uid[:8], e)
        user_stub = {"id": uid}
        queue = list(incoming)
        while queue:
            for up_id in queue:
                try:
                    await _sync_upload_analytics_core(user_stub, up_id, skip_token_refresh=True)
                except HTTPException:
                    pass
                except Exception as e:
                    logger.warning("sync-analytics/all upload=%s: %s", up_id, e)
                await asyncio.sleep(0.35)
            queue = _sync_analytics_pending.pop(uid, [])
        try:
            n = await mirror_pci_metrics_into_uploads(core.state.db_pool, user_id=uid)
            if n:
                logger.info("sync-analytics/all: mirrored PCI→uploads n=%s user=%s", n, uid[:8])
        except Exception as e:
            logger.warning("sync-analytics/all PCI mirror user=%s: %s", uid[:8], e)
        # Refresh Smart Insights packaging rollups from the engagement we just synced.
        try:
            from services.ml_scoring_job import maybe_recompute_quality_after_analytics_sync

            await maybe_recompute_quality_after_analytics_sync(core.state.db_pool, uid)
        except Exception as e:
            logger.warning("sync-analytics/all quality recompute user=%s: %s", uid[:8], e)
    finally:
        _sync_analytics_running.discard(uid)


async def _background_sync_uploads_thumbnails(user_id: str, upload_ids: list[str]) -> None:
    await background_sync_posted_thumbnails(core.state.db_pool, user_id, upload_ids)


