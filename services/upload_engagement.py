"""Upload-row engagement from ``platform_results`` + column fallbacks (user KPI, dashboard, digest)."""
from __future__ import annotations

from typing import Any, Optional

from core.json_utils import safe_json
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN


def rollup_engagement_from_platform_results(
    entries: list,
    *,
    shortform_only: bool = False,
    successful_only: bool = True,
) -> dict[str, int]:
    """Sum per-platform metrics stored on platform_results when uploads.views/likes are stale."""
    tv = tl = tc = ts = 0
    if not entries:
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}

    def _pick_int(d: dict, *keys: str) -> int:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return int(d[k] or 0)
                except (TypeError, ValueError):
                    return 0
        return 0

    shortform_platforms = {"tiktok", "youtube", "instagram", "facebook"}
    successful_statuses = {"published", "succeeded", "success", "completed", "partial"}

    for e in entries:
        if not isinstance(e, dict):
            continue
        plat = str(e.get("platform") or "").strip().lower()
        if shortform_only and plat and plat not in shortform_platforms:
            continue
        if successful_only:
            ok = bool(e.get("success") is True)
            st = str(e.get("status") or "").strip().lower()
            if (not ok) and (st not in successful_statuses):
                continue
        tv += _pick_int(e, "views", "view_count", "play_count", "playCount", "video_views")
        tl += _pick_int(e, "likes", "like_count", "likeCount")
        tc += _pick_int(e, "comments", "comment_count", "commentCount")
        ts += _pick_int(e, "shares", "share_count", "shareCount")
    return {"views": tv, "likes": tl, "comments": tc, "shares": ts}


def normalize_upload_platform_results_list(raw: Any) -> list:
    pr = safe_json(raw, [])
    if isinstance(pr, dict):
        return [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in pr.items()]
    if isinstance(pr, list):
        return pr
    return []


def title_and_metrics_from_upload_platform_results(
    raw: Any,
    platform: Optional[str],
    platform_video_id: Optional[str],
) -> tuple[Optional[str], dict[str, int]]:
    """
    Pick title + engagement for one catalog row from uploads.platform_results JSON,
    matching platform (and platform_video_id when provided). Used by GET /api/catalog/content.
    """
    entries = normalize_upload_platform_results_list(raw)
    plat = (platform or "").strip().lower()
    vid = str(platform_video_id or "").strip()
    candidates: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        ep = str(e.get("platform") or "").strip().lower()
        if plat and ep != plat:
            continue
        candidates.append(e)
    if not candidates:
        return None, {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    picked: Optional[dict] = None
    if vid:
        for e in candidates:
            ev = str(
                e.get("platform_video_id")
                or e.get("video_id")
                or e.get("media_id")
                or e.get("post_id")
                or ""
            ).strip()
            if ev == vid:
                picked = e
                break
    if picked is None:
        picked = candidates[0]
    title_out: Optional[str] = None
    for key in ("title", "name", "video_title", "caption"):
        t = picked.get(key)
        if isinstance(t, str) and t.strip():
            title_out = t.strip()
            break
    roll = rollup_engagement_from_platform_results(
        [picked], shortform_only=False, successful_only=False
    )
    return title_out, roll


async def compute_upload_engagement_totals(
    conn: Any,
    user_id: str,
    *,
    since: Optional[Any] = None,
    until: Optional[Any] = None,
    platform: Optional[str] = None,
) -> dict[str, int]:
    """
    User-scoped engagement from upload rows, using per-upload rollups when DB columns are stale.
    ``platform`` filters to uploads whose platforms[] contains that slug (case-insensitive).
    """
    where = f"WHERE user_id = $1 AND status IN {SUCCESSFUL_STATUS_SQL_IN}"
    params: list[Any] = [user_id]
    if since is not None:
        where += f" AND created_at >= ${len(params) + 1}"
        params.append(since)
    if until is not None:
        where += f" AND created_at < ${len(params) + 1}"
        params.append(until)
    if platform:
        where += (
            f" AND EXISTS (SELECT 1 FROM unnest(COALESCE(platforms, ARRAY[]::text[])) AS _plat "
            f"WHERE lower(_plat::text) = ${len(params) + 1})"
        )
        params.append(platform)
    rows = await conn.fetch(
        f"""
        SELECT views, likes, comments, shares, platform_results
          FROM uploads
          {where}
        """,
        *params,
    )
    totals = {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    for r in rows:
        pr = normalize_upload_platform_results_list(r.get("platform_results"))
        roll = rollup_engagement_from_platform_results(
            pr,
            shortform_only=True,
            successful_only=True,
        )
        totals["views"] += max(int(r.get("views") or 0), int(roll["views"] or 0))
        totals["likes"] += max(int(r.get("likes") or 0), int(roll["likes"] or 0))
        totals["comments"] += max(int(r.get("comments") or 0), int(roll["comments"] or 0))
        totals["shares"] += max(int(r.get("shares") or 0), int(roll["shares"] or 0))
    return totals
