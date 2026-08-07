"""Upload-row engagement from ``platform_results`` + column fallbacks (user KPI, dashboard, digest).

Also powers coach / ML quality scoring so TikTok, YouTube, and Meta (Instagram/Facebook)
engagement on ``platform_results`` feeds “What’s working for you” tips — not only stale
``uploads.views/likes/comments/shares`` columns.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.json_utils import safe_json
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN

# Platforms whose engagement must flow into coach / quality / packaging tips.
COACH_ENGAGEMENT_PLATFORMS = frozenset({"tiktok", "youtube", "instagram", "facebook"})


def engagement_rate_pct(views: int, likes: int, comments: int, shares: int) -> float:
    v = max(0, int(views or 0))
    if v <= 0:
        return 0.0
    interactions = max(0, int(likes or 0)) + max(0, int(comments or 0)) + max(0, int(shares or 0))
    return (interactions / float(v)) * 100.0


def _column_metrics(row: Any) -> Dict[str, int]:
    if row is None:
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    return {
        "views": max(0, int(get("views") or 0)),
        "likes": max(0, int(get("likes") or 0)),
        "comments": max(0, int(get("comments") or 0)),
        "shares": max(0, int(get("shares") or 0)),
    }


def _max_metrics(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {
        "views": max(int(a.get("views") or 0), int(b.get("views") or 0)),
        "likes": max(int(a.get("likes") or 0), int(b.get("likes") or 0)),
        "comments": max(int(a.get("comments") or 0), int(b.get("comments") or 0)),
        "shares": max(int(a.get("shares") or 0), int(b.get("shares") or 0)),
    }


def rollup_engagement_from_platform_results(
    entries: list,
    *,
    shortform_only: bool = False,
    successful_only: bool = True,
    platforms: Optional[frozenset] = None,
) -> dict[str, int]:
    """Sum per-platform metrics stored on platform_results when uploads.views/likes are stale."""
    from services.content_success_features import entry_metrics, entry_successful

    tv = tl = tc = ts = 0
    if not entries:
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}

    allow = platforms or (COACH_ENGAGEMENT_PLATFORMS if shortform_only else None)

    for e in entries:
        if not isinstance(e, dict):
            continue
        plat = str(e.get("platform") or "").strip().lower()
        if allow is not None and plat and plat not in allow:
            continue
        if successful_only and not entry_successful(e):
            continue
        m = entry_metrics(e, plat or "unknown")
        tv += int(m["views"])
        tl += int(m["likes"])
        tc += int(m["comments"])
        ts += int(m["shares"])
    return {"views": tv, "likes": tl, "comments": tc, "shares": ts}


def effective_upload_metrics(row: Any, *, shortform_only: bool = True) -> Dict[str, int]:
    """
    Element-wise max of upload columns and successful platform_results rollup.

    Prefer PR metrics (TikTok / YouTube / Instagram / Facebook, including Meta
    reactions / impressions aliases) when columns are stale or zero.
    """
    cols = _column_metrics(row)
    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    pr = normalize_upload_platform_results_list(get("platform_results"))
    roll = rollup_engagement_from_platform_results(
        pr,
        shortform_only=shortform_only,
        successful_only=True,
        platforms=COACH_ENGAGEMENT_PLATFORMS if shortform_only else None,
    )
    return _max_metrics(cols, roll)


def per_platform_upload_metrics(row: Any) -> List[Dict[str, Any]]:
    """
    One metric blob per TikTok / YouTube / Instagram / Facebook target.

    Uses true per-platform ``platform_results`` stats when present; otherwise falls
    back to upload columns for single-platform posts (or shared columns when
    multi-platform but PR has no metrics yet).
    """
    from services.content_success_features import entry_metrics, entry_successful

    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    cols = _column_metrics(row)
    pr = normalize_upload_platform_results_list(get("platform_results"))
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for e in pr:
        if not isinstance(e, dict) or not entry_successful(e):
            continue
        plat = str(e.get("platform") or "").strip().lower()
        if not plat or plat not in COACH_ENGAGEMENT_PLATFORMS:
            continue
        m = entry_metrics(e, plat)
        if (m["views"] + m["likes"] + m["comments"] + m["shares"]) <= 0:
            continue
        seen.add(plat)
        out.append(
            {
                "platform": plat,
                **m,
                "engagement_rate_pct": engagement_rate_pct(
                    m["views"], m["likes"], m["comments"], m["shares"]
                ),
            }
        )

    platforms = [
        str(p).strip().lower()
        for p in (get("platforms") or [])
        if str(p).strip()
    ]
    platforms = [p for p in platforms if p in COACH_ENGAGEMENT_PLATFORMS]

    if not out:
        # No usable PR metrics — attribute columns to each declared platform
        # (same prior behavior) so single-platform posts still score.
        for plat in platforms or ["all"]:
            if (cols["views"] + cols["likes"] + cols["comments"] + cols["shares"]) <= 0:
                continue
            out.append(
                {
                    "platform": plat if plat != "all" else "all",
                    **cols,
                    "engagement_rate_pct": engagement_rate_pct(
                        cols["views"], cols["likes"], cols["comments"], cols["shares"]
                    ),
                }
            )
        return out

    # Some platforms already have real PR metrics. Do not assign rolled-up
    # upload-column totals to the remaining platforms (double-counts multi-post).
    return out


def strategy_key_from_artifacts(output_artifacts: Any) -> str:
    """Match ml_scoring_job SQL attribution key for quality daily rows."""
    oa = safe_json(output_artifacts, {})
    if not isinstance(oa, dict):
        oa = {}
    key = str(oa.get("content_attribution_key") or "").strip()
    if key:
        return key
    tsel = str(oa.get("thumbnail_selection_method") or "").strip() or "na"
    trend = str(oa.get("thumbnail_render_method") or "").strip() or "na"
    return f"legacy|tsel={tsel}|trend={trend}"


def grounding_score_from_artifacts(output_artifacts: Any) -> Optional[float]:
    oa = safe_json(output_artifacts, {})
    if not isinstance(oa, dict):
        return None
    hr = oa.get("hydration_report") if isinstance(oa.get("hydration_report"), dict) else {}
    gs = hr.get("grounding_score")
    if gs is None:
        gsv = oa.get("grounding_score_v1") if isinstance(oa.get("grounding_score_v1"), dict) else {}
        gs = gsv.get("grounding_score")
    try:
        if gs is None or str(gs).strip() == "":
            return None
        return float(gs)
    except (TypeError, ValueError):
        return None


def normalize_upload_platform_results_list(raw: Any) -> list:
    pr = safe_json(raw, [])
    # Legacy rows may be double-encoded jsonb strings ('"[{...}]"').
    for _ in range(3):
        if isinstance(pr, str):
            pr = safe_json(pr, [])
        else:
            break
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
