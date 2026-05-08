"""
Canonical user engagement rollups for product KPIs.

Headline views/likes/comments/shares use a **deduplicated** merge:
  • Rows from ``platform_content_items`` (DB, synced catalog) keyed by
    (platform, account_id, platform_video_id) — aligned with the PCI unique constraint.
  • Successful ``platform_results`` entries on successful uploads keyed the same way
    after resolving ``account_id`` (entry fields, ``token_row_id``, ``target_accounts``,
    or primary account per platform — same spirit as ``_enrich_platform_results``).
  • Per key: take element-wise max(metrics) so DB catalog and upload JSON stay in sync
    without double-counting the same video.
  • ``platform_results`` entries without a resolvable video id contribute as **orphans**
    (cannot dedupe against catalog).
  • Successful uploads with **no** successful platform result row use one
    **upload-level** max(row columns, rolled-up pr) so pipeline jobs still count.
  • If every successful ``platform_results`` row lacks a video id, the same
    **upload-level** max(row, roll) is used once (not a sum of empty per-entry metrics),
    so ``uploads.views`` / worker-synced columns are not dropped.

``live_aggregate`` / ``platform_metrics_cache`` (worker + OAuth account polls) is
**not** mixed into these totals — it is an account-level snapshot for pills/badges;
see ``kpi_sources`` on API payloads.

Time windows: optional half-open UTC ``[window_start, window_end_exclusive)`` on
``platform_content_items.published_at`` and ``uploads.created_at``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from services.catalog_identity import normalize_platform_account_id, token_row_to_map_entry
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger("uploadm8.canonical_engagement")

# Bump when merge semantics or API shape changes (clients / exports can branch on this).
ROLLUP_VERSION = 2
SLOW_ROLLUP_WARN_MS = 2000.0

_ENG_KEYS = ("views", "likes", "comments", "shares")
# Defensive cap for JSON / UI (Postgres BIGINT can exceed JS safe int).
_MAX_METRIC = (2**63) - 1


def parse_tenant_user_id(user_id: str) -> str:
    """Validate tenant scope id (UUID). Raises ValueError if not a valid UUID string."""
    return str(uuid.UUID(str(user_id).strip()))


def _clamp_nonneg(n: int) -> int:
    try:
        x = int(n)
    except (TypeError, ValueError):
        return 0
    if x < 0:
        return 0
    if x > _MAX_METRIC:
        return _MAX_METRIC
    return x

# Must match GET /api/analytics ``minutes_map`` in ``routers/analytics``.
ANALYTICS_RANGE_MINUTES: Dict[str, int] = {
    "30m": 30,
    "1h": 60,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "7d": 10080,
    "30d": 43200,
    "90d": 129600,
    "6m": 262800,
    "1y": 525600,
}


def engagement_time_window_for_analytics_range(
    range_key: str, *, now: datetime
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Half-open UTC window [start, end_exclusive) for analytics range presets.
    ``end_exclusive`` is ``now`` (UTC). Unbounded ``all`` → (None, None).
    """
    rk = (range_key or "30d").strip().lower()
    if rk == "all":
        return None, None
    minutes = int(ANALYTICS_RANGE_MINUTES.get(rk, 43200))
    end = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if end.tzinfo != timezone.utc:
        end = end.astimezone(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return start, end


def engagement_time_window_for_overview_days(days: int, *, now: datetime) -> Tuple[datetime, datetime]:
    """Half-open UTC window [now - days, now) for analytics overview."""
    d = max(1, min(int(days or 30), 3650))
    end = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if end.tzinfo != timezone.utc:
        end = end.astimezone(timezone.utc)
    start = end - timedelta(days=d)
    return start, end


def engagement_window_api_dict(
    *,
    start: Optional[datetime],
    end_exclusive: Optional[datetime],
) -> Dict[str, Any]:
    """Serializable window metadata for API responses."""
    return {
        "start": start.isoformat() if start else None,
        "end_exclusive": end_exclusive.isoformat() if end_exclusive else None,
        "uploads_filter_column": "created_at",
        "catalog_filter_column": "published_at",
        "interval_semantics": "half_open_utc",
    }


def _zero_vec() -> Dict[str, int]:
    return {k: 0 for k in _ENG_KEYS}


def _pick_int(d: dict, *keys: str) -> int:
    """First parseable non-null value wins; try every key (do not stop on one bad cast)."""
    for k in keys:
        if k not in d or d[k] is None:
            continue
        try:
            v = d[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, float):
                return _clamp_nonneg(int(round(v)))
            return _clamp_nonneg(int(v))
        except (TypeError, ValueError):
            continue
    return 0


def _vec_max(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: _clamp_nonneg(max(int(a.get(k) or 0), int(b.get(k) or 0))) for k in _ENG_KEYS}


def _vec_add(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: _clamp_nonneg(int(a.get(k) or 0) + int(b.get(k) or 0)) for k in _ENG_KEYS}


def _normalize_platform_results_list(raw: Any) -> List[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        return [{"platform": k, **v} if isinstance(v, dict) else {"platform": k} for k, v in raw.items()]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _pr_entry_successful(e: dict) -> bool:
    ok = bool(e.get("success") is True)
    st = str(e.get("status") or "").strip().lower()
    good = {"published", "succeeded", "success", "completed", "partial"}
    return ok or st in good


def _metrics_from_pr_entry(e: dict) -> Dict[str, int]:
    """Per-platform-result metrics (all platforms — matches catalog breadth)."""
    plat = str(e.get("platform") or "").strip().lower()
    likes_keys = ("likes", "like_count", "likeCount", "reactions", "reaction_count")
    if plat == "facebook":
        likes_keys = ("reactions", "reaction_count", "likes", "like_count", "likeCount")
    return {
        "views": _clamp_nonneg(_pick_int(e, "views", "view_count", "play_count", "playCount", "video_views", "impressions")),
        "likes": _clamp_nonneg(_pick_int(e, *likes_keys)),
        "comments": _clamp_nonneg(_pick_int(e, "comments", "comment_count", "commentCount")),
        "shares": _clamp_nonneg(_pick_int(e, "shares", "share_count", "shareCount")),
    }


def _video_key_from_pr_entry(e: dict) -> Optional[Tuple[str, str]]:
    plat = str(e.get("platform") or "").strip().lower()
    if not plat:
        return None
    vid = (
        e.get("platform_video_id")
        or e.get("video_id")
        or e.get("videoId")
        or e.get("tiktok_video_id")
        or e.get("youtube_video_id")
        or e.get("publish_id")
        or e.get("publishId")
        or e.get("media_id")
        or e.get("post_id")
        or e.get("share_id")
        or e.get("id")
    )
    if vid is None:
        return None
    s = str(vid).strip()
    if not s:
        return None
    return plat, s


def _direct_account_hints_from_pr_entry(e: dict) -> Optional[str]:
    """Platform-native ids sometimes stored on the publish result."""
    for k in (
        "account_id",
        "page_id",
        "channel_id",
        "ig_user_id",
        "open_id",
        "facebook_page_id",
        "instagram_user_id",
    ):
        v = e.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _resolve_pr_account_id(
    e: dict,
    *,
    plat: str,
    token_map: Dict[str, Dict[str, Any]],
    tokens_by_id: Dict[str, Dict[str, Any]],
    primary_by_platform: Dict[str, Dict[str, Any]],
    used_token_ids: set,
) -> str:
    """
    Match ``_enrich_platform_results`` account selection: explicit ids, token_row_id,
    target_accounts multi-account rotation, then primary per platform.
    When ``target_accounts`` is empty, ``token_row_id`` resolves against all user tokens.
    """
    hint = _direct_account_hints_from_pr_entry(e)
    if hint:
        return normalize_platform_account_id(plat, hint)

    tid = str(e.get("token_row_id") or "").strip()
    p = plat.lower()

    if token_map:
        if tid and tid in token_map:
            return normalize_platform_account_id(plat, token_map[tid].get("account_id"))
        candidates = [
            v
            for v in token_map.values()
            if str(v.get("platform") or "").lower() == p and str(v.get("token_row_id") or "") not in used_token_ids
        ]
        acct = (
            candidates[0]
            if candidates
            else next((v for v in token_map.values() if str(v.get("platform") or "").lower() == p), None)
        )
        if acct:
            tr = str(acct.get("token_row_id") or "")
            if tr:
                used_token_ids.add(tr)
            return normalize_platform_account_id(plat, acct.get("account_id"))

    if tid and tid in tokens_by_id:
        return normalize_platform_account_id(plat, tokens_by_id[tid].get("account_id"))

    fb = primary_by_platform.get(p)
    if fb:
        return normalize_platform_account_id(plat, fb.get("account_id"))
    return ""


def _dedupe_key_from_pr_entry(
    e: dict,
    *,
    token_map: Dict[str, Dict[str, Any]],
    tokens_by_id: Dict[str, Dict[str, Any]],
    primary_by_platform: Dict[str, Dict[str, Any]],
    used_token_ids: set,
) -> Optional[Tuple[str, str, str]]:
    """(platform, normalized_account_id, platform_video_id) or None if no video id."""
    vk = _video_key_from_pr_entry(e)
    if vk is None:
        return None
    plat, vid = vk
    acct = _resolve_pr_account_id(
        e,
        plat=plat,
        token_map=token_map,
        tokens_by_id=tokens_by_id,
        primary_by_platform=primary_by_platform,
        used_token_ids=used_token_ids,
    )
    return plat, acct, vid


def _rollup_successful_pr_entries(entries: List[dict]) -> Dict[str, int]:
    """Sum metrics from successful entries (all platforms)."""
    t = _zero_vec()
    for e in entries:
        if not isinstance(e, dict):
            continue
        if not _pr_entry_successful(e):
            continue
        t = _vec_add(t, _metrics_from_pr_entry(e))
    return t


def _single_upload_row_engagement(row: dict) -> Dict[str, int]:
    """max(row columns, sum successful pr metrics) for one upload row."""
    pr = _normalize_platform_results_list(row.get("platform_results"))
    roll = _rollup_successful_pr_entries(pr)
    return {
        "views": _clamp_nonneg(max(int(row.get("views") or 0), roll["views"])),
        "likes": _clamp_nonneg(max(int(row.get("likes") or 0), roll["likes"])),
        "comments": _clamp_nonneg(max(int(row.get("comments") or 0), roll["comments"])),
        "shares": _clamp_nonneg(max(int(row.get("shares") or 0), roll["shares"])),
    }


async def compute_canonical_engagement_rollup(
    conn: "asyncpg.Connection",
    user_id: str,
    *,
    window_start: Optional[datetime] = None,
    window_end_exclusive: Optional[datetime] = None,
    platform: Optional[str] = None,
    catalog_content_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deduplicated headline engagement from ``platform_content_items`` + ``uploads``.

    Returns views/likes/comments/shares, breakdown, rollup_rule, catalog_tracked_videos,
    ``rollup_version``, ``compute`` (latency, health, scope), and ``kpi_sources``.
    """
    t0 = time.perf_counter()
    warnings: List[str] = []
    health = {"platform_tokens": True, "platform_content_items": True, "uploads": True}

    try:
        uid = parse_tenant_user_id(user_id)
    except ValueError as e:
        logger.warning("canonical rollup: invalid user_id: %s", e)
        raise

    combined: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    catalog_row_count = 0
    catalog_rows_with_valid_key = 0

    tokens_by_id: Dict[str, Dict[str, Any]] = {}
    primary_by_platform: Dict[str, Dict[str, Any]] = {}
    try:
        tok_rows = await conn.fetch(
            """
            SELECT id, platform, account_id, is_primary, updated_at
              FROM platform_tokens
             WHERE user_id = $1::uuid AND revoked_at IS NULL
            """,
            uid,
        )
        for tr in tok_rows:
            ent = token_row_to_map_entry(dict(tr))
            tid = ent.get("token_row_id") or ""
            if tid:
                tokens_by_id[tid] = ent
        rows_sorted = sorted(
            tok_rows,
            key=lambda r: (0 if r.get("is_primary") else 1, str(r.get("updated_at") or "")),
        )
        for tr in rows_sorted:
            p = str(tr.get("platform") or "").lower()
            if p not in primary_by_platform:
                primary_by_platform[p] = token_row_to_map_entry(dict(tr))
    except Exception as e:
        health["platform_tokens"] = False
        warnings.append("platform_tokens_fetch_failed")
        logger.warning("canonical rollup: platform_tokens fetch failed: %s", e, exc_info=True)

    c_where = ["user_id = $1"]
    c_params: List[Any] = [uid]
    if window_start is not None:
        c_where.append(f"published_at >= ${len(c_params) + 1}")
        c_params.append(window_start)
    if window_end_exclusive is not None:
        c_where.append(f"published_at < ${len(c_params) + 1}")
        c_params.append(window_end_exclusive)
    if window_start is not None or window_end_exclusive is not None:
        c_where.append("published_at IS NOT NULL")
    if platform:
        c_where.append(f"lower(platform) = ${len(c_params) + 1}")
        c_params.append(platform.lower().strip())
    if catalog_content_kind:
        ck = str(catalog_content_kind).strip().lower()
        if ck == "reel":
            c_where.append("(lower(COALESCE(content_kind, '')) IN ('reel', 'reels'))")
        else:
            c_where.append(f"lower(COALESCE(content_kind, '')) = ${len(c_params) + 1}")
            c_params.append(ck)
    c_sql = f"""
        SELECT platform, account_id, platform_video_id, views, likes, comments, shares
          FROM platform_content_items
         WHERE {" AND ".join(c_where)}
    """
    try:
        crows = await conn.fetch(c_sql, *c_params)
    except Exception as e:
        health["platform_content_items"] = False
        warnings.append("platform_content_items_fetch_failed")
        logger.warning("canonical rollup: catalog fetch failed: %s", e, exc_info=True)
        crows = []

    for r in crows:
        catalog_row_count += 1
        plat = str(r["platform"] or "").strip().lower()
        vid = str(r["platform_video_id"] or "").strip()
        if not plat or not vid:
            continue
        catalog_rows_with_valid_key += 1
        acct = normalize_platform_account_id(plat, r.get("account_id"))
        key = (plat, acct, vid)
        m = {
            "views": _clamp_nonneg(r["views"]),
            "likes": _clamp_nonneg(r["likes"]),
            "comments": _clamp_nonneg(r["comments"]),
            "shares": _clamp_nonneg(r["shares"]),
        }
        combined[key] = _vec_max(combined.get(key, _zero_vec()), m)

    u_where = ["user_id = $1", f"status IN {SUCCESSFUL_STATUS_SQL_IN}"]
    u_params: List[Any] = [uid]
    if window_start is not None:
        u_where.append(f"created_at >= ${len(u_params) + 1}")
        u_params.append(window_start)
    if window_end_exclusive is not None:
        u_where.append(f"created_at < ${len(u_params) + 1}")
        u_params.append(window_end_exclusive)
    if platform:
        u_where.append(
            f"EXISTS (SELECT 1 FROM unnest(COALESCE(platforms, ARRAY[]::text[])) AS _plat "
            f"WHERE lower(_plat::text) = ${len(u_params) + 1})"
        )
        u_params.append(platform.lower().strip())
    u_sql = f"""
        SELECT id, platform_results, views, likes, comments, shares, target_accounts
          FROM uploads
         WHERE {" AND ".join(u_where)}
    """
    try:
        uprows = await conn.fetch(u_sql, *u_params)
    except Exception as e:
        health["uploads"] = False
        warnings.append("uploads_fetch_failed")
        logger.warning("canonical rollup: uploads fetch failed: %s", e, exc_info=True)
        uprows = []

    orphan = _zero_vec()
    upload_pr_only_keys = 0
    jobs_fallback = 0
    pr_entries_missing_video_key = 0
    upload_jobs_all_pr_keyless_fallback = 0

    for row in uprows:
        pr = _normalize_platform_results_list(row.get("platform_results"))
        successful = [e for e in pr if isinstance(e, dict) and _pr_entry_successful(e)]
        if not successful:
            fb = _single_upload_row_engagement(dict(row))
            orphan = _vec_add(orphan, fb)
            jobs_fallback += 1
            continue

        target_ids = list(dict.fromkeys(str(t) for t in (row.get("target_accounts") or []) if t))
        token_map: Dict[str, Dict[str, Any]] = {}
        for tid in target_ids:
            if tid in tokens_by_id:
                token_map[tid] = tokens_by_id[tid]
        used_token_ids: set = set()

        had_key = False
        upload_resolved_keys: List[Tuple[str, str, str]] = []
        entry_orphan = _zero_vec()
        for e in successful:
            m = _metrics_from_pr_entry(e)
            dkey = _dedupe_key_from_pr_entry(
                e,
                token_map=token_map,
                tokens_by_id=tokens_by_id,
                primary_by_platform=primary_by_platform,
                used_token_ids=used_token_ids,
            )
            if dkey is None:
                entry_orphan = _vec_add(entry_orphan, m)
                pr_entries_missing_video_key += 1
                continue
            had_key = True
            upload_resolved_keys.append(dkey)
            if dkey in combined:
                combined[dkey] = _vec_max(combined[dkey], m)
            else:
                combined[dkey] = {
                    "views": _clamp_nonneg(m["views"]),
                    "likes": _clamp_nonneg(m["likes"]),
                    "comments": _clamp_nonneg(m["comments"]),
                    "shares": _clamp_nonneg(m["shares"]),
                }
                upload_pr_only_keys += 1

        if had_key:
            orphan = _vec_add(orphan, entry_orphan)
            # For single-platform uploads the upload row columns (updated by sync-analytics)
            # hold the authoritative per-video totals.  Apply them as a non-destructive max
            # so that views/likes/comments/shares synced after publish are reflected even
            # when platform_content_items hasn't been catalogued yet.
            if len(upload_resolved_keys) == 1:
                row_v: Dict[str, int] = {
                    "views":    _clamp_nonneg(int(row.get("views")    or 0)),
                    "likes":    _clamp_nonneg(int(row.get("likes")    or 0)),
                    "comments": _clamp_nonneg(int(row.get("comments") or 0)),
                    "shares":   _clamp_nonneg(int(row.get("shares")   or 0)),
                }
                k = upload_resolved_keys[0]
                combined[k] = _vec_max(combined[k], row_v)
        else:
            # Successful pr rows but none had a video id — sum(pr) may be all zeros while
            # uploads.views/likes columns hold truth; use one upload-level max(row, roll).
            upload_jobs_all_pr_keyless_fallback += 1
            orphan = _vec_add(orphan, _single_upload_row_engagement(dict(row)))

    total = _zero_vec()
    for _k, m in combined.items():
        total = _vec_add(total, m)
    total = _vec_add(total, orphan)

    unique_keys = len(combined)

    pv = _zero_vec()
    for _k, m in combined.items():
        pv = _vec_add(pv, m)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if latency_ms >= SLOW_ROLLUP_WARN_MS:
        logger.warning(
            "canonical rollup slow: user=%s latency_ms=%.1f catalog_rows=%s upload_rows=%s",
            uid[:8],
            latency_ms,
            len(crows),
            len(uprows),
        )

    compute_complete = health["platform_tokens"] and health["platform_content_items"] and health["uploads"]
    if not compute_complete:
        warnings.append("partial_data")

    breakdown = {
        "dedupe": {
            "unique_video_keys": unique_keys,
            "dedupe_key": "platform_account_id_platform_video_id",
            "catalog_rows_scanned": catalog_row_count,
            "catalog_rows_valid_video_key": catalog_rows_with_valid_key,
            "upload_pr_only_keys_added": max(0, upload_pr_only_keys),
            "upload_jobs_no_successful_pr_row_fallback": jobs_fallback,
            "pr_entries_missing_video_key": pr_entries_missing_video_key,
            "upload_jobs_all_pr_keyless_fallback": upload_jobs_all_pr_keyless_fallback,
            "single_platform_upload_row_boost_applied": True,
        },
        "vectors": {
            "per_video_deduped_total": dict(pv),
            "orphan_pr_metrics_plus_row_fallback": dict(orphan),
        },
        "compute": {
            "rollup_version": ROLLUP_VERSION,
            "latency_ms": round(latency_ms, 2),
            "health": dict(health),
            "warnings": warnings,
            "scope": {
                "connected_token_rows": len(tokens_by_id),
                "catalog_rows_fetched": len(crows),
                "upload_rows_fetched": len(uprows),
                "catalog_content_kind_filter": catalog_content_kind,
            },
            "complete": compute_complete,
        },
    }

    out: Dict[str, Any] = {
        "views": _clamp_nonneg(total["views"]),
        "likes": _clamp_nonneg(total["likes"]),
        "comments": _clamp_nonneg(total["comments"]),
        "shares": _clamp_nonneg(total["shares"]),
        "breakdown": breakdown,
        "catalog_tracked_videos": catalog_row_count,
        "rollup_version": ROLLUP_VERSION,
        "rollup_rule": "dedupe_max_per_platform_account_video_pci_plus_pr_orphans_fallback",
        "kpi_sources": {
            "rollup_version": ROLLUP_VERSION,
            "headline_engagement": "db_platform_content_items_plus_uploads_platform_results",
            "headline_excludes": "platform_metrics_cache_live_account_poll",
            "live_aggregate": "oauth_account_snapshot_worker_refresh_separate_pill",
            "workers": "catalog_sync_metrics_pci; per_upload_sync_updates_uploads_and_pci",
            "tenant_scope": "single_user_uuid",
            "time_semantics": "half_open_utc_uploads_created_at_pci_published_at",
            "enterprise_note": "Deterministic per-tenant merge; compare tenants using rates or normalized scores, not raw totals alone.",
        },
    }
    return out


def merge_upload_and_catalog_engagement(
    upload_engagement: Optional[Dict[str, Any]],
    catalog: Optional[Dict[str, Any]],
    *,
    engagement_window_utc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Legacy merge: per-metric max(upload totals, catalog aggregate).

    Prefer :func:`compute_canonical_engagement_rollup` for non-duplicate headline KPIs.
    """
    u_raw = upload_engagement or {}
    c_raw = catalog or {}
    u = {k: max(0, int(u_raw.get(k) or 0)) for k in _ENG_KEYS}
    c = {k: max(0, int(c_raw.get(k) or 0)) for k in _ENG_KEYS}
    merged = {k: max(u[k], c[k]) for k in _ENG_KEYS}
    out: Dict[str, Any] = {
        "views": merged["views"],
        "likes": merged["likes"],
        "comments": merged["comments"],
        "shares": merged["shares"],
        "breakdown": {"uploads": u, "catalog": c, "legacy_max_merge": True},
        "catalog_tracked_videos": int(c_raw.get("total_videos") or 0),
        "rollup_rule": "max_per_metric_uploads_vs_catalog_legacy",
        "kpi_sources": {
            "headline_engagement": "legacy_max_upload_sum_vs_catalog_sum",
            "note": "May double-count overlapping video semantics; use compute_canonical_engagement_rollup",
        },
    }
    if engagement_window_utc is not None:
        out["engagement_window_utc"] = engagement_window_utc
    return out