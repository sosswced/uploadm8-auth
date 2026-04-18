"""Shared platform-metrics helpers for analytics, dashboard, and catalog date windows."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def parse_iso_ts(ts: Any) -> Optional[datetime]:
    try:
        if not ts:
            return None
        s = str(ts).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def aggregate_platform_metrics_live(platforms_result: dict) -> dict:
    """
    Sum engagement from platforms that returned status=='live' (TikTok/YouTube/etc.).
    Meta platforms that only return errors do not contribute — avoids zeroing the sum.
    """
    agg = {"views": 0, "likes": 0, "comments": 0, "shares": 0, "platforms_included": []}

    def _n(x) -> int:
        try:
            return int(x or 0)
        except Exception:
            return 0

    if not isinstance(platforms_result, dict):
        return agg

    for plat, d in platforms_result.items():
        if not isinstance(d, dict) or d.get("status") != "live":
            continue
        if plat == "youtube":
            views = _n(d.get("shorts_views")) or _n(d.get("views"))
            likes = _n(d.get("shorts_likes")) or _n(d.get("likes"))
            comments = _n(d.get("shorts_comments")) or _n(d.get("comments"))
        else:
            views = _n(d.get("views"))
            likes = _n(d.get("reactions")) if plat == "facebook" else _n(d.get("likes"))
            comments = _n(d.get("comments"))
        shares = _n(d.get("shares"))
        if views or likes or comments or shares:
            agg["views"] += views
            agg["likes"] += likes
            agg["comments"] += comments
            agg["shares"] += shares
            agg["platforms_included"].append(plat)
    return agg
