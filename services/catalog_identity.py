"""
Stable identity helpers for platform catalog rows and engagement deduplication.

``platform_content_items`` is unique on (user_id, platform, account_id, platform_video_id).
Rollups must use the same triple (within a user) so two connected accounts never
merge metrics under a bare platform_video_id key.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def normalize_platform_account_id(platform: str, raw: Any) -> str:
    """Normalize platform account id for dedupe keys; empty string if unknown."""
    _ = (platform or "").strip().lower()
    if raw is None:
        return ""
    s = str(raw).strip()
    return s


def token_row_to_map_entry(row: Any) -> Dict[str, Any]:
    """Normalize a platform_tokens row for account resolution (matches enrich helpers)."""
    if row is None:
        return {}
    rid = row.get("id")
    return {
        "token_row_id": str(rid) if rid is not None else "",
        "platform": str(row.get("platform") or ""),
        "account_id": str(row.get("account_id") or ""),
    }


def parse_facebook_dual_cursor(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Facebook sync stores pagination for both /videos and /video_reels in ``next_cursor``.
    Legacy value: plain string = videos cursor only.
    """
    if not raw:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    if s.startswith("{"):
        import json

        try:
            d = json.loads(s)
            return d.get("v") or None, d.get("r") or None
        except json.JSONDecodeError:
            return s, None
    return s, None


def dump_facebook_dual_cursor(videos_after: Optional[str], reels_after: Optional[str]) -> Optional[str]:
    import json

    if not videos_after and not reels_after:
        return None
    return json.dumps({"v": videos_after, "r": reels_after})
