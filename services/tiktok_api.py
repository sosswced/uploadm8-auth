"""
TikTok Open API v2 — URL and response envelope rules (single source for app, worker, jobs).

Docs: video list / video query use ?fields= in the query string, not the JSON body.
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

TIKTOK_VIDEO_LIST_FIELDS = (
    "id,title,view_count,like_count,comment_count,share_count,duration,create_time"
)
TIKTOK_VIDEO_QUERY_FIELDS = "id,view_count,like_count,comment_count,share_count"


def tiktok_video_list_url() -> str:
    return (
        "https://open.tiktokapis.com/v2/video/list/"
        f"?fields={quote(TIKTOK_VIDEO_LIST_FIELDS)}"
    )


def tiktok_video_query_url() -> str:
    return (
        "https://open.tiktokapis.com/v2/video/query/"
        f"?fields={quote(TIKTOK_VIDEO_QUERY_FIELDS)}"
    )


def tiktok_envelope_error(body: Any) -> Optional[str]:
    """TikTok often returns HTTP 200 with error.code != 'ok' in JSON."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if code is None or str(code).lower() == "ok":
        return None
    return (err.get("message") or str(code) or "tiktok_api_error")[:500]
