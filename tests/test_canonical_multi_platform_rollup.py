"""Regression: multi-platform uploads must contribute their combined synced totals.

Per-upload sync-analytics writes the SUM across every platform a video was posted to
into ``uploads.views/likes/comments/shares``.  The per-entry ``platform_results``
metrics are frequently stale/zero (captured at publish time).  The canonical rollup
must not drop the authoritative combined row totals for multi-platform uploads, or
headline KPIs under-report videos cross-posted to TikTok/YouTube/Instagram/Facebook.
"""

from __future__ import annotations

import asyncio
import json

from unittest.mock import AsyncMock, MagicMock

from services.canonical_engagement import compute_canonical_engagement_rollup

_UID = "11111111-1111-1111-1111-111111111111"


def _conn_for(*, tokens, pci_rows, upload_rows):
    """asyncpg-like conn whose ``fetch`` dispatches by table name in the SQL."""

    async def _fetch(sql, *args):
        s = " ".join(str(sql).split()).lower()
        if "from platform_tokens" in s:
            return tokens
        if "from platform_content_items" in s:
            return pci_rows
        if "from uploads" in s:
            return upload_rows
        return []

    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=_fetch)
    return conn


def _upload_row(*, platforms, pr_entries, views, likes, comments=0, shares=0):
    return {
        "id": "aaaaaaaa-0000-0000-0000-000000000001",
        "platform_results": json.dumps(pr_entries),
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "target_accounts": [],
        "platforms": platforms,
    }


def test_multi_platform_stale_pr_uses_combined_row_total():
    """PR entries all report 0, uploads row holds the real combined sum -> counted."""
    pr = [
        {"platform": "tiktok", "success": True, "platform_video_id": "tt1", "views": 0, "likes": 0},
        {"platform": "youtube", "success": True, "platform_video_id": "yt1", "views": 0, "likes": 0},
    ]
    conn = _conn_for(
        tokens=[],
        pci_rows=[],
        upload_rows=[_upload_row(platforms=["tiktok", "youtube"], pr_entries=pr, views=799, likes=27, comments=0)],
    )

    out = asyncio.run(compute_canonical_engagement_rollup(conn, _UID))

    # The whole combined synced total must be reflected (shortfall = 799 - 0).
    assert out["views"] == 799
    assert out["likes"] == 27
    assert out["breakdown"]["dedupe"]["multi_platform_upload_row_shortfall_applied"] is True


def test_multi_platform_no_double_count_when_pr_already_has_metrics():
    """When per-entry PR metrics already sum to the row total, no extra is added."""
    pr = [
        {"platform": "tiktok", "success": True, "platform_video_id": "tt1", "views": 500, "likes": 20},
        {"platform": "youtube", "success": True, "platform_video_id": "yt1", "views": 299, "likes": 7},
    ]
    conn = _conn_for(
        tokens=[],
        pci_rows=[],
        upload_rows=[_upload_row(platforms=["tiktok", "youtube"], pr_entries=pr, views=799, likes=27)],
    )

    out = asyncio.run(compute_canonical_engagement_rollup(conn, _UID))

    # 500+299 == 799 already; shortfall is 0, so no double counting.
    assert out["views"] == 799
    assert out["likes"] == 27


def test_multi_platform_partial_pr_lifts_to_row_total():
    """PR entries partially populated; shortfall lifts the total to the synced truth."""
    pr = [
        {"platform": "tiktok", "success": True, "platform_video_id": "tt1", "views": 300, "likes": 10},
        {"platform": "youtube", "success": True, "platform_video_id": "yt1", "views": 0, "likes": 0},
    ]
    conn = _conn_for(
        tokens=[],
        pci_rows=[],
        upload_rows=[_upload_row(platforms=["tiktok", "youtube"], pr_entries=pr, views=799, likes=27)],
    )

    out = asyncio.run(compute_canonical_engagement_rollup(conn, _UID))

    # tt1 key holds 300 views / 10 likes; shortfall 499 views / 17 likes -> total truth.
    assert out["views"] == 799
    assert out["likes"] == 27


def test_single_platform_row_boost_still_works():
    """Single-platform behaviour unchanged: row total applied as max on the one key."""
    pr = [
        {"platform": "tiktok", "success": True, "platform_video_id": "tt1", "views": 0, "likes": 0},
    ]
    conn = _conn_for(
        tokens=[],
        pci_rows=[],
        upload_rows=[_upload_row(platforms=["tiktok"], pr_entries=pr, views=123, likes=4)],
    )

    out = asyncio.run(compute_canonical_engagement_rollup(conn, _UID))

    assert out["views"] == 123
    assert out["likes"] == 4
