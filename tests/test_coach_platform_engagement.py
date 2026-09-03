"""Unit tests: coach / ML quality scoring uses TikTok, YouTube, Meta engagement from platform_results."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from services.ml_scoring_job import aggregate_quality_score_rows
from services.upload_engagement import (
    effective_upload_metrics,
    engagement_rate_pct,
    per_platform_upload_metrics,
    rollup_engagement_from_platform_results,
)


def test_rollup_includes_facebook_reactions_and_impressions():
    """Meta Graph often returns reactions/impressions instead of likes/views."""
    roll = rollup_engagement_from_platform_results(
        [
            {
                "platform": "facebook",
                "success": True,
                "status": "published",
                "impressions": 1000,
                "reactions": 40,
                "comments": 5,
                "shares": 2,
            }
        ],
        shortform_only=True,
        successful_only=True,
    )
    assert roll["views"] == 1000
    assert roll["likes"] == 40
    assert roll["comments"] == 5
    assert roll["shares"] == 2


def test_effective_upload_metrics_prefers_platform_results_over_stale_columns():
    row = {
        "views": 0,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "platform_results": [
            {
                "platform": "youtube",
                "success": True,
                "status": "published",
                "views": 500,
                "likes": 20,
                "comments": 3,
                "shares": 0,
            },
            {
                "platform": "tiktok",
                "success": True,
                "status": "published",
                "play_count": 1200,
                "like_count": 80,
                "comment_count": 10,
                "share_count": 4,
            },
            {
                "platform": "instagram",
                "success": True,
                "status": "published",
                "views": 300,
                "likes": 25,
                "comments": 2,
                "shares": 1,
            },
        ],
    }
    m = effective_upload_metrics(row)
    assert m["views"] == 2000
    assert m["likes"] == 125
    assert m["comments"] == 15
    assert m["shares"] == 5
    assert engagement_rate_pct(m["views"], m["likes"], m["comments"], m["shares"]) > 0


def test_per_platform_metrics_differ_across_tiktok_youtube_meta():
    row = {
        "views": 0,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "platforms": ["tiktok", "youtube", "facebook"],
        "platform_results": [
            {
                "platform": "tiktok",
                "success": True,
                "views": 1000,
                "likes": 100,
                "comments": 10,
                "shares": 5,
            },
            {
                "platform": "youtube",
                "success": True,
                "views": 200,
                "likes": 4,
                "comments": 1,
                "shares": 0,
            },
            {
                "platform": "facebook",
                "success": True,
                "impressions": 400,
                "reactions": 20,
                "comments": 2,
                "shares": 1,
            },
        ],
    }
    plats = {p["platform"]: p for p in per_platform_upload_metrics(row)}
    assert set(plats) == {"tiktok", "youtube", "facebook"}
    assert plats["tiktok"]["engagement_rate_pct"] > plats["youtube"]["engagement_rate_pct"]
    assert plats["facebook"]["views"] == 400
    assert plats["facebook"]["likes"] == 20


def test_per_platform_metrics_does_not_double_count_missing_platforms():
    """When one platform has PR metrics, do not assign rolled-up columns to others."""
    row = {
        "views": 2046,
        "likes": 108,
        "comments": 3,
        "shares": 0,
        "platforms": ["tiktok", "youtube", "instagram"],
        "platform_results": [
            {
                "platform": "tiktok",
                "success": True,
                "views": 1700,
                "likes": 98,
                "comments": 2,
                "shares": 0,
            }
        ],
    }
    plats = {p["platform"]: p for p in per_platform_upload_metrics(row)}
    assert set(plats) == {"tiktok"}
    assert plats["tiktok"]["views"] == 1700


def test_aggregate_quality_scores_uses_pr_engagement_not_zero_columns():
    uid = uuid4()
    day = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "user_id": uid,
            "created_at": day,
            "views": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "platforms": ["youtube", "instagram"],
            "output_artifacts": {
                "content_attribution_key": "v1|cs=punchy|ct=cinematic|cv=teacher",
            },
            "platform_results": [
                {
                    "platform": "youtube",
                    "success": True,
                    "views": 346,
                    "likes": 10,
                    "comments": 2,
                    "shares": 0,
                },
                {
                    "platform": "instagram",
                    "success": True,
                    "views": 120,
                    "likes": 15,
                    "comments": 1,
                    "shares": 0,
                },
            ],
        }
    ]
    out = aggregate_quality_score_rows(rows)
    by_plat = {(r["platform"], r["strategy_key"]): r for r in out}
    all_row = by_plat[("all", "v1|cs=punchy|ct=cinematic|cv=teacher")]
    assert all_row["mean_views"] == 466
    assert all_row["mean_engagement"] > 0.0
    yt = by_plat[("youtube", "v1|cs=punchy|ct=cinematic|cv=teacher")]
    ig = by_plat[("instagram", "v1|cs=punchy|ct=cinematic|cv=teacher")]
    assert yt["mean_views"] == 346
    assert ig["mean_views"] == 120
    # Per-platform ERs must not both inherit a shared zero column rate.
    assert yt["mean_engagement"] != ig["mean_engagement"] or yt["mean_views"] != ig["mean_views"]


def test_slim_upload_item_uses_platform_results_engagement():
    from services.upload.list_detail import _build_slim_upload_item

    item = _build_slim_upload_item(
        {
            "id": "u1",
            "title": "Speed clip",
            "filename": "x.mp4",
            "platforms": ["tiktok", "youtube"],
            "status": "completed",
            "created_at": None,
            "completed_at": None,
            "views": 0,
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "platform_results": [
                {
                    "platform": "tiktok",
                    "success": True,
                    "views": 1700,
                    "likes": 98,
                    "comments": 2,
                    "shares": 0,
                },
                {
                    "platform": "youtube",
                    "success": True,
                    "views": 346,
                    "likes": 10,
                    "comments": 1,
                    "shares": 0,
                },
            ],
        }
    )
    assert item["views"] == 2046
    assert item["likes"] == 108
    assert item["comments"] == 3
