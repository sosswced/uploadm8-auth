"""TikTok analytics aggregation + verify video_id ledger stamping."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_update_publish_attempt_verified_accepts_platform_post_id():
    from stages import db as db_stage

    src = inspect.getsource(db_stage.update_publish_attempt_verified)
    assert "platform_post_id" in src
    assert "COALESCE($4, platform_post_id)" in src or "platform_post_id = COALESCE" in src


def test_verify_stage_stamps_tiktok_ledger_video_id():
    from stages import verify_stage

    src = inspect.getsource(verify_stage.verify_single_attempt)
    assert "platform_post_id=" in src
    assert "tiktok_video_id" in src
    assert "update_publish_attempt_verified" in src


def test_fetch_tiktok_metrics_paginates_and_checks_envelope():
    from routers import analytics as analytics_router

    src = inspect.getsource(analytics_router._fetch_tiktok_metrics)
    assert "tiktok_envelope_error" in src
    assert "has_more" in src
    assert "cursor" in src
    assert "scope_missing" in src


def test_tiktok_webhook_patches_list_platform_results():
    from routers import webhooks as webhooks_router

    src = inspect.getsource(webhooks_router._handle_tiktok_event)
    assert 'existing["tiktok"]' not in src
    assert "platform_video_id" in src
    assert "publish_attempts" in src
    assert "_tiktok_webhook_pr_targets" in src
    assert "_find_tiktok_upload_for_webhook" in src
    assert "post.publish.publicly_available" in src
    assert "LIMIT 1" in src


def test_tiktok_webhook_matches_succeeded_awaiting_video_id():
    """Succeeded uploads with only publish_id must still be stampable by webhooks."""
    from routers import webhooks as webhooks_router

    src = inspect.getsource(webhooks_router._find_tiktok_upload_for_webhook)
    assert "succeeded" in src
    assert "48 hours" in src
    assert "_tiktok_pr_missing_video_id" in src


def test_tiktok_pr_missing_video_id_detects_step_a_only():
    from routers.webhooks import _tiktok_pr_missing_video_id

    assert _tiktok_pr_missing_video_id(
        [{"platform": "tiktok", "publish_id": "v_pub~1", "success": True}]
    )
    assert not _tiktok_pr_missing_video_id(
        [
            {
                "platform": "tiktok",
                "publish_id": "v_pub~1",
                "platform_video_id": "7123456789012345678",
                "success": True,
            }
        ]
    )


def test_tiktok_webhook_pr_targets_exact_publish_id_only():
    from routers.webhooks import _tiktok_webhook_pr_targets

    rows = [
        {"platform": "tiktok", "publish_id": "share-a", "platform_video_id": ""},
        {"platform": "tiktok", "publish_id": "share-b", "platform_video_id": ""},
        {"platform": "youtube", "publish_id": "share-a"},
    ]
    targets = _tiktok_webhook_pr_targets(rows, "share-b")
    assert len(targets) == 1
    assert targets[0]["publish_id"] == "share-b"


def test_tiktok_webhook_pr_targets_does_not_patch_siblings_after_first():
    from routers.webhooks import _tiktok_webhook_pr_targets

    rows = [
        {"platform": "tiktok", "publish_id": "", "platform_video_id": ""},
        {"platform": "tiktok", "publish_id": "", "platform_video_id": ""},
    ]
    targets = _tiktok_webhook_pr_targets(rows, "share-x")
    assert len(targets) == 1


def test_frontend_honors_public_publish_enabled():
    """Upload UI shows Direct Post enabled state (audit-approved)."""
    js = Path("frontend/js/tiktok-export.js").read_text(encoding="utf-8")
    html = Path("frontend/upload.html").read_text(encoding="utf-8")
    assert "syncDirectPostBanner" in js
    assert "tiktokAuditedNotice" in html
    assert "Direct Post enabled" in html
    assert "privacyClamped" in js
    assert "creatorInfoLoaded" in js
    assert "tt-private-account-hint" in js
    assert "tt-refresh-privacy" in js
    assert "forceRefresh" in js
    copy = Path("frontend/js/tiktok-ux-copy.js").read_text(encoding="utf-8")
    assert "privateAccountPrivacyHint" in copy
