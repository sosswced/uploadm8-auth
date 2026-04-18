"""Unit tests for services.canonical_engagement rollup helpers (no DB)."""

from __future__ import annotations

from datetime import datetime, timezone

from services import canonical_engagement as ce


def test_pick_int_tries_next_key_after_invalid_string() -> None:
    d = {"views": "not-a-number", "view_count": 42}
    assert ce._pick_int(d, "views", "view_count") == 42


def test_pick_int_skips_bool() -> None:
    d = {"views": True, "view_count": 7}
    assert ce._pick_int(d, "views", "view_count") == 7


def test_pick_int_float_rounds() -> None:
    d = {"views": 3.6}
    assert ce._pick_int(d, "views") == 4


def test_video_key_publish_id_before_generic_id() -> None:
    e = {"platform": "youtube", "publish_id": "abc123", "id": "internal-99"}
    assert ce._video_key_from_pr_entry(e) == ("youtube", "abc123")


def test_video_key_prefers_tiktok_video_id_over_generic_id() -> None:
    e = {
        "platform": "tiktok",
        "tiktok_video_id": "tt-999",
        "id": "internal-row",
    }
    assert ce._video_key_from_pr_entry(e) == ("tiktok", "tt-999")


def test_metrics_facebook_prefers_reactions() -> None:
    e = {"platform": "facebook", "reactions": 50, "likes": 10}
    m = ce._metrics_from_pr_entry(e)
    assert m["likes"] == 50


def test_single_upload_row_engagement_maxes_row_over_zero_roll() -> None:
    row = {
        "views": 500,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "platform_results": [
            {"platform": "tiktok", "success": True, "views": 0, "likes": 0},
        ],
    }
    m = ce._single_upload_row_engagement(row)
    assert m["views"] == 500


def test_engagement_time_window_all() -> None:
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    s, e = ce.engagement_time_window_for_analytics_range("all", now=now)
    assert s is None and e is None


def test_engagement_time_window_30d_end_is_now() -> None:
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    s, e = ce.engagement_time_window_for_analytics_range("30d", now=now)
    assert e == now
    assert s < e


def test_vec_max_clamps_overflow() -> None:
    huge = {"views": ce._MAX_METRIC + 999, "likes": 0, "comments": 0, "shares": 0}
    z = ce._zero_vec()
    out = ce._vec_max(z, huge)
    assert out["views"] == ce._MAX_METRIC


def test_normalize_platform_results_dict_shape() -> None:
    raw = {"tiktok": {"success": True, "views": 1, "platform_video_id": "v1"}}
    lst = ce._normalize_platform_results_list(raw)
    assert len(lst) == 1
    assert lst[0]["platform"] == "tiktok"
    assert lst[0]["views"] == 1


def test_simulate_all_keyless_successful_uses_row() -> None:
    """Mirror upload loop: successful pr rows but no video id → one row-level rollup."""
    orphan = ce._zero_vec()
    combined: dict = {}
    row = {
        "views": 200,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "platform_results": [{"platform": "instagram", "success": True, "status": "published"}],
    }
    pr = ce._normalize_platform_results_list(row.get("platform_results"))
    successful = [e for e in pr if isinstance(e, dict) and ce._pr_entry_successful(e)]
    assert successful
    had_key = False
    entry_orphan = ce._zero_vec()
    for e in successful:
        key = ce._video_key_from_pr_entry(e)
        m = ce._metrics_from_pr_entry(e)
        if key is None:
            entry_orphan = ce._vec_add(entry_orphan, m)
            continue
        had_key = True
    if had_key:
        orphan = ce._vec_add(orphan, entry_orphan)
    else:
        orphan = ce._vec_add(orphan, ce._single_upload_row_engagement(dict(row)))
    assert orphan["views"] >= 200


def test_pr_entry_successful_partial_status() -> None:
    assert ce._pr_entry_successful({"success": False, "status": "partial"})


def test_facebook_dual_cursor_roundtrip() -> None:
    from services.catalog_identity import dump_facebook_dual_cursor, parse_facebook_dual_cursor

    assert parse_facebook_dual_cursor(None) == (None, None)
    assert parse_facebook_dual_cursor("legacy_cursor") == ("legacy_cursor", None)
    dumped = dump_facebook_dual_cursor("v_after", "r_after")
    assert dumped is not None
    assert parse_facebook_dual_cursor(dumped) == ("v_after", "r_after")


def test_dedupe_key_scopes_by_account() -> None:
    """Same platform_video_id on two accounts must not share one combined key."""
    tokens_by_id = {
        "tok-a": {"token_row_id": "tok-a", "platform": "tiktok", "account_id": "acc-a"},
        "tok-b": {"token_row_id": "tok-b", "platform": "tiktok", "account_id": "acc-b"},
    }
    primary = {"tiktok": tokens_by_id["tok-a"]}
    used: set = set()
    e1 = {
        "platform": "tiktok",
        "success": True,
        "platform_video_id": "vid1",
        "views": 10,
        "token_row_id": "tok-a",
    }
    e2 = {**e1, "views": 20, "token_row_id": "tok-b"}
    k1 = ce._dedupe_key_from_pr_entry(
        e1,
        token_map={},
        tokens_by_id=tokens_by_id,
        primary_by_platform=primary,
        used_token_ids=used,
    )
    used.clear()
    k2 = ce._dedupe_key_from_pr_entry(
        e2,
        token_map={},
        tokens_by_id=tokens_by_id,
        primary_by_platform=primary,
        used_token_ids=used,
    )
    assert k1 is not None and k2 is not None
    assert k1[2] == k2[2]
    assert k1[1] != k2[1]
