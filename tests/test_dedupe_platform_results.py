"""Dedupe bloated platform_results for queue chips."""

from __future__ import annotations

from services.uploads_api import dedupe_platform_result_entries


def test_dedupe_collapses_same_account_repeats():
    items = [
        {"platform": "tiktok", "success": True, "account_username": "cedybandz5254", "publish_id": "p1"},
        {"platform": "tiktok", "success": True, "account_username": "cedybandz5254", "publish_id": "p2", "platform_url": "https://tiktok.com/x"},
        {"platform": "youtube", "success": True, "account_username": "realroadrunner7"},
        {"platform": "youtube", "success": True, "account_username": "realroadrunner7"},
    ]
    out = dedupe_platform_result_entries(items)
    assert len(out) == 2
    tt = next(x for x in out if x["platform"] == "tiktok")
    assert tt.get("platform_url")  # preferred richer entry


def test_dedupe_keeps_distinct_accounts_same_platform():
    items = [
        {"platform": "tiktok", "success": True, "token_row_id": "aaa", "account_username": "a"},
        {"platform": "tiktok", "success": True, "token_row_id": "bbb", "account_username": "b"},
    ]
    assert len(dedupe_platform_result_entries(items)) == 2


def test_dedupe_anonymous_identical_publish_id_collapses():
    items = [
        {"platform": "tiktok", "success": True, "publish_id": "same-pub"}
        for _ in range(10)
    ]
    out = dedupe_platform_result_entries(items)
    assert len(out) == 1


def test_dedupe_anonymous_distinct_video_ids_kept():
    """Distinct live posts without token_row_id must not collapse to one chip."""
    items = [
        {"platform": "tiktok", "success": True, "platform_video_id": f"vid{i}"}
        for i in range(3)
    ]
    assert len(dedupe_platform_result_entries(items)) == 3


def test_dedupe_anonymous_no_ids_collapse_by_platform():
    items = [{"platform": "youtube", "success": True} for _ in range(5)]
    assert len(dedupe_platform_result_entries(items)) == 1


def test_dedupe_empty():
    assert dedupe_platform_result_entries([]) == []
    assert dedupe_platform_result_entries(None) == []  # type: ignore[arg-type]
