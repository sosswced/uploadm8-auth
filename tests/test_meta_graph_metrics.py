"""Unit tests for Meta Graph metrics fallbacks (no HTTP)."""

from services.meta_graph_metrics import instagram_account_degraded_live


def test_instagram_degraded_is_live_for_rollup():
    d = instagram_account_degraded_live(http_status=403, ig_user_id="1784")
    assert d["status"] == "live"
    assert d["views"] == 0
    assert d["analytics_source"] == "insufficient_scope"
    assert "instagram_basic" in d.get("analytics_note", "")
