"""Catalog aggregate period defaults and dropdown-aligned keys."""

from services.catalog_sync import _normalize_period_key, _parse_period_to_sql_interval


def test_parse_period_aliases():
    assert _parse_period_to_sql_interval("30d") == "30 days"
    assert _parse_period_to_sql_interval("1y") == "365 days"
    assert _parse_period_to_sql_interval("365d") == "365 days"
    assert _parse_period_to_sql_interval("all") is None
    assert _parse_period_to_sql_interval(None) is None


def test_normalize_period_key_matches_ui():
    assert _normalize_period_key(None) == "30d"
    assert _normalize_period_key("30d") == "30d"
    assert _normalize_period_key("1y") == "1y"
    assert _normalize_period_key("365d") == "1y"
    assert _normalize_period_key("all") == "all"
    assert _normalize_period_key(None, days=7) == "7d"
