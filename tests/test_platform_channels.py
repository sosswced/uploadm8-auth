"""Unit tests for services.platform_channels (analytics aliases, display labels)."""

from __future__ import annotations

import pytest

from services import platform_channels as pc


def test_resolve_instagram_reels_alias() -> None:
    f = pc.resolve_analytics_platform_filter("instagram_reels")
    assert f.platform == "instagram"
    assert f.catalog_content_kind == "reel"
    assert "Reels" in f.display_name


def test_resolve_facebook_reels_alias() -> None:
    f = pc.resolve_analytics_platform_filter("facebook_reels")
    assert f.platform == "facebook"
    assert f.catalog_content_kind == "reel"


def test_resolve_canonical_instagram_no_pci_kind_filter() -> None:
    f = pc.resolve_analytics_platform_filter("instagram")
    assert f.platform == "instagram"
    assert f.catalog_content_kind is None


def test_resolve_all() -> None:
    f = pc.resolve_analytics_platform_filter("all")
    assert f.platform is None
    assert f.display_name == "All platforms"


def test_resolve_invalid() -> None:
    with pytest.raises(ValueError):
        pc.resolve_analytics_platform_filter("not_a_platform")


def test_platform_display_labels() -> None:
    assert "Reels" in pc.platform_display_label("instagram")
    assert "Reels" in pc.platform_display_label("facebook")
