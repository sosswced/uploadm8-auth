"""Composition-first: LOCATION / place banners must never paint on covers."""

from __future__ import annotations

from core.thumbnail_text import (
    clean_thumbnail_headline,
    is_location_banner_headline,
    is_unusable_thumbnail_headline,
)
from stages.pikzels_api import _build_pikzels_v2_prompt


def test_location_labeled_strings_are_unusable():
    assert is_location_banner_headline("Location: Federal Way, Washington")
    assert is_location_banner_headline("LOCATION FEDERAL WAY WASHINGTON")
    assert is_unusable_thumbnail_headline("LOCATION FEDERAL WAY WASHINGTON")
    # After clean, leading LOCATION is stripped — remaining city must still not paint via pack.
    cleaned = clean_thumbnail_headline("Location: Federal Way, Washington")
    assert not cleaned.startswith("LOCATION")
    assert "FEDERAL" in cleaned or cleaned == ""


def test_pikzels_never_paints_federal_way_banner():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "LOCATION FEDERAL WAY WASHINGTON",
            "_uploadm8_paint_policy": "hook_only",  # even if mis-set, must not paint place
            "_uploadm8_hook_line": "LOCATION FEDERAL WAY WASHINGTON",
            "hook_line": "LOCATION FEDERAL WAY WASHINGTON",
            "geo_context": "Federal Way, Washington",
        },
        category="automotive",
        platform="instagram",
    )
    assert 'reading "LOCATION' not in prompt
    assert 'reading "FEDERAL WAY' not in prompt
    assert "CREATIVE COMPOSITION MODE" in prompt or "STRICT NO-TEXT" in prompt
