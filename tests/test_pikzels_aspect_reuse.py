"""Pikzels credit burn guards: aspect formats + optional paid extras default off."""

from __future__ import annotations

from stages.pikzels_api import pikzels_format_for_platform
from stages.thumbnail_stage import (
    _hydration_pikzels_edit_enabled,
    _upload_pikzels_text_brief_enabled,
)


def test_pikzels_format_youtube_landscape_verticals_share():
    assert pikzels_format_for_platform("youtube") == "16:9"
    assert pikzels_format_for_platform("instagram") == "9:16"
    assert pikzels_format_for_platform("facebook") == "9:16"
    assert pikzels_format_for_platform("tiktok") == "9:16"
    # All-platform upload → only two unique formats (was 4 billed API calls).
    formats = {
        pikzels_format_for_platform(p)
        for p in ("youtube", "instagram", "facebook", "tiktok")
    }
    assert formats == {"16:9", "9:16"}


def test_hydration_edit_and_text_brief_default_off(monkeypatch):
    monkeypatch.delenv("THUMBNAIL_HYDRATION_PIKZELS_EDIT", raising=False)
    monkeypatch.delenv("PIKZELS_TEXT_BRIEF_ON_UPLOAD", raising=False)
    assert _hydration_pikzels_edit_enabled() is False
    assert _upload_pikzels_text_brief_enabled() is False
    monkeypatch.setenv("THUMBNAIL_HYDRATION_PIKZELS_EDIT", "1")
    monkeypatch.setenv("PIKZELS_TEXT_BRIEF_ON_UPLOAD", "true")
    assert _hydration_pikzels_edit_enabled() is True
    assert _upload_pikzels_text_brief_enabled() is True
