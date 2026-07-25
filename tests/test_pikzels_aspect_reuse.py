"""Pikzels credit burn guards: aspect formats + optional paid extras default off."""

from __future__ import annotations

from stages.pikzels_api import (
    pikzels_format_for_platform,
    pikzels_max_image_calls_per_job,
    unique_pikzels_aspect_leaders,
)
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


def test_unique_pikzels_aspect_leaders_one_vertical():
    leaders = unique_pikzels_aspect_leaders(
        ["youtube", "instagram", "facebook", "tiktok"]
    )
    assert leaders == {"16:9": "youtube", "9:16": "instagram"}
    # Only two billable leaders for four platforms.
    assert len(leaders) == 2


def test_unique_pikzels_aspect_leaders_skips_studio_winner():
    leaders = unique_pikzels_aspect_leaders(
        ["youtube", "instagram", "facebook", "tiktok"],
        skip_platforms={"youtube"},
    )
    assert "16:9" not in leaders
    assert leaders == {"9:16": "instagram"}


def test_pikzels_max_image_calls_default_two(monkeypatch):
    monkeypatch.delenv("PIKZELS_MAX_IMAGE_CALLS_PER_JOB", raising=False)
    assert pikzels_max_image_calls_per_job() == 2
    monkeypatch.setenv("PIKZELS_MAX_IMAGE_CALLS_PER_JOB", "0")
    assert pikzels_max_image_calls_per_job() == 0


def test_hydration_edit_and_text_brief_default_off(monkeypatch):
    monkeypatch.delenv("THUMBNAIL_HYDRATION_PIKZELS_EDIT", raising=False)
    monkeypatch.delenv("PIKZELS_TEXT_BRIEF_ON_UPLOAD", raising=False)
    assert _hydration_pikzels_edit_enabled() is False
    assert _upload_pikzels_text_brief_enabled() is False
    monkeypatch.setenv("THUMBNAIL_HYDRATION_PIKZELS_EDIT", "1")
    monkeypatch.setenv("PIKZELS_TEXT_BRIEF_ON_UPLOAD", "true")
    assert _hydration_pikzels_edit_enabled() is True
    assert _upload_pikzels_text_brief_enabled() is True
