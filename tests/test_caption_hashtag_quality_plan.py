"""Regression coverage for caption/title/hashtag quality plan.

Music run-ons, OCR routes, generic Vision bans, title word-boundary clip,
and writing-mix title lead facets.
"""

from __future__ import annotations

from core.caption_creative import compose_creative_directive, title_lead_facet
from core.helpers import (
    clip_at_word_boundary,
    extract_highway_route_tokens,
    extract_local_business_boards,
    is_instructional_road_sign,
    music_track_hashtag_bodies,
    sanitize_hashtag_body,
)
from core.upload_baseline_defaults import UNIVERSAL_UPLOAD_BASELINE
from core.vision_labels import is_generic_vision_label, is_junk_hashtag_body
from services.signal_hashtags import build_signal_hashtags
from stages.context import JobContext, TelemetryData


def test_music_track_short_title_kept_long_clause_dropped():
    short = music_track_hashtag_bodies("Drake", "Hotline Bling")
    assert "drake" in short
    assert "hotlinebling" in short

    long = music_track_hashtag_bodies(
        "Usher",
        "Love in This Club, Pt. II",
        "R&B/Soul/Funk",
    )
    assert "usher" in long
    assert "loveinthisclubptii" not in long
    assert "loveinthisclub" in long
    assert sanitize_hashtag_body("Love in This Club, Pt. II") not in long


def test_clip_at_word_boundary_never_mid_word():
    s = "West Valley Highway dashcam at forty seven MPH near Tukwila Washington"
    clipped = clip_at_word_boundary(s, 40)
    assert len(clipped) <= 40
    assert not clipped.endswith("Highw")
    assert " " in clipped or len(clipped) < 20
    # Last char should not be mid-token shred from the next word.
    assert clipped == clipped.rstrip()
    remainder = s[len(clipped) :].lstrip()
    if remainder:
        assert clipped[-1].isalnum() or clipped[-1] in "—"


def test_ocr_route181_and_instructional_drop():
    tokens = extract_highway_route_tokens("181 SOUTH KEEP LEFT I-5 near Tukwila")
    assert "181south" in tokens
    assert "route181" not in tokens
    assert "i5" in tokens
    assert is_instructional_road_sign("KEEP LEFT")
    assert is_instructional_road_sign("SIGNAL")
    assert not is_instructional_road_sign("West Valley Highway")


def test_local_business_board_from_ocr_all_caps():
    boards = extract_local_business_boards("SALAL CREDIT UNION EDIT UNION")
    assert any("salal" in sanitize_hashtag_body(b) for b in boards)


def test_generic_vehicle_and_mood_banned():
    for slug in ("sedan", "familycar", "compactcar", "chill", "forest"):
        assert is_junk_hashtag_body(slug) or is_generic_vision_label(slug)


def test_title_lead_facet_by_mix():
    assert title_lead_facet("factual") == "place/geo"
    assert title_lead_facet("freestyle", "hype") == "free/any-evidence"
    assert title_lead_facet("punchy", "hype") == "speed/telemetry"
    assert title_lead_facet("diary") == "music/audio"
    assert title_lead_facet("listicle") == "sign/OCR"
    assert title_lead_facet("story") == "env/route"
    brief = compose_creative_directive("factual", "documentary", "teacher")
    assert "TITLE LEAD FACET" in brief
    assert "place/geo" in brief
    free = compose_creative_directive("freestyle", "hype", "teacher")
    assert "free/any-evidence" in free or "no forced evidence class" in free
    punchy = compose_creative_directive("punchy", "hype", "hypebeast")
    assert "speed/telemetry" in punchy
    assert brief != punchy


def test_signal_hashtags_tukwila_usher_style_upload():
    ctx = JobContext(job_id="job-test", upload_id="test-hashtag-quality", user_id="u-test")
    ctx.platforms = ["instagram"]
    tel = TelemetryData()
    tel.location_city = "Tukwila"
    tel.location_state = "Washington"
    tel.location_road = "West Valley Highway"
    ctx.telemetry = tel
    ctx.telemetry_data = tel
    ctx.audio_context = {
        "music_detected": True,
        "music_artist": "Usher",
        "music_title": "Love in This Club, Pt. II",
        "music_genre": "R&B/Soul/Funk",
    }
    ctx.vision_context = {
        "ocr_text": "181 SOUTH KEEP LEFT SALAL CREDIT UNION",
        "logo_names": [],
        "landmark_names": [],
    }
    tags = build_signal_hashtags(ctx, max_extra=12)
    assert "usher" in tags
    assert "loveinthisclubptii" not in tags
    assert "loveinthisclub" in tags
    assert "tukwila" in tags or "washington" in tags
    assert "westvalleyhighway" in tags or any("westvalley" in t for t in tags)
    assert "181south" in tags
    assert "route181" not in tags
    assert "keepleft" not in tags
    assert "sedan" not in tags
    assert "familycar" not in tags
    assert "chill" not in tags
    assert not any("salal" in t for t in tags)


def test_signal_hashtags_env_lane_under_strong_geo_music():
    ctx = JobContext(job_id="job-env", upload_id="test-env-lane", user_id="u-test")
    ctx.platforms = ["instagram"]
    tel = TelemetryData()
    tel.location_city = "Tukwila"
    tel.location_state = "Washington"
    ctx.telemetry = tel
    ctx.telemetry_data = tel
    ctx.audio_context = {
        "music_detected": True,
        "music_artist": "Usher",
        "music_title": "Love in This Club, Pt. II",
    }
    ctx.vision_context = {
        "ocr_text": "181 SOUTH",
        "labels": ["heavy snowfall", "tree", "outdoors", "sedan"],
        "logo_names": [],
        "landmark_names": [],
    }
    tags = build_signal_hashtags(ctx, max_extra=12)
    assert "snowfall" in tags
    assert "sedan" not in tags
    assert "tree" not in tags
    assert "outdoors" not in tags
    assert "usher" in tags


def test_baseline_ai_hashtag_count_aligned():
    assert str(UNIVERSAL_UPLOAD_BASELINE.get("aiHashtagCount")) == "15"
    assert int(UNIVERSAL_UPLOAD_BASELINE.get("ai_hashtag_count") or 0) == 15


def test_rare_env_hashtag_allowlist_not_coarse():
    from core.vision_labels import rare_env_hashtag_bodies

    assert rare_env_hashtag_bodies("tree", "outdoors", "plant", limit=2) == []
    got = rare_env_hashtag_bodies("heavy snowfall", "cherry blossoms", "car", limit=2)
    assert "snowfall" in got
    assert "cherryblossoms" in got
    assert len(got) <= 2
    ferry = rare_env_hashtag_bodies("Puget Sound ferry crossing", limit=2)
    assert "ferry" in ferry


def test_prefer_pack_title_if_generic_replaces_clickbait():
    from core.publish_pack import prefer_pack_title_if_generic

    pack = {
        "subject": "Dashcam near Tukwila Washington",
        "hook_line": "Usher — Love in This Club",
    }
    out = prefer_pack_title_if_generic("POV: wait until you see this drive", pack)
    assert out
    assert len(out) <= 100
    assert "pov" not in out.lower()
    assert "tukwila" in out.lower() or "usher" in out.lower()
    keep = prefer_pack_title_if_generic("Usher on West Valley Highway near Tukwila", pack)
    assert keep == ""


def test_title_clip_budget_is_100():
    long = "West Valley Highway dashcam at forty seven MPH near Tukwila Washington with Usher playing"
    clipped = clip_at_word_boundary(long, 100)
    assert len(clipped) <= 100
    assert not clipped.endswith("Washing")
