"""Place/artist hashtags must stay split — never city+state or artist|artist run-ons."""

from __future__ import annotations

from core.helpers import (
    expand_geo_runon_hashtag,
    normalize_hashtag_bodies,
    split_hashtag_source_phrases,
)
from services.signal_hashtags import build_signal_hashtags
from stages.context import JobContext, TelemetryData


def test_expand_geo_runon_los_angeles_california():
    assert expand_geo_runon_hashtag("losangelescalifornia") == ["losangeles", "california"]
    assert expand_geo_runon_hashtag("lasvegasnv") == ["lasvegas", "nevada"]
    assert expand_geo_runon_hashtag("#LosAngelesCA") == ["losangeles", "california"]
    assert expand_geo_runon_hashtag("tumwaterwa") == ["tumwater", "washington"]
    # Short brands / given names ending in state letters must not be shredded.
    assert expand_geo_runon_hashtag("tesla") == ["tesla"]
    assert expand_geo_runon_hashtag("angelica") == ["angelica"]
    assert expand_geo_runon_hashtag("veronica") == ["veronica"]
    assert expand_geo_runon_hashtag("america") == ["america"]



def test_split_city_state_display_and_artists():
    assert split_hashtag_source_phrases("Las Vegas, NV") == ["Las Vegas", "nevada"]
    assert split_hashtag_source_phrases("Los Angeles, California") == [
        "Los Angeles",
        "california",
    ]
    assert split_hashtag_source_phrases("Destroy Lonely|Lil Uzi Vert") == [
        "Destroy Lonely",
        "Lil Uzi Vert",
    ]


def test_collab_x_splits_but_leading_x_names_survive():
    # "A x B" collab delimiter still splits…
    assert split_hashtag_source_phrases("Drake x Future") == ["Drake", "Future"]
    assert split_hashtag_source_phrases("Drake X Future") == ["Drake", "Future"]
    # …but names that merely start/end with the word X are never shredded.
    assert split_hashtag_source_phrases("X Games") == ["X Games"]
    assert split_hashtag_source_phrases("X Ambassadors") == ["X Ambassadors"]
    assert split_hashtag_source_phrases("Racer X") == ["Racer X"]


def test_normalize_hashtag_bodies_splits_runons():
    tags = normalize_hashtag_bodies(
        [
            "#lasvegasnv",
            "Destroy Lonely|Lil Uzi Vert",
            "lasvegas",
            "nevada",
        ]
    )
    assert "lasvegas" in tags
    assert "nevada" in tags
    assert "lasvegasnv" not in tags
    assert "destroylonely" in tags
    assert "liluzivert" in tags
    assert "destroylonelyliluzivert" not in tags


def test_merge_unsquashes_smashed_artist_when_source_has_pipe():
    from services.signal_hashtags import merge_signal_hashtags_into_ctx

    ctx = JobContext(
        job_id="j-unsquash",
        upload_id="unsquash",
        user_id="u",
        filename="clip.mp4",
        platforms=["youtube"],
        telemetry=TelemetryData(
            max_speed_mph=90.0,
            location_city="Los Angeles",
            location_state="California",
            location_country="US",
        ),
        audio_context={
            "music_detected": True,
            "music_artist": "Destroy Lonely|Lil Uzi Vert",
            "music_title": "LOVE HURTS",
        },
        ai_hashtags=["destroylonelyliluzivert", "lovehurts", "fyp"],
    )
    merge_signal_hashtags_into_ctx(ctx)
    tags = {t.lower() for t in (ctx.ai_hashtags or [])}
    assert "destroylonely" in tags
    assert "liluzivert" in tags
    assert "destroylonelyliluzivert" not in tags
    assert "losangeles" in tags
    assert "california" in tags
    assert "losangelesca" not in tags
    assert "losangelescalifornia" not in tags


def test_signal_hashtags_emit_split_city_state_and_artists():
    ctx = JobContext(
        job_id="j-geo-ht",
        upload_id="geo-ht",
        user_id="u",
        filename="clip.mp4",
        platforms=["youtube"],
        telemetry=TelemetryData(
            max_speed_mph=128.0,
            location_city="Las Vegas",
            location_state="Nevada",
            location_country="US",
            location_road="Las Vegas Freeway",
            location_start_display="Las Vegas, NV",
        ),
        audio_context={
            "music_detected": True,
            "music_artist": "Destroy Lonely|Lil Uzi Vert",
            "music_title": "LOVE HURTS",
        },
    )
    tags = {t.lower() for t in build_signal_hashtags(ctx)}
    assert "lasvegas" in tags
    assert "nevada" in tags
    assert "lasvegasnv" not in tags
    assert "lasvegasnevada" not in tags
    assert "destroylonely" in tags
    assert "liluzivert" in tags
    assert "destroylonelyliluzivert" not in tags
    assert "lovehurts" in tags
