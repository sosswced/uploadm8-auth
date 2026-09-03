"""Lane integrity: Vision/VI POIs survive ambient filter; music ≠ Whisper junk."""

from __future__ import annotations

from types import SimpleNamespace

from core.vision_entities import (
    VisualEntityBundle,
    build_scene_hook_line,
    collect_visual_entities,
)
from core.vision_labels import filter_vision_labels_for_context, is_ambient_redundant_vision_label
from stages.context import (
    build_hydration_story_text,
    build_video_story_timeline,
    transcript_usable_for_story,
)


def test_gas_station_not_ambient_on_automotive():
    assert not is_ambient_redundant_vision_label(
        "Gas station",
        ambient_profiles=("automotive", "dashcam"),
    )
    assert not is_ambient_redundant_vision_label(
        "Fuel pump",
        ambient_profiles=("automotive",),
    )
    kept = filter_vision_labels_for_context(
        ["Gas station", "Car wash", "Windshield", "Rear-view mirror", "Lane"],
        category="automotive",
        filename="dashcam_night.mp4",
    )
    low = " ".join(kept).lower()
    assert "gas station" in low
    assert "car wash" in low
    assert "windshield" not in low


def test_bare_gas_still_ambient_furniture():
    assert is_ambient_redundant_vision_label("gas", ambient_profiles=("automotive",))


def test_collect_visual_entities_keeps_gas_station_poi():
    bundle = collect_visual_entities(
        vision_context={
            "label_names": ["Gas station", "Car wash", "Vehicle", "Night"],
            "logo_names": ["Shell"],
        },
        category="automotive",
        filename="livermore.mp4",
    )
    assert any("gas" in x.lower() for x in bundle.scene_labels)
    assert "Shell" in bundle.brands


def test_scene_hook_includes_poi_with_place_and_music():
    bundle = VisualEntityBundle(
        scene_labels=["Gas station", "Car wash"],
        brands=["Shell"],
    )
    hook = build_scene_hook_line(
        place="Livermore",
        music_artist="A Boogie Wit da Hoodie",
        music_title="24 Hours",
        bundle=bundle,
    )
    low = hook.lower()
    assert "gas station" in low
    assert "livermore" in low
    assert "boogie" in low or "24 hours" in low


def test_transcript_junk_demoted_when_music_detected():
    junk = "슥슥슥슥 슥슥슥 슥슥슥 슥슥슥"
    assert not transcript_usable_for_story(junk, music_detected=True)
    assert transcript_usable_for_story(
        "We are pulling into the station for gas.",
        music_detected=True,
    )
    assert not transcript_usable_for_story(
        "la la la la la",
        music_detected=True,
        transcript_role="third_party_lyrics",
    )


def test_hydration_story_prefers_poi_over_whisper_junk():
    ctx = SimpleNamespace(
        upload_id="lane-1",
        filename="livermore.mp4",
        thumbnail_category="automotive",
        hydration_payload={"category": "automotive"},
        telemetry=SimpleNamespace(
            location_city="Livermore",
            location_state="California",
            location_display="Livermore, California",
            gazetteer_place_name="Livermore",
            location_road="First Street",
            padus_unit_name=None,
            mid_lat=37.7,
            mid_lon=-121.74,
            max_speed_mph=0,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={
            "label_names": ["Gas station", "Car wash", "Fuel pump"],
            "logo_names": ["Shell"],
            "ocr_text": "",
        },
        video_intelligence={},
        video_intelligence_context={},
        audio_context={
            "music_detected": True,
            "music_artist": "A Boogie Wit da Hoodie",
            "music_title": "24 Hours (feat. Lil Durk)",
            "transcript": "슥슥슥슥 슥슥슥 슥슥슥",
            "transcript_segments": [
                {"start": 0.0, "text": "슥슥슥슥 슥슥슥"},
                {"start": 19.0, "text": "슥슥슥 슥슥슥"},
            ],
        },
        ai_transcript="슥슥슥슥 슥슥슥",
        video_understanding={},
        visual_recognition={},
        output_artifacts={},
        trill=None,
        trill_score=None,
    )
    story = build_hydration_story_text(ctx, max_chars=900).lower()
    assert "gas station" in story or "shell" in story or "car wash" in story
    assert "speech/transcript cue" not in story
    assert "boogie" in story or "24 hours" in story

    events = build_video_story_timeline(ctx, max_events=40)
    kinds = [str(e.get("kind") or "") for e in events]
    assert "music" in kinds
    assert "transcript" not in kinds
