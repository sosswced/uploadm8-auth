"""24/7 scene fusion: TL-independent scene understanding + welcome OCR."""

from __future__ import annotations

from types import SimpleNamespace

from services.scene_fusion import (
    apply_scene_fusion,
    extract_place_signs,
    build_fusion_scene,
    enrich_thin_fusion_scene,
    fusion_scene_is_thin,
    _parse_enrich_response,
)
from services.hydration_enforcer import (
    build_title_anchor_phrase,
    collect_evidence,
    enforce_hydration,
)


def test_extract_welcome_to_place_signs():
    signs = extract_place_signs(
        "WELCOME TO ASHLAND City Limit Population 21,000",
        "Entering Logandale — next exit 2 miles",
    )
    assert any("Ashland" in s for s in signs)
    assert any("Logandale" in s for s in signs)


def test_fusion_fills_scene_when_twelvelabs_empty():
    ctx = SimpleNamespace(
        upload_id="fusion-1",
        telemetry=SimpleNamespace(
            max_speed_mph=154.0,
            avg_speed_mph=120.0,
            location_city="Logandale",
            location_state="California",
            location_country="US",
            location_road="Westside Freeway",
            location_display="Logandale, California",
            location_start_display="Near I-15",
            gazetteer_place_name="Logandale",
            padus_unit_name=None,
            near_padus=False,
            mid_lat=39.37,
            mid_lon=-122.19,
            points=[],
        ),
        telemetry_data=None,
        dashcam_osd_context={"max_speed_mph": 154.0},
        vision_context={"ocr_text": "Welcome to Logandale", "logo_names": ["Nike"]},
        audio_context={
            "music_detected": True,
            "music_artist": "Fetty Wap",
            "music_title": "The Truth",
        },
        video_intelligence={"on_screen_text": [{"text": "Westside Freeway"}]},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="Keep it moving through the night.",
        thumbnail_category="automotive",
        filename="run.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
    )
    report = apply_scene_fusion(ctx)
    assert not report.get("skipped")
    vu = ctx.video_understanding
    assert vu.get("source") == "fusion"
    assert vu.get("scene_description")
    assert vu.get("description") == vu.get("scene_description")
    assert "154" in vu["scene_description"] or "Logandale" in vu["scene_description"]
    assert vu.get("title_suggestion")
    assert "154 MPH" in vu["title_suggestion"]
    assert "Logandale" in vu["title_suggestion"]
    cq = vu.get("custom_queries") or {}
    assert cq.get("music_id") and "Fetty" in cq["music_id"]
    assert cq.get("location_clue")
    assert cq.get("peak_speed") == "154 MPH"
    assert cq.get("brands_visible") and "Nike" in cq["brands_visible"]


def test_fusion_skips_when_twelve_labs_present():
    ctx = SimpleNamespace(
        upload_id="fusion-2",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={
            "scene_description": "A cinematic night run on the coast.",
            "source": "twelve_labs",
        },
        ai_transcript="",
        thumbnail_category="automotive",
        filename="run.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
    )
    report = apply_scene_fusion(ctx)
    assert report.get("skipped")
    assert ctx.video_understanding["source"] == "twelve_labs"
    assert "cinematic night run" in ctx.video_understanding["scene_description"]


def test_welcome_sign_place_when_gps_city_missing():
    ctx = SimpleNamespace(
        upload_id="fusion-3",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={"max_speed_mph": 88.0},
        vision_context={"ocr_text": "Welcome to Ashland"},
        audio_context={},
        video_intelligence={
            "on_screen_text": [{"text": "WELCOME TO ASHLAND"}],
        },
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        thumbnail_category="automotive",
        filename="clip.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
        ai_title="Untitled",
        ai_caption="",
        ai_hashtags=[],
        m8_platform_captions={},
        m8_platform_titles={},
        m8_platform_hashtags={},
    )
    fused = build_fusion_scene(ctx)
    assert "Ashland" in (fused.get("scene_description") or "")
    apply_scene_fusion(ctx)
    pool = collect_evidence(ctx)
    assert pool.place_sign and "Ashland" in pool.place_sign
    title = build_title_anchor_phrase(pool, ctx)
    assert "Ashland" in title
    assert "Captured at" not in title


def test_enforce_hydration_uses_fusion_title_not_place_only():
    ctx = SimpleNamespace(
        upload_id="fusion-4",
        telemetry=SimpleNamespace(
            max_speed_mph=154.0,
            avg_speed_mph=120.0,
            location_city="Logandale",
            location_state="California",
            location_country="US",
            location_road="Westside Freeway",
            gazetteer_place_name="Logandale",
            location_start_display=None,
            padus_unit_name=None,
            near_padus=False,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={
            "music_detected": True,
            "music_artist": "Fetty Wap",
            "music_title": "The Truth",
        },
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        thumbnail_category="automotive",
        filename="clip.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
        ai_title="Logandale, CA",
        ai_caption="Captured at 154 MPH, near Logandale, CA, with Fetty Wap — The Truth on the speakers.",
        ai_hashtags=["logandale"],
        m8_platform_captions={},
        m8_platform_titles={"youtube": "Logandale, CA"},
        m8_platform_hashtags={},
    )
    apply_scene_fusion(ctx)
    report = enforce_hydration(ctx)
    assert report.get("rewrote_title") or "154 MPH" in (ctx.ai_title or "")
    assert "154 MPH" in (ctx.ai_title or "")
    assert "Captured at" not in (ctx.ai_title or "")
    assert "Captured at 154 MPH" in (ctx.ai_caption or "")


def test_twelvelabs_skip_when_vi_rich_defaults_false():
    from stages import twelvelabs_stage as tl

    # Module default is env-parsed at import; assert the planned default string.
    import os

    assert os.environ.get("TWELVELABS_SKIP_WHEN_VI_RICH", "false").lower() in (
        "false",
        "0",
        "",
    ) or tl.TWELVELABS_SKIP_WHEN_VI_RICH in (True, False)
    # When env unset in this process, expected product default is false.
    if "TWELVELABS_SKIP_WHEN_VI_RICH" not in os.environ:
        assert tl.TWELVELABS_SKIP_WHEN_VI_RICH is False


def test_title_suggestion_cannot_override_trusted_peak():
    """TL/fusion suggestion with wrong MPH must lose to OSD/telemetry peak."""
    from services.hydration_enforcer import build_title_anchor_phrase, collect_evidence

    ctx = SimpleNamespace(
        upload_id="peak-guard",
        telemetry=SimpleNamespace(
            max_speed_mph=154.0,
            avg_speed_mph=120.0,
            location_city="Logandale",
            location_state="California",
            location_country="US",
            location_road="Westside Freeway",
            gazetteer_place_name="Logandale",
            location_start_display=None,
            padus_unit_name=None,
            near_padus=False,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={"music_detected": True, "music_artist": "Fetty Wap"},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={
            "source": "twelve_labs",
            "scene_description": "Night freeway run.",
            "title_suggestion": "46 MPH · Logandale, CA · Fetty Wap",
        },
        ai_transcript="",
        thumbnail_category="automotive",
        filename="clip.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
    )
    pool = collect_evidence(ctx)
    title = build_title_anchor_phrase(pool, ctx)
    assert "154 MPH" in title
    assert "46 MPH" not in title
    assert "Logandale" in title


def test_long_placeish_title_still_upgraded_when_speed_missing():
    """Creative geo titles without MPH are thin when trusted peak exists."""
    from services.hydration_enforcer import enforce_hydration

    ctx = SimpleNamespace(
        upload_id="long-thin",
        telemetry=SimpleNamespace(
            max_speed_mph=154.0,
            avg_speed_mph=120.0,
            location_city="Logandale",
            location_state="California",
            location_country="US",
            location_road="Westside Freeway",
            gazetteer_place_name="Logandale",
            location_start_display=None,
            padus_unit_name=None,
            near_padus=False,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={
            "music_detected": True,
            "music_artist": "Fetty Wap",
            "music_title": "The Truth",
        },
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        thumbnail_category="automotive",
        filename="clip.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
        ai_title="Sunset cruise near Logandale California westside",
        ai_caption="Cruise vibes.",
        ai_hashtags=["logandale"],
        m8_platform_captions={},
        m8_platform_titles={},
        m8_platform_hashtags={},
    )
    apply_scene_fusion(ctx)
    enforce_hydration(ctx)
    assert "154 MPH" in (ctx.ai_title or "")


def test_welcome_sign_becomes_hashtag_and_timeline_beat():
    from services.hydration_enforcer import build_evidence_hashtags, collect_evidence
    from stages.context import build_video_story_timeline

    ctx = SimpleNamespace(
        upload_id="welcome-tags",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={"max_speed_mph": 88.0},
        vision_context={"ocr_text": "Welcome to Ashland"},
        audio_context={},
        video_intelligence={"on_screen_text": [{"text": "WELCOME TO ASHLAND"}]},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        thumbnail_category="automotive",
        filename="clip.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
    )
    apply_scene_fusion(ctx)
    pool = collect_evidence(ctx)
    tags = [t.lower() for t in build_evidence_hashtags(pool)]
    assert any("ashland" in t for t in tags)
    beats = build_video_story_timeline(ctx, max_events=40)
    kinds = {str(b.get("kind") or "") for b in beats if isinstance(b, dict)}
    assert "welcome_sign" in kinds


def test_collect_evidence_survives_non_dict_vision_context():
    """Malformed vision_context must not abort hydration (recognition_flat guard)."""
    from services.hydration_enforcer import collect_evidence

    ctx = SimpleNamespace(
        upload_id="bad-vc",
        telemetry=SimpleNamespace(
            max_speed_mph=40.0,
            location_city="Ashland",
            location_state="Oregon",
            location_country="US",
            location_road=None,
            gazetteer_place_name=None,
            location_start_display=None,
            padus_unit_name=None,
            near_padus=False,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context="not-a-dict",
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        thumbnail_category="general",
        filename="x.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
        visual_recognition=None,
    )
    pool = collect_evidence(ctx)
    assert pool.city == "Ashland"
    assert pool.max_speed_mph >= 5


def test_m8_deterministic_title_speed_first_with_place_sign():
    from stages.m8_engine import _deterministic_evidence_title, _validate_title

    sg = {
        "geo": {
            "place_sign": "Ashland",
            "max_speed_mph": 88.0,
        },
        "dashcam_osd": {"max_speed_mph": 88.0},
        "music": {"artist": "The Weeknd", "title": "Take Me Back To LA"},
        "trill": {},
        "video_intelligence": {},
        "vision": {},
    }
    title = _deterministic_evidence_title(sg, platform="youtube")
    assert title
    assert title.startswith("88 MPH")
    assert "Ashland" in title
    assert " · " not in title
    assert "through" in title.lower()
    ok, reason = _validate_title(title, sg, platform="youtube")
    assert ok, reason


def test_m8_rejects_checklist_dot_stack_titles():
    from stages.m8_engine import _validate_title

    sg = {"geo": {}, "transcript": {}}
    ok, reason = _validate_title("Garlock Road · The Eagles · Spirited", sg, platform="instagram")
    assert not ok
    assert reason == "checklist_dot_stack"
    ok2, reason2 = _validate_title("110 MPH · Garlock Road", sg, platform="instagram")
    assert not ok2
    assert reason2 in ("formula_stub", "checklist_dot_stack")


def test_m8_title_from_caption_voice():
    from stages.m8_engine import _platform_title_from_caption

    cap = (
        "Night air locks in at 110 MPH through Garlock Road while the cabin stays quiet. "
        "Another beat follows."
    )
    yt = _platform_title_from_caption("youtube", cap)
    ig = _platform_title_from_caption("instagram", cap)
    assert yt and "110 MPH" in yt and "Garlock" in yt
    assert ig and "Garlock" in ig
    assert " · " not in (yt or "")

def test_parse_enrich_response_requires_substance():
    assert _parse_enrich_response("not json") is None
    assert _parse_enrich_response('{"scene_description":"too short"}') is None
    ok = _parse_enrich_response(
        '{"scene_description":"Night dashcam run near Bieber with Kodak Black on the speakers while the cabin stays locked in.",'
        '"title_suggestion":"Bieber night run"}'
    )
    assert ok and "Bieber" in ok["scene_description"]


def test_fusion_scene_is_thin_threshold():
    ctx = SimpleNamespace(
        video_understanding={"scene_description": "Fast run.", "source": "fusion"}
    )
    assert fusion_scene_is_thin(ctx) is True
    ctx.video_understanding["scene_description"] = (
        "A longer fused scene description that already clears the thin threshold "
        "with place music and motion context baked in."
    )
    assert fusion_scene_is_thin(ctx) is False


import asyncio


def test_enrich_thin_fusion_skips_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = SimpleNamespace(
        upload_id="enrich-1",
        video_understanding={"scene_description": "Fast run.", "source": "fusion"},
        output_artifacts={},
    )
    report = asyncio.run(enrich_thin_fusion_scene(ctx))
    assert report.get("enriched") is False
    assert report.get("reason") == "no_openai_key"
