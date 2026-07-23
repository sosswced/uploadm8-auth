"""Caption / title prompts must carry trusted HUD speeds + timeline spine."""

from __future__ import annotations

from types import SimpleNamespace

from services.hydration_enforcer import _vision_ocr_peak_mph, collect_evidence
from stages.context import build_hydration_story_text, build_video_story_timeline
from stages.dashcam_osd_stage import _aggregate, parse_osd_line
from stages.m8_engine import (
    M8_CAPTION_STYLES,
    M8_CAPTION_TONES,
    M8_CAPTION_VOICES,
    _build_hydration_timeline_brief,
    _build_m8_prompt,
    build_must_use_shortlist,
)


def _osd_with_samples():
    samples = [
        parse_osd_line(
            "2025/03/01 03:13:18 PM 39.349976° -122.194778° 92MPH C Walker",
            t_s=10.0,
        ),
        parse_osd_line(
            "2025/03/01 03:14:34 PM 39.375992° -122.192772° 87MPH C Walker",
            t_s=86.0,
        ),
        parse_osd_line(
            "2025/03/01 03:15:00 PM 39.376340° -122.192770° 90MPH C Walker",
            t_s=112.0,
        ),
        # OCR spike that must not become the published peak
        parse_osd_line(
            "2025/03/01 03:15:10 PM 39.376400° -122.192800° 154MPH C Walker",
            t_s=122.0,
        ),
    ]
    return _aggregate(samples)


def test_osd_aggregate_rejects_154_spike_when_samples_are_90s():
    osd = _osd_with_samples()
    assert osd["max_speed_mph"] <= 95
    assert any(abs(s["mph"] - 92) < 0.1 for s in osd["speed_series"])
    assert osd.get("max_speed_at_s") is not None


def test_vision_ocr_peak_rejects_outlier_spike():
    ocr = "39.35 -122.19 92MPH\n---\n87MPH\n---\n154 MPH ESCORT."
    assert _vision_ocr_peak_mph(ocr) <= 95


def test_collect_evidence_caps_spike_with_speed_series():
    osd = _osd_with_samples()
    # Simulate a stale inflated peak that somehow stayed on the dict.
    osd["max_speed_mph"] = 154.0
    ctx = SimpleNamespace(
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context=osd,
        vision_context={},
        audio_context={},
        trill=None,
        trill_score=None,
        ai_transcript="",
        video_intelligence=None,
        video_intelligence_context=None,
        video_understanding=None,
        filename="clip.mp4",
    )
    pool = collect_evidence(ctx)
    assert pool.max_speed_mph <= 95
    assert "series" in (pool.speed_source or "")


def test_collect_evidence_does_not_cap_map_telemetry_with_osd_series():
    """Bugbot high: .map peak must win even when HUD samples are lower."""
    osd = _osd_with_samples()
    osd["max_speed_mph"] = 92.0
    tel = SimpleNamespace(
        max_speed_mph=154.0,
        avg_speed_mph=110.0,
        location_road=None,
        location_city="Logandale",
        location_state="CA",
        location_country="US",
        gazetteer_place_name=None,
        padus_unit_name=None,
        near_padus=False,
        points=[{"lat": 1, "lon": 2, "speed_mph": 154}],
    )
    ctx = SimpleNamespace(
        telemetry=tel,
        telemetry_data=tel,
        dashcam_osd_context=osd,
        vision_context={},
        audio_context={},
        trill=None,
        trill_score=None,
        ai_transcript="",
        video_intelligence=None,
        video_intelligence_context=None,
        video_understanding=None,
        filename="clip.mp4",
    )
    pool = collect_evidence(ctx)
    assert pool.max_speed_mph == 154.0
    assert pool.speed_source == "telemetry"


def test_hydration_story_peak_matches_trusted_not_raw_spike():
    osd = _osd_with_samples()
    osd["max_speed_mph"] = 154.0  # poisoned aggregate
    ctx = SimpleNamespace(
        dashcam_osd_context=osd,
        telemetry=None,
        telemetry_data=None,
        vision_context={},
        audio_context={},
        video_intelligence=None,
        video_intelligence_context=None,
        visual_recognition=None,
        filename="clip.mp4",
        thumbnail_category="automotive",
        hydration_payload={},
        ai_transcript="",
    )
    story = build_hydration_story_text(ctx, max_chars=900)
    assert "154" not in story
    assert "peak speed about" in story
    assert "HUD speed samples" in story


def test_evidence_matrix_includes_all_registered_prefs():
    from core.caption_creative import (
        CAPTION_STYLES,
        CAPTION_TONES,
        CAPTION_VOICES,
        evidence_matrix_cell_specs,
    )

    cells = evidence_matrix_cell_specs("freestyle", "documentary", "passenger")
    styles = {c[0] for c in cells}
    tones = {c[1] for c in cells}
    voices = {c[2] for c in cells}
    assert styles == set(CAPTION_STYLES)
    assert tones == set(CAPTION_TONES)
    assert voices == set(CAPTION_VOICES)
    assert ("freestyle", "documentary", "passenger") in cells


def test_timeline_uses_max_speed_at_s_and_sample_beats():
    osd = _osd_with_samples()
    ctx = SimpleNamespace(
        dashcam_osd_context=osd,
        telemetry=None,
        telemetry_data=None,
        vision_context={},
        audio_context={},
        video_intelligence=None,
        video_intelligence_context=None,
        video_info={"duration": 150},
        ai_transcript="",
        output_artifacts={},
    )
    events = build_video_story_timeline(ctx, max_events=40)
    kinds = {e.get("kind") for e in events}
    assert "osd_speed" in kinds or "osd_speed_beat" in kinds
    texts = " ".join(str(e.get("text") or "") for e in events)
    assert "92 MPH" in texts or "90 MPH" in texts
    assert "154 MPH" not in texts


def test_hydration_story_includes_sample_speeds():
    osd = _osd_with_samples()
    ctx = SimpleNamespace(
        dashcam_osd_context=osd,
        telemetry=SimpleNamespace(
            location_city="Logandale",
            location_state="California",
            location_display="Logandale, California",
            location_road=None,
            gazetteer_place_name=None,
            padus_unit_name=None,
            mid_lat=39.37634,
            mid_lon=-122.19277,
            max_speed_mph=92.0,
            points=[],
        ),
        telemetry_data=None,
        vision_context={},
        audio_context={
            "music_detected": True,
            "music_artist": "Fetty Wap",
            "music_title": "The Truth",
            "music_genre": "Rap/Hip Hop",
        },
        video_intelligence=None,
        video_intelligence_context=None,
        visual_recognition=None,
        filename="clip.mp4",
        thumbnail_category="automotive",
        hydration_payload={},
        ai_transcript="",
    )
    story = build_hydration_story_text(ctx, max_chars=900)
    assert "HUD speed samples" in story
    assert "92 MPH" in story
    assert "Fetty Wap" in story or "The Truth" in story


def test_m8_brief_and_must_use_prefer_sample_speeds():
    osd = _osd_with_samples()
    osd["max_speed_mph"] = 154.0  # poisoned peak
    sg = {
        "platforms": ["youtube"],
        "hydration_story": "Logandale run with Fetty Wap.",
        "dashcam_osd": {
            "max_speed_mph": 154.0,
            "speed_series": osd["speed_series"],
            "driver_name": "C Walker",
            "first_seen": {"date": "2025-03-01", "time": "15:13:18"},
            "last_seen": {"date": "2025-03-01", "time": "15:15:50"},
        },
        "geo": {"city": "Logandale", "state": "CA"},
        "music": {"artist": "Fetty Wap", "title": "The Truth"},
        "timeline": [
            {"t_seconds": 10, "kind": "osd_speed_beat", "text": "92 MPH"},
            {"t_seconds": 86, "kind": "osd_speed_beat", "text": "87 MPH"},
            {"t_seconds": 112, "kind": "osd_speed_beat", "text": "90 MPH"},
        ],
    }
    brief = _build_hydration_timeline_brief(sg)
    assert "TIMELINE SPINE" in brief
    assert "92mph" in brief.replace(" ", "").lower() or "92 MPH" in brief
    assert "peak=154" not in brief.replace(" ", "")
    assert "peak=92" in brief.replace(" ", "") or "peak=90" in brief.replace(" ", "")
    must = build_must_use_shortlist(sg)
    assert any("MPH" in t for t in must)
    assert not any(t.startswith("154") for t in must)


def test_scene_graph_osd_max_is_trusted_not_raw_spike():
    from stages.m8_engine import build_scene_graph

    osd = _osd_with_samples()
    osd["max_speed_mph"] = 154.0
    ctx = SimpleNamespace(
        upload_id="test-upload",
        user_id="test-user",
        audio_context={},
        vision_context={},
        video_understanding={},
        video_intelligence=None,
        video_intelligence_context=None,
        visual_recognition=None,
        video_info={"duration": 150},
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context=osd,
        trill=None,
        trill_score=None,
        platforms=["youtube"],
        filename="clip.mp4",
        output_artifacts={},
        hydration_payload={},
        ai_transcript="",
        entitlements=None,
        thumbnail_category="automotive",
        fusion_context=None,
        content_signals=None,
        trend_intel_context=None,
        user_settings={},
    )
    sg = build_scene_graph(ctx, "automotive")
    osd_sg = sg.get("dashcam_osd") or {}
    assert float(osd_sg.get("max_speed_mph") or 0) <= 95
    assert float(osd_sg.get("max_speed_raw_mph") or 0) == 154.0
    assert "series" in str(osd_sg.get("max_speed_source") or "")


def test_expanded_style_tone_voice_allowlists():
    assert "freestyle" in M8_CAPTION_STYLES
    assert "documentary" in M8_CAPTION_TONES
    assert "passenger" in M8_CAPTION_VOICES


def test_m8_prompt_includes_hydration_brief():
    osd = _osd_with_samples()
    ctx = SimpleNamespace(
        user_settings={"captionStyle": "freestyle", "captionTone": "documentary", "captionVoice": "passenger"},
        audio_context={},
        vision_context={},
        video_understanding={},
        video_intelligence=None,
        video_intelligence_context=None,
        visual_recognition=None,
        video_info={"duration": 150},
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context=osd,
        trill=None,
        trill_score=None,
        platforms=["youtube"],
        filename="clip.mp4",
        output_artifacts={},
        hydration_payload={},
        ai_transcript="",
        entitlements=None,
        thumbnail_category="automotive",
        fusion_context=None,
        content_signals=None,
    )
    sg = {
        "platforms": ["youtube"],
        "category": "automotive",
        "hydration_story": "Fast run near Logandale vibing to Fetty Wap The Truth.",
        "dashcam_osd": {
            "max_speed_mph": osd["max_speed_mph"],
            "speed_series": osd["speed_series"],
            "driver_name": "C Walker",
        },
        "geo": {"city": "Logandale", "state": "CA"},
        "music": {"artist": "Fetty Wap", "title": "The Truth"},
        "timeline": [
            {"t_seconds": 10, "kind": "osd_speed_beat", "text": "92 MPH"},
            {"t_seconds": 86, "kind": "osd_speed_beat", "text": "87 MPH"},
        ],
    }
    prompt = _build_m8_prompt(
        ctx,
        sg,
        "automotive",
        "freestyle",
        "documentary",
        "mixed",
        5,
        True,
        True,
        True,
        caption_voice_ui="passenger",
    )
    assert "HYDRATION + TIMELINE BRIEF" in prompt
    assert "FREESTYLE" in prompt
    assert "high-energy first-person dashcam" in prompt.lower() or "generic wrappers" in prompt.lower()


def test_vi_label_dump_never_enters_title_or_caption_anchor():
    """Tumwater-style VI dumps must not pollute publishable title/caption."""
    from services.hydration_enforcer import (
        build_anchor_phrase,
        build_title_anchor_phrase,
        collect_evidence,
        enforce_hydration,
        scrub_machine_publish_dump,
    )

    dump = (
        "Video Intelligence — labels: land vehicle, motor vehicle, driving, car, "
        "highway, vehicle, road trip, road, lane, windshield | objects: car (12-13s), "
        "car (0-6s),"
    )
    assert scrub_machine_publish_dump(
        f"Captured at 128 MPH, near Tumwater, WA. {dump}"
    ) == "Captured at 128 MPH, near Tumwater, WA"

    from services.hydration_enforcer import _is_machine_label_dump

    assert _is_machine_label_dump(dump)

    ctx = SimpleNamespace(
        telemetry=SimpleNamespace(
            max_speed_mph=128.0,
            avg_speed_mph=90.0,
            location_city="Tumwater",
            location_state="Washington",
            location_country="US",
            location_road="I 5",
            gazetteer_place_name="Tumwater",
            padus_unit_name=None,
            near_padus=False,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={},
        trill=None,
        trill_score=None,
        ai_transcript="",
        video_intelligence={
            "summary_text": dump,
            "machine_summary": dump,
            "top_labels": [
                "land vehicle (0.9)",
                "motor vehicle (0.9)",
                "car (0.8)",
                "highway (0.7)",
            ],
            "object_tracks": [
                {"description": "car", "start_s": 12, "end_s": 13, "confidence": 0.9},
            ],
            "logos": [],
            "on_screen_text": [],
        },
        video_intelligence_context={"summary_text": dump, "machine_summary": dump},
        video_understanding={"scene_description": "Fast run near Tumwater on I-5."},
        filename="clip.mp4",
        thumbnail_category="automotive",
        ai_title=f"Captured at 128 MPH, near Tumwater, WA. {dump}",
        ai_caption=f"Captured at 128 MPH, near Tumwater, WA. {dump}",
        ai_hashtags=["66mphcwalker", "car", "tumwaterwa", "i5"],
        m8_platform_captions={},
        m8_platform_titles={},
        m8_platform_hashtags={},
        output_artifacts={},
        upload_id="test-tumwater",
    )
    pool = collect_evidence(ctx)
    assert pool.video_summary_phrase is None or not _is_machine_label_dump(
        pool.video_summary_phrase or ""
    )
    caption_anchor = build_anchor_phrase(pool, ctx)
    assert "Video Intelligence" not in caption_anchor
    assert "labels:" not in caption_anchor.lower()
    assert "objects:" not in caption_anchor.lower()
    assert "land vehicle" not in caption_anchor.lower()
    assert "windshield" not in caption_anchor.lower()
    assert "128" in caption_anchor or "Tumwater" in caption_anchor

    title_anchor = build_title_anchor_phrase(pool, ctx)
    assert "Video Intelligence" not in title_anchor
    assert "Captured at" not in title_anchor
    assert "labels:" not in title_anchor.lower()
    assert "128 MPH" not in title_anchor

    report = enforce_hydration(ctx)
    assert "Video Intelligence" not in (ctx.ai_title or "")
    assert "Video Intelligence" not in (ctx.ai_caption or "")
    assert "labels:" not in (ctx.ai_title or "").lower()
    assert "labels:" not in (ctx.ai_caption or "").lower()
    assert "Captured at" not in (ctx.ai_title or "")
    assert "66mphcwalker" not in [t.lower() for t in (ctx.ai_hashtags or [])]
    assert "car" not in [t.lower() for t in (ctx.ai_hashtags or [])]
    assert report.get("title_anchor")


def test_hashtags_drop_ocr_mashups_and_generic_taxonomy():
    """HUD OCR + Vision filler must not publish; geo/brands must survive."""
    from services.hydration_enforcer import (
        _scrub_leaked_junk_hashtags,
        build_evidence_hashtags,
        collect_evidence,
        enforce_hydration,
    )
    from core.vision_labels import is_junk_hashtag_body

    junk = [
        "66mphcwalker",
        "7omphcwalker",
        "73mphcwalker",
        "20250303111931am46950603",
        "nature",
        "modeoftransport",
        "horizon",
        "car",
        "vehicle",
    ]
    for j in junk:
        assert is_junk_hashtag_body(j), j
    scrubbed = _scrub_leaked_junk_hashtags(
        junk + ["i5", "tumwaterwa", "costco", "matsoninc", "cwalker", "tripledigits"]
    )
    low = {t.lower() for t in scrubbed}
    assert "i5" in low
    assert "tumwaterwa" in low
    assert "costco" in low
    assert "matsoninc" in low
    assert "cwalker" in low
    assert "tripledigits" in low
    for j in junk:
        assert j.lower() not in low

    ctx = SimpleNamespace(
        telemetry=SimpleNamespace(
            max_speed_mph=128.0,
            avg_speed_mph=90.0,
            location_city="Tumwater",
            location_state="Washington",
            location_country="US",
            location_road="I 5",
            gazetteer_place_name="Tumwater",
            padus_unit_name=None,
            near_padus=False,
        ),
        telemetry_data=None,
        dashcam_osd_context={"driver_name": "C Walker", "max_speed_mph": 92.0},
        vision_context={"logo_names": ["Costco"], "landmark_names": [], "ocr_text": ""},
        audio_context={},
        trill=SimpleNamespace(bucket="gloryBoy", score=100.0),
        trill_score=None,
        ai_transcript="",
        video_intelligence={
            "logos": [{"description": "Matson Inc"}],
            "on_screen_text": [
                {"text": "66MPH C Walker", "confidence": 0.9},
                {"text": "2025/03/03 11:19:31 AM 46.950603", "confidence": 0.8},
                {"text": "COSTCO", "confidence": 0.7},
            ],
            "top_labels": ["nature", "Mode of transport", "horizon", "car"],
            "summary_text": "",
        },
        video_intelligence_context={},
        video_understanding={},
        filename="clip.mp4",
        thumbnail_category="automotive",
        ai_title="Fast run near Tumwater",
        ai_caption="Fast run near Tumwater on I-5.",
        ai_hashtags=list(junk) + ["gloryboytour", "sendit"],
        m8_platform_captions={},
        m8_platform_titles={},
        m8_platform_hashtags={},
        output_artifacts={},
        upload_id="hash-quality",
    )
    pool = collect_evidence(ctx)
    tags = [t.lower() for t in build_evidence_hashtags(pool, max_extra=16)]
    assert "i5" in tags or "i5tumwater" in tags or "tumwateri5" in tags
    assert "tumwaterwa" in tags or "tumwater" in tags
    assert "costco" in tags
    assert "matsoninc" in tags
    for j in ("nature", "modeoftransport", "horizon", "car", "66mphcwalker"):
        assert j not in tags

    enforce_hydration(ctx)
    final = {t.lower() for t in (ctx.ai_hashtags or [])}
    for j in junk:
        assert j.lower() not in final
    assert "costco" in final or "matsoninc" in final
    assert "tumwaterwa" in final or "tumwater" in final


def test_no_color_nature_or_vague_category_copy():
    from core.vision_labels import is_junk_hashtag_body, is_vague_taxonomy_copy
    from services.hydration_enforcer import _is_generic_caption, _scrub_leaked_junk_hashtags

    for tag in (
        "nature",
        "horizon",
        "modeoftransport",
        "car",
        "blue",
        "green",
        "yellow",
        "colorful",
        "scenery",
        "landscape",
        "aesthetic",
        "vibes",
        "bluesky",
        "greentrees",
        "naturevibes",
    ):
        assert is_junk_hashtag_body(tag), tag

    for copy in (
        "Nature",
        "Horizon",
        "Mode of transport",
        "Blue skies",
        "Green trees",
        "Nature views",
        "Beautiful landscape",
        "Outdoor scenery",
    ):
        assert is_vague_taxonomy_copy(copy), copy
        assert _is_generic_caption(copy) or len(copy) < 12

    kept = _scrub_leaked_junk_hashtags(
        [
            "nature",
            "blue",
            "horizon",
            "modeoftransport",
            "i5",
            "tumwaterwa",
            "costco",
            "cwalker",
        ]
    )
    assert {t.lower() for t in kept} == {"i5", "tumwaterwa", "costco", "cwalker"}
