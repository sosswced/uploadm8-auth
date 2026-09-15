"""Publish pack + logo-on-dashcam paint eligibility (Jordan Kuwait Bank class)."""

from __future__ import annotations

from types import SimpleNamespace

from core.publish_pack import (
    PUBLISH_PACK_ARTIFACT,
    attach_publish_pack,
    build_publish_pack,
    is_paintable_pack_headline,
)
from stages.pikzels_api import _build_pikzels_v2_prompt
from stages.thumbnail_stage import _concrete_thumbnail_headline, _sanitize_thumbnail_brief


def _dashcam_ctx(**kwargs):
    identity = {
        "version": 1,
        "subject": "U-Haul driving in Iron County, Utah",
        "confidence": "high",
        "resolver": "llm+deterministic",
        "hero_facts": [
            {"text": "Iron County, Utah", "class": "place", "score": 3.0},
            {"text": "78 MPH RUN", "class": "speed", "score": 2.5},
            {"text": "Chief Keef", "class": "music", "score": 2.0},
            {"text": "Jordan Kuwait Bank", "class": "logo", "score": 1.0},
        ],
        "domain_tags": [{"tag": "automotive", "confidence": 0.9}],
    }
    base = dict(
        filename="20250224_0016_CAM.MP4",
        title="20250224_0016_CAM.MP4",
        caption="",
        ai_title="",
        ai_caption="",
        ai_transcript="",
        vision_context={
            "logo_names": ["Jordan Kuwait Bank", "Real United", "U-Haul"],
            "ocr_text": "ESCORT 56MPH C Walker",
            "landmarks": [],
            "labels": [],
        },
        audio_context={"music_artist": "Chief Keef", "music_title": "Oh My Goodness"},
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={"max_speed_mph": 77.6, "driver_name": "C Walker"},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        visual_recognition={},
        hydration_payload={
            "category": "automotive",
            "anchor_phrase": "Captured at 78 MPH, on SR 20, near Iron County, UT, with Chief Keef",
            "evidence": {
                "geo": {"display": "Iron County, Utah", "road": "SR 20", "city": "Iron County", "state": "UT"},
                "osd": {"max_speed_mph": 77.6},
                "music": {"artist": "Chief Keef", "title": "Oh My Goodness"},
            },
            "signal_hashtags": [
                "chiefkeef",
                "ironcounty",
                "utah",
                "jordankuwaitbank",
                "realunited",
            ],
        },
        thumbnail_category="automotive",
        output_artifacts={"content_identity_v1": identity},
        content_identity=identity,
        user_settings={},
        get_effective_title=lambda: "20250224_0016_CAM.MP4",
        get_effective_caption=lambda: "",
        get_thumbnail_brief_vars=lambda category=None: {
            "fusion_summary": "brand cues: U-Haul, Jordan Kuwait Bank",
            "hydration_story": "peak speed about 78 MPH; brand cues: Jordan Kuwait Bank",
            "geo_context": "Iron County, Utah",
            "osd_context": "peak speed: 77.6 mph",
            "music_context": "Chief Keef — Oh My Goodness",
            "speech_context": "",
            "trill_context": "",
            "signal_hashtags": "jordankuwaitbank",
        },
    )
    base.update(kwargs)
    ctx = SimpleNamespace(**base)

    # Driving evidence stub via speed consensus artifact path
    class _Tel:
        max_speed_mph = 77.6
        location_city = "Iron County"
        location_state = "UT"
        location_road = "SR 20"

    if ctx.telemetry is None:
        ctx.telemetry = _Tel()
        ctx.telemetry_data = _Tel()
    return ctx


def test_publish_pack_prefers_speed_or_place_not_logo(monkeypatch):
    ctx = _dashcam_ctx()

    monkeypatch.setattr(
        "core.driving_evidence.has_driving_evidence",
        lambda _ctx: True,
    )
    monkeypatch.setattr(
        "core.speed_consensus.publishable_peak_mph",
        lambda _ctx: 78.0,
    )
    pack = build_publish_pack(ctx)
    assert "JORDAN" not in (pack.get("hook_line") or "").upper()
    assert "KUWAIT" not in (pack.get("hook_line") or "").upper()
    # Composition-first: speed earns paint; place alone would be paint_policy=none.
    assert pack.get("paint_policy") == "hook_only"
    assert pack.get("hook_class") == "speed"
    assert "78" in (pack.get("hook_line") or "")
    assert (pack.get("visual_brief") or {}).get("text") == "hook_lower_third"
    seeds = [str(s).lower() for s in (pack.get("hashtag_seeds") or [])]
    assert "jordankuwaitbank" not in seeds


def test_publish_pack_place_only_is_composition_not_paint(monkeypatch):
    ctx = _dashcam_ctx()
    monkeypatch.setattr("core.driving_evidence.has_driving_evidence", lambda _ctx: True)
    monkeypatch.setattr("core.speed_consensus.publishable_peak_mph", lambda _ctx: 0.0)
    pack = build_publish_pack(ctx)
    assert pack.get("paint_policy") == "none"
    assert (pack.get("visual_brief") or {}).get("text") == "none"
    if pack.get("hook_line"):
        assert not is_paintable_pack_headline(pack["hook_line"], pack)


def test_concrete_headline_rejects_jordan_kuwait_bank(monkeypatch):
    ctx = _dashcam_ctx()
    monkeypatch.setattr("core.driving_evidence.has_driving_evidence", lambda _ctx: True)
    monkeypatch.setattr("core.speed_consensus.publishable_peak_mph", lambda _ctx: 78.0)
    attach_publish_pack(ctx)
    headline = _concrete_thumbnail_headline(ctx, "automotive")
    assert "JORDAN" not in headline.upper()
    assert "KUWAIT" not in headline.upper()
    assert "BANK" not in headline.upper()


def test_sanitize_brief_no_logo_options(monkeypatch):
    ctx = _dashcam_ctx()
    monkeypatch.setattr("core.driving_evidence.has_driving_evidence", lambda _ctx: True)
    monkeypatch.setattr("core.speed_consensus.publishable_peak_mph", lambda _ctx: 78.0)
    attach_publish_pack(ctx)
    brief = _sanitize_thumbnail_brief(
        ctx,
        {
            "selected_headline": "JORDAN KUWAIT BANK",
            "headline_options": ["JORDAN KUWAIT BANK", "REAL UNITED", "KLAN KOSOVA"],
        },
        "automotive",
    )
    selected = str(brief.get("selected_headline") or "").upper()
    assert "JORDAN" not in selected
    assert "KUWAIT" not in selected
    for opt in brief.get("headline_options") or []:
        assert "JORDAN" not in str(opt).upper()
        assert "KUWAIT" not in str(opt).upper()


def test_pikzels_prompt_no_text_for_logo_bleed(monkeypatch):
    ctx = _dashcam_ctx()
    monkeypatch.setattr("core.driving_evidence.has_driving_evidence", lambda _ctx: True)
    monkeypatch.setattr("core.speed_consensus.publishable_peak_mph", lambda _ctx: 78.0)
    pack = attach_publish_pack(ctx)
    brief = {
        "selected_headline": "JORDAN KUWAIT BANK",
        "_uploadm8_paint_policy": pack.get("paint_policy") or "none",
        "_uploadm8_pack_subject": pack.get("subject"),
        "_uploadm8_hook_line": pack.get("hook_line"),
        "pack_subject": pack.get("subject"),
        "hook_line": pack.get("hook_line"),
        "_uploadm8_dashcam_pov": True,
    }
    prompt = _build_pikzels_v2_prompt(brief, category="automotive", platform="instagram")
    assert 'reading "JORDAN KUWAIT BANK"' not in prompt
    assert (
        "CREATIVE COMPOSITION MODE" in prompt
        or "STRICT NO-TEXT" in prompt
        or "78 MPH" in prompt
    )


def test_is_paintable_rejects_logo_without_subject_agree():
    pack = {
        "subject": "U-Haul driving in Iron County, Utah",
        "hook_line": "78 MPH",
        "paint_policy": "hook_only",
        "hook_class": "speed",
        "hashtag_seeds": ["ironcounty", "utah"],
    }
    assert not is_paintable_pack_headline("JORDAN KUWAIT BANK", pack, fact_class="logo")
    assert is_paintable_pack_headline("78 MPH", pack, fact_class="speed")
    assert not is_paintable_pack_headline("FEDERAL WAY WASHINGTON", pack, fact_class="place")
    assert not is_paintable_pack_headline("LOCATION FEDERAL WAY WASHINGTON", pack)


def test_attach_publish_pack_artifact():
    ctx = _dashcam_ctx()
    pack = attach_publish_pack(ctx)
    assert ctx.output_artifacts.get(PUBLISH_PACK_ARTIFACT) is pack
