"""Stock motorsport / hype openers — shared patterns, ranking, hydration labels."""

from __future__ import annotations

from types import SimpleNamespace

from core.caption_creative import interaction_contract, title_lead_facet
from core.prose_cliche_patterns import (
    hydration_cliche_patterns,
    is_sole_motorsport_opener,
    prompt_motorsport_ban_line,
    ranking_cliche_patterns,
)
from services.hydration_enforcer import _is_generic_caption, enforce_hydration
from services.m8_grounding_pass import is_formula_stub_caption
from stages.m8_engine import _penalize_generic, _quality_gate_penalty


def test_start_your_engines_is_generic_not_receipt_stub():
    assert is_formula_stub_caption("Start your engines!") is False
    assert _is_generic_caption("Start your engines!") is True
    assert _is_generic_caption("Buckle up for Seattle") is True
    assert _is_generic_caption("Fasten your seatbelts") is True
    grounded = (
        "Start your engines in Iron County, Utah as C Walker hits 78 MPH to The Weeknd"
    )
    assert _penalize_generic(grounded) >= 8.0
    assert _penalize_generic("Start your engines!") >= 8.0
    assert _penalize_generic("50 MPH — Seattle rain braid with The Weeknd") == 0.0


def test_shared_prose_cliche_module_feeds_hydration_and_ranking():
    hyd = {p.pattern for p in hydration_cliche_patterns()}
    rank = {p.pattern for p in ranking_cliche_patterns()}
    assert r"\bstart your engines?\b" in hyd
    assert r"\bstart your engines?\b" in rank
    assert "Start your engines" in prompt_motorsport_ban_line()
    assert is_sole_motorsport_opener("Start your engines!") is True
    assert is_sole_motorsport_opener(grounded := (
        "Start your engines in Iron County, Utah as C Walker hits 78 MPH"
    )) is False


def test_quality_gate_hard_rejects_sole_motorsport_under_persona():
    soft = _quality_gate_penalty(
        "instagram",
        "Start your engines!",
        "A longer caption about the drive through Seattle with concrete place nouns here.",
        persona_required=False,
    )
    hard = _quality_gate_penalty(
        "instagram",
        "Start your engines!",
        "A longer caption about the drive through Seattle with concrete place nouns here.",
        persona_required=True,
    )
    assert hard >= soft + 40.0


def test_high_heat_interaction_bans_motorsport_for_any_style():
    punchy = interaction_contract("punchy", "hype", "teacher")
    freestyle = interaction_contract("freestyle", "hype", "teacher")
    assert "Start your engines" in punchy
    assert "Start your engines" in freestyle


def test_freestyle_title_lead_is_free_not_forced_geo():
    assert title_lead_facet("freestyle", "hype") == "free/any-evidence"
    assert title_lead_facet("factual") == "place/geo"


def test_enforce_hydration_marks_engines_as_generic_rejected():
    """Full hydrate path: sole motorsport opener under freestyle×hype×teacher."""
    ctx = SimpleNamespace(
        user_settings={
            "captionStyle": "freestyle",
            "captionTone": "hype",
            "captionVoice": "teacher",
        },
        audio_context={
            "music_detected": True,
            "music_artist": "The Weeknd",
            "music_title": "Enjoy The Show",
        },
        vision_context={"labels": ["car", "road"], "ocr": "MSC"},
        video_understanding={"scene_description": "Driving near Seattle on I-90."},
        video_intelligence={"object_tracks": [{"description": "car"}]},
        video_intelligence_context={},
        visual_recognition=None,
        video_info={"duration": 90},
        telemetry=SimpleNamespace(
            location_city="Seattle",
            location_state="Washington",
            location_road="I 90",
            max_speed_mph=50.0,
            avg_speed_mph=40.0,
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        trill=None,
        trill_score=None,
        platforms=["instagram"],
        filename="cam.mp4",
        thumbnail_category="automotive",
        fusion_context=None,
        content_signals=None,
        ai_transcript="",
        entitlements=None,
        hydration_payload={},
        output_artifacts={},
        upload_id="engines-generic-wipe",
        ai_title="Start your engines!",
        ai_caption="Start your engines!",
        ai_hashtags=["seattle"],
        m8_platform_titles={"instagram": "Start your engines!"},
        m8_platform_captions={
            "instagram": "Start your engines! Vague hype with no place tokens yet."
        },
        m8_platform_hashtags={"instagram": ["seattle"]},
    )
    # Point telemetry_data at same object used by evidence pool helpers.
    ctx.telemetry_data = ctx.telemetry
    report = enforce_hydration(ctx)
    assert report.get("persona_required") is True
    assert report.get("generic_rejected") is True
    assert report.get("receipt_rejected") is False
    assert report.get("wipe_reason") == "generic_rejected"
    assert is_formula_stub_caption(report.get("title_before") or "") is False
    final = (ctx.m8_platform_titles or {}).get("instagram") or ctx.ai_title or ""
    assert final.strip() != "Start your engines!"
    assert not is_sole_motorsport_opener(final)
