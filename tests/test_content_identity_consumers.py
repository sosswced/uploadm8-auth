"""Identity-driven consumers: category adapter, hero-fact headline, overrides.

Regression coverage for the plan's Phase 2 exit gates:
  * an OCR speed spike never becomes a headline on a slower verified drive;
  * non-driving footage (gardening) never gets an MPH headline;
  * novel content produces a grounded headline;
  * keyword scanning is gone — detection reads content_identity_v1.
"""

from __future__ import annotations

from types import SimpleNamespace

from stages.thumbnail_stage import (
    _concrete_thumbnail_headline,
    _detect_category,
    _hero_fact_headlines,
    effective_thumbnail_category,
)


def _ctx(**overrides) -> SimpleNamespace:
    base = dict(
        upload_id="cidc-1",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        title=None,
        caption=None,
        filename="clip.mp4",
        output_artifacts={},
        get_effective_title=lambda: "",
        get_effective_caption=lambda: "",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tel(mph: float) -> SimpleNamespace:
    return SimpleNamespace(
        max_speed_mph=mph,
        avg_speed_mph=mph * 0.7,
        location_city="Logandale",
        location_state="CA",
        location_road=None,
        location_display="Logandale, CA",
        location_start_display=None,
        gazetteer_place_name=None,
        padus_unit_name=None,
    )


def _gardening_ctx() -> SimpleNamespace:
    return _ctx(
        video_understanding={
            "scene_description": "A gardener harvests ripe roma tomatoes from raised garden beds."
        },
        vision_context={
            "label_names": ["tomato", "plant", "garden bed", "vegetable"],
            "face_count": 0,
        },
        video_intelligence_context={"labels": ["tomato", "gardening", "harvest"]},
    )


# ── Category adapter (identity-driven, no keywords) ──────────────────────


def test_detect_category_driving_from_sensor_domain():
    ctx = _ctx(telemetry=_tel(154.0))
    assert _detect_category(ctx) == "automotive"


def test_detect_category_gardening_from_agreement():
    assert _detect_category(_gardening_ctx()) == "gardening"


def test_detect_category_llm_tag_direct_match():
    # A worker-resolved identity with an LLM domain tag maps directly.
    ctx = _ctx(
        output_artifacts={
            "content_identity_v1": {
                "version": 1,
                "subject": "sourdough loaf scoring before bake",
                "domain_tags": [{"tag": "food", "confidence": 0.9}],
                "hero_facts": [],
                "do_not_invent": [],
                "confidence": "high",
            }
        }
    )
    assert _detect_category(ctx) == "food"


def test_detect_category_novel_content_stays_general():
    ctx = _ctx(
        video_understanding={
            "scene_description": "An artisan shapes molten material inside a workshop."
        },
    )
    assert _detect_category(ctx) == "general"


def test_effective_category_general_niche_means_auto():
    us = {"thumbnail_studio_default_strategy": {"audience_niche": "general"}}
    assert effective_thumbnail_category(us, "gardening") == "gardening"


def test_effective_category_specific_niche_overrides():
    us = {"thumbnail_studio_default_strategy": {"audience_niche": "gaming"}}
    assert effective_thumbnail_category(us, "gardening") == "gaming"


# ── Hero-fact headline selector ──────────────────────────────────────────


def test_ocr_spike_never_headlines_over_verified_peak():
    # Telemetry says 68; Vision OCR spikes to 200. Headline must carry 68, never 200.
    ctx = _ctx(
        telemetry=_tel(68.0),
        vision_context={"ocr_text": "200 MPH", "label_names": ["highway", "road"], "face_count": 0},
        video_understanding={"scene_description": "Dashcam highway drive at dusk."},
    )
    headline = _concrete_thumbnail_headline(ctx, "automotive")
    assert "200" not in headline
    candidates = _hero_fact_headlines(ctx, "automotive")
    assert any("68 MPH" in c for c in candidates)
    assert not any("200" in c for c in candidates)


def test_gardening_never_gets_mph_headline():
    headline = _concrete_thumbnail_headline(_gardening_ctx(), "gardening")
    assert "MPH" not in headline.upper()
    assert headline  # still concrete, not empty


def test_speed_requires_driving_domain_even_when_verified():
    # Verified 40 MPH exists but content is not a drive (e.g. GPS in a backpack
    # during a garden timelapse) — the category gate keeps MPH off the cover.
    ctx = _gardening_ctx()
    ctx.telemetry = _tel(40.0)
    candidates = _hero_fact_headlines(ctx, "gardening")
    assert not any("MPH" in c.upper() for c in candidates)


def test_novel_content_headline_is_grounded():
    ctx = _ctx(
        video_understanding={
            "scene_description": "An artisan shapes molten glass into a vase inside a hot workshop."
        },
        vision_context={"label_names": ["glass", "furnace"], "face_count": 1},
    )
    headline = _concrete_thumbnail_headline(ctx, "general")
    assert headline
    assert "MPH" not in headline.upper()


def test_empty_context_falls_back_to_category_fallback():
    headline = _concrete_thumbnail_headline(_ctx(filename=""), "gardening")
    assert headline == "GARDEN UPDATE"
