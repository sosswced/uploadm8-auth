"""Grounding eval, depth router, place-evidence, and M8 pass-2 unit tests."""

from __future__ import annotations

from types import SimpleNamespace

from core.vision_labels import vision_labels_are_weak
from services.grounding_eval import compute_grounding_score
from services.hydration_enforcer import EvidencePool
from services.m8_grounding_pass import (
    apply_grounding_pass2_to_ranked,
    build_evidence_catalog,
    ensure_must_use_coverage,
    m8_grounding_pass2_enabled,
    synthesize_claims_from_text,
)
from services.multimodal_depth_router import classify_clip_kind, route_multimodal_depth
from services.place_evidence import extract_place_evidence, merge_place_evidence_into_pool


def test_vision_labels_are_weak_generic_only():
    assert vision_labels_are_weak(["outdoor", "vehicle", "person", "sky"])
    assert not vision_labels_are_weak(
        ["outdoor", "vehicle"],
        landmark_names=["Golden Gate Bridge"],
    )
    assert not vision_labels_are_weak(
        ["outdoor"],
        ocr_text="WELCOME TO SANTA MONICA BEACH",
    )
    assert not vision_labels_are_weak(["ferrari", "lighthouse", "pier"])


def test_depth_router_forces_tl_when_vision_weak():
    ctx = SimpleNamespace(
        filename="vacation_clip.mp4",
        thumbnail_category="travel",
        vision_context={
            "label_names": ["outdoor", "sky", "person", "vehicle"],
            "landmark_names": [],
            "logo_names": [],
            "ocr_text": "",
        },
        user_settings={},
        telemetry_data=None,
        dashcam_osd_context={},
        ai_transcript="",
        audio_context={},
        place_evidence={},
        output_artifacts={},
        duration_seconds=45,
    )
    route = route_multimodal_depth(ctx)
    assert route["vision_weak"] is True
    assert route["force_twelvelabs"] is True
    assert "vision_labels_weak" in route["reason"]


def test_depth_router_no_force_with_landmark():
    ctx = SimpleNamespace(
        filename="trip.mp4",
        thumbnail_category="travel",
        vision_context={
            "label_names": ["outdoor", "sky"],
            "landmark_names": ["Statue of Liberty"],
            "logo_names": [],
            "ocr_text": "",
        },
        user_settings={},
        telemetry_data=SimpleNamespace(location_city="New York", points=[]),
        dashcam_osd_context={},
        ai_transcript="",
        audio_context={},
        place_evidence={"landmarks": [{"name": "Statue of Liberty"}]},
        output_artifacts={},
        duration_seconds=30,
    )
    route = route_multimodal_depth(ctx)
    assert route["vision_weak"] is False
    assert route["force_twelvelabs"] is False


def test_classify_dashcam_filename():
    ctx = SimpleNamespace(
        filename="M8_2024_DRIVE.MP4",
        thumbnail_category="general",
        vision_context={},
        duration_seconds=10,
        audio_context={},
        ai_transcript="",
    )
    assert classify_clip_kind(ctx) == "dashcam"


def test_place_evidence_extracts_beach_plate_team():
    ctx = SimpleNamespace(
        vision_context={
            "ocr_text": "SANTA MONICA BEACH\nLakers Store\n7ABC123\nLincoln Memorial",
            "landmark_names": ["Santa Monica Pier"],
            "landmarks": [
                {"description": "Santa Monica Pier", "lat": 34.0094, "lon": -118.4973, "score": 0.9}
            ],
            "logo_names": ["NBA"],
            "label_names": ["outdoor"],
        },
        video_intelligence={
            "on_screen_text": [{"text": "Dodgers Stadium"}],
        },
        audio_context={
            "transcript_structured": {
                "named_entities": {"places": ["Venice Beach"], "organizations": ["Lakers"]},
            }
        },
        telemetry_data=None,
        place_evidence=None,
    )
    report = extract_place_evidence(ctx)
    assert any("Santa Monica" in p or "Pier" in p for p in report["places"])
    assert report["beaches"]
    assert report["monuments"] or any("Memorial" in m for m in report.get("monuments") or [])
    assert report["license_plates"]
    assert report["sports_teams"] or report["stadiums"]
    assert "vision_landmark" in report["sources"]


def test_merge_place_evidence_into_pool():
    pool = EvidencePool()
    merge_place_evidence_into_pool(
        pool,
        {
            "places": ["Bondi Beach"],
            "beaches": ["Bondi Beach"],
            "monuments": ["Sydney Opera House"],
            "stadiums": [],
            "license_plates": ["ABC123"],
            "sports_teams": ["Lakers"],
            "sources": ["ocr"],
            "geocode_from_landmark": {"location_display": "Bondi, NSW, Australia"},
        },
    )
    assert "Bondi Beach" in pool.vision_landmarks
    assert pool.place_beaches == ["Bondi Beach"]
    assert pool.sports_teams == ["Lakers"]
    assert pool.city and "Bondi" in pool.city


def test_grounding_score_hits_place():
    pool = EvidencePool()
    pool.city = "Guadalupe"
    pool.vision_landmarks = ["Guadalupe Dunes"]
    pool.max_speed_mph = 46.0
    scored = compute_grounding_score(
        text="Rolling through Guadalupe near the Dunes at 46 mph.",
        pool=pool,
        evidence_present=True,
    )
    assert scored["hit_count"] >= 2
    assert scored["grounding_score"] > 0.3
    assert scored["lanes"]["geo"] is True


def test_grounding_score_zero_when_generic():
    pool = EvidencePool()
    pool.city = "Guadalupe"
    pool.vision_landmarks = ["Guadalupe Dunes"]
    scored = compute_grounding_score(
        text="Cruise under vast skies! Endless horizons await.",
        pool=pool,
        evidence_present=True,
    )
    assert scored["grounding_score"] < 0.35
    assert scored["hit_count"] == 0


def test_m8_pass2_enabled_by_default():
    assert m8_grounding_pass2_enabled({}) is True


def test_evidence_catalog_and_synthesize_claims():
    scene = {
        "geo": {"city": "Guadalupe", "state": "CA"},
        "vision": {"landmarks": ["Guadalupe Dunes"], "logos": []},
        "place_evidence": {"beaches": [], "monuments": [], "sports_teams": []},
    }
    must_use = ["46 MPH", "Guadalupe, CA"]
    catalog = build_evidence_catalog(scene, must_use)
    assert len(catalog) >= 2
    claims = synthesize_claims_from_text(
        "Rolling at 46 MPH near Guadalupe, CA by the Dunes.",
        catalog,
    )
    assert claims
    assert claims[0]["evidence_ids"]


def test_ensure_must_use_injects():
    text, injected = ensure_must_use_coverage(
        "Cruise under vast skies!",
        ["46 MPH", "Guadalupe, CA"],
        min_required=2,
    )
    assert injected is True
    assert "46 MPH" in text or "Guadalupe" in text


def test_apply_grounding_pass2_to_ranked():
    scene = {
        "geo": {"city": "Guadalupe", "state": "CA", "max_speed_mph": 46},
        "vision": {"landmarks": ["Guadalupe Dunes"], "logos": []},
        "place_evidence": {},
    }
    ranked = {
        "must_use": ["46 MPH", "Guadalupe, CA"],
        "platforms": {
            "tiktok": {
                "winner": {
                    "title": None,
                    "caption": "Cruise under vast skies! Endless horizons await.",
                    "hashtags": ["travel"],
                    "claims": [],
                },
                "variants_ranked": [],
            }
        },
    }
    out = apply_grounding_pass2_to_ranked(ranked, scene)
    winner = out["platforms"]["tiktok"]["winner"]
    assert "Guadalupe" in winner["caption"] or "46" in winner["caption"]
    assert out["grounding_pass2"]["must_use_injected"] >= 1
    assert out["evidence_catalog"]


def test_gold_set_fixture_grounding_gate():
    """CI gate: gold captions beat min score; bad captions stay below."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "fixtures" / "grounding_gold_set.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    min_score = float(data["min_grounding_score"])
    assert data["clips"], "gold set must include at least one clip"

    for clip in data["clips"]:
        pool = EvidencePool()
        for k, v in (clip.get("evidence") or {}).items():
            setattr(pool, k, v)
        gold = compute_grounding_score(
            text=clip["gold_caption"],
            pool=pool,
            evidence_present=True,
        )
        bad = compute_grounding_score(
            text=clip["bad_caption"],
            pool=pool,
            evidence_present=True,
        )
        assert gold["grounding_score"] >= min_score, clip["id"]
        assert bad["grounding_score"] < min_score, clip["id"]
        assert gold["grounding_score"] > bad["grounding_score"], clip["id"]


def test_upload_qa_answers_with_citations():
    from services.upload_qa import answer_from_evidence

    arts = {
        "hydration_report": {
            "evidence": {"city": "Santa Monica", "max_speed_mph": 42},
            "hydration_story": "Drive along Santa Monica Beach",
        },
        "place_evidence_v1": {
            "beaches": [{"name": "Santa Monica Beach"}],
            "landmarks": [],
        },
        "shot_list_v1": [{"summary": "Coastal road at golden hour"}],
    }
    out = answer_from_evidence("Where was this filmed?", arts)
    assert out["grounding_ok"] is True
    assert out["evidence_ids"]
    assert "Santa Monica" in out["answer"]
    assert out["citations"]


def test_whisper_aic_exempt_alone_charges_zero():
    from stages.ai_service_costs import (
        AIC_BILLING_EXEMPT,
        SERVICE_WEIGHTS,
        compute_aic_service_charge,
    )

    assert "audio_whisper" in AIC_BILLING_EXEMPT
    assert SERVICE_WEIGHTS["audio_whisper"] == 0
    aic = compute_aic_service_charge(
        enabled={"audio_whisper"},
        duration_seconds=120,
        max_caption_frames=4,
        num_thumbnails=1,
        weights=SERVICE_WEIGHTS,
    )
    assert aic == 0
    mixed = compute_aic_service_charge(
        enabled={"audio_whisper", "caption_llm"},
        duration_seconds=60,
        max_caption_frames=4,
        num_thumbnails=1,
        weights=SERVICE_WEIGHTS,
    )
    assert mixed >= SERVICE_WEIGHTS["caption_llm"]
