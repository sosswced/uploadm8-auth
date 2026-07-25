"""Content identity consensus: open-vocabulary fusion, grounding, speed gating."""

from __future__ import annotations

from types import SimpleNamespace

from core.content_identity import (
    CONTENT_IDENTITY_ARTIFACT,
    build_content_identity,
    build_identity_evidence,
    fact_is_grounded,
    get_content_identity,
    merge_llm_identity,
    top_domain_tag,
)
from services.content_identity_llm import build_identity_prompt, parse_identity_response


def _ctx(**overrides) -> SimpleNamespace:
    base = dict(
        upload_id="cid-1",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        filename="clip.mp4",
        output_artifacts={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tel(mph: float, road: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        max_speed_mph=mph,
        avg_speed_mph=mph * 0.7,
        location_city="Logandale",
        location_state="CA",
        location_road=road or None,
        location_display="Logandale, CA",
        location_start_display=None,
        gazetteer_place_name=None,
        padus_unit_name=None,
    )


def _driving_ctx() -> SimpleNamespace:
    return _ctx(
        telemetry=_tel(154.0, road="I-15"),
        video_understanding={
            "scene_description": "Dashcam view of a highway drive through Logandale under a blue sky."
        },
        vision_context={
            "label_names": ["windshield", "highway", "road"],
            "ocr_text": "154 MPH",
            "face_count": 0,
        },
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


def _novel_ctx() -> SimpleNamespace:
    # Content no fixed taxonomy covers — identity must still work.
    return _ctx(
        video_understanding={
            "scene_description": "An artisan shapes molten glass into a vase inside a hot workshop."
        },
        vision_context={"label_names": ["glass", "furnace"], "face_count": 1},
    )


# ── Evidence harvest ─────────────────────────────────────────────────────


def test_evidence_attributes_providers():
    ev = build_identity_evidence(_gardening_ctx())
    providers = {t["provider"] for t in ev["tokens"]}
    assert "vision" in providers
    assert "video_intelligence" in providers
    assert "twelvelabs" in providers
    assert "tomato" in " ".join(t["text"].lower() for t in ev["tokens"])


def test_evidence_empty_context():
    ev = build_identity_evidence(_ctx())
    assert ev["tokens"] == []
    assert ev["prose"] == {}


# ── Deterministic identity ───────────────────────────────────────────────


def test_gardening_identity_cross_provider_agreement():
    ident = build_content_identity(_gardening_ctx())
    assert ident["confidence"] == "high"  # tomato agreed by vision + VI + scene
    facts = " ".join(f["text"].lower() for f in ident["hero_facts"])
    assert "tomato" in facts
    # No speed data → speed never appears, and do_not_invent says so.
    assert not any(f["class"] == "speed" for f in ident["hero_facts"])
    assert any("never state a speed" in d for d in ident["do_not_invent"])


def test_driving_identity_gets_verified_speed_fact():
    ident = build_content_identity(_driving_ctx())
    speed_facts = [f for f in ident["hero_facts"] if f["class"] == "speed"]
    assert len(speed_facts) == 1
    assert "154" in speed_facts[0]["text"]
    assert any("154 MPH" in d for d in ident["do_not_invent"])


def test_low_confidence_speed_never_becomes_fact():
    # Vision OCR spike only — consensus confidence is not high, so no speed fact.
    ctx = _ctx(
        vision_context={"ocr_text": "202 MPH", "label_names": ["road"], "face_count": 0},
    )
    ident = build_content_identity(ctx)
    assert not any(f["class"] == "speed" for f in ident["hero_facts"])


def test_novel_content_still_produces_subject():
    ident = build_content_identity(_novel_ctx())
    assert "glass" in ident["subject"].lower()
    assert ident["hero_facts"]


def test_empty_context_identity():
    ident = build_content_identity(_ctx())
    assert ident["confidence"] == "none"
    assert ident["hero_facts"] == []
    assert ident["novel_content"] is True


def test_no_faces_guard_in_do_not_invent():
    ident = build_content_identity(_gardening_ctx())
    assert any("do not invent or add people" in d for d in ident["do_not_invent"])


def test_get_content_identity_caches_artifact():
    ctx = _gardening_ctx()
    first = get_content_identity(ctx)
    assert ctx.output_artifacts[CONTENT_IDENTITY_ARTIFACT] is first
    ctx.vision_context = {}
    assert get_content_identity(ctx) is first


# ── Grounding validator + LLM merge ──────────────────────────────────────


def _llm_payload(**overrides):
    base = {
        "subject": "roma tomato harvest in raised garden beds",
        "activity": "harvesting tomatoes",
        "setting": "backyard garden",
        "domain_tags": [{"tag": "gardening", "confidence": 0.93}],
        "hero_facts": [
            {"text": "ripe roma tomatoes fill the harvest", "class": "entity", "providers": ["vision"]},
        ],
        "peak_metric_candidates": [],
        "do_not_invent": ["no pets visible"],
        "novel_content": False,
    }
    base.update(overrides)
    return base


def test_merge_accepts_grounded_llm_output():
    ctx = _gardening_ctx()
    ev = build_identity_evidence(ctx)
    base = build_content_identity(ctx, evidence=ev)
    out = merge_llm_identity(base, _llm_payload(), evidence=ev)
    assert out["resolver"] == "llm+deterministic"
    assert top_domain_tag(out) == "gardening"
    assert out["subject"].startswith("roma tomato")


def test_merge_drops_ungrounded_facts():
    ctx = _gardening_ctx()
    ev = build_identity_evidence(ctx)
    base = build_content_identity(ctx, evidence=ev)
    out = merge_llm_identity(
        base,
        _llm_payload(
            subject="skydiving over the alps",
            hero_facts=[
                {"text": "parachute deployment at altitude", "class": "entity", "providers": ["vision"]},
            ],
        ),
        evidence=ev,
    )
    # Ungrounded subject rejected — deterministic subject survives.
    assert "skydiving" not in out["subject"].lower()
    assert not any("parachute" in f["text"].lower() for f in out["hero_facts"])


def test_merge_gates_llm_speed_facts_on_consensus():
    ctx = _gardening_ctx()
    ev = build_identity_evidence(ctx)
    base = build_content_identity(ctx, evidence=ev)
    out = merge_llm_identity(
        base,
        _llm_payload(
            hero_facts=[
                {"text": "tomatoes at 90 mph", "class": "speed", "providers": ["vision"]},
            ]
        ),
        evidence=ev,
        speed_consensus={"peak_mph": 0.0, "confidence": "none"},
    )
    assert not any(f["class"] == "speed" for f in out["hero_facts"])


def test_merge_none_keeps_deterministic():
    ctx = _driving_ctx()
    ev = build_identity_evidence(ctx)
    base = build_content_identity(ctx, evidence=ev)
    out = merge_llm_identity(base, None, evidence=ev)
    assert out["resolver"] == "deterministic"


def test_fact_is_grounded():
    corpus = {"tomato", "garden", "harvest"}
    assert fact_is_grounded("ripe tomato closeup", corpus)
    assert not fact_is_grounded("skydiving parachute", corpus)


# ── LLM service (no network) ─────────────────────────────────────────────


def test_prompt_includes_evidence_and_speed_contract():
    ctx = _driving_ctx()
    ev = build_identity_evidence(ctx)
    prompt = build_identity_prompt(ev, {"peak_mph": 154.0, "confidence": "high"})
    assert "154 MPH" in prompt
    assert "highway" in prompt.lower()
    prompt_no_speed = build_identity_prompt(ev, {"peak_mph": 0.0, "confidence": "none"})
    assert "NO verified speed" in prompt_no_speed


def test_parse_identity_response_rejects_garbage():
    assert parse_identity_response("not json") is None
    assert parse_identity_response('{"no_subject": 1}') is None
    assert parse_identity_response('{"subject": "a", "hero_facts": []}') is not None
