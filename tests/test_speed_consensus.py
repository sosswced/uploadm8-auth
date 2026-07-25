"""Speed consensus: canonical peak, wrong-MPH scrub, equal-weight timeline."""

from __future__ import annotations

from types import SimpleNamespace

from core.speed_consensus import (
    SPEED_CONSENSUS_ARTIFACT,
    build_speed_consensus,
    consensus_peak_mph,
    get_speed_consensus,
    scrub_untrusted_speed_claims,
    speed_tolerance_mph,
)
from services.hydration_enforcer import (
    _title_is_timeline_thin,
    _title_suggestion_matches_trusted_peak,
    build_title_anchor_phrase,
    collect_evidence,
)


def _ctx(**overrides) -> SimpleNamespace:
    base = dict(
        upload_id="spc-1",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        thumbnail_category="automotive",
        filename="run.mp4",
        output_artifacts={},
        trill=None,
        trill_score=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tel(mph: float) -> SimpleNamespace:
    return SimpleNamespace(
        max_speed_mph=mph,
        avg_speed_mph=mph * 0.7,
        location_city=None,
        location_state=None,
        location_country=None,
        location_road=None,
        location_display=None,
        location_start_display=None,
        gazetteer_place_name=None,
        padus_unit_name=None,
        near_padus=False,
        points=[],
    )


# ── Consensus artifact ──────────────────────────────────────────────────


def test_consensus_telemetry_is_high_confidence():
    ctx = _ctx(telemetry=_tel(154.0))
    c = build_speed_consensus(ctx)
    assert c["peak_mph"] == 154.0
    assert c["source"] == "telemetry"
    assert c["confidence"] == "high"


def test_consensus_flags_outlier_sources():
    # OSD aggregate spikes to 200 while trusted series says 120 → series wins,
    # the spiked aggregate is listed as an outlier.
    ctx = _ctx(
        dashcam_osd_context={
            "max_speed_mph": 200.0,
            "speed_series": [{"mph": 118.0, "t_s": 4.0}, {"mph": 120.0, "t_s": 9.0}],
        },
    )
    c = build_speed_consensus(ctx)
    assert c["peak_mph"] == 120.0
    assert "osd" in c["outliers"]
    assert "osd_series" in c["agreeing"]


def test_consensus_two_agreeing_sources_high_confidence():
    ctx = _ctx(
        dashcam_osd_context={
            "max_speed_mph": 122.0,
            "speed_series": [{"mph": 120.0, "t_s": 5.0}],
        },
    )
    c = build_speed_consensus(ctx)
    assert c["confidence"] == "high"
    assert set(c["agreeing"]) >= {"osd", "osd_series"}


def test_consensus_none_when_no_speed():
    c = build_speed_consensus(_ctx())
    assert c["peak_mph"] == 0.0
    assert c["confidence"] == "none"


def test_get_speed_consensus_caches_artifact():
    ctx = _ctx(telemetry=_tel(88.0))
    first = get_speed_consensus(ctx)
    assert ctx.output_artifacts[SPEED_CONSENSUS_ARTIFACT] is first
    # Mutating sources afterwards must not change the cached artifact.
    ctx.telemetry = _tel(10.0)
    assert get_speed_consensus(ctx) is first
    assert consensus_peak_mph(ctx) == 88.0


# ── Wrong-MPH scrub ─────────────────────────────────────────────────────


def test_scrub_removes_wrong_claim_keeps_trusted():
    out = scrub_untrusted_speed_claims(
        "The car cruises at 46 MPH before surging to 154 MPH on the freeway.",
        154.0,
    )
    assert "46" not in out
    assert "154 MPH" in out


def test_scrub_keeps_speed_limit_sign_context():
    out = scrub_untrusted_speed_claims(
        "Passes a 45 mph speed limit sign while doing 154 mph.",
        154.0,
    )
    assert "45 mph" in out
    assert "154 mph" in out


def test_scrub_drops_all_claims_when_no_trusted_peak():
    out = scrub_untrusted_speed_claims("Flying at 130 MPH down the straight.", 0.0)
    assert "130" not in out
    assert "MPH" not in out.upper()


def test_scrub_converts_kph_before_comparing():
    # 248 km/h ≈ 154 mph → agrees with the trusted peak, keep it.
    out = scrub_untrusted_speed_claims("Hits 248 km/h on the autobahn.", 154.0)
    assert "248 km/h" in out
    # 80 km/h ≈ 50 mph → contradicts, drop it.
    out2 = scrub_untrusted_speed_claims("Hits 80 km/h on the autobahn.", 154.0)
    assert "80" not in out2


# ── Title gates use the consensus ───────────────────────────────────────


def test_wrong_mph_title_is_thin_and_rewritten():
    ctx = _ctx(
        telemetry=_tel(154.0),
        vision_context={"ocr_text": "Welcome to Logandale"},
        audio_context={
            "music_detected": True,
            "music_artist": "Fetty Wap",
            "music_title": "The Truth",
        },
    )
    pool = collect_evidence(ctx)
    assert pool.max_speed_mph == 154.0
    # Wrong-number MPH title no longer counts as speed coverage.
    assert _title_is_timeline_thin("Cruising Logandale at 46 MPH", pool) is True
    # Correct peak keeps the creative title.
    assert _title_is_timeline_thin("154 MPH flyby — Logandale heat", pool) is False


def test_suggestion_with_mph_rejected_when_no_trusted_peak():
    pool = collect_evidence(_ctx())
    assert _title_suggestion_matches_trusted_peak("Wild 130 MPH run", pool) is False


def test_no_mph_suggestion_cannot_beat_compact_with_speed():
    ctx = _ctx(
        telemetry=_tel(154.0),
        audio_context={
            "music_detected": True,
            "music_artist": "Fetty Wap",
            "music_title": "The Truth",
        },
        video_understanding={
            "scene_description": "A long scenic drive through golden hills.",
            "title_suggestion": "A long scenic drive through golden hills at dusk",
        },
    )
    pool = collect_evidence(ctx)
    anchor = build_title_anchor_phrase(pool, ctx)
    assert "154 MPH" in anchor


def test_collect_evidence_scrubs_invented_scene_speed():
    ctx = _ctx(
        telemetry=_tel(154.0),
        video_understanding={
            "scene_description": "Dashcam rolls at 46 MPH through the valley."
        },
    )
    pool = collect_evidence(ctx)
    assert "46" not in str(pool.video_understanding_phrase or "")


# ── Equal-weight timeline + place-sign cache ────────────────────────────


def test_timeline_has_scene_and_vi_label_beats():
    from stages.context import build_video_story_timeline

    ctx = _ctx(
        video_understanding={
            "scene_description": "A dashcam run down a desert freeway. Traffic thins out late."
        },
        video_intelligence_context={
            "segment_labels": [
                {"description": "freeway driving"},
                {"description": "desert landscape"},
            ]
        },
    )
    events = build_video_story_timeline(ctx)
    kinds = {e["kind"] for e in events}
    assert "scene" in kinds
    assert "vi_label" in kinds
    scene_events = [e for e in events if e["kind"] == "scene"]
    assert "desert freeway" in scene_events[0]["text"]


def test_m8_brief_has_speed_contract_and_diverse_spine():
    from stages.m8_engine import _build_hydration_timeline_brief

    brief = _build_hydration_timeline_brief(
        {
            "hydration_story": "154 MPH run through Logandale.",
            "dashcam_osd": {
                "max_speed_mph": 154.0,
                "speed_series": [{"mph": 150.0, "t_s": 3.0}, {"mph": 154.0, "t_s": 8.0}],
            },
            "geo": {"max_speed_mph": 154.0, "city": "Logandale"},
            "music": {"artist": "Fetty Wap", "title": "The Truth"},
            "timeline": [
                {"t_seconds": 0.0, "kind": "scene", "text": "Desert freeway run"},
                {"t_seconds": 0.0, "kind": "welcome_sign", "text": "Welcome to Logandale"},
                {"t_seconds": 1.0, "kind": "osd_speed_beat", "text": "80 MPH"},
                {"t_seconds": 3.0, "kind": "osd_speed_beat", "text": "120 MPH"},
                {"t_seconds": 5.0, "kind": "osd_speed_beat", "text": "140 MPH"},
                {"t_seconds": 8.0, "kind": "osd_speed", "text": "OSD peak 154 MPH"},
                {"t_seconds": 2.0, "kind": "music", "text": "Music detected: Fetty Wap — The Truth"},
                {"t_seconds": 4.0, "kind": "transcript", "text": "Look at this stretch of road"},
            ],
        }
    )
    assert "SPEED CONTRACT" in brief
    assert "154 MPH" in brief
    # Non-speed providers must survive in the spine.
    assert "Welcome to Logandale" in brief
    assert "Fetty Wap" in brief
    assert "Look at this stretch of road" in brief


def test_collect_place_signs_cached_once(monkeypatch):
    import services.scene_fusion as sf

    calls = {"n": 0}
    real = sf.extract_place_signs

    def _counting(*blobs):
        calls["n"] += 1
        return real(*blobs)

    monkeypatch.setattr(sf, "extract_place_signs", _counting)
    ctx = _ctx(vision_context={"ocr_text": "Welcome to Ashland"})
    first = sf.collect_place_signs(ctx)
    second = sf.collect_place_signs(ctx)
    assert first == second
    assert any("Ashland" in s for s in first)
    assert calls["n"] == 1


def test_signal_hashtags_speed_bucket_uses_consensus():
    from services.signal_hashtags import build_signal_hashtags

    # OSD aggregate is a 200 MPH OCR spike; trusted series says 68 → the
    # bucket must come from the consensus (68), not the spike.
    ctx = _ctx(
        dashcam_osd_context={
            "max_speed_mph": 200.0,
            "speed_series": [{"mph": 66.0, "t_s": 2.0}, {"mph": 68.0, "t_s": 6.0}],
        },
    )
    tags = build_signal_hashtags(ctx)
    blob = " ".join(tags).lower()
    # 68 MPH → FreewayDrive bucket; the 200 spike would have hit TopSpeed.
    assert "topspeed" not in blob
    assert "freewaydrive" in blob


def test_speed_tolerance_floor_and_pct():
    assert speed_tolerance_mph(0) == 8.0
    assert speed_tolerance_mph(50) == 8.0
    assert speed_tolerance_mph(150) == 18.0
