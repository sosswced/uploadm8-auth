"""FactLedger goldens — 9020642f shape + class coverage / hashtag pad."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _ctx_9020642f(*, persona: bool = True):
    """Mirror upload 9020642f-6dc3-4cfc-8e61-1ba79b52f73e evidence + thin publish copy."""
    settings = (
        {
            "captionStyle": "punchy",
            "captionTone": "cinematic",
            "captionVoice": "teacher",
            "maxHashtags": "15",
            "aiHashtagCount": "5",
        }
        if persona
        else {
            "captionStyle": "story",
            "captionTone": "authentic",
            "captionVoice": "default",
            "maxHashtags": "15",
        }
    )
    # Weak LLM leftovers that omit road + vehicle + full music weave.
    weak_title = "Allendale night send"
    weak_caption = "Punchy teacher energy near Allendale while the cabin stays locked in."
    weak_tags = ["allendale", "openroad", "travelvibes", "dashcam"]
    return SimpleNamespace(
        telemetry=SimpleNamespace(
            max_speed_mph=128.0,
            avg_speed_mph=114.0,
            location_city="Allendale",
            location_state="California",
            location_country="US",
            location_road="I 505",
            gazetteer_place_name="Allendale",
            padus_unit_name=None,
            near_padus=False,
            location_display="Allendale, California",
        ),
        telemetry_data=None,
        dashcam_osd_context={"driver_name": "C Walker"},
        vision_context={},
        audio_context={
            "music_detected": True,
            "music_artist": "iLoveMakonnen",
            "music_title": "Maneuvering",
            "music_genre": "Hip Hop",
        },
        trill=SimpleNamespace(score=97, bucket="gloryBoy"),
        trill_score=None,
        ai_transcript="Got to make the stack, so I'm out here movin' it",
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={
            "scene_description": (
                "Fast run in a GAC Group near Allendale, California "
                "vibing to iLoveMakonnen Maneuvering."
            ),
        },
        filename="20250301_0041_CAM.MP4",
        thumbnail_category="automotive",
        ai_title=weak_title,
        ai_caption=weak_caption,
        ai_hashtags=list(weak_tags),
        m8_platform_captions={
            "tiktok": weak_caption,
            "youtube": weak_caption,
            "instagram": weak_caption,
            "facebook": weak_caption,
        },
        m8_platform_titles={
            "tiktok": weak_title,
            "youtube": weak_title,
            "instagram": weak_title,
            "facebook": weak_title,
        },
        m8_platform_hashtags={
            "youtube": list(weak_tags),
            "tiktok": list(weak_tags),
        },
        output_artifacts={},
        upload_id="9020642f-6dc3-4cfc-8e61-1ba79b52f73e",
        user_settings=settings,
        vehicle_make_id=4083,
        vehicle_model_id=9,
        vehicle_make_name="Toyota",
        vehicle_model_name="Camry",
        platforms=["tiktok", "youtube", "instagram", "facebook"],
    )


def test_get_effective_hashtags_reserves_ledger_before_always():
    """Always-tags must not starve FactLedger music/geo under maxHashtags."""
    from stages.context import JobContext

    ctx = JobContext(
        job_id="j-reserve",
        upload_id="reserve-ht",
        user_id="u",
        filename="clip.mp4",
        platforms=["youtube"],
        ai_hashtags=[
            "ilovemakonnen",
            "maneuvering",
            "allendale",
            "i505",
            "toyota",
            "camry",
            "gloryboy",
            "tripledigits",
        ],
        m8_platform_hashtags={
            "youtube": [
                "ilovemakonnen",
                "maneuvering",
                "allendale",
                "i505",
                "toyota",
                "camry",
                "gloryboy",
                "tripledigits",
            ]
        },
        user_settings={
            "maxHashtags": "10",
            "alwaysHashtags": [
                "brand1",
                "brand2",
                "brand3",
                "brand4",
                "brand5",
                "brand6",
                "brand7",
                "brand8",
                "brand9",
                "brand10",
            ],
        },
        output_artifacts={
            "fact_ledger_v1": {
                "facts": {
                    "music_artist": {
                        "cls": "music_artist",
                        "value": "iLoveMakonnen",
                        "slug": "ilovemakonnen",
                        "publishable": True,
                    },
                    "music_title": {
                        "cls": "music_title",
                        "value": "Maneuvering",
                        "slug": "maneuvering",
                        "publishable": True,
                    },
                    "place_primary": {
                        "cls": "place_primary",
                        "value": "Allendale",
                        "slug": "allendale",
                        "publishable": True,
                    },
                    "vehicle_make": {
                        "cls": "vehicle_make",
                        "value": "Toyota",
                        "slug": "toyota",
                        "publishable": True,
                    },
                }
            }
        },
    )
    final = [t.lower().lstrip("#") for t in ctx.get_effective_hashtags("youtube")]
    assert len(final) == 10
    assert "ilovemakonnen" in final
    assert "maneuvering" in final
    assert "allendale" in final
    assert "toyota" in final


def test_class_coverage_score_and_of_classes():
    from services.fact_ledger import (
        class_coverage_score,
        publishable_facts_from_scene_graph,
    )

    sg = {
        "speed_consensus": {"peak_mph": 128.0, "confidence": "high"},
        "geo": {"city": "Allendale", "state": "California", "road": "I 505"},
        "music": {"artist": "iLoveMakonnen", "title": "Maneuvering"},
        "vehicle": {"make": "Toyota", "model": "Camry"},
        "trill": {"bucket": "gloryBoy"},
    }
    facts = publishable_facts_from_scene_graph(sg)
    # Partial voice (place only) is soft-penalized, not hard-killed.
    partial = class_coverage_score(
        "Allendale night",
        "Punchy near Allendale",
        facts,
    )
    assert partial < 0
    assert partial > -200.0
    empty = class_coverage_score("Vibes only", "Just vibes tonight.", facts)
    assert empty == -200.0
    good_cap = (
        "128 MPH locked in near Allendale on I 505 with 'Maneuvering' by "
        "iLoveMakonnen in the Toyota Camry — Trill gloryBoy"
    )
    assert class_coverage_score("Allendale send", good_cap, facts) == 24.0


def test_ranking_rejects_music_less_winner_when_artist_publishable():
    """Music-less voice loses to a variant that cites publishable artist/title."""
    from stages.m8_engine import rank_and_select

    scene = {
        "platforms": ["youtube"],
        "geo": {"city": "Allendale", "state": "California", "road": "I 505"},
        "speed_consensus": {"peak_mph": 128.0, "confidence": "high"},
        "music": {"artist": "iLoveMakonnen", "title": "Maneuvering"},
        "vehicle": {"make": "Toyota", "model": "Camry"},
        "trill": {"bucket": "gloryBoy"},
        "transcript": {},
        "video_understanding": {"scene": "Night run near Allendale."},
    }
    parsed = {
        "platforms": {
            "youtube": {
                "variants": [
                    {
                        "variant_index": 0,
                        "title": "Allendale night send",
                        "caption": (
                            "Punchy teacher energy near Allendale while the cabin "
                            "stays locked in on I 505 at 128 MPH in the Toyota Camry."
                        ),
                        "hashtags": ["allendale", "toyota", "camry"],
                        "score": 95.0,
                    },
                    {
                        "variant_index": 1,
                        "title": "Allendale send with Makonnen",
                        "caption": (
                            "128 MPH locked in near Allendale on I 505 with "
                            "'Maneuvering' by iLoveMakonnen in the Toyota Camry — "
                            "Trill gloryBoy energy."
                        ),
                        "hashtags": [
                            "allendale",
                            "ilovemakonnen",
                            "maneuvering",
                            "toyota",
                            "camry",
                        ],
                        "score": 70.0,
                    },
                ]
            }
        }
    }
    out = rank_and_select(parsed, scene, {})
    block = (out.get("platforms") or {}).get("youtube") or {}
    winner = block.get("winner") or {}
    blob = f"{winner.get('title') or ''} {winner.get('caption') or ''}".lower()
    assert "ilovemakonnen" in blob.replace(" ", "") or "makonnen" in blob
    assert "maneuvering" in blob


def test_signal_hashtags_music_without_detected_flag():
    from services.signal_hashtags import build_signal_hashtags

    ctx = SimpleNamespace(
        audio_context={
            "music_detected": False,
            "music_artist": "iLoveMakonnen",
            "music_title": "Maneuvering",
        },
        telemetry=None,
        telemetry_data=None,
        vision_context={},
        dashcam_osd_context={},
        trill=None,
        trill_score=None,
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        output_artifacts={},
    )
    tags = {t.lower() for t in build_signal_hashtags(ctx)}
    assert "ilovemakonnen" in tags
    assert "maneuvering" in tags


def test_ensure_fact_ledger_at_publish_fail_soft(monkeypatch):
    monkeypatch.delenv("UPLOADM8_FACT_LEDGER_STRICT", raising=False)
    monkeypatch.setenv("UPLOADM8_FACT_LEDGER", "1")
    from services.fact_ledger import ensure_fact_ledger_at_publish

    ctx = _ctx_9020642f()
    ctx.ai_caption = "Just vibes."
    ctx.ai_title = "Vibes"
    ctx.ai_hashtags = []
    ctx.m8_platform_captions = {"youtube": "Just vibes."}
    ctx.m8_platform_titles = {"youtube": "Vibes"}
    ctx.m8_platform_hashtags = {"youtube": []}
    # No telemetry speed consensus path — still has telemetry attrs on ctx
    report = ensure_fact_ledger_at_publish(ctx)
    assert report.get("enabled") is True
    assert report.get("blocked") is False
    # Should have woven or padded something
    assert (ctx.ai_caption and len(ctx.ai_caption) > len("Just vibes.")) or (
        ctx.ai_hashtags and len(ctx.ai_hashtags) >= 5
    )


def test_hydration_report_includes_title_trail(monkeypatch):
    monkeypatch.delenv("UPLOADM8_FACT_LEDGER", raising=False)
    from services.hydration_enforcer import enforce_hydration

    ctx = _ctx_9020642f()
    enforce_hydration(ctx)
    hr = (ctx.output_artifacts or {}).get("hydration_report") or {}
    assert "title_before" in hr
    assert "title_after" in hr
    assert "wipe_reason" in hr
    assert "fact_ledger" in hr
    assert "fact_ledger_v1" in (ctx.output_artifacts or {})


def test_soft_weave_keeps_voice_adds_missing_classes():
    from services.fact_ledger import build_fact_ledger, soft_weave_missing_into_caption
    from services.hydration_enforcer import collect_evidence
    from services.m8_grounding_pass import is_formula_stub_caption

    ctx = _ctx_9020642f()
    ledger = build_fact_ledger(ctx, collect_evidence(ctx))
    new, woven = soft_weave_missing_into_caption(
        ctx.ai_caption, ledger, title=ctx.ai_title, hashtags=ctx.ai_hashtags
    )
    assert woven
    assert "128" in new or "MPH" in new
    assert "Maneuvering" in new or "iLoveMakonnen" in new
    assert "Toyota" in new or "Camry" in new
    assert "I 505" in new or "505" in new
    assert not is_formula_stub_caption(new)
    assert not new.strip().lower().startswith("through ")
    assert "Punchy teacher" in new or "cabin" in new.lower()


def test_pad_hashtags_front_loads_ledger_to_fifteen():
    from services.fact_ledger import build_fact_ledger, pad_hashtags_with_ledger
    from services.hydration_enforcer import build_evidence_hashtags, collect_evidence

    ctx = _ctx_9020642f()
    pool = collect_evidence(ctx)
    ledger = build_fact_ledger(ctx, pool)
    extras = build_evidence_hashtags(pool, max_extra=16)
    merged, _ = pad_hashtags_with_ledger(
        ctx.ai_hashtags, ledger, target=15, extras=extras
    )
    assert len(merged) == 15
    low = {t.lower() for t in merged}
    assert "ilovemakonnen" in low
    assert "maneuvering" in low
    assert "allendale" in low
    assert "toyota" in low or "camry" in low
    assert any("i505" in t or t == "i505" for t in low)
    assert "fyp" not in low
    assert "viral" not in low


def test_enforce_hydration_fact_ledger_9020642f_end_to_end(monkeypatch):
    monkeypatch.delenv("UPLOADM8_FACT_LEDGER", raising=False)
    from services.hydration_enforcer import enforce_hydration
    from services.m8_grounding_pass import is_formula_stub_caption

    ctx = _ctx_9020642f()
    # Start from receipt (historical wipe) — persona + ledger must recover facts
    # without shipping Through-Place compact as the final title.
    receipt = "128 MPH through Allendale, CA — with iLoveMakonnen"
    ctx.ai_title = receipt
    for pl in ctx.m8_platform_titles:
        ctx.m8_platform_titles[pl] = receipt

    report = enforce_hydration(ctx)
    assert report.get("fact_ledger", {}).get("enabled") is True

    title = (ctx.m8_platform_titles or {}).get("youtube") or ctx.ai_title or ""
    caption = (ctx.m8_platform_captions or {}).get("youtube") or ctx.ai_caption or ""
    tags = [t.lower() for t in (ctx.ai_hashtags or [])]

    assert title.strip()
    assert not is_formula_stub_caption(title), f"title still receipt: {title!r}"
    assert "Through Allendale" not in title

    blob = f"{title} {caption}".lower()
    assert "128" in blob or "mph" in blob
    assert "allendale" in blob
    assert "ilovemakonnen" in blob.replace(" ", "") or "makonnen" in blob
    assert "maneuvering" in blob
    assert "toyota" in blob or "camry" in blob
    assert "505" in blob or "i505" in "".join(tags)

    assert len(tags) >= 12
    assert "ilovemakonnen" in tags
    assert "maneuvering" in tags
    assert "allendale" in tags
    assert "toyota" in tags or "camry" in tags

    arts = ctx.output_artifacts or {}
    assert "fact_ledger_v1" in arts
    pubs = set((arts["fact_ledger_v1"] or {}).get("publishable") or [])
    assert "music_artist" in pubs and "vehicle_make" in pubs


def test_llm_title_survives_with_fact_ledger_pad():
    """23a3eb23-style: keep LLM title; pad tags; weave into caption only."""
    from services.hydration_enforcer import enforce_hydration
    from services.m8_grounding_pass import is_formula_stub_caption

    llm_title = "Livermore Chill with A Boogie"
    ctx = SimpleNamespace(
        telemetry=SimpleNamespace(
            max_speed_mph=None,
            avg_speed_mph=None,
            location_city="Livermore",
            location_state="California",
            location_country="US",
            location_road=None,
            gazetteer_place_name="Livermore",
            padus_unit_name=None,
            near_padus=False,
            location_display="Livermore, California",
        ),
        telemetry_data=None,
        dashcam_osd_context={},
        vision_context={},
        audio_context={
            "music_detected": True,
            "music_artist": "A Boogie Wit da Hoodie",
            "music_title": "Still Thinking",
        },
        trill=SimpleNamespace(score=40, bucket="cruise"),
        trill_score=None,
        ai_transcript="",
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={"scene_description": "Night drive near Livermore."},
        filename="clip.mp4",
        thumbnail_category="automotive",
        ai_title=llm_title,
        ai_caption="Night vibes near Livermore stay loose.",
        ai_hashtags=["livermore"],
        m8_platform_captions={"facebook": "Night vibes near Livermore stay loose."},
        m8_platform_titles={"facebook": llm_title},
        m8_platform_hashtags={"facebook": ["livermore"]},
        output_artifacts={},
        upload_id="23a3eb23-ledger",
        user_settings={
            "captionStyle": "punchy",
            "captionTone": "chaotic",
            "captionVoice": "radio_host",
            "maxHashtags": "15",
        },
        vehicle_make_name=None,
        vehicle_model_name=None,
        platforms=["facebook"],
    )
    enforce_hydration(ctx)
    fb = (ctx.m8_platform_titles or {}).get("facebook") or ""
    assert "Livermore" in fb and "Boogie" in fb
    assert not is_formula_stub_caption(fb)
    assert "Through Livermore" not in fb
    tags = ctx.ai_hashtags or []
    assert len(tags) >= 5
    assert any("boogie" in t.lower() for t in tags)
    cap = ctx.ai_caption or ""
    assert "Still Thinking" in cap or "Thinking" in cap


def test_must_use_includes_vehicle_from_scene_graph():
    from stages.m8_engine import build_must_use_shortlist

    sg = {
        "geo": {"city": "Allendale", "state": "California", "road": "I 505"},
        "speed_consensus": {"peak_mph": 128.0, "confidence": "high"},
        "music": {"artist": "iLoveMakonnen", "title": "Maneuvering"},
        "vehicle": {"make": "Toyota", "model": "Camry"},
        "trill": {"bucket": "gloryBoy"},
        "dashcam_osd": {},
        "vision": {},
        "place_evidence": {},
    }
    tokens = build_must_use_shortlist(sg)
    blob = " ".join(tokens).lower()
    assert "toyota" in blob
    assert "camry" in blob
    assert "ilovemakonnen" in blob or "maneuvering" in blob
