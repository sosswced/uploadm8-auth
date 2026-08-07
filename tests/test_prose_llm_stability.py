"""Caption prose + pick-mode + matrix + voice-fallback stability (plan phases A–F)."""

from __future__ import annotations

from types import SimpleNamespace

from services.upload.prefs import merge_upload_init_caption_creative


def test_merge_omit_preserves_account_cycle():
    prefs = {
        "captionCreativePickMode": "cycle",
        "caption_creative_pick_mode": "cycle",
        "randomizeCaptionCreative": True,
    }
    merge_upload_init_caption_creative(
        prefs,
        SimpleNamespace(
            caption_creative_pick_mode=None,
            captionCreativePickMode=None,
            randomize_caption_creative=None,
            randomizeCaptionCreative=None,
            caption_creative_override=None,
            captionCreativeOverride=None,
            multi_style_captions=None,
            multiStyleCaptions=None,
            caption_creative_vary_style=None,
            captionCreativeVaryStyle=None,
            caption_creative_vary_tone=None,
            captionCreativeVaryTone=None,
            caption_creative_vary_voice=None,
            captionCreativeVaryVoice=None,
            caption_style=None,
            captionStyle=None,
            caption_tone=None,
            captionTone=None,
            caption_voice=None,
            captionVoice=None,
            caption_creative_combo_index=None,
            captionCreativeComboIndex=None,
        ),
    )
    assert prefs["captionCreativePickMode"] == "cycle"


def test_merge_bare_off_does_not_clobber_cycle():
    prefs = {
        "captionCreativePickMode": "cycle",
        "randomizeCaptionCreative": True,
    }
    merge_upload_init_caption_creative(
        prefs,
        SimpleNamespace(
            caption_creative_pick_mode="off",
            captionCreativePickMode=None,
            randomize_caption_creative=False,
            randomizeCaptionCreative=None,
            caption_creative_override=None,
            captionCreativeOverride=None,
            multi_style_captions=None,
            multiStyleCaptions=None,
            caption_creative_vary_style=None,
            captionCreativeVaryStyle=None,
            caption_creative_vary_tone=None,
            captionCreativeVaryTone=None,
            caption_creative_vary_voice=None,
            captionCreativeVaryVoice=None,
            caption_style=None,
            captionStyle=None,
            caption_tone=None,
            captionTone=None,
            caption_voice=None,
            captionVoice=None,
            caption_creative_combo_index=None,
            captionCreativeComboIndex=None,
        ),
    )
    assert prefs["captionCreativePickMode"] == "cycle"


def test_matrix_default_on_when_unset(monkeypatch):
    from stages.m8_engine import m8_evidence_matrix_enabled

    monkeypatch.delenv("M8_CAPTION_STYLE_MATRIX", raising=False)
    assert m8_evidence_matrix_enabled({}) is True
    assert m8_evidence_matrix_enabled({"multiStyleCaptions": False}) is False
    assert m8_evidence_matrix_enabled({"multi_style_captions": True}) is True
    monkeypatch.setenv("M8_CAPTION_STYLE_MATRIX", "0")
    assert m8_evidence_matrix_enabled({}) is False


def test_platform_prompt_has_no_persona_line():
    from stages.m8_engine import _platform_prompt

    text = _platform_prompt(
        "tiktok",
        {"persona": "hype_friend", "risk_level": "safe", "constraints": {}},
    )
    assert "Persona=" not in text
    assert "risk=safe" in text


def test_repair_does_not_prepend_pov():
    from stages.m8_engine import _repair_artifacts_selective

    winner = {"title": "", "caption": "short", "hashtags": ["viral"]}
    repaired, _ = _repair_artifacts_selective(
        "tiktok",
        winner,
        [],
        {"platforms": ["tiktok"]},
        {},
    )
    assert not str(repaired.get("caption") or "").lower().startswith("pov:")


def test_voice_over_checklist_in_score():
    from stages.m8_engine import score_variant

    scene = {
        "platforms": ["instagram"],
        "geo": {"city": "Logandale", "road": "Garlock Road"},
        "speed_consensus": {"peak_mph": 88.0},
    }
    must = ["88 MPH", "Logandale", "Garlock"]
    voice = {
        "caption": "Rolling through Logandale on Garlock with the needle kissing 88 MPH.",
        "title": "88 MPH through Logandale heat",
        "hashtags": ["logandale"],
    }
    checklist = {
        "caption": "Anchored in 88 MPH, Garlock Road, Logandale",
        "title": "88 MPH · Garlock Road · Logandale",
        "hashtags": ["logandale"],
    }
    sv, _ = score_variant(
        "instagram", voice, scene, must_use=must, min_must_use=2,
        caption_style="story", caption_tone="authentic", caption_voice="default",
    )
    sc, _ = score_variant(
        "instagram", checklist, scene, must_use=must, min_must_use=2,
        caption_style="story", caption_tone="authentic", caption_voice="default",
    )
    assert sv > sc


def test_matrix_cell_can_win_publish():
    from stages.m8_engine import merge_matrix_cells_into_ranked

    scene = {
        "platforms": ["tiktok"],
        "geo": {"city": "Logandale"},
        "speed_consensus": {"peak_mph": 110.0},
    }
    ranked = {
        "must_use": ["110 MPH", "Logandale"],
        "platforms": {
            "tiktok": {
                "variants_ranked": [
                    {
                        "variant_index": 0,
                        "caption": "Anchored in 110 MPH · Logandale",
                        "title": None,
                        "hashtags": [],
                        "score": 10.0,
                        "winner_source": "main",
                    }
                ],
                "winner": {
                    "variant_index": 0,
                    "caption": "Anchored in 110 MPH · Logandale",
                    "title": None,
                    "hashtags": [],
                    "score": 10.0,
                    "winner_source": "main",
                },
            }
        },
    }
    matrix = {
        "cells": [
            {
                "caption_style": "punchy",
                "caption_tone": "hype",
                "caption_voice": "hypebeast",
                "tiktok_caption": (
                    "Needle hits 110 MPH outside Logandale and the whole cabin goes quiet for a second."
                ),
                "hashtags": ["logandale", "speed"],
            }
        ]
    }
    out = merge_matrix_cells_into_ranked(ranked, matrix, scene)
    w = (out["platforms"]["tiktok"] or {}).get("winner") or {}
    assert w.get("winner_source") == "matrix"
    assert "110" in str(w.get("caption") or "")
    assert out.get("selection_meta", {}).get("matrix_in_pool") is True


def test_voice_fallback_not_formula_stub():
    from stages.m8_engine import build_voice_fallback_selection
    from services.m8_grounding_pass import is_formula_stub_caption

    scene = {
        "platforms": ["instagram", "tiktok"],
        "geo": {"city": "Logandale", "road": "Garlock Road"},
        "speed_consensus": {"peak_mph": 88.0},
        "music": {"detected": True, "artist": "Fetty Wap", "title": "The Truth"},
    }
    sel = build_voice_fallback_selection(
        scene,
        caption_style="story",
        caption_tone="authentic",
        caption_voice="passenger",
        platforms=["instagram", "tiktok"],
    )
    assert sel["selection_meta"]["winner_source"] == "voice_fallback"
    for pl, block in sel["platforms"].items():
        cap = str((block.get("winner") or {}).get("caption") or "")
        assert cap
        assert not is_formula_stub_caption(cap)
        assert " · " not in cap or "through" in cap.lower()
        assert block.get("winner_source") == "voice_fallback"


def test_hero_priors_automotive_only_keeps_bootstrap_global(tmp_path):
    from core.hero_fact_priors import (
        _BOOTSTRAP_GLOBAL,
        class_rank_for_cluster,
        rebuild_hero_fact_priors,
    )

    rows = []
    for i in range(30):
        rows.append({
            "identity_domain_tag": "automotive",
            "identity_headline_class": "speed" if i % 2 == 0 else "place",
            "is_hot": 1,
            "hotness_score": 2.0,
        })
    out = tmp_path / "priors.json"
    payload = rebuild_hero_fact_priors(rows, out_path=out, min_rows=25)
    assert "automotive" in payload["clusters"]
    assert payload["global"] == list(_BOOTSTRAP_GLOBAL)
    order = class_rank_for_cluster("gardening", priors=payload)
    assert order == list(_BOOTSTRAP_GLOBAL)


def test_live_anchored_stub_loses_to_voice():
    """Regression: live_demo 0d70258c style stub must lose ranking."""
    from services.m8_grounding_pass import is_formula_stub_caption, apply_grounding_pass2_to_ranked
    from stages.m8_engine import score_variant

    cap = "Anchored in 121 MPH, Cascade Wonderland Highway"
    title = "121 MPH · Hornbrook, California · The Weeknd"
    assert is_formula_stub_caption(cap)
    scene = {
        "platforms": ["instagram"],
        "geo": {"city": "Hornbrook", "road": "Cascade Wonderland Highway"},
        "speed_consensus": {"peak_mph": 121.0},
        "music": {"detected": True, "artist": "The Weeknd", "title": "x"},
    }
    must = ["121 MPH", "Hornbrook", "Cascade Wonderland Highway"]
    stub_sc, _ = score_variant(
        "instagram",
        {"caption": cap, "title": title, "hashtags": []},
        scene,
        must_use=must,
        min_must_use=2,
    )
    assert stub_sc < 45.0

    ranked = {
        "must_use": must,
        "platforms": {
            "instagram": {
                "winner": {"caption": cap, "title": title, "hashtags": []},
                "variants_ranked": [
                    {"caption": cap, "title": title},
                    {
                        "caption": (
                            "Needle kisses 121 MPH on Cascade Wonderland Highway "
                            "outside Hornbrook with The Weeknd in the cabin."
                        ),
                        "title": "121 MPH through Hornbrook",
                    },
                ],
            }
        },
        "scene_graph": scene,
    }
    out = apply_grounding_pass2_to_ranked(ranked, scene, must_use=must)
    winner_cap = str(((out["platforms"]["instagram"] or {}).get("winner") or {}).get("caption") or "")
    assert "Anchored in" not in winner_cap
    assert "121" in winner_cap


def test_soft_mph_gate_keeps_place_voice():
    from services.hydration_enforcer import (
        EvidencePool,
        _title_is_timeline_thin,
        _title_is_salvageable_voice,
    )

    pool = EvidencePool(
        max_speed_mph=154.0,
        city="Logandale",
        gazetteer_place="Logandale",
        music_artist="Fetty Wap",
        music_title="The Truth",
    )
    creative = "Sunset pull through Logandale with Fetty Wap on the stereo"
    assert _title_is_salvageable_voice(creative, pool) is True
    assert _title_is_timeline_thin(creative, pool) is False
    assert _title_is_timeline_thin("Cruising Logandale at 46 MPH", pool) is True
