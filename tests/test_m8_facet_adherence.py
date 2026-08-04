"""Facet adherence scoring for M8 caption ranker."""

from __future__ import annotations

from stages.m8_engine import _facet_adherence_score, _hook_strength_score, score_variant


def test_hook_strength_does_not_reward_pov_or_this():
    pov = _hook_strength_score("POV: you won't believe this drive")
    concrete = _hook_strength_score("88 MPH near Flagstaff at dusk")
    assert concrete > pov


def test_facet_adherence_rewards_punchy_length_and_calm_no_bang():
    short = "88 MPH. Flagstaff. Done."
    long = "x" * 400
    punchy_ok = _facet_adherence_score(short, "", style_ui="punchy", tone_ui="calm", voice_ui="default")
    punchy_long = _facet_adherence_score(long, "", style_ui="punchy", tone_ui="calm", voice_ui="default")
    assert punchy_ok > punchy_long

    calm_clean = _facet_adherence_score(
        "Measured run near Flagstaff at 88 MPH",
        "",
        style_ui="story",
        tone_ui="calm",
        voice_ui="default",
    )
    calm_loud = _facet_adherence_score(
        "Measured run near Flagstaff at 88 MPH!!!",
        "",
        style_ui="story",
        tone_ui="calm",
        voice_ui="default",
    )
    assert calm_clean > calm_loud


def test_score_variant_includes_facet_signal():
    scene = {
        "platforms": ["tiktok"],
        "geo": {"city": "Flagstaff"},
        "dashcam_osd": {"max_speed_mph": 88},
        "vision": {"labels": []},
    }
    good = {
        "caption": "88 MPH through Flagstaff — short and sharp.",
        "title": None,
        "hashtags": ["Flagstaff", "88MPH"],
    }
    weak = {
        "caption": "POV: this is insane!!! " + ("words " * 80),
        "title": None,
        "hashtags": ["nature", "horizon"],
    }
    sg, _ = score_variant(
        "tiktok",
        good,
        scene,
        must_use=["88 MPH", "Flagstaff"],
        min_must_use=1,
        caption_style="punchy",
        caption_tone="calm",
        caption_voice="journalist",
    )
    sw, _ = score_variant(
        "tiktok",
        weak,
        scene,
        must_use=["88 MPH", "Flagstaff"],
        min_must_use=1,
        caption_style="punchy",
        caption_tone="calm",
        caption_voice="journalist",
    )
    assert sg > sw
