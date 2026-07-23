"""Single-source caption creative registry stays consistent across layers."""

from __future__ import annotations

from core.caption_creative import (
    CAPTION_STYLES,
    CAPTION_TONES,
    CAPTION_VOICES,
    STYLE_DIRECTIVES,
    TONE_DIRECTIVES,
    VOICE_DIRECTIVES,
    normalize_caption_style,
    normalize_caption_tone,
    normalize_caption_voice,
    trusted_peak_speed_mph,
)
from stages.caption_stage import STYLE_DIRECTIVES as LEGACY_STYLE
from stages.caption_stage import TONE_DIRECTIVES as LEGACY_TONE
from stages.caption_stage import VOICE_PROFILES as LEGACY_VOICE
from stages.m8_engine import M8_CAPTION_STYLES, M8_CAPTION_TONES, M8_CAPTION_VOICES


def test_m8_allowlists_are_registry_keys():
    assert M8_CAPTION_STYLES == CAPTION_STYLES
    assert M8_CAPTION_TONES == CAPTION_TONES
    assert M8_CAPTION_VOICES == CAPTION_VOICES


def test_legacy_caption_stage_directives_cover_registry():
    assert set(LEGACY_STYLE) == set(STYLE_DIRECTIVES)
    assert set(LEGACY_TONE) == set(TONE_DIRECTIVES)
    assert set(LEGACY_VOICE) == set(VOICE_DIRECTIVES)


def test_normalize_unknown_falls_back():
    assert normalize_caption_style("nope") == "story"
    assert normalize_caption_tone("nope") == "authentic"
    assert normalize_caption_voice("nope") == "default"
    assert normalize_caption_voice("creator_coach") == "mentor"


def test_trusted_peak_telemetry_never_capped_by_series():
    peak, src = trusted_peak_speed_mph(
        telemetry_max=154.0,
        osd_max=92.0,
        series_peak=90.0,
    )
    assert peak == 154.0
    assert src == "telemetry"


def test_trusted_peak_osd_spike_capped_by_series():
    peak, src = trusted_peak_speed_mph(
        telemetry_max=0.0,
        osd_max=154.0,
        series_peak=92.0,
    )
    assert peak == 92.0
    assert "series_cap" in src
