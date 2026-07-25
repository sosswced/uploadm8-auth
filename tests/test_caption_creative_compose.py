"""Combinatorial style × tone × voice composer produces distinct, owned briefs."""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace

from core.caption_creative import (
    CAPTION_STYLES,
    CAPTION_TONES,
    CAPTION_VOICES,
    STYLE_DIRECTIVES,
    TONE_DIRECTIVES,
    VOICE_DIRECTIVES,
    cell_micro_brief,
    combination_index,
    compose_creative_directive,
    evidence_matrix_cell_specs,
    interaction_contract,
    total_combinations,
)

_STYLE_FACETS = ("architecture", "hook", "length", "beats", "rotation")
_TONE_FACETS = ("intensity", "pacing", "punctuation", "word_field")
_VOICE_FACETS = ("pov", "diction", "habits", "signature")


def test_every_style_has_complete_facets():
    for key, entry in STYLE_DIRECTIVES.items():
        facets = entry.get("facets")
        assert isinstance(facets, dict), f"style {key} missing facets"
        for f in _STYLE_FACETS:
            assert str(facets.get(f) or "").strip(), f"style {key} missing facet {f}"


def test_every_tone_has_complete_facets_and_valid_intensity():
    for key, entry in TONE_DIRECTIVES.items():
        facets = entry.get("facets")
        assert isinstance(facets, dict), f"tone {key} missing facets"
        for f in _TONE_FACETS:
            assert str(facets.get(f) or "").strip() != "", f"tone {key} missing facet {f}"
        assert 1 <= int(facets["intensity"]) <= 5


def test_every_voice_has_complete_facets():
    for key, entry in VOICE_DIRECTIVES.items():
        facets = entry.get("facets")
        assert isinstance(facets, dict), f"voice {key} missing facets"
        for f in _VOICE_FACETS:
            assert str(facets.get(f) or "").strip(), f"voice {key} missing facet {f}"


def test_all_combinations_compose_uniquely():
    briefs = {}
    for s, t, v in product(CAPTION_STYLES, CAPTION_TONES, CAPTION_VOICES):
        brief = compose_creative_directive(s, t, v)
        assert brief not in briefs.values(), f"duplicate brief for {(s, t, v)}"
        briefs[(s, t, v)] = brief
    assert len(briefs) == total_combinations()


def test_single_knob_swap_changes_brief_materially():
    base = compose_creative_directive("story", "calm", "passenger")
    assert base != compose_creative_directive("punchy", "calm", "passenger")
    assert base != compose_creative_directive("story", "chaotic", "passenger")
    assert base != compose_creative_directive("story", "calm", "journalist")


def test_facet_ownership_visible_in_composed_brief():
    brief = compose_creative_directive("punchy", "hype", "radio_host")
    # STYLE owns length + hook
    assert "under 120 characters" in brief
    assert "first 3 words" in brief
    # TONE owns intensity + pacing
    assert "intensity 4/5" in brief
    assert "tight clauses" in brief
    # VOICE owns POV + diction
    assert "drive-time host" in brief
    assert "track + town + speed callouts" in brief


def test_tension_rules_are_combination_specific():
    # low heat on a compact style
    assert "Low heat (1/5)" in compose_creative_directive("punchy", "calm", "default")
    # high heat inside an arc style
    assert "High heat (5/5)" in compose_creative_directive("story", "chaotic", "default")
    # high heat through a disciplined voice
    assert "disciplined voice" in compose_creative_directive("listicle", "hype", "teacher")
    # hypebeast at low heat keeps cadence, drops caps
    assert "Hypebeast at low heat" in compose_creative_directive("factual", "calm", "hypebeast")
    # first-person style vs third-person voice
    assert "POV clash" in compose_creative_directive("diary", "authentic", "cinematic_narrator")
    # a neutral combination gets none of the special tensions
    neutral = compose_creative_directive("story", "cinematic", "default")
    for marker in ("Low heat", "High heat", "POV clash", "Hypebeast at low heat"):
        assert marker not in neutral


def test_variant_seed_rotates_lead_and_is_deterministic():
    no_seed = compose_creative_directive("story", "calm", "default")
    assert "FRESHNESS ROTATION" not in no_seed
    seeded = [
        compose_creative_directive("story", "calm", "default", variant_seed=i)
        for i in range(5)
    ]
    assert all("FRESHNESS ROTATION" in b for b in seeded)
    assert len(set(seeded)) == 5
    again = compose_creative_directive("story", "calm", "default", variant_seed=3)
    assert again == seeded[3]


def test_combination_index_is_unique_and_bounded():
    seen = set()
    for s, t, v in product(CAPTION_STYLES, CAPTION_TONES, CAPTION_VOICES):
        idx = combination_index(s, t, v)
        assert 1 <= idx <= total_combinations()
        assert idx not in seen
        seen.add(idx)


def test_cell_micro_briefs_unique_across_matrix():
    specs = evidence_matrix_cell_specs("story", "authentic", "default")
    briefs = [cell_micro_brief(s, t, v) for s, t, v in specs]
    assert len(briefs) == len(set(briefs)), "matrix micro-briefs must be distinct per cell"
    for b in briefs:
        assert len(b) < 300


def test_interaction_contract_normalizes_unknown_inputs():
    text = interaction_contract("bogus", "bogus", "bogus")
    assert "STORY structure" in text
    assert "AUTHENTIC intensity" in text
    assert "spoken as DEFAULT" in text


def test_m8_directive_block_uses_composer_and_seed():
    from stages.m8_engine import _m8_creative_directive_block, _m8_variant_seed

    block = _m8_creative_directive_block("listicle", "dry", "best_friend", variant_seed=7)
    assert "LISTICLE × DRY × BEST_FRIEND" in block
    assert "FRESHNESS ROTATION (seed 7)" in block

    ctx = SimpleNamespace(upload_id="abc-123", job_id="")
    seed = _m8_variant_seed(ctx)
    assert isinstance(seed, int) and 0 <= seed < 997
    assert _m8_variant_seed(SimpleNamespace(upload_id="", job_id="")) is None
