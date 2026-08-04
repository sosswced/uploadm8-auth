"""Randomize / cycle caption style × tone × voice with per-axis locks."""

from __future__ import annotations

import random
from types import SimpleNamespace

from core.caption_creative import (
    CAPTION_STYLES,
    CAPTION_TONES,
    CAPTION_VOICES,
    combination_at,
    combination_index,
    normalize_caption_creative_pick_mode,
    parse_caption_creative_vary_flags,
    pick_random_combination,
    resolve_caption_creative_knobs,
    total_combinations,
)
from services.upload.prefs import merge_upload_init_caption_creative


def test_total_combinations_matches_product():
    assert total_combinations() == len(CAPTION_STYLES) * len(CAPTION_TONES) * len(CAPTION_VOICES)
    assert total_combinations() == 378


def test_combination_at_round_trips_all_indices():
    seen = set()
    for i in range(total_combinations()):
        s, t, v = combination_at(i)
        assert s in CAPTION_STYLES
        assert t in CAPTION_TONES
        assert v in CAPTION_VOICES
        seen.add((s, t, v))
        assert combination_index(s, t, v) == i + 1
    assert len(seen) == total_combinations()


def test_combination_at_wraps():
    assert combination_at(0) == combination_at(total_combinations())
    assert combination_at(-1) == combination_at(total_combinations() - 1)


def test_pick_random_combination_is_deterministic_with_rng():
    a = pick_random_combination(random.Random(42))
    b = pick_random_combination(random.Random(42))
    assert a == b
    assert a[0] in CAPTION_STYLES


def test_normalize_pick_mode():
    assert normalize_caption_creative_pick_mode(True) == "random"
    assert normalize_caption_creative_pick_mode("cycle") == "cycle"
    assert normalize_caption_creative_pick_mode("sweep") == "cycle"
    assert normalize_caption_creative_pick_mode(False) == "off"
    assert normalize_caption_creative_pick_mode(None) == "off"


def test_parse_vary_flags_default_all_true():
    assert parse_caption_creative_vary_flags({}) == {"style": True, "tone": True, "voice": True}


def test_parse_vary_flags_lock_inverts():
    assert parse_caption_creative_vary_flags({"captionCreativeLockTone": True}) == {
        "style": True,
        "tone": False,
        "voice": True,
    }


def test_resolve_off_uses_saved_defaults():
    out = resolve_caption_creative_knobs(
        {"captionStyle": "punchy", "captionTone": "hype", "captionVoice": "mentor"}
    )
    assert out["randomized"] is False
    assert out["pick_mode"] == "off"
    assert out["style"] == "punchy"
    assert out["tone"] == "hype"
    assert out["voice"] == "mentor"


def test_resolve_random_overrides_defaults():
    out = resolve_caption_creative_knobs(
        {
            "captionStyle": "punchy",
            "captionTone": "hype",
            "captionVoice": "mentor",
            "randomizeCaptionCreative": True,
        },
        rng=random.Random(7),
    )
    assert out["randomized"] is True
    assert out["pick_mode"] == "random"
    assert out["style"] in CAPTION_STYLES
    assert out == resolve_caption_creative_knobs(
        {"randomizeCaptionCreative": True},
        rng=random.Random(7),
    )


def test_resolve_lock_tone_keeps_fixed_tone():
    out = resolve_caption_creative_knobs(
        {
            "randomizeCaptionCreative": True,
            "captionTone": "documentary",
            "captionCreativeVaryStyle": True,
            "captionCreativeVaryTone": False,
            "captionCreativeVaryVoice": True,
        },
        rng=random.Random(3),
    )
    assert out["randomized"] is True
    assert out["tone"] == "documentary"
    assert out["vary"]["tone"] is False
    assert out["subspace_size"] == len(CAPTION_STYLES) * len(CAPTION_VOICES)


def test_resolve_lock_two_axes_only_varies_one():
    out = resolve_caption_creative_knobs(
        {
            "captionCreativePickMode": "cycle",
            "captionCreativeComboIndex": 2,
            "captionStyle": "story",
            "captionTone": "calm",
            "captionVoice": "mentor",
            "captionCreativeVaryStyle": False,
            "captionCreativeVaryTone": False,
            "captionCreativeVaryVoice": True,
        }
    )
    assert out["style"] == "story"
    assert out["tone"] == "calm"
    assert out["voice"] == CAPTION_VOICES[2 % len(CAPTION_VOICES)]
    assert out["subspace_size"] == len(CAPTION_VOICES)


def test_resolve_all_locked_is_not_randomized():
    out = resolve_caption_creative_knobs(
        {
            "randomizeCaptionCreative": True,
            "captionStyle": "diary",
            "captionTone": "dry",
            "captionVoice": "journalist",
            "captionCreativeVaryStyle": False,
            "captionCreativeVaryTone": False,
            "captionCreativeVaryVoice": False,
        }
    )
    assert out["randomized"] is False
    assert out["style"] == "diary"
    assert out["tone"] == "dry"
    assert out["voice"] == "journalist"


def test_resolve_cycle_uses_explicit_index():
    out = resolve_caption_creative_knobs(
        {"captionCreativePickMode": "cycle", "captionCreativeComboIndex": 0}
    )
    assert out["randomized"] is True
    assert out["pick_mode"] == "cycle"
    assert (out["style"], out["tone"], out["voice"]) == combination_at(0)
    assert out["combo_index"] == 1


def test_resolve_cycle_stable_from_upload_id():
    a = resolve_caption_creative_knobs(
        {"caption_creative_pick_mode": "cycle"},
        upload_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    b = resolve_caption_creative_knobs(
        {"caption_creative_pick_mode": "cycle"},
        upload_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert a == b
    assert a["randomized"] is True


def test_merge_upload_init_caption_creative_random():
    prefs: dict = {"captionStyle": "story"}
    merge_upload_init_caption_creative(
        prefs,
        SimpleNamespace(
            randomize_caption_creative=True,
            caption_creative_pick_mode=None,
            caption_creative_combo_index=None,
            caption_creative_vary_style=False,
            caption_creative_vary_tone=True,
            caption_creative_vary_voice=True,
            caption_style="punchy",
            caption_tone=None,
            caption_voice=None,
            randomizeCaptionCreative=None,
            captionCreativePickMode=None,
            captionCreativeComboIndex=None,
            captionCreativeVaryStyle=None,
            captionCreativeVaryTone=None,
            captionCreativeVaryVoice=None,
            captionStyle=None,
            captionTone=None,
            captionVoice=None,
        ),
    )
    assert prefs["randomizeCaptionCreative"] is True
    assert prefs["captionCreativePickMode"] == "random"
    assert prefs["captionCreativeVaryStyle"] is False
    assert prefs["captionStyle"] == "punchy"


def test_merge_upload_init_caption_creative_cycle_ignores_client_index():
    prefs: dict = {}
    merge_upload_init_caption_creative(
        prefs,
        SimpleNamespace(
            randomize_caption_creative=True,
            caption_creative_pick_mode="cycle",
            caption_creative_combo_index=12,
            caption_creative_vary_style=None,
            caption_creative_vary_tone=None,
            caption_creative_vary_voice=None,
            caption_style=None,
            caption_tone=None,
            caption_voice=None,
            randomizeCaptionCreative=None,
            captionCreativePickMode=None,
            captionCreativeComboIndex=None,
            captionCreativeVaryStyle=None,
            captionCreativeVaryTone=None,
            captionCreativeVaryVoice=None,
            captionStyle=None,
            captionTone=None,
            captionVoice=None,
        ),
    )
    assert prefs["captionCreativePickMode"] == "cycle"
    assert "captionCreativeComboIndex" in prefs
    assert prefs.get("captionCreativeShuffleSeed")
    assert prefs.get("captionCreativeDeckFingerprint")
    resolved = resolve_caption_creative_knobs(prefs)
    assert resolved["randomized"] is True
    assert resolved["pick_mode"] == "cycle"


def test_shuffled_deck_order_is_permutation():
    from core.caption_creative import shuffled_deck_order

    n = 20
    order = shuffled_deck_order("seed-a", n)
    assert sorted(order) == list(range(n))
    assert order != list(range(n))
    assert shuffled_deck_order("seed-a", n) == order
    assert shuffled_deck_order("seed-b", n) != order


def test_allocate_shuffled_cycle_draw_no_repeats_until_exhausted():
    from core.caption_creative import allocate_shuffled_cycle_draw

    prefs: dict = {
        "captionCreativePickMode": "cycle",
        "captionCreativeVaryStyle": True,
        "captionCreativeVaryTone": True,
        "captionCreativeVaryVoice": True,
    }
    seen = []
    n = total_combinations()
    for _ in range(n):
        allocate_shuffled_cycle_draw(prefs)
        seen.append(prefs["captionCreativeComboIndex"])
    assert len(seen) == n
    assert len(set(seen)) == n
    allocate_shuffled_cycle_draw(prefs)
    assert prefs["captionCreativeComboIndex"] in range(n)


def test_allocate_uses_account_cursor_across_calls():
    from core.caption_creative import allocate_shuffled_cycle_draw

    prefs: dict = {
        "captionCreativePickMode": "cycle",
        "captionCreativeVaryStyle": False,
        "captionCreativeVaryTone": False,
        "captionCreativeVaryVoice": True,
        "captionStyle": "story",
        "captionTone": "calm",
        "captionVoice": "default",
    }
    allocate_shuffled_cycle_draw(prefs)
    first = prefs["captionCreativeComboIndex"]
    seed = prefs["captionCreativeDrawSeed"]
    allocate_shuffled_cycle_draw(prefs)
    second = prefs["captionCreativeComboIndex"]
    assert first != second
    assert prefs["captionCreativeDrawSeed"] == seed
