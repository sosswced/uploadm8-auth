"""Regression: degenerate repeated-token titles/captions must never publish.

Covers the "the the the the the" stuck-decoding bug that bypassed every
cliché / formula / length guard and shipped to Instagram.
"""

from __future__ import annotations

from core.publish_text_sanitize import (
    collapse_repeated_words,
    is_degenerate_publish_text,
    sanitize_publish_text,
    strip_trailing_hashtag_run,
)


# --------------------------------------------------------------------------
# Sanitizer primitives
# --------------------------------------------------------------------------

def test_collapse_single_word_stutter():
    assert collapse_repeated_words("the the the the the") == "the"
    assert collapse_repeated_words("go go go") == "go"
    assert collapse_repeated_words("Barcelona Barcelona tunnel") == "Barcelona tunnel"


def test_collapse_preserves_real_prose():
    # Legitimate copy with no immediate repetition is untouched.
    assert collapse_repeated_words("Walking out of the tunnel at Camp Nou") == (
        "Walking out of the tunnel at Camp Nou"
    )
    # "that that" style is repetition, but a single normal repeat elsewhere stays.
    assert collapse_repeated_words("the road less traveled") == "the road less traveled"


def test_collapse_phrase_stutter():
    assert collapse_repeated_words("on the road on the road on the road") == "on the road"
    assert collapse_repeated_words("camp nou camp nou tunnel") == "camp nou tunnel"


def test_collapse_case_insensitive_keeps_first_casing():
    assert collapse_repeated_words("The the THE the") == "The"


def test_is_degenerate_detects_stutter():
    assert is_degenerate_publish_text("the the the the the")
    assert is_degenerate_publish_text("the the the")
    assert is_degenerate_publish_text("go go go go")
    # Mostly-repetition long strings.
    assert is_degenerate_publish_text("the a the a the a the a the a the")


def test_is_degenerate_allows_real_titles():
    assert not is_degenerate_publish_text("Camp Nou Tunnel Walk Before Kickoff")
    assert not is_degenerate_publish_text("Walking out into the roar of Camp Nou")
    assert not is_degenerate_publish_text("110 MPH through the canyon")
    assert not is_degenerate_publish_text("")


def test_sanitize_publish_text_idempotent():
    once = sanitize_publish_text("the the the the the")
    assert once == "the"
    assert sanitize_publish_text(once) == "the"


def test_strip_trailing_hashtag_run_keeps_prose():
    raw = (
        "Join the electric atmosphere as Luis Montos Chuty proudly stands "
        "as Barça's representative. ⚽️ #LuisMontosChuty #Barça"
    )
    cleaned = strip_trailing_hashtag_run(raw)
    assert "#LuisMontosChuty" not in cleaned
    assert "#Barça" not in cleaned
    assert "Barça's representative" in cleaned
    assert strip_trailing_hashtag_run("No tags here") == "No tags here"


# --------------------------------------------------------------------------
# Integration: generation-time rejection in _validate_title
# --------------------------------------------------------------------------

def test_validate_title_rejects_degenerate():
    from stages.m8_engine import _validate_title

    ok, reason = _validate_title("the the the the the", {"transcript": {}}, platform="instagram")
    assert ok is False
    assert reason == "degenerate_repetition"


def test_validate_title_allows_real_title():
    from stages.m8_engine import _validate_title

    ok, _reason = _validate_title(
        "Walking out into the roar of Camp Nou", {"transcript": {}}, platform="instagram"
    )
    assert ok is True


# --------------------------------------------------------------------------
# Integration: publish-time safety net on effective getters
# --------------------------------------------------------------------------

def test_effective_title_collapses_stutter():
    from stages.context import JobContext

    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="u1",
        filename="clip.mp4",
        title="",
        ai_title="the the the the the",
    )
    # The degenerate ai_title collapses to a single token rather than shipping stutter.
    assert ctx.get_effective_title() == "the"


def test_effective_caption_collapses_stutter_per_platform():
    from stages.context import JobContext

    ctx = JobContext(
        job_id="j2",
        upload_id="u2",
        user_id="u2",
        filename="clip.mp4",
        caption="",
        ai_caption="",
    )
    ctx.m8_platform_captions = {"instagram": "the the the the the"}
    assert ctx.get_effective_caption("instagram") == "the"
    # Real caption is preserved unchanged.
    ctx.m8_platform_captions = {"instagram": "Out of the tunnel at Camp Nou before kickoff."}
    assert ctx.get_effective_caption("instagram") == (
        "Out of the tunnel at Camp Nou before kickoff."
    )


# --------------------------------------------------------------------------
# Integration: hydration rewrite gate prefers anchor over degenerate copy
# --------------------------------------------------------------------------

def test_hydrate_title_replaces_degenerate_with_anchor():
    from services.hydration_enforcer import _hydrate_title

    out = _hydrate_title("the the the the the", "Camp Nou tunnel walk")
    assert "the the" not in out.lower()
    assert out == "Camp Nou tunnel walk"


def test_hydrate_caption_replaces_degenerate_with_anchor():
    from services.hydration_enforcer import _hydrate_caption

    out = _hydrate_caption("the the the the the the the", "Out of the tunnel at Camp Nou.")
    assert "the the" not in out.lower()
    # Anchor wins over the degenerate caption (trailing punctuation may be
    # normalized by the machine-dump scrubber).
    assert out.rstrip(".") == "Out of the tunnel at Camp Nou"
