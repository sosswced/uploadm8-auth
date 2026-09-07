"""Regression: camera dump filenames + hydration meta must never paint on covers."""

from __future__ import annotations

from types import SimpleNamespace

from core.thumbnail_text import (
    clean_thumbnail_headline,
    is_empty_hydration_story_fallback,
    is_hydration_meta_headline,
    is_media_dump_filename,
    is_unusable_thumbnail_headline,
)
from stages.context import build_hydration_story_text
from stages.pikzels_api import _build_pikzels_v2_prompt
from stages.thumbnail_stage import _concrete_thumbnail_headline, _sanitize_thumbnail_brief


def _empty_ctx(**kwargs):
    base = dict(
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        caption="",
        ai_title="",
        ai_caption="",
        ai_transcript="",
        vision_context={},
        audio_context={},
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        visual_recognition={},
        hydration_payload={},
        thumbnail_category="general",
        output_artifacts={},
        content_identity={},
        get_effective_title=lambda: "IMG_5135.MOV",
        get_effective_caption=lambda: "",
        get_thumbnail_brief_vars=lambda category=None: {},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_media_dump_filename_detection():
    assert is_media_dump_filename("IMG_5135.MOV")
    assert is_media_dump_filename("img_5135")
    assert is_media_dump_filename("IMG 5135.MOV")
    assert is_media_dump_filename(clean_thumbnail_headline("IMG_5135.MOV"))
    assert not is_media_dump_filename("Camp Nou Tunnel Walk")


def test_hydration_meta_headline_detection():
    assert is_hydration_meta_headline("HYDRATION STORY")
    assert is_hydration_meta_headline("HYDRATION STORY IMG 5135.MOV")
    assert is_hydration_meta_headline(
        clean_thumbnail_headline(
            "Hydration story: IMG_5135.MOV has no strong analysis signals yet"
        )
    )
    assert not is_hydration_meta_headline("BARCELONA TUNNEL WALK")


def test_empty_hydration_story_no_filename_bleed():
    story = build_hydration_story_text(_empty_ctx(), max_chars=900)
    assert story == ""
    assert not is_empty_hydration_story_fallback(story)


def test_legacy_empty_hydration_string_is_flagged():
    legacy = (
        "Hydration story: IMG_5135.MOV has no strong analysis signals yet; "
        "use the actual frame and filename only."
    )
    assert is_empty_hydration_story_fallback(legacy)


def test_concrete_headline_skips_img_dump_filename():
    headline = _concrete_thumbnail_headline(_empty_ctx(), "general")
    assert "IMG" not in headline.upper()
    assert "5135" not in headline
    assert "HYDRATION" not in headline.upper()
    assert headline == "VIDEO HIGHLIGHT"


def test_sanitize_rejects_hydration_story_llm_bleed():
    ctx = _empty_ctx()
    brief = _sanitize_thumbnail_brief(
        ctx,
        {
            "selected_headline": "HYDRATION STORY IMG_5135.MOV",
            "headline_options": ["IMG_5135.MOV", "HYDRATION STORY"],
        },
        "general",
    )
    selected = str(brief.get("selected_headline") or "")
    assert "HYDRATION" not in selected.upper()
    assert "IMG" not in selected.upper()
    assert "5135" not in selected
    for opt in brief.get("headline_options") or []:
        assert "HYDRATION" not in str(opt).upper()
        assert not is_media_dump_filename(opt)


def test_pikzels_prompt_no_text_for_dump_or_meta_headline():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "HYDRATION STORY IMG 5135.MOV",
            "hydration_story": (
                "Hydration story: IMG_5135.MOV has no strong analysis signals yet; "
                "use the actual frame and filename only."
            ),
        },
        category="general",
        platform="instagram",
    )
    assert "STRICT NO-TEXT MODE" in prompt
    assert "HYDRATION STORY" not in prompt
    assert "IMG_5135" not in prompt
    assert "Story:" not in prompt


def test_pikzels_prompt_keeps_real_headline_and_story():
    prompt = _build_pikzels_v2_prompt(
        {
            "selected_headline": "CAMP NOU TUNNEL",
            "hydration_story": "Walking out of the tunnel into Camp Nou before kickoff.",
        },
        category="sports",
        platform="instagram",
    )
    assert "CAMP NOU TUNNEL" in prompt
    assert "STRICT NO-TEXT MODE" not in prompt
    assert "Story:" in prompt
    assert is_unusable_thumbnail_headline("IMG_5135.MOV")


def test_effective_title_skips_camera_dump_name():
    from stages.context import JobContext, is_placeholder_upload_title

    assert is_placeholder_upload_title("IMG_5135.MOV", "IMG_5135.MOV")
    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="u1",
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        ai_title="Camp Nou tunnel walk",
    )
    assert ctx.get_effective_title() == "Camp Nou tunnel walk"
    ctx_no_ai = JobContext(
        job_id="j2",
        upload_id="u2",
        user_id="u2",
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        ai_title="",
    )
    assert ctx_no_ai.get_effective_title() == ""
