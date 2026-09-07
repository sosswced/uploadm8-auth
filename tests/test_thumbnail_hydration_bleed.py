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
    assert clean_thumbnail_headline("IMG_5135.MOV") == ""
    assert clean_thumbnail_headline("IMG 5135") == ""
    assert not is_media_dump_filename("Camp Nou Tunnel Walk")


def test_hydration_meta_headline_detection():
    assert is_hydration_meta_headline("HYDRATION STORY")
    assert is_hydration_meta_headline("HYDRATION STORY IMG 5135.MOV")
    assert clean_thumbnail_headline(
        "Hydration story: IMG_5135.MOV has no strong analysis signals yet"
    ) == ""
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


def test_thumbnail_brief_vars_never_pass_dump_filename():
    """Pikzels / brief LLM must not be told to paint IMG_5135 after dump titles are refused."""
    from stages.context import JobContext

    ctx = JobContext(
        job_id="j3",
        upload_id="u3",
        user_id="u3",
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        ai_title="",
        vision_context={
            "label_names": ["Soccer", "Stadium", "Jersey"],
            "web_entities": ["FC Barcelona", "Camp Nou"],
            "logo_names": ["FC Barcelona"],
            "ocr_text": "FC BARCELONA HYDRATION POINT",
            "dominant_colors": [
                {"name": "red", "rgb": [190, 20, 40], "score": 0.22},
                {"name": "blue", "rgb": [20, 40, 170], "score": 0.18},
            ],
        },
    )
    ctx.thumbnail_category = "sports"
    brief = ctx.get_thumbnail_brief_vars(category="sports")
    title = str(brief.get("effective_title") or "")
    assert "IMG" not in title.upper()
    assert "5135" not in title
    assert "barcelona" in title.lower() or title == "Video"


def test_filename_never_eligible_for_thumbnail_or_pikzels():
    """Hard invariant: no thumbnail/Pikzels path may use the upload filename as text."""
    from core.thumbnail_text import (
        is_filename_like_thumbnail_text,
        safe_thumbnail_prompt_title,
    )
    from stages.context import JobContext
    from stages.pikzels_api import _build_pikzels_v2_prompt
    from stages.thumbnail_stage import _concrete_thumbnail_headline, _sanitize_thumbnail_brief

    assert is_filename_like_thumbnail_text("IMG_5135.MOV")
    assert is_filename_like_thumbnail_text("IMG 5135", filename="IMG_5135.MOV")
    assert is_filename_like_thumbnail_text("holiday.mp4")
    assert safe_thumbnail_prompt_title("IMG_5135.MOV") == "Video"
    assert safe_thumbnail_prompt_title("Camp Nou Tunnel") == "Camp Nou Tunnel"

    ctx = JobContext(
        job_id="j4",
        upload_id="u4",
        user_id="u4",
        filename="IMG_5135.MOV",
        title="IMG_5135.MOV",
        ai_title="",
        vision_context={},
    )
    brief = ctx.get_thumbnail_brief_vars(category="general")
    assert brief["effective_title"] == "Video"
    assert "5135" not in brief["effective_title"]
    assert "IMG" not in brief["effective_title"].upper()

    headline = _concrete_thumbnail_headline(ctx, "general")
    assert "5135" not in headline
    assert "IMG" not in headline.upper()

    sanitized = _sanitize_thumbnail_brief(
        ctx,
        {
            "selected_headline": "IMG_5135.MOV",
            "badge_text": "IMG 5135",
            "headline_options": ["IMG_5135.MOV", "holiday.mp4"],
        },
        "general",
    )
    assert "5135" not in str(sanitized.get("selected_headline") or "")
    assert "5135" not in str(sanitized.get("badge_text") or "")
    for opt in sanitized.get("headline_options") or []:
        assert "5135" not in str(opt)
        assert not str(opt).lower().endswith(".mp4")

    prompt = _build_pikzels_v2_prompt(
        {"selected_headline": "IMG 5135"},
        category="general",
        platform="instagram",
    )
    assert "STRICT NO-TEXT MODE" in prompt
    assert "IMG 5135" not in prompt
    assert "IMG_5135" not in prompt
