"""Regression tests for hashtag coercion, platform merge, caption cleanup, TikTok chunks."""
from __future__ import annotations

import pytest

from core.helpers import (
    coerce_hashtag_list,
    merge_platform_hashtag_overlay,
    sanitize_hashtag_body,
)
from stages.context import JobContext
from stages.entitlements import Entitlements
from stages.publish_stage import (
    _build_platform_caption,
    _strip_stray_hashtag_json_blob,
    _tiktok_file_upload_chunk_plan,
)
from worker import _merge_job_preferences


def test_coerce_hashtag_list_json_and_nested():
    assert coerce_hashtag_list('["a", "b"]') == ["a", "b"]
    assert coerce_hashtag_list('["x"]') == ["x"]
    assert coerce_hashtag_list(['["nested"]']) == ["nested"]
    assert coerce_hashtag_list("a, b, c") == ["a", "b", "c"]


def test_sanitize_hashtag_body_strips_json_noise():
    assert sanitize_hashtag_body('#"[\\"tester\\"' ) == "tester"
    assert sanitize_hashtag_body("qwe") == "qwe"


def test_merge_job_preferences_empty_platform_hashtags_does_not_wipe_db():
    ctx = JobContext(job_id="j", upload_id="u", user_id="1")
    ctx.user_settings = {
        "platformHashtags": {"facebook": ["fbonly"], "youtube": ["yt"]},
    }
    _merge_job_preferences(ctx, {"preferences": {"platformHashtags": {}, "theme": "dark"}})
    ph = ctx.user_settings.get("platformHashtags") or {}
    assert ph.get("facebook") == ["fbonly"]
    assert ph.get("youtube") == ["yt"]
    assert ctx.user_settings.get("theme") == "dark"


def test_merge_job_preferences_partial_platform_hashtags_merges():
    ctx = JobContext(job_id="j", upload_id="u", user_id="1")
    ctx.user_settings = {
        "platformHashtags": {"facebook": ["keep"], "instagram": ["igold"]},
    }
    _merge_job_preferences(
        ctx,
        {"preferences": {"platformHashtags": {"instagram": ["ignew"]}}},
    )
    ph = ctx.user_settings["platformHashtags"]
    assert ph["facebook"] == ["keep"]
    assert ph["instagram"] == ["ignew"]


def test_get_effective_hashtags_respects_max_total():
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        hashtags=["a", "b", "c"],
        ai_hashtags=["d", "e"],
        user_settings={"alwaysHashtags": ["z"], "maxHashtags": 3},
    )
    tags = ctx.get_effective_hashtags("youtube")
    assert len(tags) == 3
    assert tags[0] == "#z"


def test_get_effective_hashtags_order_and_platform_keys():
    e = Entitlements()
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=e,
        hashtags=["base"],
        ai_hashtags=["ai"],
        user_settings={
            "alwaysHashtags": ["always"],
            "platformHashtags": {"facebook": ["fb1", "fb2"]},
            "blockedHashtags": ["blocked"],
        },
    )
    # duplicate body 'fb1' later should dedupe
    ctx.hashtags = ["fb1", "base"]
    tags = ctx.get_effective_hashtags("facebook")
    bodies = [t.lstrip("#") for t in tags]
    assert bodies[0] == "always"
    assert "fb1" in bodies and "fb2" in bodies
    assert "base" in bodies
    assert "ai" in bodies
    assert "blocked" not in bodies
    assert bodies.index("always") < bodies.index("fb1")


def test_get_effective_hashtags_youtube_alias_google():
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        user_settings={"platformHashtags": {"google": ["fromgoogle"]}},
    )
    tags = ctx.get_effective_hashtags("youtube")
    assert tags == ["#fromgoogle"]


def test_get_effective_hashtags_m8_platform_merged():
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        hashtags=["upload"],
        user_settings={"platformHashtags": {"tiktok": ["user_tt"]}},
        m8_platform_hashtags={"tiktok": ["m8a", "m8b"]},
    )
    tags = ctx.get_effective_hashtags("tiktok")
    joined = " ".join(tags)
    assert "#user_tt" in joined
    assert "#m8a" in joined
    assert "#m8b" in joined
    assert "#upload" in joined


def test_get_effective_hashtags_upload_not_starved_by_m8_under_cap():
    """Per-upload hashtags must survive maxHashtags when M8 suggests many tags."""
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        hashtags=["mybrand", "custom", "vlog"],
        ai_hashtags=["fyp", "viral"],
        user_settings={"maxHashtags": 5},
        m8_platform_hashtags={"tiktok": ["m8a", "m8b", "m8c"]},
    )
    tags = ctx.get_effective_hashtags("tiktok")
    bodies = [t.lstrip("#") for t in tags]
    assert bodies == ["mybrand", "custom", "vlog", "m8a", "m8b"]


@pytest.mark.parametrize(
    "dirty,expect_sub",
    [
        ('Caption here. #"[\\"tester\\" #\\"qwe\\"]" #ok', "Caption here."),
        ("Hello #[\\\"a\\\" \\\"b\\\"] tail", "Hello tail"),
    ],
)
def test_strip_stray_hashtag_json_blob(dirty: str, expect_sub: str):
    cleaned = _strip_stray_hashtag_json_blob(dirty)
    assert expect_sub in cleaned
    assert "#[" not in cleaned
    assert '\\"' not in cleaned


def test_tiktok_chunk_plan_small_and_multipart():
    c, n = _tiktok_file_upload_chunk_plan(3 * 1024 * 1024)
    assert n == 1 and c == 3 * 1024 * 1024

    c, n = _tiktok_file_upload_chunk_plan(70 * 1024 * 1024)
    assert n >= 2
    assert c == 10 * 1024 * 1024 or c >= 5 * 1024 * 1024
    last = 70 * 1024 * 1024 - (n - 1) * c
    assert 0 < last <= 128 * 1024 * 1024


def test_tiktok_chunk_plan_rejects_empty():
    with pytest.raises(ValueError):
        _tiktok_file_upload_chunk_plan(0)


def test_merge_platform_hashtag_overlay_empty_incoming_keeps_base():
    base = {"tiktok": ["1", "2"], "youtube": ["3"]}
    assert merge_platform_hashtag_overlay(base, {}) == {
        "tiktok": ["1", "2"],
        "youtube": ["3"],
    }


def test_merge_platform_hashtag_overlay_partial_incoming_merges():
    base = {"tiktok": ["keep"], "youtube": ["3", "4"], "instagram": [], "facebook": []}
    inc = {"tiktok": ["only_tt"]}
    assert merge_platform_hashtag_overlay(base, inc)["youtube"] == ["3", "4"]
    assert merge_platform_hashtag_overlay(base, inc)["tiktok"] == ["only_tt"]


def test_merge_platform_hashtag_overlay_lowercases_keys():
    assert merge_platform_hashtag_overlay({}, {"YouTube": ["a"]}) == {"youtube": ["a"]}


def test_instagram_first_comment_mode_caption_excludes_hashtags():
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        caption="Hello route",
        ai_hashtags=["roadtrip", "sunset"],
        user_settings={"hashtagPosition": "comment"},
    )
    body = _build_platform_caption(ctx, "instagram")
    assert "Hello route" in body
    assert "#roadtrip" not in body and "#sunset" not in body


def test_instagram_start_mode_includes_hashtags():
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        caption="Hi",
        ai_hashtags=["a"],
        user_settings={"hashtagPosition": "start"},
    )
    body = _build_platform_caption(ctx, "instagram")
    assert body.startswith("#a") or "#a" in body[:20]


def test_facebook_comment_setting_still_inlines_hashtags():
    """'First comment' is IG-only; Facebook keeps tags in the description body."""
    ctx = JobContext(
        job_id="j",
        upload_id="u",
        user_id="1",
        entitlements=Entitlements(),
        caption="Desc",
        ai_hashtags=["fb"],
        user_settings={"hashtagPosition": "comment"},
    )
    body = _build_platform_caption(ctx, "facebook")
    assert "#fb" in body
    assert "Desc" in body
