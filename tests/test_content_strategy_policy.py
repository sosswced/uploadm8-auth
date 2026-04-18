"""content_style_policy.json load path (safe_parse + defaults)."""

from stages.content_strategy import (
    _load_policy,
    build_content_strategy,
    map_ui_caption_style_for_strategy,
    map_ui_caption_tone_for_strategy,
    map_ui_caption_voice_to_persona,
)
from stages.context import JobContext


def test_load_policy_returns_dict_with_defaults():
    p = _load_policy()
    assert isinstance(p, dict)
    assert "defaults" in p
    assert "rules" in p
    assert isinstance(p["defaults"], dict)
    assert isinstance(p["rules"], list)


def test_map_ui_style_tone_voice():
    assert map_ui_caption_style_for_strategy("punchy") == "hook"
    assert map_ui_caption_style_for_strategy("factual") == "factual"
    assert map_ui_caption_tone_for_strategy("hype") == "bold"
    assert map_ui_caption_voice_to_persona("hypebeast") == "hype_friend"
    assert map_ui_caption_voice_to_persona("mentor") == "creator_coach"
    assert map_ui_caption_voice_to_persona("cinematic_narrator") == "storyteller"


def test_build_content_strategy_seeds_from_user_settings():
    ctx = JobContext(job_id="j", upload_id="u", user_id="usr", platforms=["tiktok"])
    ctx.audio_context = {}
    s = build_content_strategy(
        ctx,
        category="general",
        user_caption_style="punchy",
        user_caption_tone="hype",
        user_caption_voice="mentor",
    )
    assert s["outputs"]["caption_style"] == "hook"
    assert s["outputs"]["tone"] == "bold"
    assert s["outputs"]["voice_persona"] == "creator_coach"
    assert s["inputs"].get("user_caption_seed", {}).get("voice_ui") == "mentor"
