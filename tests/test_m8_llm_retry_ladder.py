"""M8 OpenAI retry ladder: full → expand(on length/parse) → compact → text_only; stop on quota."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from stages.m8_engine import (
    m8_completion_budget,
    m8_temperature_for_tone,
    run_m8_caption_engine,
    serialize_scene_graph_for_prompt,
)


def _ctx():
    return SimpleNamespace(
        upload_id="m8-ladder-1",
        user_id="u1",
        platforms=["youtube", "tiktok"],
        user_settings={},
        filename="clip.mp4",
        thumbnail_category="automotive",
        telemetry=None,
        telemetry_data=None,
        dashcam_osd_context={"max_speed_mph": 88.0},
        vision_context={"labels": ["car"], "ocr_text": ""},
        audio_context={},
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        ai_transcript="",
        output_artifacts={},
        hashtags=[],
        m8_platform_titles={},
        m8_platform_captions={},
        m8_platform_hashtags={},
        ai_title=None,
        ai_caption=None,
        ai_hashtags=[],
        m8_scene_graph=None,
        m8_engine_meta=None,
    )


def _ok_payload():
    return {
        "platforms": {
            "youtube": {
                "variants": [
                    {
                        "title": "88 MPH on the freeway",
                        "caption": "Triple-digit energy on a clear stretch.",
                        "hashtags": ["freeway", "sendit"],
                    }
                ]
            },
            "tiktok": {
                "variants": [
                    {
                        "title": None,
                        "caption": "88 MPH send it",
                        "hashtags": ["sendit"],
                    }
                ]
            },
        }
    }


def test_m8_completion_budget_scales_and_floors():
    assert m8_completion_budget(1) == 5200  # floor beats 280*5
    assert m8_completion_budget(4) == 5600  # 280*4*5
    assert m8_completion_budget(12) <= 16000


def test_m8_temperature_scales_with_tone_intensity():
    calm = m8_temperature_for_tone("calm")
    chaotic = m8_temperature_for_tone("chaotic")
    assert calm < chaotic
    assert 0.5 <= calm <= 0.75
    assert 0.8 <= chaotic <= 1.1


def test_serialize_scene_graph_never_mid_cuts_and_keeps_priority():
    fat = {
        "hydration_story": "Driver on Route 66 at dusk.",
        "timeline": [{"t_seconds": 1, "kind": "osd_speed", "text": "88 MPH"}],
        "geo": {"road": "I-40", "city": "Flagstaff"},
        "music": {"artist": "Test", "title": "Song"},
        "dashcam_osd": {"max_speed_mph": 88},
        "platforms": ["youtube", "tiktok"],
        "vision": {
            "labels": ["car"] * 5000,
            "raw_frames": [{"x": "y" * 200}] * 100,
        },
        "trend_intel": {"rows": [{"title": "x" * 500}] * 50},
    }
    out = serialize_scene_graph_for_prompt(fat, char_budget=8000)
    assert len(out) <= 8000
    parsed = __import__("json").loads(out)
    assert "timeline" in parsed or "hydration_story" in parsed
    assert "geo" in parsed or "dashcam_osd" in parsed
    # Must be valid JSON (no mid-string truncation)
    assert out.strip().endswith("}") or out.strip().endswith("]")


def test_m8_ladder_falls_to_text_only_after_full_expand_and_compact_fail():
    ctx = _ctx()
    calls = []

    async def _fake(*, frames, prompt, model, max_completion_tokens, http_timeout_sec, temperature=0.55):
        calls.append(
            {
                "n_frames": len(frames or []),
                "max_compl": max_completion_tokens,
            }
        )
        # full + expand + compact fail; text_only succeeds
        if len(calls) < 4:
            return {}, {"prompt": 0, "completion": 0}, "parse_failed"
        return _ok_payload(), {"prompt": 10, "completion": 20}, ""

    async def _run():
        with patch("stages.m8_engine._call_openai_m8_json", new=AsyncMock(side_effect=_fake)):
            with patch(
                "stages.m8_engine.build_scene_graph",
                return_value={
                    "platforms": ["youtube", "tiktok"],
                    "vision": {},
                    "transcript": {},
                    "video_intelligence": {},
                },
            ):
                with patch("stages.m8_engine.m8_evidence_matrix_enabled", return_value=False):
                    with patch(
                        "stages.m8_engine.rank_and_select",
                        return_value={
                            "platforms": _ok_payload()["platforms"],
                            "must_use": [],
                        },
                    ):
                        with patch(
                            "stages.m8_engine._ensure_platform_completeness",
                            side_effect=lambda r, s: r,
                        ):
                            def _apply(ctx_in, *_a, **_k):
                                ctx_in.m8_platform_captions = {
                                    "youtube": "Triple-digit energy on a clear stretch.",
                                    "tiktok": "88 MPH send it",
                                }
                                ctx_in.m8_platform_titles = {
                                    "youtube": "88 MPH on the freeway"
                                }
                                ctx_in.ai_caption = "88 MPH send it"
                                ctx_in.ai_title = "88 MPH on the freeway"

                            with patch(
                                "stages.m8_engine.apply_selection_to_context",
                                side_effect=_apply,
                            ):
                                return await run_m8_caption_engine(
                                    ctx,
                                    frames=["a.jpg", "b.jpg", "c.jpg"],
                                    category="automotive",
                                    caption_style="punchy",
                                    caption_tone="cinematic",
                                    caption_voice="teacher",
                                    hashtag_style="mixed",
                                    hashtag_count=5,
                                    generate_title=True,
                                    generate_caption=True,
                                    generate_hashtags=True,
                                    model="gpt-4o",
                                    blocked_tags=[],
                                    always_tags=[],
                                    base_tags=[],
                                    db_pool=None,
                                    strategy=None,
                                )

    meta = asyncio.run(_run())
    assert meta.get("ok") is True
    assert meta.get("llm_tier") == "text_only"
    assert len(calls) == 4
    assert calls[0]["n_frames"] == 3
    assert calls[1]["max_compl"] > calls[0]["max_compl"]  # expand
    assert calls[2]["n_frames"] == 2
    assert calls[3]["n_frames"] == 0


def test_m8_ladder_stops_on_quota_without_extra_calls():
    ctx = _ctx()
    calls = []

    async def _fake(*, frames, prompt, model, max_completion_tokens, http_timeout_sec, temperature=0.55):
        calls.append(len(frames or []))
        return {}, {"prompt": 0, "completion": 0}, "openai_quota"

    async def _run():
        with patch("stages.m8_engine._call_openai_m8_json", new=AsyncMock(side_effect=_fake)):
            with patch(
                "stages.m8_engine.build_scene_graph",
                return_value={
                    "platforms": ["youtube"],
                    "vision": {},
                    "transcript": {},
                    "video_intelligence": {},
                },
            ):
                with patch("stages.m8_engine.m8_evidence_matrix_enabled", return_value=False):
                    return await run_m8_caption_engine(
                        ctx,
                        frames=["a.jpg"],
                        category="automotive",
                        caption_style="punchy",
                        caption_tone="cinematic",
                        caption_voice="teacher",
                        hashtag_style="mixed",
                        hashtag_count=5,
                        generate_title=True,
                        generate_caption=True,
                        generate_hashtags=True,
                        model="gpt-4o",
                        blocked_tags=[],
                        always_tags=[],
                        base_tags=[],
                        db_pool=None,
                        strategy=None,
                    )

    meta = asyncio.run(_run())
    assert meta.get("ok") is False
    assert meta.get("error_class") == "openai_quota"
    assert len(calls) == 1


def test_prompt_contains_creative_spine_and_authority():
    from stages.m8_engine import _build_m8_prompt

    ctx = _ctx()
    ctx.platforms = ["tiktok"]
    prompt = _build_m8_prompt(
        ctx,
        {
            "platforms": ["tiktok"],
            "hydration_story": "88 MPH near Flagstaff",
            "timeline": [{"t_seconds": 1, "kind": "osd", "text": "88 MPH"}],
            "geo": {"city": "Flagstaff"},
            "vision": {},
            "transcript": {},
        },
        "automotive",
        "punchy",
        "cinematic",
        "mixed",
        5,
        True,
        True,
        True,
        caption_voice_ui="teacher",
    )
    assert "CREATIVE AUTHORITY" in prompt
    assert "CREATIVE SPINE" in prompt
    # Spine must appear after PLATFORM RULES / near TASK (recency)
    spine_i = prompt.find("CREATIVE SPINE")
    task_i = prompt.find("\nTASK:")
    assert spine_i > 0 and task_i > spine_i
    assert "REJECTED by the ranker" not in prompt
    assert "Style: mixed" in prompt and "mix niche" in prompt
