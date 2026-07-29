"""M8 OpenAI retry ladder: full → compact → text_only; stop on quota."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from stages.m8_engine import run_m8_caption_engine


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


def test_m8_ladder_falls_to_text_only_after_full_and_compact_fail():
    ctx = _ctx()
    calls = []

    async def _fake(*, frames, prompt, model, max_completion_tokens, http_timeout_sec):
        calls.append(
            {
                "n_frames": len(frames or []),
                "max_compl": max_completion_tokens,
            }
        )
        if len(calls) < 3:
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
                                    model="gpt-4o-mini",
                                    blocked_tags=[],
                                    always_tags=[],
                                    base_tags=[],
                                    db_pool=None,
                                    strategy=None,
                                )

    meta = asyncio.run(_run())
    assert meta.get("ok") is True
    assert meta.get("llm_tier") == "text_only"
    assert len(calls) == 3
    assert calls[0]["n_frames"] == 3
    assert calls[1]["n_frames"] == 2
    assert calls[2]["n_frames"] == 0


def test_m8_ladder_stops_on_quota_without_extra_calls():
    ctx = _ctx()
    calls = []

    async def _fake(*, frames, prompt, model, max_completion_tokens, http_timeout_sec):
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
                        model="gpt-4o-mini",
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
