"""Opt-in baselines + admin full-stack defaults + Whisper feed helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from stages.context import JobContext
from stages.m8_engine import _build_m8_prompt, build_scene_graph


def _ctx_with_speech(**kwargs) -> JobContext:
    base = dict(
        job_id="j-whisper",
        upload_id="u-whisper",
        user_id="user-1",
        platforms=["youtube", "tiktok"],
        filename="talk.mp4",
        video_info={"duration": 42.0, "audio_codec": "aac"},
        ai_transcript="We are rolling through Logandale at night keep it moving.",
        audio_context={
            "transcript": "We are rolling through Logandale at night keep it moving.",
            "gpt_audio_summary": "night drive energy, Logandale mention, keep-moving vibe",
            "transcript_language": "en",
            "transcript_duration": 41.2,
            "transcript_segments": [
                {"start": 0.0, "end": 3.5, "text": "We are rolling through Logandale"},
            ],
        },
        user_settings={
            "useAudioContext": True,
            "audioTranscription": True,
            "aiServiceSpeechToText": True,
            "aiServiceAudioSummary": True,
            "aiServiceCaptionWriter": True,
        },
        entitlements=SimpleNamespace(can_ai=True, allowed_ai_services=None, max_caption_frames=6),
        vision_context={},
        video_understanding={},
        output_artifacts={},
    )
    base.update(kwargs)
    return JobContext(**base)


def test_free_defaults_opt_in_off():
    from core.upload_baseline_defaults import (
        FREE_TIER_PROCESSING_DEFAULTS,
        UNIVERSAL_UPLOAD_BASELINE,
        apply_upload_baseline_defaults,
    )

    assert UNIVERSAL_UPLOAD_BASELINE["aiServiceSpeechToText"] is False
    assert FREE_TIER_PROCESSING_DEFAULTS["aiServiceSpeechToText"] is False
    free = apply_upload_baseline_defaults({}, tier="free")
    assert free["aiServiceSpeechToText"] is False
    assert free["aiServiceCaptionWriter"] is False
    assert free["autoCaptions"] is False
    assert free["useAudioContext"] is False
    assert free["aiServiceSceneUnderstanding"] is False
    assert free["aiServiceVideoAnalyzer"] is False
    assert free["thumbnailStudioEngineEnabled"] is False


def test_paid_active_defaults_speech_tl_vi_studio_on():
    from core.upload_baseline_defaults import apply_upload_baseline_defaults

    paid = apply_upload_baseline_defaults({}, tier="creator_lite", subscription_status="active")
    assert paid["aiServiceSpeechToText"] is True
    assert paid["useAudioContext"] is True
    assert paid["audioTranscription"] is True
    assert paid["aiServiceAudioSummary"] is True
    assert paid["autoCaptions"] is True
    assert paid["aiServiceCaptionWriter"] is True
    assert paid["aiServiceSceneUnderstanding"] is True
    assert paid["aiServiceVideoAnalyzer"] is True
    assert paid["aiServiceFrameInspector"] is True
    assert paid["thumbnailStudioEnabled"] is True
    assert paid["thumbnailStudioEngineEnabled"] is True
    assert paid["thumbnailRenderPipeline"] == "auto"
    # Explicit user off is preserved.
    kept_off = apply_upload_baseline_defaults(
        {"aiServiceSpeechToText": False, "useAudioContext": False},
        tier="studio",
        subscription_status="active",
    )
    assert kept_off["aiServiceSpeechToText"] is False
    assert kept_off["useAudioContext"] is False


def test_trialing_defaults_opt_in_off():
    from core.upload_baseline_defaults import apply_upload_baseline_defaults

    trial = apply_upload_baseline_defaults(
        {}, tier="creator_pro", subscription_status="trialing"
    )
    assert trial["aiServiceSpeechToText"] is False
    assert trial["autoCaptions"] is False
    assert trial["aiServiceSceneUnderstanding"] is False
    assert trial["aiServiceVideoAnalyzer"] is False
    assert trial["thumbnailStudioEngineEnabled"] is False
    assert trial["useAudioContext"] is False


def test_canceled_paid_tier_defaults_opt_in_off():
    from core.upload_baseline_defaults import apply_upload_baseline_defaults

    canceled = apply_upload_baseline_defaults(
        {}, tier="studio", subscription_status="canceled"
    )
    assert canceled["aiServiceSpeechToText"] is False
    assert canceled["thumbnailStudioEnabled"] is False
    assert canceled["aiServiceVideoAnalyzer"] is False


def test_admin_defaults_full_stack_on():
    from core.upload_baseline_defaults import apply_upload_baseline_defaults

    admin = apply_upload_baseline_defaults({}, tier="free", role="admin")
    assert admin["aiServiceSpeechToText"] is True
    assert admin["aiServiceCaptionWriter"] is True
    assert admin["aiServiceSceneUnderstanding"] is True
    assert admin["aiServiceVideoAnalyzer"] is True
    assert admin["autoCaptions"] is True
    assert admin["useAudioContext"] is True

    master = apply_upload_baseline_defaults({}, tier="master_admin", role="user")
    assert master["aiServiceSpeechToText"] is True


def test_whisper_pref_default_off_when_unset():
    from stages.ai_service_costs import user_pref_ai_service_enabled

    assert user_pref_ai_service_enabled({}, "audio_whisper", default=False) is False
    assert (
        user_pref_ai_service_enabled(
            {"aiServiceSpeechToText": True}, "audio_whisper", default=False
        )
        is True
    )


def test_null_stt_pref_uses_default_arg():
    from stages.ai_service_costs import _pref_true

    assert _pref_true({"aiServiceSpeechToText": None}, "aiServiceSpeechToText", False) is False
    assert _pref_true({"aiServiceSpeechToText": None}, "aiServiceSpeechToText", True) is True


def test_m8_scene_graph_carries_transcript_and_gpt_summary():
    ctx = _ctx_with_speech()
    sg = build_scene_graph(ctx, "automotive")
    assert "Logandale" in (sg.get("transcript") or {}).get("text", "")
    env = sg.get("audio_environment") or {}
    assert "night drive" in str(env.get("gpt_audio_summary") or "")


def test_m8_prompt_requires_speech_and_summary_discipline():
    ctx = _ctx_with_speech()
    sg = build_scene_graph(ctx, "automotive")
    prompt = _build_m8_prompt(
        ctx,
        sg,
        "automotive",
        "conversational",
        "energetic",
        "mixed",
        5,
        True,
        True,
        True,
        historical={},
        strategy=None,
        include_evidence_matrix=False,
        caption_voice_ui="default",
    )
    assert "Logandale" in prompt
    assert "gpt_audio_summary" in prompt or "night drive" in prompt
    assert "AUDIO DISCIPLINE" in prompt or "transcript.text" in prompt


def test_thumbnail_brief_keeps_rich_speech_when_hydration_present():
    ctx = _ctx_with_speech()
    ctx.hydration_payload = {
        "v": 1,
        "category": "automotive",
        "anchor_phrase": "Logandale night",
        "evidence": {
            "speech": {
                "phrase": "We are rolling through Logandale",
                "summary": "night drive energy, Logandale mention",
            }
        },
        "fusion_summary": "",
        "hydration_story": "",
        "signal_hashtags": [],
    }
    brief = ctx.get_thumbnail_brief_vars(category="automotive")
    speech = str(brief.get("speech_context") or "")
    assert "Logandale" in speech
    assert "summary" in speech.lower() or "night drive" in speech


def test_hydration_payload_speech_includes_summary():
    from services.hydration_payload import build_hydration_payload, hydration_brief_strings

    ctx = _ctx_with_speech()
    hp = build_hydration_payload(ctx, category="automotive", category_source="test")
    speech = (hp.get("evidence") or {}).get("speech") or {}
    assert "Logandale" in str(speech.get("phrase") or "")
    assert "night drive" in str(speech.get("summary") or "")
    brief = hydration_brief_strings(hp)
    assert "Logandale" in str(brief.get("speech_context") or "")


def test_audio_stage_stores_transcript_then_gpt_summary(tmp_path: Path):
    import asyncio

    from stages.audio_stage import run_audio_context_stage

    (tmp_path / "v.mp4").write_bytes(b"fake")

    ctx = _ctx_with_speech(
        ai_transcript="",
        audio_context={},
        processed_video_path=tmp_path / "v.mp4",
        local_video_path=tmp_path / "v.mp4",
        temp_dir=tmp_path,
    )

    whisper_payload = {
        "text": "We are rolling through Logandale at night keep it moving.",
        "language": "en",
        "duration": 41.0,
        "segments": [{"start": 0.0, "end": 2.0, "text": "We are rolling through Logandale"}],
    }

    async def _run():
        with (
            patch("stages.audio_stage._extract_audio_wav", new=AsyncMock(return_value=True)),
            patch("stages.audio_stage._transcribe_wav", new=AsyncMock(return_value=whisper_payload)),
            patch(
                "stages.audio_stage._build_structured_transcript",
                new=AsyncMock(return_value={"topics": ["logandale"], "engine": "test"}),
            ),
            patch(
                "stages.audio_stage._gpt_audio_summary_from_transcript",
                new=AsyncMock(return_value="night drive energy, Logandale mention"),
            ),
            patch("stages.audio_stage.OPENAI_API_KEY", "sk-test"),
            patch("stages.audio_stage.AUDIO_STAGE_ENABLED", True),
        ):
            return await run_audio_context_stage(ctx)

    out = asyncio.run(_run())
    assert "Logandale" in (out.ai_transcript or "")
    assert "night drive" in str((out.audio_context or {}).get("gpt_audio_summary") or "")


def test_legacy_spoken_block_shape_for_prose():
    ctx = _ctx_with_speech(ai_transcript="We are rolling through Logandale at night.")
    raw_tx = getattr(ctx, "ai_transcript", None)
    assert raw_tx and "Logandale" in raw_tx
    transcript_block = (
        "\n━━ SPOKEN CONTENT (speech-to-text — factual; do not contradict) ━━\n"
        f"{str(raw_tx).strip()[:6000]}\n"
    )
    assert "SPOKEN CONTENT" in transcript_block


def test_voice_shaped_caption_prefers_whisper_phrase():
    from services.hydration_enforcer import EvidencePool, _voice_shaped_caption_from_pool

    pool = EvidencePool()
    pool.transcript_phrase = "We are rolling through Logandale keep it moving"
    pool.video_understanding_phrase = "dashcam road at night"
    pool.place_name = "Logandale"
    ctx = _ctx_with_speech()
    cap = _voice_shaped_caption_from_pool(pool, ctx)
    assert "Logandale" in cap
    assert "rolling" in cap.lower() or "Logandale" in cap
    assert "Out here with what the clip actually shows" not in cap
