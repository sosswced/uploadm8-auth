"""Settings prefs must reach upload JobContext, Trill, and Thumbnail Studio paths."""

from __future__ import annotations

from stages.context import JobContext, create_context
from stages.entitlements import Entitlements
from stages.youtube_copyright_shorts import _trim_pref_enabled
from services.trill_content import generate_trill_content


def _ents() -> Entitlements:
    # Minimal entitlements stub — create_context only needs the object.
    try:
        return Entitlements()  # type: ignore[call-arg]
    except TypeError:
        from stages.entitlements import get_entitlements_for_tier

        return get_entitlements_for_tier("pro")


def test_create_context_uj_overlay_wins_for_studio_trim_burn_trill():
    live = {
        "thumbnailStudioEnabled": False,
        "thumbnailPersonaEnabled": False,
        "thumbnailDefaultPersonaId": "live-persona",
        "youtubeShortsCopyrightTrim": False,
        "tiktokBurnStyledCover": False,
        "trillMinScore": 10,
        "captionStyle": "factual",
        "speeding_mph": 40,
        "euphoria_mph": 60,
    }
    uj = {
        "thumbnailStudioEnabled": True,
        "thumbnailStudioEngineEnabled": True,
        "thumbnailStudioStrict": True,
        "thumbnailPersonaEnabled": True,
        "thumbnailDefaultPersonaId": "studio-persona",
        "thumbnailPersonaStrength": 88,
        "thumbnailApplyMode": "force_persona",
        "thumbnailRefPersonaMode": "locked",
        "youtubeShortsCopyrightTrim": True,
        "tiktokBurnStyledCover": True,
        "trillMinScore": 55,
        "captionStyle": "punchy",
        "captionTone": "cinematic",
        "captionVoice": "teacher",
        "speeding_mph": 50,
        "euphoria_mph": 101,
        "aiServiceMusicDetection": True,
        "useAudioContext": True,
    }
    upload = {
        "id": "u1",
        "user_id": "user1",
        "filename": "clip.mp4",
        "r2_key": "k",
        "platforms": ["youtube", "tiktok"],
        "user_preferences": uj,
    }
    ctx = create_context(
        {"job_id": "j1", "upload_id": "u1", "user_id": "user1"},
        upload,
        dict(live),
        _ents(),
    )
    us = ctx.user_settings
    assert us["thumbnailStudioEnabled"] is True
    assert us["thumbnailStudioEngineEnabled"] is True
    assert us["thumbnailStudioStrict"] is True
    assert us["thumbnailPersonaEnabled"] is True
    assert us["thumbnailDefaultPersonaId"] == "studio-persona"
    assert us["thumbnailPersonaStrength"] == 88
    assert us["thumbnailApplyMode"] == "force_persona"
    assert us["youtubeShortsCopyrightTrim"] is True
    assert us["tiktokBurnStyledCover"] is True
    assert us["trillMinScore"] == 55
    assert us["captionStyle"] == "punchy"
    assert us["captionTone"] == "cinematic"
    assert us["captionVoice"] == "teacher"
    assert us["speeding_mph"] == 50
    assert us["euphoria_mph"] == 101
    assert _trim_pref_enabled(ctx) is True


def test_hydrate_includes_flow_critical_pairs():
    from routers.preferences import _hydrate_snake_camel_mirror

    d = {
        "youtube_shorts_copyright_trim": True,
        "tiktok_burn_styled_cover": False,
        "thumbnail_apply_mode": "force",
        "speeding_mph": 50,
        "caption_style": "punchy",
    }
    _hydrate_snake_camel_mirror(d)
    assert d["youtubeShortsCopyrightTrim"] is True
    assert d["tiktokBurnStyledCover"] is False
    assert d["thumbnailApplyMode"] == "force"
    assert d["speedingMph"] == 50
    assert d["captionStyle"] == "punchy"


def test_trill_content_prompt_includes_caption_prefs(monkeypatch):
    """generate_trill_content must honor Settings caption style/tone/voice."""
    captured = {}

    class _Resp:
        class choices:
            class message:
                content = '{"title":"T","caption":"C","hashtags":["a"]}'

            message = message()

        usage = type("U", (), {"total_tokens": 10})()

    class _Completions:
        @staticmethod
        def create(**kwargs):
            captured["kwargs"] = kwargs
            return type(
                "R",
                (),
                {
                    "choices": [
                        type(
                            "Ch",
                            (),
                            {
                                "message": type(
                                    "M",
                                    (),
                                    {
                                        "content": '{"title":"T","caption":"C","hashtags":["a"]}'
                                    },
                                )()
                            },
                        )()
                    ],
                    "usage": type("U", (), {"total_tokens": 12})(),
                },
            )()

    class _OpenAI:
        api_key = None
        chat = type("Chat", (), {"completions": _Completions})()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("services.trill_content.OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("openai.api_key", "sk-test", raising=False)
    monkeypatch.setattr(
        "openai.chat.completions.create",
        _Completions.create,
        raising=False,
    )

    # Import openai inside function — patch the module after import path
    import openai

    monkeypatch.setattr(openai, "api_key", "sk-test", raising=False)
    monkeypatch.setattr(openai.chat.completions, "create", _Completions.create)

    out = generate_trill_content(
        {
            "trill_score": 80,
            "speed_bucket": "SPIRITED",
            "place_name": "Austin",
            "state": "TX",
        },
        {
            "trillOpenaiModel": "gpt-4o",
            "captionStyle": "punchy",
            "captionTone": "cinematic",
            "captionVoice": "teacher",
        },
    )
    prompt = captured["kwargs"]["messages"][1]["content"]
    assert "CAPTION STYLE: punchy" in prompt
    assert "CAPTION TONE: cinematic" in prompt
    assert "CAPTION VOICE / PERSONA: teacher" in prompt
    assert captured["kwargs"]["model"] == "gpt-4o"
    assert out["title"] == "T"


def test_field_map_trill_openai_model_alias():
    """Regression: trillOpenaiModel must map to trill_openai_model, not openai_model only."""
    import inspect
    from stages import db as db_stage

    src = inspect.getsource(db_stage.load_user_settings)
    assert '"trillOpenaiModel": "trill_openai_model"' in src
    assert '"youtubeShortsCopyrightTrim": "youtube_shorts_copyright_trim"' in src
    assert '"tiktokBurnStyledCover": "tiktok_burn_styled_cover"' in src
