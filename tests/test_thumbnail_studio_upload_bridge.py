"""Studio winner → upload cover bridge + skip diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from services.thumbnail_studio_strategy import thumbnail_strategy_from_variant
from services.thumbnail_studio_upload_bridge import strategy_apply_mode, strategy_preview_r2_key
from services.upload.prefs import merge_upload_init_thumbnail_preferences
from services.upload.thumbnails import (
    pikzels_template_thumbnail_warning,
    studio_thumb_diagnostics_from_artifacts,
)
from stages.thumbnail_stage import studio_pipeline_skip_reason


def test_strategy_from_variant_includes_preview_r2_and_cover_direct():
    job = {
        "id": "11111111-1111-1111-1111-111111111111",
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "youtube_video_id": "dQw4w9WgXcQ",
        "niche": "cars",
        "topic": "night drive",
        "closeness": 70,
        "competitor_gap_mode": False,
        "persona_id": None,
    }
    variant = {
        "name": "Face left",
        "layout_pattern": "face_left",
        "format_key": "dyn-cars-0001",
        "headline": "NIGHT RUN",
        "preview_r2_key": "thumbnail-studio/previews/u1/j1/variant_1.jpg",
        "face_scale": 0.45,
        "text_position": "right",
        "contrast_profile": "high",
        "emotion": "hype",
    }
    strat = thumbnail_strategy_from_variant(
        job_row=job,
        variant_id="22222222-2222-2222-2222-222222222222",
        variant_json=variant,
    )
    assert strat["preview_r2_key"].endswith("variant_1.jpg")
    assert strat["apply_mode"] == "cover_direct"
    assert strategy_apply_mode(strat) == "cover_direct"
    assert strategy_preview_r2_key(strat).endswith("variant_1.jpg")


def test_merge_engine_on_defaults_strict_studio():
    prefs: dict = {}
    merge_upload_init_thumbnail_preferences(
        prefs, SimpleNamespace(thumbnail_use_studio_engine=True)
    )
    assert prefs["thumbnail_studio_strict"] is True
    assert prefs["thumbnailStudioStrict"] is True


def test_studio_skip_reason_tier_lacks_ai_styling(monkeypatch):
    monkeypatch.setattr(
        "stages.thumbnail_stage.studio_renderer_enabled",
        lambda: True,
    )
    ent = SimpleNamespace(can_custom_thumbnails=True, can_ai_thumbnail_styling=False)
    reason = studio_pipeline_skip_reason(
        {"thumbnail_studio_enabled": True, "thumbnail_studio_engine_enabled": True},
        ent,
        require_auto_thumbnails=False,
    )
    assert reason == "tier_lacks_ai_thumbnail_styling"


def test_pikzels_warning_requested_but_skipped():
    arts = {
        "thumbnail_render_method": "template",
        "pikzels_requested_but_skipped": "1",
        "studio_render_report": {
            "skip_reason": "tier_lacks_ai_thumbnail_styling",
            "pikzels_requested_but_skipped": True,
            "platform_render_methods": {"youtube": {"succeeded_with": "template"}},
        },
    }
    warn = pikzels_template_thumbnail_warning(arts)
    assert warn is not None
    assert warn["code"] == "pikzels_requested_skipped"
    assert "Creator Pro" in warn["message"]
    diag = studio_thumb_diagnostics_from_artifacts(arts)
    assert diag["pikzels_requested_but_skipped"] is True
    assert diag["platform_render_methods"]["youtube"]["succeeded_with"] == "template"


def test_pikzels_warning_insufficient_credits_from_provider_trace():
    arts = {
        "thumbnail_render_method": "",
        "provider_error_trace": [
            {
                "provider": "pikzels",
                "http_status": 402,
                "provider_code": "insufficient_credits",
                "message": "INSUFFICIENT_CREDITS: Your API balance is too low for this request.",
            }
        ],
    }
    warn = pikzels_template_thumbnail_warning(arts)
    assert warn is not None
    assert warn["code"] == "pikzels_insufficient_credits"
    assert warn["skip_reason"] == "pikzels_insufficient_credits"
    assert "insufficient credits" in warn["message"].lower()
    assert "Settings were already fine" in warn["message"]


def test_youtube_thumb_error_codes():
    from stages.publish_stage import _youtube_thumb_push_error_code

    assert _youtube_thumb_push_error_code(403, "channel must be verified") == "youtube_channel_not_verified"
    assert _youtube_thumb_push_error_code(403, "custom thumbnail permission") == "youtube_custom_thumbs_disabled"
    assert _youtube_thumb_push_error_code(503, "backend") == "youtube_thumb_server_error"
