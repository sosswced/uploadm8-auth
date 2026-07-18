"""Upload thumbnail preference merge + studio R2 key helpers."""

from __future__ import annotations

from types import SimpleNamespace

from core.content_attribution import build_content_attribution_snapshot
from routers.thumbnail_studio_api import _collect_studio_job_r2_keys
from services.thumbnail_studio_strategy import (
    read_thumbnail_studio_default_strategy,
    strategy_audience_niche,
)
from services.upload.prefs import merge_upload_init_thumbnail_preferences


def test_merge_engine_off_also_disables_pikzels():
    prefs: dict = {
        "thumbnail_pikzels_enabled": True,
        "thumbnailPikzelsEnabled": True,
        "thumbnail_studio_engine_enabled": True,
    }
    merge_upload_init_thumbnail_preferences(
        prefs, SimpleNamespace(thumbnail_use_studio_engine=False)
    )
    assert prefs["thumbnail_studio_engine_enabled"] is False
    assert prefs["thumbnailPikzelsEnabled"] is False
    assert prefs["thumbnail_pikzels_enabled"] is False


def test_merge_engine_off_wins_over_pikzels_true():
    prefs: dict = {
        "thumbnail_pikzels_enabled": True,
        "thumbnailPikzelsEnabled": True,
    }
    merge_upload_init_thumbnail_preferences(
        prefs,
        SimpleNamespace(
            thumbnail_use_studio_engine=False,
            thumbnail_use_pikzels=True,
        ),
    )
    assert prefs["thumbnail_studio_engine_enabled"] is False
    assert prefs["thumbnail_pikzels_enabled"] is False
    assert prefs["thumbnailPikzelsEnabled"] is False


def test_content_attribution_absent_studio_prefs_default_false():
    snap = build_content_attribution_snapshot(
        user_settings={},
        strategy={"outputs": {}},
        category="cars",
        used_m8_engine=False,
        caption_style_ui="story",
        caption_tone_ui="authentic",
        caption_voice_ui="default",
        hashtag_style="mixed",
        hashtag_count=0,
        caption_frame_count=0,
        generate_hashtags=False,
        output_artifacts={},
    )
    assert snap["thumbnail_studio_enabled"] is False
    assert snap["thumbnail_studio_engine_enabled"] is False
    assert snap["thumbnail_persona_enabled"] is False


def test_merge_persona_id_enables_persona():
    prefs: dict = {}
    merge_upload_init_thumbnail_preferences(
        prefs,
        SimpleNamespace(
            thumbnail_use_studio_engine=True,
            thumbnail_use_persona=True,
            thumbnail_persona_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            thumbnail_persona_strength=80,
        ),
    )
    assert prefs["thumbnail_default_persona_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert prefs["thumbnail_persona_enabled"] is True
    assert prefs["thumbnail_persona_strength"] == 80
    assert prefs["thumbnail_pikzels_enabled"] is True


def test_collect_studio_job_r2_keys_includes_previews_and_ab_pack():
    rows = [
        {"variant_json": {"preview_r2_key": "thumbnail-studio/previews/u1/j1/variant_1.jpg"}},
        {"variant_json": '{"preview_r2_key":"thumbnail-studio/previews/u1/j1/variant_2.jpg"}'},
        {"variant_json": {"pikzels_cdn_url": "https://cdn.example/x.jpg"}},
    ]
    keys = _collect_studio_job_r2_keys("u1", "j1", rows)
    assert "thumbnail-studio/previews/u1/j1/variant_1.jpg" in keys
    assert "thumbnail-studio/previews/u1/j1/variant_2.jpg" in keys
    assert "thumbnail-studio/ab-packs/u1/j1.zip" in keys
    assert len(keys) == 3


def test_strategy_reader_prefers_studio_canonical_key():
    prefs = {
        "thumbnailDefaultStrategy": {"audience_niche": "legacy"},
        "thumbnailStudioDefaultStrategy": {"audience_niche": "cars", "layout_pattern": "face_left"},
    }
    strat = read_thumbnail_studio_default_strategy(prefs)
    assert strat["audience_niche"] == "cars"
    assert strategy_audience_niche(prefs) == "cars"


def test_content_attribution_includes_studio_ml_fields():
    snap = build_content_attribution_snapshot(
        user_settings={
            "thumbnailStudioEnabled": True,
            "thumbnailStudioEngineEnabled": True,
            "thumbnailPersonaEnabled": True,
            "thumbnailDefaultPersonaId": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "thumbnailPersonaStrength": 80,
            "thumbnailStudioDefaultStrategy": {
                "audience_niche": "cars",
                "layout_pattern": "face_left_text_right",
            },
            "thumbnail_selection_mode": "ai",
            "thumbnail_render_pipeline": "auto",
            "styled_thumbnails": True,
            "auto_thumbnails": True,
        },
        strategy={"outputs": {}},
        category="cars",
        used_m8_engine=True,
        caption_style_ui="story",
        caption_tone_ui="authentic",
        caption_voice_ui="default",
        hashtag_style="mixed",
        hashtag_count=3,
        caption_frame_count=6,
        generate_hashtags=True,
        output_artifacts={
            "thumbnail_render_method": "studio_renderer",
            "studio_render_report": {
                "persona_kind": "linked",
                "platform_render_methods": {
                    "youtube": {"ctr_score": 0.71, "pikzels_main_score": 0.66}
                },
            },
        },
    )
    assert snap["thumbnail_studio_enabled"] is True
    assert snap["thumbnail_persona_enabled"] is True
    assert snap["thumbnail_persona_strength"] == 80
    assert snap["thumbnail_audience_niche"] == "cars"
    assert snap["thumbnail_layout_pattern"] == "face_left_text_right"
    assert snap["thumbnail_engine_mode"] == "uploadm8_pikzels_v2_pipeline"
    assert snap["studio_variant_ctr_score"] == 0.71
    assert snap["studio_pikzels_main_score"] == 0.66
