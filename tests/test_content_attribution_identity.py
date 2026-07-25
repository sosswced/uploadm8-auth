"""Phase 4: content_identity fields land on the attribution snapshot."""

from __future__ import annotations

import json

from core.content_attribution import build_content_attribution_snapshot


def test_snapshot_includes_identity_fields():
    snap = build_content_attribution_snapshot(
        user_settings={},
        strategy={},
        category="gardening",
        used_m8_engine=True,
        caption_style_ui="story",
        caption_tone_ui="authentic",
        caption_voice_ui="default",
        hashtag_style="mixed",
        hashtag_count=5,
        caption_frame_count=6,
        generate_hashtags=True,
        output_artifacts={
            "thumbnail_render_method": "ai_edit",
            "content_identity_v1": json.dumps({
                "version": 1,
                "subject": "raised-bed tomato harvest",
                "domain_tags": [{"tag": "gardening", "confidence": 0.91}],
                "hero_facts": [
                    {"text": "40 ripe tomatoes", "class": "count"},
                    {"text": "first harvest", "class": "transcript"},
                ],
                "confidence": "high",
                "novel_content": False,
            }),
        },
    )
    assert snap["identity_domain_tag"] == "gardening"
    assert snap["identity_domain_confidence"] == 0.91
    assert snap["identity_hero_fact_class"] == "count"
    assert snap["identity_confidence"] == "high"
    assert snap["identity_novel_content"] is False
    assert snap["thumbnail_engine_mode"] == "uploadm8_gpt_image_edit_pipeline"
    assert "tomato" in snap["identity_subject"]


def test_snapshot_without_identity_is_empty_safe():
    snap = build_content_attribution_snapshot(
        user_settings={},
        strategy={},
        category="general",
        used_m8_engine=False,
        caption_style_ui="story",
        caption_tone_ui="authentic",
        caption_voice_ui="default",
        hashtag_style="mixed",
        hashtag_count=0,
        caption_frame_count=6,
        generate_hashtags=False,
        output_artifacts={},
    )
    assert snap["identity_domain_tag"] == ""
    assert snap["identity_hero_fact_class"] == ""
    assert snap["identity_novel_content"] is None
