"""Thumbnail styled render order — studio must keep template fallback."""

from stages.thumbnail_stage import _thumbnail_styled_render_order


def test_studio_ok_includes_template_fallback():
    assert _thumbnail_styled_render_order("auto", studio_ok=True, ai_edit_ok=False) == [
        "studio",
        "template",
    ]
    assert _thumbnail_styled_render_order(
        "studio_renderer", studio_ok=True, ai_edit_ok=True
    ) == ["studio", "template"]


def test_studio_ok_with_stickers_then_template():
    assert _thumbnail_styled_render_order(
        "auto", studio_ok=True, ai_edit_ok=False, sticker_ok=True
    ) == ["sticker", "studio", "template"]


def test_no_studio_keeps_template_only_paths():
    assert _thumbnail_styled_render_order("template", studio_ok=False, ai_edit_ok=False) == [
        "template"
    ]
    assert _thumbnail_styled_render_order("auto", studio_ok=False, ai_edit_ok=True) == [
        "ai_edit",
        "template",
    ]


def test_pipeline_none_empty():
    assert _thumbnail_styled_render_order("none", studio_ok=True, ai_edit_ok=True) == []
