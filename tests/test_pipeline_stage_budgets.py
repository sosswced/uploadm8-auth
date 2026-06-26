"""Stage timeout budget helpers."""

from stages import pipeline_stage_budgets as budgets


def test_get_all_budgets_keys():
    b = budgets.get_all_budgets()
    assert "watermark" in b
    assert "transcode" in b
    assert "thumbnail" in b
    assert b["thumbnail"] >= 60


def test_watermark_vf_builder():
    from stages.watermark_stage import build_drawtext_filter

    vf = build_drawtext_filter(text="Upload M8", font_size=42)
    assert "drawtext=" in vf
    assert "Upload M8" in vf
