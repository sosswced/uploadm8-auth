"""Unit tests for expanded watermark branding (logo, fonts, sponsored-by)."""

from stages.db import normalize_watermark_settings, sanitize_watermark_font_family
from stages.watermark_stage import (
    build_drawtext_filter,
    build_logo_overlay_chain,
    format_watermark_display_text,
    should_apply_watermark,
    watermark_requires_logo_prepass,
)


class _Ent:
    def __init__(self, can_watermark: bool):
        self.can_watermark = can_watermark


class _Ctx:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_normalize_watermark_defaults_and_logo_mode_fallback():
    s = normalize_watermark_settings({})
    assert s["text"] == "Upload M8"
    assert s["mode"] == "text"
    assert s["font_family"] == "dejavu"
    assert s["sponsored_prefix"] is False
    # logo/both without key → text
    s2 = normalize_watermark_settings({"mode": "both", "logo_r2_key": ""})
    assert s2["mode"] == "text"
    s3 = normalize_watermark_settings(
        {"mode": "both", "logo_r2_key": "watermarks/admin/logo.png"}
    )
    assert s3["mode"] == "both"
    assert s3["logo_r2_key"] == "watermarks/admin/logo.png"


def test_sanitize_font_family():
    assert sanitize_watermark_font_family("arial") == "arial"
    assert sanitize_watermark_font_family("nope") == "dejavu"


def test_sponsored_prefix_formatting():
    text = format_watermark_display_text(
        {
            "text": "UploadM8",
            "sponsored_prefix": True,
            "sponsored_prefix_text": "Sponsored by",
        }
    )
    assert text.startswith("Sponsored by ")
    assert "UploadM8" in text
    # No double prefix
    text2 = format_watermark_display_text(
        {
            "text": "Sponsored by UploadM8",
            "sponsored_prefix": True,
            "sponsored_prefix_text": "Sponsored by",
        }
    )
    assert text2.count("Sponsored by") == 1


def test_logo_prepass_and_overlay_filter():
    assert watermark_requires_logo_prepass(
        {"mode": "both", "logo_r2_key": "watermarks/admin/logo.png"}
    )
    assert not watermark_requires_logo_prepass({"mode": "text", "logo_r2_key": "watermarks/admin/logo.png"})
    chain = build_logo_overlay_chain(
        logo_width=120,
        opacity=0.9,
        position="bottom-left",
        include_drawtext="drawtext=text='x'",
    )
    assert "[1:v]scale=120:-1" in chain
    assert "overlay=" in chain
    assert "drawtext=text='x'" in chain


def test_drawtext_includes_font_and_color():
    vf = build_drawtext_filter(
        text="Sponsored by Upload M8",
        font_size=36,
        opacity=0.8,
        position="bottom-right",
        fontfile="C:/Windows/Fonts/arial.ttf",
        text_color="#ff6600",
        font_weight="bold",
    )
    assert "drawtext=text=" in vf
    assert "0xFF6600@0.8" in vf
    assert "fontfile=" in vf


def test_should_apply_watermark_free_and_opt_in():
    free = _Ctx(entitlements=_Ent(True), user_settings={}, apply_watermark=None)
    assert should_apply_watermark(free) is True
    paid = _Ctx(entitlements=_Ent(False), user_settings={}, apply_watermark=None)
    assert should_apply_watermark(paid) is False
    opt = _Ctx(
        entitlements=_Ent(False),
        user_settings={"sponsorWatermarkOptIn": True},
        apply_watermark=None,
    )
    assert should_apply_watermark(opt) is True
    explicit = _Ctx(entitlements=_Ent(False), user_settings={}, apply_watermark=True)
    assert should_apply_watermark(explicit) is True
