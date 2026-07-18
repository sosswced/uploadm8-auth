"""Studio → Upload apply-mode bridge contracts."""

from __future__ import annotations

from types import SimpleNamespace

from services.thumbnail_apply_mode import (
    allow_persona_on_render,
    allow_youtube_support_image,
    apply_structured_strategy_to_brief,
    bind_source_ids_into_prefs,
    merge_strategy_into_studio_options,
    normalize_apply_mode,
    normalize_ref_persona_mode,
    resolve_ref_persona_mode,
    strategy_summary_for_ui,
    structured_strategy_payload,
    to_bridge_apply_mode,
)
from services.thumbnail_studio_upload_bridge import strategy_apply_mode, strategy_preview_r2_key
from services.upload.prefs import merge_upload_init_thumbnail_preferences


def test_normalize_apply_mode_aliases():
    assert normalize_apply_mode("cover_direct") == "pinned_cover"
    assert normalize_apply_mode("support_image") == "fresh_generate"
    assert normalize_apply_mode("pinned") == "pinned_cover"
    assert to_bridge_apply_mode("pinned_cover") == "cover_direct"
    assert to_bridge_apply_mode("strategy_only") == "strategy_only"


def test_ref_persona_xor_rules():
    face = {"thumbnail_persona_enabled": True, "thumbnail_apply_mode": "fresh_generate"}
    assert resolve_ref_persona_mode(face) == "face_brand"
    assert allow_youtube_support_image(face) is False
    assert allow_persona_on_render(face) is True

    recreate = {
        "thumbnail_persona_enabled": False,
        "thumbnail_apply_mode": "fresh_generate",
        "thumbnail_ref_persona_mode": "recreate_style",
    }
    assert allow_youtube_support_image(recreate) is True
    assert allow_persona_on_render(recreate) is False

    strategy_only = {"thumbnail_apply_mode": "strategy_only"}
    assert allow_youtube_support_image(strategy_only) is False


def test_merge_presign_apply_mode_and_source_ids():
    prefs: dict = {}
    merge_upload_init_thumbnail_preferences(
        prefs,
        SimpleNamespace(
            thumbnail_use_studio_engine=True,
            thumbnail_apply_mode="pinned_cover",
            thumbnail_ref_persona_mode="recreate_style",
            thumbnail_source_job_id="job-1",
            thumbnail_source_variant_id="var-1",
        ),
    )
    assert prefs["thumbnail_apply_mode"] == "pinned_cover"
    assert prefs["thumbnailApplyMode"] == "pinned_cover"
    assert prefs["thumbnail_source_job_id"] == "job-1"
    assert prefs["thumbnail_source_variant_id"] == "var-1"
    assert prefs["thumbnail_ref_persona_mode"] == "recreate_style"


def test_structured_strategy_into_brief_and_options():
    strategy = {
        "format_key": "face_left",
        "layout_pattern": "face_left_text_right",
        "reference_strength": 80,
        "emotion": "excited",
        "job_id": "j1",
        "variant_id": "v1",
        "platforms": {"tiktok": {"format_key": "vertical_hook"}},
    }
    brief = apply_structured_strategy_to_brief({}, strategy)
    assert brief["_uploadm8_strategy_structured"]["format_key"] == "face_left"
    assert brief["_uploadm8_reference_strength"] == 80
    opts = merge_strategy_into_studio_options({}, strategy, platform="tiktok")
    assert opts["strategy_structured"]["format_key"] == "vertical_hook"
    assert opts["reference_strength"] == 80
    assert structured_strategy_payload(strategy)["layout_pattern"] == "face_left_text_right"


def test_bind_source_ids_and_summary():
    prefs: dict = {"thumbnail_studio_default_strategy": {"layout_name": "Bold"}}
    bind_source_ids_into_prefs(prefs, job_id="aaaaaaaa", variant_id="bbbbbbbb")
    assert prefs["thumbnail_source_job_id"] == "aaaaaaaa"
    assert prefs["thumbnailStudioDefaultStrategy"]["source_variant_id"] == "bbbbbbbb"
    line = strategy_summary_for_ui(
        {"selected_at": "2026-07-12T12:00:00+00:00", "layout_name": "Bold", "job_id": "aaaaaaaa", "variant_id": "bbbbbbbb"},
        prefs,
    )
    assert "Saved run" in line
    assert "Variant" in line


def test_bridge_strategy_apply_mode_pref_overlay():
    strategy = {"apply_mode": "support_image", "preview_r2_key": "thumbnail-studio/previews/u/j/v.jpg"}
    assert strategy_preview_r2_key(strategy).endswith("v.jpg")
    assert strategy_apply_mode(strategy) == "support_image"
    assert strategy_apply_mode(strategy, {"thumbnail_apply_mode": "pinned_cover"}) == "cover_direct"


def test_normalize_ref_persona_mode():
    assert normalize_ref_persona_mode("face") == "face_brand"
    assert normalize_ref_persona_mode("both") == "both"
