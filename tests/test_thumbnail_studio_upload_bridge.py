"""Studio winner → upload cover bridge + skip diagnostics."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from PIL import Image

from services.thumbnail_studio_strategy import thumbnail_strategy_from_variant
from services.thumbnail_studio_upload_bridge import (
    COVER_DIRECT_PLATFORMS,
    VERTICAL_PLATFORMS,
    apply_studio_winner_to_upload_thumbs,
    hydrate_bridge_strategy,
    strategy_apply_mode,
    strategy_preview_r2_key,
)
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
    assert strat["platforms"]["facebook"]["apply_mode"] == "letterbox"
    assert "facebook" in VERTICAL_PLATFORMS
    assert COVER_DIRECT_PLATFORMS == frozenset({"youtube"})


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


def test_cover_direct_letterboxes_meta_and_tiktok(tmp_path, monkeypatch):
    """Pinned Studio JPEG: YT copy 16:9; FB/IG/TT letterbox 9:16."""
    src = tmp_path / "winner.jpg"
    Image.new("RGB", (1280, 720), (20, 40, 80)).save(src, format="JPEG")

    async def _fake_dl(r2_key, dest):
        dest.write_bytes(src.read_bytes())
        return True

    monkeypatch.setattr(
        "services.thumbnail_studio_upload_bridge.download_studio_preview_to_path",
        _fake_dl,
    )
    strategy = {
        "apply_mode": "cover_direct",
        "preview_r2_key": "thumbnail-studio/previews/u/j/v.jpg",
    }
    report: dict = {}

    async def _run():
        return await apply_studio_winner_to_upload_thumbs(
            strategy=strategy,
            platforms=["youtube", "facebook", "instagram", "tiktok"],
            temp_dir=tmp_path,
            upload_id="up1",
            brief={},
            studio_opts={},
            platform_map={},
            report=report,
            user_settings={"thumbnail_apply_mode": "pinned_cover"},
        )

    platform_map, _brief, skip, _opts = asyncio.run(_run())
    assert set(skip) == {"youtube", "facebook", "instagram", "tiktok"}
    assert set(report.get("studio_winner_letterbox_platforms") or []) == {
        "facebook",
        "instagram",
        "tiktok",
    }
    yt = Image.open(platform_map["youtube"])
    assert yt.size == (1280, 720)
    for plat in ("facebook", "instagram", "tiktok"):
        im = Image.open(platform_map[plat])
        assert im.size == (1080, 1920)


def test_hydrate_bridge_strategy_loads_pin_variant_r2(monkeypatch):
    async def _fake_fetch(*, user_id, variant_id, job_id="", db_pool=None):
        assert variant_id.endswith("bbbb")
        return "thumbnail-studio/previews/u/j/pin.jpg", {
            "variant_id": variant_id,
            "job_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "layout_name": "Pinned",
        }

    monkeypatch.setattr(
        "services.thumbnail_studio_upload_bridge.fetch_studio_variant_preview_r2",
        _fake_fetch,
    )
    locked = {
        "variant_id": "11111111-1111-1111-1111-111111111111",
        "preview_r2_key": "thumbnail-studio/previews/u/j/locked.jpg",
        "apply_mode": "cover_direct",
    }
    us = {
        "thumbnail_source_variant_id": "22222222-2222-2222-2222-22222222bbbb",
        "thumbnail_source_job_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "thumbnail_apply_mode": "pinned_cover",
    }
    report: dict = {}
    out = asyncio.run(
        hydrate_bridge_strategy(
            locked,
            us,
            user_id="33333333-3333-3333-3333-333333333333",
            db_pool=object(),
            report=report,
        )
    )
    assert out["preview_r2_key"].endswith("pin.jpg")
    assert report.get("studio_winner_hydrate_ok") is True
    assert out["variant_id"].endswith("bbbb")


def test_unresolved_persona_does_not_skip_entire_pikzels_pipeline():
    from stages.thumbnail_stage import run_thumbnail_stage

    src = inspect.getsource(run_thumbnail_stage)
    assert "persona_unresolved_continue" in src
    assert 'raise SkipStage(\n                    "Linked Pikzels persona required' not in src


def test_tiktok_burn_allowlist_includes_studio_winner_cover_direct():
    import worker

    src = inspect.getsource(worker._maybe_burn_tiktok_styled_cover)
    assert "studio_winner_cover_direct" in src


def test_hydrate_noop_when_strategy_already_matches_pin():
    strat = {
        "variant_id": "22222222-2222-2222-2222-22222222bbbb",
        "preview_r2_key": "thumbnail-studio/previews/u/j/same.jpg",
        "apply_mode": "cover_direct",
    }
    us = {
        "thumbnail_source_variant_id": "22222222-2222-2222-2222-22222222bbbb",
    }
    report: dict = {}
    out = asyncio.run(
        hydrate_bridge_strategy(strat, us, user_id="u", db_pool=object(), report=report)
    )
    assert out["preview_r2_key"].endswith("same.jpg")
    assert report.get("studio_winner_hydrate_needed") is False
