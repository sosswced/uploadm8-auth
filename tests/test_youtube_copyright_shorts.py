"""YouTube Shorts + ACR catalogue trim (≤60s YouTube-only)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from services.acrcloud_identify import parse_acr_identify_response
from stages.context import JobContext
from stages.publish_stage import _youtube_avoid_shorts_markers_for_rights
from stages.youtube_copyright_shorts import (
    COPYRIGHT_SHORTS_MAX_SEC,
    _acr_catalog_copyright_signal,
    _trim_pref_enabled,
    apply_youtube_copyright_shorts_after_audio,
    youtube_copyright_shorts_acr_risk,
    youtube_copyright_shorts_trim_applied,
)


def _ctx(**kwargs) -> JobContext:
    base = dict(
        job_id="j1",
        upload_id="u1",
        user_id="user1",
        platforms=["youtube", "tiktok"],
        video_info={"duration": 90.0},
        audio_context={
            "copyright_risk": True,
            "music_detected": True,
            "music_title": "Hit Song",
            "music_artist": "Artist",
            "content_signals": ["acr_catalog_match"],
        },
        user_settings={"youtubeShortsCopyrightTrim": True},
        platform_videos={},
        output_artifacts={},
        temp_dir="/tmp",
    )
    base.update(kwargs)
    return JobContext(**base)


def test_acr_catalog_signal_from_copyright_risk():
    assert _acr_catalog_copyright_signal({"copyright_risk": True}) is True


def test_acr_catalog_signal_from_content_signals():
    assert _acr_catalog_copyright_signal(
        {
            "copyright_risk": False,
            "content_signals": ["acr_catalog_match"],
            "music_title": "X",
        }
    ) is True


def test_acr_catalog_signal_false_without_music():
    assert _acr_catalog_copyright_signal(
        {"content_signals": ["acr_catalog_match"], "music_detected": False}
    ) is False


def test_risk_requires_youtube_and_long_clip():
    assert youtube_copyright_shorts_acr_risk(_ctx(platforms=["tiktok"])) is False
    assert youtube_copyright_shorts_acr_risk(_ctx(video_info={"duration": 30})) is False
    assert youtube_copyright_shorts_acr_risk(_ctx()) is True


def test_trim_pref_camel_snake_and_string():
    assert _trim_pref_enabled(_ctx(user_settings={"youtubeShortsCopyrightTrim": True})) is True
    assert _trim_pref_enabled(_ctx(user_settings={"youtube_shorts_copyright_trim": True})) is True
    assert _trim_pref_enabled(_ctx(user_settings={"youtubeShortsCopyrightTrim": "false"})) is False
    assert _trim_pref_enabled(_ctx(user_settings={})) is False


def test_parse_acr_sets_copyright_risk():
    parsed = parse_acr_identify_response(
        {
            "status": {"code": 0},
            "metadata": {
                "music": [
                    {
                        "title": "Catalogue Track",
                        "artists": [{"name": "Big Label"}],
                        "score": 88,
                        "acrid": "abc",
                    }
                ]
            },
        }
    )
    assert parsed["music_detected"] is True
    assert parsed["copyright_risk"] is True
    assert parsed["music_title"] == "Catalogue Track"


def test_avoid_shorts_markers_when_no_trim():
    ctx = _ctx(user_settings={"youtubeShortsCopyrightTrim": False})
    assert _youtube_avoid_shorts_markers_for_rights(ctx) is True


def test_avoid_shorts_markers_off_when_trim_applied():
    ctx = _ctx()
    ctx.output_artifacts["youtube_copyright_shorts"] = json.dumps(
        {"trim_applied": True, "source": "acr_catalog"}
    )
    assert youtube_copyright_shorts_trim_applied(ctx) is True
    assert _youtube_avoid_shorts_markers_for_rights(ctx) is False


def test_apply_writes_notice_without_trim_when_pref_off():
    ctx = _ctx(user_settings={"youtubeShortsCopyrightTrim": False})
    merge = AsyncMock()

    async def _run():
        with patch(
            "stages.pipeline_checkpoint.merge_output_artifacts_patch", merge
        ), patch(
            "stages.youtube_copyright_shorts._retrim_youtube_deliverable",
            new_callable=AsyncMock,
        ) as retrim:
            await apply_youtube_copyright_shorts_after_audio(ctx, MagicMock())
            retrim.assert_not_awaited()

    asyncio.run(_run())
    assert "youtube_copyright_shorts" in ctx.output_artifacts
    notice = json.loads(ctx.output_artifacts["youtube_copyright_shorts"])
    assert notice["trim_applied"] is False
    assert notice["trim_pref_enabled"] is False
    merge.assert_awaited()


def test_apply_trims_youtube_only_when_pref_on(tmp_path: Path):
    yt = tmp_path / "youtube.mp4"
    tt = tmp_path / "tiktok.mp4"
    yt.write_bytes(b"yt-full")
    tt.write_bytes(b"tt-full")
    trimmed = tmp_path / "transcoded_youtube_copyright_shorts_trim.mp4"
    trimmed.write_bytes(b"yt-60")

    ctx = _ctx(
        temp_dir=str(tmp_path),
        platform_videos={"youtube": yt, "tiktok": tt},
        user_settings={"youtubeShortsCopyrightTrim": True},
    )
    merge = AsyncMock()
    refresh = AsyncMock(return_value=True)

    async def _fake_retrim(c, _pool):
        c.platform_videos["youtube"] = trimmed
        return True

    info_full = SimpleNamespace(duration=90.0)
    info_trim = SimpleNamespace(duration=59.5)

    async def _run():
        with patch(
            "stages.pipeline_checkpoint.merge_output_artifacts_patch", merge
        ), patch(
            "stages.youtube_copyright_shorts._retrim_youtube_deliverable",
            new=_fake_retrim,
        ), patch(
            "stages.pipeline_checkpoint.refresh_transcode_checkpoint_platform",
            refresh,
        ), patch(
            "stages.youtube_copyright_shorts.get_video_info",
            new=AsyncMock(side_effect=[info_trim]),
        ):
            await apply_youtube_copyright_shorts_after_audio(ctx, MagicMock())

    asyncio.run(_run())
    assert ctx.platform_videos["youtube"] == trimmed
    assert ctx.platform_videos["tiktok"] == tt
    notice = json.loads(ctx.output_artifacts["youtube_copyright_shorts"])
    assert notice["trim_applied"] is True
    assert notice.get("youtube_output_duration_sec") == 59.5
    refresh.assert_awaited()
    assert COPYRIGHT_SHORTS_MAX_SEC == 60.0


def test_apply_noop_without_acr_risk():
    ctx = _ctx(audio_context={})
    merge = AsyncMock()

    async def _run():
        with patch("stages.pipeline_checkpoint.merge_output_artifacts_patch", merge):
            await apply_youtube_copyright_shorts_after_audio(ctx, MagicMock())

    asyncio.run(_run())
    assert "youtube_copyright_shorts" not in ctx.output_artifacts
    merge.assert_not_awaited()


def test_apply_preserves_trim_applied_on_resume_when_already_short(tmp_path: Path):
    """Resume after checkpoint: youtube already ≤60s → retrim skips but trim_applied stays true."""
    yt = tmp_path / "youtube.mp4"
    yt.write_bytes(b"yt-60")
    ctx = _ctx(
        temp_dir=str(tmp_path),
        platform_videos={"youtube": yt},
        user_settings={"youtubeShortsCopyrightTrim": True},
        output_artifacts={
            "youtube_copyright_shorts": json.dumps(
                {"trim_applied": True, "youtube_output_duration_sec": 59.0, "source": "acr_catalog"}
            )
        },
    )
    merge = AsyncMock()
    info_short = SimpleNamespace(duration=59.0)

    async def _run():
        with patch(
            "stages.pipeline_checkpoint.merge_output_artifacts_patch", merge
        ), patch(
            "stages.youtube_copyright_shorts._retrim_youtube_deliverable",
            new=AsyncMock(return_value=False),
        ), patch(
            "stages.youtube_copyright_shorts.get_video_info",
            new=AsyncMock(return_value=info_short),
        ):
            await apply_youtube_copyright_shorts_after_audio(ctx, MagicMock())

    asyncio.run(_run())
    notice = json.loads(ctx.output_artifacts["youtube_copyright_shorts"])
    assert notice["trim_applied"] is True
    assert notice.get("youtube_output_duration_sec") == 59.0


def test_apply_does_not_trust_prior_trim_when_file_still_long(tmp_path: Path):
    """Resume with stale trim_applied but long youtube file → do not claim trim."""
    yt = tmp_path / "youtube.mp4"
    yt.write_bytes(b"yt-long")
    ctx = _ctx(
        temp_dir=str(tmp_path),
        platform_videos={"youtube": yt},
        user_settings={"youtubeShortsCopyrightTrim": True},
        output_artifacts={
            "youtube_copyright_shorts": json.dumps(
                {"trim_applied": True, "youtube_output_duration_sec": 59.0, "source": "acr_catalog"}
            )
        },
    )
    merge = AsyncMock()
    info_long = SimpleNamespace(duration=180.0)

    async def _run():
        with patch(
            "stages.pipeline_checkpoint.merge_output_artifacts_patch", merge
        ), patch(
            "stages.youtube_copyright_shorts._retrim_youtube_deliverable",
            new=AsyncMock(return_value=False),
        ), patch(
            "stages.youtube_copyright_shorts.get_video_info",
            new=AsyncMock(return_value=info_long),
        ):
            await apply_youtube_copyright_shorts_after_audio(ctx, MagicMock())

    asyncio.run(_run())
    notice = json.loads(ctx.output_artifacts["youtube_copyright_shorts"])
    assert notice["trim_applied"] is False
    assert notice.get("trim_error")


def test_retrim_falls_back_to_local_video_when_youtube_key_missing(tmp_path: Path):
    """Pref on + ACR risk: resolve source from local_video_path when map lacks youtube."""
    from stages.youtube_copyright_shorts import _resolve_youtube_source_path, _retrim_youtube_deliverable

    src = tmp_path / "source.mp4"
    src.write_bytes(b"full-length")

    ctx = _ctx(
        temp_dir=str(tmp_path),
        platform_videos={"tiktok": src},  # no youtube key
        user_settings={"youtubeShortsCopyrightTrim": True},
    )
    ctx.local_video_path = src

    assert _resolve_youtube_source_path(ctx) == src

    info_full = SimpleNamespace(duration=120.0)
    info_trim = SimpleNamespace(duration=59.5)

    async def _fake_transcode(_src, out_path, *_a, **_k):
        Path(out_path).write_bytes(b"trimmed")

    async def _run():
        with patch(
            "stages.youtube_copyright_shorts.get_video_info",
            new=AsyncMock(side_effect=[info_full, info_trim]),
        ), patch(
            "stages.youtube_copyright_shorts.transcode_video",
            new=_fake_transcode,
        ), patch(
            "stages.youtube_copyright_shorts.resolve_reframe_action",
            return_value="none",
        ):
            return await _retrim_youtube_deliverable(ctx, MagicMock())

    assert asyncio.run(_run()) is True
    assert Path(ctx.platform_videos["youtube"]).name == "transcoded_youtube_copyright_shorts_trim.mp4"
    assert Path(ctx.platform_videos["tiktok"]) == src


def test_list_item_exposes_youtube_copyright_shorts_from_artifacts():
    from services.upload.list_detail import (
        build_upload_list_item,
        youtube_copyright_shorts_notice_from_artifacts,
    )

    arts = {
        "youtube_copyright_shorts": {
            "level": "warning",
            "message": "Catalogue music detected",
            "trim_applied": True,
        }
    }
    notice = youtube_copyright_shorts_notice_from_artifacts(arts)
    assert notice["trim_applied"] is True
    row = {
        "id": "u1",
        "filename": "clip.mp4",
        "platforms": ["youtube"],
        "status": "processing",
        "output_artifacts": arts,
        "created_at": None,
    }
    item = build_upload_list_item(row, [], creator_map={}, presign_r2_thumbnails=False)
    assert item["youtubeCopyrightShorts"]["message"] == "Catalogue music detected"
    assert item["youtubeCopyrightShorts"]["trim_applied"] is True


def test_duration_backfill_probes_local_file_when_video_info_empty(tmp_path: Path):
    """Checkpoint resume with empty video_info must not silently disable the trim gate."""
    from stages.youtube_copyright_shorts import _backfill_missing_duration

    yt = tmp_path / "youtube.mp4"
    yt.write_bytes(b"yt")
    ctx = _ctx(
        temp_dir=str(tmp_path),
        video_info={},  # resume path lost duration
        platform_videos={"youtube": yt},
    )
    info = SimpleNamespace(duration=95.0)

    async def _run():
        with patch(
            "stages.youtube_copyright_shorts.get_video_info",
            new=AsyncMock(return_value=info),
        ):
            await _backfill_missing_duration(ctx)

    asyncio.run(_run())
    assert ctx.video_info["duration"] == 95.0
    assert youtube_copyright_shorts_acr_risk(ctx) is True


def test_duration_backfill_noop_when_duration_present():
    from stages.youtube_copyright_shorts import _backfill_missing_duration

    ctx = _ctx(video_info={"duration": 90.0})

    async def _run():
        with patch(
            "stages.youtube_copyright_shorts.get_video_info",
            new=AsyncMock(side_effect=AssertionError("must not probe")),
        ):
            await _backfill_missing_duration(ctx)

    asyncio.run(_run())
    assert ctx.video_info["duration"] == 90.0


def test_create_context_keeps_prior_notice_json_parseable():
    """Nested artifact dicts must be JSON-encoded, not Python repr — otherwise the
    prior trim notice is silently dropped on resume (str(dict) → single quotes)."""
    from stages.context import create_context
    from stages.entitlements import get_entitlements_for_tier
    from stages.youtube_copyright_shorts import get_youtube_copyright_notice

    notice = {"level": "warning", "trim_applied": True, "message": "trimmed"}
    upload = {
        "id": "u1",
        "user_id": "user1",
        "filename": "clip.mp4",
        "r2_key": "k",
        "platforms": ["youtube"],
        "output_artifacts": {"youtube_copyright_shorts": notice},
    }
    ctx = create_context({}, upload, {}, get_entitlements_for_tier("pro"))
    restored = get_youtube_copyright_notice(ctx)
    assert restored is not None
    assert restored.get("trim_applied") is True


def test_post_transcode_checkpoint_persists_and_restores_video_info():
    """Resume must carry video_info or audio/trim duration gates silently fail."""
    import inspect
    from stages import pipeline_checkpoint as pc

    save_src = inspect.getsource(pc.save_post_transcode_checkpoint)
    assert '"video_info"' in save_src

    resume_src = inspect.getsource(pc.try_resume_from_checkpoint)
    assert 'cp.get("video_info")' in resume_src


def test_shell_wanted_includes_output_artifacts():
    """Regression: shell/bootstrap must SELECT output_artifacts or the notice blinks off."""
    import inspect
    from services.upload import list_detail as ld

    src = inspect.getsource(ld.fetch_user_uploads_list)
    assert '"output_artifacts"' in src
    # Ensure shell branch still exists and includes the column near shell wanted list.
    assert "elif shell:" in src
    shell_idx = src.index("elif shell:")
    chunk = src[shell_idx : shell_idx + 1200]
    assert "output_artifacts" in chunk
