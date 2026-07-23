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
