"""TikTok music compliance (ACR mute + inbox/Sounds path)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from services.tiktok_api import normalize_tiktok_post_settings, validate_tiktok_post_settings
from stages.context import JobContext
from stages.tiktok_music_compliance import (
    ARTIFACT_KEY,
    apply_tiktok_music_compliance_after_audio,
    get_tiktok_music_compliance_notice,
    tiktok_acr_catalog_risk,
)


def _ctx(**kwargs) -> JobContext:
    base = dict(
        job_id="j1",
        upload_id="tt-music-1",
        user_id="user1",
        platforms=["tiktok"],
        video_info={"duration": 45.0},
        audio_context={
            "copyright_risk": True,
            "music_detected": True,
            "music_title": "Hit Song",
            "music_artist": "Artist",
            "content_signals": ["acr_catalog_match"],
        },
        tiktok_post_settings={},
        platform_videos={},
        output_artifacts={},
        temp_dir="/tmp",
    )
    base.update(kwargs)
    return JobContext(**base)


def test_normalize_music_flags_and_mute_wins():
    out = normalize_tiktok_post_settings(
        {
            "privacy_level": "SELF_ONLY",
            "user_consent": True,
            "mute_audio_for_tiktok": True,
            "keep_catalog_audio_licensed": True,
            "finish_in_tiktok_app": False,
        }
    )
    assert out["mute_audio_for_tiktok"] is True
    assert out["keep_catalog_audio_licensed"] is False


def test_finish_in_app_clears_keep_licensed_and_skips_privacy():
    out = normalize_tiktok_post_settings(
        {
            "finish_in_tiktok_app": True,
            "keep_catalog_audio_licensed": True,
            "user_consent": True,
        }
    )
    assert out["finish_in_tiktok_app"] is True
    assert out["keep_catalog_audio_licensed"] is False
    errs = validate_tiktok_post_settings(out)
    assert errs == []


def test_direct_post_still_requires_privacy():
    errs = validate_tiktok_post_settings(
        {"user_consent": True, "finish_in_tiktok_app": False}
    )
    assert any("privacy" in e.lower() or "view" in e.lower() for e in errs)


def test_acr_catalog_risk_any_duration():
    assert tiktok_acr_catalog_risk(_ctx()) is True
    assert tiktok_acr_catalog_risk(_ctx(platforms=["youtube"])) is False
    assert tiktok_acr_catalog_risk(_ctx(audio_context={"music_detected": False})) is False


def test_apply_mutes_on_acr_risk(tmp_path):
    src = tmp_path / "tiktok.mp4"
    src.write_bytes(b"fake")
    muted = tmp_path / "tiktok_muted_tt-music-1.mp4"
    ctx = _ctx(
        temp_dir=str(tmp_path),
        platform_videos={"tiktok": src},
        tiktok_post_settings={
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "user_consent": True,
        },
    )

    async def _mute(c):
        muted.write_bytes(b"muted")
        c.platform_videos["tiktok"] = muted
        return muted

    with (
        patch(
            "stages.tiktok_music_compliance.mute_tiktok_deliverable",
            new=AsyncMock(side_effect=_mute),
        ),
        patch(
            "stages.pipeline_checkpoint.merge_output_artifacts_patch",
            new=AsyncMock(),
        ),
        patch(
            "stages.pipeline_checkpoint.refresh_transcode_checkpoint_platform",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(apply_tiktok_music_compliance_after_audio(ctx, MagicMock()))

    notice = get_tiktok_music_compliance_notice(ctx)
    assert notice is not None
    assert notice["status"] == "muted"
    assert notice["muted"] is True
    assert Path(ctx.platform_videos["tiktok"]) == muted


def test_keep_licensed_skips_mute():
    ctx = _ctx(
        tiktok_post_settings={
            "privacy_level": "SELF_ONLY",
            "user_consent": True,
            "keep_catalog_audio_licensed": True,
        }
    )
    with (
        patch(
            "stages.tiktok_music_compliance.mute_tiktok_deliverable",
            new=AsyncMock(),
        ) as mute,
        patch(
            "stages.pipeline_checkpoint.merge_output_artifacts_patch",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(apply_tiktok_music_compliance_after_audio(ctx, MagicMock()))
    mute.assert_not_called()
    notice = json.loads(ctx.output_artifacts[ARTIFACT_KEY])
    assert notice["status"] == "kept_licensed"


def test_list_item_exposes_tiktok_music_compliance():
    from services.upload.list_detail import (
        build_upload_list_item,
        tiktok_music_compliance_notice_from_artifacts,
    )

    arts = {
        "tiktok_music_compliance": {
            "status": "muted",
            "message": "TikTok file was muted",
            "muted": True,
            "finish_in_tiktok_app": True,
        }
    }
    notice = tiktok_music_compliance_notice_from_artifacts(arts)
    assert notice["muted"] is True

    item = build_upload_list_item(
        {
            "id": "u1",
            "filename": "clip.mp4",
            "platforms": ["tiktok"],
            "status": "processing",
            "output_artifacts": arts,
            "created_at": None,
        },
        [],
        creator_map={},
        presign_r2_thumbnails=False,
    )
    assert item["tiktokMusicCompliance"]["message"] == "TikTok file was muted"
    assert item["tiktokMusicCompliance"]["finish_in_tiktok_app"] is True


def test_frontend_wires_sounds_and_inbox_copy():
    js = Path("frontend/js/tiktok-export.js").read_text(encoding="utf-8")
    copy = Path("frontend/js/tiktok-ux-copy.js").read_text(encoding="utf-8")
    queue = Path("frontend/queue.html").read_text(encoding="utf-8")
    assert "tt-finish-in-app" in js
    assert "finish_in_tiktok_app" in js
    assert "mute_audio_for_tiktok" in js
    assert "finishInAppLabel" in copy
    assert "queueAddSoundNote" in copy
    assert "add Sound in TikTok" in queue
    assert "tiktokMusicCompliance" in queue
    assert "_tiktok_init_inbox_upload" in Path("stages/publish_stage.py").read_text(
        encoding="utf-8"
    )
