"""Process-wide Pikzels 402 latch — stop billing after empty balance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from stages.pikzels_api import (
    artifacts_show_pikzels_credits_exhausted,
    clear_pikzels_credits_exhausted_latch,
    mark_pikzels_credits_exhausted,
    pikzels_credits_blocked,
    render_thumbnail_with_studio_renderer,
)
from services.thumbnail_ops import record_pikzels_render_failures_incident
from services.upload.prefs import merge_upload_init_thumbnail_preferences


@pytest.fixture(autouse=True)
def _clear_latch():
    clear_pikzels_credits_exhausted_latch()
    yield
    clear_pikzels_credits_exhausted_latch()


def test_mark_and_block_credits_latch(monkeypatch):
    monkeypatch.setenv("PIKZELS_CREDITS_EXHAUSTED_TTL_SEC", "3600")
    assert pikzels_credits_blocked() is False
    mark_pikzels_credits_exhausted(ttl_sec=3600)
    assert pikzels_credits_blocked() is True
    clear_pikzels_credits_exhausted_latch()
    assert pikzels_credits_blocked() is False


def test_clear_latch_env_flag(monkeypatch):
    mark_pikzels_credits_exhausted(ttl_sec=3600)
    assert pikzels_credits_blocked() is True
    monkeypatch.setenv("PIKZELS_CLEAR_CREDITS_LATCH", "1")
    assert pikzels_credits_blocked() is False
    assert pikzels_credits_blocked() is False  # flag consumed


def test_artifacts_show_prior_402():
    assert artifacts_show_pikzels_credits_exhausted(
        {"pikzels_credits_exhausted": "1"}
    )
    assert artifacts_show_pikzels_credits_exhausted(
        {
            "pikzels_render_failures": [
                {"platform": "tiktok", "http_status": 402, "reason": "insufficient_credits"}
            ]
        }
    )
    assert not artifacts_show_pikzels_credits_exhausted({"pikzels_render_failures": []})


def test_render_refuses_when_latch_armed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PIKZELS_API_KEY", "test-key-not-real")
    mark_pikzels_credits_exhausted(ttl_sec=3600)
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9" + b"\x00" * 64)
    out = tmp_path / "o.jpg"
    post = AsyncMock(return_value=(200, {"image_base64": "aaaa"}))
    with patch("stages.pikzels_api.pikzels_v2_post", new=post):
        ok = asyncio.run(
            render_thumbnail_with_studio_renderer(
                frame, {}, "tiktok", out, upload_id="u1", job_context=SimpleNamespace()
            )
        )
    assert ok is False
    post.assert_not_awaited()


def test_presign_does_not_auto_enable_pikzels_when_engine_omitted(monkeypatch):
    monkeypatch.setenv("PIKZELS_API_KEY", "test-key-not-real")
    prefs: dict = {}
    merge_upload_init_thumbnail_preferences(prefs, SimpleNamespace())
    assert prefs.get("thumbnail_pikzels_enabled") is not True
    assert prefs.get("thumbnail_studio_engine_enabled") is not True


def test_incident_dedupes_duplicate_tiktok_402():
    artifacts = {
        "pikzels_render_failures": [
            {"platform": "tiktok", "http_status": 402, "message": "leader fail"},
            {"platform": "tiktok", "http_status": 402, "message": "skipped no credits"},
        ]
    }

    async def _run():
        with patch(
            "services.ops_incidents.record_operational_incident",
            new=AsyncMock(),
        ) as mock_incident:
            await record_pikzels_render_failures_incident(
                object(),
                upload_id="dup-tiktok",
                user_id="u1",
                output_artifacts=artifacts,
            )
            return mock_incident

    mock = asyncio.run(_run())
    kwargs = mock.await_args.kwargs
    assert kwargs["details"]["failures"] == [
        {
            "platform": "tiktok",
            "http_status": 402,
            "http_label": "402",
            "message": "leader fail",
        }
    ]
    assert "tiktok,tiktok" not in kwargs["subject"]
    assert "(1 platform" in kwargs["subject"]
