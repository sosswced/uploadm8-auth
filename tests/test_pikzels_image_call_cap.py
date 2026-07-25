"""Hard cap on billable Pikzels /v2/thumbnail/image calls per job."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from stages.pikzels_api import render_thumbnail_with_studio_renderer


@pytest.fixture()
def tiny_jpeg(tmp_path: Path) -> Path:
    # Minimal valid-enough JPEG header bytes for prepare path mock bypass.
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9" + b"\x00" * 64)
    return p


def test_pikzels_image_call_cap_blocks_third_invoke(tiny_jpeg: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PIKZELS_MAX_IMAGE_CALLS_PER_JOB", "2")
    monkeypatch.setenv("PIKZELS_API_KEY", "test-key-not-real")

    ctx = SimpleNamespace(_pikzels_image_calls=0, output_artifacts={}, user_settings={})
    out1 = tmp_path / "a.jpg"
    out2 = tmp_path / "b.jpg"
    out3 = tmp_path / "c.jpg"

    async def _fake_post(_path, _payload):
        return 500, {"error": {"code": "NOPE", "message": "fail"}}

    with (
        patch("stages.pikzels_api.studio_renderer_enabled", return_value=True),
        patch("stages.pikzels_api._jpeg_bytes_for_pikzels_frame", return_value=b"abc123"),
        patch("stages.pikzels_api.pikzels_v2_post", new=AsyncMock(side_effect=_fake_post)),
        patch(
            "services.platform_colors.resolve_platform_colors",
            return_value={},
        ),
        patch(
            "services.platform_colors.platform_color_for",
            return_value=None,
        ),
    ):
        ok1 = asyncio.run(
            render_thumbnail_with_studio_renderer(
                tiny_jpeg, {}, "youtube", out1, upload_id="u1", job_context=ctx
            )
        )
        ok2 = asyncio.run(
            render_thumbnail_with_studio_renderer(
                tiny_jpeg, {}, "instagram", out2, upload_id="u1", job_context=ctx
            )
        )
        ok3 = asyncio.run(
            render_thumbnail_with_studio_renderer(
                tiny_jpeg, {}, "tiktok", out3, upload_id="u1", job_context=ctx
            )
        )

    assert ok1 is False and ok2 is False and ok3 is False
    assert int(ctx._pikzels_image_calls) == 2
