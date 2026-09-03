"""YouTube resumable upload path — mock proof."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from stages.context import JobContext
from stages.publish_stage import publish_to_youtube


class _FakeResp:
    def __init__(self, status=200, data=None, text="", headers=None):
        self.status_code = status
        self._data = data if data is not None else {}
        self.text = text if text else (json.dumps(self._data) if self._data else "")
        self.headers = headers or {}

    def json(self):
        return self._data


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.posts = []
        self.puts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.posts.append({"url": url, "kwargs": kwargs})
        return _FakeResp(
            200,
            {},
            headers={
                "Location": "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=fake"
            },
        )

    async def put(self, url, **kwargs):
        self.puts.append({"url": url, "kwargs": kwargs})
        return _FakeResp(200, {"id": "ytVideo123"})


def test_publish_to_youtube_resumable_upload(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00fake-mp4-bytes")

    ctx = JobContext(
        job_id="j-yt",
        upload_id="u-yt",
        user_id="user-1",
        platforms=["youtube"],
        title="Test Short #shorts",
        caption="hello",
        privacy="public",
    )

    token = {"access_token": "ya29.fake"}

    with patch("stages.publish_stage.httpx.AsyncClient", _FakeAsyncClient):
        with patch(
            "stages.publish_stage._refresh_youtube_token",
            new_callable=AsyncMock,
            return_value=token,
        ):
            with patch(
                "stages.publish_stage._ensure_platform_thumbnail_local",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = asyncio.run(publish_to_youtube(video, ctx, token))

    assert result.success is True
    assert result.platform_video_id == "ytVideo123"
    assert result.platform_url and "ytVideo123" in result.platform_url
