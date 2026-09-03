"""Meta Graph verify polls media/video ids."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from stages.verify_stage import verify_meta_media


class _FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data if data is not None else {}
        self.content = b"1" if data is not None else b""

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.get_url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kwargs):
        self.get_url = url
        return _FakeResp(200, {"id": "178414000", "permalink": "https://instagram.com/reel/x"})


def test_verify_meta_media_instagram_confirmed():
    with patch("stages.verify_stage.httpx.AsyncClient", _FakeClient):
        status = asyncio.run(
            verify_meta_media("instagram", "178414000", {"access_token": "EAA"})
        )
    assert status == "confirmed"


def test_verify_meta_media_facebook_processing():
    class _ProcClient(_FakeClient):
        async def get(self, url, **kwargs):
            return _FakeResp(200, {"id": "vid1", "status": "processing"})

    with patch("stages.verify_stage.httpx.AsyncClient", _ProcClient):
        status = asyncio.run(
            verify_meta_media("facebook", "vid1", {"access_token": "EAA"})
        )
    assert status == "pending"
