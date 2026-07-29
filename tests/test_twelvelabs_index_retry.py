"""Twelve Labs stale-index 404 → force-create + retry once + persist state."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import stages.twelvelabs_stage as tl


def _ctx(video: Path) -> SimpleNamespace:
    return SimpleNamespace(
        upload_id="tl-404",
        user_id="u",
        user_settings={"aiServiceSceneUnderstanding": True},
        subscription_tier="studio",
        entitlements=None,
        processed_video_path=str(video),
        local_video_path=None,
        video_intelligence={},
        video_intelligence_context={},
        video_understanding={},
        mark_stage=lambda *_a, **_k: None,
        output_artifacts={},
    )


def test_upload_404_recreates_index_and_retries(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    state_path = tmp_path / "tl_index.json"

    ctx = _ctx(video)
    tl._IGNORE_ENV_INDEX = False
    tl.TWELVELABS_INDEX_ID = "stale-index-id"
    tl.TWELVE_LABS_API_KEY = "test-key"
    tl._INDEX_STATE_PATH = state_path

    upload_calls = {"n": 0, "ids": []}

    async def fake_upload(video_path, index_id, upload_id, *, ctx=None):
        upload_calls["n"] += 1
        upload_calls["ids"].append(index_id)
        if index_id == "stale-index-id":
            return None, "upload HTTP 404"
        return "vid-ok", ""

    async def fake_create(*, ctx=None, unique=False):
        assert unique is True
        return "fresh-index-id"

    async def fake_desc(video_id, *, ctx=None):
        return "A fast dashcam run on the highway."

    async def fake_title(video_id, *, ctx=None):
        return "Highway heat"

    with (
        patch.object(tl, "_upload_and_index", side_effect=fake_upload),
        patch.object(tl, "_create_index", side_effect=fake_create),
        patch.object(tl, "_generate_description", side_effect=fake_desc),
        patch.object(tl, "_generate_title", side_effect=fake_title),
        patch.object(tl, "user_pref_ai_service_enabled", return_value=True),
    ):
        # _create_index is patched — still need state save on success path.
        # Call real save after fake create via wrapper.
        async def create_and_save(*, ctx=None, unique=False):
            assert unique is True
            tl._save_index_state(ignore_env=True, resolved_index_id="fresh-index-id")
            return "fresh-index-id"

        with patch.object(tl, "_create_index", side_effect=create_and_save):
            out = asyncio.run(tl.run_twelvelabs_stage(ctx))

    assert upload_calls["n"] == 2
    assert tl._IGNORE_ENV_INDEX is True
    assert out.video_understanding.get("video_id") == "vid-ok"
    assert out.video_understanding.get("index_id") == "fresh-index-id"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.get("ignore_env") is True
    assert state.get("resolved_index_id") == "fresh-index-id"

    # New "worker": process latch cleared, file state still heals env.
    tl._IGNORE_ENV_INDEX = False
    tl.TWELVELABS_INDEX_ID = "stale-index-id"
    assert tl._resolve_index_id() == "fresh-index-id"


def test_get_or_create_prefers_healed_index_over_first_uploadm8(tmp_path: Path):
    state_path = tmp_path / "tl_pref.json"
    tl._INDEX_STATE_PATH = state_path
    tl._IGNORE_ENV_INDEX = False
    tl.TWELVE_LABS_API_KEY = "test-key"
    tl._save_index_state(ignore_env=True, resolved_index_id="healed-999")

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": [
                    {"_id": "stale-111", "name": "uploadm8-content"},
                    {"_id": "healed-999", "name": "uploadm8-content-abcdef12"},
                    {"_id": "other-222", "name": "uploadm8-old"},
                ]
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    with patch.object(tl.httpx, "AsyncClient", _Client):
        picked = asyncio.run(tl._get_or_create_index())
    assert picked == "healed-999"


def test_get_or_create_prefers_newest_healed_name_without_state(tmp_path: Path):
    state_path = tmp_path / "tl_nostate.json"
    if state_path.exists():
        state_path.unlink()
    tl._INDEX_STATE_PATH = state_path
    tl._IGNORE_ENV_INDEX = False
    tl.TWELVE_LABS_API_KEY = "test-key"

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": [
                    {"_id": "old-aaa", "name": "uploadm8-content"},
                    {"_id": "heal-bbb", "name": "uploadm8-content-bbbbbbbb"},
                    {"_id": "heal-ccc", "name": "uploadm8-content-cccccccc"},
                ]
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    with patch.object(tl.httpx, "AsyncClient", _Client):
        picked = asyncio.run(tl._get_or_create_index())
    # reverse name sort → cccccccc first
    assert picked == "heal-ccc"


def test_upload_404_retry_still_skips_when_fresh_fails(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    state_path = tmp_path / "tl_index_fail.json"

    ctx = _ctx(video)
    ctx.upload_id = "tl-404-fail"
    tl._IGNORE_ENV_INDEX = False
    tl.TWELVELABS_INDEX_ID = "stale"
    tl.TWELVE_LABS_API_KEY = "test-key"
    tl._INDEX_STATE_PATH = state_path

    async def always_404(*_a, **_k):
        return None, "upload HTTP 404"

    with (
        patch.object(tl, "_upload_and_index", side_effect=always_404),
        patch.object(tl, "_create_index", AsyncMock(return_value="fresh")),
        patch.object(tl, "user_pref_ai_service_enabled", return_value=True),
    ):
        try:
            asyncio.run(tl.run_twelvelabs_stage(ctx))
            raised = False
        except tl.SkipStage as e:
            raised = True
            assert "404" in str(e)

    assert raised
    assert tl._IGNORE_ENV_INDEX is True
    assert state_path.is_file()
    assert json.loads(state_path.read_text(encoding="utf-8")).get("ignore_env") is True
