"""Thumbnail Studio + Pikzels v2 proxy route tests (mocked external API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import core.state
from app import app
from core.deps import get_current_user

USER_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

FAKE_USER = {
    "id": USER_ID,
    "subscription_tier": "creator_pro",
    "billing_user_id": USER_ID,
    "email": "studio@test.uploadm8.com",
}


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_args):
        return False


class _StudioPrefsConn:
    """Minimal conn for Studio preference gate (default: Studio enabled)."""

    def __init__(self, preferences=None):
        self._preferences = {} if preferences is None else preferences

    async def fetchrow(self, *_a, **_k):
        return {"preferences": self._preferences}


class FakePool:
    def __init__(self, conn=None):
        self._conn = conn if conn is not None else _StudioPrefsConn()

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.fixture(scope="module")
def studio_client():
    async def _fake_user():
        return FAKE_USER

    core.state.db_pool = FakePool()
    app.dependency_overrides[get_current_user] = _fake_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_current_user, None)
    core.state.db_pool = None


def test_ts_estimate_returns_wallet_breakdown(studio_client: TestClient):
    r = studio_client.post(
        "/api/thumbnail-studio/estimate",
        json={"variant_count": 2, "has_persona": False, "has_channel_memory": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["put_cost"] >= 0
    assert body["aic_cost"] >= 0
    assert isinstance(body.get("breakdown"), dict)


def test_pikzels_v2_recreate_debits_and_proxies(studio_client: TestClient):
    pikzels_response = {"output_url": "https://cdn.pikzels.test/out.jpg", "job_id": "pkz-1"}
    with patch(
        "routers.thumbnail_studio_api.atomic_debit_tokens",
        new=AsyncMock(return_value=True),
    ) as mock_debit, patch(
        "routers.thumbnail_studio_api.pikzels_v2_post",
        new=AsyncMock(return_value=(200, pikzels_response)),
    ) as mock_post, patch(
        "routers.thumbnail_studio_api.record_studio_usage_event",
        new=AsyncMock(),
    ), patch(
        "routers.thumbnail_studio_api._maybe_notify_pikzels_discord",
        new=AsyncMock(),
    ):
        r = studio_client.post(
            "/api/thumbnail-studio/pikzels-v2/recreate",
            json={
                "prompt": "Bold title, high contrast",
                "image_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
                "model": "pkz_4",
                "format": "16:9",
            },
        )

    assert r.status_code == 200
    assert r.json()["output_url"] == pikzels_response["output_url"]
    mock_debit.assert_awaited_once()
    mock_post.assert_awaited_once()
    assert mock_post.await_args.args[0] == "/v2/thumbnail/image"


def test_pikzels_v2_edit_debits_and_proxies(studio_client: TestClient):
    pikzels_response = {"output_url": "https://cdn.pikzels.test/edited.jpg"}
    with patch(
        "routers.thumbnail_studio_api.atomic_debit_tokens",
        new=AsyncMock(return_value=True),
    ), patch(
        "routers.thumbnail_studio_api.pikzels_v2_post",
        new=AsyncMock(return_value=(200, pikzels_response)),
    ) as mock_post, patch(
        "routers.thumbnail_studio_api.record_studio_usage_event",
        new=AsyncMock(),
    ), patch(
        "routers.thumbnail_studio_api._maybe_notify_pikzels_discord",
        new=AsyncMock(),
    ):
        r = studio_client.post(
            "/api/thumbnail-studio/pikzels-v2/edit",
            json={
                "prompt": "Add neon outline",
                "image_url": "https://cdn.uploadm8.test/thumb.jpg",
                "format": "16:9",
            },
        )

    assert r.status_code == 200
    assert r.json()["output_url"] == pikzels_response["output_url"]
    mock_post.assert_awaited_once()
    assert mock_post.await_args.args[0] == "/v2/thumbnail/edit"


def test_pikzels_v2_recreate_insufficient_tokens_returns_429(studio_client: TestClient):
    with patch(
        "routers.thumbnail_studio_api.atomic_debit_tokens",
        new=AsyncMock(return_value=False),
    ):
        r = studio_client.post(
            "/api/thumbnail-studio/pikzels-v2/recreate",
            json={
                "prompt": "test",
                "image_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            },
        )

    assert r.status_code == 429
    detail = r.json()["detail"]
    assert detail["code"] == "insufficient_tokens"


def test_pikzels_recreate_integration_mocked_no_api_key(studio_client: TestClient):
    """Full recreate proxy path with mocked Pikzels — no PIKZELS_API_KEY required."""
    fake_output = {
        "output_url": "https://cdn.pikzels.test/mock-recreate.png",
        "variants": [{"id": "v1"}],
    }
    with patch(
        "routers.thumbnail_studio_api.atomic_debit_tokens",
        new=AsyncMock(return_value=True),
    ), patch(
        "routers.thumbnail_studio_api.pikzels_v2_post",
        new=AsyncMock(return_value=(200, fake_output)),
    ) as mock_post, patch(
        "routers.thumbnail_studio_api.record_studio_usage_event",
        new=AsyncMock(),
    ), patch(
        "routers.thumbnail_studio_api._maybe_notify_pikzels_discord",
        new=AsyncMock(),
    ):
        r = studio_client.post(
            "/api/thumbnail-studio/pikzels-v2/recreate",
            json={
                "prompt": "Recreate with persona styling",
                "image_url": "https://i.ytimg.com/vi/abc123/hqdefault.jpg",
                "image_weight": "medium",
                "upload_id": "0ad8c0a8-a1d1-49dd-841b-1d85630929d6",
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["output_url"] == fake_output["output_url"]
    payload = mock_post.await_args.args[1]
    assert payload.get("prompt")
    assert payload.get("image_url")


def test_assert_thumbnail_studio_enabled_blocks_explicit_off():
    import asyncio
    from routers.thumbnail_studio_api import _assert_thumbnail_studio_enabled
    from fastapi import HTTPException

    class _Conn:
        async def fetchrow(self, *_a, **_k):
            return {"preferences": {"thumbnailStudioEnabled": False}}

    class _Pool:
        def acquire(self):
            return _FakeAcquire(_Conn())

    prev = core.state.db_pool
    core.state.db_pool = _Pool()
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_assert_thumbnail_studio_enabled(FAKE_USER))
        assert ei.value.status_code == 403
        detail = ei.value.detail
        assert isinstance(detail, dict)
        assert detail.get("code") == "thumbnail_studio_disabled"
        assert detail.get("settings_url") == "/settings.html#preferences"
    finally:
        core.state.db_pool = prev


def test_assert_thumbnail_studio_enabled_allows_default():
    import asyncio
    from routers.thumbnail_studio_api import _assert_thumbnail_studio_enabled

    class _Conn:
        async def fetchrow(self, *_a, **_k):
            return {"preferences": {}}

    class _Pool:
        def acquire(self):
            return _FakeAcquire(_Conn())

    prev = core.state.db_pool
    core.state.db_pool = _Pool()
    try:
        asyncio.run(_assert_thumbnail_studio_enabled(FAKE_USER))
    finally:
        core.state.db_pool = prev


def test_assert_thumbnail_studio_enabled_fails_closed_without_pool():
    import asyncio
    from routers.thumbnail_studio_api import _assert_thumbnail_studio_enabled
    from fastapi import HTTPException

    prev = core.state.db_pool
    core.state.db_pool = None
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_assert_thumbnail_studio_enabled(FAKE_USER))
        assert ei.value.status_code == 503
        assert ei.value.detail.get("code") == "preferences_unavailable"
    finally:
        core.state.db_pool = prev


def test_assert_thumbnail_studio_enabled_fails_closed_on_db_error():
    import asyncio
    from routers.thumbnail_studio_api import _assert_thumbnail_studio_enabled
    from fastapi import HTTPException

    class _Conn:
        async def fetchrow(self, *_a, **_k):
            raise RuntimeError("db down")

    class _Pool:
        def acquire(self):
            return _FakeAcquire(_Conn())

    prev = core.state.db_pool
    core.state.db_pool = _Pool()
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_assert_thumbnail_studio_enabled(FAKE_USER))
        assert ei.value.status_code == 503
        assert ei.value.detail.get("code") == "preferences_unavailable"
    finally:
        core.state.db_pool = prev


def test_pikzels_v2_recreate_blocked_when_studio_disabled(studio_client: TestClient):
    core.state.db_pool = FakePool(_StudioPrefsConn({"thumbnailStudioEnabled": False}))
    try:
        with patch(
            "routers.thumbnail_studio_api.atomic_debit_tokens",
            new=AsyncMock(return_value=True),
        ) as mock_debit:
            r = studio_client.post(
                "/api/thumbnail-studio/pikzels-v2/recreate",
                json={
                    "prompt": "Bold title",
                    "image_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
                },
            )
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "thumbnail_studio_disabled"
        mock_debit.assert_not_awaited()
    finally:
        core.state.db_pool = FakePool()
