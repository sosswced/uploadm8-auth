"""Proof that every requested Meta permission is exercised on Graph.

pages_manage_posts must POST /{page-id}/videos (Page VOD/Reels, not live).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from stages.context import JobContext
from services.meta_oauth import (
    FACEBOOK_GRAPH_PERMISSION_PROOF,
    INSTAGRAM_GRAPH_PERMISSION_PROOF,
    facebook_page_videos_publish_url,
    facebook_pages_manage_posts_proof,
    fetch_managed_pages,
    instagram_reels_container_url,
    instagram_reels_publish_url,
    meta_facebook_oauth_scope,
    meta_graph_is_rate_limited,
    meta_instagram_oauth_scope,
    split_oauth_scope,
)


class _FakeResp:
    def __init__(self, status=200, data=None, text="", headers=None):
        self.status_code = status
        self._data = data if data is not None else {}
        self.text = text if text else (json.dumps(self._data) if self._data else "")
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if int(self.status_code) >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text[:200]}")


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.posts = []
        self.gets = []
        self._post_resp = _FakeResp(200, {"id": "vid-pages-manage-posts"})
        self._get_resp = _FakeResp(200, {"permalink_url": "https://www.facebook.com/watch/?v=vid-pages-manage-posts"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.posts.append({"url": url, "kwargs": kwargs})
        return self._post_resp

    async def get(self, url, **kwargs):
        self.gets.append({"url": url, "kwargs": kwargs})
        return self._get_resp


def test_every_facebook_scope_has_graph_proof(monkeypatch):
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    monkeypatch.delenv("META_FACEBOOK_OAUTH_SCOPE", raising=False)
    requested = set(split_oauth_scope(meta_facebook_oauth_scope()))
    proven = {row["permission"] for row in FACEBOOK_GRAPH_PERMISSION_PROOF}
    missing = requested - proven
    assert not missing, f"Facebook scopes with no Graph proof: {missing}"
    assert "publish_video" not in requested
    assert "publish_video" not in proven


def test_every_instagram_scope_has_graph_proof(monkeypatch):
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    monkeypatch.delenv("META_INSTAGRAM_OAUTH_SCOPE", raising=False)
    requested = set(split_oauth_scope(meta_instagram_oauth_scope()))
    proven = {row["permission"] for row in INSTAGRAM_GRAPH_PERMISSION_PROOF}
    missing = requested - proven
    assert not missing, f"Instagram scopes with no Graph proof: {missing}"


def test_pages_manage_posts_proof_is_page_videos_post():
    proof = facebook_pages_manage_posts_proof()
    assert proof["permission"] == "pages_manage_posts"
    assert proof["method"] == "POST"
    assert proof["path"] == "/{page-id}/videos"
    assert "live" not in proof["purpose"].lower() or "not live" in proof["purpose"].lower()
    assert "/videos" in facebook_page_videos_publish_url("PAGE123")
    assert facebook_page_videos_publish_url("PAGE123").endswith("/PAGE123/videos")
    assert "live_videos" not in facebook_page_videos_publish_url("PAGE123")


def test_approved_usage_covers_requested_facebook_scopes(monkeypatch):
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    monkeypatch.delenv("META_FACEBOOK_OAUTH_SCOPE", raising=False)
    from services.meta_oauth import META_PERMISSION_ALLOWED_USAGE

    requested = set(split_oauth_scope(meta_facebook_oauth_scope()))
    missing = requested - set(META_PERMISSION_ALLOWED_USAGE)
    assert not missing, f"No Approved Usage notes for: {missing}"
    pmp = META_PERMISSION_ALLOWED_USAGE["pages_manage_posts"]
    assert "Publish a post, photo, or video" in pmp["official_allowed_usage"]
    assert "POST /{page-id}/videos" in pmp["we_use"]
    assert "live-stream" in pmp["we_do_not"].lower() or "Live streaming" in pmp["we_do_not"]
    assert "publish_video" in pmp["we_do_not"]


def test_approved_usage_covers_requested_instagram_scopes(monkeypatch):
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    monkeypatch.delenv("META_INSTAGRAM_OAUTH_SCOPE", raising=False)
    from services.meta_oauth import META_PERMISSION_ALLOWED_USAGE

    requested = set(split_oauth_scope(meta_instagram_oauth_scope()))
    missing = requested - set(META_PERMISSION_ALLOWED_USAGE)
    assert not missing, f"No Approved Usage notes for: {missing}"


def test_graph_rate_limiting_guidelines():
    from services.meta_oauth import META_GRAPH_RATE_LIMITING

    notes = META_GRAPH_RATE_LIMITING["app_review_notes"]
    for code in (4, 17, 32, 613):
        assert str(code) in notes
    assert "rate limit" in notes.lower()
    assert meta_graph_is_rate_limited(
        200, "", {"X-App-Usage": '{"call_count": 100, "total_time": 10, "total_cputime": 10}'}
    ) is True
    body = json.dumps({"error": {"message": "Application request limit reached", "code": 4}})
    assert meta_graph_is_rate_limited(400, body) is True
    assert meta_graph_is_rate_limited(400, {"error": {"code": 17}}) is True
    assert meta_graph_is_rate_limited(400, {"error": {"code": 32}}) is True
    assert meta_graph_is_rate_limited(400, {"error": {"code": 613}}) is True
    assert meta_graph_is_rate_limited(429, "") is True
    assert meta_graph_is_rate_limited(400, {"error": {"code": 190}}) is False


def test_fetch_managed_pages_hits_me_accounts():
    class _Client:
        def __init__(self):
            self.urls = []

        async def get(self, url, params=None, timeout=None):
            self.urls.append(url)
            return _FakeResp(200, {"data": [{"id": "222", "name": "Demo Page", "access_token": "pt"}]})

    client = _Client()
    pages = asyncio.run(fetch_managed_pages(client, "user-llt", fields="id,name,access_token"))
    assert any("/me/accounts" in u for u in client.urls)
    assert pages[0]["id"] == "222"


def test_publish_to_facebook_posts_page_videos_pages_manage_posts(tmp_path):
    from stages.publish_stage import META_API_VERSION, publish_to_facebook

    video = tmp_path / "reel.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    ctx = JobContext(
        job_id="j1",
        upload_id="u1",
        user_id="user-1",
        platforms=["facebook"],
        caption="pages_manage_posts proof",
        privacy="public",
    )
    token = {
        "access_token": "page-token",
        "page_id": "PAGE123",
        "access_non_expiring": True,
        "access_obtained_at": "2026-08-01T00:00:00+00:00",
        "meta_permissions": [
            {"permission": "pages_manage_posts", "status": "granted"},
        ],
    }
    fake = _FakeAsyncClient()

    async def _identity(token_data, *args, **kwargs):
        return token_data

    with (
        patch("stages.publish_stage._refresh_meta_token", new=AsyncMock(side_effect=_identity)),
        patch("stages.publish_stage._ensure_platform_thumbnail_local", new=AsyncMock(return_value=None)),
        patch("stages.publish_stage.httpx.AsyncClient", return_value=fake),
    ):
        result = asyncio.run(publish_to_facebook(video, ctx, token, video_url=None))

    assert result.success is True
    assert result.platform_video_id == "vid-pages-manage-posts"
    assert fake.posts, "publish_to_facebook never POSTed Graph"
    posted = fake.posts[0]["url"]
    expected = facebook_page_videos_publish_url("PAGE123", version=META_API_VERSION)
    assert posted == expected
    assert posted.endswith("/PAGE123/videos")
    assert "live_videos" not in posted


def test_publish_to_facebook_maps_graph_rate_limit(tmp_path):
    from stages.publish_stage import publish_to_facebook

    video = tmp_path / "reel.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    ctx = JobContext(job_id="j2", upload_id="u2", user_id="user-1", platforms=["facebook"])
    token = {
        "access_token": "page-token",
        "page_id": "PAGE123",
        "access_non_expiring": True,
        "access_obtained_at": "2026-08-01T00:00:00+00:00",
        "meta_permissions": [{"permission": "pages_manage_posts", "status": "granted"}],
    }
    fake = _FakeAsyncClient()
    fake._post_resp = _FakeResp(
        400,
        {"error": {"message": "Application request limit reached", "code": 4}},
    )

    async def _identity(token_data, *args, **kwargs):
        return token_data

    with (
        patch("stages.publish_stage._refresh_meta_token", new=AsyncMock(side_effect=_identity)),
        patch("stages.publish_stage.httpx.AsyncClient", return_value=fake),
    ):
        result = asyncio.run(publish_to_facebook(video, ctx, token))

    assert result.success is False
    assert result.error_code == "PLATFORM_RATE_LIMIT"


def test_facebook_metrics_hit_videos_and_insights():
    from routers.analytics import _fetch_facebook_metrics

    class _MetricsClient(_FakeAsyncClient):
        async def get(self, url, **kwargs):
            self.gets.append({"url": url, "params": (kwargs.get("params") or {})})
            params = kwargs.get("params") or {}
            fields = str(params.get("fields") or "")
            if url.endswith("/PAGE123/videos"):
                return _FakeResp(200, {"data": [{"id": "v1", "created_time": "2026-08-01T00:00:00+0000"}]})
            if "insights.metric" in fields:
                return _FakeResp(
                    200,
                    {
                        "insights": {
                            "data": [
                                {"name": "total_video_views", "values": [{"value": 10}]},
                            ]
                        }
                    },
                )
            if "followers_count" in fields:
                return _FakeResp(200, {"followers_count": 5})
            return _FakeResp(200, {})

    fake = _MetricsClient()
    with patch("routers.analytics.httpx.AsyncClient", return_value=fake):
        out = asyncio.run(_fetch_facebook_metrics("page-token", "PAGE123"))

    urls = [c["url"] for c in fake.gets]
    assert any(u.endswith("/PAGE123/videos") for u in urls)
    assert any("insights.metric" in str(c.get("params", {}).get("fields") or "") for c in fake.gets)
    assert out.get("status") == "live"
    assert out.get("analytics_source") == "read_insights+pages_read_engagement"
    assert out.get("views") == 10


def test_publish_to_instagram_posts_media_and_media_publish(tmp_path):
    from stages.publish_stage import META_API_VERSION, publish_to_instagram

    video = tmp_path / "reel.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    ctx = JobContext(
        job_id="j-ig",
        upload_id="u-ig",
        user_id="user-1",
        platforms=["instagram"],
        caption="instagram_content_publish proof",
        privacy="public",
    )
    token = {
        "access_token": "page-token",
        "ig_user_id": "IG123",
        "access_non_expiring": True,
        "access_obtained_at": "2026-08-01T00:00:00+00:00",
        "meta_permissions": [
            {"permission": "instagram_content_publish", "status": "granted"},
        ],
    }

    class _IgClient(_FakeAsyncClient):
        async def post(self, url, **kwargs):
            self.posts.append({"url": url, "kwargs": kwargs})
            if str(url).rstrip("/").endswith("/media_publish"):
                return _FakeResp(200, {"id": "ig-media-1"})
            return _FakeResp(200, {"id": "ig-creation-1"})

        async def get(self, url, **kwargs):
            self.gets.append({"url": url, "kwargs": kwargs})
            if "ig-media-1" in str(url):
                return _FakeResp(
                    200,
                    {"permalink": "https://www.instagram.com/reel/abc/", "shortcode": "abc"},
                )
            return _FakeResp(200, {"status_code": "FINISHED"})

    fake = _IgClient()

    async def _identity(token_data, *args, **kwargs):
        return token_data

    with (
        patch("stages.publish_stage._refresh_meta_token", new=AsyncMock(side_effect=_identity)),
        patch("stages.publish_stage._ensure_platform_thumbnail_local", new=AsyncMock(return_value=None)),
        patch("stages.publish_stage.IG_POLL_INTERVAL", 0),
        patch("stages.publish_stage.httpx.AsyncClient", return_value=fake),
    ):
        result = asyncio.run(
            publish_to_instagram(
                video,
                ctx,
                token,
                video_url="https://cdn.example/reel.mp4",
            )
        )

    assert result.success is True
    assert result.platform_video_id == "ig-media-1"
    posted = [p["url"] for p in fake.posts]
    assert instagram_reels_container_url("IG123", version=META_API_VERSION) in posted
    assert instagram_reels_publish_url("IG123", version=META_API_VERSION) in posted


def test_publish_to_instagram_blocks_when_content_publish_declined(tmp_path):
    from stages.publish_stage import publish_to_instagram

    video = tmp_path / "reel.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    ctx = JobContext(job_id="j-ig2", upload_id="u-ig2", user_id="user-1", platforms=["instagram"])
    token = {
        "access_token": "page-token",
        "ig_user_id": "IG123",
        "meta_permissions": [
            {"permission": "instagram_content_publish", "status": "declined"},
        ],
    }
    fake = _FakeAsyncClient()

    async def _identity(token_data, *args, **kwargs):
        return token_data

    with (
        patch("stages.publish_stage._refresh_meta_token", new=AsyncMock(side_effect=_identity)),
        patch("stages.publish_stage.httpx.AsyncClient", return_value=fake),
    ):
        result = asyncio.run(
            publish_to_instagram(video, ctx, token, video_url="https://cdn.example/reel.mp4")
        )

    assert result.success is False
    assert result.error_code == "MISSING_PERMISSION"
    assert not fake.posts


def test_instagram_metrics_hit_media_and_insights():
    from routers.analytics import _fetch_instagram_metrics

    class _IgMetrics(_FakeAsyncClient):
        async def get(self, url, **kwargs):
            self.gets.append({"url": url, "params": (kwargs.get("params") or {})})
            if url.endswith("/IG123/media"):
                return _FakeResp(
                    200,
                    {"data": [{"id": "m1", "media_type": "REELS", "timestamp": "2026-08-01T00:00:00+0000"}]},
                )
            if url.endswith("/m1/insights"):
                return _FakeResp(
                    200,
                    {
                        "data": [
                            {"name": "plays", "values": [{"value": 9}]},
                            {"name": "likes", "values": [{"value": 2}]},
                            {"name": "reach", "values": [{"value": 7}]},
                        ]
                    },
                )
            return _FakeResp(200, {})

    fake = _IgMetrics()
    with patch("routers.analytics.httpx.AsyncClient", return_value=fake):
        out = asyncio.run(_fetch_instagram_metrics("page-token", "IG123"))

    urls = [c["url"] for c in fake.gets]
    assert any(u.endswith("/IG123/media") for u in urls)
    assert any(u.endswith("/m1/insights") for u in urls)
    assert out.get("status") == "live"
    assert out.get("analytics_source") == "instagram_manage_insights"
    assert out.get("views") == 9
    assert out.get("likes") == 2


def test_catalog_facebook_videos_hits_page_videos():
    from services.catalog_sync import _list_facebook_videos
    from services.meta_oauth import META_GRAPH_API_VERSION

    class _Cat(_FakeAsyncClient):
        async def get(self, url, **kwargs):
            self.gets.append({"url": url})
            return _FakeResp(
                200,
                {
                    "data": [
                        {
                            "id": "v1",
                            "title": "Page clip",
                            "length": 12,
                            "created_time": "2026-08-01T00:00:00+0000",
                        }
                    ]
                },
            )

    fake = _Cat()
    with patch("services.catalog_sync.httpx.AsyncClient", return_value=fake):
        videos, _, _ = asyncio.run(_list_facebook_videos("page-token", "PAGE123"))

    assert videos and videos[0]["platform_video_id"] == "v1"
    urls = [c["url"] for c in fake.gets]
    assert any(
        f"/{META_GRAPH_API_VERSION}/PAGE123/videos" in u for u in urls
    )
