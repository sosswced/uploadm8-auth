"""Unit tests for TikTok OAuth token/profile helpers (no live TikTok calls)."""

from __future__ import annotations

import asyncio

import httpx

from services import tiktok_api


def test_parse_token_flat():
    raw = {
        "access_token": "act.x",
        "open_id": "oid-1",
        "refresh_token": "rft.y",
        "expires_in": 3600,
    }
    access, oid, rft, src = tiktok_api.tiktok_parse_oauth_token_response(raw)
    assert access == "act.x"
    assert oid == "oid-1"
    assert rft == "rft.y"
    assert src.get("expires_in") == 3600


def test_parse_token_nested_data():
    raw = {
        "data": {
            "access_token": "act.n",
            "open_id": "oid-2",
            "refresh_token": "rft.n",
            "expires_in": 7200,
        }
    }
    access, oid, rft, src = tiktok_api.tiktok_parse_oauth_token_response(raw)
    assert access == "act.n"
    assert oid == "oid-2"
    assert rft == "rft.n"
    assert src.get("expires_in") == 7200


def test_extract_user_nested():
    body = {
        "data": {
            "user": {
                "open_id": "u1",
                "display_name": "Creator",
                "avatar_url": "https://cdn.example/a.jpg",
            }
        },
        "error": {"code": "ok", "message": ""},
    }
    u = tiktok_api.tiktok_extract_user_from_info_body(body)
    assert u["display_name"] == "Creator"


def test_extract_user_flat_data():
    body = {
        "data": {"open_id": "u2", "display_name": "Flat", "avatar_url_100": "https://x/y.png"},
        "error": {"code": "ok", "message": ""},
    }
    u = tiktok_api.tiktok_extract_user_from_info_body(body)
    assert u["open_id"] == "u2"


def test_identity_prefers_large_avatar():
    u = {
        "open_id": "a",
        "display_name": "N",
        "avatar_large_url": "https://big",
        "avatar_url": "https://small",
    }
    ident = tiktok_api.tiktok_identity_from_user_object(u)
    assert ident["account_avatar"] == "https://big"
    assert ident["account_name"] == "N"


def test_envelope_error_ok_is_none():
    assert tiktok_api.tiktok_envelope_error({"error": {"code": "ok", "message": ""}}) is None


def test_envelope_error_invalid():
    msg = tiktok_api.tiktok_envelope_error(
        {"error": {"code": "invalid_scope", "message": "missing"}}
    )
    assert msg and "missing" in msg


def test_fetch_profile_success_on_first_get():
    payload = {
        "data": {
            "user": {
                "open_id": "723f24d7-e717-40f8-a2b6-cb8464cd23b4",
                "display_name": "Display",
                "avatar_url": "https://p19-sign.tiktokcdn-us.com/a.jpeg",
            }
        },
        "error": {"code": "ok", "message": ""},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "tiktokapis.com" in str(request.url)
        assert "fields=" in str(request.url) or request.method == "POST"
        return httpx.Response(200, json=payload)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            prof = await tiktok_api.fetch_tiktok_user_profile_for_oauth(client, "BearerToken")
        assert prof["account_name"] == "Display"
        assert prof["account_id"] == "723f24d7-e717-40f8-a2b6-cb8464cd23b4"
        assert "tiktokcdn" in prof["account_avatar"]

    asyncio.run(run())


def test_fetch_profile_falls_back_when_full_fields_rejected():
    """Simulate stats fields rejected: first attempts fail, basic fields succeed."""

    ok_body = {
        "data": {
            "user": {
                "open_id": "x",
                "display_name": "OnlyBasic",
                "avatar_url_100": "https://av",
            }
        },
        "error": {"code": "ok", "message": ""},
    }
    err_body = {
        "error": {"code": "scope_not_authorized", "message": "stats"},
        "data": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        u = str(request.url)
        if "follower_count" in u or (
            request.method == "POST"
            and request.content
            and b"follower_count" in request.content
        ):
            return httpx.Response(200, json=err_body)
        return httpx.Response(200, json=ok_body)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            prof = await tiktok_api.fetch_tiktok_user_profile_for_oauth(client, "tok")
        assert prof["account_name"] == "OnlyBasic"

    asyncio.run(run())
