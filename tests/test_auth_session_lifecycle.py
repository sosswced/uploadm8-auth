"""Auth session lifecycle: logout with expired access, login must_reset flag."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import core.state
from app import app
from core.config import AUTH_REFRESH_COOKIE
from core.helpers import _sha256_hex


USER_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_args):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.fixture
def auth_client():
    with TestClient(app) as client:
        yield client


def test_logout_clears_cookies_without_access_token(auth_client: TestClient):
    """Expired/missing access must still clear cookies when refresh identifies the user."""
    refresh_plain = "test-refresh-token-plain"
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"user_id": USER_ID})
    conn.execute = AsyncMock()
    core.state.db_pool = FakePool(conn)

    auth_client.cookies.set(AUTH_REFRESH_COOKIE, refresh_plain)
    r = auth_client.post("/api/auth/logout", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "logged_out"
    conn.execute.assert_awaited()


def test_logout_with_refresh_in_body(auth_client: TestClient):
    refresh_plain = "body-refresh-token"
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"user_id": USER_ID})
    conn.execute = AsyncMock()
    core.state.db_pool = FakePool(conn)

    r = auth_client.post("/api/auth/logout", json={"refresh_token": refresh_plain})
    assert r.status_code == 200
    assert r.json()["status"] == "logged_out"
    args = conn.fetchrow.await_args.args
    assert args[1] == _sha256_hex(refresh_plain)


def test_logout_without_session_still_ok(auth_client: TestClient):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    core.state.db_pool = FakePool(conn)

    r = auth_client.post("/api/auth/logout", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "logged_out"
    conn.execute.assert_not_awaited()


def test_login_includes_must_reset_password(auth_client: TestClient):
    from core.auth import hash_password

    pw = "CorrectHorseBattery1"
    row = {
        "id": USER_ID,
        "password_hash": hash_password(pw),
        "status": "active",
        "email_verified": True,
        "must_reset_password": True,
    }
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    conn.execute = AsyncMock()
    core.state.db_pool = FakePool(conn)

    with patch("services.auth_credentials.create_refresh_token", new=AsyncMock(return_value="rtok")):
        with patch("services.auth_credentials.create_access_jwt", return_value="atok"):
            r = auth_client.post(
                "/api/auth/login",
                json={"email": "reset@test.uploadm8.com", "password": pw},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["must_reset_password"] is True
    assert body["access_token"] == "atok"
    assert body["refresh_token"] == "rtok"


def test_build_me_response_includes_must_reset_password():
    from services.me_profile import build_me_response

    payload = build_me_response(
        {
            "id": USER_ID,
            "email": "a@b.com",
            "subscription_tier": "free",
            "role": "user",
            "wallet": {},
            "must_reset_password": True,
        }
    )
    assert payload["must_reset_password"] is True
