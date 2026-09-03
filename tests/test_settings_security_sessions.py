"""Settings security: email change, sessions list, logout-other, timezone profile."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import core.state
from app import app
from core.auth import hash_password
from core.config import AUTH_REFRESH_COOKIE
from services.me_profile import apply_settings_profile_update
from core.models import ProfileUpdateSettings


USER_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"


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
def client():
    with TestClient(app) as c:
        yield c


def _auth_user():
    return {
        "id": USER_ID,
        "email": "tester@example.com",
        "name": "Tester",
        "role": "user",
        "subscription_tier": "free",
        "status": "active",
        "email_verified": True,
    }


def test_apply_settings_profile_update_persists_timezone():
    import asyncio

    conn = AsyncMock()
    conn.execute = AsyncMock()
    data = ProfileUpdateSettings(first_name="A", last_name="B", timezone="America/Chicago")

    async def _run():
        return await apply_settings_profile_update(conn, USER_ID, data, {"name": "Old"})

    did, msg = asyncio.run(_run())
    assert did is True
    assert "updated" in msg.lower()
    sql = conn.execute.await_args.args[0]
    assert "timezone" in sql
    assert "America/Chicago" in conn.execute.await_args.args


def test_settings_email_endpoint_exists_and_requires_password():
    from routers import me as me_router

    src = inspect.getsource(me_router.update_email_settings)
    assert "verify_password" in src
    assert "send_email_change_email" in src
    assert "pending_verification" in src or "email_changes" in src
    assert "email_verified=false" not in src.replace(" ", "")


def test_put_settings_email_success(client: TestClient):
    pw = "CorrectHorseBattery1"
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "id": USER_ID,
            "email": "tester@example.com",
            "name": "Tester",
            "password_hash": hash_password(pw),
        }
    )
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    core.state.db_pool = FakePool(conn)

    from core.deps import get_current_user

    app.dependency_overrides[get_current_user] = lambda: _auth_user()
    try:
        with patch("routers.me.send_email_change_email", new=AsyncMock()):
            r = client.put(
                "/api/settings/email",
                json={"new_email": "new@example.com", "current_password": pw},
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"] == "tester@example.com"
    assert body.get("pending_email") == "new@example.com"
    assert body.get("status") == "pending_verification"
    assert body.get("email_verified") is False


def test_put_settings_email_wrong_password(client: TestClient):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "id": USER_ID,
            "email": "tester@example.com",
            "name": "Tester",
            "password_hash": hash_password("CorrectHorseBattery1"),
        }
    )
    core.state.db_pool = FakePool(conn)
    from core.deps import get_current_user

    app.dependency_overrides[get_current_user] = lambda: _auth_user()
    try:
        r = client.put(
            "/api/settings/email",
            json={"new_email": "new@example.com", "current_password": "wrong-password"},
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert r.status_code == 401


def test_list_sessions_and_logout_other(client: TestClient):
    now = datetime.now(timezone.utc)
    refresh_plain = "current-refresh-token-plain"
    current_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    other_sid = "ffffffff-ffff-ffff-ffff-ffffffffffff"

    class Conn:
        def __init__(self):
            self._mode = "list"

        async def fetch(self, sql, *args):
            sql_l = sql.lower()
            if "update refresh_tokens" in sql_l:
                return [{"id": other_sid}]
            # list sessions
            return [
                {
                    "id": current_id,
                    "created_at": now - timedelta(hours=1),
                    "expires_at": now + timedelta(days=7),
                },
                {
                    "id": other_sid,
                    "created_at": now - timedelta(days=1),
                    "expires_at": now + timedelta(days=7),
                },
            ]

        async def fetchrow(self, sql, *args):
            # current session lookup by hash
            return {"id": current_id}

        async def fetchval(self, sql, *args):
            return current_id

        async def execute(self, *a, **k):
            return None

    conn = Conn()
    core.state.db_pool = FakePool(conn)
    from core.deps import get_current_user

    app.dependency_overrides[get_current_user] = lambda: _auth_user()
    try:
        client.cookies.set(AUTH_REFRESH_COOKIE, refresh_plain)
        listed = client.get("/api/auth/sessions")
        assert listed.status_code == 200, listed.text
        payload = listed.json()
        assert payload["active_count"] == 2
        assert payload["other_count"] == 1
        assert any(s["is_current"] for s in payload["sessions"])

        revoked = client.post("/api/auth/logout-other-sessions", json={})
        assert revoked.status_code == 200, revoked.text
        assert revoked.json()["sessions_revoked"] == 1
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_settings_html_wires_security_actions():
    from pathlib import Path

    html = Path("frontend/settings.html").read_text(encoding="utf-8")
    assert "/api/settings/email" in html
    assert "/api/auth/sessions" in html
    assert "/api/auth/logout-other-sessions" in html
    assert "/api/settings/password" in html
    assert "encodeURIComponent(platform)" in html
    assert "timezone: tzEl" in html or "timezone: tzEl ?" in html
