"""Light integration checks for critical HTTP surface (no DB required for webhooks GET)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_tiktok_webhook_challenge_ok():
    from app import app

    c = TestClient(app)
    r = c.get("/api/webhooks/tiktok")
    assert r.status_code == 200
    assert r.json().get("status") == "tiktok-webhook-endpoint-ok"


def test_facebook_webhook_challenge_health():
    from app import app

    c = TestClient(app)
    r = c.get("/api/webhooks/facebook")
    assert r.status_code == 200


def test_presign_requires_auth():
    from app import app

    c = TestClient(app)
    r = c.post(
        "/api/uploads/presign",
        json={
            "filename": "x.mp4",
            "file_size": 1000,
            "content_type": "video/mp4",
            "platforms": ["youtube"],
        },
    )
    assert r.status_code in (401, 403)


def test_catalog_sync_status_requires_auth():
    from app import app

    c = TestClient(app)
    r = c.get("/api/catalog/sync-status")
    assert r.status_code in (401, 403)


def test_analytics_requires_auth():
    from app import app

    c = TestClient(app)
    r = c.get("/api/analytics")
    assert r.status_code in (401, 403)


def test_shell_bootstrap_requires_auth():
    from app import app

    c = TestClient(app)
    r = c.get("/api/shell/bootstrap?context=dashboard&upload_limit=50")
    assert r.status_code in (401, 403)
