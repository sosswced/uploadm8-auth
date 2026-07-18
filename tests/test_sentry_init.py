"""Sentry before_send filters — keep production signal high."""

from __future__ import annotations

from core.sentry_init import (
    _before_send,
    _before_send_transaction,
    _env_name,
    _is_localhost_url,
)


def _event(message: str = "", exc_type: str = "", exc_value: str = "") -> dict:
    ev: dict = {}
    if message:
        ev["logentry"] = {"message": message}
    if exc_type:
        ev["exception"] = {"values": [{"type": exc_type, "value": exc_value}]}
    return ev


def test_drop_none_pool_acquire():
    ev = _event(
        exc_type="AttributeError",
        exc_value="'NoneType' object has no attribute 'acquire'",
    )
    assert _before_send(ev, None) is None


def test_drop_connection_lost_noise():
    ev = _event(
        message="Future exception was never retrieved\nfuture: <Future finished exception=ConnectionError('unexpected connection_lost() call')>",
        exc_type="ConnectionError",
        exc_value="unexpected connection_lost() call",
    )
    assert _before_send(ev, None) is None


def test_drop_localhost_acquire_timeout():
    ev = {
        "request": {"url": "http://127.0.0.1:8000/api/admin/kpis"},
        "exception": {"values": [{"type": "TimeoutError", "value": ""}]},
    }
    assert _before_send(ev, None) is None


def test_keep_real_error():
    ev = _event(exc_type="ValueError", exc_value="unexpected billing state")
    assert _before_send(ev, None) is not None


def test_keep_transient_db_visible():
    """Do not hide production DB errors — fix root cause and resolve in Sentry."""
    ev = {
        "request": {"url": "https://auth.uploadm8.com/api/me"},
        "exception": {
            "values": [
                {
                    "type": "CannotConnectNowError",
                    "value": "the database system is not yet accepting connections",
                }
            ]
        },
    }
    assert _before_send(ev, None) is not None


def test_drop_localhost_performance_transactions():
    """Local E2E must not regress production Slow DB Query issues."""
    ev = {
        "request": {"url": "http://127.0.0.1:8000/api/admin/kpis"},
        "spans": [{"description": "SELECT 1", "op": "db"}],
    }
    assert _before_send_transaction(ev, None) is None


def test_keep_remote_transactions():
    ev = {
        "request": {"url": "https://auth.uploadm8.com/api/me"},
        "spans": [{"description": "SELECT 1", "op": "db"}],
    }
    assert _before_send_transaction(ev, None) is not None


def test_localhost_url_helper():
    assert _is_localhost_url("http://127.0.0.1:8000/health")
    assert not _is_localhost_url("https://auth.uploadm8.com/api/me")


def test_env_name_prefers_development_for_local_e2e(monkeypatch):
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "production")
    monkeypatch.setenv("E2E_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("BASE_URL", "https://auth.uploadm8.com")
    # Without local uvicorn argv, E2E_BASE_URL alone + production tag → development
    assert _env_name() == "development"
