"""Sentry before_send filters — keep production signal high."""

from __future__ import annotations

from core.sentry_init import _before_send


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


def test_keep_real_error():
    ev = _event(exc_type="ValueError", exc_value="unexpected billing state")
    assert _before_send(ev, None) is not None


def test_keep_transient_db_visible():
    """Do not hide DB errors — fix root cause and resolve in Sentry."""
    ev = _event(
        exc_type="CannotConnectNowError",
        exc_value="the database system is not yet accepting connections",
    )
    assert _before_send(ev, None) is not None
