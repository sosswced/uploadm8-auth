"""Unit tests for the presign recent-duplicate-source guard.

Re-submitting the same source file must be rejected while a prior copy is in
flight or recently posted — otherwise one file multiplies across every
connected platform (409 duplicate_upload, bypass via allowDuplicate).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from services.upload.presign import (
    duplicate_upload_guard_enabled,
    duplicate_upload_window_hours,
    reject_recent_duplicate_source,
)


class FakeConn:
    def __init__(self, row=None):
        self.row = row
        self.calls: list[tuple] = []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        return self.row


def _data(**kw):
    base = {"filename": "clip.mp4", "file_size": 1234, "allow_duplicate": False}
    base.update(kw)
    return SimpleNamespace(**base)


def test_duplicate_raises_409(monkeypatch):
    monkeypatch.delenv("DUPLICATE_UPLOAD_GUARD", raising=False)
    monkeypatch.delenv("DUPLICATE_UPLOAD_WINDOW_HOURS", raising=False)
    conn = FakeConn(row={"id": "abc", "status": "processing", "created_at": None})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(reject_recent_duplicate_source(conn, "user-1", _data()))
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "duplicate_upload"
    assert exc.value.detail["existing_upload_id"] == "abc"
    assert len(conn.calls) == 1
    sql, args = conn.calls[0]
    assert "filename" in sql and "file_size" in sql
    assert args[0] == "user-1"
    assert args[1] == "clip.mp4"
    assert args[2] == 1234


def test_no_duplicate_passes(monkeypatch):
    monkeypatch.delenv("DUPLICATE_UPLOAD_GUARD", raising=False)
    conn = FakeConn(row=None)
    asyncio.run(reject_recent_duplicate_source(conn, "user-1", _data()))
    assert len(conn.calls) == 1


def test_allow_duplicate_bypasses(monkeypatch):
    monkeypatch.delenv("DUPLICATE_UPLOAD_GUARD", raising=False)
    conn = FakeConn(row={"id": "abc", "status": "processing", "created_at": None})
    asyncio.run(reject_recent_duplicate_source(conn, "user-1", _data(allow_duplicate=True)))
    assert conn.calls == []


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("DUPLICATE_UPLOAD_GUARD", "0")
    conn = FakeConn(row={"id": "abc", "status": "processing", "created_at": None})
    asyncio.run(reject_recent_duplicate_source(conn, "user-1", _data()))
    assert conn.calls == []


def test_zero_window_disables(monkeypatch):
    monkeypatch.delenv("DUPLICATE_UPLOAD_GUARD", raising=False)
    monkeypatch.setenv("DUPLICATE_UPLOAD_WINDOW_HOURS", "0")
    conn = FakeConn(row={"id": "abc", "status": "processing", "created_at": None})
    asyncio.run(reject_recent_duplicate_source(conn, "user-1", _data()))
    assert conn.calls == []


def test_window_default_and_parsing(monkeypatch):
    monkeypatch.delenv("DUPLICATE_UPLOAD_WINDOW_HOURS", raising=False)
    assert duplicate_upload_window_hours() == 6.0
    monkeypatch.setenv("DUPLICATE_UPLOAD_WINDOW_HOURS", "12")
    assert duplicate_upload_window_hours() == 12.0
    monkeypatch.setenv("DUPLICATE_UPLOAD_WINDOW_HOURS", "bogus")
    assert duplicate_upload_window_hours() == 6.0
    monkeypatch.setenv("DUPLICATE_UPLOAD_WINDOW_HOURS", "-3")
    assert duplicate_upload_window_hours() == 0.0


def test_guard_enabled_parsing(monkeypatch):
    monkeypatch.delenv("DUPLICATE_UPLOAD_GUARD", raising=False)
    assert duplicate_upload_guard_enabled() is True
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("DUPLICATE_UPLOAD_GUARD", off)
        assert duplicate_upload_guard_enabled() is False
