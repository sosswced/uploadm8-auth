"""Accurate / quiet worker scale Discord notifications."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from stages.notify_stage import (
    _scale_should_discord,
    _worker_scale_discord_mode,
    _worker_scale_embed,
    notify_admin_worker_start,
    notify_admin_worker_stop,
)


def test_scale_discord_mode_default_edges(monkeypatch):
    monkeypatch.delenv("WORKER_SCALE_DISCORD_MODE", raising=False)
    assert _worker_scale_discord_mode() == "edges"


def test_scale_should_discord_edges_only_first_and_last(monkeypatch):
    monkeypatch.setenv("WORKER_SCALE_DISCORD_MODE", "edges")
    assert _scale_should_discord("start", 1) is True
    assert _scale_should_discord("start", 0) is False  # no heartbeat yet
    assert _scale_should_discord("start", 2) is False
    assert _scale_should_discord("stop", 0, stale=0) is True
    assert _scale_should_discord("stop", 0, stale=2) is False  # stale peers ≠ empty
    assert _scale_should_discord("stop", 2) is False


def test_scale_embed_uses_alive_not_projected():
    with patch(
        "stages.notify_stage._fleet_alive_counts",
        new=AsyncMock(return_value={"alive": 2, "stale": 0, "dead": 4, "recent_dead": 0}),
    ):
        stop_payload = asyncio.run(_worker_scale_embed(None, "stop"))
        start_payload = asyncio.run(_worker_scale_embed(None, "start"))
    assert stop_payload["alive"] == 2
    assert "2 still active" in stop_payload["content"]
    assert "Projected" not in str(stop_payload)
    assert start_payload["alive"] == 2
    assert "2 active" in start_payload["content"]
    # Informational colors — not alarm red/green
    assert stop_payload["embeds"][0]["color"] == 0x64748B
    assert start_payload["embeds"][0]["color"] == 0x3B82F6


def test_scale_embed_stop_not_empty_when_stale_peers():
    with patch(
        "stages.notify_stage._fleet_alive_counts",
        new=AsyncMock(return_value={"alive": 0, "stale": 1, "dead": 2, "recent_dead": 1}),
    ):
        payload = asyncio.run(_worker_scale_embed(None, "stop"))
    assert "not empty" in payload["content"]
    assert payload["stale"] == 1
    assert _scale_should_discord("stop", 0, stale=1) is False


def test_notify_stop_suppressed_when_peers_remain(monkeypatch):
    monkeypatch.setenv("WORKER_SCALE_DISCORD_MODE", "edges")
    with patch(
        "stages.notify_stage._worker_scale_embed",
        new=AsyncMock(
            return_value={
                "content": "Worker left — 2 still active",
                "alive": 2,
                "stale": 0,
                "embeds": [],
            }
        ),
    ), patch(
        "stages.notify_stage._send_discord_webhook",
        new=AsyncMock(),
    ) as send, patch(
        "stages.notify_stage._get_admin_webhook",
        new=AsyncMock(return_value="https://discord.com/api/webhooks/1/x"),
    ):
        asyncio.run(notify_admin_worker_stop(None))
    send.assert_not_awaited()


def test_notify_stop_pages_when_fleet_empty(monkeypatch):
    monkeypatch.setenv("WORKER_SCALE_DISCORD_MODE", "edges")
    with patch(
        "stages.notify_stage._worker_scale_embed",
        new=AsyncMock(
            return_value={
                "content": "Last worker offline — fleet empty",
                "alive": 0,
                "stale": 0,
                "embeds": [{"title": "Worker fleet"}],
            }
        ),
    ), patch(
        "stages.notify_stage._send_discord_webhook",
        new=AsyncMock(),
    ) as send, patch(
        "stages.notify_stage._get_admin_webhook",
        new=AsyncMock(return_value="https://discord.com/api/webhooks/1/x"),
    ):
        asyncio.run(notify_admin_worker_stop(None))
    send.assert_awaited_once()


def test_notify_start_pages_first_worker(monkeypatch):
    monkeypatch.setenv("WORKER_SCALE_DISCORD_MODE", "edges")
    with patch(
        "stages.notify_stage._worker_scale_embed",
        new=AsyncMock(
            return_value={
                "content": "Worker online — fleet was empty",
                "alive": 1,
                "stale": 0,
                "embeds": [{"title": "Worker fleet"}],
            }
        ),
    ), patch(
        "stages.notify_stage._send_discord_webhook",
        new=AsyncMock(),
    ) as send, patch(
        "stages.notify_stage._get_admin_webhook",
        new=AsyncMock(return_value="https://discord.com/api/webhooks/1/x"),
    ):
        asyncio.run(notify_admin_worker_start(None))
    send.assert_awaited_once()


def test_notify_start_suppressed_when_alive_zero(monkeypatch):
    monkeypatch.setenv("WORKER_SCALE_DISCORD_MODE", "edges")
    with patch(
        "stages.notify_stage._worker_scale_embed",
        new=AsyncMock(
            return_value={
                "content": "Worker online — heartbeat not visible yet",
                "alive": 0,
                "stale": 0,
                "embeds": [],
            }
        ),
    ), patch(
        "stages.notify_stage._send_discord_webhook",
        new=AsyncMock(),
    ) as send, patch(
        "stages.notify_stage._get_admin_webhook",
        new=AsyncMock(return_value="https://discord.com/api/webhooks/1/x"),
    ):
        asyncio.run(notify_admin_worker_start(None))
    send.assert_not_awaited()
