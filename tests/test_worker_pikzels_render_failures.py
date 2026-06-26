"""Worker thumbnail stage — Pikzels render failure ops incident."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from services.thumbnail_ops import record_pikzels_render_failures_incident


def test_pikzels_render_failures_emits_ops_incident():
    upload_id = "c1c1c1c1-c1c1-c1c1-c1c1-c1c1c1c1c1c1"
    user_id = "user-pikzels-ops"
    failures = [
        {"platform": "youtube", "http_status": 500, "message": "upstream timeout"},
        {"platform": "tiktok", "http_status": 402, "message": "payment required"},
    ]
    artifacts = {"pikzels_render_failures": failures}

    async def _run():
        with patch(
            "services.ops_incidents.record_operational_incident",
            new=AsyncMock(),
        ) as mock_incident:
            await record_pikzels_render_failures_incident(
                object(),
                upload_id=upload_id,
                user_id=user_id,
                output_artifacts=artifacts,
            )
            return mock_incident

    mock_incident = asyncio.run(_run())
    mock_incident.assert_awaited_once()
    kwargs = mock_incident.await_args.kwargs
    assert kwargs["source"] == "thumbnail"
    assert kwargs["upload_id"] == upload_id
    assert kwargs["user_id"] == user_id
    assert "youtube:500" in kwargs["incident_type"]
    assert len(kwargs["details"]["failures"]) == 2
    assert kwargs["alert_email"] is True  # mixed 402 + 500 → not payment-only


def test_pikzels_render_failures_json_string_artifacts():
    upload_id = "d2d2d2d2-d2d2-d2d2-d2d2-d2d2d2d2d2d2"
    raw = json.dumps([{"platform": "instagram", "http_status": 503, "message": "busy"}])
    artifacts = {"pikzels_render_failures": raw}

    async def _run():
        with patch(
            "services.ops_incidents.record_operational_incident",
            new=AsyncMock(),
        ) as mock_incident:
            await record_pikzels_render_failures_incident(
                object(),
                upload_id=upload_id,
                user_id=None,
                output_artifacts=artifacts,
            )
            return mock_incident

    mock_incident = asyncio.run(_run())
    mock_incident.assert_awaited_once()
    assert mock_incident.await_args.kwargs["details"]["failures"][0]["platform"] == "instagram"


def test_pikzels_render_failures_skips_when_absent():
    async def _run():
        with patch(
            "services.ops_incidents.record_operational_incident",
            new=AsyncMock(),
        ) as mock_incident:
            await record_pikzels_render_failures_incident(
                object(),
                upload_id="skip-me",
                user_id="u1",
                output_artifacts={},
            )
            return mock_incident

    mock_incident = asyncio.run(_run())
    mock_incident.assert_not_called()


def test_pikzels_402_only_suppresses_email_alerts():
    artifacts = {
        "pikzels_render_failures": [
            {"platform": "youtube", "http_status": 402, "message": "quota"},
        ]
    }

    async def _run():
        with patch(
            "services.ops_incidents.record_operational_incident",
            new=AsyncMock(),
        ) as mock_incident:
            await record_pikzels_render_failures_incident(
                object(),
                upload_id="paywall-upload",
                user_id="u1",
                output_artifacts=artifacts,
            )
            return mock_incident

    mock_incident = asyncio.run(_run())
    kwargs = mock_incident.await_args.kwargs
    assert kwargs["alert_email"] is False
    assert kwargs["alert_discord"] is False
