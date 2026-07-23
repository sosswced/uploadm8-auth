"""Ops alert gating for pikzels_template_fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from services.thumbnail_ops import record_pikzels_template_render_incident


def test_template_fallback_suppressed_when_empty_method_and_report():
    record = AsyncMock()

    async def _run():
        with patch("stages.pikzels_api.studio_renderer_enabled", return_value=True), patch(
            "services.ops_incidents.record_operational_incident", record
        ):
            await record_pikzels_template_render_incident(
                object(),
                upload_id="u1",
                user_id="user-1",
                render_method="",
                studio_report={},
            )

    asyncio.run(_run())
    record.assert_not_awaited()


def test_template_fallback_fires_for_template_with_skip_reason():
    record = AsyncMock()

    async def _run():
        with patch("stages.pikzels_api.studio_renderer_enabled", return_value=True), patch(
            "services.ops_incidents.record_operational_incident", record
        ):
            await record_pikzels_template_render_incident(
                object(),
                upload_id="u2",
                user_id="user-1",
                render_method="template",
                studio_report={"skip_reason": "studio_engine_disabled"},
            )

    asyncio.run(_run())
    record.assert_awaited_once()
    kwargs = record.await_args.kwargs
    assert kwargs["incident_type"] == "pikzels_template_fallback"
    assert kwargs["details"]["skip_reason"] == "studio_engine_disabled"
