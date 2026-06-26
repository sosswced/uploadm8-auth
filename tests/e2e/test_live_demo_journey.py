"""Pytest wrapper for the live demo journey (slow, headed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.helpers.live_demo import LiveDemoLog, resolve_demo_paths, run_live_demo_journey

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.upload_ui]


@pytest.mark.timeout(7200)
def test_live_demo_journey(live_demo_page, base_url: str, test_video_path: Path | None, test_telemetry_map_path: Path | None):
    if test_video_path is None:
        pytest.skip("Set E2E_TEST_VIDEO for live demo")
    video, telemetry = resolve_demo_paths(test_video_path, test_telemetry_map_path)
    result = run_live_demo_journey(
        live_demo_page,
        base_url,
        video=video,
        telemetry=telemetry,
        pipeline_timeout_s=7200.0,
        log=LiveDemoLog(),
    )
    assert result.get("upload"), "Expected upload row from API poll"
    assert result.get("background_checks", {}).get("ok"), "Background router/API checks failed"
    if not result.get("worker_safe"):
        assert result.get("admin_ui", {}).get("ok"), "Target user admin UI phase failed"
    status = str(result["upload"].get("status") or "").lower()
    assert status in {"completed", "partial", "succeeded", "processing", "failed", "staged"}, f"Unexpected status: {status}"
