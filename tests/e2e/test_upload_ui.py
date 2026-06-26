"""Browser upload flow — video + optional paired .map telemetry."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.helpers.browser_session import navigate_to_page_human, wait_for_authenticated_shell
from tests.e2e.helpers.upload_files import set_upload_file_pair

pytestmark = [pytest.mark.e2e, pytest.mark.upload_ui, pytest.mark.slow, pytest.mark.overnight]


@pytest.mark.timeout(600)
def test_upload_page_accepts_video_and_telemetry(
    human_session_page,
    base_url: str,
    test_video_path: Path | None,
    test_telemetry_map_path: Path | None,
):
    if test_video_path is None:
        pytest.skip("Set E2E_TEST_VIDEO in .env for upload UI tests")

    navigate_to_page_human(human_session_page, base_url, "upload.html")
    wait_for_authenticated_shell(human_session_page)

    set_upload_file_pair(human_session_page, test_video_path, test_telemetry_map_path)
    human_session_page.locator("#fileList:not(.hidden) .file-item").first.wait_for(
        state="visible",
        timeout=30_000,
    )

    if test_telemetry_map_path is not None:
        body = human_session_page.locator("body").inner_text(timeout=5000)
        assert "Paired with" in body or test_telemetry_map_path.name in body, (
            f"Expected telemetry pair UI for {test_telemetry_map_path.name}"
        )

    api_fails = getattr(human_session_page, "_e2e_failed_requests", [])
    hard_fails = [f for f in api_fails if "presign" in f.lower() or " 500 " in f]
    assert not hard_fails, f"Upload API errors: {hard_fails[:5]}"
