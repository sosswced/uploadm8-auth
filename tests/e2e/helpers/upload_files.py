"""Playwright helpers for upload.html file pairing (video + .map)."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page

# Playwright rejects in-memory buffers above ~50MB.
_MAX_BUFFER_BYTES = 48 * 1024 * 1024


def set_upload_file_pair(page: Page, video: Path, telemetry: Path | None = None) -> None:
    """
    Set files on #fileInput with correct video/* MIME for upload.html.

    upload.html filters videos with file.type.startsWith('video/'); Windows path
    uploads often leave type empty unless MIME is set explicitly.
    """
    page.wait_for_selector("#fileInput", state="attached", timeout=90_000)
    page.wait_for_function("() => window.__uploadm8DropZoneWired === true", timeout=90_000)
    inp = page.locator("#fileInput")
    if inp.count() == 0:
        inp = page.locator('input[type="file"]')

    video_size = video.stat().st_size
    if video_size <= _MAX_BUFFER_BYTES:
        payload: list[dict | str] = [
            {
                "name": video.name,
                "mimeType": "video/mp4",
                "buffer": video.read_bytes(),
            }
        ]
        if telemetry is not None:
            payload.append(
                {
                    "name": telemetry.name,
                    "mimeType": "application/octet-stream",
                    "buffer": telemetry.read_bytes(),
                }
            )
        inp.first.set_input_files(payload)
        _dispatch_handle_files(page)
        return

    paths = [str(video)]
    if telemetry is not None:
        paths.append(str(telemetry))
    inp.first.set_input_files(paths, timeout=180_000)
    page.wait_for_function(
        "() => document.getElementById('fileInput')?.files?.length > 0",
        timeout=180_000,
    )
    _fix_mime_and_dispatch_change(page)
    page.wait_for_timeout(500)


def _dispatch_handle_files(page: Page) -> None:
    """Fire change on #fileInput after buffer-based set_input_files."""
    page.evaluate(
        """() => {
            const input = document.getElementById('fileInput');
            if (!input || !input.files || !input.files.length) return false;
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }"""
    )


def _fix_mime_and_dispatch_change(page: Page) -> None:
    """Re-wrap chosen files with explicit MIME types and fire change (large local files)."""
    page.evaluate(
        """() => {
            const input = document.getElementById('fileInput');
            if (!input || !input.files || !input.files.length) return false;
            const dt = new DataTransfer();
            for (const f of input.files) {
                const lower = f.name.toLowerCase();
                let type = f.type;
                if (!type || type === 'application/octet-stream') {
                    if (lower.endsWith('.mp4') || lower.endsWith('.mov') || lower.endsWith('.webm')) {
                        type = 'video/mp4';
                    } else if (lower.endsWith('.map')) {
                        type = 'application/octet-stream';
                    }
                }
                dt.items.add(new File([f], f.name, { type }));
            }
            input.files = dt.files;
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }"""
    )
