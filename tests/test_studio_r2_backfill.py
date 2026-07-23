"""Free CDN→R2 backfill for Thumbnail Studio saved runs."""

import asyncio
from unittest.mock import AsyncMock, patch

from services.thumbnail_studio import (
    backfill_job_variants_to_r2,
    backfill_variant_preview_to_r2,
    resolve_variant_cdn_url,
)


def test_resolve_variant_cdn_url_from_stored_field():
    u = resolve_variant_cdn_url(
        {"pikzels_cdn_url": "https://cdn.pikzels.com/rest-api/x/y.jpg"}
    )
    assert u.startswith("https://cdn.pikzels.com/")


def test_backfill_skips_when_already_on_r2():
    out = asyncio.run(
        backfill_variant_preview_to_r2(
            user_id="u1",
            job_id="j1",
            variant_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            variant={"preview_r2_key": "thumbnail-studio/previews/u1/j1/variant_1.jpg"},
        )
    )
    assert out["status"] == "already_r2"


def test_backfill_mirrors_live_cdn():
    async def _run():
        with patch(
            "services.thumbnail_studio._download_bytes",
            new=AsyncMock(return_value=b"x" * 4096),
        ), patch(
            "services.thumbnail_studio._r2_put_bytes",
            new=AsyncMock(return_value=None),
        ):
            return await backfill_variant_preview_to_r2(
                user_id="u1",
                job_id="j1",
                variant_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                variant={
                    "index": 2,
                    "pikzels_cdn_url": "https://cdn.pikzels.com/rest-api/live.jpg",
                },
            )

    out = asyncio.run(_run())
    assert out["status"] == "mirrored"
    assert out["preview_r2_key"].startswith("thumbnail-studio/previews/u1/j1/")


def test_backfill_job_marks_cdn_gone():
    async def _run():
        variants = [
            {
                "variant_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "index": 1,
                "pikzels_cdn_url": "https://cdn.pikzels.com/rest-api/gone.jpg",
            }
        ]
        with patch(
            "services.thumbnail_studio._download_bytes",
            new=AsyncMock(return_value=None),
        ):
            return await backfill_job_variants_to_r2(
                user_id="u1", job_id="j1", variants=variants
            )

    summary = asyncio.run(_run())
    assert summary["cdn_gone"] == 1
    assert summary["mirrored"] == 0
    assert summary["free"] is True
