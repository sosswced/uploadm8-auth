"""Durable Thumbnail Studio R2 preview helpers."""

from services.thumbnail_studio import (
    STUDIO_PREVIEW_PRESIGN_TTL_SEC,
    STUDIO_PREVIEW_RETENTION_DAYS,
    attach_preview_urls_to_variants,
)


def test_studio_preview_retention_defaults_near_ten_months():
    assert 240 <= STUDIO_PREVIEW_RETENTION_DAYS <= 366
    assert STUDIO_PREVIEW_PRESIGN_TTL_SEC == 7 * 86400


def test_attach_preview_urls_marks_r2_storage(monkeypatch):
    monkeypatch.setattr(
        "services.thumbnail_studio.presign_variant_preview_url",
        lambda key, ttl=None: f"https://signed.example/{key}?t={ttl}",
    )
    variants = [
        {
            "index": 1,
            "preview_r2_key": "thumbnail-studio/previews/u/j/variant_1.jpg",
            "pikzels_cdn_url": "https://cdn.pikzels.com/old.jpg",
        }
    ]
    attach_preview_urls_to_variants(variants)
    assert variants[0]["preview_storage"] == "r2"
    assert variants[0]["preview_retention_days"] == STUDIO_PREVIEW_RETENTION_DAYS
    assert variants[0]["preview_url"].startswith("https://signed.example/")
