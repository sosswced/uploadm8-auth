"""
FastAPI smoke: public routes that mirror stages/entitlements (no DB round-trip).

Importing `app` initialises Sentry when configured — keep these few and fast.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app import app

    return TestClient(app)


def test_api_entitlements_tiers_matches_module_slugs(client) -> None:
    from stages.entitlements import TIER_SLUGS

    r = client.get("/api/entitlements/tiers")
    assert r.status_code == 200
    data = r.json()
    assert set(data.get("tier_slugs", [])) == set(TIER_SLUGS)
    assert len(data.get("tiers", [])) == len(TIER_SLUGS)
    assert data.get("entitlement_keys")


def test_api_entitlements_alias(client) -> None:
    a = client.get("/api/entitlements/tiers").json()
    b = client.get("/api/entitlements").json()
    assert a == b


def test_thumbnail_studio_layout_formats_public(client) -> None:
    """No Authorization header — must stay 200 (Thumbnail Studio niche chips)."""
    r = client.get("/api/entitlements/thumbnail-studio-formats?niche=automotive")
    assert r.status_code == 200
    data = r.json()
    assert "formats" in data
    assert isinstance(data["formats"], list)
    assert len(data["formats"]) >= 1


def test_api_pricing_returns_tiers_and_topups(client) -> None:
    from stages.entitlements import TOPUP_PRODUCTS

    r = client.get("/api/pricing")
    assert r.status_code == 200
    body = r.json()
    assert len(body["tiers"]) == 5
    assert len(body["topups"]) == len(TOPUP_PRODUCTS)
    for tier in body["tiers"]:
        assert "max_thumbnails" in tier
        assert "max_caption_frames" in tier
        assert "trial_days" in tier
    assert body["tiers"][0]["slug"] == "free"
    assert body["tiers"][0]["trial_days"] == 0
    assert body["tiers"][1]["trial_days"] == 7
    assert body["tiers"][0]["max_caption_frames"] == 3
    assert body["tiers"][0]["max_thumbnails"] == 3
    assert body["tiers"][4]["max_thumbnails"] == 20
    assert body["tiers"][4]["max_caption_frames"] == 15
