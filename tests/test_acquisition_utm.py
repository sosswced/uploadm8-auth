"""UTM helpers and acquisition rollup shapes for Acquisition Command Center."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from services.auth_credentials import _clean_utm


def test_clean_utm_strips_and_truncates():
    assert _clean_utm("  meta  ") == "meta"
    assert _clean_utm("") is None
    assert _clean_utm(None) is None
    long = "x" * 200
    assert len(_clean_utm(long) or "") == 128


def test_clean_utm_drops_markup_chars():
    # Admin CRM renders UTM values; never persist markup-capable input.
    assert _clean_utm("<img src=x onerror=alert(1)>") == "img srcx onerroralert1"
    assert _clean_utm('"><script>') == "script"
    assert _clean_utm("<<>>") is None
    assert _clean_utm("summer_sale-2026/v2") == "summer_sale-2026/v2"


def test_user_create_accepts_optional_utm():
    from core.models import UserCreate

    u = UserCreate(
        email="a@b.co",
        password="password1",
        name="Test User",
        utm_source="meta",
        utm_medium="paid_social",
        utm_campaign="traction_v1",
        utm_content="hook_1",
    )
    assert u.utm_source == "meta"
    assert u.utm_campaign == "traction_v1"


def test_fetch_acquisition_by_utm_empty():
    from services.growth_intelligence import fetch_acquisition_by_utm

    async def _run():
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        until = datetime(2026, 2, 1, tzinfo=timezone.utc)
        return await fetch_acquisition_by_utm(conn, since, until)

    out = asyncio.run(_run())
    assert out["signups_with_utm"] == 0
    assert out["paid_converts"] == 0
    assert out["campaigns"] == []


def test_fetch_acquisition_by_utm_rows():
    from services.growth_intelligence import fetch_acquisition_by_utm

    async def _run():
        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "utm_source": "meta",
                    "utm_medium": "paid_social",
                    "utm_campaign": "traction_v1",
                    "signups": 3,
                    "paid_converts": 1,
                    "revenue_usd": 19.0,
                }
            ]
        )
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        until = datetime(2026, 2, 1, tzinfo=timezone.utc)
        return await fetch_acquisition_by_utm(conn, since, until)

    out = asyncio.run(_run())
    assert out["signups_with_utm"] == 3
    assert out["paid_converts"] == 1
    assert out["campaigns"][0]["utm_campaign"] == "traction_v1"
    assert out["campaigns"][0]["revenue_usd"] == 19.0


def test_build_marketing_intel_includes_acquisition(monkeypatch):
    from services import growth_intelligence as gi

    async def _funnel(*a, **k):
        return {"shown": 0}

    async def _levers(*a, **k):
        return {}

    async def _promos(*a, **k):
        return []

    async def _comms(*a, **k):
        return []

    async def _acq(*a, **k):
        return {"signups_with_utm": 2, "paid_converts": 0, "campaigns": []}

    async def _life(*a, **k):
        return {"by_signup_source": [{"signup_source": "meta", "signups": 2}], "by_lifecycle_stage": []}

    monkeypatch.setattr(gi, "fetch_marketing_funnel", _funnel)
    monkeypatch.setattr(gi, "fetch_sales_opportunity_levers", _levers)
    monkeypatch.setattr(gi, "fetch_promo_schedule_hints", _promos)
    monkeypatch.setattr(gi, "build_recommended_comms", _comms)
    monkeypatch.setattr(gi, "fetch_acquisition_by_utm", _acq)
    monkeypatch.setattr(gi, "fetch_lifecycle_crm_rollup", _life)

    out = asyncio.run(gi.build_marketing_intel_bundle(MagicMock(), "30d"))
    assert "acquisition_by_utm" in out
    assert out["acquisition_by_utm"]["signups_with_utm"] == 2
    assert out["lifecycle_crm"]["by_signup_source"][0]["signup_source"] == "meta"


def test_derive_signup_source_and_lifecycle():
    from services.activation_onboarding import derive_signup_source, effective_lifecycle_stage

    assert derive_signup_source("meta") == "meta"
    assert derive_signup_source(None, "paid_social", "c1") == "campaign_unknown"
    assert derive_signup_source(None) == "direct"
    assert effective_lifecycle_stage(subscription_status="trialing") == "trial"
    assert effective_lifecycle_stage(subscription_status="active", subscription_tier="creator_pro") == "paid"
    assert effective_lifecycle_stage(subscription_status="canceled") == "churned"
    assert effective_lifecycle_stage(stored="signed_up") == "signed_up"


def test_fetch_activation_checklist_shape():
    from services.activation_onboarding import fetch_activation_checklist

    async def _run():
        conn = AsyncMock()
        # preferences, connected, uploaded, scheduled
        conn.fetchval = AsyncMock(side_effect=[None, True, False, False])
        return await fetch_activation_checklist(conn, "00000000-0000-0000-0000-000000000001")

    out = asyncio.run(_run())
    assert out["total"] == 3
    assert out["steps"][0]["id"] == "connect"
    assert out["steps"][0]["done"] is True
    assert out["steps"][1]["done"] is False
    assert out["show_card"] is True
    assert out["show_playbook_modal"] is False  # connected → no playbook nag
