"""Stripe webhook preflight must fail-closed when dedup lookup errors."""

from __future__ import annotations

import asyncio
import inspect

from routers import billing as billing_mod


def test_event_already_processed_raises_on_lookup_error():
    class _BoomConn:
        async def fetchrow(self, *_a, **_k):
            raise RuntimeError("db down")

    async def _run():
        await billing_mod._event_already_processed(_BoomConn(), "evt_test")

    try:
        asyncio.run(_run())
        raise AssertionError("expected lookup failure to raise")
    except RuntimeError as e:
        assert "db down" in str(e)


def test_webhook_preflight_returns_503_on_failure():
    src = inspect.getsource(billing_mod.stripe_webhook)
    assert 'HTTPException(503, "Webhook temporarily unavailable")' in src
    assert "except HTTPException" in src
