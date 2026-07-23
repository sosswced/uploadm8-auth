"""Shell bootstrap helpers (unit-level, no DB)."""

import asyncio
from unittest.mock import AsyncMock, patch

from services.shell_bootstrap import _bootstrap_schedule_repair, _ok


def test_ok_coerces_exceptions_to_none():
    assert _ok({"uploads": []}) == {"uploads": []}
    assert _ok(RuntimeError("boom")) is None


def test_dashboard_bootstrap_keeps_stats_when_uploads_fail():
    """Quota/stats must still return if the uploads list sub-call fails."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from services.shell_bootstrap import shell_bootstrap_payload

    user = {
        "id": "u1",
        "subscription_tier": "free",
        "role": "user",
        "wallet": {"put_balance": 0, "put_reserved": 0, "aic_balance": 0, "aic_reserved": 0},
    }
    stats = {
        "uploads": {"total": 1, "total_scope": "all_time"},
        "quota": {"uploads_used": 3, "uploads_limit": 100, "uploads_unlimited": False},
        "engagement": {"views": 0, "likes": 0, "comments": 0, "shares": 0, "scope": "all_time"},
        "accounts": {"connected": 0, "limit": 1},
        "scheduled": 0,
        "success_rate": 100,
        "recent": [],
        "wallet": {},
        "credits": {},
        "plan": {},
    }

    async def _run():
        with patch(
            "services.shell_bootstrap.dashboard_stats_for_user",
            new=AsyncMock(return_value=stats),
        ), patch(
            "services.shell_bootstrap.fetch_user_uploads_list",
            new=AsyncMock(side_effect=RuntimeError("uploads down")),
        ), patch(
            "services.shell_bootstrap._fetch_platforms_bundle",
            new=AsyncMock(return_value={"platforms": {}, "total_accounts": 0, "max_accounts": 1, "can_add_more": True}),
        ):
            return await shell_bootstrap_payload(
                MagicMock(),
                user,
                context="dashboard",
                upload_limit=50,
                upload_view=None,
                meta=False,
            )

    out = asyncio.run(_run())
    assert out["dashboard_stats"]["quota"]["uploads_used"] == 3
    assert out["uploads"] == []


def test_bootstrap_schedule_repair_timeout_returns_none():
    async def _wait_for_timeout(coro, *_a, **_k):
        if hasattr(coro, "close"):
            coro.close()
        raise asyncio.TimeoutError

    async def _run():
        with patch("services.shell_bootstrap.asyncio.wait_for", new=_wait_for_timeout):
            return await _bootstrap_schedule_repair(object(), "user-1")

    result = asyncio.run(_run())
    assert result is None
