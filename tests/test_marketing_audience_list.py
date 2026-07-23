"""list_campaign_audience_users returns matched rows for CSV / handoff."""

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4


def test_list_campaign_audience_users_includes_matches():
    from services.marketing_execution import list_campaign_audience_users

    uid = str(uuid4())
    urow = {
        "id": uid,
        "email": "a@example.com",
        "name": "Ada",
        "subscription_tier": "free",
        "role": "user",
    }

    async def _run():
        conn = AsyncMock()
        with patch(
            "services.marketing_execution._iter_candidate_users",
            new=AsyncMock(return_value=[urow]),
        ), patch(
            "services.marketing_execution._user_campaign_features",
            new=AsyncMock(
                return_value={
                    "uploads_window": 5,
                    "enterprise_fit_score": 40.0,
                    "nudge_ctr_pct": 12.5,
                    "revenue_7d": 0,
                }
            ),
        ), patch(
            "services.marketing_execution._match_fail_reason",
            return_value=None,
        ):
            out = await list_campaign_audience_users(
                conn,
                targeting={"tiers": [], "min_uploads_30d": 0},
                range_key="30d",
                result_limit=50,
                dedupe_within_days=0,
            )
        return out

    out = asyncio.run(_run())
    assert out["matched_count"] == 1
    assert out["returned"] == 1
    assert out["users"][0]["email"] == "a@example.com"
    assert out["users"][0]["user_id"] == uid
    assert out["users"][0]["send_ready"] is True
    assert out["users"][0]["uploads"] == 5
