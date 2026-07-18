"""Connected Accounts limit payload should follow role-aware entitlements."""

from datetime import datetime, timezone

from services.platform_accounts import serialize_platform_account
from stages.entitlements import entitlements_to_dict, get_entitlements_from_user


def test_master_admin_platforms_limit_is_effectively_unlimited():
    plan = entitlements_to_dict(
        get_entitlements_from_user(
            {"role": "master_admin", "subscription_tier": "free"}
        )
    )
    max_accounts = int(plan.get("max_accounts", 1) or 1)
    assert max_accounts >= 999
    # Same rule as GET /api/platforms can_add_more
    assert 4 < max_accounts


def test_free_tier_platforms_limit_is_finite():
    plan = entitlements_to_dict(
        get_entitlements_from_user({"role": "user", "subscription_tier": "free"})
    )
    max_accounts = int(plan.get("max_accounts", 1) or 1)
    assert max_accounts == 4
    assert not (4 < max_accounts)


def test_serialize_platform_account_includes_reconnect_fields():
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    row = {
        "id": "tok-1",
        "account_id": "ext-1",
        "account_name": "Demo",
        "account_username": "demo",
        "account_avatar": "https://example.com/a.png",
        "is_primary": True,
        "created_at": now,
        "last_oauth_reconnect_at": now,
        "last_used_at": now,
    }
    active = serialize_platform_account(row, auth_error_by_token={}, presign=False)
    assert active["status"] == "active"
    assert active["first_connected_at"]
    assert active["last_reconnected_at"]
    assert active["last_used_at"]
    assert active["avatar"] == "https://example.com/a.png"

    stale = serialize_platform_account(
        row,
        auth_error_by_token={"tok-1": "TOKEN_EXPIRED"},
        presign=False,
    )
    assert stale["status"] == "needs_reconnection"
