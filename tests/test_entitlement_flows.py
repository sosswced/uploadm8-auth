"""
Tier / entitlement matrix — single source is stages/entitlements.py + worker usage.

Covers: paid tiers, trialing (same entitlements as tier), friends_family, lifetime,
master_admin via role, queue routing, connect/queue guards, Stripe lookup keys.
"""

from __future__ import annotations

import pytest

from stages.entitlements import (
    ENTITLEMENT_KEYS,
    NORMAL_QUEUE_CLASSES,
    PRIORITY_QUEUE_CLASSES,
    STRIPE_LOOKUP_TO_TIER,
    TIER_CONFIG,
    TIER_SLUGS,
    TOPUP_PRODUCTS,
    check_queue_depth,
    can_user_connect_platform,
    entitlements_to_dict,
    get_entitlements_for_tier,
    get_entitlements_from_user,
    get_next_public_upgrade_tier,
    get_tiers_for_api,
    normalize_tier,
)


def _user(tier: str, **extra) -> dict:
    row = {"subscription_tier": tier, "role": "user"}
    row.update(extra)
    return row


@pytest.mark.parametrize("slug", list(TIER_CONFIG.keys()))
def test_every_configured_tier_builds_entitlements(slug: str) -> None:
    ent = get_entitlements_for_tier(slug)
    assert ent.tier == normalize_tier(slug)
    assert ent.priority_class in PRIORITY_QUEUE_CLASSES | NORMAL_QUEUE_CLASSES
    if slug in ("friends_family", "lifetime", "master_admin"):
        assert ent.is_internal is True
    else:
        assert ent.is_internal is False


def test_entitlements_to_dict_matches_schema() -> None:
    ent = get_entitlements_for_tier("creator_pro")
    d = entitlements_to_dict(ent)
    for k in ENTITLEMENT_KEYS:
        assert k in d, f"missing API key {k}"


def test_normalize_unknown_tier_falls_back_to_free() -> None:
    assert normalize_tier("nope_not_a_tier") == "free"
    assert get_entitlements_for_tier("nope_not_a_tier").tier == "free"


def test_public_upgrade_ladder() -> None:
    assert get_next_public_upgrade_tier("free") == "creator_lite"
    assert get_next_public_upgrade_tier("agency") is None
    assert get_next_public_upgrade_tier("friends_family") is None


def test_stripe_lookup_keys_resolve_to_configured_tiers() -> None:
    for _lookup, tier in STRIPE_LOOKUP_TO_TIER.items():
        assert tier in TIER_CONFIG, f"{_lookup} -> {tier}"


def test_topup_products_have_wallet_and_amount() -> None:
    for key, meta in TOPUP_PRODUCTS.items():
        assert meta.get("wallet") in ("put", "aic"), key
        assert int(meta.get("amount", 0)) > 0, key


@pytest.mark.parametrize(
    "tier,expect_priority_lane",
    [
        ("free", False),
        ("creator_lite", False),
        ("creator_pro", True),
        ("studio", True),
        ("agency", True),
        ("friends_family", True),
        ("lifetime", True),
        ("master_admin", True),
    ],
)
def test_can_priority_matches_queue_routing_intent(tier: str, expect_priority_lane: bool) -> None:
    """can_priority mirrors priority_class in p0–p2 vs p3–p4 (worker uses entitlements.priority_class)."""
    ent = get_entitlements_for_tier(tier)
    assert ent.can_priority == expect_priority_lane
    pc = ent.priority_class
    if expect_priority_lane:
        assert pc in PRIORITY_QUEUE_CLASSES
    else:
        assert pc in NORMAL_QUEUE_CLASSES


def test_trialing_user_gets_tier_entitlements_not_free() -> None:
    """Trialing is a billing state; feature caps follow subscription_tier (see get_entitlements_from_user)."""
    u = _user(
        "creator_pro",
        subscription_status="trialing",
    )
    ent = get_entitlements_from_user(u)
    assert ent.tier == "creator_pro"
    assert ent.max_parallel_uploads == get_entitlements_for_tier("creator_pro").max_parallel_uploads


def test_active_paid_same_as_trial_tier() -> None:
    u_trial = _user("studio", subscription_status="trialing")
    u_active = _user("studio", subscription_status="active")
    assert entitlements_to_dict(get_entitlements_from_user(u_trial)) == entitlements_to_dict(
        get_entitlements_from_user(u_active)
    )


def test_friends_and_family_tier_matches_internal_high_caps() -> None:
    ent = get_entitlements_from_user(_user("friends_family"))
    assert ent.tier == "friends_family"
    assert ent.priority_class == "p0"
    assert ent.can_white_label is True
    assert ent.queue_depth >= 9999


def test_lifetime_tier() -> None:
    ent = get_entitlements_from_user(_user("lifetime"))
    assert ent.tier == "lifetime"
    assert ent.priority_class == "p0"


def test_master_admin_role_forces_master_entitlements() -> None:
    """Only role=master_admin maps to master_admin tier; role=admin keeps subscription quotas."""
    ent_admin = get_entitlements_from_user(
        {"subscription_tier": "free", "role": "admin"},
    )
    assert ent_admin.tier == "free"
    ent2 = get_entitlements_from_user(
        {"subscription_tier": "creator_lite", "role": "master_admin"},
    )
    assert ent2.tier == "master_admin"


def test_wallet_bypass_matches_staff_and_internal_tiers() -> None:
    from stages.entitlements import wallet_bypass_for_user_record

    assert wallet_bypass_for_user_record({"subscription_tier": "free", "role": "admin"}) is True
    assert wallet_bypass_for_user_record({"subscription_tier": "free", "role": "master_admin"}) is True
    assert wallet_bypass_for_user_record({"subscription_tier": "friends_family", "role": "user"}) is True
    assert wallet_bypass_for_user_record({"subscription_tier": "lifetime", "role": "user"}) is True
    assert wallet_bypass_for_user_record({"subscription_tier": "master_admin", "role": "user"}) is True
    assert wallet_bypass_for_user_record({"subscription_tier": "free", "role": "user"}) is False


def test_flex_enabled_on_user_row() -> None:
    ent = get_entitlements_from_user(_user("studio", flex_enabled=True))
    assert ent.can_flex is True


def test_entitlement_overrides_apply() -> None:
    overrides = {"can_ai": False, "max_parallel_uploads": 1}
    ent = get_entitlements_from_user(_user("creator_pro"), overrides=overrides)
    assert ent.can_ai is False
    assert ent.max_parallel_uploads == 1


def test_connect_platform_guard_at_limits() -> None:
    ent = get_entitlements_for_tier("free")
    ok, reason = can_user_connect_platform(_user("free"), current_total=ent.max_accounts)
    assert ok is False
    assert "limit" in reason.lower()

    ok2, _ = can_user_connect_platform(_user("free"), current_total=0, current_for_platform=ent.max_accounts_per_platform)
    assert ok2 is False


def test_queue_depth_guard() -> None:
    ent = get_entitlements_for_tier("free")
    ok, reason = check_queue_depth(_user("free"), current_pending=ent.queue_depth)
    assert ok is False
    assert "queue" in reason.lower()

    ok2, _ = check_queue_depth(_user("free"), current_pending=ent.queue_depth - 1)
    assert ok2 is True


def test_get_tiers_for_api_covers_all_slugs() -> None:
    rows = get_tiers_for_api()
    slugs = {r["slug"] for r in rows}
    for s in TIER_SLUGS:
        assert s in slugs, f"missing tier in /api payload: {s}"


def test_kpi_tool_costs_are_non_negative() -> None:
    from stages.kpi_collector import TOOL_COST_ESTIMATES_USD_PER_UPLOAD

    for _tool, usd in TOOL_COST_ESTIMATES_USD_PER_UPLOAD.items():
        assert float(usd) >= 0.0, _tool
