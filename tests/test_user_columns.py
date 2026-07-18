"""Projected users SELECT helpers (UPLOADM8-30)."""

from core.user_columns import (
    USERS_AUTH_COLUMNS,
    USERS_ME_COLUMNS,
    USERS_WALLET_COLUMNS,
    USERS_WORKER_COLUMNS,
    users_select_sql,
)


def test_users_select_sql_no_star():
    sql = users_select_sql(USERS_AUTH_COLUMNS)
    assert "SELECT *" not in sql
    assert "FROM users WHERE id = $1" in sql
    assert "password_hash" not in USERS_AUTH_COLUMNS
    assert "preferences" not in USERS_AUTH_COLUMNS


def test_me_and_auth_share_required_billing_fields():
    for col in ("subscription_tier", "role", "flex_enabled", "email_verified", "status"):
        assert col in USERS_AUTH_COLUMNS
        assert col in USERS_ME_COLUMNS


def test_wallet_and_worker_are_slim():
    assert "password_hash" not in USERS_WALLET_COLUMNS
    assert "preferences" not in USERS_WORKER_COLUMNS
    assert "subscription_tier" in USERS_WORKER_COLUMNS
