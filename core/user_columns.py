"""Projected ``users`` SELECT lists for hot auth / me / worker paths (UPLOADM8-30).

Avoid ``SELECT *`` on the PK lookup — the row is wide (``preferences`` JSONB,
``password_hash``, Stripe fields) and is fetched many times per page load.
"""

from __future__ import annotations

# Auth gates + entitlements + workspace billing attach (deps / require_verified_user_on_conn).
USERS_AUTH_COLUMNS = (
    "id, email, name, role, status, email_verified, timezone, "
    "subscription_tier, subscription_status, flex_enabled, "
    "stripe_customer_id, stripe_subscription_id, current_period_end, trial_end, "
    "active_workspace_id, must_reset_password, created_at, avatar_r2_key, "
    "first_name, last_name"
)

# Same as auth — ``GET /api/me`` needs display + billing fields for ``build_me_response``.
USERS_ME_COLUMNS = USERS_AUTH_COLUMNS

# Wallet page only needs auth gates + tier for plan limits.
USERS_WALLET_COLUMNS = (
    "id, email, name, role, status, email_verified, "
    "subscription_tier, flex_enabled"
)

# Workspace owner billing profile (``apply_owner_billing_profile``).
USERS_OWNER_BILLING_COLUMNS = (
    "id, subscription_tier, flex_enabled, subscription_status, "
    "stripe_customer_id, stripe_subscription_id, current_period_end, trial_end, "
    "role, status, email_verified, name, email"
)

# Member row in workspace context (identity + auth gates).
USERS_MEMBER_COLUMNS = (
    "id, email, name, role, status, email_verified, "
    "subscription_tier, flex_enabled, active_workspace_id"
)

# Worker pipeline entitlements (preferences come from ``load_user_settings``).
USERS_WORKER_COLUMNS = (
    "id, email, name, role, status, email_verified, timezone, "
    "subscription_tier, flex_enabled"
)


def users_select_sql(columns: str) -> str:
    return f"SELECT {columns} FROM users WHERE id = $1"
