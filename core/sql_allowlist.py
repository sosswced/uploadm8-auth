"""Guardrails for dynamic SQL fragments built from Python (not user SQL)."""
from __future__ import annotations

from typing import Iterable

# Account deletion: COUNT(*) before DELETE — must match tables we delete from.
ACCOUNT_DELETION_COUNT_TABLES: frozenset[str] = frozenset(
    {
        "uploads",
        "platform_tokens",
        "token_ledger",
        "wallets",
        "user_settings",
        "user_preferences",
        "refresh_tokens",
        "user_color_preferences",
        "account_groups",
        "white_label_settings",
    }
)

# UPDATE users SET fragments like "name = $2" — left-hand column must be allowlisted.
USERS_UPDATE_COLUMNS_ME: frozenset[str] = frozenset({"name", "timezone"})
USERS_UPDATE_COLUMNS_PROFILE: frozenset[str] = frozenset(
    {"first_name", "last_name", "avatar_r2_key", "name"}
)
USERS_UPDATE_COLUMNS_ADMIN: frozenset[str] = frozenset(
    {"subscription_tier", "role", "status", "flex_enabled"}
)

ALLOWED_WALLET_BALANCE_COLUMNS: frozenset[str] = frozenset({"put_balance", "aic_balance"})

# uploads: PATCH metadata + complete-upload body
UPLOADS_METADATA_PATCH_COLUMNS: frozenset[str] = frozenset(
    {"title", "caption", "hashtags", "scheduled_time", "schedule_metadata", "schedule_mode", "updated_at"}
)
UPLOADS_COMPLETE_BODY_COLUMNS: frozenset[str] = frozenset({"title", "caption", "hashtags", "target_accounts"})

ACCOUNT_GROUPS_UPDATE_COLUMNS: frozenset[str] = frozenset(
    {"name", "description", "color", "account_ids", "updated_at"}
)

USER_COLOR_PREFERENCES_UPDATE_COLUMNS: frozenset[str] = frozenset(
    {"tiktok_color", "youtube_color", "instagram_color", "facebook_color", "accent_color", "updated_at"}
)

# Worker: save_generated_metadata — AI fields written back to uploads
UPLOADS_AI_GENERATED_METADATA_COLUMNS: frozenset[str] = frozenset(
    {"ai_generated_title", "ai_generated_caption", "ai_generated_hashtags"}
)

# OAuth token refresh persistence (stages/db.save_refreshed_token)
OAUTH_TOKEN_STORAGE_TABLES_ORDERED: tuple[str, ...] = ("platform_tokens", "connected_accounts")
OAUTH_TOKEN_STORAGE_TABLES: frozenset[str] = frozenset(OAUTH_TOKEN_STORAGE_TABLES_ORDERED)


def assert_account_deletion_table(table: str) -> str:
    if table not in ACCOUNT_DELETION_COUNT_TABLES:
        raise ValueError(f"disallowed table for account deletion count: {table!r}")
    return table


def assert_user_update_set_clauses(set_clauses: Iterable[str], allowed: frozenset[str]) -> None:
    assert_set_fragments_columns(set_clauses, allowed)


def assert_set_fragments_columns(set_clauses: Iterable[str], allowed: frozenset[str]) -> None:
    """Ensure each `col = ...` fragment uses only allowlisted column names."""
    for frag in set_clauses:
        if "=" not in frag:
            continue
        col = frag.split("=", 1)[0].strip()
        if col not in allowed:
            raise ValueError(f"disallowed column in dynamic UPDATE: {col!r}")


def assert_wallet_balance_column(col: str) -> str:
    if col not in ALLOWED_WALLET_BALANCE_COLUMNS:
        raise ValueError(f"disallowed wallet column: {col!r}")
    return col


def assert_relation_name(name: str, allowed: frozenset[str]) -> str:
    """Validate a SQL table/relation identifier against an explicit allowlist."""
    if name not in allowed:
        raise ValueError(f"disallowed SQL relation name: {name!r}")
    return name
