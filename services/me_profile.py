"""
User profile / ``GET /api/me`` payload and profile PATCH helpers for routers/me.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException

from core.config import BILLING_MODE
from core.helpers import _safe_col
from core.r2 import generate_presigned_download_url
from core.sql_allowlist import assert_set_fragments_columns, USERS_UPDATE_COLUMNS_ME, USERS_UPDATE_COLUMNS_PROFILE
from core.models import ProfileUpdate, ProfileUpdateSettings
from stages.entitlements import get_entitlements_from_user, entitlements_to_dict

logger = logging.getLogger("uploadm8-api")


def build_me_response(user: dict) -> dict:
    """Shape the authenticated user dict for ``GET /api/me``."""
    raw_tier = user.get("subscription_tier", "free")
    ent = get_entitlements_from_user(dict(user))
    plan = entitlements_to_dict(ent)
    wallet = user.get("wallet", {})
    role = user.get("role", "user")

    avatar_r2_key = user.get("avatar_r2_key")
    avatar_signed_url = None
    if avatar_r2_key:
        try:
            avatar_signed_url = generate_presigned_download_url(avatar_r2_key)
        except Exception as e:
            logger.warning(f"Failed to presign avatar for user {user.get('id')}: {e}")

    raw_name = user.get("name")
    first = (user.get("first_name") or "").strip()
    last = (user.get("last_name") or "").strip()
    combined = f"{first} {last}".strip() if (first or last) else None
    email_prefix = (user.get("email") or "").split("@")[0] if user.get("email") else None
    display_name = raw_name or combined or email_prefix or "User"

    return {
        "id": user["id"],
        "email": user["email"],
        "name": display_name,
        "role": role,
        "timezone": user.get("timezone") or "America/Chicago",
        "avatar_r2_key": avatar_r2_key,
        "avatar_url": avatar_signed_url,
        "avatarUrl": avatar_signed_url,
        "avatar_signed_url": avatar_signed_url,
        "avatarSignedUrl": avatar_signed_url,
        "subscription_tier": raw_tier,
        "tier": ent.tier,
        "tier_display": ent.tier_display,
        "subscription_status": user.get("subscription_status"),
        "current_period_end": user.get("current_period_end").isoformat()
        if user.get("current_period_end")
        else None,
        "trial_end": user.get("trial_end").isoformat() if user.get("trial_end") else None,
        "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
        "stripe_subscription_id": user.get("stripe_subscription_id"),
        "billing_mode": BILLING_MODE,
        "wallet": {
            "put_balance": float(wallet.get("put_balance", 0.0) or 0.0),
            "aic_balance": float(wallet.get("aic_balance", 0.0) or 0.0),
            "put_reserved": float(wallet.get("put_reserved", 0.0) or 0.0),
            "aic_reserved": float(wallet.get("aic_reserved", 0.0) or 0.0),
            "updated_at": wallet.get("updated_at"),
        },
        "plan": plan,
        "features": {
            "uploads": plan.get("put_monthly", 0) > 0,
            "scheduler": plan.get("can_schedule", False),
            "analytics": bool(plan.get("analytics") and plan.get("analytics") != "basic"),
            "watermark": plan.get("can_watermark", True),
            "white_label": plan.get("can_white_label", False),
            "support": True,
        },
        "entitlements": entitlements_to_dict(get_entitlements_from_user(dict(user))),
    }


async def apply_me_profile_update(conn, user_id: str, data: ProfileUpdate) -> bool:
    """
    Apply ``PUT /api/me`` column updates. Returns True if any row was updated.
    """
    updates: List[str] = []
    params: List[Any] = [user_id]
    if data.name:
        updates.append(f"{_safe_col('name', USERS_UPDATE_COLUMNS_ME)} = ${len(params) + 1}")
        params.append(data.name)
    if data.timezone:
        updates.append(f"{_safe_col('timezone', USERS_UPDATE_COLUMNS_ME)} = ${len(params) + 1}")
        params.append(data.timezone)
    if not updates:
        return False
    assert_set_fragments_columns(updates, USERS_UPDATE_COLUMNS_ME)
    await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
    return True


async def apply_settings_profile_update(
    conn, user_id: str, data: ProfileUpdateSettings, user: dict
) -> Tuple[bool, str]:
    """
    Apply ``PUT /api/settings/profile``. Returns (did_update, message).
    """
    updates: List[str] = []
    params: List[Any] = [user_id]

    if data.first_name is not None:
        updates.append(f"{_safe_col('first_name', USERS_UPDATE_COLUMNS_PROFILE)} = ${len(params) + 1}")
        params.append(data.first_name.strip())

    if data.last_name is not None:
        updates.append(f"{_safe_col('last_name', USERS_UPDATE_COLUMNS_PROFILE)} = ${len(params) + 1}")
        params.append(data.last_name.strip())

    if data.first_name is not None or data.last_name is not None:
        first = data.first_name.strip() if data.first_name else user.get("first_name", "")
        last = data.last_name.strip() if data.last_name else user.get("last_name", "")
        full_name = f"{first} {last}".strip() or user.get("name", "User")
        updates.append(f"{_safe_col('name', USERS_UPDATE_COLUMNS_PROFILE)} = ${len(params) + 1}")
        params.append(full_name)

    if not updates:
        return False, "No changes made"

    assert_set_fragments_columns(updates, USERS_UPDATE_COLUMNS_PROFILE)
    await conn.execute(
        f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1",
        *params,
    )
    logger.info(f"Profile updated for user {user_id}")
    return True, "Profile updated successfully"
