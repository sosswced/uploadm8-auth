"""
User profile / ``GET /api/me`` payload and profile PATCH helpers for routers/me.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException

from core.config import BILLING_MODE
from core.deps import _attach_workspace_context
from core.db_pool import acquire_db
from core.helpers import _safe_col
from core.r2 import resolve_user_profile_avatar_url
from core.sql_allowlist import assert_set_fragments_columns, USERS_UPDATE_COLUMNS_ME, USERS_UPDATE_COLUMNS_PROFILE
from core.user_columns import USERS_ME_COLUMNS, users_select_sql
from core.models import ProfileUpdate, ProfileUpdateSettings
from services.workspace import ensure_personal_workspace, workspace_capabilities
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
    avatar_signed_url = resolve_user_profile_avatar_url(avatar_r2_key, presign=True) or None

    raw_name = user.get("name")
    first = (user.get("first_name") or "").strip()
    last = (user.get("last_name") or "").strip()
    combined = f"{first} {last}".strip() if (first or last) else None
    email_prefix = (user.get("email") or "").split("@")[0] if user.get("email") else None
    display_name = raw_name or combined or email_prefix or "User"

    out = {
        "id": user["id"],
        "user_id": str(user["id"]),
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
        "entitlements": plan,
        "must_reset_password": bool(user.get("must_reset_password")),
    }
    trill = user.get("trill")
    if isinstance(trill, dict):
        out["trill"] = trill
    ws = user.get("workspace")
    if isinstance(ws, dict):
        out["workspace"] = ws
    out["workspace_capabilities"] = workspace_capabilities(user)
    return out


async def fetch_me_endpoint_data(pool, user_id: str) -> tuple[dict, list]:
    """
    Data for ``GET /api/me``: one pooled connection for users row (auth gates), wallet,
    and thumbnail personas — avoids a second checkout after ``get_current_user_readonly``
    queries (Sentry duplicate ``pg_advisory_unlock_all`` spans; see also UPLOADM8-11).

    Caller should pass a JWT-verified ``user_id`` (e.g. from ``get_verified_user_id``).
    """
    async with acquire_db(pool) as conn:
        user = await conn.fetchrow(users_select_sql(USERS_ME_COLUMNS), user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        if user["status"] == "banned":
            raise HTTPException(status_code=403, detail="Account suspended")
        if user.get("email_verified") is False:
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Please verify your email to use the app.",
                    "code": "email_not_verified",
                },
            )
        user_dict = dict(user)
        await ensure_personal_workspace(
            conn,
            user_id,
            user_dict.get("name") or user_dict.get("email") or "My Workspace",
        )
        user_dict = await _attach_workspace_context(conn, user_dict, None)
        personas = []
        try:
            from services.thumbnail_personas_list import list_thumbnail_studio_personas

            personas = await list_thumbnail_studio_personas(conn, user_id)
        except Exception:
            logger.debug("GET /api/me thumbnail_personas skipped", exc_info=True)
        try:
            from services.trill_access import user_trill_map_unlocked

            map_unlocked = await user_trill_map_unlocked(conn, user_id)
            prefs_row = await conn.fetchrow(
                """
                SELECT trill_enabled, default_vehicle_make_id, default_vehicle_model_id
                FROM user_preferences WHERE user_id = $1
                """,
                user_id,
            )
            pd = dict(prefs_row) if prefs_row else {}
            user_dict["trill"] = {
                "map_unlocked": bool(map_unlocked),
                "enabled": True if pd.get("trill_enabled") is None else bool(pd.get("trill_enabled")),
                "default_vehicle_make_id": pd.get("default_vehicle_make_id"),
                "default_vehicle_model_id": pd.get("default_vehicle_model_id"),
            }
        except Exception:
            logger.debug("GET /api/me trill summary skipped", exc_info=True)
            user_dict["trill"] = {"map_unlocked": False, "enabled": True}
        return user_dict, personas


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
