"""
UploadM8 "Me" routes — user profile, wallet, settings, preferences, account deletion.
Extracted from app.py.
"""

import json
import re
import uuid
import logging

import httpx
import stripe
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, BackgroundTasks

from core.config import (
    BILLING_MODE,
    R2_BUCKET_NAME,
    STRIPE_SECRET_KEY,
    STRIPE_SUCCESS_URL,
    STRIPE_CANCEL_URL,
    TIKTOK_CLIENT_KEY,
    TIKTOK_CLIENT_SECRET,
)
from core.state import db_pool
from core.deps import get_current_user
from core.auth import hash_password, verify_password, encrypt_blob, decrypt_blob
from core.wallet import get_wallet, credit_wallet, transfer_tokens
from core.r2 import (
    generate_presigned_download_url,
    r2_presign_get_url,
    get_s3_client,
    _normalize_r2_key,
    _delete_r2_objects,
)
from core.helpers import _now_utc, _safe_json, get_plan, _safe_col
from core.models import (
    ProfileUpdate,
    ProfileUpdateSettings,
    SettingsUpdate,
    PasswordChange,
    PreferencesUpdate,
    TransferRequest,
    CheckoutRequest,
)
from core.oauth import _revoke_platform_token
from stages.emails import send_account_deleted_email, send_password_changed_email
from stages.entitlements import (
    TOPUP_PRODUCTS,
    get_entitlements_from_user,
    entitlements_to_dict,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["me"])


# ============================================================
# User Profile & Wallet
# ============================================================
@router.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    raw_tier = user.get("subscription_tier", "free")
    ent = get_entitlements_from_user(dict(user))
    plan = entitlements_to_dict(ent)  # plan = entitlements (single source)
    wallet = user.get("wallet", {})
    role = user.get("role", "user")

    # Avatar: single source of truth = users.avatar_r2_key (private bucket -> signed URL)
    avatar_r2_key = user.get("avatar_r2_key")
    avatar_signed_url = None
    if avatar_r2_key:
        try:
            avatar_signed_url = generate_presigned_download_url(avatar_r2_key)
        except Exception as e:
            logger.warning(f"Failed to presign avatar for user {user.get('id')}: {e}")

    # Display name: prefer name, then first_name+last_name, then email prefix
    raw_name = user.get("name")
    first = (user.get("first_name") or "").strip()
    last = (user.get("last_name") or "").strip()
    combined = f"{first} {last}".strip() if (first or last) else None
    email_prefix = (user.get("email") or "").split("@")[0] if user.get("email") else None
    display_name = raw_name or combined or email_prefix or "User"

    # Stabilization window: return both snake_case + camelCase keys
    return {
        "id": user["id"],
        "email": user["email"],
        "name": display_name,
        "role": role,
        "timezone": user.get("timezone") or "America/Chicago",

        # Avatar outputs (private, signed)
        "avatar_r2_key": avatar_r2_key,
        "avatar_url": avatar_signed_url,
        "avatarUrl": avatar_signed_url,
        "avatar_signed_url": avatar_signed_url,
        "avatarSignedUrl": avatar_signed_url,

        "subscription_tier":      raw_tier,
        "tier":                  ent.tier,           # canonical slug (launch->creator_lite)
        "tier_display":          ent.tier_display,   # human-readable name
        "subscription_status":     user.get("subscription_status"),
        "current_period_end":      user.get("current_period_end").isoformat() if user.get("current_period_end") else None,
        "trial_end":               user.get("trial_end").isoformat() if user.get("trial_end") else None,
        "stripe_subscription_id":  user.get("stripe_subscription_id"),
        "billing_mode":            BILLING_MODE,   # "test" | "live"
        "wallet": {
            "put_balance":  float(wallet.get("put_balance", 0.0) or 0.0),
            "aic_balance":  float(wallet.get("aic_balance", 0.0) or 0.0),
            "put_reserved": float(wallet.get("put_reserved", 0.0) or 0.0),
            "aic_reserved": float(wallet.get("aic_reserved", 0.0) or 0.0),
            "updated_at":   wallet.get("updated_at"),
        },
        "plan": plan,
        "features": {
            "uploads":     plan.get("put_monthly", 0) > 0,
            "scheduler":   plan.get("can_schedule", False),
            "analytics":   bool(plan.get("analytics") and plan.get("analytics") != "basic"),
            "watermark":   plan.get("can_watermark", True),
            "white_label": plan.get("can_white_label", False),
            "support":     True,
        },
        # Full entitlements dict — used by hasEntitlement() in app.js
        "entitlements": entitlements_to_dict(
            get_entitlements_from_user(dict(user))
        ),
    }


@router.put("/api/me")
async def update_me(data: ProfileUpdate, user: dict = Depends(get_current_user)):
    """Update user profile"""
    _ME_COLS = frozenset({"name", "timezone"})
    updates, params = [], [user["id"]]
    if data.name:
        updates.append(f"{_safe_col('name', _ME_COLS)} = ${len(params)+1}")
        params.append(data.name)
    if data.timezone:
        updates.append(f"{_safe_col('timezone', _ME_COLS)} = ${len(params)+1}")
        params.append(data.timezone)
    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1", *params)
    return {"status": "updated"}


# ============================================================
# Settings Endpoints
# ============================================================
@router.put("/api/settings/profile")
async def update_profile_settings(data: ProfileUpdateSettings, user: dict = Depends(get_current_user)):
    """Update user profile (first name, last name)"""
    _PROFILE_COLS = frozenset({"first_name", "last_name", "name"})
    updates, params = [], [user["id"]]

    if data.first_name is not None:
        updates.append(f"{_safe_col('first_name', _PROFILE_COLS)} = ${len(params)+1}")
        params.append(data.first_name.strip())

    if data.last_name is not None:
        updates.append(f"{_safe_col('last_name', _PROFILE_COLS)} = ${len(params)+1}")
        params.append(data.last_name.strip())

    # Also update the combined name field for backwards compatibility
    if data.first_name is not None or data.last_name is not None:
        first = data.first_name.strip() if data.first_name else user.get("first_name", "")
        last = data.last_name.strip() if data.last_name else user.get("last_name", "")
        full_name = f"{first} {last}".strip() or user.get("name", "User")
        updates.append(f"{_safe_col('name', _PROFILE_COLS)} = ${len(params)+1}")
        params.append(full_name)

    if updates:
        async with db_pool.acquire() as conn:
            await conn.execute(
                f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = $1",
                *params
            )
        logger.info(f"Profile updated for user {user['id']}")
        return {"status": "success", "message": "Profile updated successfully"}

    return {"status": "success", "message": "No changes made"}

@router.put("/api/settings/preferences/legacy")
async def update_preferences_legacy(data: PreferencesUpdate, user: dict = Depends(get_current_user)):
    """Update user preferences (notifications, theme, hashtags, etc.)"""
    async with db_pool.acquire() as conn:
        # Ensure user_settings row exists
        await conn.execute(
            "INSERT INTO user_settings (user_id, preferences_json) VALUES ($1, '{}') ON CONFLICT (user_id) DO NOTHING",
            user["id"]
        )

        # Get current preferences
        current_prefs = await conn.fetchval(
            "SELECT preferences_json FROM user_settings WHERE user_id = $1",
            user["id"]
        )

        # Parse current preferences
        prefs = current_prefs if current_prefs else {}
        if isinstance(prefs, str):
            prefs = json.loads(prefs)

        # Update with new values (only update fields that are provided)
        if data.emailNotifs is not None:
            prefs["emailNotifs"] = data.emailNotifs
        if data.uploadCompleteNotifs is not None:
            prefs["uploadCompleteNotifs"] = data.uploadCompleteNotifs
        if data.marketingEmails is not None:
            prefs["marketingEmails"] = data.marketingEmails
        if data.theme is not None:
            prefs["theme"] = data.theme
        if data.accentColor is not None:
            prefs["accentColor"] = data.accentColor
        if data.defaultPrivacy is not None:
            prefs["defaultPrivacy"] = data.defaultPrivacy
        if data.autoPublish is not None:
            prefs["autoPublish"] = data.autoPublish
        if data.alwaysHashtags is not None:
            prefs["alwaysHashtags"] = data.alwaysHashtags
        if data.blockedHashtags is not None:
            prefs["blockedHashtags"] = data.blockedHashtags
        if data.tiktokHashtags is not None:
            prefs["tiktokHashtags"] = data.tiktokHashtags
        if data.youtubeHashtags is not None:
            prefs["youtubeHashtags"] = data.youtubeHashtags
        if data.instagramHashtags is not None:
            prefs["instagramHashtags"] = data.instagramHashtags
        if data.facebookHashtags is not None:
            prefs["facebookHashtags"] = data.facebookHashtags
        if data.hashtagPosition is not None:
            prefs["hashtagPosition"] = data.hashtagPosition
        if data.maxHashtags is not None:
            prefs["maxHashtags"] = data.maxHashtags
        if data.aiHashtagsEnabled is not None:
            prefs["aiHashtagsEnabled"] = data.aiHashtagsEnabled
        if data.aiHashtagCount is not None:
            prefs["aiHashtagCount"] = data.aiHashtagCount
        if data.aiHashtagStyle is not None:
            prefs["aiHashtagStyle"] = data.aiHashtagStyle
        if data.captionStyle is not None:
            prefs["captionStyle"] = data.captionStyle
        if data.captionTone is not None:
            prefs["captionTone"] = data.captionTone
        if data.captionVoice is not None:
            prefs["captionVoice"] = data.captionVoice
        if data.platformHashtags is not None:
            prefs["platformHashtags"] = data.platformHashtags

        # Save back to database
        await conn.execute(
            "UPDATE user_settings SET preferences_json = $1, updated_at = NOW() WHERE user_id = $2",
            json.dumps(prefs),
            user["id"]
        )

    logger.info(f"Preferences updated for user {user['id']}")
    return {"status": "success", "message": "Preferences saved successfully", "preferences": prefs}


@router.put("/api/settings/password")
async def update_password_settings(data: PasswordChange, user: dict = Depends(get_current_user)):
    """Change user password (settings endpoint version)"""
    async with db_pool.acquire() as conn:
        # Verify current password
        user_row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")

        # Update to new password
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, user["id"])

        # Optionally invalidate other sessions (refresh tokens)
        await conn.execute("DELETE FROM refresh_tokens WHERE user_id = $1", user["id"])

    logger.info(f"Password changed via settings for user {user['id']}")
    return {"status": "success", "message": "Password changed successfully"}

@router.post("/api/settings/avatar")
async def upload_avatar(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Must be JPEG, PNG, GIF, or WebP")

        # Read file content
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")

        # Create unique filename
        ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "png"
        r2_key = f"avatars/{user['id']}/{uuid.uuid4()}.{ext}"

        # Upload to private R2 bucket
        s3 = get_s3_client()
        if not R2_BUCKET_NAME:
            raise HTTPException(status_code=500, detail="Missing R2_BUCKET_NAME")
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=r2_key,
            Body=content,
            ContentType=file.content_type,
        )

        # Store single source of truth in DB
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET avatar_r2_key = $1, updated_at = NOW() WHERE id = $2",
                r2_key,
                user["id"],
            )

        signed_url = r2_presign_get_url(r2_key)

        logger.info(f"Avatar uploaded for user {user['id']}: {r2_key}")
        return {"success": True, "r2_key": r2_key, "avatar_url": signed_url, "avatarUrl": signed_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Avatar upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar")


# ──────────────────────────────────────────────────────────────────────────────
# Account deletion helper (TOS: paid users keep access until period end)
# ──────────────────────────────────────────────────────────────────────────────

async def _execute_account_deletion(conn, user: dict, ip_addr: str = None, initiated_by: str = "self") -> dict:
    """
    Performs full account deletion: revoke platform tokens, delete DB rows, purge R2.
    Called from DELETE /api/me (immediate) or customer.subscription.deleted (deferred).
    """
    user_id = str(user["id"])
    r2_rows = await conn.fetch(
        "SELECT r2_key, telemetry_r2_key, processed_r2_key, thumbnail_r2_key FROM uploads WHERE user_id = $1",
        user["id"],
    )
    r2_keys = []
    for row in r2_rows:
        for col in ("r2_key", "telemetry_r2_key", "processed_r2_key", "thumbnail_r2_key"):
            v = row.get(col) if col in row.keys() else None
            if v:
                r2_keys.append(v)
    avatar_key = user.get("avatar_r2_key") or ""
    if avatar_key:
        r2_keys.append(avatar_key)

    token_rows = await conn.fetch(
        "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE user_id = $1",
        user["id"],
    )
    tokens_revoked = 0
    for trow in token_rows:
        ok, err = await _revoke_platform_token(trow["platform"], trow["token_blob"])
        if ok:
            tokens_revoked += 1
        await conn.execute(
            """
            INSERT INTO platform_disconnect_log
                (user_id, platform, account_id, account_name,
                 revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            user_id,
            trow["platform"],
            trow["account_id"],
            trow["account_name"],
            ok,
            err or None,
            initiated_by,
            ip_addr,
        )

    _DELETION_TABLES = frozenset({
        "uploads", "platform_tokens", "token_ledger", "wallets",
        "user_settings", "user_preferences", "refresh_tokens",
        "user_color_preferences", "account_groups", "white_label_settings",
    })
    rows_deleted = {}
    for tbl in _DELETION_TABLES:
        try:
            n = await conn.fetchval(f"SELECT COUNT(*) FROM {_safe_col(tbl, _DELETION_TABLES)} WHERE user_id = $1", user["id"])
            rows_deleted[tbl] = int(n)
        except Exception:
            pass
    rows_deleted["users"] = 1

    import asyncio as _aio
    _aio.ensure_future(send_account_deleted_email(
        user.get("email", ""),
        user.get("name") or "there",
    ))

    await conn.execute("DELETE FROM refresh_tokens          WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM token_ledger             WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM wallets                  WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM user_settings            WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM user_preferences         WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM platform_tokens          WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM user_color_preferences   WHERE user_id = $1", user["id"])
    await conn.execute("DELETE FROM account_groups           WHERE user_id = $1", user["id"])
    try:
        await conn.execute("DELETE FROM white_label_settings WHERE user_id = $1", user["id"])
    except Exception:
        pass
    await conn.execute("DELETE FROM uploads                  WHERE user_id = $1", user["id"])
    try:
        await conn.execute(
            "UPDATE support_messages SET name = '[deleted]', email = '[deleted]' WHERE user_id = $1",
            user["id"],
        )
    except Exception:
        pass
    await conn.execute("DELETE FROM users WHERE id = $1", user["id"])

    r2_deleted = await _delete_r2_objects(r2_keys)
    return {"r2_deleted": r2_deleted, "tokens_revoked": tokens_revoked, "rows_deleted": rows_deleted}


# Self-serve account deletion  DELETE /api/me
# ──────────────────────────────────────────────────────────────────────────────

@router.delete("/api/me")
async def delete_account(request: Request, user: dict = Depends(get_current_user)):
    """
    Self-serve account deletion. TOS-aligned:
      - Free users: deletion is immediate.
      - Paid users: subscription cancelled (no future charges), access until period end,
        then full deletion when Stripe sends subscription.deleted.
    """
    user_id = str(user["id"])
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)

    if user.get("role") == "master_admin":
        raise HTTPException(403, "Master admin accounts cannot be deleted via this endpoint.")

    async with db_pool.acquire() as conn:
        deletion_log_id = await conn.fetchval(
            """
            INSERT INTO account_deletion_log
                (user_id, user_email, user_name, initiated_by, ip_address)
            VALUES ($1, $2, $3, 'self', $4)
            RETURNING id
            """,
            user_id,
            user.get("email", ""),
            user.get("name", ""),
            ip_addr,
        )

        stripe_sub_id = user.get("stripe_subscription_id")
        has_active_paid_sub = bool(stripe_sub_id and STRIPE_SECRET_KEY)

        if has_active_paid_sub:
            try:
                sub = stripe.Subscription.retrieve(stripe_sub_id)
                if sub.status in ("active", "trialing"):
                    has_active_paid_sub = True
                else:
                    has_active_paid_sub = False
            except Exception:
                has_active_paid_sub = False

        if has_active_paid_sub:
            # TOS: paid users keep access until end of billing period
            await conn.execute(
                "UPDATE users SET deletion_requested_at = NOW() WHERE id = $1",
                user["id"],
            )
            try:
                stripe.Subscription.cancel(stripe_sub_id)
            except Exception as e:
                logger.warning(f"Stripe cancel failed for {user_id}: {e}")

            period_end = user.get("current_period_end")
            access_until = period_end.strftime("%B %d, %Y") if period_end and hasattr(period_end, "strftime") else "end of billing period"

            logger.info(f"[DELETION SCHEDULED] user={user_id} access_until={access_until}")
            return {
                "status": "deletion_scheduled",
                "message": "Your account will be deleted at the end of your billing period. You retain access until then.",
                "access_until": access_until,
            }
        else:
            # Free user or no active subscription: delete immediately
            result = await _execute_account_deletion(conn, user, ip_addr=ip_addr, initiated_by="account_deletion")
            await conn.execute(
                """
                UPDATE account_deletion_log
                SET completed_at = NOW(), r2_keys_deleted = $2, tokens_revoked = $3,
                    stripe_cancelled = $4, rows_deleted = $5
                WHERE id = $1
                """,
                deletion_log_id,
                result["r2_deleted"],
                result["tokens_revoked"],
                False,
                json.dumps(result["rows_deleted"]),
            )
            logger.info(f"[DELETION COMPLETE] user={user_id} r2={result['r2_deleted']} tokens={result['tokens_revoked']}")
            return {
                "status": "account_deleted",
                "summary": {
                    "r2_objects_deleted": result["r2_deleted"],
                    "platform_tokens_revoked": result["tokens_revoked"],
                    "rows_deleted": result["rows_deleted"],
                },
            }

# ============================================================
# Wallet
# ============================================================
@router.get("/api/wallet")
async def get_wallet_endpoint(user: dict = Depends(get_current_user)):
    wallet = user.get("wallet", {})
    plan = get_plan(user.get("subscription_tier", "free"))
    async with db_pool.acquire() as conn:
        ledger = await conn.fetch("SELECT * FROM token_ledger WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50", user["id"])
    return {
        "wallet": wallet, "plan_limits": {"put_daily": plan.get("put_daily", 1), "put_monthly": plan.get("put_monthly", 30), "aic_monthly": plan.get("aic_monthly", 0)},
        "ledger": [dict(l) for l in ledger],
    }

@router.post("/api/wallet/topup")
async def wallet_topup(data: CheckoutRequest, user: dict = Depends(get_current_user)):
    product = TOPUP_PRODUCTS.get(data.lookup_key)
    if not product: raise HTTPException(400, "Invalid product")

    async with db_pool.acquire() as conn:
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(email=user["email"], name=user["name"])
            customer_id = customer.id
            await conn.execute("UPDATE users SET stripe_customer_id = $1 WHERE id = $2", customer_id, user["id"])

    prices = stripe.Price.list(lookup_keys=[data.lookup_key], active=True)
    if not prices.data: raise HTTPException(400, "Price not found")

    session = stripe.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": prices.data[0].id, "quantity": 1}],
        mode="payment",
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
        metadata={"user_id": str(user["id"]), "wallet": product["wallet"], "amount": product["amount"]},
    )
    return {"checkout_url": session.url}

@router.post("/api/wallet/transfer")
async def wallet_transfer(data: TransferRequest, user: dict = Depends(get_current_user)):
    if not user.get("flex_enabled"):
        raise HTTPException(403, "Flex add-on required for transfers")
    async with db_pool.acquire() as conn:
        success = await transfer_tokens(conn, user["id"], data.from_platform, data.to_platform, data.amount)
    if not success: raise HTTPException(400, "Transfer failed - insufficient balance")
    return {"status": "transferred", "amount": data.amount, "burn": int(data.amount * 0.02)}

# ============================================================
# Settings
# ============================================================
_SETTINGS_DEFAULTS = {
    "discord_webhook": None,
    "telemetry_enabled": True,
    "hud_enabled": True,
    "hud_position": "bottom-left",
    "speeding_mph": 80,
    "euphoria_mph": 100,
    "hud_speed_unit": "mph",
    "hud_color": "#FFFFFF",
    "hud_font_family": "Arial",
    "hud_font_size": 24,
    "ffmpeg_screenshot_interval": 5,
    "auto_generate_thumbnails": True,
    "auto_generate_captions": True,
    "auto_generate_hashtags": True,
    "default_hashtag_count": 5,
    "always_use_hashtags": False,
}

@router.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    """Get user settings including Trill preferences"""
    async with db_pool.acquire() as conn:
        try:
            settings = await conn.fetchrow("""
                SELECT
                    discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                    speeding_mph, euphoria_mph, hud_speed_unit, hud_color,
                    hud_font_family, hud_font_size, ffmpeg_screenshot_interval,
                    auto_generate_thumbnails, auto_generate_captions,
                    auto_generate_hashtags, default_hashtag_count, always_use_hashtags
                FROM user_settings
                WHERE user_id = $1
            """, user["id"])
        except Exception as e:
            # Fallback if extended columns not yet migrated (e.g. pre-707)
            logger.warning(f"Full settings SELECT failed ({e}), using base columns")
            settings = await conn.fetchrow("""
                SELECT discord_webhook, telemetry_enabled, hud_enabled, hud_position,
                    speeding_mph, euphoria_mph, hud_speed_unit, hud_color
                FROM user_settings WHERE user_id = $1
            """, user["id"])
            if settings:
                settings = dict(settings)
                for k, v in _SETTINGS_DEFAULTS.items():
                    settings.setdefault(k, v)
                return settings
            return dict(_SETTINGS_DEFAULTS)

        if not settings:
            return dict(_SETTINGS_DEFAULTS)
        result = dict(settings)
        for k, v in _SETTINGS_DEFAULTS.items():
            result.setdefault(k, v)
        return result

# Base columns that exist in user_settings from migration 5 (before 707)
_SETTINGS_BASE_FIELDS = [
    "discord_webhook", "telemetry_enabled", "hud_enabled",
    "hud_position", "speeding_mph", "euphoria_mph",
    "hud_speed_unit", "hud_color",
]
# Extended columns added in migration 707
_SETTINGS_EXTENDED_FIELDS = [
    "hud_font_family", "hud_font_size", "ffmpeg_screenshot_interval",
    "auto_generate_thumbnails", "auto_generate_captions", "auto_generate_hashtags",
    "default_hashtag_count", "always_use_hashtags",
]

@router.put("/api/settings")
async def update_settings(data: SettingsUpdate, user: dict = Depends(get_current_user)):
    """Update user settings including Trill thresholds"""
    all_fields = _SETTINGS_BASE_FIELDS + _SETTINGS_EXTENDED_FIELDS
    updates, params = [], [user["id"]]

    _ALLOWED_SETTINGS = frozenset(_SETTINGS_BASE_FIELDS + _SETTINGS_EXTENDED_FIELDS)
    for field in all_fields:
        val = getattr(data, field, None)
        if val is not None:
            updates.append(f"{_safe_col(field, _ALLOWED_SETTINGS)} = ${len(params)+1}")
            params.append(val)

    if not updates:
        return {"status": "updated"}

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO user_settings (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
        """, user["id"])

        try:
            await conn.execute(
                f"UPDATE user_settings SET {', '.join(updates)}, updated_at = NOW() WHERE user_id = $1",
                *params
            )
        except Exception as e:
            # Fallback: update only base columns if extended columns not yet migrated
            base_updates, base_params = [], [user["id"]]
            _BASE_ALLOWED = frozenset(_SETTINGS_BASE_FIELDS)
            for field in _SETTINGS_BASE_FIELDS:
                val = getattr(data, field, None)
                if val is not None:
                    base_updates.append(f"{_safe_col(field, _BASE_ALLOWED)} = ${len(base_params)+1}")
                    base_params.append(val)
            if base_updates:
                await conn.execute(
                    f"UPDATE user_settings SET {', '.join(base_updates)}, updated_at = NOW() WHERE user_id = $1",
                    *base_params
                )
                logger.warning(f"Settings update fell back to base columns after: {e}")
            else:
                raise

    logger.info(f"Updated settings for user {user['id']}: {updates}")
    return {"status": "updated"}


@router.post("/api/settings/test-discord-webhook")
async def test_user_discord_webhook(data: dict, user: dict = Depends(get_current_user)):
    """Send a test message to the user's Discord webhook (for Settings page Test Webhook button).
    Accepts webhookUrl or webhook_url in body. If empty, uses the user's saved webhook from settings.
    The same saved URL is used when admin sends via 'Send to user webhooks'."""
    webhook_url = (data.get("webhookUrl") or data.get("webhook_url") or "").strip()
    if not webhook_url:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(
                  NULLIF(TRIM(us.discord_webhook), ''),
                  NULLIF(TRIM(up.discord_webhook), ''),
                  NULLIF(TRIM(COALESCE(u.preferences->>'discordWebhook', u.preferences->>'discord_webhook')), '')
                ) AS url
                FROM users u
                LEFT JOIN user_settings us ON us.user_id = u.id
                LEFT JOIN user_preferences up ON up.user_id = u.id
                WHERE u.id = $1
                """,
                user["id"],
            )
            webhook_url = (row["url"] or "").strip() if row else ""
    if not webhook_url:
        raise HTTPException(400, "Webhook URL required. Save your webhook in Settings first, or pass webhookUrl in the request.")
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        raise HTTPException(400, "Invalid Discord webhook URL")
    test_embed = {
        "title": "\U0001f514 UploadM8 Webhook Test",
        "description": "If you see this message, your webhook is configured correctly!",
        "color": 0x22c55e,
        "fields": [
            {"name": "Status", "value": "\u2705 Connected", "inline": True},
            {"name": "Tested By", "value": user.get("email", "User"), "inline": True},
        ],
        "footer": {"text": "UploadM8 Notifications"},
        "timestamp": _now_utc().isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook_url, json={"embeds": [test_embed]})
            if r.status_code not in (200, 204):
                raise HTTPException(400, f"Discord returned status {r.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(504, "Webhook request timed out")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Failed to reach Discord: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Failed to send: {str(e)}")
    return {"status": "sent"}


# ============================================================
# Me Preferences
# ============================================================
@router.get("/api/me/preferences")
async def get_preferences(user: dict = Depends(get_current_user)):
    """Get user preferences including hashtag settings"""
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
    if prefs and prefs["preferences"]:
        return json.loads(prefs["preferences"]) if isinstance(prefs["preferences"], str) else prefs["preferences"]
    return {}

@router.put("/api/me/preferences")
async def update_preferences(request: Request, user: dict = Depends(get_current_user)):
    """Update user preferences including hashtag settings"""
    prefs = await request.json()

    # Validate and sanitize hashtag data (string-safe; avoids list('#tag') char-splitting)
    def _split_tags(v):
        if v is None:
            return []
        # Accept JSON string list
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                maybe = json.loads(s)
                if isinstance(maybe, list):
                    v = maybe
                else:
                    v = s
            except Exception:
                v = s
        if isinstance(v, str):
            parts = re.split(r"[\s,]+", v.strip())
            return [p for p in parts if p]
        if isinstance(v, (list, tuple, set)):
            out = []
            for x in v:
                sx = str(x).strip()
                if sx:
                    out.append(sx)
            return out
        sx = str(v).strip()
        return [sx] if sx else []

    def _clean_tag_list(v, limit):
        out = []
        for t in _split_tags(v)[:limit]:
            t = str(t).strip().lower().lstrip("#")[:50]
            if t:
                out.append(t)
        return out

    # Support both camelCase (frontend) and snake_case (backend/worker) keys
    if "alwaysHashtags" in prefs or "always_hashtags" in prefs:
        v = prefs.get("alwaysHashtags", None)
        if v is None:
            v = prefs.get("always_hashtags", None)
        clean = _clean_tag_list(v, 100)
        prefs["alwaysHashtags"] = [f"#{t}" for t in clean]  # UI-friendly
        prefs["always_hashtags"] = clean                   # worker-friendly (no '#')

    if "blockedHashtags" in prefs or "blocked_hashtags" in prefs:
        v = prefs.get("blockedHashtags", None)
        if v is None:
            v = prefs.get("blocked_hashtags", None)
        clean = _clean_tag_list(v, 100)
        prefs["blockedHashtags"] = [f"#{t}" for t in clean]
        prefs["blocked_hashtags"] = clean

    if "platformHashtags" in prefs or "platform_hashtags" in prefs:
        ph = prefs.get("platformHashtags", None)
        if ph is None:
            ph = prefs.get("platform_hashtags", None)

        if isinstance(ph, str):
            try:
                ph = json.loads(ph)
            except Exception:
                ph = {}
        if not isinstance(ph, dict):
            ph = {}

        cleaned_ui = {}
        cleaned_worker = {}
        for platform in ["tiktok", "youtube", "instagram", "facebook"]:
            raw = ph.get(platform) or ph.get(platform.title()) or ph.get(platform.upper())
            clean = _clean_tag_list(raw, 50)
            cleaned_ui[platform] = [f"#{t}" for t in clean]
            cleaned_worker[platform] = clean

        prefs["platformHashtags"] = cleaned_ui
        prefs["platform_hashtags"] = cleaned_worker
    # Validate numeric hashtag settings
    if "maxHashtags" in prefs:
        prefs["maxHashtags"] = max(1, min(50, int(prefs["maxHashtags"])))
    if "aiHashtagCount" in prefs:
        prefs["aiHashtagCount"] = max(1, min(30, int(prefs["aiHashtagCount"])))

    # Validate hashtag position
    if "hashtagPosition" in prefs and prefs["hashtagPosition"] not in ["start", "end", "caption", "comment"]:
        prefs["hashtagPosition"] = "end"

    # Validate AI hashtag style (must match caption_stage + UI schema)
    if "aiHashtagStyle" in prefs and prefs["aiHashtagStyle"] not in ["lowercase", "capitalized", "camelcase", "mixed"]:
        prefs["aiHashtagStyle"] = "mixed"
    if "ai_hashtag_style" in prefs and prefs["ai_hashtag_style"] not in ["lowercase", "capitalized", "camelcase", "mixed"]:
        prefs["ai_hashtag_style"] = "mixed"

    # Caption & AI Settings — style / tone / voice (worker caption_stage reads these)
    _CAPTION_STYLES = ("story", "punchy", "factual")
    _CAPTION_TONES = ("hype", "calm", "cinematic", "authentic")
    _CAPTION_VOICES = (
        "default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator",
    )
    if "captionStyle" in prefs or "caption_style" in prefs:
        v = str(prefs.get("captionStyle") or prefs.get("caption_style") or "story").strip().lower()
        if v not in _CAPTION_STYLES:
            v = "story"
        prefs["captionStyle"] = prefs["caption_style"] = v
    if "captionTone" in prefs or "caption_tone" in prefs:
        v = str(prefs.get("captionTone") or prefs.get("caption_tone") or "authentic").strip().lower()
        if v not in _CAPTION_TONES:
            v = "authentic"
        prefs["captionTone"] = prefs["caption_tone"] = v
    if "captionVoice" in prefs or "caption_voice" in prefs:
        v = str(prefs.get("captionVoice") or prefs.get("caption_voice") or "default").strip().lower()
        if v not in _CAPTION_VOICES:
            v = "default"
        prefs["captionVoice"] = prefs["caption_voice"] = v

    async with db_pool.acquire() as conn:
        # MERGE with existing preferences — never replace entirely (frontend may send partial updates)
        existing_row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
        existing = {}
        if existing_row and existing_row.get("preferences"):
            raw = existing_row["preferences"]
            if isinstance(raw, str):
                try:
                    existing = json.loads(raw) if raw else {}
                except Exception:
                    existing = {}
            elif isinstance(raw, dict):
                existing = dict(raw)
        merged = {**existing, **prefs}
        await conn.execute(
            "UPDATE users SET preferences = $1, updated_at = NOW() WHERE id = $2",
            json.dumps(merged), user["id"]
        )
        # Sync discord_webhook to user_settings and user_preferences so:
        # - Admin "Send to user webhooks" finds it (announcement query reads from these tables)
        # - Worker load_user_settings can use either source
        discord_webhook = (prefs.get("discordWebhook") or prefs.get("discord_webhook") or "").strip() or None
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, discord_webhook) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET discord_webhook = $2, updated_at = NOW()
            """,
            user["id"],
            discord_webhook,
        )
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            user["id"],
        )
        await conn.execute(
            "UPDATE user_preferences SET discord_webhook = $1, updated_at = NOW() WHERE user_id = $2",
            discord_webhook,
            user["id"],
        )
    return {"status": "updated"}
