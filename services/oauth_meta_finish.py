"""Finish Meta OAuth after destination pick — persist token + popup HTML."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi.responses import HTMLResponse

import core.state
from core.audit import log_system_event
from core.auth import encrypt_blob
from core.oauth import mirror_oauth_profile_image_to_r2
from services.meta_oauth import meta_oauth_mode
from services.platform_oauth_refresh import OAUTH_RECONNECT_RESET_SQL
from stages.entitlements import can_user_connect_platform

logger = logging.getLogger("uploadm8-api")


def oauth_popup_html(
    success: bool, platform: str, post_target: str, error_msg: str = None
) -> HTMLResponse:
    if success:
        payload = {"type": "oauth_success", "platform": str(platform or "")}
    else:
        safe_error = (error_msg or "Unknown error").replace('"', '\\"').replace("\n", " ")[:200]
        payload = {
            "type": "oauth_error",
            "platform": str(platform or ""),
            "error": safe_error,
        }
    payload_js = json.dumps(payload)
    target_js = json.dumps(post_target)
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html>
        <head><title>Connecting...</title></head>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #1a1a2e; color: white;">
            <div style="text-align: center;">
                <p>{" Connected successfully!" if success else " Connection failed"}</p>
                <p style="color: #888; font-size: 14px;">This window will close automatically...</p>
            </div>
            <script>
                (function () {{
                    var payload = {payload_js};
                    var target = {target_js};
                    try {{
                        payload.ts = Date.now();
                        localStorage.setItem('uploadm8_oauth_result', JSON.stringify(payload));
                        localStorage.removeItem('uploadm8_oauth_result');
                    }} catch (e) {{}}
                    if (window.opener) {{
                        try {{ window.opener.postMessage(payload, target); }} catch (e) {{}}
                        try {{ window.opener.postMessage(payload, '*'); }} catch (e) {{}}
                    }}
                    setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} }}, 1200);
                }})();
            </script>
        </body>
        </html>
        """
    )


async def persist_meta_destination(
    *,
    platform: str,
    user_id: str,
    account_id: str,
    account_name: str,
    account_username: str,
    account_avatar: str,
    access_token: str,
    user_access_token: str,
    meta_llt_ok: bool,
    meta_permissions: Any,
    facebook_user_asid: Optional[str],
    token_expires_in: Any,
    reconnect_account_id: Optional[str],
    reconnect_expected_provider_id: Optional[str],
    post_target: str,
) -> HTMLResponse:
    """Store a chosen Meta Page / IG account after OAuth or destination pick."""
    from core.platform_token_expiry import normalize_connect_expires_at, stamp_token_expiry

    if account_avatar and str(account_avatar).startswith("http"):
        try:
            mirrored_key = await mirror_oauth_profile_image_to_r2(
                str(user_id), platform, str(account_avatar)
            )
            if mirrored_key:
                account_avatar = mirrored_key
        except Exception as _av_e:
            logger.debug("OAuth avatar mirror skipped (%s): %s", platform, _av_e)

    if not account_id or not str(account_id).strip():
        return oauth_popup_html(False, platform, post_target, "Provider did not return account_id")

    _exp_fields = normalize_connect_expires_at(token_expires_in)
    blob_payload = {
        "access_token": access_token,
        "refresh_token": None,
        **_exp_fields,
    }
    if meta_llt_ok:
        blob_payload.update(stamp_token_expiry(blob_payload, non_expiring=True))
    else:
        blob_payload.pop("access_non_expiring", None)
    blob_payload["meta_oauth_mode"] = meta_oauth_mode()
    blob_payload["meta_permissions"] = meta_permissions or []
    if facebook_user_asid:
        blob_payload["facebook_user_id"] = str(facebook_user_asid)
    if user_access_token:
        blob_payload["meta_user_token"] = str(user_access_token)
    if token_expires_in:
        user_stamp = stamp_token_expiry({}, expires_in=token_expires_in)
        blob_payload["meta_user_expires_at"] = user_stamp.get("expires_at")
    if platform == "instagram":
        blob_payload["ig_user_id"] = str(account_id)
    if platform == "facebook":
        blob_payload["page_id"] = str(account_id)
    token_blob = encrypt_blob(blob_payload)

    async with core.state.db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM platform_tokens WHERE user_id = $1 AND platform = $2 AND account_id = $3",
            user_id,
            platform,
            account_id,
        )
        if existing:
            await conn.execute(
                f"""
                UPDATE platform_tokens SET token_blob = $1, account_name = $2, account_username = $3,
                account_avatar = $4, updated_at = NOW(), last_oauth_reconnect_at = NOW(),
                {OAUTH_RECONNECT_RESET_SQL}
                WHERE id = $5
                """,
                token_blob,
                account_name,
                account_username,
                account_avatar,
                existing["id"],
            )
            connect_action = "PLATFORM_RECONNECTED"
        elif reconnect_account_id:
            if reconnect_expected_provider_id and str(reconnect_expected_provider_id) != str(account_id):
                return oauth_popup_html(
                    False,
                    platform,
                    post_target,
                    "You authenticated a different account. Please sign in to the same account you selected for reconnect.",
                )
            await conn.execute(
                f"""
                UPDATE platform_tokens
                SET token_blob = $1, account_name = $2, account_username = $3,
                    account_avatar = $4, account_id = $5, updated_at = NOW(),
                    last_oauth_reconnect_at = NOW(), {OAUTH_RECONNECT_RESET_SQL}
                WHERE id = $6 AND user_id = $7 AND platform = $8
                """,
                token_blob,
                account_name,
                account_username,
                account_avatar,
                account_id,
                reconnect_account_id,
                user_id,
                platform,
            )
            connect_action = "PLATFORM_RECONNECTED"
        else:
            user_row = await conn.fetchrow(
                "SELECT id, role, subscription_tier FROM users WHERE id = $1",
                user_id,
            )
            current_count = int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
                    user_id,
                )
                or 0
            )
            current_for_platform = int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND platform = $2 AND revoked_at IS NULL",
                    user_id,
                    platform,
                )
                or 0
            )
            allowed, reason = can_user_connect_platform(
                dict(user_row or {}),
                current_total=current_count,
                current_for_platform=current_for_platform,
            )
            if not allowed:
                return oauth_popup_html(False, platform, post_target, reason)
            await conn.execute(
                """
                INSERT INTO platform_tokens
                (user_id, platform, account_id, account_name, account_username, account_avatar, token_blob, last_oauth_reconnect_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """,
                user_id,
                platform,
                account_id,
                account_name,
                account_username,
                account_avatar,
                token_blob,
            )
            connect_action = "PLATFORM_CONNECTED"

        await log_system_event(
            conn,
            user_id=str(user_id),
            action=connect_action,
            event_category="PLATFORM",
            resource_type="platform",
            resource_id=f"{platform}:{account_id}",
            details={
                "platform": platform,
                "account_name": account_name,
                "account_username": account_username,
            },
        )
    return oauth_popup_html(True, platform, post_target)


async def finish_meta_destination_pick(
    *,
    platform: str,
    pick_token: str,
    destination_id: str,
    parent_origin: Optional[str],
    frontend_url: str,
) -> HTMLResponse:
    from core.oauth import sanitize_oauth_parent_origin
    from services.meta_oauth_connect import pop_meta_pick_pending

    pending = await pop_meta_pick_pending(pick_token)
    post_target = (
        sanitize_oauth_parent_origin(parent_origin)
        if parent_origin
        else frontend_url.rstrip("/")
    )
    if not pending or pending.get("platform") != platform:
        return oauth_popup_html(
            False,
            platform,
            post_target,
            "Selection expired. Close this window and connect again.",
        )
    post_target = sanitize_oauth_parent_origin(pending.get("parent_origin") or parent_origin)

    want = str(destination_id or "").strip()
    chosen = None
    for d in pending.get("destinations") or []:
        if str(d.get("destination_id") or "") == want:
            chosen = d
            break
    if not chosen:
        return oauth_popup_html(False, platform, post_target, "Invalid destination selection.")

    account_name = chosen.get("name") or chosen.get("username") or (
        "Facebook Page" if platform == "facebook" else "Instagram Account"
    )
    account_username = (chosen.get("username") or "").strip() or account_name
    try:
        return await persist_meta_destination(
            platform=platform,
            user_id=str(pending["user_id"]),
            account_id=str(chosen["destination_id"]),
            account_name=account_name,
            account_username=account_username,
            account_avatar=str(chosen.get("avatar") or ""),
            access_token=str(chosen["access_token"]),
            user_access_token=str(pending.get("user_access_token") or ""),
            meta_llt_ok=bool(pending.get("meta_llt_ok")),
            meta_permissions=pending.get("meta_permissions") or [],
            facebook_user_asid=pending.get("facebook_user_asid"),
            token_expires_in=pending.get("token_expires_in"),
            reconnect_account_id=pending.get("reconnect_account_id"),
            reconnect_expected_provider_id=pending.get("reconnect_expected_provider_account_id"),
            post_target=post_target,
        )
    except Exception:
        logger.exception("oauth select-destination failed for %s", platform)
        return oauth_popup_html(
            False, platform, post_target, "Connection failed. Please try again."
        )
