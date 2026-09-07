"""OAuth routes (/api/oauth/*)."""
from __future__ import annotations

import json
import logging
import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from fastapi.responses import HTMLResponse

import core.state
from core.audit import log_system_event
from core.auth import encrypt_blob
from core.config import (
    OAUTH_CONFIG,
    FRONTEND_URL,
    TIKTOK_CLIENT_KEY,
    TIKTOK_CLIENT_SECRET,
    YOUTUBE_CLIENT_ID,
    YOUTUBE_CLIENT_SECRET,
    INSTAGRAM_CLIENT_ID,
    INSTAGRAM_CLIENT_SECRET,
    FACEBOOK_CLIENT_ID,
    FACEBOOK_CLIENT_SECRET,
)
from core.deps import get_current_user_readonly_no_wallet
from core.oauth import (
    get_oauth_redirect_uri,
    mirror_oauth_profile_image_to_r2,
    sanitize_oauth_parent_origin,
    tiktok_pkce_verifier_and_challenge,
    _pop_oauth_state,
    _store_oauth_state,
)
from core.time_utils import now_utc as _now_utc
from services.meta_oauth import (
    FACEBOOK_GRAPH_PERMISSION_PROOF,
    INSTAGRAM_GRAPH_PERMISSION_PROOF,
    META_GRAPH_API_VERSION,
    META_GRAPH_RATE_LIMITING,
    META_PERMISSION_ALLOWED_USAGE,
    exchange_long_lived_user_token,
    facebook_pages_manage_posts_proof,
    fetch_granted_permissions,
    meta_facebook_oauth_scope,
    meta_instagram_oauth_scope,
    meta_oauth_auth_type,
    meta_oauth_mode,
)
from services.platform_oauth_refresh import OAUTH_RECONNECT_RESET_SQL
from services.workspace import require_can_manage_platforms, resolve_billing_user_id
from stages.entitlements import can_user_connect_platform

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["oauth"])


@router.get("/api/oauth/{platform}/start")
async def oauth_start(
    platform: str,
    parent_origin: Optional[str] = Query(None),
    force_login: bool = Query(False, description="Force account chooser/reauth where provider supports it"),
    reconnect_account_id: Optional[str] = Query(None, description="Existing platform_tokens.id to reconnect"),
    user: dict = Depends(get_current_user_readonly_no_wallet),
):
    """Start OAuth flow for a platform.

    Provider URLs are built so the user always sees an explicit auth/consent step
    (not a silent bind to whoever is already signed in on that browser). TikTok:
    ``disable_auto_auth=1``; Google: ``prompt`` includes ``login``; Meta:
    ``auth_type=reauthenticate,rerequest`` and a per-request ``auth_nonce``.
    """
    if platform not in OAUTH_CONFIG:
        raise HTTPException(400, f"Unsupported platform: {platform}")

    require_can_manage_platforms(user)
    platform_user_id = resolve_billing_user_id(user)
    actor_user_id = str(user["id"])

    # Account limits are enforced in the OAuth callback on *new* rows only, so users at the limit
    # can still reconnect (UPDATE) existing platform identities.

    config = OAUTH_CONFIG[platform]
    state = secrets.token_urlsafe(32)

    reconnect_target = None
    if reconnect_account_id:
        async with core.state.db_pool.acquire() as conn:
            reconnect_target = await conn.fetchrow(
                """
                SELECT id, account_id
                FROM platform_tokens
                WHERE id = $1
                  AND user_id = $2
                  AND platform = $3
                  AND revoked_at IS NULL
                """,
                reconnect_account_id,
                platform_user_id,
                platform,
            )
        if not reconnect_target:
            raise HTTPException(404, "Reconnect target account not found")

    tiktok_code_verifier: str | None = None
    if platform == "tiktok":
        tiktok_code_verifier, tiktok_code_challenge = tiktok_pkce_verifier_and_challenge()

    # Store state with workspace platform owner + acting member
    state_payload: dict = {
        "user_id": platform_user_id,
        "actor_user_id": actor_user_id,
        "platform": platform,
        "created_at": _now_utc().isoformat(),
        "parent_origin": sanitize_oauth_parent_origin(parent_origin),
        "reconnect_account_id": str(reconnect_target["id"]) if reconnect_target else None,
        "reconnect_expected_provider_account_id": str(reconnect_target["account_id"]) if reconnect_target else None,
    }
    if tiktok_code_verifier:
        state_payload["tiktok_code_verifier"] = tiktok_code_verifier
    await _store_oauth_state(state, state_payload)

    redirect_uri = get_oauth_redirect_uri(platform)

    if platform == "tiktok":
        # TikTok Login Kit: disable_auto_auth=0 skips the auth/consent UI when the browser
        # already has a valid TikTok session — every OAuth then returns the same account
        # (breaks multi-account / "add another" flows). Always show the auth page.
        # See https://developers.tiktok.com/doc/login-kit-web (disable_auto_auth).
        params = {
            "client_key": TIKTOK_CLIENT_KEY,
            "scope": config["scope"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": tiktok_code_challenge,
            "code_challenge_method": "S256",
            "disable_auto_auth": 1,
        }
        if force_login:
            params["prompt"] = "login"
    elif platform == "youtube":
        # Google OAuth: without `login`, a single Google session can still authorize silently.
        # `select_account` + `consent` + `login` shows the picker and requires sign-in again
        # so "Add another channel" is not stuck on the default browser account.
        params = {
            "client_id": YOUTUBE_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": config["scope"],
            "state": state,
            "access_type": "offline",
            "prompt": "select_account consent login",
        }
    elif platform == "instagram":
        # Meta Login: `reauthenticate` forces a fresh Facebook login (multi-account).
        # `rerequest` re-prompts declined *and* newly approved permissions (e.g. pages_manage_posts
        # after App Review). `auth_nonce` avoids a cached silent dialog on repeat popups.
        params = {
            "client_id": INSTAGRAM_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": meta_instagram_oauth_scope(),
            "response_type": "code",
            "state": state,
            "auth_type": meta_oauth_auth_type(),
            "auth_nonce": secrets.token_hex(12),
        }
    elif platform == "facebook":
        params = {
            "client_id": FACEBOOK_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": meta_facebook_oauth_scope(),
            "response_type": "code",
            "state": state,
            "auth_type": meta_oauth_auth_type(),
            "auth_nonce": secrets.token_hex(12),
        }
    
    auth_url = f"{config['auth_url']}?{urlencode(params)}"
    return {"auth_url": auth_url, "state": state}


@router.get("/api/oauth/meta/config")
async def meta_oauth_config():
    """
    Which Meta OAuth scope bundle the server requests (for App Review demos vs production).
    Does not expose app secrets. Public read — platforms UI only displays this for master admins.
    """
    return {
        "meta_oauth_mode": meta_oauth_mode(),
        "instagram_scope": meta_instagram_oauth_scope(),
        "facebook_scope": meta_facebook_oauth_scope(),
        "facebook_permission_proof": FACEBOOK_GRAPH_PERMISSION_PROOF,
        "instagram_permission_proof": INSTAGRAM_GRAPH_PERMISSION_PROOF,
        "pages_manage_posts": facebook_pages_manage_posts_proof(),
        "approved_usage": META_PERMISSION_ALLOWED_USAGE,
        "rate_limiting": META_GRAPH_RATE_LIMITING,
        "notes": (
            "META_OAUTH_MODE=full is production (Meta-approved publish + insights). "
            "minimal requests only pages_show_list, pages_read_engagement, business_management "
            "for a restricted reviewer demo. Facebook Page video uses pages_manage_posts "
            "(POST /{page-id}/videos), not publish_video."
        ),
    }


@router.get("/api/oauth/{platform}/callback")
async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):
    """Handle OAuth callback - returns HTML that communicates with parent window"""
    post_target = FRONTEND_URL.rstrip("/")
    state_data = None
    if state:
        state_data = await _pop_oauth_state(state)
        if state_data:
            post_target = sanitize_oauth_parent_origin(state_data.get("parent_origin"))

    def popup_response(success: bool, platform: str, error_msg: str = None):
        """Generate HTML that posts message to parent window and closes popup"""
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

        return HTMLResponse(f"""
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
                        // Fallback when parent_origin drifted (www vs apex) — parent validates message type.
                        try {{ window.opener.postMessage(payload, '*'); }} catch (e) {{}}
                    }}
                    setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} }}, 1200);
                }})();
            </script>
        </body>
        </html>
        """)

    if error:
        return popup_response(False, platform, error)

    if not state_data:
        return popup_response(False, platform, "Invalid or expired session. Please try again.")

    user_id = state_data["user_id"]
    reconnect_row_id = state_data.get("reconnect_account_id")
    existing_reconnect_profile = None
    if reconnect_row_id:
        try:
            async with core.state.db_pool.acquire() as conn:
                existing_reconnect_profile = await conn.fetchrow(
                    """
                    SELECT account_id, account_name, account_username, account_avatar
                    FROM platform_tokens
                    WHERE id = $1
                      AND user_id = $2
                      AND platform = $3
                    """,
                    reconnect_row_id,
                    user_id,
                    platform,
                )
        except Exception:
            existing_reconnect_profile = None

    if not code:
        return popup_response(False, platform, "No authorization code received")
    
    config = OAUTH_CONFIG[platform]
    redirect_uri = get_oauth_redirect_uri(platform)
    
    try:
        # follow_redirects: TikTok docs use curl -L for user/info; redirects would otherwise yield non-JSON bodies.
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # Exchange code for tokens based on platform
            token_payload = {}
            meta_llt_ok = False
            if platform == "tiktok":
                from services.tiktok_api import (
                    fetch_tiktok_user_profile_for_oauth,
                    tiktok_parse_oauth_token_response,
                )

                token_body = {
                    "client_key": TIKTOK_CLIENT_KEY,
                    "client_secret": TIKTOK_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                }
                cv = (state_data or {}).get("tiktok_code_verifier")
                if cv:
                    token_body["code_verifier"] = cv
                token_response = await client.post(config["token_url"], data=token_body)
                try:
                    token_data = token_response.json() if token_response.content else {}
                except Exception:
                    token_data = {}
                if token_response.status_code >= 400:
                    hint = (
                        token_data.get("error_description")
                        or token_data.get("description")
                        or token_data.get("error")
                        or (token_response.text or "")[:300]
                    )
                    raise Exception(f"TikTok token exchange failed ({token_response.status_code}): {hint}")

                access_token, token_open_id, _, token_payload = tiktok_parse_oauth_token_response(token_data)
                if not access_token:
                    raise Exception(
                        "TikTok did not return an access_token. Check client credentials, "
                        "redirect URI (must match portal, including OAUTH_PUBLIC_BASE_URL if used), and PKCE."
                    )

                account_id = token_open_id or secrets.token_hex(8)
                account_name = "TikTok User"
                account_username = ""
                account_avatar = ""

                prof = await fetch_tiktok_user_profile_for_oauth(client, access_token)
                if prof.get("account_id"):
                    account_id = str(prof["account_id"]).strip() or account_id
                if prof.get("account_name"):
                    account_name = str(prof["account_name"]).strip() or account_name
                if prof.get("account_username"):
                    account_username = str(prof["account_username"]).strip()
                if prof.get("account_avatar"):
                    account_avatar = str(prof["account_avatar"]).strip()

                # Reconnect fallback: keep prior profile fields if provider didn't return them.
                if reconnect_row_id and existing_reconnect_profile:
                    if not account_name or account_name == "TikTok User":
                        prev_nm = (existing_reconnect_profile.get("account_name") or "").strip()
                        if prev_nm and prev_nm != "TikTok User":
                            account_name = prev_nm
                    if (not account_username or not str(account_username).strip()) and existing_reconnect_profile.get(
                        "account_username"
                    ):
                        account_username = existing_reconnect_profile.get("account_username")
                    if (not account_avatar or not str(account_avatar).strip()) and existing_reconnect_profile.get(
                        "account_avatar"
                    ):
                        account_avatar = existing_reconnect_profile.get("account_avatar")
                if account_name == "TikTok User":
                    if account_username:
                        account_name = account_username
                    elif account_id:
                        aid = str(account_id)
                        account_name = f"TikTok {aid[-6:]}" if len(aid) > 6 else f"TikTok {aid}"

            elif platform == "youtube":
                token_response = await client.post(config["token_url"], data={
                    "client_id": YOUTUBE_CLIENT_ID,
                    "client_secret": YOUTUBE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                access_token = token_data.get("access_token")
                
                # Get channel info
                user_response = await client.get(
                    "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                channels = user_response.json().get("items", [])
                if channels:
                    channel = channels[0]
                    snippet = channel.get("snippet", {})
                    account_id = channel.get("id")
                    account_name = snippet.get("title", "YouTube Channel")
                    # customUrl can be empty for channels without custom URL; use channel title as fallback
                    account_username = (snippet.get("customUrl") or "").strip() or account_name
                    account_avatar = snippet.get("thumbnails", {}).get("default", {}).get("url", "")
                else:
                    account_id = secrets.token_hex(8)
                    account_name = "YouTube Channel"
                    account_username = ""
                    account_avatar = ""
                    
            elif platform == "instagram":
                # Instagram Graph API: authenticate via Facebook, get Instagram Business Account
                from services.meta_oauth_connect import list_instagram_destinations

                token_response = await client.get(config["token_url"], params={
                    "client_id": INSTAGRAM_CLIENT_ID,
                    "client_secret": INSTAGRAM_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                user_access_token = token_data.get("access_token")
                
                if not user_access_token:
                    raise Exception(f"No access token: {token_data}")

                user_access_token, meta_user_expires_in = await exchange_long_lived_user_token(
                    client,
                    user_access_token,
                    app_id=INSTAGRAM_CLIENT_ID,
                    app_secret=INSTAGRAM_CLIENT_SECRET,
                )
                if meta_user_expires_in:
                    token_data["expires_in"] = meta_user_expires_in
                    meta_llt_ok = True

                meta_perms = await fetch_granted_permissions(client, user_access_token)

                # App-scoped Facebook user id (ASID) — required to honor Meta data-deletion callbacks.
                facebook_user_asid = None
                try:
                    me_resp = await client.get(
                        f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/me",
                        params={"fields": "id", "access_token": user_access_token},
                    )
                    if me_resp.status_code == 200:
                        facebook_user_asid = str((me_resp.json() or {}).get("id") or "").strip() or None
                except Exception as _asid_e:
                    logger.debug("Instagram OAuth ASID fetch skipped: %s", _asid_e)

                destinations = await list_instagram_destinations(client, user_access_token)
                if not destinations:
                    raise Exception(
                        "No Instagram Business/Creator account found on your Facebook Pages. "
                        "Connect Instagram to a Page in Meta Business Suite, then try again."
                    )

                from core.config import BASE_URL as _API_BASE
                from services.meta_oauth_connect import resolve_or_pick_destination

                reconnect_expected = state_data.get("reconnect_expected_provider_account_id")
                chosen, pick_html = await resolve_or_pick_destination(
                    platform="instagram",
                    destinations=destinations,
                    expected_provider_id=reconnect_expected,
                    pending_fields={
                        "user_id": user_id,
                        "token_expires_in": token_data.get("expires_in"),
                        "meta_llt_ok": meta_llt_ok,
                        "meta_permissions": meta_perms,
                        "facebook_user_asid": facebook_user_asid,
                        "user_access_token": user_access_token,
                        "reconnect_account_id": state_data.get("reconnect_account_id"),
                    },
                    post_target=post_target,
                    api_base=_API_BASE,
                )
                if pick_html is not None:
                    return pick_html
                if not chosen:
                    raise Exception("Could not resolve an Instagram account to connect.")

                account_id = chosen["destination_id"]
                account_name = chosen.get("name") or chosen.get("username") or "Instagram Account"
                account_username = (chosen.get("username") or "").strip() or account_name
                account_avatar = chosen.get("avatar") or ""
                access_token = chosen["access_token"]
                
            elif platform == "facebook":
                from services.meta_oauth_connect import (
                    list_facebook_page_destinations,
                    resolve_or_pick_destination,
                )

                token_response = await client.get(config["token_url"], params={
                    "client_id": FACEBOOK_CLIENT_ID,
                    "client_secret": FACEBOOK_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                user_access_token = token_data.get("access_token")

                if not user_access_token:
                    raise Exception(f"No access token returned: {token_data}")

                user_access_token, meta_user_expires_in = await exchange_long_lived_user_token(
                    client,
                    user_access_token,
                    app_id=FACEBOOK_CLIENT_ID,
                    app_secret=FACEBOOK_CLIENT_SECRET,
                )
                if meta_user_expires_in:
                    token_data["expires_in"] = meta_user_expires_in
                    meta_llt_ok = True

                meta_perms_fb = await fetch_granted_permissions(client, user_access_token)

                facebook_user_asid = None
                try:
                    me_resp = await client.get(
                        f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/me",
                        params={"fields": "id", "access_token": user_access_token},
                    )
                    if me_resp.status_code == 200:
                        facebook_user_asid = str((me_resp.json() or {}).get("id") or "").strip() or None
                except Exception as _asid_e:
                    logger.debug("Facebook OAuth ASID fetch skipped: %s", _asid_e)

                destinations = await list_facebook_page_destinations(client, user_access_token)
                if not destinations:
                    raise Exception(
                        "No Facebook Pages found. You need a Facebook Page to publish Reels. "
                        "Create a Page at facebook.com/pages/create and try again."
                    )

                from core.config import BASE_URL as _API_BASE

                reconnect_expected = state_data.get("reconnect_expected_provider_account_id")
                chosen, pick_html = await resolve_or_pick_destination(
                    platform="facebook",
                    destinations=destinations,
                    expected_provider_id=reconnect_expected,
                    pending_fields={
                        "user_id": user_id,
                        "token_expires_in": token_data.get("expires_in"),
                        "meta_llt_ok": meta_llt_ok,
                        "meta_permissions": meta_perms_fb,
                        "facebook_user_asid": facebook_user_asid,
                        "user_access_token": user_access_token,
                        "reconnect_account_id": state_data.get("reconnect_account_id"),
                    },
                    post_target=post_target,
                    api_base=_API_BASE,
                )
                if pick_html is not None:
                    return pick_html
                if not chosen:
                    raise Exception("Could not resolve a Facebook Page to connect.")

                account_id = chosen["destination_id"]
                account_name = chosen.get("name") or "Facebook Page"
                account_username = (chosen.get("username") or "").strip() or account_name
                account_avatar = chosen.get("avatar") or ""
                access_token = chosen["access_token"]

            # Copy profile image into R2 — FB/IG/TikTok CDN URLs often 403 when hotlinked from the browser.
            _avatar_before_mirror = str(account_avatar)[:120] if account_avatar else ""
            if account_avatar and str(account_avatar).startswith("http") and platform in (
                "facebook",
                "instagram",
                "tiktok",
                "youtube",
            ):
                try:
                    mirrored_key = await mirror_oauth_profile_image_to_r2(
                        str(user_id), platform, str(account_avatar)
                    )
                    if mirrored_key:
                        account_avatar = mirrored_key
                except Exception as _av_e:
                    logger.debug(f"OAuth avatar mirror skipped ({platform}): {_av_e}")
            
            # Refuse to persist "ghost" connections with no identity
            if not account_id or (isinstance(account_id, str) and account_id.strip() == ""):
                raise Exception("Provider did not return account_id; refusing to store token (prevents phantom accounts).")

            # Store the token — include platform-specific IDs in the blob so
            # publish_stage can read them without needing a separate DB lookup.
            from core.platform_token_expiry import normalize_connect_expires_at, stamp_token_expiry

            _raw_exp = token_data.get("expires_in") or token_payload.get("expires_in")
            _exp_fields = normalize_connect_expires_at(_raw_exp)
            blob_payload = {
                "access_token": access_token,
                "refresh_token": token_data.get("refresh_token") or token_payload.get("refresh_token"),
                **_exp_fields,
            }
            # Page tokens derived from a long-lived user token do not expire.
            # If LLT exchange failed, the page token is short-lived (~1–2h) —
            # never stamp access_non_expiring or keepalive will skip refresh.
            if platform in ("instagram", "facebook"):
                if meta_llt_ok:
                    blob_payload.update(
                        stamp_token_expiry(blob_payload, non_expiring=True)
                    )
                else:
                    logger.warning(
                        "Meta LLT exchange failed for %s — storing short-lived page token expiry",
                        platform,
                    )
                    blob_payload.pop("access_non_expiring", None)
                blob_payload["meta_oauth_mode"] = meta_oauth_mode()
                if platform == "instagram":
                    blob_payload["meta_permissions"] = meta_perms
                elif platform == "facebook":
                    blob_payload["meta_permissions"] = meta_perms_fb
                # Persist ASID so Meta data-deletion / deauthorize callbacks can find rows.
                _asid = locals().get("facebook_user_asid")
                if _asid:
                    blob_payload["facebook_user_id"] = str(_asid)
                # Keep user LLT for fb_exchange_token (page tokens often cannot re-exchange).
                _uat = locals().get("user_access_token")
                if _uat:
                    blob_payload["meta_user_token"] = str(_uat)
                if _raw_exp:
                    user_stamp = stamp_token_expiry({}, expires_in=_raw_exp)
                    blob_payload["meta_user_expires_at"] = user_stamp.get("expires_at")
            if platform == "instagram" and account_id:
                blob_payload["ig_user_id"] = str(account_id)
            if platform == "facebook" and account_id:
                blob_payload["page_id"] = str(account_id)
            token_blob = encrypt_blob(blob_payload)
            
            async with core.state.db_pool.acquire() as conn:
                # Check if account already connected
                existing = await conn.fetchrow(
                    "SELECT id FROM platform_tokens WHERE user_id = $1 AND platform = $2 AND account_id = $3",
                    user_id, platform, account_id
                )

                if existing:
                    await conn.execute(f"""
                        UPDATE platform_tokens SET token_blob = $1, account_name = $2, account_username = $3,
                        account_avatar = $4, updated_at = NOW(), last_oauth_reconnect_at = NOW(),
                        {OAUTH_RECONNECT_RESET_SQL}
                        WHERE id = $5
                    """, token_blob, account_name, account_username, account_avatar, existing["id"])
                    connect_action = "PLATFORM_RECONNECTED"
                else:
                    reconnect_row_id = state_data.get("reconnect_account_id")
                    reconnect_expected_provider_id = state_data.get("reconnect_expected_provider_account_id")
                    if reconnect_row_id:
                        if reconnect_expected_provider_id and str(reconnect_expected_provider_id) != str(account_id):
                            return popup_response(
                                False,
                                platform,
                                "You authenticated a different account. Please sign in to the same account you selected for reconnect.",
                            )
                        await conn.execute(
                            f"""
                            UPDATE platform_tokens
                            SET token_blob = $1,
                                account_name = $2,
                                account_username = $3,
                                account_avatar = $4,
                                account_id = $5,
                                updated_at = NOW(),
                                last_oauth_reconnect_at = NOW(),
                                {OAUTH_RECONNECT_RESET_SQL}
                            WHERE id = $6
                              AND user_id = $7
                              AND platform = $8
                            """,
                            token_blob,
                            account_name,
                            account_username,
                            account_avatar,
                            account_id,
                            reconnect_row_id,
                            user_id,
                            platform,
                        )
                        connect_action = "PLATFORM_RECONNECTED"
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
                                "reconnect_row_id": reconnect_row_id,
                            },
                        )
                        return popup_response(True, platform)
                    user_row = await conn.fetchrow(
                        "SELECT id, role, subscription_tier FROM users WHERE id = $1",
                        user_id,
                    )
                    current_count = int(await conn.fetchval(
                        "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL",
                        user_id,
                    ) or 0)
                    current_for_platform = int(await conn.fetchval(
                        "SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1 AND platform = $2 AND revoked_at IS NULL",
                        user_id, platform,
                    ) or 0)
                    allowed, reason = can_user_connect_platform(
                        dict(user_row or {}),
                        current_total=current_count,
                        current_for_platform=current_for_platform,
                    )
                    if not allowed:
                        return popup_response(
                            False,
                            platform,
                            reason,
                        )
                    await conn.execute("""
                        INSERT INTO platform_tokens (user_id, platform, account_id, account_name, account_username, account_avatar, token_blob, last_oauth_reconnect_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """, user_id, platform, account_id, account_name, account_username, account_avatar, token_blob)
                    connect_action = "PLATFORM_CONNECTED"

                await log_system_event(conn, user_id=str(user_id), action=connect_action,
                                       event_category="PLATFORM", resource_type="platform",
                                       resource_id=f"{platform}:{account_id}",
                                       details={"platform": platform, "account_name": account_name,
                                                "account_username": account_username})
            
            return popup_response(True, platform)
    
    except Exception as e:
        # Sanitize error before logging — exception message may contain tokens or secrets
        err_type = type(e).__name__
        err_safe = str(e)
        import re as _re
        err_safe = _re.sub(r'[A-Za-z0-9_-]{40,}', '***', err_safe)
        logger.error(f"OAuth callback error for {platform} ({err_type}): {err_safe}")

        # Auto-record as operational incident so admin-incidents.html shows
        # OAuth failures even when no exception escapes to the global handler.
        try:
            if core.state.db_pool is not None:
                from services.ops_incidents import record_operational_incident

                await record_operational_incident(
                    core.state.db_pool,
                    source="oauth",
                    incident_type=f"oauth_callback_failed:{platform}"[:120],
                    subject=f"OAuth callback failed for {platform} ({err_type})",
                    body=err_safe[:8000],
                    details={
                        "platform": platform,
                        "error_type": err_type,
                        "error_sanitized": err_safe[:2000],
                    },
                    user_id=locals().get("user_id"),
                )
        except Exception as _ix:
            logger.debug("oauth incident record failed: %s", _ix)

        user_msg = str(e) if len(str(e)) < 200 and "token" not in str(e).lower() else "Connection failed. Please try again."
        return popup_response(False, platform, user_msg)


@router.post("/api/oauth/{platform}/select-destination")
async def oauth_select_destination(
    platform: str,
    pick_token: str = Form(...),
    destination_id: str = Form(...),
    parent_origin: Optional[str] = Form(None),
):
    """Complete Meta OAuth after the user picks a Page / Instagram account."""
    if platform not in ("facebook", "instagram"):
        raise HTTPException(400, "Destination pick is only for Facebook and Instagram")
    from services.oauth_meta_finish import finish_meta_destination_pick

    return await finish_meta_destination_pick(
        platform=platform,
        pick_token=pick_token,
        destination_id=destination_id,
        parent_origin=parent_origin,
        frontend_url=FRONTEND_URL,
    )

