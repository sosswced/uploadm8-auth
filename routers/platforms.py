"""
UploadM8 Platform & OAuth routes — extracted from app.py.
Platform account listing, disconnection, and OAuth start/callback flows.
"""

import json
import secrets
import logging
import re as _re
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import HTMLResponse

import core.state
from core.deps import get_current_user
from core.auth import encrypt_blob, decrypt_blob
from core.oauth import (
    _store_oauth_state,
    _pop_oauth_state,
    get_oauth_redirect_uri,
    _revoke_platform_token,
)
from core.config import (
    OAUTH_CONFIG,
    TIKTOK_CLIENT_KEY,
    TIKTOK_CLIENT_SECRET,
    YOUTUBE_CLIENT_ID,
    YOUTUBE_CLIENT_SECRET,
    META_APP_ID,
    META_APP_SECRET,
    INSTAGRAM_CLIENT_ID,
    INSTAGRAM_CLIENT_SECRET,
    FACEBOOK_CLIENT_ID,
    FACEBOOK_CLIENT_SECRET,
    FRONTEND_URL,
)
from core.audit import log_system_event
from core.helpers import _now_utc, get_plan

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["platforms"])


@router.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])

    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms: platforms[p] = []
        platforms[p].append({"id": str(acc["id"]), "account_id": acc["account_id"], "name": acc["account_name"], "username": acc["account_username"], "avatar": acc["account_avatar"], "is_primary": acc["is_primary"], "status": "active", "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None})

    plan = get_plan(user.get("subscription_tier", "free"))
    total = sum(len(v) for v in platforms.values())
    return {"platforms": platforms, "total_accounts": total, "max_accounts": plan.get("max_accounts", 1), "can_add_more": total < plan.get("max_accounts", 1)}

# Alias endpoint for frontend compatibility
@router.get("/api/platform-accounts")
async def get_platform_accounts(user: dict = Depends(get_current_user)):
    """Returns flat list of accounts for frontend compatibility"""
    async with core.state.db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])

    result = []
    for acc in accounts:
        result.append({
            "id": str(acc["id"]),
            "platform": acc["platform"],
            "account_id": acc["account_id"],
            "account_name": acc["account_name"],
            "account_username": acc["account_username"],
            "account_avatar_url": acc["account_avatar"],
            "is_primary": acc["is_primary"],
            "status": "active",
            "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None,
        })
    return {"accounts": result}

@router.get("/api/accounts")
async def get_accounts_simple(user: dict = Depends(get_current_user)):
    """Simple accounts list for dashboard"""
    async with core.state.db_pool.acquire() as conn:
        accounts = await conn.fetch("SELECT DISTINCT ON (platform, account_id) id, platform, account_name, account_username, account_avatar FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL AND account_id IS NOT NULL AND account_id <> '' AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '') ORDER BY platform, account_id, created_at DESC", user["id"])
    return [{"id": str(a["id"]), "platform": a["platform"], "name": a["account_name"], "username": a["account_username"], "avatar": a["account_avatar"], "status": "active"} for a in accounts]

@router.delete("/api/platforms/{platform}/accounts/{account_id}")
async def disconnect_account(
    platform: str,
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Disconnect a linked platform account.
    1. Revokes the access token at the provider.
    2. Hard-deletes the platform_tokens row.
    3. Writes a platform_disconnect_log record.
    """
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE id = $1 AND user_id = $2",
            account_id,
            user["id"],
        )
        if not row:
            raise HTTPException(404, "Account not found")

        # Revoke token at provider
        ok, err = await _revoke_platform_token(row["platform"], row["token_blob"])

        # Hard-delete (mark revoked_at first to satisfy the partial unique index,
        # then delete so no stale token lingers)
        await conn.execute(
            "UPDATE platform_tokens SET revoked_at = NOW() WHERE id = $1", row["id"]
        )
        await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])

        # Audit log — platform_disconnect_log (existing) + system_event_log (new)
        await conn.execute(
            """
            INSERT INTO platform_disconnect_log
                (user_id, platform, account_id, account_name,
                 revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
            VALUES ($1,$2,$3,$4,$5,$6,'self',$7)
            """,
            str(user["id"]),
            row["platform"],
            row["account_id"],
            row["account_name"],
            ok,
            err or None,
            ip_addr,
        )
        await log_system_event(conn, user_id=str(user["id"]), action="PLATFORM_DISCONNECTED",
                               event_category="PLATFORM", resource_type="platform",
                               resource_id=f"{row['platform']}:{row['account_id']}",
                               details={"platform": row["platform"], "account_name": row["account_name"],
                                        "provider_revoked": ok, "provider_error": err},
                               severity="WARNING")

    return {"status": "disconnected", "provider_revoked": ok}


@router.delete("/api/platform-accounts/{account_id}")
async def disconnect_account_by_id(
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Alias for the disconnect endpoint (used by older frontend code)."""
    return await disconnect_account(
        platform="",      # platform is looked up from the DB row
        account_id=account_id,
        request=request,
        user=user,
    )


@router.get("/api/oauth/{platform}/start")
async def oauth_start(platform: str, user: dict = Depends(get_current_user)):
    """Start OAuth flow for a platform"""
    if platform not in OAUTH_CONFIG:
        raise HTTPException(400, f"Unsupported platform: {platform}")

    # Check account limits
    plan = get_plan(user.get("subscription_tier", "free"))
    async with core.state.db_pool.acquire() as conn:
        current_count = await conn.fetchval("SELECT COUNT(*) FROM platform_tokens WHERE user_id = $1", user["id"])

    if current_count >= plan.get("max_accounts", 1):
        raise HTTPException(403, f"Account limit reached ({plan.get('max_accounts', 1)}). Upgrade to add more.")

    config = OAUTH_CONFIG[platform]
    state = secrets.token_urlsafe(32)

    # Store state in Redis (required for multi-instance safety)
    await _store_oauth_state(state, {
        "user_id": str(user["id"]),
        "platform": platform,
        "created_at": _now_utc().isoformat()
    })

    redirect_uri = get_oauth_redirect_uri(platform)

    if platform == "tiktok":
        params = {
            "client_key": TIKTOK_CLIENT_KEY,
            "scope": config["scope"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
        }
    elif platform == "youtube":
        params = {
            "client_id": YOUTUBE_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": config["scope"],
            "state": state,
            "access_type": "offline",
            "prompt": "select_account consent",  # Forces account picker + fresh consent
        }
    elif platform == "instagram":
        params = {
            "client_id": INSTAGRAM_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": config["scope"],
            "response_type": "code",
            "state": state,
            "auth_type": "rerequest",  # Force re-authentication
        }
    elif platform == "facebook":
        params = {
            "client_id": FACEBOOK_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": config["scope"],
            "response_type": "code",
            "state": state,
            "auth_type": "rerequest",  # Force re-authentication
        }

    auth_url = f"{config['auth_url']}?{urlencode(params)}"
    return {"auth_url": auth_url, "state": state}

@router.get("/api/oauth/{platform}/callback")
async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):
    """Handle OAuth callback - returns HTML that communicates with parent window"""

    def popup_response(success: bool, platform: str, error_msg: str = None):
        """Generate HTML that posts message to parent window and closes popup"""
        if success:
            message = f'{{"type": "oauth_success", "platform": "{platform}"}}'
        else:
            safe_error = (error_msg or "Unknown error").replace('"', '\\"').replace('\n', ' ')[:200]
            message = f'{{"type": "oauth_error", "platform": "{platform}", "error": "{safe_error}"}}'

        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Connecting...</title></head>
        <body style="font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #1a1a2e; color: white;">
            <div style="text-align: center;">
                <p>{"✓ Connected successfully!" if success else "✗ Connection failed"}</p>
                <p style="color: #888; font-size: 14px;">This window will close automatically...</p>
            </div>
            <script>
                if (window.opener) {{
                    window.opener.postMessage({message}, '{FRONTEND_URL}');
                }}
                setTimeout(() => window.close(), 1500);
            </script>
        </body>
        </html>
        """)

    if error:
        return popup_response(False, platform, error)

    state_data = await _pop_oauth_state(state) if state else None
    if not state_data:
        return popup_response(False, platform, "Invalid or expired session. Please try again.")
    user_id = state_data["user_id"]

    if not code:
        return popup_response(False, platform, "No authorization code received")

    config = OAUTH_CONFIG[platform]
    redirect_uri = get_oauth_redirect_uri(platform)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Exchange code for tokens based on platform
            if platform == "tiktok":
                token_response = await client.post(config["token_url"], data={
                    "client_key": TIKTOK_CLIENT_KEY,
                    "client_secret": TIKTOK_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                })
                token_data = token_response.json()
                access_token = token_data.get("access_token")
                account_id = token_data.get("open_id", secrets.token_hex(8))
                account_name = "TikTok User"
                account_username = ""
                account_avatar = ""

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
                    account_username = snippet.get("customUrl", "")
                    account_avatar = snippet.get("thumbnails", {}).get("default", {}).get("url", "")
                else:
                    account_id = secrets.token_hex(8)
                    account_name = "YouTube Channel"
                    account_username = ""
                    account_avatar = ""

            elif platform == "instagram":
                # Instagram Graph API: authenticate via Facebook, get Instagram Business Account
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

                # Get Facebook Pages the user manages
                pages_response = await client.get(
                    f"https://graph.facebook.com/v18.0/me/accounts?access_token={user_access_token}"
                )
                pages_data = pages_response.json()
                pages = pages_data.get("data", [])

                if not pages:
                    raise Exception("No Facebook Pages found. You need a Facebook Page connected to an Instagram Business account.")

                # Find Instagram Business Account connected to any page
                instagram_account = None
                page_access_token = None

                for page in pages:
                    page_id = page.get("id")
                    page_token = page.get("access_token")

                    # Check if this page has an Instagram Business Account
                    ig_response = await client.get(
                        f"https://graph.facebook.com/v18.0/{page_id}?fields=instagram_business_account&access_token={page_token}"
                    )
                    ig_data = ig_response.json()

                    if "instagram_business_account" in ig_data:
                        ig_account_id = ig_data["instagram_business_account"]["id"]

                        # Get Instagram account details
                        ig_details_response = await client.get(
                            f"https://graph.facebook.com/v18.0/{ig_account_id}?fields=id,username,name,profile_picture_url&access_token={page_token}"
                        )
                        ig_details = ig_details_response.json()

                        instagram_account = ig_details
                        page_access_token = page_token
                        break

                if not instagram_account:
                    raise Exception("No Instagram Business account found connected to your Facebook Pages. Connect your Instagram Business/Creator account to a Facebook Page first.")

                account_id = instagram_account.get("id")
                account_name = instagram_account.get("name") or instagram_account.get("username", "Instagram Account")
                account_username = instagram_account.get("username", "")
                account_avatar = instagram_account.get("profile_picture_url", "")
                access_token = page_access_token  # Use Page token for API calls

            elif platform == "facebook":
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

                # Facebook Reels require a Page token, not a user token.
                # Fetch the user's Pages and use the first one.
                pages_response = await client.get(
                    "https://graph.facebook.com/v18.0/me/accounts",
                    params={"access_token": user_access_token, "fields": "id,name,access_token,picture"},
                )
                pages_data = pages_response.json()
                pages = pages_data.get("data", [])

                if not pages:
                    raise Exception(
                        "No Facebook Pages found. You need a Facebook Page to publish Reels. "
                        "Create a Page at facebook.com/pages/create and try again."
                    )

                # Use the first Page
                page = pages[0]
                account_id    = page["id"]                                         # Page ID
                account_name  = page.get("name", "Facebook Page")
                account_username = ""
                account_avatar   = page.get("picture", {}).get("data", {}).get("url", "")
                access_token     = page["access_token"]                            # Page token

            # Refuse to persist "ghost" connections with no identity
            if not account_id or (isinstance(account_id, str) and account_id.strip() == ""):
                raise Exception("Provider did not return account_id; refusing to store token (prevents phantom accounts).")

            # Store the token — include platform-specific IDs in the blob so
            # publish_stage can read them without needing a separate DB lookup.
            blob_payload = {
                "access_token": access_token,
                "refresh_token": token_data.get("refresh_token"),
                "expires_at": token_data.get("expires_in"),
            }
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
                    await conn.execute("""
                        UPDATE platform_tokens SET token_blob = $1, account_name = $2, account_username = $3,
                        account_avatar = $4, updated_at = NOW() WHERE id = $5
                    """, json.dumps(token_blob), account_name, account_username, account_avatar, existing["id"])
                    connect_action = "PLATFORM_RECONNECTED"
                else:
                    await conn.execute("""
                        INSERT INTO platform_tokens (user_id, platform, account_id, account_name, account_username, account_avatar, token_blob)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, user_id, platform, account_id, account_name, account_username, account_avatar, json.dumps(token_blob))
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
        # Strip anything that looks like a token (long alphanumeric strings)
        err_safe = _re.sub(r'[A-Za-z0-9_-]{40,}', '***', err_safe)
        logger.error(f"OAuth callback error for {platform} ({err_type}): {err_safe}")
        # Return a stable user-facing message — never echo raw exception text to the browser
        user_msg = str(e) if len(str(e)) < 200 and "token" not in str(e).lower() else "Connection failed. Please try again."
        return popup_response(False, platform, user_msg)
