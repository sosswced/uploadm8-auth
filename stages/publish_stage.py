"""
UploadM8 Publish Stage
======================
Publish per-platform transcoded videos to social media platforms.

Platform-specific publish flows:
  TikTok   - Content Posting API v2: init upload -> PUT binary -> publish_id
  YouTube  - Resumable upload: init -> PUT binary -> video_id
  Instagram - Graph API Reels: create container (video_url) -> poll status -> publish
  Facebook - Graph API: multipart upload or URL-based Reels

Key behaviors:
  - Each platform gets its OWN transcoded video (ctx.platform_videos[platform])
  - Meta platforms (IG/FB) use presigned R2 URLs when public URL needed
  - Per-platform error isolation: one failure does NOT kill others
  - Ledger row per attempt (publish_attempts table)
  - verify_stage later confirms "accepted" -> "confirmed live"

Exports used by verify_stage:
  - decrypt_token(token_row) -> dict or None
  - init_enc_keys() -> None
"""

import os
import json
import asyncio
import logging
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import PublishError, ErrorCode
from .context import JobContext, PlatformResult
from . import db as db_stage
from . import r2 as r2_stage


logger = logging.getLogger("uploadm8-worker")

# Graph API version for Meta platforms (Instagram + Facebook)
META_API_VERSION = "v21.0"

# Instagram container polling
IG_POLL_INTERVAL = 5       # seconds between polls
IG_POLL_MAX_ATTEMPTS = 36  # 3 minutes max (36 * 5s)

# Presigned URL expiry for Meta uploads (1 hour)
PRESIGNED_URL_EXPIRY = 3600


# =====================================================================
# Token Encryption (used by this stage + verify_stage)
# =====================================================================

TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")
_ENC_KEYS: Dict[str, bytes] = {}


def init_enc_keys():
    """Parse encryption keys from environment."""
    global _ENC_KEYS
    if not TOKEN_ENC_KEYS:
        return

    clean = TOKEN_ENC_KEYS.strip().strip('"').replace("\\n", "")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        kid, b64key = part.split(":", 1)
        try:
            raw = base64.b64decode(b64key.strip())
            if len(raw) == 32:
                _ENC_KEYS[kid.strip()] = raw
        except Exception:
            pass


def decrypt_token_blob(blob: Any) -> dict:
    """Decrypt platform token blob."""
    if isinstance(blob, str):
        blob = json.loads(blob)

    kid = blob.get("kid", "v1")
    key = _ENC_KEYS.get(kid)
    if not key:
        raise ValueError(f"Unknown key id: {kid}")

    nonce = base64.b64decode(blob["nonce"])
    ciphertext = base64.b64decode(blob["ciphertext"])
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))


def decrypt_token(token_row: Any) -> Optional[dict]:
    """Best-effort decrypt/parse of a stored platform token row.

    Accepts:
      - dict: either already plaintext token data OR an encrypted blob dict
      - str: JSON string for either plaintext dict or encrypted blob

    Returns:
      - dict token payload (e.g., {"access_token": "..."})
      - None if missing/invalid

    NOTE: verify_stage imports this function directly.
    """
    if token_row is None:
        return None
    try:
        if isinstance(token_row, str):
            token_row = json.loads(token_row)
        if not isinstance(token_row, dict):
            return None

        # If this looks like an encrypted blob, decrypt it.
        if "ciphertext" in token_row and "nonce" in token_row:
            try:
                return decrypt_token_blob(token_row)
            except Exception:
                return None

        # Already plaintext token dict
        return token_row
    except Exception:
        return None


# =====================================================================
# Content Derivation
# =====================================================================

def _get_title(ctx: JobContext) -> str:
    """Get the best available title for publishing."""
    return ctx.get_effective_title() if hasattr(ctx, "get_effective_title") else (
        getattr(ctx, "ai_title", None)
        or getattr(ctx, "title", None)
        or getattr(ctx, "video_title", None)
        or getattr(ctx, "name", None)
        or f"UploadM8 {getattr(ctx, 'upload_id', '')}".strip()
    )


def _get_caption(ctx: JobContext) -> str:
    """Get the best available caption/description for publishing."""
    return ctx.get_effective_caption() if hasattr(ctx, "get_effective_caption") else (
        getattr(ctx, "ai_caption", None)
        or getattr(ctx, "caption", None)
        or getattr(ctx, "description", None)
        or ""
    )


def _get_hashtags(ctx: JobContext) -> str:
    """Get hashtag string for publishing."""
    tags = []
    if hasattr(ctx, "get_effective_hashtags"):
        tags = ctx.get_effective_hashtags()
    else:
        tags = getattr(ctx, "ai_hashtags", None) or getattr(ctx, "hashtags", None) or []
    return " ".join(tags) if tags else ""


def _build_full_caption(ctx: JobContext) -> str:
    """Build caption + hashtags combined (for IG/FB)."""
    caption = _get_caption(ctx)
    hashtags = _get_hashtags(ctx)
    parts = [p for p in [caption, hashtags] if p]
    return "\n\n".join(parts) if parts else ""


def _get_video_public_url(ctx: JobContext, platform: str) -> Optional[str]:
    """Get a publicly accessible URL for the platform's video.

    Priority:
      1. R2 public URL (if R2_PUBLIC_URL configured)
      2. R2 presigned URL (always works, expires in 1 hour)
      3. None (no URL available)
    """
    # Check if we have a processed asset R2 key for this platform
    r2_key = None

    # Try from output_artifacts stored during upload stage
    try:
        assets_json = ctx.output_artifacts.get("processed_assets", "{}")
        assets = json.loads(assets_json) if isinstance(assets_json, str) else assets_json
        r2_key = assets.get(platform)
    except Exception:
        pass

    # Fallback: construct expected key
    if not r2_key:
        r2_key = f"processed/{ctx.user_id}/{ctx.upload_id}/{platform}.mp4"

    # Try public URL first (fastest, no expiry)
    public_url = r2_stage.get_public_url(r2_key)
    if public_url:
        return public_url

    # Fall back to presigned URL (always works)
    try:
        return r2_stage.generate_presigned_url(r2_key, expires=PRESIGNED_URL_EXPIRY)
    except Exception as e:
        logger.warning(f"Could not generate presigned URL for {platform}: {e}")
        return None


# =====================================================================
# Platform Publishers
# =====================================================================


async def _refresh_tiktok_token(token_data: dict, db_pool=None, user_id: str = None) -> dict:
    """Refresh a TikTok access token using the stored refresh_token.
    TikTok access tokens expire after 24 hours; refresh tokens after 365 days.
    Persists the new token blob to DB if db_pool and user_id are provided.
    """
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        logger.warning("TikTok: No refresh_token stored, cannot refresh — user must reconnect")
        return token_data

    client_key = os.environ.get("TIKTOK_CLIENT_KEY", "")
    client_secret = os.environ.get("TIKTOK_CLIENT_SECRET", "")

    if not client_key or not client_secret:
        logger.warning("TikTok: Missing TIKTOK_CLIENT_KEY/SECRET env vars, cannot refresh token")
        return token_data

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/oauth/token/",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_key": client_key,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            if resp.status_code == 200:
                new_tokens = resp.json()
                logger.info("TikTok: Token refreshed successfully")
                updated = {
                    **token_data,
                    "access_token": new_tokens.get("access_token", token_data["access_token"]),
                    "refresh_token": new_tokens.get("refresh_token", refresh_token),
                    "expires_at": new_tokens.get("expires_in"),
                }
                # Persist back to DB so future jobs use the fresh token
                if db_pool and user_id:
                    await db_stage.save_refreshed_token(db_pool, user_id, "tiktok", updated)
                return updated
            else:
                logger.warning(f"TikTok: Token refresh failed: {resp.status_code} {resp.text[:200]}")
                return token_data
    except Exception as e:
        logger.warning(f"TikTok: Token refresh exception: {e}")
        return token_data



async def _refresh_meta_token(token_data: dict, platform: str, db_pool=None, user_id: str = None) -> dict:
    """Refresh a Meta (Instagram/Facebook) Page access token.
    
    Meta Page tokens obtained via /me/accounts are long-lived (~60 days).
    They are renewed by exchanging the stored user access token for a new
    long-lived token via the fb_exchange_token flow, then re-fetching the Page token.
    
    If the user token has expired entirely, returns original token_data and logs a warning
    so publish proceeds and fails with a clear error rather than silently skipping.
    """
    access_token = token_data.get("access_token")
    if not access_token:
        return token_data

    app_id = os.environ.get("META_APP_ID", "") or os.environ.get("FACEBOOK_CLIENT_ID", "")
    app_secret = os.environ.get("META_APP_SECRET", "") or os.environ.get("FACEBOOK_CLIENT_SECRET", "")

    if not app_id or not app_secret:
        logger.warning(f"{platform}: Missing META_APP_ID/SECRET env vars, cannot refresh token")
        return token_data

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Exchange current token for a fresh long-lived user token
            exchange_resp = await client.get(
                "https://graph.facebook.com/v18.0/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "fb_exchange_token": access_token,
                },
            )
            if exchange_resp.status_code != 200:
                logger.warning(
                    f"{platform}: Meta token exchange failed: "
                    f"{exchange_resp.status_code} {exchange_resp.text[:200]}"
                )
                return token_data

            new_user_token = exchange_resp.json().get("access_token")
            if not new_user_token:
                logger.warning(f"{platform}: Meta exchange returned no access_token")
                return token_data

            logger.info(f"{platform}: Meta user token refreshed successfully")

            # Step 2: For Instagram/Facebook Page tokens, re-fetch the Page token
            # Page tokens derived from a long-lived user token do not expire,
            # so re-fetching gives us a stable non-expiring page token.
            page_id = token_data.get("page_id") or token_data.get("ig_user_id")

            if platform == "facebook" and page_id:
                pages_resp = await client.get(
                    "https://graph.facebook.com/v18.0/me/accounts",
                    params={"access_token": new_user_token, "fields": "id,access_token"},
                )
                if pages_resp.status_code == 200:
                    pages = pages_resp.json().get("data", [])
                    matching = next((p for p in pages if p.get("id") == page_id), None)
                    if matching:
                        new_page_token = matching.get("access_token", new_user_token)
                        updated = {
                            **token_data,
                            "access_token": new_page_token,
                            "expires_at": None,  # Page tokens from LLT don't expire
                        }
                        if db_pool and user_id:
                            await db_stage.save_refreshed_token(db_pool, user_id, platform, updated)
                        return updated

            # ── Recover missing ig_user_id / page_id ────────────────────────
            # Old tokens stored before the OAuth fix may lack these IDs.
            # Now that we have a fresh user token we can fetch them on the fly
            # so the publish succeeds without forcing the user to reconnect.
            pages_resp = await client.get(
                "https://graph.facebook.com/v18.0/me/accounts",
                params={"access_token": new_user_token, "fields": "id,name,access_token,instagram_business_account"},
            )

            recovered_ig_user_id = token_data.get("ig_user_id") or token_data.get("instagram_user_id")
            recovered_page_id    = token_data.get("page_id") or token_data.get("facebook_page_id")
            recovered_page_token = new_user_token  # fallback

            if pages_resp.status_code == 200:
                pages = pages_resp.json().get("data", [])

                if platform == "instagram" and not recovered_ig_user_id:
                    # Find the Page that has an Instagram Business Account linked
                    for page in pages:
                        ig_biz = page.get("instagram_business_account")
                        if ig_biz and ig_biz.get("id"):
                            recovered_ig_user_id = ig_biz["id"]
                            recovered_page_token = page.get("access_token", new_user_token)
                            logger.info(
                                f"instagram: Recovered ig_user_id={recovered_ig_user_id} "
                                f"from Page '{page.get('name', '?')}' during token refresh"
                            )
                            break

                if platform == "facebook" and not recovered_page_id and pages:
                    first = pages[0]
                    recovered_page_id    = first["id"]
                    recovered_page_token = first.get("access_token", new_user_token)
                    logger.info(
                        f"facebook: Recovered page_id={recovered_page_id} "
                        f"from Page '{first.get('name', '?')}' during token refresh"
                    )

            # Build updated blob — preserve any existing IDs, override with recovered ones
            updated = {
                **token_data,
                "access_token":  recovered_page_token if platform == "instagram" else new_user_token,
                "expires_at":    None,
            }
            if platform == "instagram" and recovered_ig_user_id:
                updated["ig_user_id"] = recovered_ig_user_id
            if platform == "facebook" and recovered_page_id:
                updated["page_id"]    = recovered_page_id
                updated["access_token"] = recovered_page_token

            if db_pool and user_id:
                await db_stage.save_refreshed_token(db_pool, user_id, platform, updated)
            return updated

    except Exception as e:
        logger.warning(f"{platform}: Meta token refresh exception: {e}")
        return token_data


async def publish_to_tiktok(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    db_pool=None,
) -> PlatformResult:
    """Publish video to TikTok using Content Posting API v2.

    Flow: init upload -> PUT binary to upload_url -> returns publish_id
    """
    # Always refresh token first — TikTok access tokens expire after 24 hours
    token_data = await _refresh_tiktok_token(token_data, db_pool=db_pool, user_id=getattr(ctx, "user_id", None))

    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    title = _get_title(ctx)
    caption = _build_full_caption(ctx)
    # TikTok uses title field (max 150 chars), caption goes in description
    tiktok_title = (caption or title)[:150]

    try:
        file_size = video_path.stat().st_size
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Initialize upload
            init_resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "post_info": {
                        "title": tiktok_title,
                        "privacy_level": "PUBLIC_TO_EVERYONE",
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": file_size,
                    }
                }
            )

            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    http_status=init_resp.status_code,
                    error_code="INIT_FAILED",
                    error_message=f"Init failed: {init_resp.text[:200]}"
                )

            init_data = init_resp.json().get("data", {})
            upload_url = init_data.get("upload_url")
            publish_id = init_data.get("publish_id")

            if not upload_url:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error_code="NO_UPLOAD_URL",
                    error_message="No upload URL returned"
                )

            # Step 2: Upload video binary
            with open(video_path, "rb") as f:
                video_data = f.read()

            upload_resp = await client.put(
                upload_url,
                content=video_data,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(file_size),
                }
            )

            if upload_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    http_status=upload_resp.status_code,
                    error_code="UPLOAD_FAILED",
                    error_message=f"Upload failed: {upload_resp.status_code}"
                )

            logger.info(f"TikTok publish accepted: publish_id={publish_id}")
            return PlatformResult(
                platform="tiktok",
                success=True,
                publish_id=publish_id,
                verify_status="pending",
            )

    except Exception as e:
        logger.error(f"TikTok publish error: {e}")
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


async def _refresh_youtube_token(token_data: dict, db_pool=None, user_id: str = None) -> dict:
    """Attempt to refresh a YouTube access token using the stored refresh_token.
    Returns updated token_data dict with new access_token, or original if refresh fails.
    Persists updated token to DB if db_pool and user_id are provided.
    """
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        logger.warning("YouTube: No refresh_token stored, cannot refresh")
        return token_data

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "") or os.environ.get("YOUTUBE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "") or os.environ.get("YOUTUBE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        logger.warning("YouTube: Missing GOOGLE_CLIENT_ID/SECRET env vars, cannot refresh token")
        return token_data

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )
            if resp.status_code == 200:
                new_tokens = resp.json()
                logger.info("YouTube: Token refreshed successfully")
                updated = {
                    **token_data,
                    "access_token": new_tokens["access_token"],
                }
                if db_pool and user_id:
                    await db_stage.save_refreshed_token(db_pool, user_id, "youtube", updated)
                return updated
            else:
                logger.warning(f"YouTube: Token refresh failed: {resp.status_code} {resp.text[:200]}")
                return token_data
    except Exception as e:
        logger.warning(f"YouTube: Token refresh exception: {e}")
        return token_data


async def publish_to_youtube(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    db_pool=None,
) -> PlatformResult:
    """Publish video to YouTube Shorts using resumable upload.

    Flow: POST metadata (get Location header) -> PUT binary -> video_id
    """
    # Always refresh token first — YouTube access tokens expire after 1 hour
    token_data = await _refresh_youtube_token(token_data, db_pool=db_pool, user_id=getattr(ctx, "user_id", None))

    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(
            platform="youtube",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    title = _get_title(ctx)[:100]
    caption = _build_full_caption(ctx)
    description = caption[:5000] if caption else ""

    try:
        file_size = video_path.stat().st_size
        async with httpx.AsyncClient(timeout=300) as client:
            # Step 1: Initialize resumable upload
            metadata = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "categoryId": "22"  # People & Blogs
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False
                }
            }

            init_resp = await client.post(
                "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(file_size)
                },
                json=metadata
            )

            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    http_status=init_resp.status_code,
                    error_code="INIT_FAILED",
                    error_message=f"Init failed: {init_resp.text[:200]}"
                )

            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    error_code="NO_UPLOAD_URL",
                    error_message="No upload URL in response"
                )

            # Step 2: Upload video binary
            with open(video_path, "rb") as f:
                video_data = f.read()

            upload_resp = await client.put(
                upload_url,
                content=video_data,
                headers={"Content-Type": "video/mp4"}
            )

            if upload_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    http_status=upload_resp.status_code,
                    error_code="UPLOAD_FAILED",
                    error_message=f"Upload failed: {upload_resp.status_code}"
                )

            video_id = upload_resp.json().get("id")
            platform_url = f"https://youtube.com/shorts/{video_id}" if video_id else None

            logger.info(f"YouTube publish accepted: video_id={video_id}, url={platform_url}")
            return PlatformResult(
                platform="youtube",
                success=True,
                platform_video_id=video_id,
                platform_url=platform_url,
                verify_status="pending",
            )

    except Exception as e:
        logger.error(f"YouTube publish error: {e}")
        return PlatformResult(
            platform="youtube",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


async def publish_to_instagram(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    video_url: Optional[str] = None,
    db_pool=None,
) -> PlatformResult:
    """Publish video to Instagram Reels via Graph API.

    Flow (URL-based, required by Meta):
      1. POST /{ig_user_id}/media  (media_type=REELS, video_url, caption)
         -> creation_id
      2. Poll GET /{creation_id}?fields=status_code
         until status_code == "FINISHED"
      3. POST /{ig_user_id}/media_publish  (creation_id)
         -> media_id

    Requires:
      - ig_user_id in token_data (Instagram Business/Creator account ID)
      - Publicly accessible video URL (presigned R2 or public R2)
      - instagram_content_publish permission on the access token
    """
    # Refresh Meta token before use — long-lived tokens expire after ~60 days
    token_data = await _refresh_meta_token(
        token_data, "instagram", db_pool=db_pool, user_id=getattr(ctx, "user_id", None)
    )

    access_token = token_data.get("access_token")
    ig_user_id = (
        token_data.get("ig_user_id")
        or token_data.get("instagram_user_id")
        or token_data.get("instagram_page_id")
        or token_data.get("page_id")
    )

    if not access_token:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    if not ig_user_id:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="NO_IG_USER_ID",
            error_message="No Instagram user ID found in token data. Connect Instagram Business account."
        )

    # Instagram Graph API requires a public URL - cannot upload binary directly
    if not video_url:
        video_url = _get_video_public_url(ctx, "instagram")

    if not video_url:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="NO_PUBLIC_URL",
            error_message="Instagram requires a public video URL. Configure R2_PUBLIC_URL or ensure presigned URLs work."
        )

    caption = _build_full_caption(ctx)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Create media container
            logger.info(f"Instagram: Creating Reels container for ig_user_id={ig_user_id}")
            create_resp = await client.post(
                f"https://graph.facebook.com/{META_API_VERSION}/{ig_user_id}/media",
                params={
                    "access_token": access_token,
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption[:2200] if caption else "",
                    "share_to_feed": "true",
                }
            )

            if create_resp.status_code != 200:
                error_body = create_resp.text[:300]
                logger.error(f"Instagram container creation failed: {error_body}")
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    http_status=create_resp.status_code,
                    error_code="CONTAINER_FAILED",
                    error_message=f"Container creation failed: {error_body}"
                )

            creation_id = create_resp.json().get("id")
            if not creation_id:
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    error_code="NO_CREATION_ID",
                    error_message="No creation_id returned from container endpoint"
                )

            logger.info(f"Instagram: Container created, creation_id={creation_id}, polling status...")

            # Step 2: Poll until container is ready
            container_ready = False
            final_status = "UNKNOWN"

            for attempt in range(IG_POLL_MAX_ATTEMPTS):
                await asyncio.sleep(IG_POLL_INTERVAL)

                status_resp = await client.get(
                    f"https://graph.facebook.com/{META_API_VERSION}/{creation_id}",
                    params={
                        "access_token": access_token,
                        "fields": "status_code,status",
                    }
                )

                if status_resp.status_code != 200:
                    logger.warning(f"Instagram: Status poll failed (attempt {attempt + 1}): {status_resp.status_code}")
                    continue

                status_data = status_resp.json()
                final_status = status_data.get("status_code", "UNKNOWN")

                if final_status == "FINISHED":
                    container_ready = True
                    logger.info(f"Instagram: Container ready after {(attempt + 1) * IG_POLL_INTERVAL}s")
                    break
                elif final_status == "ERROR":
                    error_detail = status_data.get("status", "Unknown error")
                    return PlatformResult(
                        platform="instagram",
                        success=False,
                        error_code="CONTAINER_ERROR",
                        error_message=f"Container processing failed: {error_detail}"
                    )
                elif final_status == "EXPIRED":
                    return PlatformResult(
                        platform="instagram",
                        success=False,
                        error_code="CONTAINER_EXPIRED",
                        error_message="Container expired before publishing"
                    )
                else:
                    # IN_PROGRESS or other status
                    if (attempt + 1) % 6 == 0:  # Log every 30s
                        logger.info(f"Instagram: Still processing... status={final_status} ({(attempt + 1) * IG_POLL_INTERVAL}s)")

            if not container_ready:
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    error_code="CONTAINER_TIMEOUT",
                    error_message=f"Container not ready after {IG_POLL_MAX_ATTEMPTS * IG_POLL_INTERVAL}s (last status: {final_status})"
                )

            # Step 3: Publish the container
            publish_resp = await client.post(
                f"https://graph.facebook.com/{META_API_VERSION}/{ig_user_id}/media_publish",
                params={
                    "access_token": access_token,
                    "creation_id": creation_id,
                }
            )

            if publish_resp.status_code != 200:
                error_body = publish_resp.text[:300]
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    http_status=publish_resp.status_code,
                    error_code="PUBLISH_FAILED",
                    error_message=f"Publish failed: {error_body}"
                )

            media_id = publish_resp.json().get("id")
            platform_url = f"https://www.instagram.com/reel/{media_id}/" if media_id else None

            logger.info(f"Instagram publish accepted: media_id={media_id}")
            return PlatformResult(
                platform="instagram",
                success=True,
                platform_video_id=media_id,
                platform_url=platform_url,
                publish_id=creation_id,
                verify_status="pending",
            )

    except httpx.TimeoutException:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="TIMEOUT",
            error_message="Instagram API request timed out"
        )
    except Exception as e:
        logger.error(f"Instagram publish error: {e}")
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


async def publish_to_facebook(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    video_url: Optional[str] = None,
    db_pool=None,
) -> PlatformResult:
    """Publish video to Facebook as a Reel.

    Supports two modes:
      1. Binary upload (multipart) - when local file available
      2. URL-based - when public URL available (preferred for large files)

    As of June 2025, all Facebook videos are treated as Reels.
    """
    # Refresh Meta token before use — long-lived tokens expire after ~60 days
    token_data = await _refresh_meta_token(
        token_data, "facebook", db_pool=db_pool, user_id=getattr(ctx, "user_id", None)
    )

    access_token = token_data.get("access_token")
    page_id = (
        token_data.get("page_id")
        or token_data.get("facebook_page_id")
        or token_data.get("fb_page_id")
    )

    if not access_token:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    if not page_id:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="NO_PAGE_ID",
            error_message="No Facebook page ID found in token data"
        )

    description = _build_full_caption(ctx)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Prefer binary upload (simpler, works without public URL)
            if video_path and video_path.exists():
                with open(video_path, "rb") as f:
                    files = {"source": ("video.mp4", f, "video/mp4")}
                    resp = await client.post(
                        f"https://graph.facebook.com/{META_API_VERSION}/{page_id}/videos",
                        params={
                            "access_token": access_token,
                            "description": description[:5000] if description else "",
                        },
                        files=files,
                    )
            elif video_url:
                # URL-based upload (fallback)
                resp = await client.post(
                    f"https://graph.facebook.com/{META_API_VERSION}/{page_id}/videos",
                    params={
                        "access_token": access_token,
                        "file_url": video_url,
                        "description": description[:5000] if description else "",
                    },
                )
            else:
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    error_code="NO_VIDEO",
                    error_message="No video file or URL available for Facebook"
                )

            if resp.status_code != 200:
                error_body = resp.text[:300]
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    http_status=resp.status_code,
                    error_code="UPLOAD_FAILED",
                    error_message=f"Upload failed: {error_body}"
                )

            video_id = resp.json().get("id")
            logger.info(f"Facebook publish accepted: video_id={video_id}")
            return PlatformResult(
                platform="facebook",
                success=True,
                platform_video_id=video_id,
                verify_status="pending",
            )

    except httpx.TimeoutException:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="TIMEOUT",
            error_message="Facebook API request timed out"
        )
    except Exception as e:
        logger.error(f"Facebook publish error: {e}")
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


# =====================================================================
# Stage Entry Point
# =====================================================================

async def run_publish_stage(ctx: JobContext, db_pool) -> JobContext:
    """Publish to each platform independently + write ledger rows.

    Per-platform video selection priority:
      1. ctx.platform_videos[platform]  (local temp file from transcode)
      2. Fallback: ctx.processed_video_path or ctx.local_video_path

    Per-platform error isolation:
      - Each platform publishes independently
      - One failure does NOT prevent others from publishing
      - Results stored per-platform in ctx.platform_results

    Ledger contract:
      - INSERT publish_attempts row BEFORE the API call
      - UPDATE row after call (success/fail + metadata)
      - verify_stage later turns verify_status into confirmed/rejected/unknown
    """
    if not ctx.platforms:
        logger.warning(f"No platforms specified for upload {ctx.upload_id}")
        return ctx

    # Determine default fallback video
    default_video = ctx.processed_video_path or ctx.local_video_path
    has_any_video = False

    # Check we have at least one video available
    if ctx.platform_videos:
        has_any_video = any(
            p and p.exists() for p in ctx.platform_videos.values()
        )
    if not has_any_video and default_video and default_video.exists():
        has_any_video = True

    if not has_any_video:
        raise PublishError("No video files available to publish", code=ErrorCode.UPLOAD_FAILED)

    logger.info(f"Publishing to platforms: {ctx.platforms}")
    init_enc_keys()

    # Token DB key mapping (platform -> token table platform key)
    platform_to_db_key = {
        "tiktok": "tiktok",
        "youtube": "youtube",
        "instagram": "instagram",
        "facebook": "facebook",
    }

    for platform in ctx.platforms:
        db_key = platform_to_db_key.get(platform, platform)

        # -- Select the correct video file for this platform --
        video_file = ctx.get_video_for_platform(platform)
        if not video_file or not video_file.exists():
            video_file = default_video
            if video_file:
                logger.warning(
                    f"{platform}: Platform-specific video not found, "
                    f"using fallback: {video_file.name}"
                )

        # -- Create ledger row (before API call) --
        attempt_id = None
        try:
            attempt_id = await db_stage.insert_publish_attempt(
                db_pool,
                upload_id=str(ctx.upload_id),
                user_id=str(ctx.user_id),
                platform=str(platform),
            )
        except Exception as e:
            logger.warning(f"{platform}: Could not create publish_attempt row: {e}")
            attempt_id = None

        # -- Load platform token --
        token_data = None
        try:
            raw_token = await db_stage.load_platform_token(db_pool, ctx.user_id, db_key)
            # Decrypt if encrypted blob (token_blob is always encrypted via encrypt_blob())
            token_data = decrypt_token(raw_token) if raw_token else None
        except Exception as e:
            logger.warning(f"{platform}: Token load failed: {e}")
            token_data = None

        if not token_data:
            msg = f"Not connected to {platform}"
            logger.warning(f"{platform}: {msg}")
            if attempt_id:
                try:
                    await db_stage.update_publish_attempt_failed(
                        db_pool,
                        attempt_id=attempt_id,
                        error_code="NOT_CONNECTED",
                        error_message=msg,
                    )
                except Exception:
                    pass
            ctx.platform_results.append(PlatformResult(
                platform=platform,
                success=False,
                attempt_id=attempt_id,
                error_code="NOT_CONNECTED",
                error_message=msg,
            ))
            continue

        # -- Publish to platform --
        result = None
        try:
            if platform == "tiktok":
                result = await publish_to_tiktok(video_file, ctx, token_data, db_pool=db_pool)

            elif platform == "youtube":
                result = await publish_to_youtube(video_file, ctx, token_data, db_pool=db_pool)

            elif platform == "instagram":
                # Instagram needs a public URL (Graph API requirement)
                video_url = _get_video_public_url(ctx, "instagram")
                result = await publish_to_instagram(
                    video_file, ctx, token_data, video_url=video_url, db_pool=db_pool
                )

            elif platform == "facebook":
                # Facebook can use binary upload OR URL
                video_url = _get_video_public_url(ctx, "facebook")
                result = await publish_to_facebook(
                    video_file, ctx, token_data, video_url=video_url, db_pool=db_pool
                )

            else:
                result = PlatformResult(
                    platform=platform,
                    success=False,
                    error_code="UNSUPPORTED",
                    error_message=f"Unsupported platform: {platform}"
                )

        except Exception as e:
            logger.exception(f"Error publishing to {platform}")
            result = PlatformResult(
                platform=platform,
                success=False,
                error_code="PUBLISH_EXCEPTION",
                error_message=str(e)
            )

        # -- Record result --
        result.attempt_id = attempt_id
        ctx.platform_results.append(result)

        status_icon = "OK" if result.success else "FAIL"
        extra = ""
        if result.publish_id:
            extra += f" publish_id={result.publish_id}"
        if result.platform_video_id:
            extra += f" video_id={result.platform_video_id}"
        if result.error_message and not result.success:
            extra += f" error={result.error_message[:100]}"

        logger.info(f"{platform} [{status_icon}]: {result.error_code or 'accepted'}{extra}")

        # -- Update ledger row --
        if attempt_id:
            try:
                if result.success:
                    await db_stage.update_publish_attempt_success(
                        db_pool,
                        attempt_id=attempt_id,
                        platform_post_id=result.platform_video_id,
                        platform_url=result.platform_url,
                        http_status=result.http_status,
                        response_payload=result.response_payload,
                        publish_id=result.publish_id,
                    )
                else:
                    await db_stage.update_publish_attempt_failed(
                        db_pool,
                        attempt_id=attempt_id,
                        error_code=result.error_code or "PUBLISH_FAILED",
                        error_message=result.error_message or "Publish failed",
                        http_status=result.http_status,
                        response_payload=result.response_payload,
                    )
            except Exception as e:
                logger.warning(f"{platform}: Could not update publish_attempt: {e}")

    # -- Summary --
    succeeded = [r.platform for r in ctx.platform_results if r.success]
    failed = [r.platform for r in ctx.platform_results if not r.success]
    logger.info(f"Publish complete: succeeded={succeeded}, failed={failed}")

    return ctx
