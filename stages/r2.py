"""
UploadM8 Publish Stage
======================
Publish videos to social media platforms.
Wraps existing platform upload modules.
"""

import os
import json
import logging
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import PublishError, ErrorCode
from .context import JobContext, PlatformResult
from . import db as db_stage


logger = logging.getLogger("uploadm8-worker")


# Encryption keys (initialized at startup)
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




# ---------------------------------------------------------------------
# Compatibility exports for verify_stage
# verify_stage imports: from .publish_stage import decrypt_token, init_enc_keys
# This project stores tokens sometimes as encrypted blobs; other times as plain dicts.
# decrypt_token returns a dict with access_token etc, or None on failure.
# ---------------------------------------------------------------------
def decrypt_token(token_row: Any) -> Optional[dict]:
    """Best-effort decrypt/parse of a stored platform token row.

    Accepts:
      - dict: either already plaintext token data OR an encrypted blob dict
      - str: JSON string for either plaintext dict or encrypted blob

    Returns:
      - dict token payload (e.g., {"access_token": "..."})
      - None if missing/invalid
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


def _derive_title_desc(ctx: JobContext) -> tuple[str, str]:
    """Derive best-effort title/description from JobContext without assuming schema."""
    title = getattr(ctx, "title", None) or getattr(ctx, "video_title", None) or getattr(ctx, "name", None)
    desc = getattr(ctx, "description", None) or getattr(ctx, "caption", None) or ""
    if not title:
        title = f"UploadM8 {getattr(ctx, 'upload_id', '')}".strip()
    return str(title), str(desc)


async def publish_to_tiktok_ctx(video_path: Path, ctx: JobContext, token_data: dict) -> PlatformResult:
    title, _ = _derive_title_desc(ctx)
    return await publish_to_tiktok(video_path, title, token_data)


async def publish_to_youtube_ctx(video_path: Path, ctx: JobContext, token_data: dict) -> PlatformResult:
    title, desc = _derive_title_desc(ctx)
    return await publish_to_youtube(video_path, title, desc, token_data)
async def publish_to_tiktok(
    video_path: Path,
    title: str,
    token_data: dict
) -> PlatformResult:
    """Publish video to TikTok using Content Posting API."""
    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_message="No access token"
        )
    
    try:
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
                        "title": title[:150],
                        "privacy_level": "PUBLIC_TO_EVERYONE",
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": video_path.stat().st_size,
                    }
                }
            )
            
            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error_message=f"Init failed: {init_resp.text[:200]}"
                )
            
            init_data = init_resp.json().get("data", {})
            upload_url = init_data.get("upload_url")
            publish_id = init_data.get("publish_id")
            
            if not upload_url:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error_message="No upload URL returned"
                )
            
            # Step 2: Upload video
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            upload_resp = await client.put(
                upload_url,
                content=video_data,
                headers={"Content-Type": "video/mp4"}
            )
            
            if upload_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error_message=f"Upload failed: {upload_resp.status_code}"
                )
            
            return PlatformResult(
                platform="tiktok",
                success=True,
                publish_id=publish_id
            )
    
    except Exception as e:
        logger.error(f"TikTok publish error: {e}")
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_message=str(e)
        )


async def publish_to_youtube(
    video_path: Path,
    title: str,
    description: str,
    token_data: dict
) -> PlatformResult:
    """Publish video to YouTube using resumable upload."""
    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(
            platform="youtube",
            success=False,
            error_message="No access token"
        )
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Initialize resumable upload
            metadata = {
                "snippet": {
                    "title": title[:100],
                    "description": description[:5000] if description else "",
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
                    "X-Upload-Content-Length": str(video_path.stat().st_size)
                },
                json=metadata
            )
            
            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    error_message=f"Init failed: {init_resp.text[:200]}"
                )
            
            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    error_message="No upload URL"
                )
            
            # Upload video
            with open(video_path, 'rb') as f:
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
                    error_message=f"Upload failed: {upload_resp.status_code}"
                )
            
            video_id = upload_resp.json().get("id")
            return PlatformResult(
                platform="youtube",
                success=True,
                platform_video_id=video_id,
                platform_url=f"https://youtube.com/shorts/{video_id}" if video_id else None
            )
    
    except Exception as e:
        logger.error(f"YouTube publish error: {e}")
        return PlatformResult(
            platform="youtube",
            success=False,
            error_message=str(e)
        )


async def publish_to_instagram(
    video_path: Path,
    caption: str,
    token_data: dict,
    page_id: Optional[str] = None
) -> PlatformResult:
    """Publish video to Instagram Reels."""
    access_token = token_data.get("access_token")
    if not access_token or not page_id:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_message="Missing access token or page ID"
        )
    
    # Instagram Reels requires a public URL
    # This is a limitation - would need CDN or public R2 URL
    return PlatformResult(
        platform="instagram",
        success=False,
        error_message="Instagram Reels upload requires public video URL (coming soon)"
    )


async def publish_to_facebook(
    video_path: Path,
    description: str,
    token_data: dict,
    page_id: Optional[str] = None
) -> PlatformResult:
    """Publish video to Facebook."""
    access_token = token_data.get("access_token")
    if not access_token or not page_id:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_message="Missing access token or page ID"
        )
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(video_path, 'rb') as f:
                files = {"source": ("video.mp4", f, "video/mp4")}
                resp = await client.post(
                    f"https://graph.facebook.com/v19.0/{page_id}/videos",
                    params={
                        "access_token": access_token,
                        "description": description[:5000] if description else ""
                    },
                    files=files
                )
            
            if resp.status_code != 200:
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    error_message=f"Upload failed: {resp.text[:200]}"
                )
            
            video_id = resp.json().get("id")
            return PlatformResult(
                platform="facebook",
                success=True,
                platform_video_id=video_id
            )
    
    except Exception as e:
        logger.error(f"Facebook publish error: {e}")
        return PlatformResult(
            platform="facebook",
            success=False,
            error_message=str(e)
        )


async def run_publish_stage(ctx: JobContext, db_pool) -> JobContext:
    """Publish to platforms + write a ledger row per publish attempt.

    Ledger contract:
      - INSERT publish_attempts row before the API call
      - UPDATE after call (success/fail)
      - verification loop later turns verify_status into confirmed/rejected/unknown
    """

    if not ctx.platforms:
        logger.warning(f"No platforms specified for upload {ctx.upload_id}")
        return ctx

    default_video = ctx.processed_video_path or ctx.local_video_path
    if not default_video or not default_video.exists():
        raise PublishError("No video file to publish", code=ErrorCode.UPLOAD_FAILED)

    logger.info(f"Publishing to platforms: {ctx.platforms}")
    init_enc_keys()

    platform_to_db_key = {"tiktok": "tiktok", "youtube": "google", "instagram": "meta", "facebook": "meta"}

    for platform in ctx.platforms:
        db_key = platform_to_db_key.get(platform, platform)

        video_file = ctx.get_video_for_platform(platform)
        if not video_file or not video_file.exists():
            video_file = default_video

        attempt_id = None
        try:
            attempt_id = await db_stage.insert_publish_attempt(
                db_pool,
                upload_id=str(ctx.upload_id),
                user_id=str(ctx.user_id),
                platform=str(platform),
            )
        except Exception:
            attempt_id = None

        token_data = None
        try:
            token_data = await db_stage.load_platform_token(db_pool, ctx.user_id, db_key)
        except Exception:
            token_data = None

        if not token_data:
            msg = f"Not connected to {platform}"
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
            ctx.platform_results.append(PlatformResult(platform=platform, success=False, attempt_id=attempt_id, error_code="NOT_CONNECTED", error_message=msg))
            continue

        try:
            if platform == "tiktok":
                result = await publish_to_tiktok_ctx(video_file, ctx, token_data)
            elif platform == "youtube":
                result = await publish_to_youtube_ctx(video_file, ctx, token_data)
            elif platform == "instagram":
                page_id = token_data.get('page_id') or token_data.get('instagram_page_id') or token_data.get('ig_user_id')
                result = await publish_to_instagram(video_file, (getattr(ctx, 'caption', None) or getattr(ctx, 'description', '') or ''), token_data, page_id)
            elif platform == "facebook":
                page_id = token_data.get('page_id') or token_data.get('facebook_page_id') or token_data.get('fb_page_id')
                result = await publish_to_facebook(video_file, (getattr(ctx, 'description', None) or getattr(ctx, 'caption', '') or ''), token_data, page_id)
            else:
                result = PlatformResult(platform=platform, success=False, error_code="UNSUPPORTED", error_message=f"Unsupported platform: {platform}")
        except Exception as e:
            logger.exception(f"Error publishing to {platform}")
            result = PlatformResult(platform=platform, success=False, error_code="PUBLISH_EXCEPTION", error_message=str(e))

        result.attempt_id = attempt_id
        ctx.platform_results.append(result)

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
            except Exception:
                pass

    return ctx
