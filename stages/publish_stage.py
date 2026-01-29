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
            error="No access token"
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
                    error=f"Init failed: {init_resp.text[:200]}"
                )
            
            init_data = init_resp.json().get("data", {})
            upload_url = init_data.get("upload_url")
            publish_id = init_data.get("publish_id")
            
            if not upload_url:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error="No upload URL returned"
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
                    error=f"Upload failed: {upload_resp.status_code}"
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
            error=str(e)
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
            error="No access token"
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
                    error=f"Init failed: {init_resp.text[:200]}"
                )
            
            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    error="No upload URL"
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
                    error=f"Upload failed: {upload_resp.status_code}"
                )
            
            video_id = upload_resp.json().get("id")
            return PlatformResult(
                platform="youtube",
                success=True,
                video_id=video_id,
                url=f"https://youtube.com/shorts/{video_id}" if video_id else None
            )
    
    except Exception as e:
        logger.error(f"YouTube publish error: {e}")
        return PlatformResult(
            platform="youtube",
            success=False,
            error=str(e)
        )


async def publish_to_instagram(
    video_path: Path,
    caption: str,
    token_data: dict,
    page_id: str
) -> PlatformResult:
    """Publish video to Instagram Reels."""
    access_token = token_data.get("access_token")
    if not access_token or not page_id:
        return PlatformResult(
            platform="instagram",
            success=False,
            error="Missing access token or page ID"
        )
    
    # Instagram Reels requires a public URL
    # This is a limitation - would need CDN or public R2 URL
    return PlatformResult(
        platform="instagram",
        success=False,
        error="Instagram Reels upload requires public video URL (coming soon)"
    )


async def publish_to_facebook(
    video_path: Path,
    description: str,
    token_data: dict,
    page_id: str
) -> PlatformResult:
    """Publish video to Facebook."""
    access_token = token_data.get("access_token")
    if not access_token or not page_id:
        return PlatformResult(
            platform="facebook",
            success=False,
            error="Missing access token or page ID"
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
                    error=f"Upload failed: {resp.text[:200]}"
                )
            
            video_id = resp.json().get("id")
            return PlatformResult(
                platform="facebook",
                success=True,
                video_id=video_id
            )
    
    except Exception as e:
        logger.error(f"Facebook publish error: {e}")
        return PlatformResult(
            platform="facebook",
            success=False,
            error=str(e)
        )


async def run_publish_stage(
    ctx: JobContext,
    token_loader
) -> JobContext:
    """
    Execute platform publishing stage.
    
    Args:
        ctx: Job context
        token_loader: Async function to load platform tokens
        
    Returns:
        Updated context with publish results
    """
    if not ctx.platforms:
        logger.warning(f"No platforms specified for upload {ctx.upload_id}")
        return ctx
    
    # Check we have at least one video
    default_video = ctx.processed_video_path or ctx.local_video_path
    if not default_video or not default_video.exists():
        raise PublishError(
            "No video file to publish",
            code=ErrorCode.UPLOAD_FAILED
        )
    
    logger.info(f"Publishing to platforms: {ctx.platforms}")
    
    # Initialize encryption keys
    init_enc_keys()
    
    # Platform key mapping
    platform_to_db_key = {
        "tiktok": "tiktok",
        "youtube": "google",
        "instagram": "meta",
        "facebook": "meta"
    }
    
    for platform in ctx.platforms:
        db_key = platform_to_db_key.get(platform, platform)
        
        # Get the best video file for this platform
        # This uses the platform-specific transcoded version if available
        video_path = ctx.get_video_for_platform(platform)
        if not video_path or not video_path.exists():
            video_path = default_video
        
        logger.info(f"Using video for {platform}: {video_path.name}")
        
        # Load token
        token_blob = await token_loader(ctx.user_id, db_key)
        if not token_blob:
            ctx.platform_results.append(PlatformResult(
                platform=platform,
                success=False,
                error_message="Platform not connected"
            ))
            continue
        
        # Decrypt token
        try:
            token_data = decrypt_token_blob(token_blob)
        except Exception as e:
            ctx.platform_results.append(PlatformResult(
                platform=platform,
                success=False,
                error_message=f"Token decrypt failed: {e}"
            ))
            continue
        
        # Get final title and caption
        final_title = ctx.get_effective_title()
        final_caption = ctx.get_effective_caption()
        
        # Publish based on platform
        if platform == "tiktok":
            result = await publish_to_tiktok(
                video_path,
                final_title,
                token_data
            )
        elif platform == "youtube":
            result = await publish_to_youtube(
                video_path,
                final_title,
                final_caption,
                token_data
            )
        elif platform == "instagram":
            page_id = ctx.user_settings.get("selected_page_id")
            result = await publish_to_instagram(
                video_path,
                final_caption,
                token_data,
                page_id
            )
        elif platform == "facebook":
            page_id = ctx.user_settings.get("selected_page_id")
            result = await publish_to_facebook(
                video_path,
                final_caption,
                token_data,
                page_id
            )
        else:
            result = PlatformResult(
                platform=platform,
                success=False,
                error_message=f"Unknown platform: {platform}"
            )
        
        ctx.platform_results.append(result)
        
        if result.success:
            logger.info(f"Published to {platform}: {result.platform_video_id}")
        else:
            logger.warning(f"Failed to publish to {platform}: {result.error_message}")
    
    return ctx
