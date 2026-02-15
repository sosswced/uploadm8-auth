"""
UploadM8 Verify Stage — Platform Post Verification
====================================================
Polls platform APIs to confirm posts are actually live.

This is the "hard confirmation" layer that turns "API accepted" into "confirmed live".

Backoff schedule: 10s → 30s → 60s → 120s → 300s → mark unknown
Max ~10-15 minutes of verification attempts per publish.

Verification strategies per platform:
  - TikTok:    GET /v2/post/publish/status/fetch/ with publish_id
  - YouTube:   GET /v3/videos?id={video_id}&part=status
  - Instagram: GET /{container_id}?fields=status_code (future)
  - Facebook:  GET /{video_id}?fields=status (future)
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
import asyncpg

from . import db as db_stage
from .publish_stage import decrypt_token_blob, init_enc_keys

logger = logging.getLogger("uploadm8-worker")

# Max verification attempts before marking as unknown
MAX_VERIFY_ATTEMPTS = 7


# ============================================================================
# PLATFORM-SPECIFIC VERIFICATION FUNCTIONS
# ============================================================================

async def verify_tiktok(
    platform_post_id: str,
    token_data: dict
) -> dict:
    """
    Check TikTok publish status.
    Returns: {"status": "confirmed"|"rejected"|"pending", "payload": {...}}
    """
    access_token = token_data.get("access_token")
    if not access_token:
        return {"status": "pending", "payload": {"error": "no_token"}}
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json={"publish_id": platform_post_id}
            )
            
            if resp.status_code != 200:
                return {"status": "pending", "payload": {"http_status": resp.status_code, "body": resp.text[:500]}}
            
            data = resp.json()
            publish_status = data.get("data", {}).get("status", "").upper()
            
            # TikTok statuses: PUBLISH_COMPLETE, FAILED, PROCESSING_UPLOAD, PROCESSING_DOWNLOAD, SENDING_TO_USER_INBOX
            if publish_status == "PUBLISH_COMPLETE":
                return {"status": "confirmed", "payload": data}
            elif publish_status == "FAILED":
                return {"status": "rejected", "payload": data}
            else:
                # Still processing
                return {"status": "pending", "payload": data}
    
    except Exception as e:
        logger.warning(f"TikTok verify error: {e}")
        return {"status": "pending", "payload": {"error": str(e)}}


async def verify_youtube(
    platform_post_id: str,
    token_data: dict
) -> dict:
    """
    Check YouTube video status.
    Returns: {"status": "confirmed"|"rejected"|"pending", "payload": {...}, "url": "..."}
    """
    access_token = token_data.get("access_token")
    if not access_token:
        return {"status": "pending", "payload": {"error": "no_token"}}
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "id": platform_post_id,
                    "part": "status,snippet",
                },
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if resp.status_code != 200:
                return {"status": "pending", "payload": {"http_status": resp.status_code, "body": resp.text[:500]}}
            
            data = resp.json()
            items = data.get("items", [])
            
            if not items:
                # Video not found — could be deleted or still processing
                return {"status": "pending", "payload": data}
            
            video = items[0]
            upload_status = video.get("status", {}).get("uploadStatus", "")
            privacy_status = video.get("status", {}).get("privacyStatus", "")
            
            # YouTube statuses: processed, uploaded, failed, rejected, deleted
            if upload_status == "processed":
                url = f"https://youtube.com/shorts/{platform_post_id}"
                return {"status": "confirmed", "payload": data, "url": url}
            elif upload_status in ("failed", "rejected", "deleted"):
                return {"status": "rejected", "payload": data}
            else:
                # Still uploading/processing
                return {"status": "pending", "payload": data}
    
    except Exception as e:
        logger.warning(f"YouTube verify error: {e}")
        return {"status": "pending", "payload": {"error": str(e)}}


async def verify_facebook(
    platform_post_id: str,
    token_data: dict
) -> dict:
    """
    Check Facebook video status.
    Returns: {"status": "confirmed"|"rejected"|"pending", "payload": {...}}
    """
    access_token = token_data.get("access_token")
    if not access_token:
        return {"status": "pending", "payload": {"error": "no_token"}}
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"https://graph.facebook.com/v19.0/{platform_post_id}",
                params={
                    "access_token": access_token,
                    "fields": "status,published,permalink_url,title"
                }
            )
            
            if resp.status_code == 404:
                return {"status": "rejected", "payload": {"http_status": 404, "reason": "not_found"}}
            
            if resp.status_code != 200:
                return {"status": "pending", "payload": {"http_status": resp.status_code, "body": resp.text[:500]}}
            
            data = resp.json()
            status = data.get("status", {})
            published = data.get("published", False)
            
            # Facebook video statuses
            video_status = status.get("video_status", "") if isinstance(status, dict) else str(status)
            
            if published or video_status == "ready":
                url = data.get("permalink_url")
                return {"status": "confirmed", "payload": data, "url": url}
            elif video_status in ("error", "expired"):
                return {"status": "rejected", "payload": data}
            else:
                return {"status": "pending", "payload": data}
    
    except Exception as e:
        logger.warning(f"Facebook verify error: {e}")
        return {"status": "pending", "payload": {"error": str(e)}}


async def verify_instagram(
    platform_post_id: str,
    token_data: dict
) -> dict:
    """
    Check Instagram container/media status.
    Note: Instagram Reels not yet implemented in publish, so this is a placeholder.
    """
    # Instagram verification will be implemented when publish is supported
    return {"status": "pending", "payload": {"note": "Instagram verification not yet implemented"}}


# ============================================================================
# VERIFICATION DISPATCHER
# ============================================================================

VERIFY_FUNCTIONS = {
    "tiktok": verify_tiktok,
    "youtube": verify_youtube,
    "facebook": verify_facebook,
    "instagram": verify_instagram,
}


async def verify_single_attempt(
    pool: asyncpg.Pool,
    attempt: dict
) -> str:
    """
    Verify a single publish attempt. Returns the resulting verify_status.
    
    Args:
        pool: Database pool
        attempt: Dict from get_pending_verifications()
    
    Returns:
        "confirmed", "rejected", "pending" (will retry), or "unknown" (gave up)
    """
    attempt_id = str(attempt["id"])
    platform = attempt["platform"]
    platform_post_id = attempt["platform_post_id"]
    verify_count = attempt["verify_attempts"]
    user_id = str(attempt["user_id"])
    
    # Check if we've exceeded max attempts
    if verify_count >= MAX_VERIFY_ATTEMPTS:
        await db_stage.update_verify_unknown(pool, attempt_id, {"reason": "max_attempts_exceeded"})
        return "unknown"
    
    # Load token for verification API call
    platform_to_db_key = {
        "tiktok": "tiktok",
        "youtube": "google",
        "instagram": "meta",
        "facebook": "meta"
    }
    db_key = platform_to_db_key.get(platform, platform)
    
    token_blob = await db_stage.load_platform_token(pool, user_id, db_key)
    if not token_blob:
        # Can't verify without a token — schedule retry in case token is re-added
        await db_stage.update_verify_retry(pool, attempt_id, verify_count)
        return "pending"
    
    try:
        token_data = decrypt_token_blob(token_blob)
    except Exception as e:
        logger.warning(f"Token decrypt failed during verify: {e}")
        await db_stage.update_verify_retry(pool, attempt_id, verify_count)
        return "pending"
    
    # Initialize encryption keys (idempotent)
    init_enc_keys()
    
    # Call platform-specific verify function
    verify_fn = VERIFY_FUNCTIONS.get(platform)
    if not verify_fn:
        logger.warning(f"No verify function for platform: {platform}")
        await db_stage.update_verify_unknown(pool, attempt_id, {"reason": f"unsupported_platform: {platform}"})
        return "unknown"
    
    result = await verify_fn(platform_post_id, token_data)
    verify_status = result.get("status", "pending")
    payload = result.get("payload")
    url = result.get("url")
    
    if verify_status == "confirmed":
        await db_stage.update_verify_confirmed(pool, attempt_id, payload, url)
        logger.info(f"✅ VERIFIED: {platform} post {platform_post_id} is LIVE")
        return "confirmed"
    
    elif verify_status == "rejected":
        await db_stage.update_verify_rejected(pool, attempt_id, payload)
        logger.warning(f"❌ REJECTED: {platform} post {platform_post_id} was rejected/removed")
        return "rejected"
    
    else:
        # Still pending — schedule next retry with backoff
        await db_stage.update_verify_retry(pool, attempt_id, verify_count)
        logger.debug(f"⏳ PENDING: {platform} post {platform_post_id}, attempt {verify_count + 1}")
        return "pending"


# ============================================================================
# VERIFICATION WORKER LOOP (runs alongside main job loop)
# ============================================================================

async def run_verification_loop(pool: asyncpg.Pool, shutdown_event: asyncio.Event = None):
    """
    Background loop that polls pending verifications and processes them.
    This runs independently of the main job queue.
    
    Call this from worker.py as a background task.
    """
    logger.info("Verification worker started")
    
    while True:
        if shutdown_event and shutdown_event.is_set():
            logger.info("Verification worker shutting down")
            break
        
        try:
            # Get attempts that need verification
            pending = await db_stage.get_pending_verifications(pool, limit=20)
            
            if pending:
                logger.info(f"Processing {len(pending)} pending verifications")
                
                for attempt in pending:
                    try:
                        await verify_single_attempt(pool, attempt)
                    except Exception as e:
                        logger.error(f"Verify error for attempt {attempt.get('id')}: {e}")
                
                # Small delay between batches
                await asyncio.sleep(2)
            else:
                # No pending verifications — wait before checking again
                await asyncio.sleep(10)
        
        except asyncio.CancelledError:
            logger.info("Verification worker cancelled")
            break
        except Exception as e:
            logger.error(f"Verification loop error: {e}")
            await asyncio.sleep(5)
    
    logger.info("Verification worker stopped")
