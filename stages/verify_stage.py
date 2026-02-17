"""
UploadM8 Verify Stage
======================
Background verification loop that polls platform APIs to confirm
published videos actually went live (Step B confirmation).

The publish stage records "accepted" status (Step A). This stage
asynchronously verifies that videos actually appear on platforms.

Exports: run_verification_loop(db_pool, shutdown_event)
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any

import asyncpg
import httpx

from . import db as db_stage
from .publish_stage import decrypt_token, init_enc_keys

logger = logging.getLogger("uploadm8-worker")

VERIFY_INTERVAL_SECONDS = int(os.environ.get("VERIFY_INTERVAL_SECONDS", "60"))
VERIFY_MAX_AGE_HOURS = int(os.environ.get("VERIFY_MAX_AGE_HOURS", "24"))


async def verify_tiktok(publish_id: str, token_data: dict) -> str:
    """
    Check TikTok publish status.
    Returns: 'confirmed', 'rejected', 'pending', or 'unknown'.
    """
    access_token = token_data.get("access_token")
    if not access_token or not publish_id:
        return "unknown"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={"publish_id": publish_id},
            )

            if resp.status_code != 200:
                return "unknown"

            data = resp.json().get("data", {})
            status = data.get("status", "").upper()

            if status == "PUBLISH_COMPLETE":
                return "confirmed"
            elif status in ("FAILED", "UPLOAD_ERROR"):
                return "rejected"
            elif status in ("PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD", "SENDING_TO_USER_INBOX"):
                return "pending"
            else:
                return "unknown"

    except Exception as e:
        logger.debug(f"TikTok verify failed: {e}")
        return "unknown"


async def verify_youtube(video_id: str, token_data: dict) -> str:
    """
    Check YouTube video status.
    Returns: 'confirmed', 'rejected', 'pending', or 'unknown'.
    """
    access_token = token_data.get("access_token")
    if not access_token or not video_id:
        return "unknown"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "id": video_id,
                    "part": "status",
                },
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if resp.status_code != 200:
                return "unknown"

            items = resp.json().get("items", [])
            if not items:
                return "rejected"

            upload_status = items[0].get("status", {}).get("uploadStatus", "")
            if upload_status == "processed":
                return "confirmed"
            elif upload_status in ("failed", "rejected", "deleted"):
                return "rejected"
            elif upload_status == "uploaded":
                return "pending"
            else:
                return "unknown"

    except Exception as e:
        logger.debug(f"YouTube verify failed: {e}")
        return "unknown"


async def verify_single_attempt(
    db_pool: asyncpg.Pool,
    attempt: dict,
) -> None:
    """Verify a single publish attempt and update the DB."""
    platform = attempt.get("platform", "")
    attempt_id = str(attempt.get("id", ""))
    publish_id = attempt.get("publish_id")
    platform_post_id = attempt.get("platform_post_id")
    user_id = str(attempt.get("user_id", ""))

    if not attempt_id:
        return

    # Load platform token for this user
    platform_to_db_key = {
        "tiktok": "tiktok",
        "youtube": "google",
        "instagram": "meta",
        "facebook": "meta",
    }
    db_key = platform_to_db_key.get(platform, platform)

    token_data = None
    try:
        token_data = await db_stage.load_platform_token(db_pool, user_id, db_key)
    except Exception:
        pass

    if not token_data:
        # Can't verify without token — mark unknown
        await db_stage.update_publish_attempt_verified(db_pool, attempt_id, "unknown")
        return

    # Decrypt if needed
    try:
        token_data = decrypt_token(token_data) or token_data
    except Exception:
        pass

    # Platform-specific verification
    verify_status = "unknown"

    if platform == "tiktok" and publish_id:
        verify_status = await verify_tiktok(publish_id, token_data)
    elif platform == "youtube" and platform_post_id:
        verify_status = await verify_youtube(platform_post_id, token_data)
    else:
        # Instagram/Facebook verification not yet implemented
        verify_status = "unknown"

    # Update DB
    await db_stage.update_publish_attempt_verified(db_pool, attempt_id, verify_status)
    logger.debug(f"Verify {platform}/{attempt_id}: {verify_status}")


async def run_verification_loop(
    db_pool: asyncpg.Pool,
    shutdown_event: asyncio.Event,
):
    """
    Background loop that polls pending publish attempts and verifies them.

    Runs alongside the main job processing loop. Exits when shutdown_event is set.
    """
    logger.info("Verification loop started")
    init_enc_keys()

    while not shutdown_event.is_set():
        try:
            # Load pending verifications
            pending = await db_stage.load_pending_verifications(db_pool, limit=50)

            if pending:
                logger.info(f"Verifying {len(pending)} publish attempts")
                for attempt in pending:
                    if shutdown_event.is_set():
                        break
                    try:
                        await verify_single_attempt(db_pool, attempt)
                    except Exception as e:
                        logger.warning(f"Verify attempt failed: {e}")
                    # Small delay between API calls to avoid rate limits
                    await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"Verification loop error: {e}")

        # Wait for next cycle or shutdown
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=VERIFY_INTERVAL_SECONDS,
            )
            break  # shutdown_event was set
        except asyncio.TimeoutError:
            pass  # Normal timeout — loop again

    logger.info("Verification loop stopped")
