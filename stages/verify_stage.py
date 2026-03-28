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


async def verify_tiktok(publish_id: str, token_data: dict):
    """
    Check TikTok publish status.
    Returns: (status_str, video_id_or_None)
      status: 'confirmed', 'rejected', 'pending', or 'unknown'
      video_id: the real TikTok video_id when PUBLISH_COMPLETE (None otherwise)

    TikTok's status response includes published_element.video_id once live.
    We capture it so sync-analytics can query per-video metrics later.
    """
    access_token = token_data.get("access_token")
    if not access_token or not publish_id:
        return "unknown", None

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
                return "unknown", None

            data = resp.json().get("data", {})
            status = data.get("status", "").upper()
            # TikTok includes the real video_id once publishing is complete
            video_id = (
                data.get("published_element", {}).get("video_id")
                or data.get("video_id")
                or None
            )
            if video_id:
                video_id = str(video_id)

            if status == "PUBLISH_COMPLETE":
                return "confirmed", video_id
            elif status in ("FAILED", "UPLOAD_ERROR"):
                return "rejected", None
            elif status in ("PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD", "SENDING_TO_USER_INBOX"):
                return "pending", None
            else:
                return "unknown", None

    except Exception as e:
        logger.debug(f"TikTok verify failed: {e}")
        return "unknown", None


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
    tiktok_video_id: Optional[str] = None

    if platform == "tiktok" and publish_id:
        verify_status, tiktok_video_id = await verify_tiktok(publish_id, token_data)
    elif platform == "youtube" and platform_post_id:
        verify_status = await verify_youtube(platform_post_id, token_data)
    else:
        # Instagram/Facebook verification not yet implemented
        verify_status = "unknown"

    # Update publish_attempts row
    await db_stage.update_publish_attempt_verified(db_pool, attempt_id, verify_status)
    logger.debug(f"Verify {platform}/{attempt_id}: {verify_status}")

    # When TikTok confirms, save the real video_id back into platform_results on the uploads row.
    # The initial publish only returns a publish_id — the video_id is only available after
    # PUBLISH_COMPLETE.  Without it, sync-analytics can't query TikTok metrics.
    if platform == "tiktok" and verify_status == "confirmed" and tiktok_video_id:
        upload_id = str(attempt.get("upload_id", ""))
        attempt_publish_id = str(attempt.get("publish_id", ""))
        if upload_id:
            try:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT platform_results FROM uploads WHERE id = $1", upload_id
                    )
                    if row:
                        import json as _json
                        pr = row["platform_results"]
                        pr_list = _json.loads(pr) if isinstance(pr, str) else (pr or [])
                        if isinstance(pr_list, list):
                            updated = False
                            for item in pr_list:
                                if isinstance(item, dict) and item.get("platform") == "tiktok":
                                    matched = False
                                    if attempt_publish_id and str(item.get("publish_id", "")) == attempt_publish_id:
                                        matched = True
                                    elif not attempt_publish_id:
                                        matched = True
                                    if matched:
                                        uname = (item.get("account_username") or "").strip().lstrip("@")
                                        if uname:
                                            tiktok_url = f"https://www.tiktok.com/@{uname}/video/{tiktok_video_id}"
                                        else:
                                            tiktok_url = f"https://www.tiktok.com/video/{tiktok_video_id}"
                                        item["platform_video_id"] = tiktok_video_id
                                        item["video_id"] = tiktok_video_id
                                        item["platform_url"] = tiktok_url
                                        item["url"] = tiktok_url
                                        updated = True
                                        break
                            if updated:
                                await conn.execute(
                                    "UPDATE uploads SET platform_results = $1::jsonb, updated_at = NOW() WHERE id = $2",
                                    _json.dumps(pr_list), upload_id
                                )
                                logger.info(
                                    f"Saved TikTok video_id={tiktok_video_id} back to "
                                    f"platform_results for upload={upload_id}"
                                )
            except Exception as e:
                logger.warning(f"Could not save TikTok video_id to platform_results: {e}")


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
