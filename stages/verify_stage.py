"""
UploadM8 Verify Stage
======================
Background verification loop that polls platform APIs to confirm
published videos actually went live (Step B confirmation).

The publish stage records "accepted" status (Step A). This stage
asynchronously verifies that videos actually appear on platforms.

Exports: run_verification_loop(db_pool, shutdown_event, redis_client=None)
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import asyncpg
import httpx

from services.worker_leader_lock import acquire_leader_lock, release_leader_lock

from . import db as db_stage
from .notify_stage import notify_user_publish_confirmed, notify_user_publish_rejected
from .publish_stage import decrypt_token, init_enc_keys

logger = logging.getLogger("uploadm8-worker")


def _tiktok_web_video_url(video_id: str, username: Optional[str]) -> str:
    """TikTok often 404s for www.tiktok.com/video/{id} without the creator handle."""
    v = str(video_id or "").strip()
    if not v:
        return ""
    u = (username or "").strip().lstrip("@")
    if u:
        return f"https://www.tiktok.com/@{u}/video/{v}"
    return f"https://www.tiktok.com/video/{v}"


VERIFY_INTERVAL_SECONDS = int(os.environ.get("VERIFY_INTERVAL_SECONDS", "60"))
VERIFY_MAX_AGE_HOURS = int(os.environ.get("VERIFY_MAX_AGE_HOURS", "24"))
VERIFY_LOCK_TTL_SECONDS = int(
    os.environ.get(
        "WORKER_LEADER_LOCK_TTL_VERIFY_SEC",
        str(max(120, VERIFY_INTERVAL_SECONDS * 4)),
    )
)

# Retryable verify outcomes must stay "pending" so load_pending_verifications
# keeps polling. Terminal outcomes stop the loop.
_TERMINAL_VERIFY_STATUSES = frozenset({"confirmed", "rejected", "failed"})


def _is_terminal_verify_status(status: Optional[str]) -> bool:
    return str(status or "").strip().lower() in _TERMINAL_VERIFY_STATUSES


def _next_verify_status(
    platform: str,
    raw_status: str,
    *,
    has_video_id: bool,
) -> str:
    """
    Map platform API outcomes to a durable verify_status.

    ``unknown`` / missing-token blips must NOT park the attempt forever —
    that left TikTok rows stuck on "Awaiting confirmation" in the UI.
    """
    plat = str(platform or "").strip().lower()
    status = str(raw_status or "").strip().lower()

    if status == "rejected" or status == "failed":
        return status if status in _TERMINAL_VERIFY_STATUSES else "rejected"

    if plat == "tiktok":
        # TikTok Step B is only done when PUBLISH_COMPLETE *and* we have video_id
        # (sync-analytics / UI need the real id, not publish_id).
        if status == "confirmed" and has_video_id:
            return "confirmed"
        return "pending"

    if plat == "youtube":
        if status == "confirmed":
            return "confirmed"
        if status == "pending":
            return "pending"
        # unknown → keep polling within the age window
        return "pending"

    # Instagram / Facebook: publish already stores media/post id as platform_post_id.
    if has_video_id:
        return "confirmed"
    return "pending"


def _tiktok_items_to_update(pr_list: Any, publish_id: Optional[str]) -> list:
    """Pick TikTok platform_results rows to stamp with the confirmed video_id."""
    if not isinstance(pr_list, list):
        return []
    tiktoks = [
        i
        for i in pr_list
        if isinstance(i, dict) and str(i.get("platform") or "").strip().lower() == "tiktok"
    ]
    if not tiktoks:
        return []
    pub = str(publish_id or "").strip()
    if pub:
        by_pub = [i for i in tiktoks if str(i.get("publish_id") or "").strip() == pub]
        if by_pub:
            return by_pub
    awaiting = [
        i
        for i in tiktoks
        if not str(i.get("platform_video_id") or i.get("video_id") or "").strip()
    ]
    return awaiting or tiktoks[:1]


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

    prev_verify = str(attempt.get("verify_status") or "").strip().lower()
    tiktok_post_url: Optional[str] = None

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
        # Token missing/revoked — keep pending so reconnect can finish confirmation.
        logger.debug("Verify %s/%s: no token — leave pending", platform, attempt_id)
        if prev_verify != "pending":
            await db_stage.update_publish_attempt_verified(db_pool, attempt_id, "pending")
        return

    # Decrypt if needed
    try:
        token_data = decrypt_token(token_data) or token_data
    except Exception:
        pass

    # Platform-specific verification
    raw_status = "unknown"
    tiktok_video_id: Optional[str] = None

    if platform == "tiktok" and publish_id:
        raw_status, tiktok_video_id = await verify_tiktok(publish_id, token_data)
    elif platform == "youtube" and platform_post_id:
        raw_status = await verify_youtube(platform_post_id, token_data)
    elif platform in ("instagram", "facebook") and platform_post_id:
        # Publish already stored the media/post id — treat as confirmed live.
        raw_status = "confirmed"
    elif platform in ("instagram", "facebook"):
        raw_status = "pending"
    else:
        raw_status = "unknown"

    has_video_id = bool(
        (tiktok_video_id and str(tiktok_video_id).strip())
        or (platform_post_id and str(platform_post_id).strip() and platform != "tiktok")
    )
    verify_status = _next_verify_status(
        platform, raw_status, has_video_id=has_video_id
    )

    # Persist status (pending stays pending so the loop keeps polling).
    if verify_status != prev_verify or _is_terminal_verify_status(verify_status):
        await db_stage.update_publish_attempt_verified(db_pool, attempt_id, verify_status)
    logger.debug(
        "Verify %s/%s: raw=%s → %s video_id=%s",
        platform,
        attempt_id,
        raw_status,
        verify_status,
        bool(tiktok_video_id),
    )

    # When TikTok confirms, save the real video_id back into platform_results on the uploads row.
    # The initial publish only returns a publish_id — the video_id is only available after
    # PUBLISH_COMPLETE.  Without it, sync-analytics can't query TikTok metrics.
    if platform == "tiktok" and verify_status == "confirmed" and tiktok_video_id:
        upload_id = str(attempt.get("upload_id", ""))
        if upload_id:
            try:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT platform_results FROM uploads WHERE id = $1", upload_id
                    )
                    if row:
                        import json as _json
                        pr = row["platform_results"]
                        pr_list = pr
                        for _ in range(4):
                            if isinstance(pr_list, str):
                                try:
                                    pr_list = _json.loads(pr_list)
                                except Exception:
                                    pr_list = []
                                    break
                            else:
                                break
                        if not isinstance(pr_list, list):
                            pr_list = []
                        targets = _tiktok_items_to_update(pr_list, publish_id)
                        if targets:
                            updated = False
                            for item in targets:
                                item["platform_video_id"] = tiktok_video_id
                                item["video_id"] = tiktok_video_id
                                item["verify_status"] = "confirmed"
                                uname = (item.get("account_username") or "").strip().lstrip("@")
                                if not uname and isinstance(token_data, dict):
                                    uname = (
                                        str(token_data.get("_account_username") or "")
                                        .strip()
                                        .lstrip("@")
                                    )
                                tt_url = _tiktok_web_video_url(tiktok_video_id, uname)
                                item["platform_url"] = tt_url
                                item["url"] = tt_url
                                tiktok_post_url = tt_url
                                updated = True
                            if updated:
                                await conn.execute(
                                    "UPDATE uploads SET platform_results = $1::jsonb, updated_at = NOW() WHERE id = $2",
                                    # Pass list — codecs that always dumps() must not see a pre-dumped str.
                                    pr_list,
                                    upload_id,
                                )
                                logger.info(
                                    f"Saved TikTok video_id={tiktok_video_id} back to "
                                    f"platform_results for upload={upload_id}"
                                )
            except Exception as e:
                logger.warning(f"Could not save TikTok video_id to platform_results: {e}")

    if (
        db_pool
        and verify_status == "confirmed"
        and prev_verify != "confirmed"
        and user_id
    ):
        uid_upload = str(attempt.get("upload_id", ""))
        if uid_upload and platform in ("tiktok", "youtube"):
            post_url = ""
            if platform == "youtube" and platform_post_id:
                vid = str(platform_post_id).strip()
                if vid:
                    post_url = f"https://www.youtube.com/shorts/{vid}"
            elif platform == "tiktok" and tiktok_video_id:
                post_url = (tiktok_post_url or "").strip() or _tiktok_web_video_url(
                    tiktok_video_id,
                    (
                        str((token_data or {}).get("_account_username") or "").strip().lstrip("@")
                        if isinstance(token_data, dict)
                        else None
                    ),
                )
            try:
                await notify_user_publish_confirmed(
                    db_pool,
                    user_id=user_id,
                    upload_id=uid_upload,
                    platform=platform,
                    post_url=post_url,
                )
            except Exception as e:
                logger.warning("publish_confirmed notify failed upload=%s: %s", uid_upload, e)

    if (
        db_pool
        and verify_status == "rejected"
        and prev_verify != "rejected"
        and user_id
    ):
        uid_upload = str(attempt.get("upload_id", ""))
        if uid_upload and platform in ("tiktok", "youtube"):
            if platform == "tiktok":
                detail = (
                    "TikTok reported a terminal publish failure "
                    "(FAILED or UPLOAD_ERROR), or another hard reject state."
                )
            else:
                detail = (
                    "YouTube reports the Short as failed, rejected, deleted, "
                    "or no longer accessible via your channel."
                )
            try:
                await notify_user_publish_rejected(
                    db_pool,
                    user_id=user_id,
                    upload_id=uid_upload,
                    platform=platform,
                    detail=detail,
                )
            except Exception as e:
                logger.warning("publish_rejected notify failed upload=%s: %s", uid_upload, e)

    if (
        db_pool
        and verify_status == "failed"
        and prev_verify != "failed"
        and user_id
    ):
        uid_upload = str(attempt.get("upload_id", ""))
        if uid_upload and platform in ("tiktok", "youtube"):
            if platform == "tiktok":
                detail = (
                    "TikTok verification returned FAILED or could not confirm the publish "
                    "(timeout, API error, or publish_id no longer valid)."
                )
            else:
                detail = (
                    "YouTube verification could not confirm the Short is live "
                    "(API error, deleted video, or verification timeout)."
                )
            try:
                await notify_user_publish_rejected(
                    db_pool,
                    user_id=user_id,
                    upload_id=uid_upload,
                    platform=platform,
                    detail=detail,
                    verify_outcome="failed",
                )
            except Exception as e:
                logger.warning("publish_verify_failed notify failed upload=%s: %s", uid_upload, e)


async def run_verification_loop(
    db_pool: asyncpg.Pool,
    shutdown_event: asyncio.Event,
    redis_client: Optional[Any] = None,
):
    """
    Background loop that polls pending publish attempts and verifies them.

    Runs alongside the main job processing loop. Exits when shutdown_event is set.
    With redis_client and leader locks enabled, only one worker replica runs each tick.
    """
    logger.info("Verification loop started")
    init_enc_keys()

    while not shutdown_event.is_set():
        lock_token = await acquire_leader_lock(
            redis_client,
            "verification",
            VERIFY_LOCK_TTL_SECONDS,
        )
        if lock_token is None:
            logger.debug("Verification skipping cycle (peer holds leader lock)")
            try:
                await asyncio.wait_for(
                    asyncio.shield(shutdown_event.wait()),
                    timeout=VERIFY_INTERVAL_SECONDS,
                )
                break
            except asyncio.TimeoutError:
                continue
        try:
            try:
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
                        await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Verification loop error: {e}")
        finally:
            await release_leader_lock(redis_client, "verification", lock_token)

        try:
            await asyncio.wait_for(
                asyncio.shield(shutdown_event.wait()),
                timeout=VERIFY_INTERVAL_SECONDS,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("Verification loop stopped")
