"""
UploadM8 Verify Stage
======================
Background verification loop that polls platform APIs to confirm
published videos actually went live (Step B confirmation).

The publish stage records "accepted" status (Step A). This stage
asynchronously verifies that videos actually appear on platforms.

Supported platforms (Graph / Google / TikTok APIs):
  - TikTok: ``publish_id`` → status fetch; captures ``video_id`` when complete.
  - YouTube: ``platform_post_id`` (video id) → videos.list status.
  - Instagram: ``platform_post_id`` (media id) → Graph ``/{media_id}``.
  - Facebook: ``platform_post_id`` (video id) → Graph ``/{video_id}``.

Verify status semantics
-----------------------
``confirmed`` / ``rejected`` / ``pending`` are explicit platform outcomes.
``unknown`` is **not** "unimplemented"; it means we could not classify yet,
e.g. missing token or post id, unsupported platform for this row, HTTP or
parse failure, or an API response shape we do not map. Meta 400/403/404 right
after publish are treated as ``pending`` (transient) where applicable.

Exports: run_verification_loop(db_pool, shutdown_event)
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any, Tuple

import asyncpg
import httpx

from . import db as db_stage
from .platform_tokens import platform_tokens_db_key
from .publish_stage import decrypt_token, init_enc_keys, META_API_VERSION
from .safe_parse import json_list

logger = logging.getLogger("uploadm8-worker")

VERIFY_INTERVAL_SECONDS = int(os.environ.get("VERIFY_INTERVAL_SECONDS", "60"))
VERIFY_HTTP_TIMEOUT_SECONDS = 20
VERIFY_TIKTOK_TIMEOUT_SECONDS = 15
VERIFY_YOUTUBE_TIMEOUT_SECONDS = 15

STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
STATUS_PENDING = "pending"
STATUS_UNKNOWN = "unknown"

_GRAPH_TRANSIENT_PENDING_HTTP = {400, 403, 404}

# HTTP client + JSON shape failures during verify (never mask asyncio cancellation).
_VERIFY_CLIENT_ERRORS = (
    httpx.RequestError,
    httpx.HTTPError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)

_VERIFY_ATTEMPT_ERRORS = _VERIFY_CLIENT_ERRORS + (
    asyncpg.PostgresError,
    asyncpg.InterfaceError,
    OSError,
    TimeoutError,
    TypeError,
    ValueError,
    AttributeError,
    RuntimeError,
)
_IG_CONFIRMED = {"FINISHED", "PUBLISHED", "READY", "SUCCESS"}
_IG_REJECTED = {"ERROR", "EXPIRED", "FAILED", "REJECTED"}
_IG_PENDING = {"IN_PROGRESS", "PROCESSING", "PENDING", "UPLOADING"}
_TT_REJECTED = {"FAILED", "UPLOAD_ERROR"}
_TT_PENDING = {"PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD", "SENDING_TO_USER_INBOX"}


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
        return STATUS_UNKNOWN, None

    try:
        async with httpx.AsyncClient(timeout=VERIFY_TIKTOK_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={"publish_id": publish_id},
            )

            if resp.status_code != 200:
                return STATUS_UNKNOWN, None

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
                return STATUS_CONFIRMED, video_id
            elif status in _TT_REJECTED:
                return STATUS_REJECTED, None
            elif status in _TT_PENDING:
                return STATUS_PENDING, None
            else:
                return STATUS_UNKNOWN, None

    except asyncio.CancelledError:
        raise
    except _VERIFY_CLIENT_ERRORS as e:
        logger.debug("TikTok verify failed: %s", e)
        return STATUS_UNKNOWN, None


async def verify_youtube(video_id: str, token_data: dict) -> str:
    """
    Check YouTube video status.
    Returns: 'confirmed', 'rejected', 'pending', or 'unknown'.
    """
    access_token = token_data.get("access_token")
    if not access_token or not video_id:
        return STATUS_UNKNOWN

    try:
        async with httpx.AsyncClient(timeout=VERIFY_YOUTUBE_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "id": video_id,
                    "part": "status",
                },
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if resp.status_code != 200:
                return STATUS_UNKNOWN

            items = resp.json().get("items", [])
            if not items:
                return STATUS_REJECTED

            upload_status = items[0].get("status", {}).get("uploadStatus", "")
            if upload_status == "processed":
                return STATUS_CONFIRMED
            elif upload_status in ("failed", "rejected", "deleted"):
                return STATUS_REJECTED
            elif upload_status == "uploaded":
                return STATUS_PENDING
            else:
                return STATUS_UNKNOWN

    except asyncio.CancelledError:
        raise
    except _VERIFY_CLIENT_ERRORS as e:
        logger.debug("YouTube verify failed: %s", e)
        return STATUS_UNKNOWN


def _interpret_instagram_verify_payload(payload: Dict[str, Any], media_id: str) -> Tuple[str, Optional[str]]:
    """
    Interpret Instagram Graph media lookup payload.

    Returns:
      (verify_status, platform_url_or_none)
    """
    if not isinstance(payload, dict):
        return STATUS_UNKNOWN, None
    if payload.get("error"):
        return STATUS_UNKNOWN, None

    status_code = str(payload.get("status_code") or payload.get("status") or "").upper()
    permalink = payload.get("permalink") or payload.get("permalink_url")
    media_type = str(payload.get("media_type") or payload.get("media_product_type") or "").upper()

    if status_code in _IG_CONFIRMED:
        return STATUS_CONFIRMED, permalink
    if status_code in _IG_REJECTED:
        return STATUS_REJECTED, None
    if status_code in _IG_PENDING:
        return STATUS_PENDING, None

    # If the media object resolves and has a known media type, treat as confirmed.
    if payload.get("id") and media_type in ("REELS", "VIDEO", "CAROUSEL_ALBUM", "IMAGE"):
        return STATUS_CONFIRMED, permalink

    # Payload resolved but didn't contain explicit status.
    if payload.get("id"):
        return STATUS_PENDING, permalink
    return STATUS_UNKNOWN, None


async def verify_instagram(media_id: str, token_data: dict) -> Tuple[str, Optional[str]]:
    """
    Check Instagram media status by querying Graph API media object.
    Returns: (verify_status, platform_url_or_none)
    """
    access_token = token_data.get("access_token")
    if not access_token or not media_id:
        return STATUS_UNKNOWN, None

    try:
        async with httpx.AsyncClient(timeout=VERIFY_HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                f"https://graph.facebook.com/{META_API_VERSION}/{media_id}",
                params={
                    "access_token": access_token,
                    "fields": "id,permalink,media_type,media_product_type,status,status_code",
                },
            )
            if resp.status_code != 200:
                # Graph can return 4xx transiently right after publish acceptance.
                if resp.status_code in _GRAPH_TRANSIENT_PENDING_HTTP:
                    return STATUS_PENDING, None
                return STATUS_UNKNOWN, None
            return _interpret_instagram_verify_payload(resp.json(), media_id)
    except asyncio.CancelledError:
        raise
    except _VERIFY_CLIENT_ERRORS as e:
        logger.debug("Instagram verify failed: %s", e)
        return STATUS_UNKNOWN, None


def _interpret_facebook_verify_payload(payload: Dict[str, Any], video_id: str) -> Tuple[str, Optional[str]]:
    """
    Interpret Facebook video lookup payload.

    Returns:
      (verify_status, platform_url_or_none)
    """
    if not isinstance(payload, dict):
        return STATUS_UNKNOWN, None
    if payload.get("error"):
        return STATUS_UNKNOWN, None

    status_val = ""
    status_obj = payload.get("status")
    if isinstance(status_obj, dict):
        status_val = str(
            status_obj.get("video_status")
            or status_obj.get("publishing_phase")
            or status_obj.get("processing_progress")
            or ""
        ).lower()
    elif status_obj is not None:
        status_val = str(status_obj).lower()

    permalink = payload.get("permalink_url")
    if not permalink and video_id:
        permalink = f"https://www.facebook.com/watch/?v={video_id}"

    if any(k in status_val for k in ("ready", "live", "published", "complete", "completed")):
        return STATUS_CONFIRMED, permalink
    if any(k in status_val for k in ("error", "failed", "rejected", "blocked")):
        return STATUS_REJECTED, None
    if any(k in status_val for k in ("processing", "upload", "transcod", "pending")):
        return STATUS_PENDING, None

    if payload.get("id"):
        return STATUS_CONFIRMED, permalink
    return STATUS_UNKNOWN, None


async def verify_facebook(video_id: str, token_data: dict) -> Tuple[str, Optional[str]]:
    """
    Check Facebook video status by querying Graph API video object.
    Returns: (verify_status, platform_url_or_none)
    """
    access_token = token_data.get("access_token")
    if not access_token or not video_id:
        return STATUS_UNKNOWN, None

    try:
        async with httpx.AsyncClient(timeout=VERIFY_HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                f"https://graph.facebook.com/{META_API_VERSION}/{video_id}",
                params={
                    "access_token": access_token,
                    "fields": "id,status,permalink_url",
                },
            )
            if resp.status_code != 200:
                if resp.status_code in _GRAPH_TRANSIENT_PENDING_HTTP:
                    return STATUS_PENDING, None
                return STATUS_UNKNOWN, None
            return _interpret_facebook_verify_payload(resp.json(), video_id)
    except asyncio.CancelledError:
        raise
    except _VERIFY_CLIENT_ERRORS as e:
        logger.debug("Facebook verify failed: %s", e)
        return STATUS_UNKNOWN, None


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

    # Load platform token for this user (same DB slugs as OAuth + publish_stage)
    db_key = platform_tokens_db_key(platform)

    token_data = None
    try:
        token_data = await db_stage.load_platform_token(db_pool, user_id, db_key)
    except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError) as e:
        logger.debug("Verify %s/%s: token load failed: %s", platform, attempt_id, e)

    if not token_data:
        # Can't verify without token — mark unknown
        await db_stage.update_publish_attempt_verified(db_pool, attempt_id, STATUS_UNKNOWN)
        return

    # Decrypt if needed (decrypt_token is best-effort; returns None on failure)
    token_data = decrypt_token(token_data) or token_data

    # Robust guard: verification clients expect a token dict with access_token.
    if not isinstance(token_data, dict) or not str(token_data.get("access_token", "")).strip():
        logger.debug(f"Verify {platform}/{attempt_id}: invalid token payload")
        await db_stage.update_publish_attempt_verified(db_pool, attempt_id, STATUS_UNKNOWN)
        return

    # Platform-specific verification
    verify_status = STATUS_UNKNOWN
    tiktok_video_id: Optional[str] = None
    verified_platform_url: Optional[str] = None

    if platform == "tiktok" and publish_id:
        verify_status, tiktok_video_id = await verify_tiktok(publish_id, token_data)
    elif platform == "youtube" and platform_post_id:
        verify_status = await verify_youtube(platform_post_id, token_data)
    elif platform == "instagram" and platform_post_id:
        verify_status, verified_platform_url = await verify_instagram(str(platform_post_id), token_data)
    elif platform == "facebook" and platform_post_id:
        verify_status, verified_platform_url = await verify_facebook(str(platform_post_id), token_data)
    else:
        # Missing IDs or unsupported platform for verification
        verify_status = STATUS_UNKNOWN

    # Update publish_attempts row
    await db_stage.update_publish_attempt_verified(
        db_pool,
        attempt_id,
        verify_status,
        platform_url=verified_platform_url,
    )
    logger.debug(f"Verify {platform}/{attempt_id}: {verify_status}")

    # When TikTok confirms, save the real video_id back into platform_results on the uploads row.
    # The initial publish only returns a publish_id — the video_id is only available after
    # PUBLISH_COMPLETE.  Without it, sync-analytics can't query TikTok metrics.
    if platform == "tiktok" and verify_status == STATUS_CONFIRMED and tiktok_video_id:
        upload_id = str(attempt.get("upload_id", ""))
        attempt_publish_id = str(attempt.get("publish_id", ""))
        if upload_id:
            try:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT platform_results FROM uploads WHERE id = $1", upload_id
                    )
                    if row:
                        pr = row["platform_results"]
                        pr_list = json_list(
                            pr, default=[], context="uploads.platform_results"
                        )
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
                                json.dumps(pr_list), upload_id
                            )
                            logger.info(
                                f"Saved TikTok video_id={tiktok_video_id} back to "
                                f"platform_results for upload={upload_id}"
                            )
            except asyncio.CancelledError:
                raise
            except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError, TypeError, ValueError) as e:
                logger.warning("Could not save TikTok video_id to platform_results: %s", e)

    # Instagram / Facebook: Graph returns the canonical permalink on verify — patch uploads.platform_results
    # so dashboard links and sync-analytics match the live post (publish used to synthesize bad IG URLs).
    plat_key = str(platform or "").lower()
    if plat_key in ("instagram", "facebook") and verify_status == STATUS_CONFIRMED and platform_post_id:
        upload_id = str(attempt.get("upload_id", ""))
        pid = str(platform_post_id)
        publish_id_attempt = str(attempt.get("publish_id") or "")
        if upload_id:
            try:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT platform_results FROM uploads WHERE id = $1", upload_id
                    )
                    if row:
                        pr = row["platform_results"]
                        pr_list = json_list(
                            pr, default=[], context="uploads.platform_results"
                        )
                        updated = False
                        for item in pr_list:
                            if not isinstance(item, dict) or str(item.get("platform") or "").lower() != plat_key:
                                continue
                            matched = (
                                str(item.get("platform_video_id") or "") == pid
                                or str(item.get("video_id") or "") == pid
                            )
                            if not matched and publish_id_attempt and str(
                                item.get("publish_id") or ""
                            ) == publish_id_attempt:
                                matched = True
                            if not matched:
                                continue
                            if verified_platform_url:
                                item["platform_url"] = verified_platform_url
                                item["url"] = verified_platform_url
                            item["platform_video_id"] = pid
                            item["video_id"] = pid
                            updated = True
                            break
                        if updated:
                            await conn.execute(
                                "UPDATE uploads SET platform_results = $1::jsonb, updated_at = NOW() WHERE id = $2",
                                json.dumps(pr_list),
                                upload_id,
                            )
                            logger.info(
                                "Saved %s permalink/video_id to platform_results for upload=%s",
                                plat_key,
                                upload_id,
                            )
            except asyncio.CancelledError:
                raise
            except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError, TypeError, ValueError) as e:
                logger.warning("Could not save %s platform_url to platform_results: %s", plat_key, e)


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
                    except asyncio.CancelledError:
                        raise
                    except _VERIFY_ATTEMPT_ERRORS as e:
                        logger.warning("Verify attempt failed: %s", e)
                    # Small delay between API calls to avoid rate limits
                    await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            raise
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError) as e:
            logger.warning("Verification loop error: %s", e)

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
