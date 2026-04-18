"""
UploadM8 Webhook routes -- extracted from app.py.

Handles TikTok and Facebook/Instagram webhook verification and event processing.
"""

import hashlib
import hmac as _hmac
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import core.state
from core.config import (
    META_APP_SECRET,
    TIKTOK_CLIENT_KEY,
    TIKTOK_WEBHOOK_SECRET,
)
from core.helpers import _now_utc, _safe_json
from core.wallet import refund_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

# ============================================================
# TikTok Webhooks
# ============================================================

TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC = 300   # reject events older than 5 minutes

FACEBOOK_WEBHOOK_VERIFY_TOKEN = os.environ.get("FACEBOOK_WEBHOOK_VERIFY_TOKEN", "")


def _verify_tiktok_signature(raw_body: bytes, header: str, secret: str) -> tuple[bool, str]:
    """
    Parse and verify the Tiktok-Signature header.

    Returns (ok: bool, reason: str).
    ok=True  -> signature is valid and timestamp is fresh.
    ok=False -> verification failed (reason explains why).

    If TIKTOK_WEBHOOK_SECRET is empty the check is skipped and we return
    (True, "sig-check-skipped-no-secret") so the developer can still receive
    events during initial setup without crashing.
    """
    if not secret:
        return True, "sig-check-skipped-no-secret"

    if not header:
        return False, "missing-Tiktok-Signature-header"

    # Parse  "t=1633174587,s=18494715036ac441..."
    parts: dict[str, str] = {}
    for segment in header.split(","):
        if "=" in segment:
            k, _, v = segment.partition("=")
            parts[k.strip()] = v.strip()

    ts_str = parts.get("t", "")
    sig_received = parts.get("s", "")

    if not ts_str or not sig_received:
        return False, f"malformed-header:{header[:80]}"

    try:
        ts = int(ts_str)
    except ValueError:
        return False, f"non-numeric-timestamp:{ts_str}"

    # Replay-attack protection
    age = abs(int(time.time()) - ts)
    if age > TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC:
        return False, f"timestamp-too-old:{age}s"

    # Compute expected signature
    signed_payload = f"{ts_str}.".encode() + raw_body
    expected = _hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()

    if not _hmac.compare_digest(expected, sig_received):
        return False, "signature-mismatch"

    return True, "ok"


async def _handle_tiktok_event(event_type: str, payload: dict, user_openid: str) -> str:
    """
    Process a verified TikTok webhook event in the background.
    Returns a short string describing what was done (stored in handling_notes).
    """
    notes = f"event={event_type}"

    try:
        async with core.state.db_pool.acquire() as conn:

            # -- video.publish.completed ----------------------------------------
            if event_type == "video.publish.completed":
                # content may contain share_id or video_id -- store it in
                # platform_results and mark the upload completed if we can
                # match it by TikTok open_id.
                content = payload.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"raw": content}

                share_id = content.get("share_id", "")
                video_id = content.get("video_id", share_id)

                # Find the most recent tiktok upload for this open_id that is
                # still in a processing/queued state so we can mark it done.
                upload = await conn.fetchrow(
                    """
                    SELECT u.id, u.user_id, u.platform_results
                    FROM uploads u
                    JOIN platform_tokens pt
                        ON u.user_id = pt.user_id
                       AND pt.platform = 'tiktok'
                       AND pt.account_id = $1
                       AND pt.revoked_at IS NULL
                    WHERE u.status NOT IN ('completed', 'succeeded', 'failed', 'cancelled')
                    ORDER BY u.created_at DESC
                    LIMIT 1
                    """,
                    user_openid,
                )

                if upload:
                    existing = _safe_json(upload["platform_results"], {})
                    existing["tiktok"] = {
                        "status": "published",
                        "video_id": video_id,
                        "share_id": share_id,
                        "published_at": _now_utc().isoformat(),
                    }
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status           = 'completed',
                            completed_at     = NOW(),
                            platform_results = $1,
                            updated_at       = NOW()
                        WHERE id = $2
                        """,
                        json.dumps(existing),
                        upload["id"],
                    )
                    notes += f" upload={upload['id']} marked=completed video_id={video_id}"
                else:
                    notes += f" no-matching-upload-found openid={user_openid}"

            # -- video.upload.failed --------------------------------------------
            elif event_type == "video.upload.failed":
                content = payload.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"raw": content}

                share_id = content.get("share_id", "")

                upload = await conn.fetchrow(
                    """
                    SELECT u.id, u.user_id, u.platform_results,
                           u.put_reserved, u.aic_reserved
                    FROM uploads u
                    JOIN platform_tokens pt
                        ON u.user_id = pt.user_id
                       AND pt.platform = 'tiktok'
                       AND pt.account_id = $1
                       AND pt.revoked_at IS NULL
                    WHERE u.status NOT IN ('completed', 'succeeded', 'failed', 'cancelled')
                    ORDER BY u.created_at DESC
                    LIMIT 1
                    """,
                    user_openid,
                )

                if upload:
                    existing = _safe_json(upload["platform_results"], {})
                    existing["tiktok"] = {
                        "status": "failed",
                        "share_id": share_id,
                        "failed_at": _now_utc().isoformat(),
                    }
                    await conn.execute(
                        """
                        UPDATE uploads
                        SET status           = 'failed',
                            error_code       = 'tiktok_upload_failed',
                            error_detail     = 'TikTok reported upload failure via webhook',
                            platform_results = $1,
                            updated_at       = NOW()
                        WHERE id = $2
                        """,
                        json.dumps(existing),
                        upload["id"],
                    )
                    # Refund reserved tokens so the user isn't charged
                    if upload["put_reserved"] or upload["aic_reserved"]:
                        await refund_tokens(
                            conn,
                            str(upload["user_id"]),
                            upload["put_reserved"] or 0,
                            upload["aic_reserved"] or 0,
                            str(upload["id"]),
                        )
                    notes += f" upload={upload['id']} marked=failed tokens-refunded"
                else:
                    notes += f" no-matching-upload-found openid={user_openid}"

            # -- authorization.removed ------------------------------------------
            elif event_type == "authorization.removed":
                # TikTok has already revoked the token on their side; we just
                # need to purge the platform_tokens row and log the disconnect.
                rows = await conn.fetch(
                    """
                    UPDATE platform_tokens
                    SET revoked_at = NOW()
                    WHERE platform = 'tiktok'
                      AND account_id = $1
                      AND revoked_at IS NULL
                    RETURNING id, user_id, account_id, account_name
                    """,
                    user_openid,
                )
                for row in rows:
                    # Hard-delete after marking revoked
                    await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])
                    await conn.execute(
                        """
                        INSERT INTO platform_disconnect_log
                            (user_id, platform, account_id, account_name,
                             revoked_at_provider, provider_revoke_error,
                             initiated_by)
                        VALUES ($1, 'tiktok', $2, $3, TRUE, NULL, 'tiktok_webhook')
                        """,
                        str(row["user_id"]),
                        row["account_id"],
                        row["account_name"],
                    )
                    notes += f" purged_token={row['id']}"

                if not rows:
                    notes += f" no-active-token-found openid={user_openid}"

            else:
                notes += " unhandled-event-type"

    except Exception as exc:
        notes += f" ERROR:{exc}"
        logger.error(f"[tiktok-webhook] background handler error: {exc}", exc_info=True)

    return notes


# ============================================================
# TikTok Webhook Routes
# ============================================================

# -- GET /tiktok  (URL challenge / health check) ----------------------------
@router.get("/tiktok")
async def tiktok_webhook_challenge(challenge: Optional[str] = Query(None)):
    """
    TikTok calls this as a GET with ?challenge=<token> to verify the endpoint
    before activating the webhook subscription.  We must echo the challenge
    back as plain text with a 200 status.

    Also returns 200 (no body) if called with no challenge -- useful for
    uptime checks.
    """
    if challenge:
        # TikTok expects the exact challenge value as the plain-text response body
        return JSONResponse(content=challenge, media_type="text/plain")
    return JSONResponse(content={"status": "tiktok-webhook-endpoint-ok"})


# -- POST /tiktok  (live event receiver) ------------------------------------
@router.post("/tiktok")
async def tiktok_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives event notifications from TikTok Open Platform.

    Security
    --------
    1.  The raw body is read BEFORE any JSON parsing so the HMAC is computed
        over exactly the bytes TikTok signed.
    2.  The Tiktok-Signature header is verified with a constant-time compare.
    3.  Replay attacks are blocked -- events older than 5 minutes are rejected.
    4.  We return 200 OK immediately and process the event in a BackgroundTask
        so TikTok never times out waiting for us.
    5.  Every event (including rejected ones) is logged to tiktok_webhook_events
        for auditability.

    TikTok docs: https://developers.tiktok.com/doc/webhooks-verification
    """
    # -- 1. Read raw body ---------------------------------------------------
    raw_body = await request.body()

    # -- 2. Verify signature ------------------------------------------------
    sig_header = request.headers.get("Tiktok-Signature", "")
    sig_ok, sig_reason = _verify_tiktok_signature(raw_body, sig_header, TIKTOK_WEBHOOK_SECRET)

    if not sig_ok:
        logger.warning(f"[tiktok-webhook] signature rejected: {sig_reason} | header={sig_header[:80]}")
        # Still 200 so TikTok doesn't keep retrying a bad delivery;
        # we just won't process the event.
        return JSONResponse(
            status_code=200,
            content={"status": "rejected", "reason": sig_reason},
        )

    # -- 3. Parse JSON payload ----------------------------------------------
    try:
        payload = json.loads(raw_body)
    except Exception:
        logger.warning("[tiktok-webhook] non-JSON body")
        return JSONResponse(status_code=200, content={"status": "ok"})

    event_type   = payload.get("event", "unknown")
    user_openid  = payload.get("user_openid", "")
    create_time  = payload.get("create_time")
    client_key   = payload.get("client_key", "")

    # content field is sometimes a JSON string, sometimes an object
    content_raw = payload.get("content", {})
    if isinstance(content_raw, str):
        try:
            content_parsed = json.loads(content_raw)
        except Exception:
            content_parsed = {"raw": content_raw}
    else:
        content_parsed = content_raw

    logger.info(
        f"[tiktok-webhook] event={event_type} openid={user_openid} "
        f"sig_ok={sig_ok} sig_reason={sig_reason}"
    )

    # -- 4. Log the raw event immediately -----------------------------------
    async def _persist_event(notes: str):
        try:
            async with core.state.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO tiktok_webhook_events
                        (client_key, event, create_time, user_openid,
                         content, raw_body, sig_verified, handling_notes)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    """,
                    client_key,
                    event_type,
                    create_time,
                    user_openid,
                    json.dumps(content_parsed),
                    raw_body.decode(errors="replace"),
                    sig_ok,
                    notes,
                )
        except Exception as exc:
            logger.error(f"[tiktok-webhook] failed to persist event log: {exc}")

    # -- 5. Dispatch handler in background ----------------------------------
    async def _process_and_log():
        notes = await _handle_tiktok_event(event_type, payload, user_openid)
        await _persist_event(notes)

    background_tasks.add_task(_process_and_log)

    # -- 6. Return 200 immediately ------------------------------------------
    return JSONResponse(status_code=200, content={"status": "ok"})


# ============================================================
# Facebook / Instagram Webhooks
# ============================================================
# Meta calls GET to verify the endpoint (hub challenge), then POST for events.
#
# Setup in Meta Developer Console:
#   Callback URL : https://auth.uploadm8.com/api/webhooks/facebook
#   Verify Token : value of FACEBOOK_WEBHOOK_VERIFY_TOKEN env var
#
# Required env var:
#   FACEBOOK_WEBHOOK_VERIFY_TOKEN  -- any secret string you set in the console
#
# Meta docs: https://developers.facebook.com/docs/graph-api/webhooks/getting-started
# ============================================================

@router.get("/facebook")
async def facebook_webhook_challenge(
    hub_mode: Optional[str]      = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str]  = Query(None, alias="hub.challenge"),
):
    """
    Meta calls this GET endpoint when you click 'Verify and save' in the
    developer console.  We must:
      1. Confirm hub.mode == 'subscribe'
      2. Confirm hub.verify_token matches our secret
      3. Return hub.challenge as plain text with HTTP 200
    """
    if hub_mode == "subscribe" and hub_verify_token:
        expected = FACEBOOK_WEBHOOK_VERIFY_TOKEN
        if expected and hub_verify_token == expected:
            logger.info("[facebook-webhook] challenge verified OK")
            return PlainTextResponse(hub_challenge or "")
        else:
            logger.warning(
                f"[facebook-webhook] verify token mismatch — "
                f"got={hub_verify_token[:20]}… expected={'(not set)' if not expected else '***'}"
            )
            raise HTTPException(403, "Verify token mismatch")

    # Health check (no params)
    return JSONResponse({"status": "facebook-webhook-endpoint-ok"})


@router.post("/facebook")
async def facebook_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives real-time event notifications from Meta (Facebook & Instagram).

    Security
    --------
    Signature is verified using X-Hub-Signature-256: sha256=<hmac>
    computed with META_APP_SECRET as the key over the raw request body.
    Events that fail signature verification are rejected with 403.
    We return 200 immediately and process in a BackgroundTask.

    Subscribed fields to configure in Meta console (under 'Webhook Fields'):
      - For Page events: feed, video_feeds
      - For Instagram: mentions, story_insights, comments, live_comments
    """
    raw_body = await request.body()

    # -- Verify signature ----------------------------------------------------
    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if META_APP_SECRET and sig_header:
        import hmac as _hmac_fb
        expected_sig = "sha256=" + _hmac_fb.new(
            META_APP_SECRET.encode(), raw_body, hashlib.sha256
        ).hexdigest()
        if not _hmac_fb.compare_digest(expected_sig, sig_header):
            logger.warning(f"[facebook-webhook] signature mismatch — header={sig_header[:60]}")
            raise HTTPException(403, "Invalid signature")
    elif META_APP_SECRET and not sig_header:
        # Signature header missing entirely -- reject in production
        logger.warning("[facebook-webhook] missing X-Hub-Signature-256 header")
        raise HTTPException(400, "Missing signature")

    # -- Parse payload -------------------------------------------------------
    try:
        payload = json.loads(raw_body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    object_type = payload.get("object", "")
    entries     = payload.get("entry", []) or []

    logger.info(
        f"[facebook-webhook] received object={object_type} entries={len(entries)}"
    )

    async def _process_fb_events():
        for entry in entries:
            entry_id   = entry.get("id", "")
            changes    = entry.get("changes", []) or []
            messaging  = entry.get("messaging", []) or []

            for change in changes:
                field  = change.get("field", "")
                value  = change.get("value", {}) or {}

                # -- Video published (Page feed) ---------------------------------
                if object_type == "page" and field in ("feed", "video_feeds"):
                    item = value.get("item", "")
                    verb = value.get("verb", "")
                    video_id = value.get("video_id") or value.get("post_id", "")
                    logger.info(
                        f"[facebook-webhook] page={entry_id} field={field} "
                        f"item={item} verb={verb} video_id={video_id}"
                    )

                # -- Instagram media / comments ----------------------------------
                elif object_type == "instagram":
                    media_id = (
                        value.get("media_id")
                        or value.get("id")
                        or value.get("item_id", "")
                    )
                    logger.info(
                        f"[facebook-webhook] instagram field={field} "
                        f"media_id={media_id} value_keys={list(value.keys())}"
                    )

            # -- (Optional) Messenger events -- ignored for now -----------------
            if messaging:
                logger.debug(
                    f"[facebook-webhook] messaging events={len(messaging)} (not handled)"
                )

    background_tasks.add_task(_process_fb_events)

    # Meta requires a fast 200 response
    return JSONResponse({"status": "ok"})
