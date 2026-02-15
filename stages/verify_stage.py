"""stages.verify_stage

Real confirmation stage.

This runs as a background loop (non-blocking to main job pipeline) and turns
"accepted" publishes into "confirmed" or "rejected" via platform status lookups.

Contract:
- Only touches publish_attempts where status='success' and verify_status='pending'.
- Uses exponential backoff via next_verify_at.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx

from . import db as db_stage
from .publish_stage import decrypt_token, init_enc_keys

logger = logging.getLogger("uploadm8-worker")

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=10.0)


def _safe_json(resp: httpx.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
        if isinstance(data, dict):
            # strip obvious secrets
            out = {}
            for k, v in data.items():
                lk = str(k).lower()
                if "token" in lk or "secret" in lk or "authorization" in lk:
                    continue
                out[k] = v
            return out
        return {"data": data}
    except Exception:
        return {"text": (resp.text or "")[:4000]}


async def _verify_tiktok(client: httpx.AsyncClient, access_token: str, publish_id: str) -> tuple[str, dict, Optional[str]]:
    """Return (verify_status, payload, url_override)."""
    # TikTok publish status endpoint (may vary by app scope)
    url = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    resp = await client.post(url, headers=headers, json={"publish_id": publish_id})
    payload = {"http_status": resp.status_code, "body": _safe_json(resp)}

    if resp.status_code == 401 or resp.status_code == 403:
        return "unknown", payload, None

    body = payload.get("body") or {}
    # Heuristics (TikTok varies): look for status field
    status = None
    for key in ("status", "publish_status", "result"):  # tolerate schema drift
        if key in body:
            status = body.get(key)
            break
    if isinstance(status, dict):
        status = status.get("status") or status.get("publish_status")

    s = str(status or "").upper()
    if "COMPLETE" in s or "PUBLISHED" in s or "SUCCESS" in s:
        return "confirmed", payload, None
    if "FAIL" in s or "REJECT" in s or "ERROR" in s:
        return "rejected", payload, None

    return "pending", payload, None


async def _verify_youtube(client: httpx.AsyncClient, access_token: str, video_id: str) -> tuple[str, dict, Optional[str]]:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"id": video_id, "part": "status,snippet"}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = await client.get(url, headers=headers, params=params)
    payload = {"http_status": resp.status_code, "body": _safe_json(resp)}

    if resp.status_code == 401 or resp.status_code == 403:
        return "unknown", payload, None

    body = payload.get("body") or {}
    items = body.get("items") or []
    if not items:
        return "pending", payload, None

    status = (items[0].get("status") or {}).get("uploadStatus")
    s = str(status or "").lower()
    if s in ("processed", "uploaded"):
        # public URL can be derived
        return "confirmed", payload, f"https://www.youtube.com/watch?v={video_id}"
    if s in ("failed", "rejected", "deleted"):
        return "rejected", payload, None

    return "pending", payload, None


async def _verify_facebook(client: httpx.AsyncClient, access_token: str, video_id: str) -> tuple[str, dict, Optional[str]]:
    url = f"https://graph.facebook.com/v19.0/{video_id}"
    params = {"fields": "status,published", "access_token": access_token}
    resp = await client.get(url, params=params)
    payload = {"http_status": resp.status_code, "body": _safe_json(resp)}

    if resp.status_code == 401 or resp.status_code == 403:
        return "unknown", payload, None

    body = payload.get("body") or {}
    if body.get("published") is True:
        return "confirmed", payload, None

    status = body.get("status")
    if isinstance(status, dict):
        status = status.get("video_status") or status.get("status")

    s = str(status or "").lower()
    if "ready" in s or "published" in s:
        return "confirmed", payload, None
    if "error" in s or "expired" in s or "failed" in s:
        return "rejected", payload, None

    return "pending", payload, None


async def verify_single_attempt(pool, attempt: dict) -> None:
    attempt_id = str(attempt.get("id"))
    platform = str(attempt.get("platform") or "").lower()
    verify_attempts = int(attempt.get("verify_attempts") or 0)

    # max polls ~7 -> unknown
    if verify_attempts >= 7:
        await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "max_attempts"})
        return

    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
        try:
            # ensure crypto available for decrypting stored tokens
            init_enc_keys()
            if platform == "tiktok":
                publish_id = attempt.get("publish_id") or attempt.get("platform_post_id")
                if not publish_id:
                    await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "missing_publish_id"})
                    return

                token_row = await db_stage.load_platform_token(pool, attempt.get("user_id"), "tiktok")
                token_row = decrypt_token(token_row) if token_row else None
                access = (token_row or {}).get("access_token")
                if not access:
                    await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "missing_access_token"})
                    return

                status, payload, url_override = await _verify_tiktok(client, access, str(publish_id))

            elif platform == "youtube":
                video_id = attempt.get("platform_post_id")
                if not video_id:
                    await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "missing_video_id"})
                    return

                token_row = await db_stage.load_platform_token(pool, attempt.get("user_id"), "google")
                token_row = decrypt_token(token_row) if token_row else None
                access = (token_row or {}).get("access_token")
                if not access:
                    await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "missing_access_token"})
                    return

                status, payload, url_override = await _verify_youtube(client, access, str(video_id))

            elif platform == "facebook":
                video_id = attempt.get("platform_post_id")
                if not video_id:
                    await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "missing_video_id"})
                    return

                token_row = await db_stage.load_platform_token(pool, attempt.get("user_id"), "meta")
                token_row = decrypt_token(token_row) if token_row else None
                access = (token_row or {}).get("access_token")
                if not access:
                    await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": "missing_access_token"})
                    return

                status, payload, url_override = await _verify_facebook(client, access, str(video_id))

            else:
                # not implemented yet
                await db_stage.update_verify_unknown(pool, attempt_id=attempt_id, verify_payload={"reason": f"unsupported_platform:{platform}"})
                return

            if status == "confirmed":
                await db_stage.update_verify_confirmed(pool, attempt_id=attempt_id, verify_payload=payload, platform_url=url_override)
            elif status == "rejected":
                await db_stage.update_verify_rejected(pool, attempt_id=attempt_id, verify_payload=payload)
            else:
                await db_stage.update_verify_retry(
                    pool,
                    attempt_id=attempt_id,
                    current_verify_attempts=verify_attempts + 1,
                    verify_payload=payload,
                )

        except Exception as e:
            # retry with backoff (but don’t loop forever)
            payload = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            await db_stage.update_verify_retry(
                pool,
                attempt_id=attempt_id,
                current_verify_attempts=verify_attempts + 1,
                verify_payload=payload,
            )


async def run_verification_loop(pool, shutdown_event: asyncio.Event, *, poll_interval_seconds: int = 10, batch_size: int = 20) -> None:
    """Continuously verify pending publish attempts."""
    logger.info("Verification loop started")

    while not shutdown_event.is_set():
        try:
            pending = await db_stage.get_pending_verifications(pool, limit=batch_size)
            if pending:
                # limit concurrency
                sem = asyncio.Semaphore(8)

                async def _run_one(a: dict):
                    async with sem:
                        await verify_single_attempt(pool, a)

                await asyncio.gather(*[_run_one(a) for a in pending])
        except Exception:
            logger.exception("Verification loop error")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval_seconds)
        except asyncio.TimeoutError:
            pass

    logger.info("Verification loop stopped")
