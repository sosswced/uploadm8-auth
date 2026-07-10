"""
Public marketing routes (open pixel, one-click unsubscribe, client events).

Extracted from ``routers.admin_contract`` to keep that module under the router line cap.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException, Response

import core.state
from core.deps import get_current_user

logger = logging.getLogger("uploadm8-api")

public_marketing_router = APIRouter(prefix="/api/marketing", tags=["marketing"])


@public_marketing_router.get("/o/{token}")
async def marketing_email_open_pixel(token: str):
    """1×1 tracking pixel for campaign email opens (signed token)."""
    import base64

    from services.marketing_promo_media import verify_tracking_token
    from services.ml_marketing import record_outcome_label

    px = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax9pZkAAAAASUVORK5CYII="
    )
    pl = verify_tracking_token(token)
    if not pl or str(pl.get("t")) != "open":
        return Response(content=px, media_type="image/png")
    uid = str(pl.get("u") or "")
    if not uid:
        return Response(content=px, media_type="image/png")
    cid = str(pl.get("c") or "")
    vid = str(pl.get("v") or "")
    did = str(pl.get("d") or "")
    try:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO marketing_events (user_id, event_type, payload)
                VALUES ($1::uuid, 'campaign_email_open', $2::jsonb)
                """,
                uid,
                json.dumps(
                    {
                        "campaign_id": cid,
                        "delivery_id": did,
                        "variant_id": vid,
                        "promo_variant_id": vid,
                    }
                ),
            )
            await record_outcome_label(
                conn,
                user_id=uid,
                upload_id=None,
                variant_id=vid or None,
                feature_snapshot={"campaign_id": cid, "delivery_id": did},
                label_json={"email_open": True},
            )
    except Exception:
        logger.debug("marketing open pixel skip", exc_info=True)
    return Response(content=px, media_type="image/png")


@public_marketing_router.api_route("/unsubscribe/{token}", methods=["GET", "POST"])
async def marketing_one_click_unsubscribe(token: str):
    """List-Unsubscribe-Post / one-click: opt out of email marketing consent."""
    from services.marketing_compliance import upsert_user_marketing_consent
    from services.marketing_promo_media import verify_tracking_token

    pl = verify_tracking_token(token)
    if not pl or str(pl.get("t")) != "unsub_email":
        raise HTTPException(400, "Invalid unsubscribe token")
    uid = str(pl.get("u") or "").strip()
    if not uid:
        raise HTTPException(400, "Invalid unsubscribe token")
    try:
        async with core.state.db_pool.acquire() as conn:
            consent = await upsert_user_marketing_consent(
                conn, uid, email_marketing=False
            )
            try:
                await conn.execute(
                    """
                    INSERT INTO marketing_events (user_id, event_type, payload)
                    VALUES ($1::uuid, 'marketing_unsubscribed', $2::jsonb)
                    """,
                    uid,
                    json.dumps({"via": "list_unsubscribe", "email_marketing": False}),
                )
            except Exception:
                pass
        return {
            "ok": True,
            "email_marketing": False,
            "consent": consent,
            "message": "You have been unsubscribed from marketing emails.",
        }
    except Exception as e:
        logger.warning("marketing one-click unsub failed: %s", e)
        raise HTTPException(500, "Could not update marketing preferences")


@public_marketing_router.post("/events")
async def marketing_events_ingest(
    payload: Dict[str, Any] = Body(default_factory=dict),
    user: dict = Depends(get_current_user),
):
    et = str(payload.get("event_type") or "unknown")[:80]
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO marketing_events (user_id, event_type, payload)
            VALUES ($1::uuid, $2, $3::jsonb)
            """,
            str(user["id"]),
            et,
            json.dumps(payload),
        )
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        cid = str(meta.get("campaign_id") or "").strip()
        if cid and et in ("shown", "clicked", "dismissed", "converted"):
            try:
                from services.ml_marketing import record_outcome_label

                vid = str(meta.get("promo_variant_id") or meta.get("variant_id") or "")[:128]
                await record_outcome_label(
                    conn,
                    user_id=str(user["id"]),
                    upload_id=None,
                    variant_id=vid or None,
                    feature_snapshot={
                        "campaign_id": cid,
                        "wallet_banner_event": et,
                        "nudge_type": payload.get("nudge_type"),
                        "page": payload.get("page"),
                    },
                    label_json={f"wallet_banner_{et}": True},
                )
            except Exception:
                logger.debug("marketing_events ml label skip", exc_info=True)
    return Response(status_code=204)
