"""
Per-user marketing card images: local sub_/topup_ PNG base + optional Pikzels v2 edit.

Callers (e.g. ``marketing_execution``) must enforce ``marketing_compliance`` (suppression,
rate limits, opt-in) before scheduling outbound. This module re-checks email/discord allow
when ``channel`` is set (defense in depth).

On Pikzels or R2 failure, returns the base PNG (data URL or presigned key) so delivery
never blocks on vendor paths.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import asyncpg
import httpx

from core.r2 import _normalize_r2_key, generate_presigned_download_url, put_object_bytes
from core.wallet import atomic_debit_tokens
from services.marketing_compliance import (
    is_suppressed,
    user_discord_marketing_allowed,
    user_email_marketing_allowed,
)
from services.ml_marketing import record_outcome_label
from services.pikzels_v2 import V2_THUMBNAIL_EDIT, resolve_public_api_key
from services.pikzels_v2_client import (
    coerce_pikzels_v2_image_base64_fields,
    pikzels_timeout_seconds,
    pikzels_v2_post,
)
from services.wallet_marketing import _user_campaign_features
from services.billing_service_weights import fetch_service_weights_map
from stages.ai_service_costs import SERVICE_WEIGHTS, merge_service_weights_from_db
from stages.pikzels_api import _pikzels_v2_response_to_bytes

logger = logging.getLogger("uploadm8.marketing_image")

_REPO = Path(__file__).resolve().parents[1]
FRONTEND_IMAGES_DIR = Path(
    os.environ.get("UPLOADM8_FRONTEND_IMAGES_DIR", str(_REPO / "frontend" / "images"))
)

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.environ.get("OPENAI_MARKETING_MODEL") or "gpt-4o-mini").strip()

_MARKETING_IMAGE_PRESIGN_TTL = int(os.environ.get("MARKETING_IMAGE_PRESIGN_TTL_SEC", "604800") or 604800)

DEFAULT_CARD_BY_KIND: Dict[str, str] = {
    "topup_aic": "topup_aic_500.png",
    "topup_put": "topup_put_500.png",
    "sub_upgrade": "sub_creator_pro.png",
    "win_back": "sub_creator_lite.png",
    "trial_remind": "sub_creator_pro.png",
}


async def _marketing_image_aic_charge_db(conn: asyncpg.Connection) -> int:
    raw = await fetch_service_weights_map(conn)
    mw = merge_service_weights_from_db(raw)
    w = int(mw.get("marketing_image", SERVICE_WEIGHTS.get("marketing_image", 1)) or 1)
    return max(1, w)


async def _channel_compliance_ok(
    conn: asyncpg.Connection, user_id: str, channel: Optional[str]
) -> bool:
    if not channel:
        return True
    c = channel.strip().lower()
    if c == "email":
        if await is_suppressed(conn, user_id, "email"):
            return False
        return await user_email_marketing_allowed(conn, user_id)
    if c == "discord":
        if await is_suppressed(conn, user_id, "discord"):
            return False
        return await user_discord_marketing_allowed(conn, user_id)
    if c == "mixed":
        return await _channel_compliance_ok(conn, user_id, "email") and await _channel_compliance_ok(
            conn, user_id, "discord"
        )
    return True


async def _get_user_features(
    conn: asyncpg.Connection, user_id: str, *, range_key: str = "30d"
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    try:
        feats = dict(await _user_campaign_features(conn, user_id, range_key))
    except Exception as exc:
        logger.debug("wallet_marketing features failed: %s", exc)
        feats = {}

    row = await conn.fetchrow(
        """
        SELECT u.id::text AS user_id,
               u.email,
               COALESCE(u.subscription_tier, 'free') AS tier,
               u.role,
               COALESCE(w.put_balance, 0)::bigint AS put_balance,
               COALESCE(w.aic_balance, 0)::bigint AS aic_balance,
               u.created_at,
               (SELECT MAX(created_at) FROM uploads WHERE user_id = u.id) AS last_upload_at,
               (SELECT COUNT(*)::bigint FROM uploads WHERE user_id = u.id) AS upload_count
        FROM users u
        LEFT JOIN wallets w ON w.user_id = u.id
        WHERE u.id = $1::uuid
        LIMIT 1
        """,
        user_id,
    )
    if not row:
        base = {"tier": "free", "put_balance": 0, "aic_balance": 0, "upload_count": 0}
        base.update(feats)
        return base
    core = dict(row)
    core["upload_count"] = int(core.get("upload_count") or 0)
    core["put_balance"] = int(core.get("put_balance") or 0)
    core["aic_balance"] = int(core.get("aic_balance") or 0)
    core.update(feats)
    return core


def _deterministic_headline(
    kind: str, feats: Dict[str, Any], amount: Optional[int] = None
) -> str:
    tier = (feats.get("tier") or "free").lower()
    put = int(feats.get("put_balance") or 0)
    aic = int(feats.get("aic_balance") or 0)
    uploads = int(feats.get("upload_count") or 0)

    if kind == "topup_aic":
        if aic < 20:
            return f"Only {aic} AIC left — top up and keep AI captions flowing"
        return f"+{amount or 500} AIC — power your next {amount or 500} AI-assisted uploads"
    if kind == "topup_put":
        if put < 10:
            return f"You're at {put} PUT — refuel before your next batch"
        return f"+{amount or 500} PUT — publish {amount or 500} videos across all platforms"
    if kind == "sub_upgrade":
        if uploads > 50 and tier in ("free", "starter", "launch"):
            return f"You've shipped {uploads} videos on Starter — Creator Pro pays for itself"
        return "Unlock priority queue, no watermark, and 6x AI credits"
    if kind == "win_back":
        return "We miss you — your next upload is on us"
    if kind == "trial_remind":
        return "Your 7-day trial is ticking — lock in your tier today"
    return "UploadM8 — 4 platforms, 1 upload, 0 chaos"


async def _resolve_base_filename_from_catalog(
    conn: asyncpg.Connection, kind: str, amount: Optional[int]
) -> Optional[str]:
    """Prefer ``catalog_products.image_filename`` over static ``DEFAULT_CARD_BY_KIND``."""
    try:
        if kind in ("topup_put", "topup_aic") and amount is not None:
            row = await conn.fetchrow(
                """
                SELECT image_filename FROM catalog_products
                WHERE product_kind = $1 AND token_amount = $2
                  AND is_archived = FALSE
                LIMIT 1
                """,
                kind,
                int(amount),
            )
            if row and row.get("image_filename"):
                return str(row["image_filename"])
        tier_for_kind = {
            "win_back": "creator_lite",
            "trial_remind": "creator_pro",
            "sub_upgrade": "creator_pro",
        }
        slug = tier_for_kind.get(kind)
        if slug:
            row = await conn.fetchrow(
                """
                SELECT image_filename FROM catalog_products
                WHERE product_kind = 'subscription' AND tier_slug = $1
                  AND is_archived = FALSE
                LIMIT 1
                """,
                slug,
            )
            if row and row.get("image_filename"):
                return str(row["image_filename"])
    except Exception as exc:
        logger.debug("catalog image_filename lookup failed: %s", exc)
    return None


async def _llm_headline(
    kind: str, feats: Dict[str, Any], amount: Optional[int] = None
) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    prompt = (
        "You are an SEO/lifecycle marketer writing a single short headline "
        "(max 70 chars) for an UploadM8 marketing card. UploadM8 publishes "
        "one video to TikTok, YouTube Shorts, Instagram Reels, and Facebook "
        "Reels in one click. Tone: confident, benefit-led, no exclamation "
        "marks, no emojis. Return JSON: {\"headline\": \"...\"}.\n\n"
        f"Campaign kind: {kind}\n"
        f"User tier: {feats.get('tier')}\n"
        f"PUT balance: {feats.get('put_balance')}\n"
        f"AIC balance: {feats.get('aic_balance')}\n"
        f"Lifetime uploads: {feats.get('upload_count')}\n"
        f"Top-up amount offered: {amount}\n"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.6,
                    "max_tokens": 80,
                },
            )
        if r.status_code >= 400:
            logger.warning("OpenAI marketing headline HTTP %s", r.status_code)
            return None
        text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(text or "{}")
        h = (parsed.get("headline") or "").strip()
        return h[:120] if h else None
    except Exception as exc:
        logger.warning("LLM headline error: %s", exc)
        return None


async def _personalize_via_pikzels(
    base_image_bytes: bytes, headline: str, _user_id: str
) -> Optional[bytes]:
    if not resolve_public_api_key():
        return None
    b64 = base64.b64encode(base_image_bytes).decode("ascii")
    instruction = (
        f'Overlay this headline near the top-center of the card in bold white '
        f'sans-serif text, with subtle shadow for readability: "{headline}". '
        f"Keep all existing card structure (header band, logo, body, footer) "
        f"untouched. Do not redraw the layout. Return at original resolution."
    )[:1000]
    body: Dict[str, Any] = {
        "image_base64": f"data:image/png;base64,{b64}",
        "prompt": instruction,
        "format": "16:9",
        "model": (os.environ.get("PIKZELS_EDIT_MODEL") or "pkz_4").strip()[:32],
    }
    coerce_pikzels_v2_image_base64_fields(body)
    timeout = pikzels_timeout_seconds()
    try:
        status, resp = await pikzels_v2_post(V2_THUMBNAIL_EDIT, body)
        if status >= 400 or not isinstance(resp, dict):
            logger.info("pikzels edit returned %s — using base", status)
            return None
        out = await _pikzels_v2_response_to_bytes(resp, timeout=timeout)
        if out and len(out) > 100:
            return out
        return None
    except Exception as exc:
        logger.warning("pikzels edit error: %s", exc)
        return None


async def _upload_to_r2(image_bytes: bytes, key: str) -> tuple[Optional[str], str]:
    """Returns (presigned_get_url_or_none, normalized_key)."""
    nk = _normalize_r2_key(key)
    try:
        await asyncio.to_thread(put_object_bytes, nk, image_bytes, "image/png")
    except Exception as exc:
        logger.warning("R2 upload error: %s", exc)
        return None, nk
    try:
        url = generate_presigned_download_url(nk, ttl=_MARKETING_IMAGE_PRESIGN_TTL) or None
    except Exception as exc:
        logger.warning("presign marketing image failed: %s", exc)
        url = None
    return url, nk


async def generate_marketing_image(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    kind: str,
    campaign_id: Optional[str] = None,
    variant_id: Optional[str] = None,
    headline: Optional[str] = None,
    amount: Optional[int] = None,
    upload_to_r2: bool = True,
    use_llm: bool = True,
    use_pikzels: bool = True,
    debit_wallet: bool = False,
    channel: Optional[str] = None,
    features_range_key: str = "30d",
) -> Dict[str, Any]:
    """
    Build a personalized marketing card for one user.

    Returns ``image_url`` (presigned HTTPS, or ``data:image/png;base64,...``),
    ``image_key``, ``variant_id``, ``headline``, and ``features_used``.
    """
    variant_id = variant_id or f"v_{uuid.uuid4().hex[:10]}"
    if not await _channel_compliance_ok(conn, user_id, channel):
        return {
            "image_url": None,
            "image_key": None,
            "variant_id": variant_id,
            "headline": headline,
            "error": "compliance_blocked",
            "features_used": {},
        }

    feats = await _get_user_features(conn, user_id, range_key=features_range_key)

    llm_used = False
    final_headline = (headline or "").strip() or None
    if not final_headline and use_llm:
        llm_h = await _llm_headline(kind, feats, amount=amount)
        if llm_h:
            final_headline = llm_h
            llm_used = True
    if not final_headline:
        final_headline = _deterministic_headline(kind, feats, amount=amount)

    base_name = await _resolve_base_filename_from_catalog(conn, kind, amount)
    if not base_name and amount is not None and kind == "topup_aic":
        base_name = f"topup_aic_{int(amount)}.png"
    if not base_name and amount is not None and kind == "topup_put":
        base_name = f"topup_put_{int(amount)}.png"
    if not base_name:
        base_name = DEFAULT_CARD_BY_KIND.get(kind) or "sub_creator_pro.png"
    base_path = FRONTEND_IMAGES_DIR / base_name
    if not base_path.is_file():
        logger.warning("base card missing: %s", base_path)
        return {
            "image_url": None,
            "image_key": None,
            "variant_id": variant_id,
            "headline": final_headline,
            "error": "base_image_missing",
            "features_used": feats,
        }
    base_bytes = base_path.read_bytes()

    if debit_wallet:
        aic_units = await _marketing_image_aic_charge_db(conn)
        ref = f"marketing_image:{variant_id}"
        ok = await atomic_debit_tokens(
            conn, user_id, 0, aic_units, ref, reason="marketing_image"
        )
        if not ok:
            return {
                "image_url": None,
                "image_key": None,
                "variant_id": variant_id,
                "headline": final_headline,
                "error": "insufficient_aic",
                "features_used": feats,
            }

    final_bytes = base_bytes
    pikzels_used = False
    if use_pikzels:
        edited = await _personalize_via_pikzels(base_bytes, final_headline, user_id)
        if edited:
            final_bytes = edited
            pikzels_used = True

    key = (
        f"marketing-assets/{user_id}/{campaign_id or 'adhoc'}/{variant_id}.png"
    )
    image_url: Optional[str] = None
    image_key = key
    if upload_to_r2:
        image_url, image_key = await _upload_to_r2(final_bytes, key)

    if not image_url:
        b64 = base64.b64encode(final_bytes).decode("ascii")
        image_url = f"data:image/png;base64,{b64}"

    try:
        await record_outcome_label(
            conn,
            user_id=user_id,
            upload_id=None,
            variant_id=variant_id,
            feature_snapshot={
                "kind": kind,
                "amount": amount,
                "tier": feats.get("tier"),
                "put_balance": feats.get("put_balance"),
                "aic_balance": feats.get("aic_balance"),
                "upload_count": feats.get("upload_count"),
                "pikzels_used": pikzels_used,
                "llm_used": llm_used,
            },
            label_json={
                "event": "image_generated",
                "headline": final_headline,
                "campaign_id": campaign_id,
            },
        )
    except Exception as exc:
        logger.warning("ml_outcome_labels insert failed: %s", exc)

    return {
        "image_url": image_url,
        "image_key": image_key,
        "variant_id": variant_id,
        "headline": final_headline,
        "pikzels_used": pikzels_used,
        "features_used": feats,
    }
