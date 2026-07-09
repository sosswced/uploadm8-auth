"""
Platform-funded Pikzels images for marketing (email, Discord, in-app promos).

- Uses server PIKZELS_API_KEY only — no user wallet debit.
- Uploads durable copies to R2; emails use presigned GET URLs.
- Rate-limited via marketing_promo_media_runs + env MARKETING_PIKZELS_MAX_PER_HOUR.

Environment (ops):
  MARKETING_PROMO_MEDIA_ENABLED — master switch for generation + outbound image URLs.
  MARKETING_PIKZELS_MAX_PER_HOUR — cap Pikzels calls per hour (default 20).
  MARKETING_PIKZELS_ALLOWED_HOSTS — optional comma list; when set, presigned URLs must match these hostnames.
  MARKETING_PROMO_PRESIGN_TTL_SEC — presigned GET TTL (default 7d).
  MARKETING_PIKZELS_SEED_IMAGE_URL — optional seed still for /v2/thumbnail/image.
  MARKETING_PROMO_USER_VISUAL_CONTEXT — when true, campaigns with ``promo_media.user_context`` (or always if unset on campaign) use per-recipient seeds from connected platform avatars + strongest upload thumbnail (marketing sends only; never upload pipeline / Studio debit).
  promo_media.personalize_product_card — when true (and MARKETING_PROMO_MEDIA_ENABLED), ``marketing_execution`` calls ``services.marketing_image.generate_marketing_image`` instead of generic Pikzels promo. Optional keys: ``card_kind`` (topup_aic|topup_put|sub_upgrade|win_back|trial_remind), ``amount``, ``headline``, ``variant_id``, ``use_llm``, ``use_pikzels``, ``debit_wallet_for_card``.
  MARKETING_TRACK_SECRET — HMAC secret for email open pixel tokens (falls back to MAILGUN prefix in dev).
  PUBLIC_API_BASE_URL / API_PUBLIC_URL — API origin embedded in open-pixel URLs for Mailgun fetches.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import asyncpg
import httpx

from core.r2 import (
    _normalize_r2_key,
    generate_presigned_download_url,
    put_object_bytes,
    r2_object_exists,
    resolve_stored_account_avatar_url,
)
from services.pikzels_v2_client import pikzels_v2_post, pikzels_api_key
from services.thumbnail_studio import _pikzels_extract_image_url
from stages.entitlements import normalize_tier

logger = logging.getLogger("uploadm8.marketing_promo_media")

PROMO_SPEC_VERSION = "1"

_MARKETING_TRACK_SECRET = (
    os.environ.get("MARKETING_TRACK_SECRET", "").strip()
    or os.environ.get("MAILGUN_API_KEY", "").strip()[:32]
    or "dev-insecure-change-me"
)


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except (TypeError, ValueError):
        return default


def promo_media_enabled() -> bool:
    return _env_bool("MARKETING_PROMO_MEDIA_ENABLED", False)


def user_visual_context_globally_enabled() -> bool:
    return _env_bool("MARKETING_PROMO_USER_VISUAL_CONTEXT", False)


def user_visual_context_enabled_for_campaign(promo_media: Dict[str, Any]) -> bool:
    """Per-campaign flag or global env — marketing/comms only."""
    if not isinstance(promo_media, dict):
        return user_visual_context_globally_enabled()
    if bool(promo_media.get("user_context")):
        return True
    return user_visual_context_globally_enabled()


def max_pikzels_per_hour() -> int:
    return max(1, _env_int("MARKETING_PIKZELS_MAX_PER_HOUR", 20))


def presign_ttl_seconds() -> int:
    return max(3600, _env_int("MARKETING_PROMO_PRESIGN_TTL_SEC", 604800))


def seed_image_url() -> str:
    u = (os.environ.get("MARKETING_PIKZELS_SEED_IMAGE_URL") or "").strip()
    if u.startswith("http"):
        return u
    base = (os.environ.get("FRONTEND_URL") or "https://app.uploadm8.com").rstrip("/")
    return f"{base}/images/logo.png"


def segment_bucket_from_targeting(targeting: Dict[str, Any]) -> str:
    """Campaign-level tier filter label (for admin previews), not a per-user bucket."""
    if not isinstance(targeting, dict):
        return "all"
    tiers = [normalize_tier(t) for t in (targeting.get("tiers") or []) if t]
    if not tiers:
        return "all"
    return "_".join(sorted(set(tiers)))[:120]


def user_segment_bucket(user_tier: str, targeting: Dict[str, Any]) -> str:
    """
    Per-user segment for promo image cache: normalized tier when the campaign filters tiers,
    otherwise a single shared bucket.
    """
    if not isinstance(targeting, dict):
        return "all"
    tiers = [normalize_tier(t) for t in (targeting.get("tiers") or []) if t]
    if tiers:
        return normalize_tier(user_tier or "free")
    return "all"


def allowed_promo_image_url(url: Optional[str]) -> bool:
    """If MARKETING_PIKZELS_ALLOWED_HOSTS is set, require hostname match; else any https URL."""
    if not url or not str(url).startswith("https://"):
        return False
    hosts = [h.strip().lower() for h in (os.environ.get("MARKETING_PIKZELS_ALLOWED_HOSTS") or "").split(",") if h.strip()]
    if not hosts:
        return True
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        return host in hosts
    except Exception:
        return False


def _html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_platform_avatars_html_row(platform_rows: Any, *, max_icons: int = 6) -> str:
    """
    Small horizontal strip of channel avatars for marketing email HTML only.
    ``platform_rows`` is a list of dicts with keys platform, avatar_url (https).
    """
    if not isinstance(platform_rows, list) or not platform_rows:
        return ""
    parts: list[str] = []
    for row in platform_rows[:max_icons]:
        if not isinstance(row, dict):
            continue
        url = str(row.get("avatar_url") or "").strip()
        if not url.startswith("https://"):
            continue
        pf = _html_escape(str(row.get("platform") or "channel")[:24])
        parts.append(
            f'<img src="{_html_escape(url)}" width="36" height="36" alt="" title="{pf}" '
            'style="width:36px;height:36px;border-radius:999px;object-fit:cover;border:2px solid #27272a;margin-right:6px;display:inline-block;vertical-align:middle;" />'
        )
    if not parts:
        return ""
    return (
        '<p style="margin:12px 0 8px;font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.08em;">Your channels</p>'
        f'<p style="margin:0 0 12px;">{"".join(parts)}</p>'
    )


def _visual_fingerprint(platform_keys: list[str], top_upload_id: str) -> str:
    raw = "|".join(sorted(platform_keys)) + "||" + (top_upload_id or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def fetch_user_marketing_visual_context(conn: asyncpg.Connection, user_id: str) -> Dict[str, Any]:
    """
    Read-only aggregates for **marketing** image seeds only (never used by upload publish).

    - Connected platforms + mirrored/presigned avatars (``platform_tokens``).
    - Strongest recent upload by ``views`` with a ``thumbnail_r2_key`` for Pikzels ``image_url`` seed.
    """
    out: Dict[str, Any] = {
        "platforms": [],
        "seed_image_https": "",
        "top_upload_id": "",
        "fingerprint": "none",
        "platform_labels": "",
    }
    uid = str(user_id).strip()
    if not uid:
        return out

    try:
        prow = await conn.fetch(
            """
            SELECT platform, account_avatar, is_primary
            FROM platform_tokens
            WHERE user_id = $1::uuid AND revoked_at IS NULL
            ORDER BY is_primary DESC NULLS LAST, created_at DESC
            LIMIT 12
            """,
            uuid.UUID(uid),
        )
    except Exception:
        prow = []

    slugs: list[str] = []
    for r in prow or []:
        pf = str(r.get("platform") or "").strip().lower()[:32]
        if not pf:
            continue
        av = resolve_stored_account_avatar_url(r.get("account_avatar"), ttl=presign_ttl_seconds())
        if av and allowed_promo_image_url(av):
            out["platforms"].append({"platform": pf, "avatar_url": av})
            aid = str(r.get("account_id") or "").strip()[:80]
            slugs.append(f"{pf}:{aid}" if aid else pf)

    top_id = ""
    thumb_key = ""
    try:
        urow = await conn.fetchrow(
            """
            SELECT id, thumbnail_r2_key, COALESCE(views, 0)::bigint AS v
            FROM uploads
            WHERE user_id = $1::uuid
              AND status IN ('completed', 'succeeded', 'partial')
              AND COALESCE(NULLIF(TRIM(thumbnail_r2_key), ''), '') <> ''
            ORDER BY COALESCE(views, 0) DESC, created_at DESC
            LIMIT 1
            """,
            uuid.UUID(uid),
        )
    except Exception:
        urow = None

    if urow:
        top_id = str(urow.get("id") or "")
        thumb_key = str(urow.get("thumbnail_r2_key") or "").strip()
        if thumb_key:
            try:
                nk = _normalize_r2_key(thumb_key)
                if nk and r2_object_exists(nk):
                    u = generate_presigned_download_url(nk, ttl=min(presign_ttl_seconds(), 86_400)) or ""
                    if u and allowed_promo_image_url(u):
                        out["seed_image_https"] = u
                        out["top_upload_id"] = top_id
            except Exception:
                pass

    if not out["seed_image_https"] and out["platforms"]:
        out["seed_image_https"] = str(out["platforms"][0].get("avatar_url") or "")

    out["fingerprint"] = _visual_fingerprint(slugs, top_id)
    pnames = sorted({(s.split(":")[0] if ":" in s else s) for s in slugs if s})
    out["platform_labels"] = ", ".join(pnames)[:120] if pnames else ""
    return out


def _sanitize_prompt_fragment(text: str, max_len: int = 400) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return ""
    # Strip emails and obvious PII patterns from image prompts
    s = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[redacted]", s, flags=re.I)
    return s[:max_len]


def variant_id_for_promo(
    *,
    entity_kind: str,
    entity_id: str,
    segment_bucket: str,
    image_prompt: str,
) -> str:
    raw = f"{PROMO_SPEC_VERSION}|{entity_kind}|{entity_id}|{segment_bucket}|{image_prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


async def _count_runs_last_hour(conn: asyncpg.Connection) -> int:
    return int(
        await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM marketing_promo_media_runs
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            """
        )
        or 0
    )


async def _log_run(
    conn: asyncpg.Connection,
    *,
    entity_kind: str,
    entity_id: Optional[str],
    variant_id: str,
    http_status: Optional[int],
    ok: bool,
    detail: str = "",
) -> None:
    try:
        await conn.execute(
            """
            INSERT INTO marketing_promo_media_runs (
                id, entity_kind, entity_id, variant_id, http_status, ok, detail
            )
            VALUES ($1::uuid, $2, $3::uuid, $4, $5, $6, $7)
            """,
            uuid.uuid4(),
            (entity_kind or "unknown")[:32],
            uuid.UUID(str(entity_id)) if entity_id else None,
            (variant_id or "")[:64],
            http_status,
            ok,
            (detail or "")[:2000],
        )
    except Exception as e:
        logger.debug("marketing_promo_media_runs insert failed: %s", e)


def _promo_media_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            j = json.loads(raw)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}


async def generate_and_store_promo_image(
    conn: asyncpg.Connection,
    *,
    entity_kind: str,
    entity_id: str,
    promo_media: Dict[str, Any],
    title: str,
    objective: str,
    segment_bucket: str,
    marketing_user_id: Optional[str] = None,
    user_visual: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Call Pikzels /v2/thumbnail/image (recreate-style), upload to R2, return (r2_key, presigned_url, variant_id).
    On skip/failure returns (None, None, variant_id or "").

    When ``marketing_user_id`` is set and user visual context is enabled for this campaign,
    uses that user's platform avatars + strongest upload thumbnail as **seed only** (comms/marketing —
    never the upload pipeline). Output is stored under ``marketing/promo/user_campaign/...`` and the
    campaign row is **not** overwritten (per-recipient art).
    """
    if not promo_media_enabled():
        return None, None, ""
    if not pikzels_api_key():
        logger.info("marketing promo media: PIKZELS_API_KEY not set; skip")
        return None, None, ""

    pm = promo_media or {}
    mode = str(pm.get("mode") or "none").strip().lower()
    if mode not in ("pikzels_static", "pikzels", "image"):
        return None, None, ""

    uid = (marketing_user_id or "").strip()
    use_user_visual = bool(uid) and user_visual_context_enabled_for_campaign(pm)
    visual: Optional[Dict[str, Any]] = dict(user_visual) if isinstance(user_visual, dict) else None
    if use_user_visual and visual is None:
        try:
            visual = await fetch_user_marketing_visual_context(conn, uid)
        except Exception as e:
            logger.debug("user marketing visual context failed: %s", e)
            visual = {}

    base_segment = (segment_bucket or "all").strip()[:120] or "all"
    vdict = visual if isinstance(visual, dict) else {}
    if use_user_visual:
        uflat = uid.replace("-", "")[:12]
        fp = str(vdict.get("fingerprint") or "na")[:16]
        segment = f"{base_segment}_u{uflat}_{fp}"[:120]
    else:
        segment = base_segment

    custom = _sanitize_prompt_fragment(str(pm.get("image_prompt") or ""), 500)
    base_prompt = (
        custom
        or _sanitize_prompt_fragment(f"{title}. {objective}", 500)
        or "UploadM8 multi-platform video publishing — bold orange accent, dark background, modern SaaS marketing hero, no small text, no logos except subtle geometric shapes."
    )
    tier_hint = segment.replace("_", " ") if segment != "all" else "creators and teams"
    visual_extra = ""
    if use_user_visual:
        plab = str(vdict.get("platform_labels") or "").strip()
        if plab:
            visual_extra += f" Connected platforms: {plab}. "
        if str(vdict.get("top_upload_id") or "").strip():
            visual_extra += (
                " Let the seed image inform composition and energy from the creator's "
                "strongest-performing published upload — abstract treatment only, no readable text, "
                "no third-party logos from the reference. "
            )
    full_prompt = (
        f"16:9 marketing hero image for a social video ops product. Audience: {tier_hint}. "
        f"Scene: {base_prompt} "
        f"{visual_extra}"
        "Style: cinematic lighting, high contrast, professional, no readable text, no watermarks, no faces."
    )

    vid = variant_id_for_promo(
        entity_kind=("user_campaign" if use_user_visual else entity_kind),
        entity_id=f"{entity_id}_{uid}" if use_user_visual else entity_id,
        segment_bucket=segment,
        image_prompt=full_prompt,
    )

    persist_row = entity_kind in ("campaign", "announcement") and not use_user_visual

    if persist_row:
        existing_key = str(pm.get("last_r2_key") or "").strip()
        cached_seg = str(pm.get("segment_bucket") or "").strip()
        if (
            existing_key
            and cached_seg == segment
            and r2_object_exists(existing_key)
        ):
            try:
                url = generate_presigned_download_url(_normalize_r2_key(existing_key), ttl=presign_ttl_seconds())
                if url and allowed_promo_image_url(url):
                    return existing_key, url or None, str(pm.get("variant_id") or vid)
            except Exception:
                pass
    else:
        ext0 = "jpg"
        r2_try = _normalize_r2_key(f"marketing/promo/user_campaign/{uid}/{entity_id}/{vid}.{ext0}")
        if r2_try and r2_object_exists(r2_try):
            try:
                url = generate_presigned_download_url(r2_try, ttl=presign_ttl_seconds())
                if url and allowed_promo_image_url(url):
                    return r2_try, url or None, vid
            except Exception:
                pass
        r2_try_png = _normalize_r2_key(f"marketing/promo/user_campaign/{uid}/{entity_id}/{vid}.png")
        if r2_try_png and r2_object_exists(r2_try_png):
            try:
                url = generate_presigned_download_url(r2_try_png, ttl=presign_ttl_seconds())
                if url and allowed_promo_image_url(url):
                    return r2_try_png, url or None, vid
            except Exception:
                pass

    if await _count_runs_last_hour(conn) >= max_pikzels_per_hour():
        logger.warning("marketing promo media: hourly cap %s reached", max_pikzels_per_hour())
        await _log_run(conn, entity_kind=entity_kind, entity_id=entity_id, variant_id=vid, http_status=None, ok=False, detail="hourly_cap")
        return None, None, vid

    if use_user_visual and str(vdict.get("seed_image_https") or "").startswith("https://"):
        seed = str(vdict["seed_image_https"])
    else:
        seed = seed_image_url()
    body: Dict[str, Any] = {
        "prompt": full_prompt[:1000],
        "image_url": seed,
        "model": str(pm.get("model") or "pkz_4")[:32],
        "format": str(pm.get("format") or "16:9")[:16],
        "image_weight": str(pm.get("image_weight") or "low")[:16],
    }

    status, data = await pikzels_v2_post("/v2/thumbnail/image", body)
    if status >= 400 or not isinstance(data, dict):
        await _log_run(
            conn,
            entity_kind=entity_kind,
            entity_id=entity_id,
            variant_id=vid,
            http_status=status,
            ok=False,
            detail=str(data)[:500],
        )
        return None, None, vid

    img_url = _pikzels_extract_image_url(data)
    if not img_url:
        await _log_run(conn, entity_kind=entity_kind, entity_id=entity_id, variant_id=vid, http_status=status, ok=False, detail="no_image_url")
        return None, None, vid

    raw: bytes = b""
    ext = "jpg"
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            r = await client.get(img_url)
            r.raise_for_status()
            raw = r.content
            ext = "png" if "png" in (r.headers.get("content-type") or "").lower() else "jpg"
    except Exception as e:
        await _log_run(conn, entity_kind=entity_kind, entity_id=entity_id, variant_id=vid, http_status=status, ok=False, detail=f"download:{e}")
        return None, None, vid

    if not raw or len(raw) > 12_000_000:
        await _log_run(conn, entity_kind=entity_kind, entity_id=entity_id, variant_id=vid, http_status=status, ok=False, detail="bad_bytes")
        return None, None, vid
    if use_user_visual and uid:
        r2_key = _normalize_r2_key(f"marketing/promo/user_campaign/{uid}/{entity_id}/{vid}.{ext}")
    else:
        r2_key = _normalize_r2_key(f"marketing/promo/{entity_kind}/{entity_id}/{vid}.{ext}")
    try:
        put_object_bytes(r2_key, raw, "image/png" if ext == "png" else "image/jpeg")
    except Exception as e:
        await _log_run(conn, entity_kind=entity_kind, entity_id=entity_id, variant_id=vid, http_status=status, ok=False, detail=f"r2:{e}")
        return None, None, vid

    presigned = ""
    try:
        presigned = generate_presigned_download_url(r2_key, ttl=presign_ttl_seconds()) or ""
    except Exception as e:
        logger.warning("promo media presign failed: %s", e)

    detail_run = "ok_user_visual" if use_user_visual else "ok"
    await _log_run(
        conn,
        entity_kind=("user_campaign" if use_user_visual else entity_kind),
        entity_id=entity_id,
        variant_id=vid,
        http_status=status,
        ok=True,
        detail=json.dumps({"user_id": uid, "base": detail_run})[:1900] if use_user_visual else detail_run,
    )

    if persist_row:
        new_pm = dict(pm)
        new_pm["mode"] = mode
        new_pm["last_r2_key"] = r2_key
        new_pm["variant_id"] = vid
        new_pm["segment_bucket"] = segment
        new_pm["spec_version"] = PROMO_SPEC_VERSION
        new_pm["updated_at"] = datetime.now(timezone.utc).isoformat()
        if custom:
            new_pm["image_prompt"] = custom

        if entity_kind == "campaign":
            await conn.execute(
                "UPDATE marketing_campaigns SET promo_media = $2::jsonb, updated_at = NOW() WHERE id = $1::uuid",
                uuid.UUID(entity_id),
                json.dumps(new_pm),
            )
        elif entity_kind == "announcement":
            await conn.execute(
                "UPDATE announcements SET promo_media = $2::jsonb WHERE id = $1::uuid",
                uuid.UUID(entity_id),
                json.dumps(new_pm),
            )

    return r2_key, presigned or None, vid


async def resolve_promo_presigned_url(conn: asyncpg.Connection, promo_media: Any) -> Optional[str]:
    """Return a fresh presigned URL for an existing R2 key, or None."""
    pm = _promo_media_dict(promo_media)
    key = str(pm.get("last_r2_key") or "").strip()
    if not key:
        return None
    nk = _normalize_r2_key(key)
    if not nk or not r2_object_exists(nk):
        return None
    try:
        u = generate_presigned_download_url(nk, ttl=presign_ttl_seconds()) or None
        if u and allowed_promo_image_url(u):
            return u
        return None
    except Exception:
        return None


async def resolve_or_generate_campaign_promo_url(
    conn: asyncpg.Connection,
    *,
    campaign_id: str,
    user_tier: str,
    user_id: Optional[str] = None,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Return (presigned_https_url_or_none, variant_id, extras) for this user's segment; may call Pikzels when enabled.
    ``extras`` may include ``marketing_platform_avatars_html`` for email templates when user visual context runs.

    When ``user_id`` is set and user visual context is enabled (``promo_media.user_context`` or
    ``MARKETING_PROMO_USER_VISUAL_CONTEXT``), generates **per-user** hero art from platform avatars +
    strongest upload thumbnail seed; does not overwrite the shared campaign ``promo_media`` row.
    """
    if not promo_media_enabled():
        return None, "", {}
    row = await conn.fetchrow(
        """
        SELECT name, objective, promo_media, targeting
        FROM marketing_campaigns
        WHERE id = $1::uuid
        """,
        uuid.UUID(campaign_id),
    )
    if not row:
        return None, "", {}
    title = str(row["name"] or "")
    objective = str(row["objective"] or "")
    tg = row["targeting"]
    if isinstance(tg, str):
        try:
            tg = json.loads(tg)
        except Exception:
            tg = {}
    if not isinstance(tg, dict):
        tg = {}
    pm0 = _promo_media_dict(row["promo_media"])
    mode = str(pm0.get("mode") or "none").strip().lower()
    if mode not in ("pikzels_static", "pikzels", "image"):
        return None, "", {}
    seg = user_segment_bucket(user_tier, tg)
    uid = (user_id or "").strip()
    user_visual_prefetch: Optional[Dict[str, Any]] = None
    if uid and user_visual_context_enabled_for_campaign(pm0):
        try:
            user_visual_prefetch = await fetch_user_marketing_visual_context(conn, uid)
        except Exception:
            user_visual_prefetch = {}
    _, url, vid = await generate_and_store_promo_image(
        conn,
        entity_kind="campaign",
        entity_id=campaign_id,
        promo_media=pm0,
        title=title,
        objective=objective,
        segment_bucket=seg,
        marketing_user_id=uid if user_visual_context_enabled_for_campaign(pm0) else None,
        user_visual=user_visual_prefetch,
    )
    extras: Dict[str, Any] = {}
    if user_visual_prefetch and user_visual_context_enabled_for_campaign(pm0):
        row_html = build_platform_avatars_html_row(user_visual_prefetch.get("platforms"))
        if row_html:
            extras["marketing_platform_avatars_html"] = row_html
    if url and allowed_promo_image_url(url):
        return url, vid, extras
    pm1 = await conn.fetchval(
        "SELECT promo_media FROM marketing_campaigns WHERE id = $1::uuid",
        uuid.UUID(campaign_id),
    )
    u2 = await resolve_promo_presigned_url(conn, pm1)
    if u2 and allowed_promo_image_url(u2):
        pm1d = _promo_media_dict(pm1)
        return u2, str(pm1d.get("variant_id") or vid or ""), extras
    return None, vid or str(pm0.get("variant_id") or ""), extras


def sign_tracking_token(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64 = base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")
    sig = hmac.new(_MARKETING_TRACK_SECRET.encode("utf-8"), b64.encode("ascii"), hashlib.sha256).hexdigest()[:32]
    return f"{b64}.{sig}"


def verify_tracking_token(token: str) -> Optional[Dict[str, Any]]:
    if not token or "." not in token:
        return None
    b64, sig = token.rsplit(".", 1)
    if len(sig) != 32:
        return None
    pad = "=" * ((4 - len(b64) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(b64 + pad)
        expect = hmac.new(_MARKETING_TRACK_SECRET.encode("utf-8"), b64.encode("ascii"), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(expect, sig):
            return None
        j = json.loads(raw.decode("utf-8"))
        return j if isinstance(j, dict) else None
    except Exception:
        return None


def tracking_pixel_url_for_delivery(delivery_id: str, user_id: str, campaign_id: Optional[str], variant_id: str = "") -> str:
    base = (os.environ.get("FRONTEND_URL") or "https://app.uploadm8.com").rstrip("/")
    # Prefer API origin for tracking (Mailgun fetches pixel from public URL)
    api = (os.environ.get("PUBLIC_API_BASE_URL") or os.environ.get("API_PUBLIC_URL") or "").strip().rstrip("/")
    if not api:
        api = base.replace("app.", "auth.").replace("//app.", "//auth.")
    tok = sign_tracking_token(
        {
            "d": str(delivery_id),
            "u": str(user_id),
            "c": str(campaign_id or ""),
            "v": (variant_id or "")[:64],
            "t": "open",
        }
    )
    return f"{api}/api/marketing/o/{tok}"


def marketing_one_click_unsub_url(user_id: str) -> str:
    """Signed List-Unsubscribe-Post URL that opts the user out of email marketing."""
    base = (os.environ.get("FRONTEND_URL") or "https://app.uploadm8.com").rstrip("/")
    api = (os.environ.get("PUBLIC_API_BASE_URL") or os.environ.get("API_PUBLIC_URL") or "").strip().rstrip("/")
    if not api:
        api = base.replace("app.", "auth.").replace("//app.", "//auth.")
    tok = sign_tracking_token({"u": str(user_id), "t": "unsub_email"})
    return f"{api}/api/marketing/unsubscribe/{tok}"
