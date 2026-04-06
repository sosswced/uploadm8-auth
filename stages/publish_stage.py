"""
UploadM8 Publish Stage
======================
Publish per-platform transcoded videos to social media platforms.

Platform-specific publish flows:
  TikTok   - Content Posting API v2: init upload -> PUT binary -> publish_id
  YouTube  - Resumable upload: init -> PUT binary -> video_id; thumbnails.set for one custom
              thumbnail. Extra A/B JPEGs are uploaded to R2 by the worker for Studio
              “Test & Compare” (no public thumbnailTests API on Data API v3 as of 2025).
  Instagram - Graph API Reels: create container (video_url) -> poll status -> publish
  Facebook - Graph API: multipart upload or URL-based Reels

Key behaviors:
  - Each platform gets its OWN transcoded video (ctx.platform_videos[platform])
  - Meta platforms (IG/FB) use presigned R2 URLs when public URL needed
  - Per-platform error isolation: one failure does NOT kill others
  - Ledger row per attempt (publish_attempts table)
  - verify_stage later confirms "accepted" -> live (TikTok, YouTube, Instagram, Facebook)

Exports used by verify_stage:
  - decrypt_token(token_row) -> dict or None
  - init_enc_keys() -> None
"""

import os
import json
import asyncio
import logging
import base64
import binascii
import math
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import asyncpg

try:
    from botocore.exceptions import BotoCoreError as _R2BotoCoreError
    from botocore.exceptions import ClientError as _R2ClientError
except ImportError:  # pragma: no cover
    _R2BotoCoreError = OSError
    _R2ClientError = OSError

from .safe_parse import json_dict
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import PublishError, ErrorCode
from .context import JobContext, PlatformResult, build_multimodal_scene_digest

from services.meta_oauth import require_facebook_publish, require_instagram_publish
from . import db as db_stage
from . import r2 as r2_stage
from .outbound_rl import outbound_slot
from .platform_tokens import platform_tokens_db_key
from .redis_publish_guard import (
    publish_circuit_open,
    publish_record_result,
    publish_wait_slot,
)


logger = logging.getLogger("uploadm8-worker")

_TOKEN_PARSE_ERRORS = (
    json.JSONDecodeError,
    ValueError,
    TypeError,
    KeyError,
    OSError,
    binascii.Error,
)
try:
    from cryptography.exceptions import InvalidTag as _TokenInvalidTag

    _TOKEN_PARSE_ERRORS = _TOKEN_PARSE_ERRORS + (_TokenInvalidTag,)
except ImportError:  # pragma: no cover
    pass

_PUBLISH_HTTP_JSON_ERRS = (
    httpx.HTTPError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)
_DB_PERSIST_ERRS = (
    asyncpg.PostgresError,
    asyncpg.InterfaceError,
    OSError,
    TimeoutError,
    TypeError,
    ValueError,
)
_PUBLISH_FILE_HTTP_ERRS = _PUBLISH_HTTP_JSON_ERRS + (OSError, PermissionError)

# Graph API version for Meta platforms (Instagram + Facebook)
META_API_VERSION = "v21.0"

# Instagram container polling
IG_POLL_INTERVAL = 5       # seconds between polls
IG_POLL_MAX_ATTEMPTS = 36  # 3 minutes max (36 * 5s)

# Presigned URL expiry for Meta uploads (1 hour)
PRESIGNED_URL_EXPIRY = 3600


# =====================================================================
# Platform Thumbnail Push
# =====================================================================

def _get_platform_thumbnail_path(ctx, platform: str):
    """
    Return the best thumbnail path for a platform.
    Prefers platform-specific styled thumbnail (16:9 YouTube, 9:16 IG/FB) when available.
    """
    pm = ctx.output_artifacts.get("platform_thumbnail_map", "{}")
    plat_map = json_dict(pm, default={}, context="platform_thumbnail_map")
    path_str = plat_map.get(platform)
    if path_str:
        p = Path(path_str)
        if p.exists():
            return p
    tp = getattr(ctx, "thumbnail_path", None)
    if tp:
        p = Path(tp) if not isinstance(tp, Path) else tp
        if p.exists():
            return p
    return None


async def _push_thumbnail_to_platform(
    platform: str,
    video_id: str,
    thumbnail_path: Optional[Path],
    access_token: str,
    client: httpx.AsyncClient,
) -> bool:
    """
    Upload the local thumbnail JPEG to the platform as the video cover.
    Called right after a successful video publish.
    Non-fatal — a failure here never blocks the upload result.

    YouTube:   POST /thumbnails/set  (multipart, requires verified channel)
    TikTok:    Cover API not available to 3rd-party apps — skipped.
    Instagram: Cover must be set at container creation time (handled there).
    Facebook:  POST /{video_id}  with thumb= multipart field.
    """
    if not thumbnail_path or not thumbnail_path.exists():
        return False
    try:
        if platform == "youtube":
            with open(thumbnail_path, "rb") as f:
                thumb_bytes = f.read()
            resp = await client.post(
                f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
                f"?videoId={video_id}&uploadType=media",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "image/jpeg",
                    "Content-Length": str(len(thumb_bytes)),
                },
                content=thumb_bytes,
                timeout=60,
            )
            if resp.status_code in (200, 201):
                logger.info(f"YouTube thumbnail set: video_id={video_id}")
                return True
            logger.warning(f"YouTube thumbnail failed: {resp.status_code} {resp.text[:200]}")
            return False

        elif platform == "facebook":
            with open(thumbnail_path, "rb") as f:
                thumb_bytes = f.read()
            resp = await client.post(
                f"https://graph.facebook.com/{META_API_VERSION}/{video_id}",
                params={"access_token": access_token},
                files={"thumb": ("thumbnail.jpg", thumb_bytes, "image/jpeg")},
                timeout=60,
            )
            if resp.status_code == 200 and resp.json().get("success"):
                logger.info(f"Facebook thumbnail set: video_id={video_id}")
                return True
            logger.warning(f"Facebook thumbnail failed: {resp.status_code} {resp.text[:200]}")
            return False

        # TikTok: cover API restricted — platform auto-selects from video
        # Instagram: cover_url injected into container creation payload
        return False
    except asyncio.CancelledError:
        raise
    except (OSError, PermissionError, httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Thumbnail push %s failed (non-fatal): %s", platform, e)
        return False


# TikTok Content Posting: public → PUBLIC_TO_EVERYONE when the app is approved.
# Default is audited (public/live). Set TIKTOK_APP_AUDITED=false only for sandbox apps that
# cannot post publicly yet (maps public → SELF_ONLY to avoid unaudited_client errors).
_tiktok_aud_raw = os.environ.get("TIKTOK_APP_AUDITED", "true").strip().lower()
TIKTOK_APP_AUDITED = _tiktok_aud_raw not in (
    "0",
    "false",
    "no",
    "off",
    "unaudited",
    "sandbox",
)


def _tiktok_error_suggests_creator_only_fallback(code: Any, message: str, raw_text: str) -> bool:
    """True when TikTok init likely failed because PUBLIC_TO_EVERYONE is not allowed for this client/token."""
    blob = " ".join(
        [
            str(code or "").lower(),
            (message or "").lower(),
            (raw_text or "").lower(),
        ]
    )
    needles = (
        "unaudited",
        "not_audited",
        "private_account",
        "can_only_post",
        "only_post_to_private",
        "sandbox",
        "content-sharing-guidelines",
        "content sharing guidelines",
        "integration guidelines",
    )
    return any(n in blob for n in needles)


# =====================================================================
# Privacy Level Resolution — single source of truth
# =====================================================================

def resolve_privacy_level(canonical: str, platform: str) -> str:
    """Map a canonical UploadM8 privacy level to each platform's native value.

    Canonical levels:  "public" | "unlisted" | "private"
    Unknown inputs default to "public" behaviour for the given platform.

    Privacy matrix:

    Canonical │ TikTok                │ YouTube   │ Instagram │ Facebook
    ──────────┼───────────────────────┼───────────┼───────────┼──────────
    public    │ PUBLIC_TO_EVERYONE    │ public    │ public    │ EVERYONE
    unlisted  │ SELF_ONLY             │ unlisted  │ private   │ SELF
    private   │ MUTUAL_FOLLOW_FRIENDS │ private   │ private   │ SELF

    TikTok notes:
      - public   → PUBLIC_TO_EVERYONE when TIKTOK_APP_AUDITED (default); else SELF_ONLY for sandbox
      - unlisted → SELF_ONLY ("Only You" — what we use for pre-approval uploads)
      - private  → MUTUAL_FOLLOW_FRIENDS (friends / mutual followers)

    Instagram notes:
      - No per-post unlisted concept via Graph API; unlisted maps to private.

    Facebook notes:
      - SELF = only you (closest to unlisted/private).
      - private also maps to SELF since FB has no followers-only Reels API value.
    """
    level = (canonical or "public").strip().lower()
    if level not in ("public", "unlisted", "private"):
        level = "public"

    if platform == "tiktok":
        # Sandbox / unaudited apps cannot use PUBLIC_TO_EVERYONE (unaudited_client_can_only_post_to_private_accounts).
        if not TIKTOK_APP_AUDITED and level == "public":
            return "SELF_ONLY"
        return {
            "public":   "PUBLIC_TO_EVERYONE",
            "unlisted": "SELF_ONLY",              # "Only You" — safe for pre-approval testing
            "private":  "MUTUAL_FOLLOW_FRIENDS",  # friends / mutual follow
        }[level]

    if platform == "youtube":
        return {
            "public":   "public",
            "unlisted": "unlisted",
            "private":  "private",
        }[level]

    if platform == "instagram":
        # Instagram Reels via Graph API: no unlisted — map unlisted → private
        return {
            "public":   "public",
            "unlisted": "private",
            "private":  "private",
        }[level]

    if platform == "facebook":
        # Facebook Graph API privacy object value
        # private → SELF (no followers-only Reel API value available)
        return {
            "public":   "EVERYONE",
            "unlisted": "SELF",
            "private":  "SELF",
        }[level]

    # Unknown platform — return canonical as-is (safe default)
    return level


# =====================================================================
# Token Encryption (used by this stage + verify_stage)
# =====================================================================

TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")
_ENC_KEYS: Dict[str, bytes] = {}


def init_enc_keys():
    """Parse encryption keys from environment."""
    global _ENC_KEYS
    if not TOKEN_ENC_KEYS:
        return

    clean = TOKEN_ENC_KEYS.strip().strip('"').replace("\\n", "")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        kid, b64key = part.split(":", 1)
        try:
            raw = base64.b64decode(b64key.strip())
            if len(raw) == 32:
                _ENC_KEYS[kid.strip()] = raw
        except (binascii.Error, TypeError, ValueError) as e:
            logger.debug("Skipping invalid TOKEN_ENC_KEYS entry for kid=%r: %s", kid, e)


def decrypt_token_blob(blob: Any) -> dict:
    """Decrypt platform token blob."""
    if isinstance(blob, str):
        blob = json.loads(blob)

    kid = blob.get("kid", "v1")
    key = _ENC_KEYS.get(kid)
    if not key:
        raise ValueError(f"Unknown key id: {kid}")

    nonce = base64.b64decode(blob["nonce"])
    ciphertext = base64.b64decode(blob["ciphertext"])
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))


def decrypt_token(token_row: Any) -> Optional[dict]:
    """Best-effort decrypt/parse of a stored platform token row.

    Accepts:
      - dict: either already plaintext token data OR an encrypted blob dict
      - str: JSON string for either plaintext dict or encrypted blob

    Returns:
      - dict token payload (e.g., {"access_token": "..."})
      - None if missing/invalid

    NOTE: verify_stage imports this function directly.
    """
    if token_row is None:
        return None
    try:
        if isinstance(token_row, str):
            token_row = json.loads(token_row)
        if not isinstance(token_row, dict):
            return None

        # If this looks like an encrypted blob, decrypt it.
        if "ciphertext" in token_row and "nonce" in token_row:
            try:
                return decrypt_token_blob(token_row)
            except _TOKEN_PARSE_ERRORS as e:
                logger.debug("Token decrypt failed: %s", e)
                return None

        # Already plaintext token dict
        return token_row
    except _TOKEN_PARSE_ERRORS as e:
        logger.debug("Token parse failed: %s", e)
        return None


# =====================================================================
# Content Derivation
# =====================================================================

def _get_title(ctx: JobContext, platform: str = "") -> str:
    """Get the best available title for publishing (optional per-platform override)."""
    if hasattr(ctx, "get_effective_title"):
        return ctx.get_effective_title(platform) or (
            getattr(ctx, "ai_title", None)
            or getattr(ctx, "title", None)
            or getattr(ctx, "video_title", None)
            or getattr(ctx, "name", None)
            or f"UploadM8 {getattr(ctx, 'upload_id', '')}".strip()
        )
    return (
        getattr(ctx, "ai_title", None)
        or getattr(ctx, "title", None)
        or getattr(ctx, "video_title", None)
        or getattr(ctx, "name", None)
        or f"UploadM8 {getattr(ctx, 'upload_id', '')}".strip()
    )


def _get_caption(ctx: JobContext, platform: str = "") -> str:
    """Get the best available caption/description (optional per-platform override)."""
    if hasattr(ctx, "get_effective_caption"):
        return ctx.get_effective_caption(platform) or ""
    return (
        getattr(ctx, "ai_caption", None)
        or getattr(ctx, "caption", None)
        or getattr(ctx, "description", None)
        or ""
    )


def _tiktok_post_title(ctx: JobContext) -> str:
    """Build TikTok Content Posting `post_info.title` (max 150 chars).

    The API exposes a single title field. We must not use ``(hook_title or caption)``
    because a non-empty upload title would skip ``_build_platform_caption`` entirely,
    dropping platform-specific / always / merged hashtags.
    """
    body = _build_platform_caption(ctx, "tiktok").strip()
    hook = (_get_title(ctx, "tiktok") or "").strip()
    if body:
        if hook and not body.lower().startswith(hook.lower()):
            merged = f"{hook}\n\n{body}".strip()
        else:
            merged = body
        return merged[:150]
    return hook[:150]


def _get_hashtags(ctx: JobContext, platform: str = "") -> str:
    """Get hashtag string for publishing.

    Uses get_effective_hashtags(platform) which merges in order:
    always_hashtags → platform_hashtags for this platform → base → AI.
    Same pipeline for all sources; blocked tags filtered throughout.
    """
    if hasattr(ctx, "get_effective_hashtags"):
        merged = ctx.get_effective_hashtags(platform=platform)
    else:
        merged = getattr(ctx, "ai_hashtags", None) or getattr(ctx, "hashtags", None) or []
    if isinstance(merged, list):
        tags = [str(t).strip() for t in merged if str(t).strip()]
    else:
        import re as _re
        tags = [p for p in _re.split(r"[\s,]+", str(merged).strip()) if p]
    normalised = [t if t.startswith("#") else f"#{t}" for t in tags if t]
    return " ".join(normalised) if normalised else ""


def _build_platform_caption(ctx: JobContext, platform: str) -> str:
    """Build caption + hashtags for a specific platform.

    Includes platform-specific hashtags from user_settings.
    Respects the user's hashtag_position preference (start / end).
    """
    caption = _get_caption(ctx, platform)
    if not caption:
        caption = _fallback_caption_from_context(ctx, platform)
    hashtags = _get_hashtags(ctx, platform=platform)

    if not hashtags:
        return caption or ""
    if not caption:
        return hashtags

    # Resolve hashtag position from user preferences
    hashtag_position = "end"
    try:
        us = ctx.user_settings or {}
        hashtag_position = (
            us.get("hashtagPosition")
            or us.get("hashtag_position")
            or "end"
        ).lower()
        if hashtag_position not in ("start", "end"):
            hashtag_position = "end"
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        logger.debug("Could not resolve hashtag position from user settings: %s", e)

    if hashtag_position == "start":
        return f"{hashtags}\n\n{caption}"
    return f"{caption}\n\n{hashtags}"


def _fallback_caption_from_context(ctx: JobContext, platform: str) -> str:
    """
    Deterministic fallback when AI/manual caption is empty.
    Keeps publish stage grounded in multimodal evidence (geo/audio/vision).
    """
    ac = getattr(ctx, "audio_context", None) or {}
    vc = getattr(ctx, "vision_context", None) or {}
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)

    parts: list[str] = []
    if getattr(ctx, "location_name", None):
        parts.append(f"Shot near {ctx.location_name}")
    if tel and getattr(tel, "location_road", None):
        parts.append(f"route: {tel.location_road}")
    if tel and getattr(tel, "max_speed_mph", 0):
        parts.append(f"peak {float(getattr(tel, 'max_speed_mph', 0)):.0f} mph")

    labels = list(vc.get("label_names") or [])
    if labels:
        parts.append("scene: " + ", ".join(str(x) for x in labels[:4]))

    if ac.get("music_detected") and (ac.get("music_title") or ac.get("music_artist")):
        mt = ac.get("music_title") or ""
        ma = ac.get("music_artist") or ""
        parts.append(f"soundtrack: {mt} {('- ' + ma) if ma else ''}".strip())
    elif ac.get("top_sound_class"):
        parts.append(f"audio vibe: {ac.get('top_sound_class')}")

    if parts:
        base = " | ".join(p for p in parts if p).strip()
        return _format_platform_fallback_caption(base[:600], platform)

    # Last resort: compact multimodal digest line
    digest = (build_multimodal_scene_digest(ctx, max_chars=600) or "").strip()
    if digest:
        return _format_platform_fallback_caption(digest.replace("\n", " | ")[:600], platform)
    return ""


def _format_platform_fallback_caption(base: str, platform: str) -> str:
    """Platform-native stylistic wrapper for deterministic fallback text."""
    txt = (base or "").strip().strip("|")
    if not txt:
        return ""

    p = (platform or "").lower()
    if p == "tiktok":
        # Hook-first short style
        return f"POV: {txt}"[:600]
    if p == "youtube":
        # Description-style sentence framing
        return f"Scene breakdown: {txt}"[:1200]
    if p in ("instagram", "facebook"):
        # Slightly aesthetic/social phrasing
        return f"{txt} | captured in the moment."[:700]
    return txt[:600]


def _get_video_public_url(ctx: JobContext, platform: str) -> Optional[str]:
    """Get a publicly accessible URL for the platform's video.

    Priority:
      1. R2 public URL (if R2_PUBLIC_URL configured)
      2. R2 presigned URL (always works, expires in 1 hour)
      3. None (no URL available)
    """
    # Check if we have a processed asset R2 key for this platform
    r2_key = None

    # Try from output_artifacts stored during upload stage
    assets_json = ctx.output_artifacts.get("processed_assets", "{}")
    assets = json_dict(assets_json, default={}, context="processed_assets")
    r2_key = assets.get(platform)

    # Fallback: construct expected key
    if not r2_key:
        r2_key = f"processed/{ctx.user_id}/{ctx.upload_id}/{platform}.mp4"

    # Try public URL first (fastest, no expiry)
    public_url = r2_stage.get_public_url(r2_key)
    if public_url:
        return public_url

    # Fall back to presigned URL (always works)
    try:
        return r2_stage.generate_presigned_url(r2_key, expires=PRESIGNED_URL_EXPIRY)
    except (_R2BotoCoreError, _R2ClientError, OSError, TypeError, ValueError, RuntimeError) as e:
        logger.warning("Could not generate presigned URL for %s: %s", platform, e)
        return None


# =====================================================================
# Platform Publishers
# =====================================================================


async def _refresh_tiktok_token(
    token_data: dict, db_pool=None, user_id: str = None, token_row_id: Optional[str] = None
) -> dict:
    """Refresh a TikTok access token using the stored refresh_token.
    TikTok access tokens expire after 24 hours; refresh tokens after 365 days.
    Persists the new token blob to DB if db_pool and user_id are provided.
    """
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        logger.warning("TikTok: No refresh_token stored, cannot refresh - user must reconnect")
        return token_data

    client_key = os.environ.get("TIKTOK_CLIENT_KEY", "")
    client_secret = os.environ.get("TIKTOK_CLIENT_SECRET", "")

    if not client_key or not client_secret:
        logger.warning("TikTok: Missing TIKTOK_CLIENT_KEY/SECRET env vars, cannot refresh token")
        return token_data

    try:
        async with outbound_slot("tiktok"):
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://open.tiktokapis.com/v2/oauth/token/",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "client_key": client_key,
                        "client_secret": client_secret,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    },
                )
                if resp.status_code == 200:
                    new_tokens = resp.json()
                    logger.info("TikTok: Token refreshed successfully")
                    new_access  = new_tokens.get("access_token", token_data["access_token"])
                    new_refresh = new_tokens.get("refresh_token", refresh_token)
                    new_open_id = new_tokens.get("open_id") or token_data.get("open_id")
                    updated = {
                        **token_data,
                        "access_token":  new_access,
                        "refresh_token": new_refresh,
                        "expires_at":    new_tokens.get("expires_in"),
                    }
                    if new_open_id:
                        updated["open_id"] = new_open_id
                    # Persist back to DB so future jobs use the fresh token
                    if db_pool and user_id:
                        try:
                            await db_stage.save_refreshed_token(
                                db_pool,
                                user_id=user_id,
                                platform="tiktok",
                                access_token=new_access,
                                refresh_token=new_refresh,
                                open_id=new_open_id,
                                token_row_id=token_row_id,
                            )
                        except _DB_PERSIST_ERRS as save_err:
                            logger.warning("TikTok: Failed to persist refreshed token: %s", save_err)
                    return updated
                else:
                    logger.warning(f"TikTok: Token refresh failed: {resp.status_code} {resp.text[:200]}")
                    return token_data
    except asyncio.CancelledError:
        raise
    except _PUBLISH_HTTP_JSON_ERRS as e:
        logger.warning("TikTok: Token refresh exception: %s", e)
        return token_data



async def _refresh_meta_token(
    token_data: dict,
    platform: str,
    db_pool=None,
    user_id: str = None,
    token_row_id: Optional[str] = None,
) -> dict:
    """Refresh a Meta (Instagram/Facebook) Page access token.
    
    Meta Page tokens obtained via /me/accounts are long-lived (~60 days).
    They are renewed by exchanging the stored user access token for a new
    long-lived token via the fb_exchange_token flow, then re-fetching the Page token.
    
    If the user token has expired entirely, returns original token_data and logs a warning
    so publish proceeds and fails with a clear error rather than silently skipping.
    """
    access_token = token_data.get("access_token")
    if not access_token:
        return token_data

    app_id = os.environ.get("META_APP_ID", "") or os.environ.get("FACEBOOK_CLIENT_ID", "")
    app_secret = os.environ.get("META_APP_SECRET", "") or os.environ.get("FACEBOOK_CLIENT_SECRET", "")

    if not app_id or not app_secret:
        logger.warning(f"{platform}: Missing META_APP_ID/SECRET env vars, cannot refresh token")
        return token_data

    try:
        async with outbound_slot("meta"):
            async with httpx.AsyncClient(timeout=30) as client:
                # Step 1: Exchange current token for a fresh long-lived user token
                exchange_resp = await client.get(
                    "https://graph.facebook.com/v18.0/oauth/access_token",
                    params={
                        "grant_type": "fb_exchange_token",
                        "client_id": app_id,
                        "client_secret": app_secret,
                        "fb_exchange_token": access_token,
                    },
                )
                if exchange_resp.status_code != 200:
                    logger.warning(
                        f"{platform}: Meta token exchange failed: "
                        f"{exchange_resp.status_code} {exchange_resp.text[:200]}"
                    )
                    return token_data
    
                new_user_token = exchange_resp.json().get("access_token")
                if not new_user_token:
                    logger.warning(f"{platform}: Meta exchange returned no access_token")
                    return token_data
    
                logger.info(f"{platform}: Meta user token refreshed successfully")
    
                # Step 2: For Instagram/Facebook Page tokens, re-fetch the Page token
                # Page tokens derived from a long-lived user token do not expire,
                # so re-fetching gives us a stable non-expiring page token.
                page_id = token_data.get("page_id") or token_data.get("ig_user_id")
    
                if platform == "facebook" and page_id:
                    pages_resp = await client.get(
                        "https://graph.facebook.com/v18.0/me/accounts",
                        params={"access_token": new_user_token, "fields": "id,access_token"},
                    )
                    if pages_resp.status_code == 200:
                        pages = pages_resp.json().get("data", [])
                        matching = next((p for p in pages if p.get("id") == str(page_id)), None)
                        if matching:
                            new_page_token = matching.get("access_token", new_user_token)
                            updated = {
                                **token_data,
                                "access_token": new_page_token,
                                "expires_at": None,
                            }
                            if db_pool and user_id:
                                try:
                                    await db_stage.save_refreshed_token(
                                        db_pool,
                                        user_id=user_id,
                                        platform=platform,
                                        access_token=new_page_token,
                                        extra_fields={"page_id": str(page_id)},
                                        token_row_id=token_row_id,
                                    )
                                except _DB_PERSIST_ERRS as save_err:
                                    logger.warning("%s: Failed to persist refreshed token: %s", platform, save_err)
                            return updated
    
                # ── Refresh page token for Instagram and Facebook ───────────────
                # Instagram Insights API requires a PAGE token (not user token).
                # Always fetch /me/accounts to get fresh page tokens for both
                # platforms — even when ig_user_id / page_id are already known.
                pages_resp = await client.get(
                    "https://graph.facebook.com/v18.0/me/accounts",
                    params={"access_token": new_user_token, "fields": "id,name,access_token"},
                )
    
                recovered_ig_user_id = token_data.get("ig_user_id") or token_data.get("instagram_user_id")
                recovered_page_id    = token_data.get("page_id") or token_data.get("facebook_page_id")
                recovered_page_token = new_user_token  # fallback

                if pages_resp.status_code == 200:
                    pages = pages_resp.json().get("data", [])

                    if platform == "instagram":
                        # Find the page whose instagram_business_account matches ig_user_id
                        for page in pages:
                            page_id_tmp    = page.get("id")
                            page_token_tmp = page.get("access_token", new_user_token)
                            if not page_id_tmp:
                                continue
                            ig_resp = await client.get(
                                f"https://graph.facebook.com/v18.0/{page_id_tmp}",
                                params={
                                    "access_token": page_token_tmp,
                                    "fields": "instagram_business_account",
                                },
                            )
                            if ig_resp.status_code == 200:
                                ig_biz = ig_resp.json().get("instagram_business_account")
                                if ig_biz and ig_biz.get("id"):
                                    found_ig_id = ig_biz["id"]
                                    if not recovered_ig_user_id or found_ig_id == str(recovered_ig_user_id):
                                        recovered_ig_user_id = found_ig_id
                                        recovered_page_token = page_token_tmp
                                        logger.info(
                                            f"instagram: Refreshed page token for ig_user_id={recovered_ig_user_id} "
                                            f"via Page '{page.get('name', '?')}'"
                                        )
                                        break

                    if platform == "facebook":
                        if recovered_page_id:
                            matching = next((p for p in pages if p.get("id") == str(recovered_page_id)), None)
                            if matching:
                                recovered_page_token = matching.get("access_token", new_user_token)
                        elif pages:
                            first = pages[0]
                            recovered_page_id    = first["id"]
                            recovered_page_token = first.get("access_token", new_user_token)
                            logger.info(
                                f"facebook: Recovered page_id={recovered_page_id} "
                                f"from Page '{first.get('name', '?')}' during token refresh"
                            )

                # Both platforms always use the page token (not user token)
                updated = {
                    **token_data,
                    "access_token": recovered_page_token,
                    "expires_at":   None,
                }
                if platform == "instagram" and recovered_ig_user_id:
                    updated["ig_user_id"] = recovered_ig_user_id
                if platform == "facebook" and recovered_page_id:
                    updated["page_id"] = recovered_page_id

                if db_pool and user_id:
                    try:
                        extra = {}
                        if platform == "instagram" and recovered_ig_user_id:
                            extra["ig_user_id"] = str(recovered_ig_user_id)
                        if platform == "facebook" and recovered_page_id:
                            extra["page_id"] = str(recovered_page_id)
                        await db_stage.save_refreshed_token(
                            db_pool,
                            user_id=user_id,
                            platform=platform,
                            access_token=updated["access_token"],
                            extra_fields=extra or None,
                            token_row_id=token_row_id,
                        )
                    except _DB_PERSIST_ERRS as save_err:
                        logger.warning("%s: Failed to persist refreshed token: %s", platform, save_err)
                return updated
    
    except asyncio.CancelledError:
        raise
    except _PUBLISH_HTTP_JSON_ERRS as e:
        logger.warning("%s: Meta token refresh exception: %s", platform, e)
        return token_data



async def publish_to_tiktok(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    db_pool=None,
) -> PlatformResult:
    """Publish video to TikTok using Content Posting API v2.

    Flow: init upload -> PUT binary to upload_url -> returns publish_id
    """
    # Always refresh token first — TikTok access tokens expire after 24 hours
    token_data = await _refresh_tiktok_token(token_data, db_pool=db_pool, user_id=getattr(ctx, "user_id", None))

    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    # Single title field: include caption + hashtag merge (platform / always / AI)
    tiktok_title = _tiktok_post_title(ctx)

    try:
        file_size = video_path.stat().st_size

        # TikTok Content Posting API v2 chunk rules:
        #   - Each chunk must be between 5 MB and 64 MB (inclusive)
        #   - The LAST chunk may be smaller than chunk_size but must be ≥ 5 MB
        #     UNLESS it is the only chunk (single-chunk upload for files ≤ 64 MB)
        #   - TikTok validates: chunk_size * (total_chunk_count - 1) < video_size
        #                                ≤ chunk_size * total_chunk_count
        #
        # ROOT CAUSE OF PREVIOUS FAILURE:
        #   Using chunk_size=64 MB and computing total_chunk_count via ceiling
        #   division sent chunk_size * total_chunk_count = 128 MB for an 88 MB
        #   file.  TikTok sees the declared chunk_size doesn't divide evenly into
        #   video_size and rejects with "total chunk count is invalid".
        #
        # FIX: Compute total_chunk_count first (targeting ~32 MB chunks, well
        #   inside TikTok's 5-64 MB range), then derive chunk_size as
        #   ceil(file_size / total_chunk_count).  This guarantees:
        #     chunk_size * total_chunk_count ≥ file_size  (last chunk ≤ chunk_size)
        #     chunk_size * (total_chunk_count-1) < file_size  (total_count not over-counted)
        #   All chunks are ~30 MB — safely within both bounds on every file size.

        _64MB  = 64 * 1024 * 1024
        _5MB   = 5  * 1024 * 1024
        TARGET = 32 * 1024 * 1024   # aim for ~32 MB chunks

        if file_size <= _64MB:
            # Single-chunk upload — simplest case, always accepted
            chunk_size = file_size
            total_chunk_count = 1
        else:
            # Compute how many chunks we need at ~32 MB each
            total_chunk_count = math.ceil(file_size / TARGET)
            # Derive chunk_size so it evenly covers the whole file.
            # ceil division ensures chunk_size * total_chunk_count >= file_size.
            chunk_size = math.ceil(file_size / total_chunk_count)
            # Safety clamp — chunk_size must be in [5 MB, 64 MB]
            chunk_size = max(_5MB, min(_64MB, chunk_size))

        requested_privacy = getattr(ctx, "privacy", None) or "public"
        privacy_level = resolve_privacy_level(requested_privacy, "tiktok")
        attempted_creator_only_retry = False

        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Initialize upload
            init_resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={
                    "post_info": {
                        "title": tiktok_title,
                        "privacy_level": privacy_level,
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": file_size,
                        "chunk_size": chunk_size,
                        "total_chunk_count": total_chunk_count,
                    },
                },
            )

            if init_resp.status_code != 200:
                err_snippet = (init_resp.text or "")[:800]
                code: Any = None
                api_message = ""
                log_id = None
                try:
                    ej = init_resp.json()
                    err = ej.get("error") or {}
                    if isinstance(err, dict):
                        code = err.get("code")
                        api_message = str(err.get("message") or "")
                        log_id = err.get("log_id")
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                    logger.debug("publish_stage.tiktok: non-JSON error body (status=%s): %s", init_resp.status_code, e)

                orig_detail = (api_message.strip() or err_snippet)[:500]
                suggest_creator_only = _tiktok_error_suggests_creator_only_fallback(
                    code, api_message, err_snippet
                )
                if suggest_creator_only:
                    bits = [orig_detail]
                    if log_id:
                        bits.append(f"log_id={log_id}")
                    err_txt = (
                        "TikTok rejected public posting. Common causes even after portal approval: "
                        "using Sandbox Client Key/Secret instead of production, stale OAuth tokens "
                        "(disconnect and reconnect TikTok), or Content Posting not Live for this app. "
                        "Verify production credentials match TIKTOK_CLIENT_KEY/SECRET, portal "
                        "Products → Content Posting API shows Live, and TIKTOK_APP_AUDITED is not "
                        "false in server env (default is true). Or set TIKTOK_APP_AUDITED=false to "
                        "post creator-only until "
                        "TikTok accepts PUBLIC_TO_EVERYONE. Details: "
                        + " | ".join(bits)
                    )
                else:
                    err_txt = orig_detail

                # Retry init with SELF_ONLY when TikTok disallows public posting for this client.
                if (
                    (not attempted_creator_only_retry)
                    and suggest_creator_only
                    and str(privacy_level).upper() == "PUBLIC_TO_EVERYONE"
                ):
                    attempted_creator_only_retry = True
                    creator_only_level = resolve_privacy_level("unlisted", "tiktok")  # SELF_ONLY
                    logger.info(
                        "TikTok unaudited public rejection detected; retrying init with SELF_ONLY "
                        f"(requested_privacy={requested_privacy}, privacy_level={privacy_level}, retry_privacy={creator_only_level})"
                    )
                    init_resp2 = await client.post(
                        "https://open.tiktokapis.com/v2/post/publish/video/init/",
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "application/json; charset=UTF-8",
                        },
                        json={
                            "post_info": {
                                "title": tiktok_title,
                                "privacy_level": creator_only_level,
                            },
                            "source_info": {
                                "source": "FILE_UPLOAD",
                                "video_size": file_size,
                                "chunk_size": chunk_size,
                                "total_chunk_count": total_chunk_count,
                            },
                        },
                    )
                    if init_resp2.status_code == 200:
                        init_resp = init_resp2
                    else:
                        err_txt = (init_resp2.text[:500] or err_txt)[:500]

                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    http_status=init_resp.status_code,
                    error_code="INIT_FAILED",
                    error_message=f"Init failed: {err_txt}",
                )

            init_data = init_resp.json().get("data", {})
            upload_url = init_data.get("upload_url")
            publish_id = init_data.get("publish_id")

            if not upload_url:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error_code="NO_UPLOAD_URL",
                    error_message="No upload URL returned"
                )

            # Step 2: Upload video in chunks (or single chunk for small files)
            # TikTok requires Content-Range: bytes {start}-{end}/{total} per chunk
            with open(video_path, "rb") as f:
                for chunk_index in range(total_chunk_count):
                    chunk_start = chunk_index * chunk_size
                    chunk_data = f.read(chunk_size)
                    chunk_end = chunk_start + len(chunk_data) - 1

                    upload_resp = await client.put(
                        upload_url,
                        content=chunk_data,
                        headers={
                            "Content-Type": "video/mp4",
                            "Content-Length": str(len(chunk_data)),
                            "Content-Range": f"bytes {chunk_start}-{chunk_end}/{file_size}",
                        }
                    )

                    if upload_resp.status_code not in (200, 201, 206):
                        return PlatformResult(
                            platform="tiktok",
                            success=False,
                            http_status=upload_resp.status_code,
                            error_code="UPLOAD_FAILED",
                            error_message=f"Upload failed chunk {chunk_index+1}/{total_chunk_count}: {upload_resp.status_code}"
                        )

            logger.info(f"TikTok: uploaded {total_chunk_count} chunk(s), {file_size/1024/1024:.1f}MB total")

            logger.info(f"TikTok publish accepted: publish_id={publish_id}")
            return PlatformResult(
                platform="tiktok",
                success=True,
                publish_id=publish_id,
                verify_status="pending",
            )

    except asyncio.CancelledError:
        raise
    except _PUBLISH_FILE_HTTP_ERRS as e:
        logger.error("TikTok publish error: %s", e)
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e),
        )


async def _refresh_youtube_token(
    token_data: dict, db_pool=None, user_id: str = None, token_row_id: Optional[str] = None
) -> dict:
    """Attempt to refresh a YouTube access token using the stored refresh_token.
    Returns updated token_data dict with new access_token, or original if refresh fails.
    Persists updated token to DB if db_pool and user_id are provided.
    """
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        logger.warning("YouTube: No refresh_token stored, cannot refresh")
        return token_data

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "") or os.environ.get("YOUTUBE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "") or os.environ.get("YOUTUBE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        logger.warning("YouTube: Missing GOOGLE_CLIENT_ID/SECRET env vars, cannot refresh token")
        return token_data

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )
            if resp.status_code == 200:
                new_tokens = resp.json()
                logger.info("YouTube: Token refreshed successfully")
                new_access = new_tokens["access_token"]
                updated = {
                    **token_data,
                    "access_token": new_access,
                }
                if db_pool and user_id:
                    try:
                        await db_stage.save_refreshed_token(
                            db_pool,
                            user_id=user_id,
                            platform="youtube",
                            access_token=new_access,
                            token_row_id=token_row_id,
                        )
                    except _DB_PERSIST_ERRS as save_err:
                        logger.warning("YouTube: Failed to persist refreshed token: %s", save_err)
                return updated
            else:
                logger.warning(f"YouTube: Token refresh failed: {resp.status_code} {resp.text[:200]}")
                return token_data
    except asyncio.CancelledError:
        raise
    except _PUBLISH_HTTP_JSON_ERRS as e:
        logger.warning("YouTube: Token refresh exception: %s", e)
        return token_data


async def publish_to_youtube(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    db_pool=None,
) -> PlatformResult:
    """Publish video to YouTube Shorts using resumable upload.

    Flow: POST metadata (get Location header) -> PUT binary -> video_id
    """
    # Always refresh token first — YouTube access tokens expire after 1 hour
    token_data = await _refresh_youtube_token(token_data, db_pool=db_pool, user_id=getattr(ctx, "user_id", None))

    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(
            platform="youtube",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    title = _get_title(ctx, "youtube")[:100]
    caption = _build_platform_caption(ctx, "youtube")
    description = caption[:5000] if caption else ""

    try:
        file_size = video_path.stat().st_size
        async with httpx.AsyncClient(timeout=300) as client:
            # Step 1: Initialize resumable upload
            metadata = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "categoryId": "22"  # People & Blogs
                },
                "status": {
                    "privacyStatus": resolve_privacy_level(
                        getattr(ctx, "privacy", None) or "public",
                        "youtube",
                    ),
                    "selfDeclaredMadeForKids": False
                }
            }

            init_resp = await client.post(
                "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(file_size)
                },
                json=metadata
            )

            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    http_status=init_resp.status_code,
                    error_code="INIT_FAILED",
                    error_message=f"Init failed: {init_resp.text[:200]}"
                )

            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    error_code="NO_UPLOAD_URL",
                    error_message="No upload URL in response"
                )

            # Step 2: Upload video binary
            with open(video_path, "rb") as f:
                video_data = f.read()

            upload_resp = await client.put(
                upload_url,
                content=video_data,
                headers={"Content-Type": "video/mp4"}
            )

            if upload_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="youtube",
                    success=False,
                    http_status=upload_resp.status_code,
                    error_code="UPLOAD_FAILED",
                    error_message=f"Upload failed: {upload_resp.status_code}"
                )

            video_id = upload_resp.json().get("id")
            platform_url = f"https://youtube.com/shorts/{video_id}" if video_id else None

            logger.info(f"YouTube publish accepted: video_id={video_id}, url={platform_url}")
            # Push thumbnail to YouTube (non-fatal — requires verified channel)
            thumb_path = _get_platform_thumbnail_path(ctx, "youtube")
            if video_id and thumb_path:
                await _push_thumbnail_to_platform(
                    "youtube", video_id, thumb_path, access_token, client
                )
            return PlatformResult(
                platform="youtube",
                success=True,
                platform_video_id=video_id,
                platform_url=platform_url,
                verify_status="pending",
            )

    except asyncio.CancelledError:
        raise
    except _PUBLISH_FILE_HTTP_ERRS as e:
        logger.error("YouTube publish error: %s", e)
        return PlatformResult(
            platform="youtube",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e),
        )


async def publish_to_instagram(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    video_url: Optional[str] = None,
    db_pool=None,
) -> PlatformResult:
    """Publish video to Instagram Reels via Graph API.

    Flow (URL-based, required by Meta):
      1. POST /{ig_user_id}/media  (media_type=REELS, video_url, caption)
         -> creation_id
      2. Poll GET /{creation_id}?fields=status_code
         until status_code == "FINISHED"
      3. POST /{ig_user_id}/media_publish  (creation_id)
         -> media_id

    Requires:
      - ig_user_id in token_data (Instagram Business/Creator account ID)
      - Publicly accessible video URL (presigned R2 or public R2)
      - instagram_content_publish permission on the access token
    """
    # Refresh Meta token before use — long-lived tokens expire after ~60 days
    token_data = await _refresh_meta_token(
        token_data, "instagram", db_pool=db_pool, user_id=getattr(ctx, "user_id", None)
    )

    access_token = token_data.get("access_token")
    ig_user_id = (
        token_data.get("ig_user_id")
        or token_data.get("instagram_user_id")
        or token_data.get("instagram_page_id")
        or token_data.get("page_id")
    )

    if not access_token:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    if not ig_user_id:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="NO_IG_USER_ID",
            error_message="No Instagram user ID found in token data. Connect Instagram Business account."
        )

    _scope_msg = require_instagram_publish(token_data)
    if _scope_msg:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="META_SCOPE_MISSING",
            error_message=_scope_msg,
        )

    # Instagram Graph API requires a public URL - cannot upload binary directly
    if not video_url:
        video_url = _get_video_public_url(ctx, "instagram")

    if not video_url:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="NO_PUBLIC_URL",
            error_message="Instagram requires a public video URL. Configure R2_PUBLIC_URL or ensure presigned URLs work."
        )

    caption = _build_platform_caption(ctx, "instagram")
    ig_privacy = resolve_privacy_level(getattr(ctx, "privacy", None) or "public", "instagram")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Create media container
            logger.info(f"Instagram: Creating Reels container for ig_user_id={ig_user_id} (privacy={ig_privacy})")
            ig_params = {
                "access_token": access_token,
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption[:2200] if caption else "",
                "share_to_feed": "true" if ig_privacy == "public" else "false",
            }
            # Instagram cover_url must be set at creation time — cannot change after publish.
            # Prefer platform-specific 9:16 thumbnail when available (styled thumbnails).
            thumb_r2_key = None
            pt_json = ctx.output_artifacts.get("platform_thumbnail_r2_keys", "{}")
            pt_keys = json_dict(pt_json, default={}, context="platform_thumbnail_r2_keys")
            thumb_r2_key = pt_keys.get("instagram")
            thumb_r2_key = thumb_r2_key or getattr(ctx, "thumbnail_r2_key", None)
            if thumb_r2_key:
                try:
                    from . import r2 as _r2
                    thumb_cover_url = _r2.generate_presigned_url(thumb_r2_key, expires=3600)
                    ig_params["cover_url"] = thumb_cover_url
                    logger.info(f"Instagram: cover_url set from R2 thumbnail")
                except (_R2BotoCoreError, _R2ClientError, OSError, TypeError, ValueError, RuntimeError) as _e:
                    logger.warning("Instagram: could not set cover_url: %s", _e)
            create_resp = await client.post(
                f"https://graph.facebook.com/{META_API_VERSION}/{ig_user_id}/media",
                params=ig_params
            )

            if create_resp.status_code != 200:
                error_body = create_resp.text[:300]
                logger.error(f"Instagram container creation failed: {error_body}")
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    http_status=create_resp.status_code,
                    error_code="CONTAINER_FAILED",
                    error_message=f"Container creation failed: {error_body}"
                )

            creation_id = create_resp.json().get("id")
            if not creation_id:
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    error_code="NO_CREATION_ID",
                    error_message="No creation_id returned from container endpoint"
                )

            logger.info(f"Instagram: Container created, creation_id={creation_id}, polling status...")

            # Step 2: Poll until container is ready
            container_ready = False
            final_status = "UNKNOWN"

            for attempt in range(IG_POLL_MAX_ATTEMPTS):
                await asyncio.sleep(IG_POLL_INTERVAL)

                status_resp = await client.get(
                    f"https://graph.facebook.com/{META_API_VERSION}/{creation_id}",
                    params={
                        "access_token": access_token,
                        "fields": "status_code,status",
                    }
                )

                if status_resp.status_code != 200:
                    logger.warning(f"Instagram: Status poll failed (attempt {attempt + 1}): {status_resp.status_code}")
                    continue

                status_data = status_resp.json()
                final_status = status_data.get("status_code", "UNKNOWN")

                if final_status == "FINISHED":
                    container_ready = True
                    logger.info(f"Instagram: Container ready after {(attempt + 1) * IG_POLL_INTERVAL}s")
                    break
                elif final_status == "ERROR":
                    error_detail = status_data.get("status", "Unknown error")
                    return PlatformResult(
                        platform="instagram",
                        success=False,
                        error_code="CONTAINER_ERROR",
                        error_message=f"Container processing failed: {error_detail}"
                    )
                elif final_status == "EXPIRED":
                    return PlatformResult(
                        platform="instagram",
                        success=False,
                        error_code="CONTAINER_EXPIRED",
                        error_message="Container expired before publishing"
                    )
                else:
                    # IN_PROGRESS or other status
                    if (attempt + 1) % 6 == 0:  # Log every 30s
                        logger.info(f"Instagram: Still processing... status={final_status} ({(attempt + 1) * IG_POLL_INTERVAL}s)")

            if not container_ready:
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    error_code="CONTAINER_TIMEOUT",
                    error_message=f"Container not ready after {IG_POLL_MAX_ATTEMPTS * IG_POLL_INTERVAL}s (last status: {final_status})"
                )

            # Step 3: Publish the container
            publish_resp = await client.post(
                f"https://graph.facebook.com/{META_API_VERSION}/{ig_user_id}/media_publish",
                params={
                    "access_token": access_token,
                    "creation_id": creation_id,
                }
            )

            if publish_resp.status_code != 200:
                error_body = publish_resp.text[:300]
                return PlatformResult(
                    platform="instagram",
                    success=False,
                    http_status=publish_resp.status_code,
                    error_code="PUBLISH_FAILED",
                    error_message=f"Publish failed: {error_body}"
                )

            media_id = publish_resp.json().get("id")
            # Public shortcode URLs are NOT derivable from media_id locally — fetch permalink from Graph.
            platform_url = None
            perm_status = None
            if media_id:
                perm_resp = await client.get(
                    f"https://graph.facebook.com/{META_API_VERSION}/{media_id}",
                    params={
                        "access_token": access_token,
                        "fields": "permalink,shortcode",
                    },
                )
                perm_status = perm_resp.status_code
                if perm_resp.status_code == 200:
                    md = perm_resp.json() or {}
                    platform_url = md.get("permalink")
                    sc = md.get("shortcode")
                    if not platform_url and sc:
                        platform_url = f"https://www.instagram.com/p/{sc}/"
                else:
                    logger.warning(
                        "Instagram: permalink lookup failed media_id=%s status=%s body=%s",
                        media_id,
                        perm_resp.status_code,
                        (perm_resp.text or "")[:400],
                    )

            if not platform_url and media_id:
                logger.warning(
                    "Instagram: no permalink after publish (media_id=%s perm_status=%s)",
                    media_id,
                    perm_status,
                )

            logger.info(f"Instagram publish accepted: media_id={media_id}")
            return PlatformResult(
                platform="instagram",
                success=True,
                platform_video_id=media_id,
                platform_url=platform_url,
                publish_id=creation_id,
                verify_status="pending",
            )

    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="TIMEOUT",
            error_message="Instagram API request timed out",
        )
    except _PUBLISH_FILE_HTTP_ERRS as e:
        logger.error("Instagram publish error: %s", e)
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e),
        )


async def publish_to_facebook(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    video_url: Optional[str] = None,
    db_pool=None,
) -> PlatformResult:
    """Publish video to Facebook as a Reel.

    Supports two modes:
      1. Binary upload (multipart) - when local file available
      2. URL-based - when public URL available (preferred for large files)

    As of June 2025, all Facebook videos are treated as Reels.
    """
    # Refresh Meta token before use — long-lived tokens expire after ~60 days
    token_data = await _refresh_meta_token(
        token_data, "facebook", db_pool=db_pool, user_id=getattr(ctx, "user_id", None)
    )

    access_token = token_data.get("access_token")
    page_id = (
        token_data.get("page_id")
        or token_data.get("facebook_page_id")
        or token_data.get("fb_page_id")
    )

    if not access_token:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="NO_TOKEN",
            error_message="No access token"
        )

    if not page_id:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="NO_PAGE_ID",
            error_message="No Facebook page ID found in token data"
        )

    _fb_scope = require_facebook_publish(token_data)
    if _fb_scope:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="META_SCOPE_MISSING",
            error_message=_fb_scope,
        )

    description = _build_platform_caption(ctx, "facebook")
    fb_privacy_value = resolve_privacy_level(getattr(ctx, "privacy", None) or "public", "facebook")
    fb_privacy_param = {"value": fb_privacy_value}  # FB Graph API format: {"value": "EVERYONE"}

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Prefer binary upload (simpler, works without public URL)
            if video_path and video_path.exists():
                with open(video_path, "rb") as f:
                    files = {"source": ("video.mp4", f, "video/mp4")}
                    resp = await client.post(
                        f"https://graph.facebook.com/{META_API_VERSION}/{page_id}/videos",
                        params={
                            "access_token": access_token,
                            "description": description[:5000] if description else "",
                            "privacy": json.dumps(fb_privacy_param),
                        },
                        files=files,
                    )
            elif video_url:
                # URL-based upload (fallback)
                resp = await client.post(
                    f"https://graph.facebook.com/{META_API_VERSION}/{page_id}/videos",
                    params={
                        "access_token": access_token,
                        "file_url": video_url,
                        "description": description[:5000] if description else "",
                        "privacy": json.dumps(fb_privacy_param),
                    },
                )
            else:
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    error_code="NO_VIDEO",
                    error_message="No video file or URL available for Facebook"
                )

            if resp.status_code != 200:
                error_body = resp.text[:300]
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    http_status=resp.status_code,
                    error_code="UPLOAD_FAILED",
                    error_message=f"Upload failed: {error_body}"
                )

            video_id = resp.json().get("id")
            platform_url = None
            if video_id:
                perm_resp = await client.get(
                    f"https://graph.facebook.com/{META_API_VERSION}/{video_id}",
                    params={
                        "access_token": access_token,
                        "fields": "permalink_url",
                    },
                )
                if perm_resp.status_code == 200:
                    platform_url = (perm_resp.json() or {}).get("permalink_url")
                if not platform_url:
                    platform_url = f"https://www.facebook.com/watch/?v={video_id}"
            logger.info(f"Facebook publish accepted: video_id={video_id}, url={platform_url}")
            # Push thumbnail to Facebook (non-fatal)
            thumb_path = _get_platform_thumbnail_path(ctx, "facebook")
            if video_id and thumb_path:
                await _push_thumbnail_to_platform(
                    "facebook", video_id, thumb_path, access_token, client
                )
            return PlatformResult(
                platform="facebook",
                success=True,
                platform_video_id=video_id,
                platform_url=platform_url,
                verify_status="pending",
            )

    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="TIMEOUT",
            error_message="Facebook API request timed out",
        )
    except _PUBLISH_FILE_HTTP_ERRS as e:
        logger.error("Facebook publish error: %s", e)
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e),
        )


# =====================================================================
# Stage Entry Point
# =====================================================================

async def run_publish_stage(ctx: JobContext, db_pool) -> JobContext:
    """Publish to each platform independently + write ledger rows.

    Per-platform video selection priority:
      1. ctx.platform_videos[platform]  (local temp file from transcode)
      2. Fallback: ctx.processed_video_path or ctx.local_video_path

    Per-platform error isolation:
      - Each platform publishes independently
      - One failure does NOT prevent others from publishing
      - Results stored per-platform in ctx.platform_results

    Ledger contract:
      - INSERT publish_attempts row BEFORE the API call
      - UPDATE row after call (success/fail + metadata)
      - verify_stage later turns verify_status into confirmed/rejected/unknown
    """
    if not ctx.platforms and not ctx.target_accounts:
        logger.warning(f"No platforms specified for upload {ctx.upload_id}")
        return ctx

    # Determine default fallback video
    default_video = ctx.processed_video_path or ctx.local_video_path
    has_any_video = False

    # Check we have at least one video available
    if ctx.platform_videos:
        has_any_video = any(
            p and p.exists() for p in ctx.platform_videos.values()
        )
    if not has_any_video and default_video and default_video.exists():
        has_any_video = True

    if not has_any_video:
        raise PublishError("No video files available to publish", code=ErrorCode.UPLOAD_FAILED)

    init_enc_keys()

    # Build publish targets: list of (platform, token_id_or_None)
    #
    # LOGIC: Users select platforms AND which accounts within each platform.
    # The frontend must send target_accounts = [platform_tokens.id, ...] for the
    # selected accounts. Only those accounts receive the upload.
    #
    # When target_accounts is provided: publish to those specific accounts only.
    # When empty (legacy): one publish per platform using most-recent token.
    publish_targets: list[tuple[str, str | None]] = []
    if ctx.target_accounts:
        for token_id in ctx.target_accounts:
            raw = await db_stage.load_platform_token_by_id(db_pool, token_id)
            if raw and isinstance(raw, dict):
                plat = raw.get("_platform", "")
                if plat:
                    publish_targets.append((plat, token_id))
                    continue
            logger.warning(f"target_account {token_id}: token not found or revoked — skipping")
        if not publish_targets:
            logger.warning(f"All target_accounts invalid for upload {ctx.upload_id}, falling back to one per platform")
    if not publish_targets:
        publish_targets = [(p, None) for p in ctx.platforms]

    PUBLISH_TIMEOUT = int(os.environ.get("PUBLISH_PER_PLATFORM_TIMEOUT", "600"))
    parallel_mode = os.environ.get("PUBLISH_PARALLEL", "true").lower() in ("1", "true", "yes")

    logger.info(
        f"Publishing to {len(publish_targets)} target(s) | "
        f"mode={'parallel' if parallel_mode else 'sequential'} | "
        f"timeout={PUBLISH_TIMEOUT}s"
    )

    async def _publish_one_target(platform: str, token_id: str | None) -> PlatformResult | None:
        """Publish to a single platform/account with timeout. Returns result or None."""
        account_label = f"{platform}:{token_id[:8]}" if token_id else platform
        db_key = platform_tokens_db_key(platform)
        rcli = getattr(ctx, "redis_client", None)
        try:
            if await publish_circuit_open(rcli, platform):
                logger.warning("%s: circuit open — skipping publish attempt", account_label)
                return PlatformResult(
                    platform=platform,
                    success=False,
                    error_code="PUBLISH_CIRCUIT_OPEN",
                    error_message="Platform APIs throttled after repeated failures — retry in a few minutes.",
                    token_row_id=token_id,
                )
            await publish_wait_slot(rcli, platform)
            res = await asyncio.wait_for(
                _publish_single(ctx, db_pool, platform, token_id, db_key, account_label, default_video),
                timeout=PUBLISH_TIMEOUT,
            )
            await publish_record_result(rcli, platform, res.success, getattr(res, "http_status", None))
            return res
        except asyncio.TimeoutError:
            logger.error(f"{account_label}: Publish timed out after {PUBLISH_TIMEOUT}s")
            await publish_record_result(rcli, platform, False, None)
            return PlatformResult(
                platform=platform,
                success=False,
                error_code="PUBLISH_TIMEOUT",
                error_message=f"Publish timed out after {PUBLISH_TIMEOUT}s",
                token_row_id=token_id,
            )
        except Exception:
            await publish_record_result(rcli, platform, False, None)
            raise

    if parallel_mode and len(publish_targets) > 1:
        results = await asyncio.gather(
            *[_publish_one_target(p, tid) for p, tid in publish_targets],
            return_exceptions=True,
        )
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                plat, tid = publish_targets[i]
                logger.exception(f"Parallel publish {plat} raised: {res}")
                ctx.platform_results.append(PlatformResult(
                    platform=plat, success=False,
                    error_code="PUBLISH_EXCEPTION", error_message=str(res)[:500],
                    token_row_id=tid,
                ))
            elif res:
                ctx.platform_results.append(res)
    else:
        for platform, token_id in publish_targets:
            res = await _publish_one_target(platform, token_id)
            if res:
                ctx.platform_results.append(res)

    # -- Summary --
    succeeded = [r.platform for r in ctx.platform_results if r.success]
    failed = [r.platform for r in ctx.platform_results if not r.success]
    logger.info(
        f"Publish complete for {ctx.upload_id}: "
        f"succeeded={succeeded}, failed={failed}"
    )

    return ctx


async def _publish_single(
    ctx: "JobContext",
    db_pool,
    platform: str,
    token_id: str | None,
    db_key: str,
    account_label: str,
    default_video,
) -> PlatformResult:
    """Publish to ONE platform/account. Isolated — exceptions caught internally."""
    from stages import db as db_stage

    def _clip(s: str, n: int) -> str:
        return (s or "")[:n]

    video_file = ctx.get_video_for_platform(platform)
    if not video_file or not video_file.exists():
        video_file = default_video
        if video_file:
            logger.warning(
                f"{account_label}: Platform-specific video not found, "
                f"using fallback: {video_file.name}"
            )

    attempt_id = None
    try:
        attempt_id = await db_stage.insert_publish_attempt(
            db_pool,
            upload_id=str(ctx.upload_id),
            user_id=str(ctx.user_id),
            platform=str(platform),
        )
    except _DB_PERSIST_ERRS as e:
        logger.warning("%s: Could not create publish_attempt row: %s", account_label, e)
        attempt_id = None

    token_data = None
    token_identity = {}
    try:
        token_data, token_identity = await db_stage.load_platform_token_with_identity(
            db_pool, ctx.user_id, db_key, token_row_id=token_id
        )
        if token_data is not None:
            token_data = decrypt_token(token_data)
    except _DB_PERSIST_ERRS as e:
        logger.warning("%s: Token load failed: %s", account_label, e)
        token_data = None
        token_identity = {}

    # Hard guard: token payload must be a dict with an access token.
    if not isinstance(token_data, dict) or not str(token_data.get("access_token", "")).strip():
        msg = (
            f"Invalid or missing token payload for {platform}"
            + (f" (account {token_id[:8]})" if token_id else "")
        )
        logger.warning(f"{account_label}: {msg}")
        if attempt_id:
            try:
                await db_stage.update_publish_attempt_failed(
                    db_pool,
                    attempt_id=attempt_id,
                    error_code="TOKEN_INVALID",
                    error_message=msg,
                )
            except _DB_PERSIST_ERRS as e:
                logger.debug("%s: Failed to mark TOKEN_INVALID on publish_attempt: %s", account_label, e)
        return PlatformResult(
            platform=platform,
            success=False,
            attempt_id=attempt_id,
            error_code="TOKEN_INVALID",
            error_message=msg,
            token_row_id=token_id,
        )

    # Audit trail: effective metadata that this publish attempt will use.
    try:
        eff_title = (
            ctx.get_effective_title(platform)
            if hasattr(ctx, "get_effective_title")
            else (getattr(ctx, "title", None) or "")
        )
        eff_caption = (
            ctx.get_effective_caption(platform)
            if hasattr(ctx, "get_effective_caption")
            else (getattr(ctx, "caption", None) or "")
        )
        eff_tags = (
            ctx.get_effective_hashtags(platform)
            if hasattr(ctx, "get_effective_hashtags")
            else (getattr(ctx, "hashtags", None) or [])
        )
        requested_privacy = getattr(ctx, "privacy", None) or "public"
        native_privacy = resolve_privacy_level(requested_privacy, platform)
        await db_stage.write_system_event_log(
            db_pool,
            user_id=str(ctx.user_id),
            event_category="UPLOAD",
            action="PUBLISH_METADATA_RESOLVED",
            resource_type="upload",
            resource_id=str(ctx.upload_id),
            details={
                "attempt_id": attempt_id,
                "platform": platform,
                "token_row_id": token_id,
                "account_id": token_identity.get("account_id"),
                "account_username": token_identity.get("account_username"),
                "requested_privacy": requested_privacy,
                "native_privacy": native_privacy,
                "effective_title": _clip(str(eff_title or ""), 300),
                "effective_caption": _clip(str(eff_caption or ""), 1200),
                "effective_hashtags": list(eff_tags or [])[:80],
            },
            severity="INFO",
            outcome="SUCCESS",
        )
    except _DB_PERSIST_ERRS as e:
        logger.debug("%s: Could not write publish metadata audit log: %s", account_label, e, exc_info=True)

    if not token_data:
        msg = f"Not connected to {platform}" + (f" (account {token_id[:8]})" if token_id else "")
        logger.warning(f"{account_label}: {msg}")
        if attempt_id:
            try:
                await db_stage.update_publish_attempt_failed(
                    db_pool, attempt_id=attempt_id,
                    error_code="NOT_CONNECTED", error_message=msg,
                )
            except _DB_PERSIST_ERRS as e:
                logger.debug("%s: Failed to mark NOT_CONNECTED on publish_attempt: %s", account_label, e)
        return PlatformResult(
            platform=platform, success=False, attempt_id=attempt_id,
            error_code="NOT_CONNECTED", error_message=msg, token_row_id=token_id,
        )

    result = None
    try:
        if platform == "tiktok":
            result = await publish_to_tiktok(video_file, ctx, token_data, db_pool=db_pool)
        elif platform == "youtube":
            result = await publish_to_youtube(video_file, ctx, token_data, db_pool=db_pool)
        elif platform == "instagram":
            video_url = _get_video_public_url(ctx, "instagram")
            result = await publish_to_instagram(
                video_file, ctx, token_data, video_url=video_url, db_pool=db_pool
            )
        elif platform == "facebook":
            video_url = _get_video_public_url(ctx, "facebook")
            result = await publish_to_facebook(
                video_file, ctx, token_data, video_url=video_url, db_pool=db_pool
            )
        else:
            result = PlatformResult(
                platform=platform, success=False,
                error_code="UNSUPPORTED", error_message=f"Unsupported platform: {platform}",
                account_id=token_id,
                account_name=(token_data.get("_account_name") or "") if token_data else "",
            )
    except asyncio.CancelledError:
        raise
    except _PUBLISH_FILE_HTTP_ERRS as e:
        logger.exception("Error publishing to %s", account_label)
        result = PlatformResult(
            platform=platform,
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e),
            account_id=token_id,
            account_name=(token_data.get("_account_name") or "") if token_data else "",
        )

    result.attempt_id = attempt_id
    if token_identity:
        result.token_row_id     = token_identity.get("token_row_id")
        result.account_id       = token_identity.get("account_id")
        result.account_username = token_identity.get("account_username")
        result.account_name     = token_identity.get("account_name")
        result.account_avatar   = token_identity.get("account_avatar")

    status_icon = "OK" if result.success else "FAIL"
    extra = ""
    if token_id:
        extra += f" account={token_id[:8]}"
    if result.publish_id:
        extra += f" publish_id={result.publish_id}"
    if result.platform_video_id:
        extra += f" video_id={result.platform_video_id}"
    if result.error_message and not result.success:
        extra += f" error={result.error_message[:100]}"
    logger.info(f"{account_label} [{status_icon}]: {result.error_code or 'accepted'}{extra}")

    if attempt_id:
        try:
            if result.success:
                await db_stage.update_publish_attempt_success(
                    db_pool, attempt_id=attempt_id,
                    platform_post_id=result.platform_video_id,
                    platform_url=result.platform_url,
                    http_status=result.http_status,
                    response_payload=result.response_payload,
                    publish_id=result.publish_id,
                )
            else:
                await db_stage.update_publish_attempt_failed(
                    db_pool, attempt_id=attempt_id,
                    error_code=result.error_code or "PUBLISH_FAILED",
                    error_message=result.error_message or "Publish failed",
                    http_status=result.http_status,
                    response_payload=result.response_payload,
                )
        except _DB_PERSIST_ERRS as e:
            logger.warning("%s: Could not update publish_attempt: %s", account_label, e)

    # Audit trail: final platform outcome for this attempt.
    try:
        await db_stage.write_system_event_log(
            db_pool,
            user_id=str(ctx.user_id),
            event_category="UPLOAD",
            action="PUBLISH_ATTEMPT_RESULT",
            resource_type="upload",
            resource_id=str(ctx.upload_id),
            details={
                "attempt_id": attempt_id,
                "platform": platform,
                "token_row_id": token_id,
                "account_id": token_identity.get("account_id"),
                "account_username": token_identity.get("account_username"),
                "success": bool(result.success),
                "error_code": result.error_code,
                "error_message": _clip(str(result.error_message or ""), 500),
                "http_status": result.http_status,
                "publish_id": result.publish_id,
                "platform_video_id": result.platform_video_id,
                "platform_url": result.platform_url,
            },
            severity="INFO" if result.success else "WARN",
            outcome="SUCCESS" if result.success else "FAIL",
        )
    except _DB_PERSIST_ERRS as e:
        logger.warning("publish_stage: write_system_event_log PUBLISH_ATTEMPT_RESULT failed: %s", e)

    return result
