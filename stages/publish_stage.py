"""
UploadM8 Publish Stage
======================
Publish per-platform transcoded videos to social media platforms.

Platform-specific publish flows:
  TikTok   - Content Posting API v2: init upload -> PUT binary -> publish_id
  YouTube  - Resumable upload: init -> PUT binary -> video_id
  Instagram - Graph API Reels: create container (video_url) -> poll status -> publish;
    optional first-comment hashtag block when hashtagPosition=comment
  Facebook - Graph API: multipart upload or URL-based Reels

Key behaviors:
  - Each platform gets its OWN transcoded video (ctx.platform_videos[platform])
  - Meta platforms (IG/FB) use presigned R2 URLs when public URL needed
  - Per-platform error isolation: one failure does NOT kill others
  - Ledger row per attempt (publish_attempts table)
  - verify_stage later confirms "accepted" -> "confirmed live"

Exports used by verify_stage:
  - decrypt_token(token_row) -> dict or None
  - init_enc_keys() -> None  (delegates to core.auth; raises if TOKEN_ENC_KEYS missing)
"""

import os
import re
import json
import asyncio
import logging
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.helpers import strip_stray_hashtag_json_blob

from core.cancel_signal import is_cancelled_fast as _cancel_is_set_fast

from .errors import CancelRequested, PublishError, ErrorCode
from .context import JobContext, PlatformResult
from . import db as db_stage
from . import r2 as r2_stage
from .image_format import ensure_jpeg_file, sniff_image_format

from services.publish_metadata_gate import assert_publish_metadata_gate
from services.tiktok_api import (
    resolve_tiktok_post_settings_for_account,
    tiktok_force_private_unaudited,
    tiktok_post_info_from_settings,
    validate_tiktok_post_settings,
)
from core.helpers import coerce_processed_assets_map


async def _publish_cancelled(ctx: JobContext) -> bool:
    """Cheap Redis-only cancel check used between platform publishes.

    The publish stage doesn't have access to the worker's db_pool reference
    here, so we only consult Redis. The worker's outer ``maybe_cancel`` will
    still pick up DB-only cancels at the next stage boundary; this just
    keeps us from publishing to platform N+1 after the user hit stop.
    """
    if bool(getattr(ctx, "cancel_requested", False)):
        return True
    try:
        import core.state as _state
        rc = getattr(_state, "redis_client", None)
    except Exception:
        rc = None
    if rc is None:
        return False
    if await _cancel_is_set_fast(rc, ctx.upload_id):
        try:
            setattr(ctx, "cancel_requested", True)
        except Exception:
            pass
        return True
    return False


logger = logging.getLogger("uploadm8-worker")

# Graph API version for Meta platforms (Instagram + Facebook)
META_API_VERSION = "v21.0"

# Instagram container polling
IG_POLL_INTERVAL = 5       # seconds between polls
IG_POLL_MAX_ATTEMPTS = 36  # 3 minutes max (36 * 5s)
# IG Comment API message limit (stay under Meta limits; UTF-8 safe slice)
IG_HASHTAG_COMMENT_MAX = 1900

# Presigned URL expiry for Meta uploads (1 hour)
PRESIGNED_URL_EXPIRY = 3600
# Meta Graph multipart uploads often 413 above ~100MB — prefer file_url when larger.
FACEBOOK_MULTIPART_MAX_BYTES = 95 * 1024 * 1024

# YouTube Shorts + Content ID: catalogue/copyright-protected audio is not allowed in Shorts
# that are 60 seconds or longer (YouTube surfaces this after upload). When our ACR stack flags
# third-party audio risk and the clip is long enough to trip that rule, we publish as a normal
# video: strip Shorts hashtags from snippet metadata and link users to /watch instead of /shorts.
YOUTUBE_SHORTS_COPYRIGHT_MAX_SEC = 60.0

_SHORTS_HASHTAG_ONLY_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])#(?:shorts|youtubeshorts|youtube_shorts|yt_shorts|ytshorts)\b"
)


# =====================================================================
# Platform Thumbnail Push
# =====================================================================

def _get_platform_thumbnail_path(ctx, platform: str) -> Optional[Path]:
    """
    Return the best local thumbnail path for a platform when the file still exists.

    Prefer ``platform_thumbnail_map`` (per-platform styled 16:9 / 9:16). Fall back to
    the sharpness-selected frame. Use ``_ensure_platform_thumbnail_local`` when paths
    may be stale (deferred publish) or bytes may be PNG masquerading as JPEG.
    """
    pm = ctx.output_artifacts.get("platform_thumbnail_map", "{}")
    try:
        plat_map = json.loads(pm) if isinstance(pm, str) else (pm or {})
    except Exception:
        plat_map = {}
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


async def _ensure_platform_thumbnail_local(ctx: JobContext, platform: str) -> Optional[Path]:
    """Resolve a platform thumbnail to a local JPEG file for post-publish APIs."""
    import tempfile

    local = _get_platform_thumbnail_path(ctx, platform)
    if local:
        try:
            return ensure_jpeg_file(local)
        except Exception as e:
            logger.warning("Thumbnail JPEG normalize failed for %s: %s", platform, e)
            return local

    thumb_r2_key = None
    pt_json = ctx.output_artifacts.get("platform_thumbnail_r2_keys", "{}")
    try:
        pt_keys = json.loads(pt_json) if isinstance(pt_json, str) else (pt_json or {})
        thumb_r2_key = pt_keys.get(platform)
    except Exception:
        pass
    if not thumb_r2_key:
        thumb_r2_key = getattr(ctx, "thumbnail_r2_key", None)

    if not thumb_r2_key:
        return None

    base_dir = getattr(ctx, "temp_dir", None)
    if base_dir:
        out = Path(base_dir) / f"thumb_{platform}_publish.jpg"
    else:
        out = Path(tempfile.mkdtemp(prefix="thumb_pub_")) / f"thumb_{platform}.jpg"
    try:
        await r2_stage.download_file(thumb_r2_key, out)
        out = ensure_jpeg_file(out)
        plat_map = {}
        pm = ctx.output_artifacts.get("platform_thumbnail_map", "{}")
        try:
            plat_map = json.loads(pm) if isinstance(pm, str) else (pm or {})
        except Exception:
            plat_map = {}
        plat_map[platform] = str(out)
        ctx.output_artifacts["platform_thumbnail_map"] = json.dumps(plat_map)
        logger.info("Publish: loaded %s thumbnail from R2 for cover push", platform)
        return out
    except Exception as e:
        logger.warning("Publish: could not load %s thumbnail from R2: %s", platform, e)
        return None


def _tiktok_cover_timestamp_ms(ctx: JobContext) -> int:
    """TikTok Direct Post uses a video frame for cover — not a custom image upload.

    When the worker burned a styled thumb into the MP4 (``tiktok_cover_burned``),
    use the same offset so the API selects that frame.
    """
    arts = ctx.output_artifacts or {}
    if str(arts.get("tiktok_cover_burned") or "").lower() in ("1", "true", "yes"):
        raw = arts.get("tiktok_cover_burn_offset_seconds")
        if raw is not None and raw != "":
            try:
                return int(max(0.0, float(raw)) * 1000)
            except (TypeError, ValueError):
                pass
    for key in ("thumbnail_frame_offset_seconds", "tiktok_thumb_offset_seconds"):
        raw = (ctx.output_artifacts or {}).get(key)
        if raw is None or raw == "":
            continue
        try:
            sec = max(0.0, float(raw))
            return int(sec * 1000)
        except (TypeError, ValueError):
            continue
    return 1500


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
    TikTok:    Uses ``video_cover_timestamp_ms`` at init (not custom image upload).
    Instagram: Cover must be set at container creation time (handled there).
    Facebook:  POST /{video_id}  with thumb= multipart field.
    """
    if not thumbnail_path or not thumbnail_path.exists():
        return False
    try:
        thumbnail_path = ensure_jpeg_file(Path(thumbnail_path))
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

        return False
    except Exception as e:
        logger.warning(f"Thumbnail push {platform} failed (non-fatal): {e}")
        return False


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
      - public   → PUBLIC_TO_EVERYONE (visible to everyone)
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


def _tiktok_force_private_unaudited_enabled() -> bool:
    """When true, TikTok Direct Post uses ``SELF_ONLY`` regardless of user privacy choice.

    Default: true while ``TIKTOK_APP_AUDITED`` is unset (app pending TikTok audit).
    After audit passes, set ``TIKTOK_APP_AUDITED=1``. Optional override:
    ``TIKTOK_FORCE_PRIVATE_UNAUDITED=1``.
    """
    return tiktok_force_private_unaudited()


def _tiktok_unaudited_private_only_error(body: str) -> bool:
    return "unaudited_client_can_only_post_to_private_accounts" in (body or "")


async def _tiktok_init_direct_post(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    post_info: dict,
    file_size: int,
    chunk_size: int,
    total_chunk_count: int,
) -> httpx.Response:
    return await client.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={
            "post_info": post_info,
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": file_size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunk_count,
            },
        },
    )


# =====================================================================
# Token Encryption (used by this stage + verify_stage)
# =====================================================================

def init_enc_keys():
    """Load TOKEN_ENC_KEYS into core.state (same contract as the API process)."""
    from core.auth import init_enc_keys as _core_init

    _core_init()


def decrypt_token_blob(blob: Any) -> dict:
    """Decrypt platform token blob."""
    import core.state

    if isinstance(blob, str):
        blob = json.loads(blob)

    kid = blob.get("kid", "v1")
    key = core.state.ENC_KEYS.get(kid)
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
            except Exception:
                return None

        # Already plaintext token dict
        return token_row
    except Exception:
        return None


# =====================================================================
# Content Derivation
# =====================================================================

def _utf16_rune_len(s: str) -> int:
    """TikTok limits `title` by UTF-16 code units (see Direct Post API)."""
    return len(s.encode("utf-16-le")) // 2


def _utf16_clip(s: str, max_runes: int) -> str:
    if not s or max_runes <= 0:
        return ""
    if _utf16_rune_len(s) <= max_runes:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _utf16_rune_len(s[:mid]) <= max_runes:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo]


def _youtube_duration_sec(ctx: JobContext) -> float:
    try:
        return float((ctx.video_info or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _youtube_avoid_shorts_markers_for_rights(ctx: JobContext) -> bool:
    """True when Shorts-oriented metadata could violate YouTube's copyright + length Shorts rule."""
    try:
        from stages.youtube_copyright_shorts import youtube_copyright_shorts_trim_applied

        if youtube_copyright_shorts_trim_applied(ctx):
            return False
    except Exception:
        pass

    dur = _youtube_duration_sec(ctx)
    if dur < YOUTUBE_SHORTS_COPYRIGHT_MAX_SEC:
        return False
    ac = getattr(ctx, "audio_context", None) or {}
    if not isinstance(ac, dict):
        return False
    if ac.get("copyright_risk"):
        return True
    # Catalogue fingerprint without explicit copyright_risk (older payloads)
    cts = ac.get("content_signals") or []
    if isinstance(cts, list) and "acr_catalog_match" in {str(x).strip() for x in cts if x}:
        return bool(ac.get("music_detected") or ac.get("music_title") or ac.get("music_artist"))
    return False


def _strip_youtube_shorts_hashtag_markers(text: str) -> str:
    """Remove #shorts-style tokens so YouTube does not classify the upload as a monetized Short."""
    if not text:
        return ""
    t = _SHORTS_HASHTAG_ONLY_RE.sub("", text)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Do not call strip_stray_hashtag_json_blob here — it collapses all whitespace and destroys
    # intentional paragraph breaks in descriptions. Caption/title paths already sanitize upstream.
    return t.strip()


def _get_title(ctx: JobContext, platform: str = "") -> str:
    """Get the best available title for publishing (per-platform when ``platform`` is set)."""
    if hasattr(ctx, "get_effective_title"):
        return ctx.get_effective_title(platform=platform)
    return (
        getattr(ctx, "ai_title", None)
        or getattr(ctx, "title", None)
        or getattr(ctx, "video_title", None)
        or getattr(ctx, "name", None)
        or f"UploadM8 {getattr(ctx, 'upload_id', '')}".strip()
    )


def _get_caption(ctx: JobContext, platform: str = "") -> str:
    """Get the best available caption/description for publishing."""
    raw = (
        ctx.get_effective_caption(platform=platform)
        if hasattr(ctx, "get_effective_caption")
        else (
            getattr(ctx, "ai_caption", None)
            or getattr(ctx, "caption", None)
            or getattr(ctx, "description", None)
            or ""
        )
    )
    return strip_stray_hashtag_json_blob(str(raw or "").strip())


def _get_hashtags(ctx: JobContext, platform: str = "") -> str:
    """Get hashtag string for publishing.

    Uses get_effective_hashtags(platform) which merges in order:
    always_hashtags → platform_hashtags → upload (base) hashtags → M8 → generic AI.
    Same pipeline for all sources; blocked tags filtered throughout.
    """
    if hasattr(ctx, "get_effective_hashtags"):
        merged = ctx.get_effective_hashtags(platform=platform)
    else:
        merged = getattr(ctx, "ai_hashtags", None) or getattr(ctx, "hashtags", None) or []
    if isinstance(merged, list):
        tags = [str(t).strip() for t in merged if str(t).strip()]
    else:
        tags = [p for p in re.split(r"[\s,]+", str(merged).strip()) if p]
    normalised = [t if t.startswith("#") else f"#{t}" for t in tags if t]
    return " ".join(normalised) if normalised else ""


def _instagram_first_comment_mode(ctx: JobContext) -> bool:
    """True when user chose 'First comment (IG only)' / legacy caption-as-comment style."""
    try:
        us = ctx.user_settings or {}
        pos = str(us.get("hashtagPosition") or us.get("hashtag_position") or "end").lower()
        return pos in ("comment", "caption")
    except Exception:
        return False


async def _instagram_post_hashtag_first_comment(
    client: httpx.AsyncClient,
    media_id: str,
    access_token: str,
    hashtag_text: str,
) -> tuple[bool, str]:
    """POST /{ig-media-id}/comments — hashtags only, after reel is live. Returns (ok, error_snippet)."""
    msg = (hashtag_text or "").strip()
    if not msg:
        return True, ""
    msg = msg[:IG_HASHTAG_COMMENT_MAX]
    url = f"https://graph.facebook.com/{META_API_VERSION}/{media_id}/comments"
    resp = await client.post(
        url,
        params={"access_token": access_token, "message": msg},
    )
    if resp.status_code != 200:
        return False, (resp.text or "")[:500]
    return True, ""


def _build_platform_caption(ctx: JobContext, platform: str) -> str:
    """Build caption + hashtags for a specific platform.

    Includes platform-specific hashtags from user_settings.
    Respects ``hashtagPosition``: ``start`` | ``end`` for all platforms; ``comment`` / ``caption``
    applies **only to Instagram** — reel caption is prose-only and hashtags are posted afterward
    via Graph ``/{media-id}/comments`` (see ``publish_to_instagram``). Other platforms treat
    ``comment``/``caption`` like ``end`` so hashtags are never dropped.

    **TikTok/Instagram/Facebook:** ``start`` is treated like ``end`` (caption then hashtags)
    so prose is always first in published text.
    """
    caption = _get_caption(ctx, platform=platform)
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
        if hashtag_position not in ("start", "end", "comment", "caption"):
            hashtag_position = "end"
    except Exception:
        pass

    pl = (platform or "").strip().lower()
    if hashtag_position in ("comment", "caption"):
        if pl == "instagram":
            # Reel caption = story only; hashtags go to first comment after publish.
            return strip_stray_hashtag_json_blob(caption)
        hashtag_position = "end"

    # Caption-first policy for short-form/social feed surfaces:
    # - TikTok: post_info.title is clipped at 2200 UTF-16 from the end.
    # - Instagram/Facebook: user requested caption before hashtags.
    if pl in ("tiktok", "instagram", "facebook") and hashtag_position == "start":
        hashtag_position = "end"

    if hashtag_position == "start":
        out = f"{hashtags}\n\n{caption}"
    else:
        out = f"{caption}\n\n{hashtags}"
    return strip_stray_hashtag_json_blob(out)


def _tiktok_post_title(ctx: JobContext) -> str:
    """TikTok `post_info.title` is the full caption (hashtags allowed), max 2200 UTF-16 runes."""
    caption = (_build_platform_caption(ctx, "tiktok") or "").strip()
    if not caption:
        return _utf16_clip(_get_title(ctx, "tiktok"), 2200)
    return _utf16_clip(caption, 2200)


def _tiktok_export_is_hashtag_only(text: str) -> bool:
    """True when export caption is only hashtag tokens (user prefilled tags, AI prose pending)."""
    t = (text or "").strip()
    if not t:
        return False
    parts = [p for p in re.split(r"[\s\n]+", t) if p]
    if not parts:
        return False
    return all(p.startswith("#") and len(p) > 1 for p in parts)


def _tiktok_ai_prose(ctx: JobContext) -> str:
    """Prose-only TikTok caption from worker AI / user metadata (no hashtag merge)."""
    cap = (_get_caption(ctx, "tiktok") or "").strip()
    if cap:
        return strip_stray_hashtag_json_blob(cap)
    return (_get_title(ctx, "tiktok") or "").strip()


def _resolve_tiktok_publish_title(ctx: JobContext, export_title: str) -> str:
    """
    Prefer user-edited export caption; fall back to worker-built caption.
    Hashtag-only export text gets AI prose prepended so batch/settings-tag flows keep AI captions.
    """
    saved = (export_title or "").strip()
    if not saved:
        return _tiktok_post_title(ctx)
    if _tiktok_export_is_hashtag_only(saved):
        prose = _tiktok_ai_prose(ctx)
        if prose:
            return _utf16_clip(f"{prose}\n\n{saved}", 2200)
    return _utf16_clip(saved, 2200)


def _build_full_caption(ctx: JobContext) -> str:
    """Backward-compat alias — no platform-specific hashtags."""
    return _build_platform_caption(ctx, platform="")


def _collect_processed_r2_keys(ctx: JobContext, platform: str) -> list[str]:
    """Ordered R2 key candidates for a platform's processed MP4 (deduped)."""
    keys: list[str] = []
    seen: set[str] = set()

    def _add(key: Optional[str]) -> None:
        k = (key or "").strip()
        if k and k not in seen and not k.startswith("thumb_"):
            seen.add(k)
            keys.append(k)

    assets_sources: list[object] = []
    try:
        assets_sources.append(ctx.output_artifacts.get("processed_assets", "{}"))
    except Exception:
        pass
    upload_rec = getattr(ctx, "upload_record", None)
    if isinstance(upload_rec, dict):
        assets_sources.append(upload_rec.get("processed_assets"))

    for raw in assets_sources:
        try:
            assets = coerce_processed_assets_map(raw)
            _add(assets.get(platform))
            _add(assets.get("default"))
        except Exception:
            continue

    _add(getattr(ctx, "processed_r2_key", None))
    _add(f"processed/{ctx.user_id}/{ctx.upload_id}/{platform}.mp4")
    _add(f"processed/{ctx.user_id}/{ctx.upload_id}/default.mp4")
    return keys


def _presigned_url_for_r2_key(r2_key: str) -> Optional[str]:
    public_url = r2_stage.get_public_url(r2_key)
    if public_url:
        return public_url
    try:
        return r2_stage.generate_presigned_url(r2_key, expires=PRESIGNED_URL_EXPIRY)
    except Exception as e:
        logger.debug("Presign failed for %s: %s", r2_key, e)
        return None


def _get_video_public_url(ctx: JobContext, platform: str) -> Optional[str]:
    """Get a publicly accessible URL for the platform's video.

    Tries each known processed R2 key (platform, default, DB snapshot) until
    a public CDN URL or presigned GET URL succeeds.
    """
    for r2_key in _collect_processed_r2_keys(ctx, platform):
        url = _presigned_url_for_r2_key(r2_key)
        if url:
            logger.info("[%s] Resolved %s video URL via r2_key=%s", ctx.upload_id, platform, r2_key)
            return url
    logger.warning(
        "[%s] No hosted video URL for %s (tried keys: %s)",
        ctx.upload_id,
        platform,
        _collect_processed_r2_keys(ctx, platform),
    )
    return None


def _facebook_file_too_large_message(
    file_size: int,
    upload_mode: str,
    *,
    missing_url: bool = False,
) -> str:
    size_mb = file_size / 1024 / 1024
    limit_mb = FACEBOOK_MULTIPART_MAX_BYTES / 1024 / 1024
    if missing_url:
        return (
            f"Video is {size_mb:.1f} MB — above Facebook's {limit_mb:.0f} MB direct-upload limit. "
            "UploadM8 could not build a hosted file URL for Meta to fetch from storage. "
            "Retry shortly or contact support if this persists."
        )
    if upload_mode.startswith("file_url"):
        return (
            f"Video is {size_mb:.1f} MB — Facebook rejected the full-quality hosted file URL (HTTP 413). "
            "UploadM8 preserves your 1080p/4K transcode; Meta may still cap ingest size for this page. "
            "Try publishing to Instagram/YouTube, or contact support if this persists."
        )
    return (
        f"Video is {size_mb:.1f} MB — too large for Facebook's direct browser-style upload "
        f"(limit ~{limit_mb:.0f} MB). UploadM8 publishes full-quality files via a hosted URL when possible."
    )


# =====================================================================
# Platform Publishers
# =====================================================================


async def _refresh_tiktok_token(
    token_data: dict,
    db_pool=None,
    user_id: str = None,
    token_row_id: str | None = None,
) -> dict:
    """Refresh a TikTok access token using the stored refresh_token.
    TikTok access tokens expire after 24 hours; refresh tokens after 365 days.
    Persists the new token blob to DB if db_pool and user_id are provided.
    """
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        logger.warning("TikTok: No refresh_token stored, cannot refresh — user must reconnect")
        return token_data

    client_key = os.environ.get("TIKTOK_CLIENT_KEY", "")
    client_secret = os.environ.get("TIKTOK_CLIENT_SECRET", "")

    if not client_key or not client_secret:
        logger.warning("TikTok: Missing TIKTOK_CLIENT_KEY/SECRET env vars, cannot refresh token")
        return token_data

    try:
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
                new_access = new_tokens.get("access_token") or token_data.get("access_token")
                if not new_access:
                    logger.warning("TikTok: refresh response missing access_token")
                    return token_data
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
                    except Exception as save_err:
                        logger.warning(f"TikTok: Failed to persist refreshed token: {save_err}")
                return updated
            else:
                logger.warning(f"TikTok: Token refresh failed: {resp.status_code} {resp.text[:200]}")
                return token_data
    except Exception as e:
        logger.warning(f"TikTok: Token refresh exception: {e}")
        return token_data



async def _refresh_meta_token(
    token_data: dict,
    platform: str,
    db_pool=None,
    user_id: str = None,
    token_row_id: str | None = None,
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
                    matching = next((p for p in pages if p.get("id") == page_id), None)
                    if matching:
                        new_page_token = matching.get("access_token", new_user_token)
                        updated = {
                            **token_data,
                            "access_token": new_page_token,
                            "expires_at": None,  # Page tokens from LLT don't expire
                        }
                        if db_pool and user_id:
                            try:
                                await db_stage.save_refreshed_token(
                                    db_pool,
                                    user_id=user_id,
                                    platform=platform,
                                    access_token=new_page_token,
                                    token_row_id=token_row_id,
                                )
                            except Exception as save_err:
                                logger.warning(f"{platform}: Failed to persist refreshed token: {save_err}")
                        return updated

            # ── Recover missing ig_user_id / page_id ────────────────────────
            # Old tokens stored before the OAuth fix may lack these IDs.
            # Now that we have a fresh user token we can fetch them on the fly
            # so the publish succeeds without forcing the user to reconnect.
            # Step 1: Fetch user's Pages — id, name, page access_token only.
            # instagram_business_account is NOT a valid inline field on /me/accounts;
            # it must be fetched per-page via /{page_id}?fields=instagram_business_account
            pages_resp = await client.get(
                "https://graph.facebook.com/v18.0/me/accounts",
                params={"access_token": new_user_token, "fields": "id,name,access_token"},
            )

            recovered_ig_user_id = token_data.get("ig_user_id") or token_data.get("instagram_user_id")
            recovered_page_id    = token_data.get("page_id") or token_data.get("facebook_page_id")
            recovered_page_token = new_user_token  # fallback

            if pages_resp.status_code == 200:
                pages = pages_resp.json().get("data", [])

                if platform == "instagram" and not recovered_ig_user_id:
                    # instagram_business_account must be fetched per-page
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
                                recovered_ig_user_id = ig_biz["id"]
                                recovered_page_token = page_token_tmp
                                logger.info(
                                    f"instagram: Recovered ig_user_id={recovered_ig_user_id} "
                                    f"from Page '{page.get('name', '?')}' during token refresh"
                                )
                                break

                if platform == "facebook" and not recovered_page_id and pages:
                    first = pages[0]
                    recovered_page_id    = first["id"]
                    recovered_page_token = first.get("access_token", new_user_token)
                    logger.info(
                        f"facebook: Recovered page_id={recovered_page_id} "
                        f"from Page '{first.get('name', '?')}' during token refresh"
                    )

            # Build updated blob — preserve any existing IDs, override with recovered ones
            updated = {
                **token_data,
                "access_token":  recovered_page_token if platform == "instagram" else new_user_token,
                "expires_at":    None,
            }
            if platform == "instagram" and recovered_ig_user_id:
                updated["ig_user_id"] = recovered_ig_user_id
            if platform == "facebook" and recovered_page_id:
                updated["page_id"]    = recovered_page_id
                updated["access_token"] = recovered_page_token

            if db_pool and user_id:
                try:
                    await db_stage.save_refreshed_token(
                        db_pool,
                        user_id=user_id,
                        platform=platform,
                        access_token=updated["access_token"],
                        token_row_id=token_row_id,
                    )
                except Exception as save_err:
                    logger.warning(f"{platform}: Failed to persist refreshed token: {save_err}")
            return updated

    except Exception as e:
        logger.warning(f"{platform}: Meta token refresh exception: {e}")
        return token_data



_TIKTOK_MIN_CHUNK = 5_000_000     # 5 MB (decimal) — TikTok Media Transfer Guide
_TIKTOK_MAX_CHUNK = 64_000_000    # 64 MB (decimal) — hard cap on chunk_size
_TIKTOK_DOC_CHUNK = 10_000_000    # 10 MB (decimal) — value used in TikTok's docs example
_TIKTOK_MAX_LAST_CHUNK = 128_000_000  # 128 MB (decimal) — final-chunk allowance
_TIKTOK_MAX_CHUNKS = 1000


def _tiktok_file_upload_chunk_plan(file_size: int) -> tuple[int, int]:
    """Return (chunk_size, total_chunk_count) for TikTok FILE_UPLOAD init.

    TikTok's Media Transfer Guide rules (all sizes in **decimal** bytes —
    5/10/64/128 MB = 5_000_000 / 10_000_000 / 64_000_000 / 128_000_000):

    * ``total_chunk_count = video_size // chunk_size`` (integer **floor**).
    * Each chunk_size must be 5 MB – 64 MB **except** the final chunk, which
      may carry the remainder (up to 128 MB).
    * **When ``total_chunk_count == 1``, ``chunk_size`` MUST equal
      ``video_size``** (single "whole upload"). Sending a smaller
      ``chunk_size`` with ``total_chunk_count = 1`` makes TikTok return
      ``invalid_params: "The total chunk count is invalid"``.
    * Videos under 5 MB must be uploaded whole.

    Strategy:

    * ``video_size <= 64 MB``  → whole upload ``(video_size, 1)``.
      Covers TikTok's <5 MB rule *and* avoids the 10–20 MB and 64–~73 MB
      traps where ``floor(size / 10 MB)`` is still 1 but
      ``chunk_size != video_size`` would be sent.
    * ``video_size > 64 MB``   → chunked at 10 MB (the doc example value);
      ``floor(size / 10 MB) >= 6`` so ``n >= 2`` and the trailing remainder
      stays well under 128 MB. Files large enough to exceed the 1000-chunk
      cap (>10 GB at 10 MB chunks) bump ``chunk_size`` upward, capped at
      64 MB.
    """
    min_c = _TIKTOK_MIN_CHUNK
    max_single = _TIKTOK_MAX_CHUNK
    doc_chunk = _TIKTOK_DOC_CHUNK
    max_last = _TIKTOK_MAX_LAST_CHUNK
    max_chunks = _TIKTOK_MAX_CHUNKS

    if file_size <= 0:
        raise ValueError("TikTok upload: empty video file")

    if file_size <= max_single:
        return file_size, 1

    c = doc_chunk
    n = file_size // c
    if n > max_chunks:
        c = max(min_c, (file_size + max_chunks - 1) // max_chunks)
        c = min(max_single, c)
        n = file_size // c

    last = file_size - (n - 1) * c
    while last > max_last and c < max_single:
        c = min(max_single, c + min_c)
        n = file_size // c
        last = file_size - (n - 1) * c

    if (
        n < 2
        or n > max_chunks
        or last <= 0
        or last > max_last
        or c < min_c
        or c > max_single
    ):
        raise ValueError(
            f"TikTok upload: invalid chunk geometry size={file_size} c={c} n={n} last={last}"
        )

    return c, n


async def publish_to_tiktok(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    db_pool=None,
    token_row_id: str | None = None,
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

    # TikTok Direct Post: `post_info.title` is the full caption (max 2200 UTF-16 runes).
    tt_settings = resolve_tiktok_post_settings_for_account(
        getattr(ctx, "tiktok_post_settings", None) or (ctx.user_settings or {}).get("tiktok_post_settings"),
        str(token_row_id or ""),
    )
    if not tt_settings:
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="TIKTOK_EXPORT_SETTINGS_MISSING",
            error_message=(
                "TikTok posting settings were not saved for this upload. "
                "Re-upload with the Post to TikTok export form completed."
            ),
        )
    tt_errors = validate_tiktok_post_settings(tt_settings)
    if tt_errors:
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="TIKTOK_EXPORT_SETTINGS_INVALID",
            error_message=tt_errors[0],
        )

    max_dur = tt_settings.get("max_video_post_duration_sec")
    if max_dur:
        try:
            max_sec = int(max_dur)
            vid_dur = float((ctx.video_info or {}).get("duration") or 0.0)
            if vid_dur > 0 and vid_dur > max_sec:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    error_code="TIKTOK_VIDEO_TOO_LONG",
                    error_message=(
                        f"Video is {vid_dur:.0f}s but this TikTok account allows "
                        f"at most {max_sec}s. Shorten the clip or pick another account."
                    ),
                )
        except (TypeError, ValueError):
            pass

    tiktok_title = _resolve_tiktok_publish_title(ctx, (tt_settings.get("title") or "").strip())

    try:
        file_size = video_path.stat().st_size

        # TikTok Media Transfer Guide: total_chunk_count = video_size // chunk_size
        # (floor).  See _tiktok_file_upload_chunk_plan.
        chunk_size, total_chunk_count = _tiktok_file_upload_chunk_plan(file_size)
        logger.info(
            "TikTok init: video_size=%s chunk_size=%s total_chunk_count=%s",
            file_size, chunk_size, total_chunk_count,
        )

        tiktok_privacy = str(tt_settings.get("privacy_level") or "").strip()
        privacy_overridden_unaudited = False
        if _tiktok_force_private_unaudited_enabled():
            if tiktok_privacy != "SELF_ONLY":
                logger.info(
                    "TikTok: TIKTOK_FORCE_PRIVATE_UNAUDITED — clamping privacy_level %s → SELF_ONLY "
                    "(user was informed of this at upload time; video will be private until audit passes)",
                    tiktok_privacy,
                )
                privacy_overridden_unaudited = True
            tiktok_privacy = "SELF_ONLY"

        cover_ms = _tiktok_cover_timestamp_ms(ctx)
        post_info = tiktok_post_info_from_settings(tt_settings, title=tiktok_title)
        post_info["video_cover_timestamp_ms"] = cover_ms
        # Apply the (possibly clamped) privacy level after building post_info from settings
        post_info["privacy_level"] = tiktok_privacy
        logger.info(
            "TikTok: video_cover_timestamp_ms=%s (%.2fs) privacy=%s unaudited_clamp=%s",
            cover_ms,
            cover_ms / 1000.0,
            post_info.get("privacy_level"),
            privacy_overridden_unaudited,
        )

        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Initialize upload
            init_resp = await _tiktok_init_direct_post(
                client,
                access_token=access_token,
                post_info=post_info,
                file_size=file_size,
                chunk_size=chunk_size,
                total_chunk_count=total_chunk_count,
            )

            if (
                init_resp.status_code != 200
                and _tiktok_unaudited_private_only_error(init_resp.text)
                and post_info.get("privacy_level") != "SELF_ONLY"
            ):
                logger.info(
                    "TikTok: unaudited app — retrying init with privacy_level=SELF_ONLY "
                    "(was %s)",
                    post_info.get("privacy_level"),
                )
                post_info = {**post_info, "privacy_level": "SELF_ONLY"}
                init_resp = await _tiktok_init_direct_post(
                    client,
                    access_token=access_token,
                    post_info=post_info,
                    file_size=file_size,
                    chunk_size=chunk_size,
                    total_chunk_count=total_chunk_count,
                )

            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="tiktok",
                    success=False,
                    http_status=init_resp.status_code,
                    error_code="INIT_FAILED",
                    error_message=(
                        f"Init failed: {init_resp.text[:200]} "
                        f"(video_size={file_size} chunk_size={chunk_size} "
                        f"total_chunk_count={total_chunk_count}; "
                        f"TikTok requires total_chunk_count=floor(video_size/chunk_size))"
                    ),
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

            # Step 2: Upload video — first (N-1) parts are exactly chunk_size bytes
            # (TikTok); final part is the remainder (may be larger than chunk_size).
            with open(video_path, "rb") as f:
                offset = 0
                for chunk_index in range(total_chunk_count):
                    if chunk_index < total_chunk_count - 1:
                        chunk_data = f.read(chunk_size)
                    else:
                        chunk_data = f.read()
                    chunk_start = offset
                    chunk_end = chunk_start + len(chunk_data) - 1
                    offset = chunk_end + 1

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
                response_payload={
                    "tiktok_privacy_level": post_info.get("privacy_level"),
                    "tiktok_privacy_overridden_unaudited": privacy_overridden_unaudited,
                    "upload_privacy": (getattr(ctx, "privacy", None) or "public"),
                    "tiktok_disable_comment": post_info.get("disable_comment"),
                    "tiktok_disable_duet": post_info.get("disable_duet"),
                    "tiktok_disable_stitch": post_info.get("disable_stitch"),
                },
            )

    except Exception as e:
        logger.error(f"TikTok publish error: {e}")
        return PlatformResult(
            platform="tiktok",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


async def _refresh_youtube_token(
    token_data: dict,
    db_pool=None,
    user_id: str = None,
    token_row_id: str | None = None,
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
                    except Exception as save_err:
                        logger.warning(f"YouTube: Failed to persist refreshed token: {save_err}")
                return updated
            else:
                logger.warning(f"YouTube: Token refresh failed: {resp.status_code} {resp.text[:200]}")
                return token_data
    except Exception as e:
        logger.warning(f"YouTube: Token refresh exception: {e}")
        return token_data


async def publish_to_youtube(
    video_path: Path,
    ctx: JobContext,
    token_data: dict,
    db_pool=None,
) -> PlatformResult:
    """Publish video to YouTube using resumable upload (Shorts vs long-form via snippet + URL).

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

    rights_long_form = _youtube_avoid_shorts_markers_for_rights(ctx)
    if rights_long_form:
        logger.info(
            "YouTube publish: applying long-form snippet guard (duration=%.1fs, copyright_risk=%s)",
            _youtube_duration_sec(ctx),
            (getattr(ctx, "audio_context", None) or {}).get("copyright_risk"),
        )

    title = _get_title(ctx, "youtube")[:100]
    caption = _build_platform_caption(ctx, "youtube")
    if rights_long_form:
        title = _strip_youtube_shorts_hashtag_markers(title)[:100]
        if not (title or "").strip():
            uid = str(getattr(ctx, "upload_id", "") or "").strip()
            title = (f"UploadM8 {uid}" if uid else "UploadM8 video")[:100]
        caption = _strip_youtube_shorts_hashtag_markers(caption)
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
            if video_id:
                if rights_long_form:
                    platform_url = f"https://www.youtube.com/watch?v={video_id}"
                else:
                    platform_url = f"https://youtube.com/shorts/{video_id}"
            else:
                platform_url = None

            logger.info(f"YouTube publish accepted: video_id={video_id}, url={platform_url}")
            # Push thumbnail to YouTube (non-fatal — requires verified channel)
            thumb_path = await _ensure_platform_thumbnail_local(ctx, "youtube")
            if video_id and thumb_path:
                pushed = await _push_thumbnail_to_platform(
                    "youtube", video_id, thumb_path, access_token, client
                )
                if not pushed:
                    logger.warning(
                        "YouTube: custom thumbnail not applied for video_id=%s "
                        "(channel may need verification)",
                        video_id,
                    )
            return PlatformResult(
                platform="youtube",
                success=True,
                platform_video_id=video_id,
                platform_url=platform_url,
                verify_status="pending",
                response_payload={"youtube_long_form_rights_guard": rights_long_form} if rights_long_form else None,
            )

    except Exception as e:
        logger.error(f"YouTube publish error: {e}")
        return PlatformResult(
            platform="youtube",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


async def _instagram_cover_public_url(
    thumb_r2_key: str,
    ctx: JobContext,
) -> Optional[str]:
    """Return a Meta-fetchable JPEG URL for Reels ``cover_url``.

    Pikzels often writes PNG bytes to ``*.jpg`` keys; Instagram rejects non-JPEG covers
    (opaque subcode 2207085). Re-encode and upload ``instagram_ig_cover.jpg`` when needed.
    """
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="ig_cover_"))
    local = tmp_dir / "cover_src.jpg"
    await r2_stage.download_file(thumb_r2_key, local)
    fmt = sniff_image_format(local)
    if fmt == "jpeg" and local.stat().st_size <= 8 * 1024 * 1024:
        pub = r2_stage.get_public_url(thumb_r2_key)
        if pub:
            return pub
        return r2_stage.generate_presigned_url(thumb_r2_key, expires=PRESIGNED_URL_EXPIRY)

    jpeg_path = tmp_dir / "cover_meta.jpg"
    jpeg_path.write_bytes(local.read_bytes())
    ensure_jpeg_file(jpeg_path)
    user_id = getattr(ctx, "user_id", None) or "unknown"
    upload_id = getattr(ctx, "upload_id", None) or "unknown"
    meta_key = f"thumbnails/{user_id}/{upload_id}/instagram_ig_cover.jpg"
    await r2_stage.upload_file(jpeg_path, meta_key, "image/jpeg")
    logger.info(
        "Instagram: cover re-encoded to JPEG (%s -> %s, was %s)",
        thumb_r2_key,
        meta_key,
        fmt,
    )
    pub = r2_stage.get_public_url(meta_key)
    if pub:
        return pub
    return r2_stage.generate_presigned_url(meta_key, expires=PRESIGNED_URL_EXPIRY)


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
            # Prefer platform-specific styled 9:16 thumbnail (Pikzels / persona / custom).
            thumb_cover_url = None
            thumb_path = await _ensure_platform_thumbnail_local(ctx, "instagram")
            if thumb_path:
                try:
                    import tempfile

                    tmp_dir = Path(tempfile.mkdtemp(prefix="ig_cover_"))
                    jpeg_path = tmp_dir / "cover_meta.jpg"
                    jpeg_path.write_bytes(thumb_path.read_bytes())
                    ensure_jpeg_file(jpeg_path)
                    user_id = getattr(ctx, "user_id", None) or "unknown"
                    upload_id = getattr(ctx, "upload_id", None) or "unknown"
                    meta_key = f"thumbnails/{user_id}/{upload_id}/instagram_ig_cover.jpg"
                    await r2_stage.upload_file(jpeg_path, meta_key, "image/jpeg")
                    thumb_cover_url = r2_stage.get_public_url(meta_key)
                    if not thumb_cover_url:
                        thumb_cover_url = r2_stage.generate_presigned_url(
                            meta_key, expires=PRESIGNED_URL_EXPIRY
                        )
                    if thumb_cover_url:
                        logger.info("Instagram: cover_url set from styled thumbnail (JPEG-safe)")
                except Exception as _e:
                    logger.warning(f"Instagram: could not set cover_url from local thumb: {_e}")
            if not thumb_cover_url:
                thumb_r2_key = None
                pt_json = ctx.output_artifacts.get("platform_thumbnail_r2_keys", "{}")
                try:
                    pt_keys = json.loads(pt_json) if isinstance(pt_json, str) else (pt_json or {})
                    thumb_r2_key = pt_keys.get("instagram")
                except Exception:
                    pass
                thumb_r2_key = thumb_r2_key or getattr(ctx, "thumbnail_r2_key", None)
                if thumb_r2_key:
                    try:
                        thumb_cover_url = await _instagram_cover_public_url(thumb_r2_key, ctx)
                        if thumb_cover_url:
                            logger.info("Instagram: cover_url set from R2 key (JPEG-safe)")
                    except Exception as _e:
                        logger.warning(f"Instagram: could not set cover_url: {_e}")
            if thumb_cover_url:
                ig_params["cover_url"] = thumb_cover_url
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
            platform_url: Optional[str] = None
            last_http: Optional[int] = None
            if media_id:
                for _ig_attempt in range(2):
                    try:
                        perm_resp = await client.get(
                            f"https://graph.facebook.com/{META_API_VERSION}/{media_id}",
                            params={
                                "access_token": access_token,
                                "fields": "permalink,shortcode",
                            },
                        )
                        last_http = perm_resp.status_code
                        if perm_resp.status_code == 200:
                            pj = perm_resp.json()
                            platform_url = (pj.get("permalink") or "").strip() or None
                            sc = (pj.get("shortcode") or "").strip()
                            if not platform_url and sc:
                                platform_url = f"https://www.instagram.com/reel/{sc}/"
                            if platform_url:
                                break
                    except Exception as _e:
                        logger.warning(f"Instagram: permalink fetch for {media_id}: {_e}")
                    if _ig_attempt == 0:
                        await asyncio.sleep(1.5)
                if not platform_url:
                    logger.warning(
                        "Instagram: no permalink after publish (media_id=%s last_http=%s)",
                        media_id,
                        last_http,
                    )

            # First-comment hashtags: reel caption stays prose-only; tag block posted as /comments.
            if media_id and _instagram_first_comment_mode(ctx):
                tag_block = _get_hashtags(ctx, "instagram").strip()
                cap_only = (_get_caption(ctx, "instagram") or "").strip()
                if tag_block and cap_only:
                    await asyncio.sleep(1.0)
                    ok_c, err_c = await _instagram_post_hashtag_first_comment(
                        client, str(media_id), access_token, tag_block
                    )
                    if ok_c:
                        logger.info("Instagram: posted hashtag block as first comment on media_id=%s", media_id)
                    else:
                        logger.warning(
                            "Instagram: first-comment hashtags failed (media still live): %s",
                            err_c,
                        )

            logger.info(f"Instagram publish accepted: media_id={media_id} url={platform_url}")
            return PlatformResult(
                platform="instagram",
                success=True,
                platform_video_id=media_id,
                platform_url=platform_url,
                publish_id=creation_id,
                verify_status="pending",
            )

    except httpx.TimeoutException:
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="TIMEOUT",
            error_message="Instagram API request timed out"
        )
    except Exception as e:
        logger.error(f"Instagram publish error: {e}")
        return PlatformResult(
            platform="instagram",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
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

    description = _build_platform_caption(ctx, "facebook")
    fb_privacy_value = resolve_privacy_level(getattr(ctx, "privacy", None) or "public", "facebook")
    fb_privacy_param = {"value": fb_privacy_value}  # FB Graph API format: {"value": "EVERYONE"}
    file_size = video_path.stat().st_size if video_path and video_path.exists() else 0
    endpoint = f"https://graph.facebook.com/{META_API_VERSION}/{page_id}/videos"
    post_params = {
        "access_token": access_token,
        "description": description[:5000] if description else "",
        "privacy": json.dumps(fb_privacy_param),
    }

    must_use_file_url = file_size > FACEBOOK_MULTIPART_MAX_BYTES
    if not video_url:
        video_url = _get_video_public_url(ctx, "facebook")
    if must_use_file_url and not video_url:
        msg = _facebook_file_too_large_message(file_size, "none", missing_url=True)
        logger.error("Facebook publish blocked: %s", msg)
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="FILE_TOO_LARGE",
            error_message=msg,
        )

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            prefer_url = bool(video_url) and (
                must_use_file_url
                or not (video_path and video_path.exists())
            )
            resp = None
            upload_mode = "none"

            if prefer_url and video_url:
                upload_mode = "file_url"
                resp = await client.post(
                    endpoint,
                    params={**post_params, "file_url": video_url},
                )
            elif video_path and video_path.exists():
                upload_mode = "multipart"
                with open(video_path, "rb") as f:
                    files = {"source": ("video.mp4", f, "video/mp4")}
                    resp = await client.post(endpoint, params=post_params, files=files)
                if resp.status_code == 413:
                    if video_url:
                        logger.warning(
                            "Facebook multipart 413 (%.1f MB) — retrying with file_url",
                            file_size / 1024 / 1024,
                        )
                        upload_mode = "file_url_retry"
                        resp = await client.post(
                            endpoint,
                            params={**post_params, "file_url": video_url},
                        )
                    else:
                        video_url = _get_video_public_url(ctx, "facebook")
                        if video_url:
                            logger.warning(
                                "Facebook multipart 413 (%.1f MB) — resolved file_url on retry",
                                file_size / 1024 / 1024,
                            )
                            upload_mode = "file_url_late"
                            resp = await client.post(
                                endpoint,
                                params={**post_params, "file_url": video_url},
                            )
            elif video_url:
                upload_mode = "file_url"
                resp = await client.post(
                    endpoint,
                    params={**post_params, "file_url": video_url},
                )
            else:
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    error_code="NO_VIDEO",
                    error_message="No video file or URL available for Facebook"
                )

            if resp.status_code != 200:
                error_body = (resp.text or "").strip()[:300]
                is_too_large = resp.status_code == 413
                if is_too_large:
                    msg = _facebook_file_too_large_message(
                        file_size,
                        upload_mode,
                        missing_url=False,
                    )
                elif error_body:
                    msg = f"Facebook upload failed (HTTP {resp.status_code}): {error_body}"
                else:
                    msg = (
                        f"Facebook upload failed (HTTP {resp.status_code}, mode={upload_mode}). "
                        "Check Meta token/page permissions or try again."
                    )
                return PlatformResult(
                    platform="facebook",
                    success=False,
                    http_status=resp.status_code,
                    error_code="FILE_TOO_LARGE" if is_too_large else "UPLOAD_FAILED",
                    error_message=msg,
                )

            video_id = resp.json().get("id")
            platform_url: Optional[str] = None
            if video_id:
                try:
                    perm_resp = await client.get(
                        f"https://graph.facebook.com/{META_API_VERSION}/{video_id}",
                        params={
                            "access_token": access_token,
                            "fields": "permalink_url",
                        },
                    )
                    if perm_resp.status_code == 200:
                        platform_url = (perm_resp.json().get("permalink_url") or "").strip() or None
                except Exception as _e:
                    logger.warning(f"Facebook: could not fetch permalink_url for {video_id}: {_e}")
                if not platform_url:
                    if str(video_id).isdigit() and str(page_id).isdigit():
                        platform_url = f"https://www.facebook.com/{page_id}/videos/{video_id}"
                    else:
                        platform_url = f"https://www.facebook.com/watch/?v={video_id}"
            logger.info(f"Facebook publish accepted: video_id={video_id}, url={platform_url}")
            # Push thumbnail to Facebook (non-fatal)
            thumb_path = await _ensure_platform_thumbnail_local(ctx, "facebook")
            if video_id and thumb_path:
                pushed = await _push_thumbnail_to_platform(
                    "facebook", video_id, thumb_path, access_token, client
                )
                if not pushed:
                    logger.warning(
                        "Facebook: custom thumbnail not applied for video_id=%s",
                        video_id,
                    )
            return PlatformResult(
                platform="facebook",
                success=True,
                platform_video_id=video_id,
                platform_url=platform_url,
                verify_status="pending",
            )

    except httpx.TimeoutException:
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="TIMEOUT",
            error_message="Facebook API request timed out"
        )
    except Exception as e:
        logger.error(f"Facebook publish error: {e}")
        return PlatformResult(
            platform="facebook",
            success=False,
            error_code="PUBLISH_EXCEPTION",
            error_message=str(e)
        )


# =====================================================================
# Publish target resolution (shared by publish_stage + deferred scheduler)
# =====================================================================

async def resolve_publish_targets(ctx: JobContext, db_pool) -> list[tuple[str, str | None]]:
    """Build (platform, token_id) publish targets for an upload."""
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
            logger.warning(
                f"All target_accounts invalid for upload {ctx.upload_id}, falling back to one per platform"
            )
    if not publish_targets:
        publish_targets = [(p, None) for p in ctx.platforms]
    return publish_targets


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

    # Hydrate processed_assets from DB when publish runs after deferred handoff
    if db_pool and not coerce_processed_assets_map(
        ctx.output_artifacts.get("processed_assets", "{}")
    ):
        try:
            upload_row = await db_stage.load_upload_record(db_pool, str(ctx.upload_id))
            if upload_row:
                db_assets = coerce_processed_assets_map(upload_row.get("processed_assets"))
                if db_assets:
                    ctx.output_artifacts["processed_assets"] = json.dumps(db_assets)
                    if not getattr(ctx, "processed_r2_key", None):
                        ctx.processed_r2_key = db_assets.get("default") or next(
                            iter(db_assets.values()), None
                        )
        except Exception as e:
            logger.debug("[%s] processed_assets DB hydrate skipped: %s", ctx.upload_id, e)

    # Token DB key mapping (platform -> token table platform key)
    platform_to_db_key = {
        "tiktok": "tiktok",
        "youtube": "youtube",
        "instagram": "instagram",
        "facebook": "facebook",
    }

    # Build publish targets: list of (platform, token_id_or_None)
    publish_targets = await resolve_publish_targets(ctx, db_pool)

    plat_filter = getattr(ctx, "deferred_publish_platform_filter", None)
    if plat_filter is not None:
        plat_filter = {str(p).strip().lower() for p in plat_filter if str(p).strip()}
        publish_targets = [
            (p, tid) for p, tid in publish_targets if str(p).strip().lower() in plat_filter
        ]

    from services.deferred_publish_schedule import publish_target_already_done

    pending_targets = [
        (p, tid)
        for p, tid in publish_targets
        if not publish_target_already_done(ctx, p, tid)
    ]
    if not pending_targets:
        logger.info(f"[{ctx.upload_id}] Deferred publish: no pending targets in this batch")
        return ctx

    logger.info(f"Publishing to {len(pending_targets)} target(s)")

    assert_publish_metadata_gate(ctx, pending_targets)

    for platform, token_id in pending_targets:
        # Per-target cancel check — once a single platform has been posted we
        # don't try to "unpost" it (those APIs don't support that), but we DO
        # stop attempting any further platforms so the user's stop click
        # actually halts the upload mid-fan-out instead of grinding through
        # every remaining destination.
        if await _publish_cancelled(ctx):
            logger.info(
                f"[{ctx.upload_id}] Publish cancelled before {platform} — "
                f"halting fan-out (already-posted targets are kept)"
            )
            raise CancelRequested(ctx.upload_id)

        account_label = f"{platform}:{token_id[:8]}" if token_id else platform
        db_key = platform_to_db_key.get(platform, platform)

        # -- Select the correct video file for this platform --
        video_file = ctx.get_video_for_platform(platform)
        if not video_file or not video_file.exists():
            video_file = default_video
            if video_file:
                logger.warning(
                    f"{account_label}: Platform-specific video not found, "
                    f"using fallback: {video_file.name}"
                )

        # -- Create ledger row (before API call) --
        attempt_id = None
        try:
            attempt_id = await db_stage.insert_publish_attempt(
                db_pool,
                upload_id=str(ctx.upload_id),
                user_id=str(ctx.user_id),
                platform=str(platform),
            )
        except Exception as e:
            logger.warning(f"{account_label}: Could not create publish_attempt row: {e}")
            attempt_id = None

        # -- Load platform token (use target_accounts token when specified) --
        token_data = None
        token_identity = {}
        try:
            token_data, token_identity = await db_stage.load_platform_token_with_identity(
                db_pool, ctx.user_id, db_key, token_row_id=token_id
            )
            if token_data:
                token_data = decrypt_token(token_data) if isinstance(token_data, dict) and token_data.get("kid") else token_data
        except Exception as e:
            logger.warning(f"{account_label}: Token load failed: {e}")
            token_data = None
            token_identity = {}

        if not token_data:
            msg = f"Not connected to {platform}" + (f" (account {token_id[:8]})" if token_id else "")
            logger.warning(f"{account_label}: {msg}")
            if attempt_id:
                try:
                    await db_stage.update_publish_attempt_failed(
                        db_pool,
                        attempt_id=attempt_id,
                        error_code="NOT_CONNECTED",
                        error_message=msg,
                    )
                except Exception:
                    pass
            pr = PlatformResult(
                platform=platform,
                success=False,
                attempt_id=attempt_id,
                error_code="NOT_CONNECTED",
                error_message=msg,
                token_row_id=token_id,
            )
            ctx.platform_results.append(pr)
            continue

        # -- Publish to platform --
        result = None
        try:
            if platform == "tiktok":
                result = await publish_to_tiktok(
                    video_file, ctx, token_data, db_pool=db_pool, token_row_id=token_id
                )

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
                    platform=platform,
                    success=False,
                    error_code="UNSUPPORTED",
                    error_message=f"Unsupported platform: {platform}",
                    account_id=token_id,
                    account_name=(token_data.get("_account_name") or "") if token_data else "",
                )

        except Exception as e:
            logger.exception(f"Error publishing to {account_label}")
            result = PlatformResult(
                platform=platform,
                success=False,
                error_code="PUBLISH_EXCEPTION",
                error_message=str(e),
                account_id=token_id,
                account_name=(token_data.get("_account_name") or "") if token_data else "",
            )

        # -- Record result (stamp account identity from token so it's saved to DB) --
        result.attempt_id = attempt_id
        if token_identity:
            result.token_row_id     = token_identity.get("token_row_id")
            result.account_id       = token_identity.get("account_id")
            result.account_username = token_identity.get("account_username")
            result.account_name     = token_identity.get("account_name")
            result.account_avatar   = token_identity.get("account_avatar")
        ctx.platform_results.append(result)

        # -- Auto-record per-platform publish failure as an operational incident --
        # Surfaces failures on admin-incidents.html even when the overall pipeline
        # succeeds for other platforms. Alerts are deduped per platform+error_code.
        if not result.success:
            try:
                from services.ops_incidents import record_operational_incident

                await record_operational_incident(
                    db_pool,
                    source="publish",
                    incident_type=f"publish_failed:{platform}:{(result.error_code or 'UNKNOWN')}"[:120],
                    subject=f"Publish failed → {account_label}: {result.error_code or 'UNKNOWN'}",
                    body=(result.error_message or "")[:8000],
                    details={
                        "platform": platform,
                        "account_label": account_label,
                        "account_id": getattr(result, "account_id", None),
                        "account_username": getattr(result, "account_username", None),
                        "error_code": result.error_code,
                        "http_status": getattr(result, "http_status", None),
                        "error_message": (result.error_message or "")[:1000] or None,
                        "attempt_id": attempt_id,
                    },
                    user_id=str(ctx.user_id) if ctx.user_id else None,
                    upload_id=str(ctx.upload_id) if ctx.upload_id else None,
                )
            except Exception as _ix:
                logger.debug("publish-stage incident record failed: %s", _ix)

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

        # -- Update ledger row --
        if attempt_id:
            try:
                if result.success:
                    await db_stage.update_publish_attempt_success(
                        db_pool,
                        attempt_id=attempt_id,
                        platform_post_id=result.platform_video_id,
                        platform_url=result.platform_url,
                        http_status=result.http_status,
                        response_payload=result.response_payload,
                        publish_id=result.publish_id,
                    )
                    if token_id:
                        await db_stage.touch_platform_token_last_used(db_pool, token_id)
                else:
                    await db_stage.update_publish_attempt_failed(
                        db_pool,
                        attempt_id=attempt_id,
                        error_code=result.error_code or "PUBLISH_FAILED",
                        error_message=result.error_message or "Publish failed",
                        http_status=result.http_status,
                        response_payload=result.response_payload,
                    )
            except Exception as e:
                logger.warning(f"{account_label}: Could not update publish_attempt: {e}")

    # -- Summary --
    succeeded = [r.platform for r in ctx.platform_results if r.success]
    failed = [r.platform for r in ctx.platform_results if not r.success]
    logger.info(f"Publish complete: succeeded={succeeded}, failed={failed}")

    return ctx