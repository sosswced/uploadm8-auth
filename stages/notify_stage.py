"""
UploadM8 Notification Stage
===========================
Send notifications via Discord webhooks and email.

Notifications:
- User webhook: Upload status (success/fail)
- Admin webhook: Signup, trial, MRR, errors
- Email: Welcome, upgrade confirmation, promotions
"""

import os
import json
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import httpx

from core.helpers import sanitize_hashtag_body

from .context import JobContext, PlatformResult
from . import db as db_stage
from .publish_stage import resolve_privacy_level

logger = logging.getLogger("uploadm8-worker")

# Configuration
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
SIGNUP_DISCORD_WEBHOOK_URL = os.environ.get("SIGNUP_DISCORD_WEBHOOK_URL", "")
TRIAL_DISCORD_WEBHOOK_URL = os.environ.get("TRIAL_DISCORD_WEBHOOK_URL", "")
MRR_DISCORD_WEBHOOK_URL = os.environ.get("MRR_DISCORD_WEBHOOK_URL", "")
ERROR_DISCORD_WEBHOOK_URL = os.environ.get("ERROR_DISCORD_WEBHOOK_URL", "")

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")

_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "instagram": "Instagram",
    "facebook": "Facebook",
}


def _platform_label(slug: str) -> str:
    k = (slug or "").lower()
    return _PLATFORM_LABELS.get(k, (slug or "Platform").title())


def _flatten_hashtag_raw(raw: Any) -> List[str]:
    """Turn stored hashtag payloads (list, JSON string, junk strings) into token strings without '#'."""
    if raw is None:
        return []
    if isinstance(raw, list):
        candidates: List[Any] = list(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    candidates = parsed
                else:
                    candidates = [s]
            except json.JSONDecodeError:
                candidates = [s]
        else:
            candidates = [s]
    else:
        candidates = [raw]

    out: List[str] = []
    for item in candidates:
        piece = str(item).strip()
        if not piece:
            continue
        # Pull word-like tokens from messy strings (e.g. #"[\"tester\" #"qwe"]")
        for m in re.finditer(r"#?([A-Za-z0-9_]{2,50})", piece):
            body = sanitize_hashtag_body(m.group(1))
            if body:
                out.append(body)
    # De-dupe preserving order
    seen: set = set()
    uniq: List[str] = []
    for b in out:
        if b.lower() in seen:
            continue
        seen.add(b.lower())
        uniq.append(b)
    return uniq


def _hashtags_for_discord_line(tokens: List[str]) -> str:
    return " ".join(f"#{t}" for t in tokens if t)


def _build_hashtags_by_platform_block(ctx: JobContext) -> str:
    """Effective caption hashtags per target platform (matches publish merge order)."""
    plat_sources = [r.platform for r in (ctx.platform_results or []) if r.success]
    if not plat_sources:
        plat_sources = list(ctx.platforms or [])
    seen: set = set()
    lines: List[str] = []
    for pl in plat_sources:
        key = (pl or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        tags = ctx.get_effective_hashtags(key)
        if not tags:
            continue
        line = f"**{_platform_label(key)}:** {' '.join(tags)}"
        lines.append(line)
    return "\n".join(lines)


def _build_m8_ai_hashtags_block(ctx: JobContext) -> str:
    """Per-platform AI hashtag variants from M8 (when they differ from a single global list)."""
    m8 = getattr(ctx, "m8_platform_hashtags", None) or {}
    if not isinstance(m8, dict) or not m8:
        return ""
    lines: List[str] = []
    for pl in sorted(m8.keys()):
        raw = m8.get(pl) or []
        if not isinstance(raw, list) or not raw:
            continue
        flat = _flatten_hashtag_raw(raw)
        if not flat:
            continue
        lines.append(f"**{_platform_label(str(pl))}:** {_hashtags_for_discord_line(flat)}")
    return "\n".join(lines)


def _canonical_privacy(ctx: JobContext) -> str:
    p = (getattr(ctx, "privacy", None) or "public").strip().lower()
    if p not in ("public", "unlisted", "private"):
        return "public"
    return p


def _tiktok_status_lines(ctx: JobContext, result: PlatformResult) -> List[str]:
    lines: List[str] = []
    payload = result.response_payload or {}
    level = payload.get("tiktok_privacy_level") or resolve_privacy_level(_canonical_privacy(ctx), "tiktok")
    canon = (payload.get("upload_privacy") or _canonical_privacy(ctx) or "public").strip().lower()

    if not result.platform_url and not result.platform_video_id:
        lines.append(
            "TikTok is still processing this upload. If it is not on your profile yet, open the TikTok app and check **Inbox** or **Drafts**."
        )

    if level == "SELF_ONLY":
        if canon == "unlisted":
            lines.append(
                "You chose **unlisted** — TikTok received **Only you**. It may appear in **Drafts** or **Inbox** until you post publicly from the app; links may not work for others until then."
            )
        else:
            lines.append(
                "Posted as **Only you** on TikTok — check **Drafts** or **Inbox** if it is not on your profile yet. A share link may not work for others until you publish publicly."
            )
    elif level == "MUTUAL_FOLLOW_FRIENDS":
        lines.append(
            "Posted with **friends / mutual followers** visibility on TikTok — links may not work for everyone."
        )
    elif canon in ("unlisted", "private") and level == "PUBLIC_TO_EVERYONE":
        lines.append(
            f"Upload privacy was **{canon}** — confirm visibility in the TikTok app if the link behaves unexpectedly."
        )
    return lines


def _normalize_post_url(result: PlatformResult) -> Optional[str]:
    u = (result.platform_url or "").strip()
    if u.startswith("http"):
        plat = (result.platform or "").lower()
        if plat == "facebook" and "facebook.com/video/" in u and "/watch/" not in u:
            vid = getattr(result, "platform_video_id", None)
            if vid:
                return f"https://www.facebook.com/watch/?v={vid}"
        return u
    return None


def _fallback_post_url(result: PlatformResult) -> Optional[str]:
    vid = getattr(result, "platform_video_id", None)
    if not vid:
        return None
    plat = (result.platform or "").lower()
    if plat == "tiktok":
        handle = getattr(result, "account_username", None) or ""
        h = str(handle).strip().lstrip("@")
        if h:
            return f"https://www.tiktok.com/@{h}/video/{vid}"
        return f"https://www.tiktok.com/video/{vid}"
    if plat == "youtube":
        payload = getattr(result, "response_payload", None) or {}
        if isinstance(payload, dict) and payload.get("youtube_long_form_rights_guard"):
            return f"https://www.youtube.com/watch?v={vid}"
        return f"https://www.youtube.com/shorts/{vid}"
    if plat == "facebook":
        return f"https://www.facebook.com/watch/?v={vid}"
    return None


async def fetch_user_discord_webhook_from_db(db_pool, user_id: str) -> Optional[str]:
    """Resolve Discord webhook from DB (user_settings → user_preferences → users.preferences)."""
    if db_pool is None or not user_id:
        return None
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(
                  NULLIF(TRIM(us.discord_webhook), ''),
                  NULLIF(TRIM(up.discord_webhook), ''),
                  NULLIF(TRIM(COALESCE(
                    u.preferences->>'discordWebhook',
                    u.preferences->>'discord_webhook'
                  )), '')
                ) AS url
                FROM users u
                LEFT JOIN user_settings us ON us.user_id = u.id
                LEFT JOIN user_preferences up ON up.user_id = u.id
                WHERE u.id = $1
                """,
                user_id,
            )
        if row and row["url"]:
            url = str(row["url"]).strip()
            return url or None
    except Exception as e:
        logger.warning("Could not resolve user discord webhook from DB user=%s: %s", user_id, e)
    return None


async def _resolve_user_discord_webhook(ctx: JobContext, db_pool=None) -> Optional[str]:
    """Find the user's Discord webhook URL.

    Checks ``ctx.user_settings`` first (snake_case + camelCase), and falls back to a
    direct DB ``COALESCE`` across ``user_settings``, ``user_preferences``, and
    ``users.preferences`` JSONB when the in-memory copy is missing or stale. This
    mirrors the lookup used by ``POST /api/settings/test-discord-webhook`` so the
    pipeline stays consistent with what the user verified in the UI.
    """
    us = ctx.user_settings or {}
    raw = us.get("discord_webhook") or us.get("discordWebhook")
    if isinstance(raw, str):
        raw = raw.strip() or None
    if raw:
        return raw

    uid = getattr(ctx, "user_id", None)
    if not uid:
        return None
    return await fetch_user_discord_webhook_from_db(db_pool, str(uid))


def _is_allowed_discord_webhook_url(url: str) -> bool:
    """True for official Discord webhook URLs (including legacy discordapp.com and PTB/canary)."""
    try:
        u = urlparse((url or "").strip())
        if u.scheme != "https" or not u.hostname:
            return False
        host = u.hostname.lower()
        if host not in (
            "discord.com",
            "discordapp.com",
            "canary.discord.com",
            "ptb.discord.com",
        ):
            return False
        return (u.path or "").startswith("/api/webhooks/")
    except Exception:
        return False


def extract_pikzels_preview_image_url(data: Any) -> Optional[str]:
    """Best-effort HTTPS image URL from a Pikzels v2 JSON body (flat or nested ``data``)."""
    if not isinstance(data, dict):
        return None
    keys = ("output", "image_url", "url", "pikzels_cdn_url", "preview_url", "thumbnail_url")
    for k in keys:
        v = data.get(k)
        if isinstance(v, str) and v.startswith("https://"):
            return v.strip()[:2048]
    nested = data.get("data")
    if isinstance(nested, dict):
        for k in keys:
            v = nested.get(k)
            if isinstance(v, str) and v.startswith("https://"):
                return v.strip()[:2048]
    return None


def _pikzels_score_embed_description(response_data: Any, operation: str, upload_id: Optional[str]) -> Optional[str]:
    """Build markdown description for score/analyze; returns None if no score in body."""
    data = response_data if isinstance(response_data, dict) else {}
    nested = data.get("data") if isinstance(data.get("data"), dict) else {}
    score = data.get("main_score")
    if score is None:
        score = data.get("score")
    if score is None and nested:
        score = nested.get("main_score", nested.get("score"))
    if score is None:
        return None
    sugg_raw = data.get("suggestion") or nested.get("suggestion") or ""
    sugg = str(sugg_raw).strip()[:900]
    op_label = (operation or "score").strip()[:80]
    parts = [f"Pikzels **{op_label}**", "", f"**Score:** {score}"]
    if sugg:
        parts.extend(["", sugg])
    uid = (upload_id or "").strip()
    if uid:
        parts.extend(["", f"Upload id: `{uid}`"])
    return "\n".join(parts)[:1800]


async def notify_user_pikzels_generation(
    db_pool,
    user_id: str,
    *,
    operation: str,
    response_data: Any,
    upload_id: Optional[str] = None,
    source_image_url: Optional[str] = None,
) -> None:
    """
    When the user has configured a Discord webhook (same resolution as upload notify),
    send a compact embed. Image flows use ``image``; score flows add analysis text and,
    when available, the analyzed thumbnail (response CDN URL, or ``source_image_url``).
    """
    if not db_pool or not user_id:
        return
    wh = await fetch_user_discord_webhook_from_db(db_pool, user_id)
    if not wh or not _is_allowed_discord_webhook_url(wh):
        return

    op_low = (operation or "").lower()
    is_score = "score" in op_low

    img = extract_pikzels_preview_image_url(response_data)
    if not img and source_image_url:
        su = str(source_image_url).strip()
        if su.startswith("https://"):
            img = su[:2048]

    score_desc = _pikzels_score_embed_description(response_data, operation, upload_id) if is_score else None

    if not img and not score_desc:
        return

    embed: Dict[str, Any] = {
        "title": "📊 Pikzels analyze" if is_score else "🖼️ Pikzels output",
        "color": 0x10B981 if is_score else 0x5865F2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "UploadM8 · Pikzels"},
    }

    if score_desc:
        embed["description"] = score_desc
    elif img:
        op = (operation or "pikzels").strip()[:80]
        desc_lines = [f"Pikzels **{op}** finished with a new image."]
        uid = (upload_id or "").strip()
        if uid:
            desc_lines.append(f"Upload / context id: `{uid}`")
        embed["description"] = "\n".join(desc_lines)[:1800]
    else:
        return

    if img:
        embed["image"] = {"url": img}

    await _send_discord_webhook(wh, embeds=[embed])


async def run_notify_stage(ctx: JobContext, db_pool=None) -> JobContext:
    """
    Send notifications for completed upload.

    Process:
    1. Resolve user's Discord webhook (ctx.user_settings → DB fallback)
    2. Send embed with title / caption / hashtags / per-platform results
    3. Log explicit reason when no notification is sent
    """
    ctx.mark_stage("notify")

    user_webhook = await _resolve_user_discord_webhook(ctx, db_pool=db_pool)

    if not user_webhook:
        logger.info(
            f"[{ctx.upload_id}] notify: no user discord webhook configured — skipping user notification"
        )
        return ctx

    if not _is_allowed_discord_webhook_url(user_webhook):
        logger.warning(
            f"[{ctx.upload_id}] notify: user discord webhook is not a recognized Discord webhook URL — skipping"
        )
        return ctx

    logger.info(
        f"[{ctx.upload_id}] notify: sending user discord webhook "
        f"(success={ctx.is_success()} partial={ctx.is_partial_success()})"
    )
    await send_user_upload_notification(user_webhook, ctx, db_pool=db_pool)

    return ctx


async def send_user_upload_notification(webhook_url: str, ctx: JobContext, db_pool=None):
    """Send upload status to user's Discord webhook.

    Embed includes:
      - Status (success / partial / failed)
      - AI-generated title, caption, and hashtags
      - Per-platform result with clickable post links
    """
    try:
        is_success = ctx.is_success()

        if is_success:
            color = 0x22c55e        # green
            status_title = "✅ Upload Completed"
            status_desc  = "Your video has been published successfully!"
        elif ctx.is_partial_success():
            color = 0xf97316        # orange
            status_title = "⚠️ Partial Upload"
            status_desc  = "Some platforms failed — check your queue for details."
        else:
            color = 0xef4444        # red
            status_title = "❌ Upload Failed"
            status_desc  = f"Upload failed: {ctx.error_message or 'Unknown error'}"

        # ── Content (same precedence as publish: explicit user copy, then hydrated AI) ─
        video_title = (
            ctx.get_effective_title()
            if hasattr(ctx, "get_effective_title")
            else (
                getattr(ctx, "title", None)
                or getattr(ctx, "ai_title", None)
                or ctx.filename
                or "Untitled"
            )
        )
        video_caption = (
            ctx.get_effective_caption()
            if hasattr(ctx, "get_effective_caption")
            else (getattr(ctx, "caption", None) or getattr(ctx, "ai_caption", None) or "")
        )
        raw_tags = getattr(ctx, "ai_hashtags", None)
        if raw_tags is None:
            raw_tags = getattr(ctx, "hashtags", None) or []
        flat_ai = _flatten_hashtag_raw(raw_tags)
        hashtag_str = _hashtags_for_discord_line(flat_ai)

        by_plat = _build_hashtags_by_platform_block(ctx)
        if not by_plat and hashtag_str:
            by_plat = ""

        # ── Build fields ─────────────────────────────────────────────────────
        fields: List[dict] = [
            {"name": "📹 Title",    "value": str(video_title)[:256],  "inline": False},
        ]

        if video_caption:
            cap_val = str(video_caption)
            fields.append({
                "name": "📝 Caption",
                "value": (cap_val[:500] + "…") if len(cap_val) > 500 else cap_val,
                "inline": False,
            })

        if by_plat:
            fields.append({
                "name": "🏷️ Hashtags (by platform)",
                "value": by_plat[:1020],
                "inline": False,
            })
        elif hashtag_str:
            fields.append({
                "name": "🏷️ Hashtags",
                "value": hashtag_str[:1020],
                "inline": False,
            })

        m8_block = _build_m8_ai_hashtags_block(ctx)
        if m8_block:
            fields.append({
                "name": "🧠 AI hashtag variants (M8)",
                "value": m8_block[:1020],
                "inline": False,
            })

        # Platforms summary (unique platforms; multi-account may repeat)
        plat_set = set(ctx.platforms) if ctx.platforms else set()
        fields.append({"name": "📤 Platforms", "value": ", ".join(sorted(plat_set)) or "None", "inline": True})
        fields.append({"name": "📁 File",      "value": (ctx.filename or "—")[:80],          "inline": True})

        preview_url = None
        try:
            from services.upload_notification_preview import (
                resolve_upload_notification_preview_https_url,
                thumbnail_quality_summary_text,
            )

            tq = thumbnail_quality_summary_text(ctx)
            if tq:
                fields.append({"name": "🖼️ Thumbnail", "value": tq[:1020], "inline": False})
            preview_url = await resolve_upload_notification_preview_https_url(db_pool, ctx)
        except Exception as e:
            logger.debug("[%s] upload notify preview prep skipped: %s", getattr(ctx, "upload_id", ""), e)

        try:
            from stages.youtube_copyright_shorts import get_youtube_copyright_notice

            yt_note = get_youtube_copyright_notice(ctx)
            if yt_note and yt_note.get("message"):
                fields.append(
                    {
                        "name": "🎵 YouTube / music (ACR)",
                        "value": str(yt_note.get("message") or "")[:900],
                        "inline": False,
                    }
                )
        except Exception:
            pass

        # ── Per-platform/account results with live post URLs ────────────────────
        for result in ctx.platform_results:
            icon      = "✅" if result.success else "❌"
            plat_name = result.platform.title()
            account_label = (
                getattr(result, "account_username", None)
                or getattr(result, "account_name", None)
                or getattr(result, "account_id", None)
            )
            if account_label:
                field_name = f"{icon} {plat_name} ({account_label})"
            else:
                field_name = f"{icon} {plat_name}"

            if result.success:
                url = _normalize_post_url(result) or _fallback_post_url(result)
                chunks: List[str] = []
                if url:
                    chunks.append(f"[View post]({url})")
                elif result.publish_id:
                    chunks.append(f"Accepted — publish_id: `{result.publish_id}`")
                else:
                    chunks.append("Published ✓")

                plat_lc = (result.platform or "").lower()
                if plat_lc == "tiktok":
                    for line in _tiktok_status_lines(ctx, result):
                        if line:
                            chunks.append(line)

                if plat_lc == "instagram" and not url:
                    chunks.append(
                        "Instagram link not available yet — open Instagram or wait for queue sync to refresh the permalink."
                    )

                value = "\n".join(chunks)[:1020]
            else:
                raw_err = result.error_message or result.error_code or "Unknown error"
                value = str(raw_err)[:256]

            fields.append({"name": field_name, "value": value, "inline": False})

        embed: Dict[str, Any] = {
            "title":       status_title,
            "description": status_desc,
            "color":       color,
            "fields":      fields,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "footer":      {"text": "UploadM8"},
        }
        if preview_url:
            embed["image"] = {"url": preview_url[:2048]}

        await _send_discord_webhook(webhook_url, embeds=[embed])

    except Exception as e:
        logger.exception(f"[{ctx.upload_id}] User webhook notification failed: {e}")


async def notify_user_publish_confirmed(
    db_pool,
    *,
    user_id: str,
    upload_id: str,
    platform: str,
    post_url: str = "",
) -> None:
    """
    Step-B confirmation: platform APIs report the post is fully live (e.g. TikTok
    PUBLISH_COMPLETE with video_id). Discord webhook ping only (no email).
    """
    if not db_pool or not user_id or not upload_id:
        return
    plat_key = (platform or "").strip().lower()
    if plat_key not in ("tiktok", "youtube"):
        return

    filename = ""
    try:
        async with db_pool.acquire() as conn:
            fn = await conn.fetchval("SELECT filename FROM uploads WHERE id = $1", upload_id)
            if fn:
                filename = str(fn)
    except Exception as e:
        logger.warning("publish_confirmed: could not load filename upload=%s: %s", upload_id, e)

    plat_label = _platform_label(plat_key)
    wh = await fetch_user_discord_webhook_from_db(db_pool, user_id)
    if wh and _is_allowed_discord_webhook_url(wh):
        desc_parts = [
            (filename or "Your video") + f" is confirmed live on **{plat_label}**.",
        ]
        if (post_url or "").strip().startswith("http"):
            desc_parts.append(f"[View post]({post_url.strip()})")
        try:
            await _send_discord_webhook(
                wh,
                embeds=[
                    {
                        "title": f"✅ {plat_label} confirmed",
                        "description": "\n\n".join(desc_parts)[:1800],
                        "color": 0x22c55e,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "footer": {"text": "UploadM8 · publish confirmation"},
                    }
                ],
            )
        except Exception as e:
            logger.warning("publish_confirmed Discord failed upload=%s: %s", upload_id, e)


# ============================================================
# Admin Notifications
# ============================================================

async def _get_admin_webhook(db_pool=None) -> str:
    # Priority: explicit env override -> admin_settings saved webhook
    if ADMIN_DISCORD_WEBHOOK_URL:
        return ADMIN_DISCORD_WEBHOOK_URL
    if db_pool is None:
        return ""
    try:
        wh = await db_stage.load_admin_notification_webhook(db_pool)
        return wh or ""
    except Exception:
        return ""


async def notify_admin_signup(email: str, name: str, tier: str = "free", db_pool=None):
    """Notify admin of new user signup."""
    webhook = SIGNUP_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": "🎉 New User Signup",
        "color": 0x3b82f6,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "Name", "value": name, "inline": True},
            {"name": "Tier", "value": tier, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_trial_started(email: str, name: str, plan: str, db_pool=None):
    """Notify admin of new trial signup."""
    webhook = TRIAL_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": "🚀 Trial Started",
        "color": 0x8b5cf6,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "Name", "value": name, "inline": True},
            {"name": "Plan", "value": plan, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_mrr_collected(amount: float, customer_email: str, plan: str, db_pool=None):
    """Notify admin of MRR collection."""
    webhook = MRR_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": "💰 MRR Collected",
        "color": 0x22c55e,
        "fields": [
            {"name": "Amount", "value": f"${amount:.2f}", "inline": True},
            {"name": "Customer", "value": customer_email, "inline": True},
            {"name": "Plan", "value": plan, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_error(error_type: str, details: dict, db_pool=None):
    """Notify admin of system error — DB incident row + email, then Discord embed."""
    if db_pool:
        try:
            from services.ops_incidents import record_operational_incident

            await record_operational_incident(
                db_pool,
                source="worker",
                incident_type=str(error_type)[:120],
                subject=f"Worker: {error_type}",
                body=str(details.get("error") or details.get("message") or "")[:8000],
                details=dict(details) if isinstance(details, dict) else {"raw": str(details)},
                user_id=details.get("user_id"),
                upload_id=details.get("upload_id"),
                alert_discord=False,
            )
        except Exception as ex:
            logger.warning("notify_admin_error incident log failed: %s", ex)

    webhook = ERROR_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return

    embed = {
        "title": f"🚨 Error: {error_type}",
        "color": 0xEF4444,
        "description": f"```json\n{json.dumps(details, indent=2, default=str)[:1500]}\n```",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_worker_start(db_pool=None):
    """Notify admin that worker has started."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    await _send_discord_webhook(
        webhook,
        content="🟢 UploadM8 Worker started"
    )


async def notify_admin_worker_stop(db_pool=None):
    """Notify admin that worker has stopped."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    await _send_discord_webhook(
        webhook,
        content="🔴 UploadM8 Worker stopped"
    )


async def notify_admin_daily_summary(data: dict, db_pool=None):
    """Send daily summary to admin."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    embed = {
        "title": "📊 Daily Summary",
        "color": 0x3b82f6,
        "fields": [
            {"name": "New Users", "value": str(data.get("new_users", 0)), "inline": True},
            {"name": "Uploads", "value": str(data.get("uploads", 0)), "inline": True},
            {"name": "MRR", "value": f"${data.get('mrr', 0):.2f}", "inline": True},
            {"name": "OpenAI Cost", "value": f"${data.get('openai_cost', 0):.2f}", "inline": True},
            {"name": "Storage Used", "value": f"{data.get('storage_gb', 0):.2f} GB", "inline": True},
            {"name": "Active Users", "value": str(data.get("active_users", 0)), "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    await _send_discord_webhook(webhook, embeds=[embed])


# ============================================================
# Email Notifications
# ============================================================

async def send_welcome_email(email: str, name: str):
    """Send welcome email to new user."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.info(f"Welcome email skipped (no Mailgun): {email}")
        return
    
    subject = "Welcome to UploadM8! 🎉"
    html = f"""
    <h1>Welcome to UploadM8, {name}!</h1>
    <p>Thanks for signing up. You're now ready to upload videos to multiple platforms with a single click.</p>
    <h2>Getting Started:</h2>
    <ol>
        <li>Connect your social media accounts (TikTok, YouTube, Instagram, Facebook)</li>
        <li>Upload your first video</li>
        <li>Let our AI generate titles, captions, and hashtags</li>
        <li>Publish to all platforms at once!</li>
    </ol>
    <p><a href="{FRONTEND_URL}/dashboard.html" style="background: #f97316; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px;">Go to Dashboard</a></p>
    <p>If you have any questions, just reply to this email.</p>
    <p>- The UploadM8 Team</p>
    """
    
    await _send_mailgun_email(email, subject, html)


async def send_upgrade_email(email: str, name: str, new_tier: str):
    """Send upgrade confirmation email."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        return
    
    subject = f"Welcome to {new_tier.title()}! 🚀"
    html = f"""
    <h1>Upgrade Confirmed!</h1>
    <p>Hi {name},</p>
    <p>Your account has been upgraded to <strong>{new_tier.title()}</strong>!</p>
    <p>You now have access to:</p>
    <ul>
        <li>Higher upload limits</li>
        <li>AI-powered captions and thumbnails</li>
        <li>Smart scheduling</li>
        <li>And more!</li>
    </ul>
    <p><a href="{FRONTEND_URL}/dashboard.html">Go to Dashboard</a></p>
    <p>- The UploadM8 Team</p>
    """
    
    await _send_mailgun_email(email, subject, html)


async def send_tier_change_email(email: str, name: str, old_tier: str, new_tier: str, is_upgrade: bool):
    """Send tier change notification email."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        return
    
    if is_upgrade:
        subject = f"🎉 Upgraded to {new_tier.title()}"
        message = f"Your account has been upgraded from {old_tier.title()} to {new_tier.title()}!"
    else:
        subject = f"Plan Changed to {new_tier.title()}"
        message = f"Your account has been changed from {old_tier.title()} to {new_tier.title()}."
    
    html = f"""
    <h1>Plan Changed</h1>
    <p>Hi {name},</p>
    <p>{message}</p>
    <p><a href="{FRONTEND_URL}/settings.html">View your account settings</a></p>
    <p>- The UploadM8 Team</p>
    """
    
    await _send_mailgun_email(email, subject, html)


# ============================================================
# Internal Helpers
# ============================================================

async def _send_discord_webhook(webhook_url: str, content: str = None, embeds: List[dict] = None):
    """Send message to Discord webhook."""
    if not webhook_url:
        return
    
    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(webhook_url, json=payload)
            if response.status_code not in (200, 204):
                # Discord returns JSON error details (e.g. "embed too large", "invalid url")
                # in the body — surface them so silent drops are diagnosable.
                body = (response.text or "")[:500]
                logger.warning(
                    f"Discord webhook failed: HTTP {response.status_code} body={body}"
                )
    except Exception as e:
        logger.warning(f"Discord webhook error: {e}")


async def _send_mailgun_email(to: str, subject: str, html: str):
    """Send email via Mailgun API."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        return
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={
                    "from": MAIL_FROM,
                    "to": to,
                    "subject": subject,
                    "html": html,
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Mailgun send failed: {response.status_code}")
            else:
                logger.info(f"Email sent to {to}: {subject}")
                
    except Exception as e:
        logger.warning(f"Mailgun error: {e}")
