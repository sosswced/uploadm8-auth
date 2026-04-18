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
        return f"https://www.youtube.com/shorts/{vid}"
    if plat == "facebook":
        return f"https://www.facebook.com/watch/?v={vid}"
    return None


async def run_notify_stage(ctx: JobContext) -> JobContext:
    """
    Send notifications for completed upload.
    
    Process:
    1. Send user webhook if configured
    2. Log completion
    """
    ctx.mark_stage("notify")
    
    # Prefer snake_case; settings JSON may only have discordWebhook while user_settings
    # row still exposes discord_webhook=NULL as a present key.
    us = ctx.user_settings or {}
    user_webhook = us.get("discord_webhook") or us.get("discordWebhook")
    if isinstance(user_webhook, str):
        user_webhook = user_webhook.strip() or None

    if user_webhook:
        await send_user_upload_notification(user_webhook, ctx)
    
    return ctx


async def send_user_upload_notification(webhook_url: str, ctx: JobContext):
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

        # ── Content (AI-first, fall back to user-supplied) ───────────────────
        video_title = (
            getattr(ctx, "ai_title", None)
            or getattr(ctx, "title", None)
            or ctx.filename
            or "Untitled"
        )
        video_caption = (
            getattr(ctx, "ai_caption", None)
            or getattr(ctx, "caption", None)
            or ""
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

        embed = {
            "title":       status_title,
            "description": status_desc,
            "color":       color,
            "fields":      fields,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "footer":      {"text": "UploadM8"},
        }

        await _send_discord_webhook(webhook_url, embeds=[embed])

    except Exception as e:
        logger.warning(f"User webhook notification failed: {e}")


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
                logger.warning(f"Discord webhook failed: {response.status_code}")
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
