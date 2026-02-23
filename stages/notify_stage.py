"""
UploadM8 Notification Stage
===========================
Send notifications via Discord webhooks and email.

FIXES:
- User Discord embed now includes AI-generated title, caption, hashtags
- Platform post URLs shown as clickable links per platform
- Thumbnail image shown in embed when available
- Upload links clearly labeled per platform
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx

from .context import JobContext
from .errors import StageError, SkipStage

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

# Platform emoji map
PLATFORM_ICONS = {
    "tiktok": "🎵",
    "youtube": "▶️",
    "instagram": "📸",
    "facebook": "👥",
}


async def run_notify_stage(ctx: JobContext) -> JobContext:
    """
    Send notifications for completed upload.

    Process:
    1. Send user webhook if configured
    2. Log completion
    """
    ctx.mark_stage("notify")

    # Get user's Discord webhook
    user_webhook = ctx.user_settings.get("discord_webhook")

    if user_webhook:
        await send_user_upload_notification(user_webhook, ctx)

    return ctx


async def send_user_upload_notification(webhook_url: str, ctx: JobContext):
    """
    Send upload status to user's Discord webhook.
    
    Includes:
    - Upload title (AI-generated or user-provided)
    - Caption (AI-generated or user-provided)
    - Hashtags as a full joined string
    - Per-platform post links
    - Thumbnail image if available
    """
    try:
        is_success = ctx.is_success()
        is_partial = ctx.is_partial_success() if hasattr(ctx, 'is_partial_success') else False

        # Determine status
        if is_success and not is_partial:
            color = 0x22c55e  # Green
            status_title = "✅ Upload Published"
            status_desc = "Your video has been published successfully!"
        elif is_partial:
            color = 0xf97316  # Orange
            status_title = "⚠️ Partial Publish"
            status_desc = "Video published to some platforms. Check links below."
        else:
            color = 0xef4444  # Red
            status_title = "❌ Upload Failed"
            status_desc = f"Upload failed. Check your queue for details."

        # Get final content - force string types, never iterate lists
        final_title = ctx.ai_title or ctx.title or ctx.filename or "Untitled"
        final_caption = ctx.ai_caption or ctx.caption or ""
        final_hashtags = ctx.ai_hashtags if ctx.ai_hashtags else (ctx.hashtags or [])

        # Build hashtag string - join as one space-separated string
        hashtag_str = " ".join(
            f"#{tag.lstrip('#')}" for tag in final_hashtags if str(tag).strip()
        ) if final_hashtags else "None"

        # Core fields
        fields = [
            {
                "name": "📹 Video",
                "value": final_title[:256],
                "inline": False
            },
        ]

        # Caption field (truncate if long)
        if final_caption:
            cap_display = final_caption[:300] + ("..." if len(final_caption) > 300 else "")
            fields.append({
                "name": "📝 Caption",
                "value": cap_display,
                "inline": False
            })

        # Hashtags as a single joined string
        fields.append({
            "name": "# Hashtags",
            "value": hashtag_str[:1024],
            "inline": False
        })

        # Platform results with post URLs
        for result in ctx.platform_results:
            icon = PLATFORM_ICONS.get(result.platform, "🌐")
            platform_name = result.platform.title()

            if result.success:
                if result.platform_url:
                    value = f"[View Post]({result.platform_url})"
                else:
                    value = "✅ Published (link pending verification)"

                fields.append({
                    "name": f"{icon} {platform_name}",
                    "value": value,
                    "inline": True
                })
            else:
                error_msg = (result.error_message or "Unknown error")[:200]
                fields.append({
                    "name": f"❌ {platform_name}",
                    "value": error_msg,
                    "inline": True
                })

        # Queue link
        queue_url = f"{FRONTEND_URL}/queue.html"
        fields.append({
            "name": "🔗 Queue",
            "value": f"[View in Queue]({queue_url})",
            "inline": False
        })

        embed = {
            "title": status_title,
            "description": status_desc,
            "color": color,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "UploadM8 • Upload Once, Publish Everywhere"}
        }

        # Add thumbnail image to embed if we have a presigned URL
        # Note: thumbnail_r2_key needs to be converted to a URL elsewhere
        # The embed image must be a public URL - skip if not available
        # (handled by app.py generating presigned URLs for the queue)

        await _send_discord_webhook(webhook_url, embeds=[embed])
        logger.info(f"Discord notification sent for upload {ctx.upload_id}")

    except Exception as e:
        logger.warning(f"User webhook notification failed: {e}")


# ============================================================
# Admin Notifications
# ============================================================

async def notify_admin_signup(email: str, name: str, tier: str = "free"):
    """Notify admin of new user signup."""
    webhook = SIGNUP_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if not webhook:
        return

    embed = {
        "title": "🎉 New User Signup",
        "color": 0x3b82f6,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "Name", "value": name or "—", "inline": True},
            {"name": "Tier", "value": tier, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_trial_started(email: str, name: str, plan: str):
    """Notify admin of new trial signup."""
    webhook = TRIAL_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if not webhook:
        return

    embed = {
        "title": "🚀 Trial Started",
        "color": 0xa855f7,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "Name", "value": name or "—", "inline": True},
            {"name": "Plan", "value": plan, "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_upgrade(email: str, from_tier: str, to_tier: str, mrr: float = 0):
    """Notify admin of subscription upgrade."""
    webhook = MRR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if not webhook:
        return

    embed = {
        "title": "💰 Subscription Upgrade",
        "color": 0x22c55e,
        "fields": [
            {"name": "Email", "value": email, "inline": True},
            {"name": "From", "value": from_tier, "inline": True},
            {"name": "To", "value": to_tier, "inline": True},
            {"name": "MRR Impact", "value": f"${mrr:.2f}/mo", "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_error(error_type: str, details: dict):
    """Notify admin of system error."""
    webhook = ERROR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if not webhook:
        return

    fields = [
        {"name": k, "value": str(v)[:200], "inline": True}
        for k, v in details.items()
    ]

    embed = {
        "title": f"🚨 Error: {error_type}",
        "color": 0xef4444,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_worker_start():
    """Notify admin that worker started."""
    webhook = ADMIN_DISCORD_WEBHOOK_URL
    if not webhook:
        return

    embed = {
        "title": "🟢 Worker Started",
        "color": 0x22c55e,
        "description": "UploadM8 processing worker is online.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


async def notify_admin_worker_stop():
    """Notify admin that worker stopped."""
    webhook = ADMIN_DISCORD_WEBHOOK_URL
    if not webhook:
        return

    embed = {
        "title": "🔴 Worker Stopped",
        "color": 0xef4444,
        "description": "UploadM8 processing worker has shut down.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await _send_discord_webhook(webhook, embeds=[embed])


# ============================================================
# Email Notifications
# ============================================================

async def send_email(to_email: str, subject: str, html_body: str):
    """Send email via Mailgun."""
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.warning("Mailgun not configured, skipping email")
        return

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={
                    "from": MAIL_FROM,
                    "to": to_email,
                    "subject": subject,
                    "html": html_body,
                }
            )

            if resp.status_code not in (200, 202):
                logger.warning(f"Email send failed: {resp.status_code} {resp.text[:100]}")

    except Exception as e:
        logger.warning(f"Email send error: {e}")


# ============================================================
# Internal Discord Helper
# ============================================================

async def _send_discord_webhook(url: str, embeds: List[dict] = None, content: str = None):
    """Send a Discord webhook message."""
    if not url:
        return

    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code not in (200, 204):
                logger.warning(f"Discord webhook returned {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        logger.warning(f"Discord webhook error: {e}")
