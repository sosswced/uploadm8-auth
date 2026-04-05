"""
UploadM8 Notification Stage
===========================
Send notifications via Discord webhooks.

Notifications:
- User webhook: Upload status (success/fail)
- Admin webhook: worker lifecycle + pipeline errors
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timezone
from typing import List

import asyncpg
import httpx

from .context import JobContext
from . import db as db_stage

logger = logging.getLogger("uploadm8-worker")

# Configuration
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")
ERROR_DISCORD_WEBHOOK_URL = os.environ.get("ERROR_DISCORD_WEBHOOK_URL", "")

async def run_notify_stage(ctx: JobContext) -> JobContext:
    """
    Send notifications for completed upload.
    
    Process:
    1. Send user webhook if configured
    """
    ctx.mark_stage("notify")
    
    # Get user's Discord webhook
    user_webhook = (ctx.user_settings or {}).get("discord_webhook")
    
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
            status_title = " Upload Completed"
            status_desc  = "Your video has been published successfully!"
        elif ctx.is_partial_success():
            color = 0xf97316        # orange
            status_title = "️ Partial Upload"
            status_desc  = "Some platforms failed — check your queue for details."
        else:
            color = 0xef4444        # red
            status_title = " Upload Failed"
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
        raw_tags = (
            getattr(ctx, "ai_hashtags", None)
            or getattr(ctx, "hashtags", None)
            or []
        )
        # Guard: never iterate a bare string character-by-character
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags] if raw_tags.strip() else []
        hashtag_str = " ".join(
            (str(t) if str(t).startswith("#") else f"#{t}")
            for t in raw_tags
            if str(t).strip()
        )

        # ── Build fields ─────────────────────────────────────────────────────
        fields: List[dict] = [
            {"name": " Title",    "value": str(video_title)[:256],  "inline": False},
        ]

        if video_caption:
            cap_val = str(video_caption)
            fields.append({
                "name": " Caption",
                "value": (cap_val[:500] + "…") if len(cap_val) > 500 else cap_val,
                "inline": False,
            })

        if hashtag_str:
            fields.append({
                "name": "️ Hashtags",
                "value": hashtag_str[:500],
                "inline": False,
            })

        # Platforms summary (unique platforms; multi-account may repeat)
        plat_set = set(ctx.platforms) if ctx.platforms else set()
        fields.append({"name": " Platforms", "value": ", ".join(sorted(plat_set)) or "None", "inline": True})
        fields.append({"name": " File",      "value": (ctx.filename or "—")[:80],          "inline": True})

        # ── Per-platform/account results with live post URLs ────────────────────
        for result in ctx.platform_results:
            icon      = "" if result.success else ""
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
                url = result.platform_url
                if not url and getattr(result, "platform_video_id", None):
                    plat = (result.platform or "").lower()
                    vid = result.platform_video_id
                    if plat == "tiktok":
                        handle = getattr(result, "account_username", None) or "_"
                        url = f"https://www.tiktok.com/@{handle}/video/{vid}"
                    elif plat == "youtube":
                        url = f"https://www.youtube.com/shorts/{vid}"
                    elif plat == "facebook":
                        url = f"https://www.facebook.com/watch/?v={vid}"
                    # Instagram needs shortcode (from platform_url); media_id alone won't work
                if url and str(url).startswith("http"):
                    value = f"[View Post]({url})"
                elif result.publish_id:
                    value = f"Accepted — publish_id: `{result.publish_id}`"
                else:
                    value = "Published "
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

    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, TypeError, ValueError, AttributeError, KeyError) as e:
        logger.warning("User webhook notification failed: %s", e)


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
    except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError) as e:
        logger.debug("_get_admin_webhook: %s", e)
        return ""


async def notify_admin_error(error_type: str, details: dict, db_pool=None):
    """Notify admin of system error."""
    webhook = ERROR_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": f" Error: {error_type}",
        "color": 0xef4444,
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
        content=" UploadM8 Worker started"
    )


async def notify_admin_worker_stop(db_pool=None):
    """Notify admin that worker has stopped."""
    webhook = await _get_admin_webhook(db_pool)
    if not webhook:
        return
    
    await _send_discord_webhook(
        webhook,
        content=" UploadM8 Worker stopped"
    )


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
    except asyncio.CancelledError:
        raise
    except httpx.HTTPError as e:
        logger.warning("Discord webhook error: %s", e)


