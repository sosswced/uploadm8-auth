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
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx

from .context import JobContext
from .errors import StageError, SkipStage
from . import db as db_stage

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
    """Send upload status to user's Discord webhook."""
    try:
        is_success = ctx.is_success()
        
        # Build embed
        if is_success:
            color = 0x22c55e  # Green
            title = "✅ Upload Completed"
            description = f"Your video has been uploaded successfully!"
        elif ctx.is_partial_success():
            color = 0xf97316  # Orange
            title = "⚠️ Partial Upload"
            description = f"Some platforms failed. Check your queue for details."
        else:
            color = 0xef4444  # Red
            title = "❌ Upload Failed"
            description = f"Upload failed: {ctx.error_message or 'Unknown error'}"
        
        fields = [
            {"name": "Filename", "value": ctx.filename[:50], "inline": True},
            {"name": "Platforms", "value": ", ".join(ctx.platforms) or "None", "inline": True},
        ]
        
        # Add platform results
        for result in ctx.platform_results:
            status = "✅" if result.success else "❌"
            value = result.platform_url if result.success else (result.error_message or "Failed")
            fields.append({
                "name": f"{status} {result.platform.title()}",
                "value": value[:100],
                "inline": False
            })
        
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "UploadM8"}
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
    """Notify admin of system error."""
    webhook = ERROR_DISCORD_WEBHOOK_URL or (await _get_admin_webhook(db_pool))
    if not webhook:
        return
    
    embed = {
        "title": f"🚨 Error: {error_type}",
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
