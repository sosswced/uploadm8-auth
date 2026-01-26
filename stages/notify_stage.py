"""
UploadM8 Notify Stage
=====================
Send Discord notifications for upload results.
"""

import os
import logging

import httpx

from .errors import NotifyError, ErrorCode
from .context import JobContext


logger = logging.getLogger("uploadm8-worker")


ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")


async def send_discord_message(webhook_url: str, message: str) -> bool:
    """
    Send message to Discord webhook.
    
    Args:
        webhook_url: Discord webhook URL
        message: Message content
        
    Returns:
        True if sent successfully
    """
    if not webhook_url:
        return False
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                webhook_url,
                json={"content": message}
            )
            return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"Discord notification failed: {e}")
        return False


async def notify_user_success(ctx: JobContext) -> None:
    """Send success notification to user."""
    if not ctx.discord_webhook:
        return
    
    platforms = ", ".join(ctx.get_succeeded_platforms())
    title = ctx.final_title or ctx.filename
    
    msg = f"✅ **Upload Complete**: {title}"
    if platforms:
        msg += f"\n📱 Published to: {platforms}"
    if ctx.trill:
        msg += f"\n⚡ Trill Score: {ctx.trill.score}/100 ({ctx.trill.bucket})"
    
    await send_discord_message(ctx.discord_webhook, msg)


async def notify_user_partial(ctx: JobContext) -> None:
    """Send partial success notification to user."""
    if not ctx.discord_webhook:
        return
    
    succeeded = ctx.get_succeeded_platforms()
    failed = ctx.get_failed_platforms()
    title = ctx.final_title or ctx.filename
    
    msg = f"⚠️ **Partial Upload**: {title}"
    if succeeded:
        msg += f"\n✅ Succeeded: {', '.join(succeeded)}"
    if failed:
        msg += f"\n❌ Failed: {', '.join(failed)}"
    
    await send_discord_message(ctx.discord_webhook, msg)


async def notify_user_failure(ctx: JobContext) -> None:
    """Send failure notification to user."""
    if not ctx.discord_webhook:
        return
    
    title = ctx.final_title or ctx.filename
    error = ctx.error_detail or ctx.error_code or "Unknown error"
    
    msg = f"❌ **Upload Failed**: {title}\nError: {error[:200]}"
    
    await send_discord_message(ctx.discord_webhook, msg)


async def notify_admin_job_complete(ctx: JobContext) -> None:
    """Send job completion notification to admin."""
    if not ADMIN_DISCORD_WEBHOOK_URL:
        return
    
    status_emoji = {
        "completed": "✅",
        "partial": "⚠️",
        "failed": "❌"
    }.get(ctx.status, "❓")
    
    msg = (
        f"{status_emoji} **Job Complete**\n"
        f"Upload: {ctx.upload_id}\n"
        f"User: {ctx.user_id}\n"
        f"Status: {ctx.status}"
    )
    
    if ctx.trill:
        msg += f"\nTrill: {ctx.trill.score} ({ctx.trill.bucket})"
    
    if ctx.platform_results:
        results = [f"{r.platform}: {'✅' if r.success else '❌'}" for r in ctx.platform_results]
        msg += f"\nPlatforms: {', '.join(results)}"
    
    if ctx.error_code:
        msg += f"\nError: {ctx.error_code}"
    
    await send_discord_message(ADMIN_DISCORD_WEBHOOK_URL, msg)


async def notify_admin_worker_start() -> None:
    """Notify admin that worker has started."""
    if ADMIN_DISCORD_WEBHOOK_URL:
        await send_discord_message(
            ADMIN_DISCORD_WEBHOOK_URL,
            "🟢 **UploadM8 Worker Started**"
        )


async def notify_admin_worker_stop() -> None:
    """Notify admin that worker has stopped."""
    if ADMIN_DISCORD_WEBHOOK_URL:
        await send_discord_message(
            ADMIN_DISCORD_WEBHOOK_URL,
            "🔴 **UploadM8 Worker Stopped**"
        )


async def notify_admin_error(error: str, context: str = "") -> None:
    """Notify admin of a critical error."""
    if not ADMIN_DISCORD_WEBHOOK_URL:
        return
    
    msg = f"🚨 **Worker Error**\n{error}"
    if context:
        msg += f"\nContext: {context}"
    
    await send_discord_message(ADMIN_DISCORD_WEBHOOK_URL, msg)


async def run_notify_stage(ctx: JobContext) -> JobContext:
    """
    Execute notification stage.
    
    Args:
        ctx: Job context
        
    Returns:
        Context (unchanged)
    """
    try:
        # Notify user based on status
        if ctx.status == "completed":
            await notify_user_success(ctx)
        elif ctx.status == "partial":
            await notify_user_partial(ctx)
        elif ctx.status == "failed":
            await notify_user_failure(ctx)
        
        # Always notify admin
        await notify_admin_job_complete(ctx)
        
    except Exception as e:
        logger.warning(f"Notification stage error: {e}")
        # Don't fail the job for notification errors
    
    return ctx
