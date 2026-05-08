"""
UploadM8 notification helpers — Discord webhooks + Mailgun.

Resolution order for the destination Discord webhook:

  1. Per-event env var (e.g. SIGNUP_DISCORD_WEBHOOK_URL)
  2. ADMIN_DISCORD_WEBHOOK_URL env var
  3. core.state.admin_settings_cache["notifications"]["admin_webhook_url"]
     (populated at app startup from admin_settings.settings_json — same value
     edited in Admin → Notification Settings).

This way the web process honours whatever the admin saved in the UI even when
no env var is set on the host. Failures are LOGGED (not silently swallowed) so
misconfigured webhooks are diagnosable from the server logs.
"""

import logging
from typing import Optional

import httpx

import core.state
from core.config import (
    ADMIN_DISCORD_WEBHOOK_URL,
    SIGNUP_DISCORD_WEBHOOK_URL,
    MRR_DISCORD_WEBHOOK_URL,
    MAILGUN_API_KEY,
    MAILGUN_DOMAIN,
    MAIL_FROM,
)

logger = logging.getLogger("uploadm8-api")


def _admin_webhook_from_cache() -> str:
    """Read the Discord webhook saved via Admin → Notification Settings."""
    try:
        notif = core.state.admin_settings_cache.get("notifications") or {}
        url = (notif.get("admin_webhook_url") or "").strip()
        return url
    except Exception:
        return ""


def _resolve_webhook(*candidates: Optional[str]) -> str:
    """First non-empty candidate, falling back to admin_settings webhook."""
    for c in candidates:
        if c and c.strip():
            return c.strip()
    return _admin_webhook_from_cache()


async def discord_notify(webhook_url: str, content: str = None, embeds: list = None) -> bool:
    """POST to a Discord webhook. Returns True on 2xx, False otherwise.

    Logs HTTP status + response body on non-2xx so silent drops are diagnosable.
    """
    if not webhook_url:
        return False
    payload: dict = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(webhook_url, json=payload)
            if r.status_code in (200, 204):
                return True
            body = (r.text or "")[:500]
            logger.warning(
                "discord_notify failed: HTTP %s body=%s", r.status_code, body
            )
            return False
    except Exception as e:
        logger.warning("discord_notify error: %s", e)
        return False


async def notify_signup(email: str, name: str) -> bool:
    wh = _resolve_webhook(SIGNUP_DISCORD_WEBHOOK_URL, ADMIN_DISCORD_WEBHOOK_URL)
    if not wh:
        logger.info("notify_signup skipped: no admin webhook configured")
        return False
    return await discord_notify(
        wh,
        embeds=[{
            "title": "\U0001f389 New Signup",
            "color": 0x22c55e,
            "fields": [
                {"name": "Email", "value": email or "—"},
                {"name": "Name", "value": name or "—"},
            ],
        }],
    )


async def notify_mrr(amount: float, email: str, plan: str, event_type: str = "charge") -> bool:
    wh = _resolve_webhook(MRR_DISCORD_WEBHOOK_URL, ADMIN_DISCORD_WEBHOOK_URL)
    if not wh:
        logger.info("notify_mrr skipped: no admin webhook configured")
        return False
    return await discord_notify(
        wh,
        embeds=[{
            "title": f"\U0001f4b0 {event_type.title()}",
            "color": 0x22c55e,
            "fields": [
                {"name": "Amount", "value": f"${amount:.2f}"},
                {"name": "Email", "value": email or "—"},
                {"name": "Plan", "value": plan or "—"},
            ],
        }],
    )


async def notify_topup(amount: float, email: str, wallet: str, tokens: int) -> bool:
    wh = _resolve_webhook(MRR_DISCORD_WEBHOOK_URL, ADMIN_DISCORD_WEBHOOK_URL)
    if not wh:
        logger.info("notify_topup skipped: no admin webhook configured")
        return False
    return await discord_notify(
        wh,
        embeds=[{
            "title": "\U0001f4b3 Top-up Purchase",
            "color": 0x8b5cf6,
            "fields": [
                {"name": "Amount", "value": f"${amount:.2f}"},
                {"name": "Wallet", "value": (wallet or "").upper() or "—"},
                {"name": "Tokens", "value": str(tokens)},
                {"name": "Email", "value": email or "—"},
            ],
        }],
    )


async def notify_weekly_costs(
    openai_cost: float, storage_cost: float, compute_cost: float, revenue: float
) -> bool:
    wh = _resolve_webhook(ADMIN_DISCORD_WEBHOOK_URL)
    if not wh:
        logger.info("notify_weekly_costs skipped: no admin webhook configured")
        return False
    margin = revenue - (openai_cost + storage_cost + compute_cost)
    return await discord_notify(
        wh,
        embeds=[{
            "title": "\U0001f4ca Weekly Cost Report",
            "color": 0x3b82f6,
            "fields": [
                {"name": "OpenAI", "value": f"${openai_cost:.2f}", "inline": True},
                {"name": "Storage", "value": f"${storage_cost:.2f}", "inline": True},
                {"name": "Compute", "value": f"${compute_cost:.2f}", "inline": True},
                {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
                {"name": "Est. Margin", "value": f"${margin:.2f}", "inline": True},
            ],
        }],
    )


async def send_email(to: str, subject: str, html: str) -> bool:
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
        logger.info("send_email skipped: Mailgun not configured")
        return False
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={
                    "from": MAIL_FROM,
                    "to": to,
                    "subject": subject,
                    "html": html,
                },
            )
            if r.status_code == 200:
                return True
            body = (r.text or "")[:500]
            logger.warning("send_email failed: HTTP %s body=%s", r.status_code, body)
            return False
    except Exception as e:
        logger.warning("Email failed: %s", e)
        return False
