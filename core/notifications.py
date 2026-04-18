"""
UploadM8 notification helpers — extracted from app.py.
Discord webhooks and email sending via Mailgun.
"""

import logging

import httpx

from core.config import (
    ADMIN_DISCORD_WEBHOOK_URL,
    SIGNUP_DISCORD_WEBHOOK_URL,
    MRR_DISCORD_WEBHOOK_URL,
    MAILGUN_API_KEY,
    MAILGUN_DOMAIN,
    MAIL_FROM,
)

logger = logging.getLogger("uploadm8-api")


async def discord_notify(webhook_url: str, content: str = None, embeds: list = None):
    if not webhook_url: return
    payload = {}
    if content: payload["content"] = content
    if embeds: payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(webhook_url, json=payload)
    except Exception: pass

async def notify_signup(email: str, name: str):
    wh = SIGNUP_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        await discord_notify(wh, embeds=[{"title": "\U0001f389 New Signup", "color": 0x22c55e, "fields": [{"name": "Email", "value": email}, {"name": "Name", "value": name}]}])

async def notify_mrr(amount: float, email: str, plan: str, event_type: str = "charge"):
    wh = MRR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        await discord_notify(wh, embeds=[{"title": f"\U0001f4b0 {event_type.title()}", "color": 0x22c55e, "fields": [{"name": "Amount", "value": f"${amount:.2f}"}, {"name": "Email", "value": email}, {"name": "Plan", "value": plan}]}])

async def notify_topup(amount: float, email: str, wallet: str, tokens: int):
    wh = MRR_DISCORD_WEBHOOK_URL or ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        await discord_notify(wh, embeds=[{"title": "\U0001f4b3 Top-up Purchase", "color": 0x8b5cf6, "fields": [{"name": "Amount", "value": f"${amount:.2f}"}, {"name": "Wallet", "value": wallet.upper()}, {"name": "Tokens", "value": str(tokens)}, {"name": "Email", "value": email}]}])

async def notify_weekly_costs(openai_cost: float, storage_cost: float, compute_cost: float, revenue: float):
    wh = ADMIN_DISCORD_WEBHOOK_URL
    if wh:
        margin = revenue - (openai_cost + storage_cost + compute_cost)
        await discord_notify(wh, embeds=[{"title": "\U0001f4ca Weekly Cost Report", "color": 0x3b82f6, "fields": [
            {"name": "OpenAI", "value": f"${openai_cost:.2f}", "inline": True},
            {"name": "Storage", "value": f"${storage_cost:.2f}", "inline": True},
            {"name": "Compute", "value": f"${compute_cost:.2f}", "inline": True},
            {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
            {"name": "Est. Margin", "value": f"${margin:.2f}", "inline": True},
        ]}])

async def send_email(to: str, subject: str, html: str):
    if not MAILGUN_API_KEY or not MAILGUN_DOMAIN: return
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages", auth=("api", MAILGUN_API_KEY), data={"from": MAIL_FROM, "to": to, "subject": subject, "html": html})
    except Exception as e:
        logger.warning(f"Email failed: {e}")
