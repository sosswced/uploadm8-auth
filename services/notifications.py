from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger("uploadm8-api")


async def discord_notify(webhook_url: str, content: str = None, embeds: list = None):
    if not webhook_url:
        return
    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(webhook_url, json=payload)
            if r.status_code not in (200, 204):
                logger.warning("Discord notify failed: status=%s", r.status_code)
    except Exception as e:
        logger.warning("Discord notify error: %s", e)


async def notify_signup(email: str, name: str, signup_webhook: Optional[str], admin_webhook: Optional[str]):
    wh = signup_webhook or admin_webhook
    if wh:
        await discord_notify(
            wh,
            embeds=[
                {
                    "title": " New Signup",
                    "color": 0x22C55E,
                    "fields": [{"name": "Email", "value": email}, {"name": "Name", "value": name}],
                }
            ],
        )


async def notify_mrr(amount: float, email: str, plan: str, event_type: str, mrr_webhook: Optional[str], admin_webhook: Optional[str]):
    wh = mrr_webhook or admin_webhook
    if wh:
        await discord_notify(
            wh,
            embeds=[
                {
                    "title": f" {event_type.title()}",
                    "color": 0x22C55E,
                    "fields": [
                        {"name": "Amount", "value": f"${amount:.2f}"},
                        {"name": "Email", "value": email},
                        {"name": "Plan", "value": plan},
                    ],
                }
            ],
        )


async def notify_topup(amount: float, email: str, wallet: str, tokens: int, mrr_webhook: Optional[str], admin_webhook: Optional[str]):
    wh = mrr_webhook or admin_webhook
    if wh:
        await discord_notify(
            wh,
            embeds=[
                {
                    "title": " Top-up Purchase",
                    "color": 0x8B5CF6,
                    "fields": [
                        {"name": "Amount", "value": f"${amount:.2f}"},
                        {"name": "Wallet", "value": wallet.upper()},
                        {"name": "Tokens", "value": str(tokens)},
                        {"name": "Email", "value": email},
                    ],
                }
            ],
        )


async def notify_weekly_costs(
    openai_cost: float,
    storage_cost: float,
    compute_cost: float,
    revenue: float,
    admin_webhook: Optional[str],
):
    if not admin_webhook:
        return
    margin = revenue - (openai_cost + storage_cost + compute_cost)
    await discord_notify(
        admin_webhook,
        embeds=[
            {
                "title": " Weekly Cost Report",
                "color": 0x3B82F6,
                "fields": [
                    {"name": "OpenAI", "value": f"${openai_cost:.2f}", "inline": True},
                    {"name": "Storage", "value": f"${storage_cost:.2f}", "inline": True},
                    {"name": "Compute", "value": f"${compute_cost:.2f}", "inline": True},
                    {"name": "Revenue", "value": f"${revenue:.2f}", "inline": True},
                    {"name": "Est. Margin", "value": f"${margin:.2f}", "inline": True},
                ],
            }
        ],
    )
