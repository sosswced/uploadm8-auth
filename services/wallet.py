from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from stages.entitlements import get_entitlements_for_tier


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def get_wallet(conn, user_id: str) -> dict:
    row = await conn.fetchrow("SELECT * FROM wallets WHERE user_id = $1", user_id)
    if not row:
        await conn.execute("INSERT INTO wallets (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
        row = await conn.fetchrow("SELECT * FROM wallets WHERE user_id = $1", user_id)
    return dict(row) if row else {"put_balance": 0, "aic_balance": 0, "put_reserved": 0, "aic_reserved": 0}


async def ledger_entry(
    conn,
    user_id: str,
    token_type: str,
    delta: int,
    reason: str,
    upload_id: Optional[str] = None,
    stripe_event_id: Optional[str] = None,
    platform: Optional[str] = None,
    meta: Optional[dict] = None,
):
    await conn.execute(
        """
        INSERT INTO token_ledger (user_id, token_type, platform, delta, reason, upload_id, stripe_event_id, meta)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """,
        user_id,
        token_type,
        platform,
        delta,
        reason,
        upload_id,
        stripe_event_id,
        json.dumps(meta) if meta else None,
    )


async def reserve_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str) -> bool:
    wallet = await get_wallet(conn, user_id)
    available_put = wallet["put_balance"] - wallet["put_reserved"]
    available_aic = wallet["aic_balance"] - wallet["aic_reserved"]
    if available_put < put_count or available_aic < aic_count:
        return False
    await conn.execute(
        "UPDATE wallets SET put_reserved = put_reserved + $1, aic_reserved = aic_reserved + $2 WHERE user_id = $3",
        put_count,
        aic_count,
        user_id,
    )
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "reserve", upload_id)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "reserve", upload_id)
    return True


async def spend_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str, platforms: Optional[list] = None):
    await conn.execute(
        "UPDATE wallets SET put_balance = put_balance - $1, aic_balance = aic_balance - $2, put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3",
        put_count,
        aic_count,
        user_id,
    )
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "spend", upload_id, platform=",".join(platforms) if platforms else None)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "spend", upload_id)


async def refund_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str):
    await conn.execute(
        "UPDATE wallets SET put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3",
        put_count,
        aic_count,
        user_id,
    )
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", put_count, "refund", upload_id)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", aic_count, "refund", upload_id)


async def credit_wallet(conn, user_id: str, wallet_type: str, amount: int, reason: str, stripe_event_id: Optional[str] = None):
    if wallet_type == "put":
        await conn.execute("UPDATE wallets SET put_balance = put_balance + $1 WHERE user_id = $2", amount, user_id)
    else:
        await conn.execute("UPDATE wallets SET aic_balance = aic_balance + $1 WHERE user_id = $2", amount, user_id)
    await ledger_entry(conn, user_id, wallet_type, amount, reason, stripe_event_id=stripe_event_id)


async def transfer_tokens(
    conn,
    user_id: str,
    from_platform: str,
    to_platform: str,
    amount: int,
    burn_pct: float = 0.02,
) -> bool:
    user = await conn.fetchrow("SELECT subscription_tier, flex_enabled FROM users WHERE id = $1", user_id)
    if not user or not user.get("flex_enabled"):
        return False
    wallet = await get_wallet(conn, user_id)
    if wallet["put_balance"] - wallet["put_reserved"] < amount:
        return False
    burn = int(amount * burn_pct)
    net = amount - burn
    await ledger_entry(conn, user_id, "put", -amount, "transfer_out", platform=from_platform)
    await ledger_entry(conn, user_id, "put", net, "transfer_in", platform=to_platform)
    if burn > 0:
        await ledger_entry(conn, user_id, "put", -burn, "transfer_burn")
        await conn.execute("UPDATE wallets SET put_balance = put_balance - $1 WHERE user_id = $2", burn, user_id)
    return True


async def daily_refill(conn, user_id: str, tier: str):
    ent = get_entitlements_for_tier(tier)
    # Add one day's allowance once per calendar day; never exceed monthly cap.
    daily = ent.put_daily
    wallet = await get_wallet(conn, user_id)
    last_refill = wallet.get("last_refill_date")
    today = _now_utc().date()
    if last_refill and last_refill >= today:
        return
    monthly_cap = ent.put_monthly
    current = wallet["put_balance"]
    if current < monthly_cap:
        add = min(daily, monthly_cap - current)
        await conn.execute(
            "UPDATE wallets SET put_balance = put_balance + $1, last_refill_date = $2 WHERE user_id = $3",
            add,
            today,
            user_id,
        )
        await ledger_entry(conn, user_id, "put", add, "daily_refill")
