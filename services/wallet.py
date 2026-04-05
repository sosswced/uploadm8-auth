from __future__ import annotations

import json
import calendar
import math
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
    async with conn.transaction():
        urow = await conn.fetchrow("SELECT subscription_tier FROM users WHERE id = $1", user_id)
        tier = (urow.get("subscription_tier") if urow else "free") or "free"
        ent = get_entitlements_for_tier(str(tier))
        # Internal tiers (master_admin/friends_family/lifetime) are unlimited.
        if getattr(ent, "is_internal", False):
            return True
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
        # Ledger: single debit on capture (upload_debit), not here — avoids double lines for one spend.
        return True


async def spend_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str, platforms: Optional[list] = None):
    async with conn.transaction():
        urow = await conn.fetchrow("SELECT subscription_tier FROM users WHERE id = $1", user_id)
        tier = (urow.get("subscription_tier") if urow else "free") or "free"
        ent = get_entitlements_for_tier(str(tier))
        if getattr(ent, "is_internal", False):
            return
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
    async with conn.transaction():
        urow = await conn.fetchrow("SELECT subscription_tier FROM users WHERE id = $1", user_id)
        tier = (urow.get("subscription_tier") if urow else "free") or "free"
        ent = get_entitlements_for_tier(str(tier))
        if getattr(ent, "is_internal", False):
            return
        await conn.execute(
            "UPDATE wallets SET put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3",
            put_count,
            aic_count,
            user_id,
        )
        # No ledger row: balance was never debited; release only clears the hold.


async def partial_refund_upload_partial_success(
    conn,
    user_id: str,
    upload_id: str,
    succeeded_platforms: list,
    failed_platforms: list,
    original_put_cost: int,
    original_aic_cost: int,
) -> None:
    """
    After capture, credit back PUT for failed publish slots (same rule as before)
    and AIC in proportion to failed_targets / total_targets (AI work is mostly
    per-job, but this aligns charges with partial delivery).
    """
    n_failed = len(failed_platforms or [])
    n_ok = len(succeeded_platforms or [])
    if n_failed == 0 or n_ok == 0:
        return

    put_refund = min(n_failed * 2, max(0, int(original_put_cost or 0) - 10))

    n_total = n_ok + n_failed
    aic_refund = int((int(original_aic_cost or 0) * n_failed) // max(1, n_total))
    aic_refund = min(aic_refund, int(original_aic_cost or 0))

    if put_refund <= 0 and aic_refund <= 0:
        return

    urow = await conn.fetchrow("SELECT subscription_tier FROM users WHERE id = $1", user_id)
    tier = (urow.get("subscription_tier") if urow else "free") or "free"
    ent = get_entitlements_for_tier(str(tier))
    if getattr(ent, "is_internal", False):
        return

    await conn.execute(
        """
        UPDATE wallets SET
            put_balance = put_balance + $1,
            aic_balance = aic_balance + $2,
            updated_at = NOW()
        WHERE user_id = $3
        """,
        put_refund,
        aic_refund,
        user_id,
    )
    if put_refund > 0:
        await ledger_entry(conn, user_id, "put", put_refund, "partial_platform_refund", upload_id)
    if aic_refund > 0:
        await ledger_entry(conn, user_id, "aic", aic_refund, "partial_platform_refund", upload_id)


async def credit_wallet(conn, user_id: str, wallet_type: str, amount: int, reason: str, stripe_event_id: Optional[str] = None):
    async with conn.transaction():
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
    async with conn.transaction():
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
    # Avoid BEGIN/COMMIT for paid/internal tiers - hot path on every authenticated request.
    ent = get_entitlements_for_tier(tier)
    if getattr(ent, "is_internal", False) or str(getattr(ent, "tier", "")) != "free":
        return

    async with conn.transaction():
        wallet = await get_wallet(conn, user_id)
        last_refill = wallet.get("last_refill_date")
        today = _now_utc().date()
        if last_refill and last_refill >= today:
            return

        # Split monthly entitlements over calendar days in current UTC month.
        # Cap by subscription budget for the month (put_drip_granted), not by wallet
        # balance — rollover/top-ups can push balance above the monthly allowance.
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        put_daily = max(0, int(math.ceil((ent.put_monthly or 0) / max(1, days_in_month))))
        aic_daily = max(0, int(math.ceil((ent.aic_monthly or 0) / max(1, days_in_month))))

        put_cap = int(ent.put_monthly or 0)
        aic_cap = int(ent.aic_monthly or 0)
        month_key = f"{today.year}-{today.month:02d}"

        drip_month = wallet.get("subscription_drip_month")
        put_g = int(wallet.get("put_drip_granted") or 0)
        aic_g = int(wallet.get("aic_drip_granted") or 0)
        if (drip_month or "") != month_key:
            put_g = 0
            aic_g = 0

        put_remaining = max(0, put_cap - put_g)
        aic_remaining = max(0, aic_cap - aic_g)

        put_add = min(put_daily, put_remaining)
        aic_add = min(aic_daily, aic_remaining)
        new_put_g = put_g + put_add
        new_aic_g = aic_g + aic_add

        if put_add <= 0 and aic_add <= 0:
            await conn.execute(
                "UPDATE wallets SET last_refill_date = $1, subscription_drip_month = $2, put_drip_granted = $3, aic_drip_granted = $4 WHERE user_id = $5",
                today,
                month_key,
                new_put_g,
                new_aic_g,
                user_id,
            )
            return

        await conn.execute(
            """
            UPDATE wallets
            SET
                put_balance = put_balance + $1,
                aic_balance = aic_balance + $2,
                last_refill_date = $3,
                subscription_drip_month = $4,
                put_drip_granted = $5,
                aic_drip_granted = $6
            WHERE user_id = $7
            """,
            put_add,
            aic_add,
            today,
            month_key,
            new_put_g,
            new_aic_g,
            user_id,
        )
        if put_add > 0:
            await ledger_entry(conn, user_id, "put", put_add, "daily_refill")
        if aic_add > 0:
            await ledger_entry(conn, user_id, "aic", aic_add, "daily_refill")
