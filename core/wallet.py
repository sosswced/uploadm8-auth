"""
UploadM8 wallet & ledger functions — extracted from app.py.
Token balances, reservations, spending, refunds, daily refills.
"""

import calendar
import json
import logging
import math
from datetime import date
from typing import Any

from core.helpers import _now_utc
from stages.entitlements import get_entitlements_for_tier, wallet_bypass_for_user_record

logger = logging.getLogger("uploadm8-api")

# ============================================================
# Wallet & Ledger Functions
# ============================================================
async def get_wallet(conn, user_id: str) -> dict:
    row = await conn.fetchrow("SELECT * FROM wallets WHERE user_id = $1", user_id)
    if not row:
        await conn.execute("INSERT INTO wallets (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id)
        row = await conn.fetchrow("SELECT * FROM wallets WHERE user_id = $1", user_id)
    return dict(row) if row else {"put_balance": 0, "aic_balance": 0, "put_reserved": 0, "aic_reserved": 0}

async def ledger_entry(conn, user_id: str, token_type: str, delta: int, reason: str, upload_id: str = None, stripe_event_id: str = None, platform: str = None, meta: dict = None):
    await conn.execute("""
        INSERT INTO token_ledger (user_id, token_type, platform, delta, reason, upload_id, stripe_event_id, meta)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """, user_id, token_type, platform, delta, reason, upload_id, stripe_event_id, json.dumps(meta, default=str) if meta else None)

def _wallet_bypass_tokens(user_record: dict | None) -> bool:
    """PUT/AIC wallet bypass — see ``wallet_bypass_for_user_record``."""
    return wallet_bypass_for_user_record(user_record)


async def reserve_tokens(
    conn,
    user_id: str,
    put_count: int,
    aic_count: int,
    upload_id: str,
    *,
    ledger_meta: dict | None = None,
) -> bool:
    urow = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user_id,
    )
    if _wallet_bypass_tokens(dict(urow) if urow else None):
        return True
    wallet = await get_wallet(conn, user_id)
    available_put = wallet["put_balance"] - wallet["put_reserved"]
    available_aic = wallet["aic_balance"] - wallet["aic_reserved"]
    if available_put < put_count or available_aic < aic_count:
        return False
    await conn.execute("UPDATE wallets SET put_reserved = put_reserved + $1, aic_reserved = aic_reserved + $2 WHERE user_id = $3", put_count, aic_count, user_id)
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "reserve", upload_id, meta=ledger_meta)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "reserve", upload_id, meta=ledger_meta)
    return True

async def atomic_reserve_tokens(
    conn,
    user_id: str,
    put_count: int,
    aic_count: int,
    upload_id: str,
    *,
    ledger_meta: dict | None = None,
) -> bool:
    """Atomically check-and-reserve tokens in a single UPDATE.
    Returns True if reserved, False if insufficient balance.
    Prevents race conditions where concurrent presign calls share the same balance."""
    urow = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user_id,
    )
    if _wallet_bypass_tokens(dict(urow) if urow else None):
        return True
    row = await conn.fetchrow(
        """UPDATE wallets
           SET put_reserved = put_reserved + $1,
               aic_reserved = aic_reserved + $2
           WHERE user_id = $3
             AND (put_balance - put_reserved) >= $1
             AND (aic_balance - aic_reserved) >= $2
           RETURNING put_balance, put_reserved, aic_balance, aic_reserved""",
        put_count, aic_count, user_id
    )
    if row is None:
        return False
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "reserve", upload_id, meta=ledger_meta)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "reserve", upload_id, meta=ledger_meta)
    return True

async def atomic_debit_tokens(
    conn,
    user_id: str,
    put_count: int,
    aic_count: int,
    ref_id: str,
    *,
    reason: str = "debit",
) -> bool:
    """Debit PUT/AIC from available balance in one UPDATE (respects reserved amounts).

    Internal tiers and admin roles bypass the wallet (same rule as ``atomic_reserve_tokens``)
    so Thumbnail Studio and other direct debits do not 429 while the UI shows unlimited tokens.
    """
    put_count = int(put_count or 0)
    aic_count = int(aic_count or 0)
    if put_count <= 0 and aic_count <= 0:
        return True
    urow = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user_id,
    )
    if _wallet_bypass_tokens(dict(urow) if urow else None):
        return True
    row = await conn.fetchrow(
        """UPDATE wallets
           SET put_balance = put_balance - $1,
               aic_balance = aic_balance - $2,
               updated_at = NOW()
           WHERE user_id = $3
             AND (put_balance - put_reserved) >= $1
             AND (aic_balance - aic_reserved) >= $2
           RETURNING put_balance, aic_balance""",
        put_count,
        aic_count,
        user_id,
    )
    if row is None:
        return False
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, reason, ref_id)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, reason, ref_id)
    return True


async def spend_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str, platforms: list = None):
    await conn.execute("UPDATE wallets SET put_balance = put_balance - $1, aic_balance = aic_balance - $2, put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3", put_count, aic_count, user_id)
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", -put_count, "spend", upload_id, platform=",".join(platforms) if platforms else None)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", -aic_count, "spend", upload_id)

async def refund_tokens(conn, user_id: str, put_count: int, aic_count: int, upload_id: str):
    urow = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user_id,
    )
    if _wallet_bypass_tokens(dict(urow) if urow else None):
        return
    await conn.execute("UPDATE wallets SET put_reserved = put_reserved - $1, aic_reserved = aic_reserved - $2 WHERE user_id = $3", put_count, aic_count, user_id)
    if put_count > 0:
        await ledger_entry(conn, user_id, "put", put_count, "refund", upload_id)
    if aic_count > 0:
        await ledger_entry(conn, user_id, "aic", aic_count, "refund", upload_id)

async def credit_wallet(conn, user_id: str, wallet_type: str, amount: int, reason: str, stripe_event_id: str = None):
    if wallet_type == "put":
        await conn.execute("UPDATE wallets SET put_balance = put_balance + $1 WHERE user_id = $2", amount, user_id)
    else:
        await conn.execute("UPDATE wallets SET aic_balance = aic_balance + $1 WHERE user_id = $2", amount, user_id)
    await ledger_entry(conn, user_id, wallet_type, amount, reason, stripe_event_id=stripe_event_id)

async def transfer_tokens(conn, user_id: str, from_platform: str, to_platform: str, amount: int, burn_pct: float = 0.02) -> bool:
    # Check flex enabled
    user = await conn.fetchrow("SELECT subscription_tier, flex_enabled FROM users WHERE id = $1", user_id)
    if not user or not user.get("flex_enabled"):
        return False
    amount = int(amount)
    if amount <= 0:
        return False
    burn = int(amount * burn_pct)
    net = amount - burn
    # Single conditional UPDATE: require full available PUT for the transfer size, then apply burn atomically.
    if burn > 0:
        row = await conn.fetchrow(
            """
            UPDATE wallets
            SET put_balance = put_balance - $1,
                updated_at = NOW()
            WHERE user_id = $2
              AND (put_balance - put_reserved) >= $3
            RETURNING put_balance
            """,
            burn,
            user_id,
            amount,
        )
    else:
        row = await conn.fetchrow(
            """
            UPDATE wallets
            SET updated_at = NOW()
            WHERE user_id = $1
              AND (put_balance - put_reserved) >= $2
            RETURNING put_balance
            """,
            user_id,
            amount,
        )
    if row is None:
        return False
    await ledger_entry(conn, user_id, "put", -amount, "transfer_out", platform=from_platform)
    await ledger_entry(conn, user_id, "put", net, "transfer_in", platform=to_platform)
    if burn > 0:
        await ledger_entry(conn, user_id, "put", -burn, "transfer_burn")
    return True

def compute_free_daily_drip(
    *,
    put_monthly: int,
    aic_monthly: int,
    today: date,
    drip_month: str | None,
    put_drip_granted: int,
    aic_drip_granted: int,
) -> dict[str, Any]:
    """Split monthly free-tier PUT/AIC over calendar days (matches GET /api/wallet daily_topup)."""
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    put_daily = max(0, int(math.ceil((int(put_monthly or 0)) / max(1, days_in_month))))
    aic_daily = max(0, int(math.ceil((int(aic_monthly or 0)) / max(1, days_in_month))))
    month_key = f"{today.year}-{today.month:02d}"
    put_g = 0 if (drip_month or "") != month_key else int(put_drip_granted or 0)
    aic_g = 0 if (drip_month or "") != month_key else int(aic_drip_granted or 0)
    put_add = min(put_daily, max(0, int(put_monthly or 0) - put_g))
    aic_add = min(aic_daily, max(0, int(aic_monthly or 0) - aic_g))
    return {
        "put_daily": put_daily,
        "aic_daily": aic_daily,
        "put_add": put_add,
        "aic_add": aic_add,
        "month_key": month_key,
        "new_put_granted": put_g + put_add,
        "new_aic_granted": aic_g + aic_add,
    }


async def daily_refill(conn, user_id: str, tier: str, wallet: dict | None = None) -> dict | None:
    """Drip the user's daily token allowance.

    Hot path: called from ``get_current_user`` on every authenticated request.
    Returns the (possibly updated) wallet dict so callers can avoid a second
    ``SELECT * FROM wallets``. Returns ``None`` only when the caller did not
    provide a wallet AND we short-circuited before fetching one (paid/internal
    tiers) — the caller is then expected to fetch the wallet itself if needed.

    Free tier: PUT + AIC split of monthly caps across calendar days (same math as
    ``daily_topup`` on GET /api/wallet). Paid/internal tiers refill on invoice.
    """
    ent = get_entitlements_for_tier(tier)
    if getattr(ent, "is_internal", False) or str(getattr(ent, "tier", "")) != "free":
        return wallet

    if wallet is None:
        wallet = await get_wallet(conn, user_id)

    last_refill = wallet.get("last_refill_date")
    today = _now_utc().date()
    if last_refill and last_refill >= today:
        return wallet

    drip = compute_free_daily_drip(
        put_monthly=int(getattr(ent, "put_monthly", 0) or 0),
        aic_monthly=int(getattr(ent, "aic_monthly", 0) or 0),
        today=today,
        drip_month=wallet.get("subscription_drip_month"),
        put_drip_granted=int(wallet.get("put_drip_granted") or 0),
        aic_drip_granted=int(wallet.get("aic_drip_granted") or 0),
    )
    put_add = int(drip["put_add"])
    aic_add = int(drip["aic_add"])

    row = await conn.fetchrow(
        """
        UPDATE wallets
        SET put_balance = put_balance + $1,
            aic_balance = aic_balance + $2,
            last_refill_date = $3,
            subscription_drip_month = $4,
            put_drip_granted = $5,
            aic_drip_granted = $6
        WHERE user_id = $7
          AND (last_refill_date IS NULL OR last_refill_date < $3)
        RETURNING put_balance, aic_balance, last_refill_date,
                  put_drip_granted, aic_drip_granted, subscription_drip_month
        """,
        put_add,
        aic_add,
        today,
        drip["month_key"],
        drip["new_put_granted"],
        drip["new_aic_granted"],
        user_id,
    )
    if not row:
        return wallet
    if put_add > 0:
        await ledger_entry(conn, user_id, "put", put_add, "daily_refill")
    if aic_add > 0:
        await ledger_entry(conn, user_id, "aic", aic_add, "daily_refill")
    wallet["put_balance"] = int(row["put_balance"])
    wallet["aic_balance"] = int(row["aic_balance"])
    wallet["last_refill_date"] = row["last_refill_date"]
    wallet["put_drip_granted"] = int(row["put_drip_granted"] or 0)
    wallet["aic_drip_granted"] = int(row["aic_drip_granted"] or 0)
    wallet["subscription_drip_month"] = row["subscription_drip_month"]
    return wallet

async def partial_refund_tokens(
    conn,
    user_id: str,
    upload_id: str,
    succeeded_platforms: list,
    failed_platforms: list,
    original_put_cost: int,
    original_aic_cost: int,
):
    """
    Pro-rate a refund for platforms that failed in a partial upload.

    PUT cost formula (mirrors entitlements.py compute_put_cost):
        base = 10  (covers the first platform)
        +2 per *extra* platform beyond the first
        (HUD / priority / thumbnail costs are per-job, not per-platform -- kept)

    For a partial failure we refund:
        2 tokens x number_of_failed_platforms
        (the base-10 is always kept because work was done)

    AIC is per-job, not per-platform -- no partial AIC refund.

    Only runs when there is at least one success AND at least one failure.
    On full failure the worker calls release/unreserve instead.
    """
    urow = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user_id,
    )
    if _wallet_bypass_tokens(dict(urow) if urow else None):
        return
    n_failed = len(failed_platforms or [])
    if n_failed == 0 or not (succeeded_platforms or []):
        return  # nothing to refund

    put_refund = n_failed * 2
    put_refund = min(put_refund, max(0, int(original_put_cost or 0) - 10))  # never refund the base-10

    if put_refund <= 0:
        return

    await conn.execute(
        """
        UPDATE wallets
        SET
            put_balance  = put_balance  + $1,
            put_reserved = GREATEST(0, put_reserved - $1),
            updated_at   = NOW()
        WHERE user_id = $2
        """,
        put_refund,
        user_id,
    )

    await conn.execute(
        """
        INSERT INTO token_ledger
            (user_id, token_type, delta, reason, upload_id, ref_type, meta)
        VALUES
            ($1, 'put', $2, 'partial_platform_refund', $3, 'upload',
             jsonb_build_object(
                 'failed_platforms',    $4::text,
                 'succeeded_platforms', $5::text,
                 'original_put_cost',   $6
             ))
        """,
        user_id,
        put_refund,
        upload_id,
        ",".join([str(x) for x in (failed_platforms or [])]),
        ",".join([str(x) for x in (succeeded_platforms or [])]),
        int(original_put_cost or 0),
    )
