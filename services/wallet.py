from __future__ import annotations

import json
from typing import Optional

from stages.entitlements import wallet_bypass_for_user_record


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
        urow = await conn.fetchrow(
            "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
            user_id,
        )
        if wallet_bypass_for_user_record(dict(urow) if urow else None):
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
        urow = await conn.fetchrow(
            "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
            user_id,
        )
        if wallet_bypass_for_user_record(dict(urow) if urow else None):
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
        urow = await conn.fetchrow(
            "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
            user_id,
        )
        if wallet_bypass_for_user_record(dict(urow) if urow else None):
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

    urow = await conn.fetchrow(
        "SELECT subscription_tier, role, flex_enabled FROM users WHERE id = $1",
        user_id,
    )
    if wallet_bypass_for_user_record(dict(urow) if urow else None):
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
        amount = int(amount)
        if amount <= 0:
            return False
        burn = int(amount * burn_pct)
        net = amount - burn
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


async def daily_refill(conn, user_id: str, tier: str, wallet: dict | None = None):
    """Delegate to ``core.wallet.daily_refill`` (live GET-user hot path)."""
    from core.wallet import daily_refill as _core_daily_refill

    return await _core_daily_refill(conn, user_id, tier, wallet=wallet)


async def capture_hold_tokens(
    conn,
    upload_id: str,
    user_id: str,
    put_cost: int,
    aic_cost: int,
    meta: Optional[dict] = None,
) -> bool:
    """
    Confirm a hold: reserved → spent. Idempotent — only one winner per upload.

    Returns True if this call captured (or hold already captured), False if
    reserved balance was insufficient / hold missing.
    """
    put_cost = int(put_cost or 0)
    aic_cost = int(aic_cost or 0)
    async with conn.transaction():
        hold = await conn.fetchrow(
            """
            SELECT status FROM wallet_holds
            WHERE upload_id = $1
            ORDER BY created_at DESC NULLS LAST
            LIMIT 1
            """,
            upload_id,
        )
        upload_hold = await conn.fetchval(
            "SELECT hold_status FROM uploads WHERE id = $1",
            upload_id,
        )
        if (hold and str(hold.get("status") or "") == "captured") or str(upload_hold or "") == "captured":
            return True

        claimed = await conn.fetchval(
            """
            UPDATE wallet_holds
            SET status = 'capturing', resolved_at = NULL
            WHERE upload_id = $1 AND status = 'held'
            RETURNING id
            """,
            upload_id,
        )
        # If no wallet_holds row, still allow capture via uploads.hold_status claim
        if not claimed:
            u_claimed = await conn.fetchval(
                """
                UPDATE uploads
                SET hold_status = 'capturing'
                WHERE id = $1 AND COALESCE(hold_status, 'held') IN ('held', 'reserved', '')
                RETURNING id
                """,
                upload_id,
            )
            if not u_claimed and str(upload_hold or "") not in ("held", "reserved", "", "None"):
                return str(upload_hold or "") == "captured"

        row = await conn.fetchrow(
            """
            UPDATE wallets SET
                put_balance  = put_balance  - $1,
                aic_balance  = aic_balance  - $2,
                put_reserved = put_reserved - $1,
                aic_reserved = aic_reserved - $2,
                updated_at   = NOW()
            WHERE user_id = $3
              AND put_reserved >= $1
              AND aic_reserved >= $2
            RETURNING user_id
            """,
            put_cost,
            aic_cost,
            user_id,
        )
        if not row:
            # Roll claim back to held so a later retry can capture
            await conn.execute(
                """
                UPDATE wallet_holds SET status = 'held', resolved_at = NULL
                WHERE upload_id = $1 AND status = 'capturing'
                """,
                upload_id,
            )
            await conn.execute(
                """
                UPDATE uploads SET hold_status = 'held'
                WHERE id = $1 AND hold_status = 'capturing'
                """,
                upload_id,
            )
            return False

        meta_obj = meta or {}
        if put_cost > 0:
            await ledger_entry(
                conn, user_id, "put", -put_cost, "upload_debit", upload_id, meta=meta_obj
            )
        if aic_cost > 0:
            await ledger_entry(
                conn, user_id, "aic", -aic_cost, "upload_debit", upload_id, meta=meta_obj
            )
        await conn.execute(
            """
            UPDATE wallet_holds SET status = 'captured', resolved_at = NOW()
            WHERE upload_id = $1 AND status IN ('held', 'capturing')
            """,
            upload_id,
        )
        await conn.execute(
            "UPDATE uploads SET hold_status = 'captured' WHERE id = $1",
            upload_id,
        )
        return True


async def release_hold_tokens(
    conn,
    upload_id: str,
    user_id: str,
    put_cost: int,
    aic_cost: int,
    reason: str = "release",
) -> bool:
    """
    Release a hold without spending. Single-shot — no phantom ledger after capture.
    """
    put_cost = int(put_cost or 0)
    aic_cost = int(aic_cost or 0)
    async with conn.transaction():
        hold_status = await conn.fetchval(
            """
            SELECT status FROM wallet_holds
            WHERE upload_id = $1 AND status = 'held'
            LIMIT 1
            """,
            upload_id,
        )
        upload_hold = await conn.fetchval(
            "SELECT hold_status FROM uploads WHERE id = $1",
            upload_id,
        )
        if str(upload_hold or "") in ("captured", "released"):
            return False
        if hold_status is None and str(upload_hold or "") not in ("held", "reserved", "", "None"):
            return False

        claimed = await conn.fetchval(
            """
            UPDATE wallet_holds
            SET status = 'releasing'
            WHERE upload_id = $1 AND status = 'held'
            RETURNING id
            """,
            upload_id,
        )
        if not claimed:
            u_claimed = await conn.fetchval(
                """
                UPDATE uploads SET hold_status = 'releasing'
                WHERE id = $1 AND COALESCE(hold_status, 'held') IN ('held', 'reserved', '')
                RETURNING id
                """,
                upload_id,
            )
            if not u_claimed:
                return False

        await conn.execute(
            """
            UPDATE wallets SET
                put_reserved = GREATEST(0, put_reserved - $1),
                aic_reserved = GREATEST(0, aic_reserved - $2),
                updated_at   = NOW()
            WHERE user_id = $3
            """,
            put_cost,
            aic_cost,
            user_id,
        )
        # No positive balance ledger — reserved was never spent; release is not a credit.
        await conn.execute(
            """
            UPDATE wallet_holds SET status = 'released', resolved_at = NOW()
            WHERE upload_id = $1 AND status IN ('held', 'releasing')
            """,
            upload_id,
        )
        await conn.execute(
            "UPDATE uploads SET hold_status = 'released' WHERE id = $1",
            upload_id,
        )
        return True


async def partial_refund_idempotent(
    conn,
    user_id: str,
    upload_id: str,
    succeeded_platforms: list,
    failed_platforms: list,
    original_put_cost: int,
    original_aic_cost: int = 0,
) -> bool:
    """Credit partial refund once per upload (guards on existing ledger reason)."""
    existing = await conn.fetchval(
        """
        SELECT 1 FROM token_ledger
        WHERE upload_id = $1 AND reason = 'partial_platform_refund'
        LIMIT 1
        """,
        upload_id,
    )
    if existing:
        return False
    await partial_refund_upload_partial_success(
        conn,
        user_id,
        upload_id,
        succeeded_platforms,
        failed_platforms,
        original_put_cost,
        original_aic_cost,
    )
    return True
