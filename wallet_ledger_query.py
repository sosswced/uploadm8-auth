"""Paginated token_ledger reads + period summary for GET /api/wallet/ledger."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from stages.entitlements import ledger_pricing_reference
from services.billing_service_weights import fetch_service_weights_map
from services.wallet_marketing import _ledger_sums, get_wallet_usage_period_anchor

logger = logging.getLogger(__name__)


def _parse_uuid(s: Optional[str]) -> Optional[UUID]:
    if not s:
        return None
    try:
        return UUID(str(s).strip())
    except (ValueError, TypeError):
        return None


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except (ValueError, TypeError):
        return None


def _row_to_ledger_item(row: Any) -> Dict[str, Any]:
    d = dict(row)
    for k in ("created_at",):
        v = d.get(k)
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    meta = d.get("meta")
    if isinstance(meta, str):
        d["meta"] = meta
    return d


async def fetch_ledger_period_summary(
    conn: Any,
    user_id: str,
    period_start: datetime,
) -> Dict[str, Any]:
    """Marketing-aligned sums plus coarse debit/credit/refund buckets for the UI."""
    base = await _ledger_sums(conn, user_id, period_start)
    ext = await conn.fetchrow(
        """
        SELECT
          COALESCE(SUM(CASE WHEN token_type = 'put' AND delta < 0
            THEN ABS(delta::bigint) ELSE 0 END), 0)::bigint AS put_all_debits,
          COALESCE(SUM(CASE WHEN token_type = 'aic' AND delta < 0
            THEN ABS(delta::bigint) ELSE 0 END), 0)::bigint AS aic_all_debits,
          COALESCE(SUM(CASE WHEN token_type = 'put' AND delta > 0
            THEN delta::bigint ELSE 0 END), 0)::bigint AS put_all_credits,
          COALESCE(SUM(CASE WHEN token_type = 'aic' AND delta > 0
            THEN delta::bigint ELSE 0 END), 0)::bigint AS aic_all_credits,
          COALESCE(SUM(CASE WHEN token_type = 'put' AND delta > 0
            AND (reason ILIKE '%refund%' OR reason IN ('release', 'partial_platform_refund'))
            THEN delta::bigint ELSE 0 END), 0)::bigint AS put_refund_like,
          COALESCE(SUM(CASE WHEN token_type = 'aic' AND delta > 0
            AND (reason ILIKE '%refund%' OR reason IN ('release', 'partial_platform_refund'))
            THEN delta::bigint ELSE 0 END), 0)::bigint AS aic_refund_like
        FROM token_ledger
        WHERE user_id = $1::uuid AND created_at >= $2
        """,
        user_id,
        period_start,
    )
    out = dict(base)
    if ext:
        for k in ext.keys():
            try:
                out[k] = int(ext[k] or 0)
            except (TypeError, ValueError):
                out[k] = 0
    return out


async def fetch_ledger_page(
    conn: Any,
    user_id: str,
    *,
    limit: int,
    token_type: Optional[str] = None,
    reason_prefix: Optional[str] = None,
    upload_id: Optional[UUID] = None,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
    cursor_created_at: Optional[datetime] = None,
    cursor_id: Optional[UUID] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Keyset pagination (newest first): pass cursor from last row's created_at + id.
    """
    lim = max(1, min(int(limit or 40), 100))
    where = ["l.user_id = $1::uuid"]
    args: List[Any] = [user_id]
    p = 2

    if token_type in ("put", "aic"):
        where.append(f"l.token_type = ${p}")
        args.append(token_type)
        p += 1
    if reason_prefix:
        where.append(f"l.reason ILIKE ${p}")
        args.append(reason_prefix.rstrip("%") + "%")
        p += 1
    if upload_id:
        where.append(f"l.upload_id = ${p}")
        args.append(upload_id)
        p += 1
    if from_ts:
        where.append(f"l.created_at >= ${p}")
        args.append(from_ts)
        p += 1
    if to_ts:
        where.append(f"l.created_at <= ${p}")
        args.append(to_ts)
        p += 1
    if cursor_created_at and cursor_id:
        where.append(
            f"(l.created_at, l.id) < (${p}::timestamptz, ${p + 1}::uuid)"
        )
        args.extend([cursor_created_at, cursor_id])
        p += 2

    where_sql = " AND ".join(where)
    q = f"""
        SELECT
            l.id, l.user_id, l.token_type, l.platform, l.delta, l.reason,
            l.upload_id, l.stripe_event_id, l.ref_type, l.meta, l.created_at,
            u.filename AS upload_filename,
            u.title AS upload_title,
            u.status AS upload_status,
            u.compute_seconds AS upload_compute_seconds
        FROM token_ledger l
        LEFT JOIN uploads u ON u.id = l.upload_id
        WHERE {where_sql}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT {lim + 1}
    """
    rows = await conn.fetch(q, *args)
    has_more = len(rows) > lim
    page = rows[:lim]
    items = [_row_to_ledger_item(r) for r in page]
    return items, has_more


async def build_wallet_ledger_payload(
    pool: Any,
    user_id: str,
    *,
    limit: int,
    token_type: Optional[str],
    reason_prefix: Optional[str],
    upload_id_raw: Optional[str],
    from_raw: Optional[str],
    to_raw: Optional[str],
    cursor_at_raw: Optional[str],
    cursor_id_raw: Optional[str],
) -> Dict[str, Any]:
    async with pool.acquire() as conn:
        period_start = await get_wallet_usage_period_anchor(conn, user_id)
        summary = await fetch_ledger_period_summary(conn, user_id, period_start)
        upload_id = _parse_uuid(upload_id_raw)
        from_ts = _parse_dt(from_raw)
        to_ts = _parse_dt(to_raw)
        cursor_at = _parse_dt(cursor_at_raw)
        cursor_id = _parse_uuid(cursor_id_raw)
        if (cursor_at is None) != (cursor_id is None):
            raise ValueError("cursor_at and cursor_id must be sent together")
        items, has_more = await fetch_ledger_page(
            conn,
            user_id,
            limit=limit,
            token_type=token_type,
            reason_prefix=reason_prefix,
            upload_id=upload_id,
            from_ts=from_ts,
            to_ts=to_ts,
            cursor_created_at=cursor_at,
            cursor_id=cursor_id,
        )
        dbw = await fetch_service_weights_map(conn)
        pricing_reference = ledger_pricing_reference(service_weights=dbw)
    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = {
            "created_at": last.get("created_at"),
            "id": str(last.get("id")) if last.get("id") else None,
        }
    ps = period_start
    period_iso = ps.isoformat() if hasattr(ps, "isoformat") else None
    return {
        "period_start": period_iso,
        "summary": summary,
        "items": items,
        "has_more": has_more,
        "next_cursor": next_cursor,
        "pricing_reference": pricing_reference,
    }
