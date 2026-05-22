"""
Compare ``catalog_products.price_usd`` (and yearly for subscriptions) to live Stripe Prices.

Usage:
  python -m scripts.stripe_reconcile_catalog
  python -m scripts.stripe_reconcile_catalog --lookup-key uploadm8_put_500
  python -m scripts.stripe_reconcile_catalog --apply

With ``--apply``, overwrites DB ``price_usd`` / ``price_usd_yearly`` from Stripe when a
Price is found by ``lookup_key`` and the amounts differ. Does not create Stripe objects.

Requires STRIPE_SECRET_KEY and DATABASE_URL.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx

STRIPE_BASE = "https://api.stripe.com/v1"


async def _stripe_get(
    client: httpx.AsyncClient,
    path: str,
    key: str,
    *,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    r = await client.get(
        f"{STRIPE_BASE}/{path}",
        headers={"Authorization": f"Bearer {key}"},
        params=params or None,
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


async def _find_price_by_lookup_key(
    client: httpx.AsyncClient, key: str, stripe_key: str
) -> Optional[Dict[str, Any]]:
    data = await _stripe_get(
        client,
        "prices/search",
        stripe_key,
        params={"query": f"lookup_key:'{key}'"},
    )
    rows = data.get("data") or []
    return rows[0] if rows else None


async def _run(lookup_key: Optional[str], apply: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    dsn = os.environ.get("DATABASE_URL", "")
    if not stripe_key or not dsn:
        print("STRIPE_SECRET_KEY and DATABASE_URL are required", file=sys.stderr)
        return [], []

    conn = await asyncpg.connect(dsn)
    out: List[Dict[str, Any]] = []
    applied: List[Dict[str, Any]] = []
    try:
        if lookup_key:
            rows = await conn.fetch(
                "SELECT lookup_key, product_kind, price_usd, price_usd_yearly "
                "FROM catalog_products WHERE lookup_key = $1 AND is_archived = FALSE",
                lookup_key,
            )
        else:
            rows = await conn.fetch(
                "SELECT lookup_key, product_kind, price_usd, price_usd_yearly "
                "FROM catalog_products WHERE is_archived = FALSE "
                "ORDER BY sort_order"
            )

        async with httpx.AsyncClient() as client:
            for row in rows:
                lk = row["lookup_key"]
                db_month = float(row["price_usd"] or 0)
                want_cents = int(round(db_month * 100))
                price = await _find_price_by_lookup_key(client, lk, stripe_key)
                stripe_cents = int(price["unit_amount"]) if price else None
                item: Dict[str, Any] = {
                    "lookup_key": lk,
                    "db_price_usd": db_month,
                    "stripe_price_cents": stripe_cents,
                    "match": stripe_cents == want_cents if stripe_cents is not None else False,
                }
                if apply and stripe_cents is not None and stripe_cents != want_cents:
                    new_usd = stripe_cents / 100.0
                    await conn.execute(
                        """
                        UPDATE catalog_products
                        SET price_usd = $1::numeric, updated_at = NOW()
                        WHERE lookup_key = $2
                        """,
                        new_usd,
                        lk,
                    )
                    applied.append(
                        {
                            "lookup_key": lk,
                            "field": "price_usd",
                            "from_usd": db_month,
                            "to_usd": new_usd,
                        }
                    )
                    item["db_price_usd"] = new_usd
                    item["match"] = True

                if row["product_kind"] == "subscription" and row.get("price_usd_yearly"):
                    ylk = lk.replace("_monthly", "_yearly")
                    db_y = float(row["price_usd_yearly"] or 0)
                    want_y = int(round(db_y * 100))
                    py = await _find_price_by_lookup_key(client, ylk, stripe_key)
                    y_cents = int(py["unit_amount"]) if py else None
                    item["yearly_lookup_key"] = ylk
                    item["db_yearly_usd"] = db_y
                    item["stripe_yearly_cents"] = y_cents
                    item["yearly_match"] = y_cents == want_y if y_cents is not None else False
                    if apply and y_cents is not None and y_cents != want_y:
                        new_y = y_cents / 100.0
                        await conn.execute(
                            """
                            UPDATE catalog_products
                            SET price_usd_yearly = $1::numeric, updated_at = NOW()
                            WHERE lookup_key = $2
                            """,
                            new_y,
                            lk,
                        )
                        applied.append(
                            {
                                "lookup_key": lk,
                                "field": "price_usd_yearly",
                                "from_usd": db_y,
                                "to_usd": new_y,
                            }
                        )
                        item["db_yearly_usd"] = new_y
                        item["yearly_match"] = True

                out.append(item)
    finally:
        await conn.close()

    return out, applied


async def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lookup-key", help="Only this catalog lookup_key")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Update catalog_products from Stripe when a Price exists and amounts differ",
    )
    args = p.parse_args()
    rows, applied = await _run(args.lookup_key, apply=args.apply)
    payload: Dict[str, Any] = {"items": rows}
    if args.apply:
        payload["applied"] = applied
    print(json.dumps(payload, indent=2))
    mism = [
        r
        for r in rows
        if (not r.get("match"))
        or ("yearly_match" in r and r.get("yearly_match") is False)
    ]
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
