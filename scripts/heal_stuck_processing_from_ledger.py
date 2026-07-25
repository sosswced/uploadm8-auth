#!/usr/bin/env python3
"""One-shot heal: terminalize stuck processing uploads from publish_attempts.

Usage:
  python scripts/heal_stuck_processing_from_ledger.py 9e677d10-60a2-4fe2-96aa-e41d220c3552
  python scripts/heal_stuck_processing_from_ledger.py --scan-limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _pool():
    import asyncpg
    from dotenv import load_dotenv

    from core.helpers import _init_asyncpg_codecs

    load_dotenv(ROOT / ".env")
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_URL_POOLED")
    if not dsn:
        raise SystemExit("DATABASE_URL not set")
    # Codecs + json_param_encoder in reconcile: either path alone is enough;
    # both keeps heal scripts aligned with worker/API pools.
    return await asyncpg.create_pool(
        dsn, min_size=1, max_size=2, init=_init_asyncpg_codecs
    )


async def heal_one(pool, upload_id: str, *, force: bool = False) -> dict:
    from services.upload.publish_ledger_reconcile import (
        reconcile_stuck_processing_from_ledger,
    )

    return await reconcile_stuck_processing_from_ledger(
        pool, upload_id, force=force
    )


async def scan_and_heal(pool, limit: int) -> list:
    from services.upload.publish_ledger_reconcile import (
        reconcile_stuck_processing_from_ledger,
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT u.id::text AS upload_id
              FROM uploads u
             WHERE u.status = 'processing'
               AND EXISTS (
                    SELECT 1 FROM publish_attempts pa
                     WHERE pa.upload_id = u.id
                       AND pa.status = 'accepted'
               )
             ORDER BY u.updated_at ASC
             LIMIT $1
            """,
            max(1, int(limit)),
        )
    out = []
    for row in rows:
        uid = str(row["upload_id"])
        out.append(await reconcile_stuck_processing_from_ledger(pool, uid))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("upload_id", nargs="?", help="Upload UUID to heal")
    ap.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Scan up to N stuck processing rows with accepted ledger",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Terminalize even when accepted < expected target slots",
    )
    args = ap.parse_args()

    pool = await _pool()
    try:
        if args.upload_id:
            results = [await heal_one(pool, args.upload_id, force=args.force)]
        elif args.scan_limit > 0:
            results = await scan_and_heal(pool, args.scan_limit)
        else:
            ap.print_help()
            return 2
    finally:
        await pool.close()

    if args.json:
        print(json.dumps({"results": results}, default=str, indent=2))
    else:
        for r in results:
            print(
                f"{r.get('upload_id')} ok={r.get('ok')} state={r.get('state')} "
                f"reason={r.get('reason')} accepted={r.get('accepted_count')}"
            )
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
