#!/usr/bin/env python3
"""Collapse bloated uploads.platform_results (re-publish / ledger storms).

Usage:
  python scripts/dedupe_upload_platform_results.py 9e677d10-60a2-4fe2-96aa-e41d220c3552
  python scripts/dedupe_upload_platform_results.py --scan-limit 50
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

    load_dotenv(ROOT / ".env")
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_URL_POOLED")
    if not dsn:
        raise SystemExit("DATABASE_URL not set")
    return await asyncpg.create_pool(dsn, min_size=1, max_size=2)


async def dedupe_one(pool, upload_id: str) -> dict:
    from stages.asyncpg_json_codecs import json_param_encoder
    from services.uploads_api import dedupe_platform_result_entries

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platform_results FROM uploads WHERE id = $1::uuid",
            upload_id,
        )
        if not row:
            return {"upload_id": upload_id, "ok": False, "reason": "not_found"}
        raw = row["platform_results"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        items = [x for x in (raw or []) if isinstance(x, dict)] if isinstance(raw, list) else []
        before = len(items)
        deduped = dedupe_platform_result_entries(items)
        after = len(deduped)
        if after == before:
            return {
                "upload_id": upload_id,
                "ok": True,
                "reason": "already_clean",
                "before": before,
                "after": after,
            }
        await conn.execute(
            """
            UPDATE uploads
               SET platform_results = $2::jsonb, updated_at = NOW()
             WHERE id = $1::uuid
            """,
            upload_id,
            json_param_encoder(deduped),
        )
        return {
            "upload_id": upload_id,
            "ok": True,
            "reason": "deduped",
            "before": before,
            "after": after,
        }


async def scan(pool, limit: int) -> list:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text AS upload_id
              FROM uploads
             WHERE platform_results IS NOT NULL
               AND CASE
                     WHEN jsonb_typeof(platform_results) = 'array'
                     THEN jsonb_array_length(platform_results)
                     ELSE 0
                   END > 8
             ORDER BY updated_at DESC
             LIMIT $1
            """,
            max(1, int(limit)),
        )
    out = []
    for r in rows:
        out.append(await dedupe_one(pool, str(r["upload_id"])))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("upload_id", nargs="?")
    ap.add_argument("--scan-limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pool = await _pool()
    try:
        if args.upload_id:
            results = [await dedupe_one(pool, args.upload_id)]
        elif args.scan_limit > 0:
            results = await scan(pool, args.scan_limit)
        else:
            ap.print_help()
            return 2
    finally:
        await pool.close()

    if args.json:
        print(json.dumps({"results": results}, indent=2))
    else:
        for r in results:
            print(
                f"{r.get('upload_id')} ok={r.get('ok')} {r.get('reason')} "
                f"{r.get('before')}→{r.get('after')}"
            )
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
