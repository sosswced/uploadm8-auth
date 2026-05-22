"""
One-shot: clear catalog_products.hud and push Stripe descriptions (no HUD overlay phrase).

Usage:
    python -m scripts.disable_catalog_hud_and_sync
    python -m scripts.disable_catalog_hud_and_sync --dry-run
    python -m scripts.disable_catalog_hud_and_sync --db-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


async def _run(*, dry_run: bool, db_only: bool) -> int:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2

    conn = await asyncpg.connect(database_url)
    try:
        before = await conn.fetch(
            """
            SELECT lookup_key, hud
            FROM catalog_products
            WHERE hud IS TRUE AND COALESCE(is_archived, FALSE) IS FALSE
            ORDER BY lookup_key
            """
        )
        print(f"Rows with hud=true before update: {len(before)}")
        for row in before:
            print(f"  - {row['lookup_key']}")

        if dry_run:
            print("(dry-run) skipping UPDATE and Stripe sync")
            return 0

        updated = await conn.execute(
            """
            UPDATE catalog_products
            SET hud = FALSE, updated_at = NOW()
            WHERE hud IS TRUE
            """
        )
        print(f"UPDATE result: {updated}")

        remaining = await conn.fetchval(
            "SELECT COUNT(*)::int FROM catalog_products WHERE hud IS TRUE"
        )
        print(f"Rows with hud=true after update: {remaining}")
        if remaining:
            print("Warning: some rows still have hud=true", file=sys.stderr)
            return 1

        if db_only:
            print("DB-only mode; skipping Stripe sync")
            return 0

        stripe_key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
        if not stripe_key:
            print("STRIPE_SECRET_KEY is not set; DB updated but Stripe not synced", file=sys.stderr)
            return 3

        from scripts.sync_stripe_catalog import sync_all

        actor = os.environ.get("USER", "disable_catalog_hud_and_sync")
        payload = await sync_all(conn, actor=actor, dry_run=False)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    finally:
        await conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Disable catalog HUD flag and sync Stripe")
    p.add_argument("--dry-run", action="store_true", help="Report only; no DB or Stripe writes")
    p.add_argument("--db-only", action="store_true", help="UPDATE catalog only; no Stripe sync")
    args = p.parse_args()
    return asyncio.run(_run(dry_run=args.dry_run, db_only=args.db_only))


if __name__ == "__main__":
    raise SystemExit(main())
