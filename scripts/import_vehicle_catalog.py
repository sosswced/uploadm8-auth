"""One-shot: populate vehicle_makes from NHTSA vPIC (requires DATABASE_URL)."""

from __future__ import annotations

import asyncio
import os
import sys

import asyncpg


async def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL required", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.vehicle_catalog import sync_consumer_makes  # noqa: E402

    conn = await asyncpg.connect(dsn)
    try:
        n = await sync_consumer_makes(conn)
        print(f"Processed {n} consumer car/truck makes from NHTSA")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
