"""Run schema migrations against DATABASE_URL (no full app lifespan)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import asyncpg

from core.config import DATABASE_URL
from core.helpers import _init_asyncpg_codecs
from migrations.runtime_migrations import run_migrations


async def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=2,
        command_timeout=30,
        init=_init_asyncpg_codecs,
    )
    try:
        await run_migrations(pool)
        print("migrations_ok")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
