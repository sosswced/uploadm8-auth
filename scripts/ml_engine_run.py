#!/usr/bin/env python
"""Manual trigger for one UploadM8 ML engine cycle (loads repo-root .env)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env")
except ImportError:
    pass


async def _main() -> int:
    dsn = (os.environ.get("DATABASE_URL") or "").strip()
    pool = None
    if dsn:
        import asyncpg

        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    from services.ml_engine import run_ml_engine_cycle

    result = await run_ml_engine_cycle(pool, force=True)
    print(json.dumps(result, indent=2))
    if pool:
        await pool.close()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
