"""
Train M8 publish-hour priors from PCI + uploads and write ``m8_publish_hour_priors``.

Usage (from repo root, with DATABASE_URL set):

    python -m jobs.train_m8_publish_hour_priors

Optional env:
    M8_TRAIN_LOOKBACK_DAYS           default 420
    M8_TRAIN_PCI_ONLY                default 1 (require pci.published_at; 0 = COALESCE with upload times)
    M8_TRAIN_SOURCE_ALLOWLIST        e.g. uploadm8,linked (comma-separated, lower)
    M8_TRAIN_CONTENT_KIND_ALLOWLIST  e.g. reel,short (optional)

Each run inserts one row into ``m8_model_runs`` (metrics JSON includes SHAP + calibration).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import asyncpg


async def _main() -> int:
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        return 1

    from services.m8_publish_hour_model import train_m8_publish_hour_priors, training_lookback_days_from_env

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        metrics = await train_m8_publish_hour_priors(pool, lookback_days=training_lookback_days_from_env())
        print(metrics)
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
