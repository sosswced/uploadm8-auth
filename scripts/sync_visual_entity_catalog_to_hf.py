# /// script
# dependencies = [
#   "asyncpg>=0.29.0",
#   "datasets>=2.18.0",
#   "huggingface_hub>=0.26.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""
Export UploadM8 visual entity catalogs to a Hugging Face dataset config.

Requires ``DATABASE_URL``, ``HF_TOKEN``, and ``UM8_HF_DATASET_REPO`` in the environment
(loads repo-root ``.env`` when python-dotenv is available).

Examples::

    uv run scripts/sync_visual_entity_catalog_to_hf.py --help
    uv run scripts/sync_visual_entity_catalog_to_hf.py
    uv run scripts/sync_visual_entity_catalog_to_hf.py --limit-users 50

HF Jobs (upload this script to a reachable URL or pass inline via ``hf jobs uv run``)::

    hf jobs uv run --flavor cpu-basic --timeout 30m --secrets HF_TOKEN \\
      scripts/sync_visual_entity_catalog_to_hf.py
"""

from __future__ import annotations

import argparse
import asyncio
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


async def _run(limit_users: int) -> int:
    import asyncpg

    from services.ml_entity_hub_sync import (
        build_platform_visual_entity_export,
        push_visual_entities_to_hub,
    )

    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        print("DATABASE_URL is required", file=sys.stderr)
        return 1

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    try:
        rows = await build_platform_visual_entity_export(
            pool, limit_per_user=limit_users
        )
        result = push_visual_entities_to_hub(rows)
        if not result.get("ok"):
            print(result, file=sys.stderr)
            return 1
        print(result)
        return 0
    finally:
        await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push user_visual_entity_catalog rows to UM8_HF_DATASET_REPO (visual_entities config)."
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=200,
        metavar="N",
        help="Max creators to export (default: 200)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(max(1, args.limit_users)))


if __name__ == "__main__":
    raise SystemExit(main())
