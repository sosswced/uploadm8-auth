"""
TTL sweep for AV training packs.

Scans uploads.output_artifacts.av_training_pack_v1 for created_at past
AV_TRAINING_PACK_TTL_DAYS, then deletes R2 keys and clears pack keys in DB.

Safe default: dry-run unless --apply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _run(*, apply: bool, limit: int) -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from services.av_training_pack_retention import (
        DEFAULT_PACK_TTL_DAYS,
        fetch_expired_pack_targets,
        mark_pack_purged,
        purge_pack_r2_keys,
    )

    dsn = (os.environ.get("DATABASE_URL") or "").strip()
    report: dict = {
        "ttl_days": DEFAULT_PACK_TTL_DAYS,
        "dry_run": not apply,
        "limit": limit,
        "expired_count": 0,
        "deleted_keys": 0,
        "expired": [],
        "note": "Account delete also calls purge_user_ml_packs; no forever masters.",
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
    if not dsn:
        report["error"] = "DATABASE_URL required"
        return report

    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        targets = await fetch_expired_pack_targets(conn, limit=limit)
        report["expired_count"] = len(targets)
        report["expired"] = [
            {
                "upload_id": t["upload_id"],
                "user_id": t["user_id"],
                "created_at": t["created_at"],
                "key_count": len(t["r2_keys"]),
                "pack_r2_key": t["pack_r2_key"],
            }
            for t in targets[:50]
        ]
        if apply and targets:
            all_keys: list[str] = []
            for t in targets:
                all_keys.extend(t["r2_keys"])
            report["deleted_keys"] = purge_pack_r2_keys(all_keys)
            for t in targets:
                try:
                    await mark_pack_purged(conn, t["upload_id"])
                except Exception as e:
                    report.setdefault("mark_errors", []).append(
                        {"upload_id": t["upload_id"], "error": str(e)[:200]}
                    )
    finally:
        await conn.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Purge expired AV training packs (TTL)")
    ap.add_argument("--apply", action="store_true", help="Actually delete (default dry-run)")
    ap.add_argument("--limit", type=int, default=500, help="Max expired packs to process")
    args = ap.parse_args()
    report = asyncio.run(_run(apply=args.apply, limit=max(1, int(args.limit))))
    print(json.dumps(report, indent=2, default=str))
    return 1 if report.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
