#!/usr/bin/env python3
"""
Re-queue a stuck upload after the UUID enqueue fix is deployed.

Usage:
  python -m dotenv run -- python scripts/requeue_stuck_upload.py 46681f6a-5655-469e-9465-35fed97b4500
  python -m dotenv run -- python scripts/requeue_stuck_upload.py 46681f6a-... --reset-to-queued

Requires DATABASE_URL and REDIS_URL (same as worker).
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


async def main() -> int:
    parser = argparse.ArgumentParser(description="Re-queue a stuck upload to the process lane")
    parser.add_argument("upload_id", help="Upload UUID")
    parser.add_argument(
        "--reset-to-queued",
        action="store_true",
        help="Set status=queued and clear processing_started_at before enqueue",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build payload only, no Redis LPUSH")
    args = parser.parse_args()

    import asyncpg
    import redis.asyncio as aioredis

    from core.upload_baseline_defaults import serialize_job_payload
    from worker import _build_process_job_payload, enqueue_process_lane_job

    db_url = os.environ.get("DATABASE_URL", "").strip()
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not db_url:
        print("DATABASE_URL required", file=sys.stderr)
        return 1

    upload_id = args.upload_id.strip()
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2)

    import worker

    worker.db_pool = pool

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, user_id, status, error_code FROM uploads WHERE id = $1",
            upload_id,
        )
    if not row:
        print(f"Upload not found: {upload_id}", file=sys.stderr)
        await pool.close()
        return 1

    user_id = str(row["user_id"])
    print(f"upload={upload_id} user={user_id} status={row['status']} error={row.get('error_code')}")

    if args.reset_to_queued:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE uploads
                   SET status = 'queued',
                       processing_started_at = NULL,
                       error_code = NULL,
                       error_detail = NULL,
                       updated_at = NOW()
                 WHERE id = $1
                """,
                upload_id,
            )
        print("Reset status → queued")

    payload = await _build_process_job_payload(
        upload_id,
        user_id,
        deferred=False,
        job_id=f"manual-requeue-{upload_id}",
        resume_from_checkpoint=True,
    )
    if not payload:
        print("Failed to build job payload", file=sys.stderr)
        await pool.close()
        return 1

    # Prove JSON-safe before Redis
    encoded = serialize_job_payload({**payload, "lane": "process"})
    json.loads(encoded)
    print(f"Payload OK ({len(encoded)} bytes), priority={payload.get('priority_class')}")

    if args.dry_run:
        print(json.dumps(json.loads(encoded), indent=2)[:2000])
        await pool.close()
        return 0

    if not redis_url:
        print("REDIS_URL required for enqueue (or use --dry-run)", file=sys.stderr)
        await pool.close()
        return 1

    worker.redis_client = aioredis.from_url(redis_url, decode_responses=True)
    ok = await enqueue_process_lane_job(payload)
    await worker.redis_client.aclose()
    await pool.close()

    if not ok:
        print("enqueue_process_lane_job returned False", file=sys.stderr)
        return 1

    print("Enqueued successfully — worker should pick up within seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
