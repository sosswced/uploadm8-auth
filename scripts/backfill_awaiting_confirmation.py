#!/usr/bin/env python3
"""
Backfill stuck "Awaiting confirmation" uploads.

Fixes two failure modes found in prod:

  1. Double-encoded ``platform_results`` jsonb (JSON string containing an array)
     — UI parsers saw an empty list → permanent Awaiting confirmation.
  2. TikTok Step A ``publish_id`` without a confirmed ``video_id`` — re-run verify
     + catalog title match.

Usage:
  python scripts/backfill_awaiting_confirmation.py --dry-run
  python scripts/backfill_awaiting_confirmation.py --apply
  python scripts/backfill_awaiting_confirmation.py --apply --days 30 --limit 500

Requires DATABASE_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def _unwrap_pr(raw: Any) -> list:
    cur = raw
    for _ in range(5):
        if isinstance(cur, str):
            try:
                cur = json.loads(cur)
            except Exception:
                return []
            continue
        break
    if isinstance(cur, dict):
        return [
            {"platform": k, **v} if isinstance(v, dict) else {"platform": k}
            for k, v in cur.items()
        ]
    if isinstance(cur, list):
        return [x for x in cur if isinstance(x, dict)]
    return []


def _is_double_encoded(raw: Any) -> bool:
    """True when the decoded value is still a JSON string (legacy double-encode).

    With the asyncpg JSON codec, a jsonb *string* row decodes to ``str``. A real
    jsonb array decodes to ``list``.
    """
    return isinstance(raw, str)


def _entry_confirmed(entry: dict) -> bool:
    return bool(
        str(
            entry.get("platform_video_id")
            or entry.get("video_id")
            or entry.get("media_id")
            or entry.get("post_id")
            or entry.get("share_id")
            or entry.get("platform_url")
            or entry.get("url")
            or ""
        ).strip()
    )


def _needs_tiktok_video_id(pr: list) -> bool:
    for e in pr:
        if str(e.get("platform") or "").lower() != "tiktok":
            continue
        if e.get("success") is False:
            continue
        if not _entry_confirmed(e) and e.get("publish_id"):
            return True
    return False


def _awaits_confirmation(pr: list, *, raw: Any) -> bool:
    if _is_double_encoded(raw):
        return True
    if not pr:
        return True
    for e in pr:
        if e.get("success") is False:
            continue
        if not _entry_confirmed(e):
            return True
    return False


async def _load_candidates(
    conn: Any, *, days: int, limit: int, user_id: Optional[str]
) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, user_id, status, title, platforms, platform_results, created_at
          FROM uploads
         WHERE status IN ('succeeded', 'completed', 'partial')
           AND platform_results IS NOT NULL
           AND created_at > NOW() - ($1::text || ' days')::interval
           AND ($2::uuid IS NULL OR user_id = $2::uuid)
         ORDER BY created_at DESC
         LIMIT $3
        """,
        str(int(days)),
        user_id,
        int(limit),
    )
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        raw = d.get("platform_results")
        pr = _unwrap_pr(raw)
        double_enc = _is_double_encoded(raw)
        needs_tt = _needs_tiktok_video_id(pr)
        if double_enc or _awaits_confirmation(pr, raw=raw) or needs_tt:
            d["platform_results"] = pr
            d["_raw_pr"] = raw
            d["_double_encoded"] = double_enc
            d["_needs_tiktok_vid"] = needs_tt
            out.append(d)
    return out


async def _sql_unwrap_all_string_platform_results(conn: Any) -> int:
    """Peel jsonb *string* rows ('\"[{...}]\"') into real arrays/objects."""
    return int(
        await conn.fetchval(
            """
            WITH u AS (
                UPDATE uploads
                   SET platform_results = (platform_results #>> '{}')::jsonb,
                       updated_at = NOW()
                 WHERE platform_results IS NOT NULL
                   AND jsonb_typeof(platform_results) = 'string'
                   AND left(platform_results #>> '{}', 1) IN ('[', '{')
             RETURNING 1
            )
            SELECT COUNT(*)::int FROM u
            """
        )
        or 0
    )


async def _rewrite_platform_results(conn: Any, upload_id: str, pr: list) -> None:
    # Prefer Python list when a JSON codec is registered; fall back to dumps for
    # raw pools that still expect a JSON text string.
    try:
        await conn.execute(
            """
            UPDATE uploads
               SET platform_results = $1::jsonb, updated_at = NOW()
             WHERE id = $2::uuid
            """,
            pr,
            upload_id,
        )
    except Exception:
        await conn.execute(
            """
            UPDATE uploads
               SET platform_results = $1::jsonb, updated_at = NOW()
             WHERE id = $2::uuid
            """,
            json.dumps(pr, separators=(",", ":"), ensure_ascii=False),
            upload_id,
        )


async def _reset_tiktok_attempts(conn: Any, upload_ids: list[str]) -> int:
    if not upload_ids:
        return 0
    return int(
        await conn.fetchval(
            """
            WITH u AS (
                UPDATE publish_attempts
                   SET verify_status = 'pending',
                       verified_at = NULL,
                       updated_at = NOW()
                 WHERE upload_id = ANY($1::uuid[])
                   AND platform = 'tiktok'
                   AND status = 'accepted'
                   AND (
                        platform_post_id IS NULL OR platform_post_id = ''
                   )
             RETURNING 1
            )
            SELECT COUNT(*)::int FROM u
            """,
            upload_ids,
        )
        or 0
    )


async def _load_tiktok_attempts(conn: Any, upload_ids: list[str]) -> list[dict]:
    if not upload_ids:
        return []
    rows = await conn.fetch(
        """
        SELECT *
          FROM publish_attempts
         WHERE upload_id = ANY($1::uuid[])
           AND platform = 'tiktok'
           AND status = 'accepted'
           AND (platform_post_id IS NULL OR platform_post_id = '')
         ORDER BY created_at ASC
        """,
        upload_ids,
    )
    return [dict(r) for r in rows]


async def _catalog_backfill_for_users(conn: Any, user_ids: list[str]) -> dict[str, int]:
    from services.catalog_sync import _backfill_tiktok_video_ids

    totals = {"accounts": 0, "uploads_patched": 0}
    for uid in user_ids:
        accounts = await conn.fetch(
            """
            SELECT DISTINCT account_id
              FROM platform_tokens
             WHERE user_id = $1::uuid
               AND platform = 'tiktok'
               AND account_id IS NOT NULL AND account_id != ''
               AND (revoked_at IS NULL OR revoked_at > NOW())
            """,
            uid,
        )
        for a in accounts:
            aid = str(a["account_id"] or "").strip()
            if not aid:
                continue
            totals["accounts"] += 1
            n = await _backfill_tiktok_video_ids(conn, uid, aid)
            totals["uploads_patched"] += int(n or 0)
    return totals


async def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill stuck Awaiting confirmation uploads")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--email", type=str, default="")
    parser.add_argument("--user-id", type=str, default="")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-catalog", action="store_true")
    parser.add_argument(
        "--sync-catalog",
        action="store_true",
        help="Run platform catalog sync for affected users before title-match backfill",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    apply = bool(args.apply)

    import asyncpg

    from stages.publish_stage import init_enc_keys
    from stages.verify_stage import verify_single_attempt

    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        print("DATABASE_URL required", file=sys.stderr)
        return 1

    from stages.asyncpg_json_codecs import apply_asyncpg_json_codecs

    async def _init_conn(conn):
        await apply_asyncpg_json_codecs(conn)

    pool = await asyncpg.create_pool(
        db_url, min_size=1, max_size=4, init=_init_conn
    )
    init_enc_keys()
    user_id: Optional[str] = (args.user_id or "").strip() or None
    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "days": args.days,
        "candidates": 0,
        "double_encoded": 0,
        "needs_tiktok_vid": 0,
        "rewritten": 0,
        "attempts_reset": 0,
        "attempts_verified": 0,
        "catalog": {},
        "remaining": 0,
        "sample": [],
    }

    try:
        async with pool.acquire() as conn:
            if args.email and not user_id:
                user_id = await conn.fetchval(
                    "SELECT id::text FROM users WHERE lower(email) = lower($1)",
                    args.email.strip(),
                )
                if not user_id:
                    print(f"No user for email={args.email}", file=sys.stderr)
                    return 1

            stuck = await _load_candidates(
                conn, days=args.days, limit=args.limit, user_id=user_id
            )
            report["candidates"] = len(stuck)
            report["double_encoded"] = sum(1 for u in stuck if u.get("_double_encoded"))
            report["needs_tiktok_vid"] = sum(1 for u in stuck if u.get("_needs_tiktok_vid"))
            report["sample"] = [
                {
                    "id": str(u["id"]),
                    "created_at": u["created_at"].isoformat() if u.get("created_at") else None,
                    "title": (u.get("title") or "")[:70],
                    "double_encoded": bool(u.get("_double_encoded")),
                    "needs_tiktok_vid": bool(u.get("_needs_tiktok_vid")),
                }
                for u in stuck[:20]
            ]

            if not args.json:
                print(
                    f"candidates={report['candidates']} "
                    f"double_encoded={report['double_encoded']} "
                    f"needs_tiktok_vid={report['needs_tiktok_vid']}"
                )
                for s in report["sample"][:10]:
                    print(
                        f"  {s['id'][:8]}… enc={s['double_encoded']} "
                        f"tt={s['needs_tiktok_vid']} {s['title']!r}"
                    )

            if not apply:
                if args.json:
                    print(json.dumps(report, indent=2, default=str))
                else:
                    print("Dry-run only. Re-run with --apply to backfill.")
                return 0

            # 1) Normalize ALL jsonb-string platform_results → real arrays/objects
            report["rewritten"] = await _sql_unwrap_all_string_platform_results(conn)
            # Also rewrite candidate rows we already unwrapped in Python (belt/suspenders).
            for u in stuck:
                if u.get("_double_encoded") or u.get("_needs_tiktok_vid"):
                    try:
                        await _rewrite_platform_results(
                            conn, str(u["id"]), u["platform_results"]
                        )
                    except Exception:
                        pass

            tiktok_ids = [str(u["id"]) for u in stuck if u.get("_needs_tiktok_vid")]
            user_ids = sorted({str(u["user_id"]) for u in stuck})

            # 2) Re-verify TikTok for missing video_ids
            if tiktok_ids and not args.skip_verify:
                report["attempts_reset"] = await _reset_tiktok_attempts(conn, tiktok_ids)
                attempts = await _load_tiktok_attempts(conn, tiktok_ids)
                if not args.json:
                    print(f"Verifying {len(attempts)} TikTok attempt(s)…")
                for attempt in attempts:
                    try:
                        await verify_single_attempt(pool, attempt)
                        report["attempts_verified"] += 1
                    except Exception as e:
                        print(f"  verify failed: {e}", file=sys.stderr)
                    await asyncio.sleep(0.4)

            # 3) Optional live catalog sync, then title-match for TikTok video ids
            if args.sync_catalog and user_ids:
                from services.catalog_sync import sync_catalog_for_user

                if not args.json:
                    print(f"Catalog sync for {len(user_ids)} user(s)…")
                sync_totals = {"discovered": 0, "upserted": 0, "errors": 0}
                for uid in user_ids:
                    try:
                        sr = await sync_catalog_for_user(pool, uid, force_full=False)
                        for k in sync_totals:
                            sync_totals[k] += int((sr or {}).get(k) or 0)
                    except Exception as e:
                        print(f"  catalog sync failed user={uid[:8]}: {e}", file=sys.stderr)
                report["catalog_sync"] = sync_totals

            if not args.skip_catalog:
                if not args.json:
                    print(f"Catalog backfill for {len(user_ids)} user(s)…")
                report["catalog"] = await _catalog_backfill_for_users(conn, user_ids)

            remaining = await _load_candidates(
                conn, days=args.days, limit=args.limit, user_id=user_id
            )
            # After rewrite, only count those still missing confirmed ids
            report["remaining"] = sum(
                1
                for u in remaining
                if any(
                    (e.get("success") is not False) and not _entry_confirmed(e)
                    for e in (u.get("platform_results") or [])
                )
                or not (u.get("platform_results") or [])
            )

        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(
                f"Done. rewritten={report['rewritten']} "
                f"verified={report['attempts_verified']} "
                f"catalog={report['catalog']} "
                f"remaining_missing_ids={report['remaining']}"
            )
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
