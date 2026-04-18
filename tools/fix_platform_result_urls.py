#!/usr/bin/env python3
"""
One-shot repair for known-bad post URLs stored in uploads.platform_results.

Fixes:
  • Facebook: legacy https://www.facebook.com/video/{id} → watch/?v= or /{page}/videos/{id}
  • TikTok:    https://www.tiktok.com/video/{id} → /@handle/video/{id} when account_username exists
  • Instagram: optional — copy permalink from platform_content_items (Graph sync) when media id matches

Usage (from repo root, DATABASE_URL in env):
  .venv\\Scripts\\python tools/fix_platform_result_urls.py --dry-run
  .venv\\Scripts\\python tools/fix_platform_result_urls.py --limit 50
  .venv\\Scripts\\python tools/fix_platform_result_urls.py
  .venv\\Scripts\\python tools/fix_platform_result_urls.py --no-instagram-from-catalog

Does not load core.config (avoids JWT_SECRET requirement); set DATABASE_URL only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import asyncpg


async def _init_asyncpg_codecs(conn: asyncpg.Connection) -> None:
    """Match core.helpers: decode json/jsonb as Python objects (no core.config import)."""
    try:
        await conn.set_type_codec(
            "json",
            encoder=lambda v: json.dumps(v, separators=(",", ":"), ensure_ascii=False),
            decoder=json.loads,
            schema="pg_catalog",
        )
    except Exception:
        pass
    try:
        await conn.set_type_codec(
            "jsonb",
            encoder=lambda v: json.dumps(v, separators=(",", ":"), ensure_ascii=False),
            decoder=json.loads,
            schema="pg_catalog",
        )
    except Exception:
        pass

BAD_FB_VIDEO_PATH = re.compile(
    r"https?://(?:www\.|m\.)?facebook\.com/video/(\d+)/?(?:\?.*)?$",
    re.IGNORECASE,
)
TIKTOK_BARE_VIDEO = re.compile(
    r"https?://(?:www\.)?tiktok\.com/video/(\d+)/?(?:\?.*)?$",
    re.IGNORECASE,
)


def _fb_canonical_url(account_id: str, video_id: str) -> str:
    aid = (account_id or "").strip()
    vid = (video_id or "").strip()
    if vid.isdigit() and aid.isdigit():
        return f"https://www.facebook.com/{aid}/videos/{vid}"
    if vid.isdigit():
        return f"https://www.facebook.com/watch/?v={vid}"
    if "_" in vid:
        return f"https://www.facebook.com/{vid}"
    return f"https://www.facebook.com/watch/?v={vid}"


def _apply_url_fields(entry: Dict[str, Any], new_url: str) -> bool:
    changed = False
    for key in ("platform_url", "url"):
        cur = entry.get(key)
        if cur != new_url:
            entry[key] = new_url
            changed = True
    return changed


def _fix_facebook_entry(entry: Dict[str, Any]) -> bool:
    plat = (entry.get("platform") or "").lower()
    if plat != "facebook":
        return False
    vid = str(entry.get("platform_video_id") or entry.get("video_id") or "").strip()
    account_id = str(entry.get("account_id") or "").strip()

    changed = False
    for ukey in ("platform_url", "url"):
        raw = entry.get(ukey)
        if not raw or not isinstance(raw, str):
            continue
        s = raw.strip()
        m = BAD_FB_VIDEO_PATH.match(s)
        if m:
            vid_from_url = m.group(1)
            use_vid = vid or vid_from_url
            new_u = _fb_canonical_url(account_id, use_vid)
            if _apply_url_fields(entry, new_u):
                changed = True
            break

    # If URL still uses /video/ but regex missed (querystring, etc.)
    for ukey in ("platform_url", "url"):
        raw = entry.get(ukey)
        if not raw or not isinstance(raw, str):
            continue
        low = raw.lower()
        if "facebook.com/video/" in low and "/watch/" not in low and "videos/" not in low:
            use_vid = vid
            if not use_vid:
                tail = raw.rstrip("/").split("/")[-1].split("?")[0]
                if tail.isdigit():
                    use_vid = tail
            if use_vid:
                new_u = _fb_canonical_url(account_id, use_vid)
                if _apply_url_fields(entry, new_u):
                    changed = True
                break
    return changed


def _fix_tiktok_entry(entry: Dict[str, Any]) -> bool:
    plat = (entry.get("platform") or "").lower()
    if plat != "tiktok":
        return False
    uname = (entry.get("account_username") or "").strip().lstrip("@")
    if not uname:
        return False
    vid = str(entry.get("platform_video_id") or entry.get("video_id") or "").strip()
    changed = False
    for ukey in ("platform_url", "url"):
        raw = entry.get(ukey)
        if not raw or not isinstance(raw, str):
            continue
        m = TIKTOK_BARE_VIDEO.match(raw.strip())
        if not m:
            continue
        vid_from_url = m.group(1)
        use_vid = vid or vid_from_url
        new_u = f"https://www.tiktok.com/@{uname}/video/{use_vid}"
        if _apply_url_fields(entry, new_u):
            changed = True
        break
    return changed


async def _instagram_pci_url(
    conn: asyncpg.Connection, user_id: Any, media_id: str
) -> Optional[str]:
    if not media_id or not str(media_id).strip():
        return None
    try:
        val = await conn.fetchval(
            """
            SELECT platform_url FROM platform_content_items
             WHERE user_id = $1::uuid
               AND platform = 'instagram'
               AND platform_video_id = $2
               AND COALESCE(TRIM(platform_url), '') <> ''
             ORDER BY updated_at DESC NULLS LAST
             LIMIT 1
            """,
            user_id,
            str(media_id).strip(),
        )
    except Exception:
        return None
    if not val or not str(val).startswith("http"):
        return None
    return str(val).strip()


async def _fix_instagram_entry(
    conn: asyncpg.Connection,
    user_id: Any,
    entry: Dict[str, Any],
    *,
    from_catalog: bool,
) -> bool:
    if not from_catalog:
        return False
    plat = (entry.get("platform") or "").lower()
    if plat != "instagram":
        return False
    mid = str(entry.get("platform_video_id") or entry.get("video_id") or "").strip()
    if not mid:
        return False
    pci = await _instagram_pci_url(conn, user_id, mid)
    if not pci:
        return False
    cur = (entry.get("platform_url") or entry.get("url") or "").strip()
    if cur == pci:
        return False
    return _apply_url_fields(entry, pci)


def _parse_pr(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else None
        except json.JSONDecodeError:
            return None
    return None


async def _process_uploads(
    conn: asyncpg.Connection,
    *,
    dry_run: bool,
    limit: Optional[int],
    instagram_from_catalog: bool,
) -> Tuple[int, int]:
    """Returns (rows_scanned, rows_updated)."""
    sql = """
        SELECT id, user_id, platform_results
          FROM uploads
         WHERE platform_results IS NOT NULL
           AND jsonb_typeof(platform_results) = 'array'
           AND jsonb_array_length(platform_results) > 0
         ORDER BY updated_at DESC NULLS LAST
    """
    args: List[Any] = []
    if limit is not None:
        sql += " LIMIT $1"
        args.append(limit)

    rows = await conn.fetch(sql, *args) if args else await conn.fetch(sql)

    updated_rows = 0
    for row in rows:
        pr = _parse_pr(row["platform_results"])
        if not pr:
            continue
        changed = False
        new_list: List[Any] = []
        for item in pr:
            if not isinstance(item, dict):
                new_list.append(item)
                continue
            e = dict(item)
            if _fix_facebook_entry(e):
                changed = True
            if _fix_tiktok_entry(e):
                changed = True
            if await _fix_instagram_entry(
                conn, row["user_id"], e, from_catalog=instagram_from_catalog
            ):
                changed = True
            new_list.append(e)

        if changed:
            updated_rows += 1
            if not dry_run:
                await conn.execute(
                    """
                    UPDATE uploads
                       SET platform_results = $1::jsonb,
                           updated_at = NOW()
                     WHERE id = $2
                    """,
                    json.dumps(new_list),
                    row["id"],
                )

    return len(rows), updated_rows


async def main() -> None:
    p = argparse.ArgumentParser(description="Repair platform_results post URLs in uploads.")
    p.add_argument("--dry-run", action="store_true", help="Report counts only; no DB writes")
    p.add_argument("--limit", type=int, default=None, help="Max uploads to scan (newest first)")
    p.add_argument(
        "--no-instagram-from-catalog",
        action="store_true",
        help="Do not overwrite Instagram URLs from platform_content_items",
    )
    args = p.parse_args()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set", file=sys.stderr)
        sys.exit(1)

    instagram_catalog = not args.no_instagram_from_catalog

    pool = await asyncpg.create_pool(
        dsn,
        min_size=1,
        max_size=4,
        command_timeout=120,
        init=_init_asyncpg_codecs,
    )
    try:
        async with pool.acquire() as conn:
            scanned, updated = await _process_uploads(
                conn,
                dry_run=args.dry_run,
                limit=args.limit,
                instagram_from_catalog=instagram_catalog,
            )
        mode = "dry_run" if args.dry_run else "applied"
        print(
            json.dumps(
                {
                    "mode": mode,
                    "uploads_scanned": scanned,
                    "uploads_updated": updated,
                    "instagram_from_catalog": instagram_catalog,
                },
                indent=2,
            )
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
