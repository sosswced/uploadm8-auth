"""Build routers/catalog.py from _catalog_block.py (extracted from app.py).

Re-extract after app.py edits (line numbers drift):
  lines = Path("app.py").read_text(encoding="utf-8").splitlines(keepends=True)
  # 1-based L..M -> slice [L-1:M]
  Path("_catalog_block.py").write_text("".join(lines[L0:M0]), encoding="utf-8")
Then run: python tools/build_catalog_router.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = ROOT / "_catalog_block.py"
OUT = ROOT / "routers" / "catalog.py"

HEADER = '''"""Unified content catalog API (/api/catalog/*)."""
from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request

from services.platform_metrics_ui import parse_iso_ts

router = APIRouter(tags=["catalog"])


async def _session_user(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    import app as m

    return await m.get_current_user(request, authorization)


async def _session_user_readonly(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    import app as m

    return await m.get_current_user_readonly(request, authorization)


'''


def main() -> None:
    b = BLOCK.read_text(encoding="utf-8")
    b = b.replace("@app.", "@router.")
    b = b.replace("Depends(get_current_user_readonly)", "Depends(_session_user_readonly)")
    b = b.replace("Depends(get_current_user)", "Depends(_session_user)")
    for old, new in [
        ("db_pool", "m.db_pool"),
        ("logger.", "m.logger."),
        ("get_s3_client()", "m.get_s3_client()"),
        ("R2_BUCKET_NAME", "m.R2_BUCKET_NAME"),
        ("_normalize_r2_key(", "m._normalize_r2_key("),
        ("_catalog_title_and_metrics_from_upload_pr(", "m._catalog_title_and_metrics_from_upload_pr("),
    ]:
        b = b.replace(old, new)

    needles = [
        (
            '    """\n    from services.catalog_sync import sync_catalog_for_user',
            '    """\n    import app as m\n\n    from services.catalog_sync import sync_catalog_for_user',
        ),
        (
            '    """Return per-token sync state (platform, status, last_synced_at, cursor, counts)."""\n    async with m.db_pool.acquire()',
            '    """Return per-token sync state (platform, status, last_synced_at, cursor, counts)."""\n    import app as m\n\n    async with m.db_pool.acquire()',
        ),
        (
            '    Each row includes source badge, upload_id if linked, and per-platform URL.\n    """\n    uid = str(user["id"])',
            '    Each row includes source badge, upload_id if linked, and per-platform URL.\n    """\n    import app as m\n\n    uid = str(user["id"])',
        ),
        (
            "    accounts — not just UploadM8-originated videos.\n    \"\"\"\n    from services.catalog_sync import get_catalog_aggregate",
            "    accounts — not just UploadM8-originated videos.\n    \"\"\"\n    import app as m\n\n    from services.catalog_sync import get_catalog_aggregate",
        ),
    ]
    b2 = b
    for old, new in needles:
        if old not in b2:
            raise SystemExit(f"missing needle fragment: {old[:60]!r}...")
        b2 = b2.replace(old, new, 1)

    OUT.write_text(HEADER + b2.lstrip("\n"), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
