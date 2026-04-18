"""Build routers/analytics.py from a slice file (e.g. _analytics_block.py).

Line numbers drift whenever app.py changes. Re-find the block with ripgrep for the first
``@app.get("/api/analytics")`` through the end of ``get_upload_counts_by-token`` (before
``SupportContactRequest``), slice to a temp file, then:

  - Remove mid-module ``import time as _time`` / ``import asyncio as _asyncio``; use ``time`` / ``asyncio``.
  - Use ``parse_iso_ts`` / ``aggregate_platform_metrics_live`` from ``services.platform_metrics_ui``
    (not local defs).

Run: python tools/build_analytics_router.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = ROOT / "_analytics_block.py"
OUT = ROOT / "routers" / "analytics.py"

HEADER = '''"""Analytics and exports API (/api/analytics/*, /api/exports/*)."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request
from fastapi.responses import Response, StreamingResponse

from services import metric_definitions as metric_defs
from services.billing import get_plan
from services.platform_channels import list_analytics_platform_query_values, resolve_analytics_platform_filter
from services.platform_metrics_ui import aggregate_platform_metrics_live, parse_iso_ts
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN

router = APIRouter(tags=["analytics"])


class _LazyApp:
    __slots__ = ()

    def __getattr__(self, name: str):
        import app as _app_module

        return getattr(_app_module, name)


M = _LazyApp()


async def _session_user(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    return await M.get_current_user(request, authorization)


async def _session_user_readonly(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    return await M.get_current_user_readonly(request, authorization)


'''


def _sub_word(pat: str, repl: str, text: str) -> str:
    return re.sub(rf"(?<![\\w.]){pat}(?![\\w])", repl, text)


def main() -> None:
    b = BLOCK.read_text(encoding="utf-8")
    b = b.replace("@app.", "@router.")
    b = b.replace("Depends(get_current_user_readonly)", "Depends(_session_user_readonly)")
    b = b.replace("Depends(get_current_user)", "Depends(_session_user)")
    # Keep keyword name `db_pool=` intact; only replace value references.
    b = b.replace("db_pool=", "__DBPOOL_KW__")
    for name in (
        "db_pool",
        "decrypt_blob",
        "audit_log",
        "_user_upload_kpi_bundle",
        "_range_to_minutes",
    ):
        b = _sub_word(name, f"M.{name}", b)
    b = b.replace("__DBPOOL_KW__", "db_pool=")
    b = b.replace("logger.", "M.logger.")
    b = b.replace("_now_utc()", "M._now_utc()")
    OUT.write_text(HEADER + b.lstrip("\n"), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
