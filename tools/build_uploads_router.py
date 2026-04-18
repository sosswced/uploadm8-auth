"""Build routers/uploads.py by splicing app.py (line numbers drift — re-check anchors).

Anchors (1-based):
  Start: ``# Uploads`` section header (``# ============================================================``).
  End main block: blank line after ``retry_upload`` (before ``# User Color Preferences``).
  Second block: ``@app.get("/api/uploads/{upload_id}")`` … through end of ``get_upload_details``.

Splice app.py with::
  lines[:3829] + lines[7310:9945] + lines[10086:]

Regenerate ``routers/uploads.py`` only::

  python tools/build_uploads_router.py

One-time splice of ``app.py`` (destructive; do not run twice)::

  set UPLOADS_SPLICE_APP=1
  python tools/build_uploads_router.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
OUT = ROOT / "routers" / "uploads.py"

# 0-based slice indices — update if anchors drift (grep "# Uploads" / retry / colors / get_upload_details)
I0, I1 = 3829, 7310
J0, J1 = 9945, 10086

HEADER = '''"""Upload and scheduled-upload API (/api/uploads/*, /api/scheduled/*)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request
from fastapi import HTTPException

from schemas.uploads_api import SmartScheduleOnlyUpdate, UploadInit, UploadUpdate
from services.api_errors import api_problem
from services.billing import get_plan
from services.upload_metrics import SUCCESSFUL_STATUS_SQL_IN
from services.uploads import calculate_smart_schedule, get_existing_scheduled_days
from services.wallet import get_wallet, refund_tokens, reserve_tokens
from stages.context import expand_hashtag_items
from stages.entitlements import compute_upload_cost, get_entitlements_for_tier

router = APIRouter(tags=["uploads"])


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


def _transform(blob: str) -> str:
    blob = re.sub(
        r"\nclass CompleteUploadBody\(BaseModel\):.*?(?=\n\n@app\.post)",
        "\n",
        blob,
        count=1,
        flags=re.DOTALL,
    )
    blob = blob.replace("@app.api_route", "@router.api_route")
    blob = blob.replace("@app.", "@router.")
    blob = blob.replace("Depends(get_current_user_readonly)", "Depends(_session_user_readonly)")
    blob = blob.replace("Depends(get_current_user)", "Depends(_session_user)")

    blob = blob.replace("db_pool=", "__DBPOOL_KW__")
    for name in (
        "db_pool",
        "redis_client",
        "enqueue_job",
        "invalidate_me_api_cache",
        "log_system_event",
        "generate_presigned_upload_url",
        "get_s3_client",
        "decrypt_blob",
        "get_user_prefs_for_upload",
        "R2_BUCKET_NAME",
        "_normalize_r2_key",
        "_maybe_reconcile_stale_processing_on_read",
        "_load_uploads_columns",
        "_pick_cols",
        "_delete_r2_objects",
        "_safe_json",
        "cache_get",
        "cache_set",
        "CACHE_TTL_SHORT",
        "_now_utc",
    ):
        blob = _sub_word(name, f"M.{name}", blob)
    blob = blob.replace("__DBPOOL_KW__", "db_pool=")
    blob = blob.replace("logger.", "M.logger.")
    return blob


def main() -> None:
    lines = APP.read_text(encoding="utf-8").splitlines(keepends=True)
    part_a = "".join(lines[I0:I1])
    part_b = "".join(lines[J0:J1])
    blob = _transform(part_a + "\n" + part_b)
    OUT.write_text(HEADER + blob.lstrip("\n"), encoding="utf-8")
    print("wrote", OUT)

    if os.environ.get("UPLOADS_SPLICE_APP") == "1":
        new_text = "".join(lines[:I0] + lines[I1:J0] + lines[J1:])
        APP.write_text(new_text, encoding="utf-8")
        print("spliced app.py (UPLOADS_SPLICE_APP=1)")
    else:
        print("skip app.py splice (set UPLOADS_SPLICE_APP=1 to enable)")


if __name__ == "__main__":
    main()
