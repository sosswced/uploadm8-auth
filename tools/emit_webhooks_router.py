"""One-off builder: merge _webhook_chunk.txt into routers/webhooks.py."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK = ROOT / "_webhook_chunk.txt"
OUT = ROOT / "routers" / "webhooks.py"

HEADER = '''"""Social webhooks (TikTok, Meta) — extracted from app.py."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import hmac as _hmac
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from core.time_utils import now_utc as _now_utc
from services.wallet import refund_tokens

router = APIRouter(tags=["webhooks"])

FACEBOOK_WEBHOOK_VERIFY_TOKEN = os.environ.get("FACEBOOK_WEBHOOK_VERIFY_TOKEN", "")

TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC = 300


'''


def main() -> None:
    lines = CHUNK.read_text(encoding="utf-8").splitlines()
    # Skip banner comment through blank line after =====
    i = 0
    while i < len(lines) and (
        lines[i].startswith("#")
        or lines[i].strip() == ""
        or lines[i].strip().startswith("=")
    ):
        i += 1
    # Skip duplicate `import hmac as _hmac` and blank and TIKTOK_WEBHOOK_REPLAY const (we define in HEADER)
    while i < len(lines) and (
        lines[i].strip() == "import hmac as _hmac"
        or lines[i].strip() == ""
        or lines[i].startswith("TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC")
    ):
        i += 1
    body = "\n".join(lines[i:])
    body = body.replace("@app.", "@router.")
    body = body.replace("db_pool.acquire", "m.db_pool.acquire")
    body = body.replace("_now_utc()", "_now_utc()")  # noop — uses core import
    body = body.replace("logger.", "m.logger.")
    body = body.replace("META_APP_SECRET", "m.META_APP_SECRET")
    body = body.replace("TIKTOK_WEBHOOK_SECRET", "m.TIKTOK_WEBHOOK_SECRET")
    body = body.replace("_safe_json(", "m._safe_json(")
    # _handle_tiktok_event: inject app import before first executable line
    needle = '    notes = f"event={event_type}"'
    inject = "    import app as m\n\n" + needle
    body = body.replace(needle, inject, 1)
    # Route handlers: add import after def line... simpler prepend to each async def that uses m.

    OUT.write_text(HEADER + body + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
