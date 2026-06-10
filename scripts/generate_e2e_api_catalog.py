#!/usr/bin/env python3
"""Dump OpenAPI GET cases for master-admin E2E review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from tests.e2e.helpers.auth import api_client
from tests.e2e.helpers.openapi_catalog import build_context, fetch_openapi, iter_read_smoke_cases


def main() -> int:
    out = ROOT / "tests" / "e2e" / "catalog" / "read_endpoints.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with api_client() as client:
        openapi = fetch_openapi(client)
        ctx = build_context(client)
        cases = [
            {"method": c.method, "path": c.path, "summary": c.summary}
            for c in iter_read_smoke_cases(openapi, ctx=ctx, safe_only=True)
        ]
    out.write_text(json.dumps({"count": len(cases), "cases": cases}, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} GET cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
