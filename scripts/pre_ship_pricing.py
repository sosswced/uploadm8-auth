#!/usr/bin/env python3
"""
Pre-ship billing / pricing automation.

1. Always: regenerate frontend pricing surfaces (guide, settings, wallet estimates).
2. Optional: force-upsert ``billing_service_weights`` from code defaults (live AIC at presign).

Usage:
  python scripts/pre_ship_pricing.py
  python scripts/pre_ship_pricing.py --force-db-weights
  python scripts/pre_ship_pricing.py --force-db-weights --json
  python scripts/pre_ship_pricing.py --check   # surfaces drift only (no DB write)

Env for DB force:
  DATABASE_URL or DATABASE_PUBLIC_URL (same as API)

Wire this before frontend robocopy (/333 step 3, uploadm8-frontend-push).
After backend deploy to Render, run once with --force-db-weights against prod DB
(or use Debit weights → Reset / BILLING_WEIGHTS_FORCE_SYNC_FROM_CODE=1 one boot).
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


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        # Minimal fallback — do not override existing env
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _run_surfaces(*, check: bool) -> dict:
    from scripts.sync_pricing_surfaces import write_all

    return write_all(check=check)


async def _force_db_weights() -> dict:
    import asyncpg

    url = (
        (os.environ.get("DATABASE_URL") or "").strip()
        or (os.environ.get("DATABASE_PUBLIC_URL") or "").strip()
    )
    if not url:
        return {
            "ok": False,
            "error": "DATABASE_URL / DATABASE_PUBLIC_URL not set — skip DB force or set env",
        }
    from services.billing_service_weights import (
        ensure_billing_weights_seeded,
        fetch_service_weights_map,
        sync_service_weights_from_code,
        weights_drift_summary,
    )

    conn = await asyncpg.connect(url)
    try:
        await ensure_billing_weights_seeded(conn)
        n = await sync_service_weights_from_code(
            conn, updated_by="pre_ship_pricing:force-db-weights"
        )
        raw = await fetch_service_weights_map(conn)
        drift = weights_drift_summary(raw)
        return {
            "ok": True,
            "rows_written": n,
            "drifted_count": drift.get("drifted_count"),
            "calibration": drift.get("calibration"),
        }
    finally:
        await conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-ship pricing surfaces + optional DB weight force")
    ap.add_argument(
        "--force-db-weights",
        action="store_true",
        help="Upsert billing_service_weights from SERVICE_WEIGHTS (requires DATABASE_URL)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Only verify pricing surfaces are in sync (no writes except report)",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--skip-surfaces",
        action="store_true",
        help="Only run DB force (no sync_pricing_surfaces)",
    )
    args = ap.parse_args()
    _load_dotenv()

    report: dict = {"ok": True, "surfaces": None, "db_weights": None}

    if not args.skip_surfaces:
        surfaces = _run_surfaces(check=args.check)
        report["surfaces"] = surfaces
        if not surfaces.get("ok"):
            report["ok"] = False

    if args.force_db_weights:
        if args.check:
            report["db_weights"] = {
                "ok": True,
                "skipped": True,
                "note": "--check does not write DB; omit --check to force",
            }
        else:
            db = asyncio.run(_force_db_weights())
            report["db_weights"] = db
            if not db.get("ok"):
                report["ok"] = False

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"pre_ship_pricing ok={report['ok']}")
        if report.get("surfaces"):
            s = report["surfaces"]
            print(
                f"  surfaces calibration={s.get('calibration')} "
                f"js_changed={s.get('js_changed')} ok={s.get('ok')}"
            )
        if report.get("db_weights"):
            d = report["db_weights"]
            print(f"  db_weights={d}")
        if not report["ok"]:
            err = (report.get("surfaces") or {}).get("error") or (
                report.get("db_weights") or {}
            ).get("error")
            if err:
                print(err, file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
