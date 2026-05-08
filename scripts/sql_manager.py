#!/usr/bin/env python
# /// script
# dependencies = [
#   "asyncpg>=0.29.0,<0.32.0",
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""
Phase 1 SQL manager for UploadM8 analytics/model exports.

Examples:
  uv run scripts/sql_manager.py describe --table uploads
  uv run scripts/sql_manager.py query --sql "SELECT COUNT(*) AS n FROM uploads"
  uv run scripts/sql_manager.py export --sql "SELECT * FROM ml_outcome_labels LIMIT 1000" --output data/ml_labels.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import asyncpg
import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")


def _require_database_url() -> str:
    dsn = (os.environ.get("DATABASE_URL") or "").strip()
    if not dsn:
        raise SystemExit("DATABASE_URL is required")
    return dsn


async def _fetch(dsn: str, sql: str) -> list[dict[str, Any]]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(sql)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _describe_table(dsn: str, table: str) -> list[dict[str, Any]]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT
                column_name,
                data_type,
                is_nullable,
                ordinal_position
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
            """,
            table,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def _print_json(rows: Sequence[dict[str, Any]], *, head: int = 20) -> None:
    out = list(rows[:head])
    print(json.dumps(out, indent=2, default=str))


def _write_output(rows: Sequence[dict[str, Any]], output: str) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(rows)
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix == ".jsonl":
        df.to_json(out_path, orient="records", lines=True)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise SystemExit("Output must end with .parquet, .jsonl, or .csv")
    print(f"Wrote {len(df)} rows to {out_path}")


async def _run(args: argparse.Namespace) -> None:
    dsn = _require_database_url()
    if args.command == "query":
        rows = await _fetch(dsn, args.sql)
        _print_json(rows, head=args.head)
        print(f"Rows: {len(rows)}")
        return
    if args.command == "describe":
        rows = await _describe_table(dsn, args.table)
        _print_json(rows, head=max(args.head, 200))
        print(f"Columns: {len(rows)}")
        return
    if args.command == "export":
        rows = await _fetch(dsn, args.sql)
        _write_output(rows, args.output)
        return
    raise SystemExit(f"Unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UploadM8 SQL export/query helper.")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Run SQL and print JSON preview.")
    q.add_argument("--sql", required=True)
    q.add_argument("--head", type=int, default=20)

    d = sub.add_parser("describe", help="Describe a DB table schema.")
    d.add_argument("--table", required=True)
    d.add_argument("--head", type=int, default=200)

    e = sub.add_parser("export", help="Run SQL and write output file.")
    e.add_argument("--sql", required=True)
    e.add_argument("--output", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
