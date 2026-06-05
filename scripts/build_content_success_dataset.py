#!/usr/bin/env python
# /// script
# dependencies = [
#   "asyncpg>=0.29.0,<0.32.0",
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "datasets>=2.20.0,<4.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
#   "trackio>=0.25.0,<1.0.0",
# ]
# ///
"""
Build the content-success training dataset for the UploadM8 ML / AI engine.

Unlike the promo-targeting dataset (user-level aggregates → revenue uplift), this
emits **one row per (upload x platform)** so the model can learn which *topic*,
*content packaging*, and *upload-flow choices* drive the strongest engagement on
each platform.

Per-platform engagement (views / likes / comments / shares) is sourced from each
upload's ``platform_results`` JSON (the per-platform publish result), falling back
to the upload row columns for single-platform uploads. The upload-flow / topic
context comes from ``uploads.output_artifacts.content_attribution_v1`` (content
category, caption style/tone/voice, hashtags, thumbnail strategy, M8 engine) plus
the upload hashtags and schedule timing.

Feature engineering + the within-user "hotness" label live in
``services.content_success_features`` so they stay unit-testable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import asyncpg
import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import OptionalTrackioRun, hf_env_status, hf_write_token
from services.hf_dataset_export import coerce_dataframe_for_hf
from services.content_success_features import build_labeled_dataframe

UPLOADS_SQL = """
SELECT
    u.id            AS upload_id,
    u.user_id,
    u.created_at,
    u.platforms,
    u.hashtags,
    COALESCE(u.views, 0)    AS views,
    COALESCE(u.likes, 0)    AS likes,
    COALESCE(u.comments, 0) AS comments,
    COALESCE(u.shares, 0)   AS shares,
    u.platform_results,
    u.output_artifacts
FROM uploads u
WHERE u.created_at >= (NOW() - ($1::int || ' days')::interval)
  AND u.status IN ('completed', 'succeeded', 'partial')
ORDER BY u.created_at DESC
LIMIT $2::int
"""


def _require_database_url() -> str:
    dsn = (os.environ.get("DATABASE_URL") or "").strip()
    if not dsn:
        raise SystemExit("DATABASE_URL is required")
    return dsn


async def _fetch(dsn: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(UPLOADS_SQL, int(lookback_days), int(limit))
    finally:
        await conn.close()
    return [dict(r) for r in rows]


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return coerce_dataframe_for_hf(df)


def _write_local(df: pd.DataFrame, output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".parquet":
        raise SystemExit("Output must be .parquet")
    _prepare_df(df).to_parquet(out, index=False)
    print(f"Wrote {len(df)} (upload x platform) rows to {out}")


def _maybe_push_hf(df: pd.DataFrame, repo_id: str, split: str, private: bool) -> None:
    if df.empty:
        print(f"Skipping HF push to {repo_id}: dataset is empty")
        return
    ok, reason = hf_env_status(require_write_token=True)
    if not ok:
        raise SystemExit(f"HF env check failed: {reason}")
    token = hf_write_token()
    if not token:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for push")
    try:
        from datasets import Dataset
    except ImportError as e:
        raise SystemExit(
            "datasets package is required for --push-to (pip install 'datasets>=2.20.0')"
        ) from e
    ds = Dataset.from_pandas(_prepare_df(df), preserve_index=False)
    ds.push_to_hub(repo_id, token=token, split=split, private=private)
    print(f"Pushed {len(ds)} rows to {repo_id} ({split})")


async def _run(args: argparse.Namespace) -> None:
    track = OptionalTrackioRun("content_success_dataset_build")
    track.start(
        config={
            "lookback_days": int(args.lookback_days),
            "limit": int(args.limit),
            "target_loop": "content_success",
            "grain": "upload_x_platform",
        }
    )
    dsn = _require_database_url()
    upload_rows = await _fetch(dsn, lookback_days=args.lookback_days, limit=args.limit)
    df = build_labeled_dataframe(upload_rows)
    hot_rate = float(df["is_hot"].mean()) if len(df) else 0.0
    track.log(
        {
            "upload_rows": int(len(upload_rows)),
            "expanded_rows": int(len(df)),
            "hot_rate": hot_rate,
            "platforms": int(df["platform"].nunique()) if len(df) else 0,
        }
    )
    _write_local(df, args.output)
    if args.push_to and len(df):
        _maybe_push_hf(df, args.push_to, args.split, args.private)
    track.finish()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build content-success ML training dataset.")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--limit", type=int, default=250000)
    p.add_argument("--output", default="data/ml/content_success_train_v1.parquet")
    p.add_argument("--push-to", default="")
    p.add_argument("--split", default="content_success")
    p.add_argument("--private", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
