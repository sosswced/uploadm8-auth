#!/usr/bin/env python
# /// script
# dependencies = [
#   "asyncpg>=0.29.0,<0.32.0",
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "datasets>=2.20.0,<4.0.0",
#   "huggingface_hub>=1.10.0,<2.0.0",
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

import logging

from services.ml_observability import OptionalTrackioRun, hf_env_status, hf_write_token
from services.hf_dataset_export import coerce_dataframe_for_hf, push_dataframe_to_hub
from services.content_success_features import build_labeled_dataframe

logger = logging.getLogger(__name__)
from services.ml_feature_registry import active_cat, active_num

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


CONTENT_BASE_SQL = """
SELECT *
FROM v_content_success_base
WHERE upload_id = ANY($1::uuid[])
"""


async def _merge_base_features(dsn: str, df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge curated per-(upload x platform) signals (age, duration, recognition)."""
    if df.empty or "upload_id" not in df.columns:
        return df
    upload_ids = [str(x) for x in df["upload_id"].dropna().unique().tolist()]
    if not upload_ids:
        return df
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(CONTENT_BASE_SQL, upload_ids)
    except Exception as e:  # noqa: BLE001 - view may not exist yet; keep base df
        print(f"Skipping content base-feature merge: {e}")
        return df
    finally:
        await conn.close()
    if not rows:
        return df
    base = pd.DataFrame.from_records([dict(r) for r in rows])
    if base.empty:
        return df
    base["upload_id"] = base["upload_id"].astype(str)
    base["platform"] = base["platform"].astype(str).str.lower()
    df = df.copy()
    df["upload_id"] = df["upload_id"].astype(str)
    df["platform"] = df["platform"].astype(str).str.lower()
    # Avoid clobbering existing columns (e.g. platform) on merge.
    overlap = [c for c in base.columns if c in df.columns and c not in ("upload_id", "platform")]
    base = base.drop(columns=overlap)
    return df.merge(base, on=["upload_id", "platform"], how="left")


def _finalize_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive age-normalized + text-shape features and guarantee all model columns."""
    if df.empty:
        return df
    df = df.copy()

    age = pd.to_numeric(df.get("age_days"), errors="coerce") if "age_days" in df.columns else None
    views = pd.to_numeric(df.get("views"), errors="coerce").fillna(0.0)
    if age is not None:
        df["views_per_day"] = views / age.clip(lower=1.0)
    else:
        df["views_per_day"] = views  # no published_at available; fall back to raw views

    if "title" in df.columns:
        df["title_len"] = df["title"].fillna("").astype(str).str.len()
    if "caption" in df.columns:
        df["caption_len"] = df["caption"].fillna("").astype(str).str.len()

    # Recognition booleans → ints for the numeric pipeline.
    if "has_people" in df.columns:
        df["has_people"] = df["has_people"].fillna(False).astype(int)

    # Guarantee every active model column exists (cold DB / missing view → NaN).
    for col in active_num("content"):
        if col not in df.columns:
            df[col] = float("nan")
    for col in active_cat("content"):
        if col not in df.columns:
            df[col] = None

    # Drop raw text we only needed for lengths.
    return df.drop(columns=[c for c in ("title", "caption") if c in df.columns])


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
    result = push_dataframe_to_hub(
        _prepare_df(df),
        repo_id=repo_id,
        token=token,
        split=split,
        private=private,
    )
    print(
        f"Pushed {result.get('rows', 0)} rows to {repo_id} ({split}) via {result.get('mode')}"
    )


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
    df = await _merge_base_features(dsn, df)
    df = _finalize_content_features(df)
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
        try:
            _maybe_push_hf(df, args.push_to, args.split, args.private)
        except Exception as e:
            logger.warning("HF push failed after local write (continuing): %s", e)
            print(f"WARNING: HF push failed after local write: {e}")
    track.finish()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build content-success ML training dataset.")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--limit", type=int, default=250000)
    p.add_argument("--output", default="data/ml/content_success_train_v1.parquet")
    p.add_argument("--push-to", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--private", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
