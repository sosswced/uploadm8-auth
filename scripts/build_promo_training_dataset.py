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
Build first Phase 1 promo-targeting training dataset from UploadM8 tables.

Target loop: marketing/promo uplift.
Label: conversion in 7 days after touchpoint (based on revenue_tracking).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import asyncpg
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import OptionalTrackioRun, hf_env_status, hf_write_token
from services.hf_dataset_export import coerce_dataframe_for_hf


PROMO_SQL = """
WITH touchpoints AS (
    SELECT
        mtd.id AS touchpoint_id,
        mtd.user_id,
        mtd.channel,
        mtd.status AS delivery_status,
        mtd.created_at AS touchpoint_at,
        COALESCE(mtd.sent_at, mtd.created_at) AS effective_sent_at
    FROM marketing_touchpoint_deliveries mtd
    WHERE mtd.created_at >= (NOW() - ($1::int || ' days')::interval)
),
user_uploads_30d AS (
    SELECT
        u.user_id,
        COUNT(*)::int AS uploads_30d,
        AVG(COALESCE(u.views, 0))::double precision AS avg_views_30d,
        AVG(
            CASE
                WHEN COALESCE(u.views, 0) > 0
                THEN ((COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::double precision / u.views::double precision) * 100.0
                ELSE 0.0
            END
        )::double precision AS avg_engagement_pct_30d
    FROM uploads u
    WHERE u.created_at >= (NOW() - INTERVAL '30 days')
    GROUP BY u.user_id
),
platform_perf_30d AS (
    SELECT
        pci.user_id,
        COUNT(*)::int AS content_items_30d,
        AVG(COALESCE(pci.views, 0))::double precision AS pci_avg_views_30d
    FROM platform_content_items pci
    WHERE pci.published_at >= (NOW() - INTERVAL '30 days')
    GROUP BY pci.user_id
),
label_7d AS (
    SELECT
        tp.touchpoint_id,
        CASE WHEN EXISTS (
            SELECT 1
            FROM revenue_tracking rt
            WHERE rt.user_id = tp.user_id
              AND rt.created_at >= tp.effective_sent_at
              AND rt.created_at < tp.effective_sent_at + INTERVAL '7 days'
              AND COALESCE(rt.amount, 0) > 0
        ) THEN 1 ELSE 0 END AS converted_7d,
        COALESCE((
            SELECT SUM(COALESCE(rt2.amount, 0))::double precision
            FROM revenue_tracking rt2
            WHERE rt2.user_id = tp.user_id
              AND rt2.created_at >= tp.effective_sent_at
              AND rt2.created_at < tp.effective_sent_at + INTERVAL '7 days'
        ), 0.0) AS revenue_7d
    FROM touchpoints tp
)
SELECT
    tp.touchpoint_id,
    tp.user_id,
    tp.channel,
    tp.delivery_status,
    tp.touchpoint_at,
    EXTRACT(DOW FROM tp.effective_sent_at)::int AS sent_dow_utc,
    EXTRACT(HOUR FROM tp.effective_sent_at)::int AS sent_hour_utc,
    COALESCE(w.put_balance, 0)::int AS put_balance,
    COALESCE(w.aic_balance, 0)::int AS aic_balance,
    COALESCE(u.subscription_tier, 'free') AS subscription_tier,
    COALESCE(uu.uploads_30d, 0)::int AS uploads_30d,
    COALESCE(uu.avg_views_30d, 0)::double precision AS avg_views_30d,
    COALESCE(uu.avg_engagement_pct_30d, 0)::double precision AS avg_engagement_pct_30d,
    COALESCE(pp.content_items_30d, 0)::int AS content_items_30d,
    COALESCE(pp.pci_avg_views_30d, 0)::double precision AS pci_avg_views_30d,
    l7.converted_7d,
    l7.revenue_7d
FROM touchpoints tp
LEFT JOIN users u ON u.id = tp.user_id
LEFT JOIN wallets w ON w.user_id = tp.user_id
LEFT JOIN user_uploads_30d uu ON uu.user_id = tp.user_id
LEFT JOIN platform_perf_30d pp ON pp.user_id = tp.user_id
LEFT JOIN label_7d l7 ON l7.touchpoint_id = tp.touchpoint_id
ORDER BY tp.touchpoint_at DESC
LIMIT $2::int
"""

FALLBACK_PROMO_SQL = """
SELECT
    mol.id AS touchpoint_id,
    mol.user_id,
    COALESCE(mol.label_json->>'channel', 'ml_label') AS channel,
    COALESCE(mol.label_json->>'event', 'ml_outcome_label') AS delivery_status,
    mol.created_at AS touchpoint_at,
    EXTRACT(DOW FROM mol.created_at)::int AS sent_dow_utc,
    EXTRACT(HOUR FROM mol.created_at)::int AS sent_hour_utc,
    COALESCE(w.put_balance, 0)::int AS put_balance,
    COALESCE(w.aic_balance, 0)::int AS aic_balance,
    COALESCE(u.subscription_tier, 'free') AS subscription_tier,
    COALESCE((mol.feature_snapshot->>'uploads_window')::int, 0)::int AS uploads_30d,
    COALESCE((mol.feature_snapshot->>'avg_views_30d')::double precision, 0)::double precision AS avg_views_30d,
    COALESCE((mol.feature_snapshot->>'engagement_30d')::double precision, 0)::double precision AS avg_engagement_pct_30d,
    COALESCE((mol.feature_snapshot->>'content_items_30d')::int, 0)::int AS content_items_30d,
    COALESCE((mol.feature_snapshot->>'pci_avg_views_30d')::double precision, 0)::double precision AS pci_avg_views_30d,
    CASE
      WHEN COALESCE((mol.label_json->>'conversion')::double precision, 0) > 0 THEN 1
      WHEN COALESCE((mol.label_json->>'revenue_7d')::double precision, 0) > 0 THEN 1
      WHEN COALESCE(mol.label_json->>'selected_variant', '') <> '' THEN 1
      ELSE 0
    END AS converted_7d,
    GREATEST(
      COALESCE((mol.label_json->>'revenue_7d')::double precision, 0),
      COALESCE((mol.label_json->>'conversion')::double precision, 0)
    )::double precision AS revenue_7d
FROM ml_outcome_labels mol
LEFT JOIN users u ON u.id = mol.user_id
LEFT JOIN wallets w ON w.user_id = mol.user_id
WHERE mol.created_at >= (NOW() - ($1::int || ' days')::interval)
ORDER BY mol.created_at DESC
LIMIT $2::int
"""


def _require_database_url() -> str:
    dsn = (os.environ.get("DATABASE_URL") or "").strip()
    if not dsn:
        raise SystemExit("DATABASE_URL is required")
    return dsn


async def _fetch_dataset(dsn: str, lookback_days: int, limit: int) -> pd.DataFrame:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(PROMO_SQL, int(lookback_days), int(limit))
        if not rows:
            rows = await conn.fetch(FALLBACK_PROMO_SQL, int(lookback_days), int(limit))
    finally:
        await conn.close()
    return pd.DataFrame.from_records([dict(r) for r in rows])


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return coerce_dataframe_for_hf(df)


def _write_local(df: pd.DataFrame, output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".parquet":
        raise SystemExit("Output must be .parquet for phase-1 baseline")
    df = _prepare_df(df)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


def _maybe_push_hf(df: pd.DataFrame, repo_id: str, split: str, private: bool) -> None:
    ok, reason = hf_env_status(require_write_token=True)
    if not ok:
        raise SystemExit(f"HF env check failed: {reason}")
    token = hf_write_token()
    if not token:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for push")
    ds = Dataset.from_pandas(_prepare_df(df), preserve_index=False)
    ds.push_to_hub(repo_id, token=token, split=split, private=private)
    print(f"Pushed {len(ds)} rows to {repo_id} ({split})")


async def _run(args: argparse.Namespace) -> None:
    track = OptionalTrackioRun("promo_targeting_dataset_build")
    track.start(
        config={
            "lookback_days": int(args.lookback_days),
            "limit": int(args.limit),
            "target_loop": "promo_targeting",
        }
    )
    dsn = _require_database_url()
    df = await _fetch_dataset(dsn, lookback_days=args.lookback_days, limit=args.limit)
    pos_rate = float(df["converted_7d"].mean()) if len(df) else 0.0
    track.log({"rows": int(len(df)), "positive_rate": pos_rate})
    _write_local(df, args.output)
    if args.push_to:
        _maybe_push_hf(df, args.push_to, args.split, args.private)
    track.finish()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build promo-targeting ML training dataset.")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--limit", type=int, default=250000)
    p.add_argument("--output", default="data/ml/promo_targeting_train_v1.parquet")
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
