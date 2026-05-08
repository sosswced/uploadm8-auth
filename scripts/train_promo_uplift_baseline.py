#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "scikit-learn>=1.4.0,<2.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
#   "trackio>=0.25.0,<1.0.0",
# ]
# ///
"""
Train and evaluate a minimal promo-targeting uplift baseline.

Input: parquet produced by build_promo_training_dataset.py
Output: JSON report with AUC/PR/lift@k and threshold suggestions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import OptionalTrackioRun


FEATURES_NUM = [
    "sent_dow_utc",
    "sent_hour_utc",
    "put_balance",
    "aic_balance",
    "uploads_30d",
    "avg_views_30d",
    "avg_engagement_pct_30d",
    "content_items_30d",
    "pci_avg_views_30d",
]

FEATURES_CAT = [
    "channel",
    "delivery_status",
    "subscription_tier",
]

TARGET = "converted_7d"


def _safe_lift_at_k(y_true: pd.Series, y_score: pd.Series, k: float) -> float:
    if len(y_true) == 0:
        return 0.0
    base = float(y_true.mean())
    if base <= 0:
        return 0.0
    n_top = max(1, int(len(y_true) * k))
    top_idx = y_score.sort_values(ascending=False).head(n_top).index
    top_rate = float(y_true.loc[top_idx].mean()) if n_top > 0 else 0.0
    return float(top_rate / base) if base > 0 else 0.0


def _threshold_table(y_true: pd.Series, y_score: pd.Series) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        pred = (y_score >= t).astype(int)
        positives = int(pred.sum())
        if positives <= 0:
            precision = 0.0
        else:
            precision = float(y_true[pred == 1].mean())
        out.append(
            {
                "threshold": t,
                "predicted_positive_count": positives,
                "precision": precision,
            }
        )
    return out


def _build_pipeline() -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, FEATURES_NUM),
            ("cat", cat_pipe, FEATURES_CAT),
        ]
    )
    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input dataset not found: {path}")

    df = pd.read_parquet(path)
    if TARGET not in df.columns:
        raise SystemExit(f"Missing target column: {TARGET}")
    if len(df) < 8:
        raise SystemExit(f"Need more rows to train baseline (got {len(df)})")
    class_count = int(df[TARGET].nunique())
    if class_count < 2:
        return {
            "task": "promo_targeting_uplift_baseline",
            "status": "insufficient_label_variance",
            "message": "Target has only one class; add more mixed conversion outcomes before supervised training.",
            "rows": int(len(df)),
            "positive_rate": float(df[TARGET].mean()) if len(df) else 0.0,
            "unique_classes": class_count,
        }

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    p_test = pd.Series(pipe.predict_proba(X_test)[:, 1], index=y_test.index)

    auc = float(roc_auc_score(y_test, p_test))
    ap = float(average_precision_score(y_test, p_test))
    lift_10 = _safe_lift_at_k(y_test, p_test, 0.10)
    lift_20 = _safe_lift_at_k(y_test, p_test, 0.20)
    thresholds = _threshold_table(y_test, p_test)

    report: Dict[str, Any] = {
        "task": "promo_targeting_uplift_baseline",
        "model": "logistic_regression_balanced_v1",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "base_positive_rate_test": float(y_test.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "lift_at_10pct": lift_10,
        "lift_at_20pct": lift_20,
        "threshold_suggestions": thresholds,
        "feature_columns": FEATURES_NUM + FEATURES_CAT,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train promo uplift baseline model.")
    parser.add_argument("--input", default="data/ml/promo_targeting_train_v1.parquet")
    parser.add_argument("--report-out", default="data/ml/promo_targeting_baseline_report.json")
    args = parser.parse_args()

    track = OptionalTrackioRun("promo_targeting_baseline_train")
    track.start(config={"input": args.input})
    report = _run(args)
    track.log(report)
    track.finish()

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
