#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "scikit-learn>=1.4.0,<2.0.0",
#   "joblib>=1.3.0,<2.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
#   "trackio>=0.25.0,<1.0.0",
# ]
# ///
"""
Train and evaluate a minimal promo-targeting uplift baseline.

Input: parquet produced by build_promo_training_dataset.py
Output: JSON report with AUC/PR/lift@k and threshold suggestions; joblib model artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import OptionalTrackioRun
from services.promo_targeting_features import (
    FEATURES_CAT,
    FEATURES_NUM,
    TARGET_CONVERTED,
    TARGET_ENGAGED,
)
from services.promo_targeting_train import (
    build_promo_pipeline,
    pick_target_column,
    recommend_score_threshold,
)


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


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input dataset not found: {path}")

    df = pd.read_parquet(path)
    target_col, used_fallback = pick_target_column(df, TARGET_CONVERTED, TARGET_ENGAGED)
    if target_col not in df.columns:
        raise SystemExit(f"Missing target column: {target_col}")
    if len(df) < 8:
        raise SystemExit(f"Need more rows to train baseline (got {len(df)})")
    class_count = int(df[target_col].nunique())
    if class_count < 2:
        return {
            "task": "promo_targeting_uplift_baseline",
            "status": "insufficient_label_variance",
            "message": "Target has only one class; add more mixed conversion outcomes before supervised training.",
            "rows": int(len(df)),
            "positive_rate": float(df[target_col].mean()) if len(df) else 0.0,
            "unique_classes": class_count,
            "target_column": target_col,
        }

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )
    pipe = build_promo_pipeline()
    pipe.fit(X_train, y_train)
    p_test = pd.Series(pipe.predict_proba(X_test)[:, 1], index=y_test.index)

    auc = float(roc_auc_score(y_test, p_test))
    ap = float(average_precision_score(y_test, p_test))
    lift_10 = _safe_lift_at_k(y_test, p_test, 0.10)
    lift_20 = _safe_lift_at_k(y_test, p_test, 0.20)
    thresholds = _threshold_table(y_test, p_test)
    rec_threshold = recommend_score_threshold(y_test, p_test)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    report: Dict[str, Any] = {
        "task": "promo_targeting_uplift_baseline",
        "status": "ok",
        "model": "logistic_regression_balanced_v1",
        "target_column": target_col,
        "label_fallback_used": used_fallback,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rows": int(len(df)),
        "positive_rate": float(df[target_col].mean()),
        "base_positive_rate_test": float(y_test.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "lift_at_10pct": lift_10,
        "lift_at_20pct": lift_20,
        "threshold_suggestions": thresholds,
        "recommended_score_threshold": rec_threshold,
        "default_score_threshold": rec_threshold,
        "feature_columns": FEATURES_NUM + FEATURES_CAT,
        "model_path": str(model_path),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train promo uplift baseline model.")
    parser.add_argument("--input", default="data/ml/promo_targeting_train_v1.parquet")
    parser.add_argument("--report-out", default="data/ml/promo_targeting_baseline_report.json")
    parser.add_argument("--model-out", default="data/ml/promo_uplift_model.joblib")
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
    if report.get("status") == "ok":
        from services.promo_targeting_model import invalidate_cache

        invalidate_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
