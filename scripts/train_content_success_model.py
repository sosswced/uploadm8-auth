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
Train the content-success model and rank the hottest topics / content.

Input: parquet produced by ``build_content_success_dataset.py`` — one row per
(upload x platform) with per-platform engagement + upload-flow / topic features.

Output: JSON report with
  * a classifier (predicts "hot" content from upload-flow choices) — ROC-AUC / PR /
    lift@k, so it plugs into the same HF eval-results plumbing as the promo model;
  * ranked tables answering the product question directly — *what is the best, most
    successful and hottest topic / content* — by topic, hashtag, platform x topic,
    and packaging (caption style/tone/voice).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

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
from services.content_success_features import (
    FEATURES_CAT,
    FEATURES_NUM,
    TARGET,
    content_rankings,
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=40)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, FEATURES_NUM),
            ("cat", cat_pipe, FEATURES_CAT),
        ]
    )
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input dataset not found: {path}")

    df = pd.read_parquet(path)
    base: Dict[str, Any] = {
        "task": "content_success_hotness",
        "model": "logistic_regression_balanced_v1",
        "rows": int(len(df)),
    }
    if TARGET not in df.columns or len(df) < 8:
        base["status"] = "insufficient_rows"
        base["message"] = "Need more (upload x platform) rows with engagement before training."
        base["rankings"] = content_rankings(df) if len(df) else {}
        return base

    rankings = content_rankings(df)
    class_count = int(df[TARGET].nunique())
    if class_count < 2:
        base["status"] = "insufficient_label_variance"
        base["message"] = "Hotness target has only one class; publish more varied content first."
        base["positive_rate"] = float(df[TARGET].mean())
        base["rankings"] = rankings
        return base

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    p_test = pd.Series(pipe.predict_proba(X_test)[:, 1], index=y_test.index)

    report: Dict[str, Any] = {
        **base,
        "status": "ok",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "base_positive_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "average_precision": float(average_precision_score(y_test, p_test)),
        "lift_at_10pct": _safe_lift_at_k(y_test, p_test, 0.10),
        "lift_at_20pct": _safe_lift_at_k(y_test, p_test, 0.20),
        "feature_columns": FEATURES_NUM + FEATURES_CAT,
        "rankings": rankings,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train content-success / hottest-topic model.")
    parser.add_argument("--input", default="data/ml/content_success_train_v1.parquet")
    parser.add_argument("--report-out", default="data/ml/content_success_report.json")
    args = parser.parse_args()

    track = OptionalTrackioRun("content_success_train")
    track.start(config={"input": args.input})
    report = _run(args)
    track.log({k: v for k, v in report.items() if k != "rankings"})
    track.finish()

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rankings"}, indent=2))
    print(f"Report (with topic/content rankings) written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
