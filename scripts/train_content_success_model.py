#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "scikit-learn>=1.8.0,<2.0.0",
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.ml_observability import OptionalTrackioRun
from services.ml_cv import build_user_groups, grouped_oof_predict, grouped_oof_regress
from services.content_success_features import (
    FEATURES_CAT,
    FEATURES_NUM,
    TARGET,
    content_rankings,
)

CONTENT_MODEL_VERSION = "hist_gradient_boosting_v2"


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


def _build_preprocessor() -> ColumnTransformer:
    # Trees handle NaN natively; categoricals one-hot encoded (dense for HGB).
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=40)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("cat", cat_pipe, FEATURES_CAT),
        ]
    )


def _build_pipeline() -> Pipeline:
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, l2_regularization=1.0,
        class_weight="balanced", random_state=42,
    )
    return Pipeline(steps=[("pre", _build_preprocessor()), ("clf", clf)])


def _build_regressor() -> Pipeline:
    reg = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=42,
    )
    return Pipeline(steps=[("pre", _build_preprocessor()), ("reg", reg)])


def _ndcg_at_k(relevance: pd.Series, scores: pd.Series, k: int) -> float:
    rel = relevance.to_numpy(dtype=float).reshape(1, -1)
    sc = scores.to_numpy(dtype=float).reshape(1, -1)
    if rel.shape[1] < 2:
        return 0.0
    kk = min(int(k), rel.shape[1])
    try:
        return float(ndcg_score(rel, sc, k=kk))
    except Exception:
        return 0.0


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input dataset not found: {path}")

    df = pd.read_parquet(path)
    base: Dict[str, Any] = {
        "task": "content_success_hotness",
        "model": CONTENT_MODEL_VERSION,
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

    # Guarantee every model column exists even if a fallback build path omitted some.
    X = df.reindex(columns=FEATURES_NUM + FEATURES_CAT).reset_index(drop=True).copy()
    y = df[TARGET].astype(int).reset_index(drop=True).copy()
    groups = build_user_groups(df, len(df))

    # Leakage-free evaluation: out-of-fold predictions grouped by user.
    oof = grouped_oof_predict(_build_pipeline, X, y, groups)
    if oof is not None:
        cv_mode = "group_kfold_user"
        mask = oof.notna()
        y_eval = y[mask]
        p_eval = oof[mask]
    else:
        cv_mode = "holdout_stratified"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        pipe = _build_pipeline()
        pipe.fit(X_train, y_train)
        y_eval = y_test
        p_eval = pd.Series(pipe.predict_proba(X_test)[:, 1], index=y_test.index)

    # Regression head on continuous hotness_score → ranking quality (NDCG@k),
    # since "hottest content" is fundamentally a ranking problem.
    ndcg_block: Dict[str, Any] = {}
    if "hotness_score" in df.columns:
        hot = pd.to_numeric(df["hotness_score"], errors="coerce").fillna(0.0).reset_index(drop=True)
        oof_reg = grouped_oof_regress(_build_regressor, X, hot, groups)
        if oof_reg is not None:
            m = oof_reg.notna()
            ndcg_block = {
                "regression_cv_mode": "group_kfold_user",
                "regression_eval_rows": int(m.sum()),
                "ndcg_at_10": _ndcg_at_k(hot[m], oof_reg[m], 10),
                "ndcg_at_20": _ndcg_at_k(hot[m], oof_reg[m], 20),
            }

    final_model = _build_pipeline()
    final_model.fit(X, y)

    report: Dict[str, Any] = {
        **base,
        "status": "ok",
        "cv_mode": cv_mode,
        "n_user_groups": int(groups.nunique()),
        "train_rows": int(len(X)),
        "eval_rows": int(len(y_eval)),
        "base_positive_rate_test": float(y_eval.mean()),
        "roc_auc": float(roc_auc_score(y_eval, p_eval)),
        "average_precision": float(average_precision_score(y_eval, p_eval)),
        "lift_at_10pct": _safe_lift_at_k(y_eval, p_eval, 0.10),
        "lift_at_20pct": _safe_lift_at_k(y_eval, p_eval, 0.20),
        **ndcg_block,
        "feature_columns": FEATURES_NUM + FEATURES_CAT,
        "rankings": rankings,
        "final_model": final_model,
    }
    return report


def main() -> int:
    import joblib

    parser = argparse.ArgumentParser(description="Train content-success / hottest-topic model.")
    parser.add_argument("--input", default="data/ml/content_success_train_v1.parquet")
    parser.add_argument("--report-out", default="data/ml/content_success_report.json")
    parser.add_argument("--model-out", default="data/ml/content_success_model.joblib")
    args = parser.parse_args()

    track = OptionalTrackioRun("content_success_train")
    track.start(config={"input": args.input})
    report = _run(args)
    track.log({k: v for k, v in report.items() if k != "rankings"})
    track.finish()

    model = report.pop("final_model", None)
    if model is not None and report.get("status") == "ok":
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rankings"}, indent=2))
    print(f"Report (with topic/content rankings) written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
