#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas>=2.0.0,<3.0.0",
#   "pyarrow>=15.0.0",
#   "scikit-learn>=1.8.0,<2.0.0",
#   "python-dotenv>=1.0.0,<2.0.0",
#   "joblib>=1.3.0",
# ]
# ///
"""Shadow AV-read distill — tabular sense+align student (never auto-wired to worker)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from services.av_read_features import CAT_FEATURES, NUM_FEATURES  # noqa: E402


def _band_to_int(band: Any) -> int:
    b = str(band or "mid").lower()
    return {"low": 0, "mid": 1, "high": 2}.get(b, 1)


def train(df: pd.DataFrame) -> Dict[str, Any]:
    from services.av_training_pack_quality import filter_pack_training_rows

    # Defense in depth: refuse low-agreement packs even if parquet was built unfiltered.
    raw_rows = df.to_dict(orient="records") if df is not None and not df.empty else []
    accepted, refused = filter_pack_training_rows(raw_rows)
    df = pd.DataFrame(accepted) if accepted else pd.DataFrame()
    refused_n = len(refused)

    if df.empty or len(df) < 8:
        return {
            "status": "insufficient_data",
            "train_rows": int(len(df)),
            "refused_rows": int(refused_n),
            "publish_status": "trained_not_published",
            "reason": "need_at_least_8_rows",
        }

    work = df.copy()
    work["grounding_score"] = pd.to_numeric(work.get("grounding_score"), errors="coerce")
    work["needs_deep_teacher"] = work["needs_deep_teacher"].astype(bool).astype(int)
    work["hero_class"] = work["identity_hero_fact_class"].fillna("unknown").astype(str)
    work["grounding_band_i"] = work["grounding_band"].map(_band_to_int)
    work["fusion_text"] = work.get("fusion_text", "").fillna("").astype(str)

    for c in NUM_FEATURES:
        if c not in work.columns:
            work[c] = 0
    for c in CAT_FEATURES:
        if c not in work.columns:
            work[c] = "unknown"

    # Cohort split for TL backup readiness reporting
    tl_ok = work[work["tl_status"].astype(str) == "ok"]
    fusion_backup = work[work["tl_status"].astype(str).isin(["skipped", "failed", "disabled"])]

    y_hero = work["hero_class"]
    y_deep = work["needs_deep_teacher"]
    y_band = work["grounding_band_i"]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=24)),
                    ]
                ),
                CAT_FEATURES,
            ),
        ]
    )

    # Hash fusion text lightly
    text_pipe = Pipeline(
        steps=[
            ("hash", HashingVectorizer(n_features=64, alternate_sign=False, norm=None)),
        ]
    )

    X_struct = pre.fit_transform(work)
    X_text = text_pipe.fit_transform(work["fusion_text"]).toarray()
    X = np.hstack([X_struct, X_text])

    report: Dict[str, Any] = {
        "status": "ok",
        "train_rows": int(len(work)),
        "refused_rows": int(refused_n),
        "publish_status": "trained_not_published",
        "features_num": NUM_FEATURES,
        "features_cat": CAT_FEATURES,
        "cohort": {
            "tl_ok_rows": int(len(tl_ok)),
            "fusion_backup_rows": int(len(fusion_backup)),
        },
    }

    # Held-out split when possible
    strat = y_hero if y_hero.nunique() > 1 and y_hero.value_counts().min() >= 2 else None
    try:
        X_tr, X_te, yh_tr, yh_te, yd_tr, yd_te, yb_tr, yb_te = train_test_split(
            X, y_hero, y_deep, y_band, test_size=0.25, random_state=42, stratify=strat
        )
    except ValueError:
        X_tr, X_te = X, X
        yh_tr, yh_te = y_hero, y_hero
        yd_tr, yd_te = y_deep, y_deep
        yb_tr, yb_te = y_band, y_band

    hero_clf = HistGradientBoostingClassifier(max_depth=4, max_iter=80, random_state=42)
    deep_clf = HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=42)
    band_clf = HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=42)

    hero_clf.fit(X_tr, yh_tr)
    deep_clf.fit(X_tr, yd_tr)
    band_clf.fit(X_tr, yb_tr)

    yh_pred = hero_clf.predict(X_te)
    yd_pred = deep_clf.predict(X_te)
    yb_pred = band_clf.predict(X_te)

    report["metrics"] = {
        "hero_class_macro_f1": float(f1_score(yh_te, yh_pred, average="macro", zero_division=0)),
        "needs_deep_accuracy": float(accuracy_score(yd_te, yd_pred)),
        "grounding_band_accuracy": float(accuracy_score(yb_te, yb_pred)),
    }
    try:
        if len(np.unique(yd_te)) > 1:
            proba = deep_clf.predict_proba(X_te)[:, 1]
            report["metrics"]["needs_deep_roc_auc"] = float(roc_auc_score(yd_te, proba))
    except Exception:
        pass

    # Hero-fact F1 vs teacher labels on held-out (same as hero_class_macro_f1 — teacher = pack labels)
    report["hero_fact_f1_vs_teacher"] = report["metrics"]["hero_class_macro_f1"]

    return report, {"hero": hero_clf, "deep": deep_clf, "band": band_clf, "pre": pre, "text": text_pipe}


def main() -> int:
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/ml/av_training_pack_v1.parquet")
    ap.add_argument("--report", type=str, default="data/ml/av_read_distill_report.json")
    ap.add_argument("--model", type=str, default="data/ml/av_read_distill_model.joblib")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        report = {
            "status": "insufficient_data",
            "train_rows": 0,
            "publish_status": "trained_not_published",
            "reason": f"missing_input:{path}",
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return 0

    df = pd.read_parquet(path)
    result = train(df)
    if isinstance(result, tuple):
        report, models = result
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, args.model)
        report["model_path"] = str(args.model)
    else:
        report = result

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "train_rows", "publish_status", "metrics", "cohort")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
