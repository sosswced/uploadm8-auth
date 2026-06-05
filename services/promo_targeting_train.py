"""Training helpers shared by the promo uplift baseline script."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.promo_targeting_features import FEATURES_CAT, FEATURES_NUM


def build_promo_pipeline() -> Pipeline:
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


def pick_target_column(df: pd.DataFrame, primary: str, fallback: str) -> tuple[str, bool]:
    """Return (target_col, used_fallback)."""
    if primary in df.columns and int(df[primary].nunique()) >= 2:
        return primary, False
    if fallback in df.columns and int(df[fallback].nunique()) >= 2:
        return fallback, True
    return primary, False


def recommend_score_threshold(y_true: pd.Series, y_score: pd.Series) -> float:
    best_t = 0.4
    best_prec = -1.0
    for t in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55):
        pred = (y_score >= t).astype(int)
        positives = int(pred.sum())
        if positives <= 0:
            continue
        precision = float(y_true[pred == 1].mean())
        if precision > best_prec:
            best_prec = precision
            best_t = t
    return float(best_t)
