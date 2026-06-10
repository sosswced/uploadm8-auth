"""Training helpers shared by the promo uplift baseline script."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from services.promo_targeting_features import FEATURES_CAT, FEATURES_NUM

# Model identifier persisted in the report + m8_model_runs.
PROMO_MODEL_VERSION = "hist_gradient_boosting_v2"


def build_promo_pipeline() -> Pipeline:
    # Gradient-boosted trees handle NaN natively and need no scaling; categoricals
    # are one-hot encoded (dense, since HGB does not accept sparse input).
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("cat", cat_pipe, FEATURES_CAT),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
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
