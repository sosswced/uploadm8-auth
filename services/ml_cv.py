"""
Cross-validation + calibration helpers shared by the ML training scripts.

- ``grouped_oof_predict``: leakage-free out-of-fold probabilities grouped by
  ``user_id`` so a user's rows never span train and test (prevents the model
  from "memorizing" individual creators).
- ``fit_calibrated``: best-effort probability calibration so the recommended
  score thresholds consumed at runtime are meaningful, with a safe fall back to
  a plain fit on tiny / degenerate datasets.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def grouped_oof_predict(
    make_pipe: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cap: int = 5,
) -> Optional[pd.Series]:
    """Out-of-fold positive-class probabilities grouped by ``groups``.

    Returns a Series aligned to ``X.index`` (NaN where a fold was skipped) or
    ``None`` when grouped CV is not feasible (caller should fall back).
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)
    n_groups = int(g.nunique())
    if n_groups < 2:
        return None
    n_splits = max(2, min(int(cap), n_groups))
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    try:
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups=g):
            if int(y.iloc[tr].nunique()) < 2:
                continue
            pipe = make_pipe()
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    except Exception as e:  # noqa: BLE001
        logger.warning("grouped OOF prediction failed: %s", e)
        return None
    valid = oof.notna()
    if int(valid.sum()) < 2 or int(y[valid].nunique()) < 2:
        return None
    return oof


def grouped_oof_regress(
    make_pipe: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cap: int = 5,
) -> Optional[pd.Series]:
    """Out-of-fold continuous predictions grouped by ``groups`` (e.g. hotness_score)."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)
    n_groups = int(g.nunique())
    if n_groups < 2:
        return None
    n_splits = max(2, min(int(cap), n_groups))
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    try:
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups=g):
            reg = make_pipe()
            reg.fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[te] = reg.predict(X.iloc[te])
    except Exception as e:  # noqa: BLE001
        logger.warning("grouped OOF regression failed: %s", e)
        return None
    if int(oof.notna().sum()) < 2:
        return None
    return oof


def fit_calibrated(
    make_pipe: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cap: int = 3,
) -> Tuple[object, str]:
    """Return (fitted_estimator, calibration_method). Falls back to a plain fit."""
    from sklearn.calibration import CalibratedClassifierCV

    y = pd.Series(y)
    vc = y.value_counts()
    min_class = int(vc.min()) if len(vc) else 0
    if min_class >= 3:
        try:
            k = max(2, min(int(cap), min_class))
            method = "isotonic" if len(y) >= 200 else "sigmoid"
            cal = CalibratedClassifierCV(make_pipe(), method=method, cv=k)
            cal.fit(X, y)
            return cal, method
        except Exception as e:  # noqa: BLE001
            logger.warning("probability calibration failed, using uncalibrated: %s", e)
    pipe = make_pipe()
    pipe.fit(X, y)
    return pipe, "none"


def build_user_groups(df: pd.DataFrame, n: int) -> pd.Series:
    """User-id groups for CV; synthetic rows (no user_id) become singleton groups."""
    if "user_id" in df.columns:
        raw = df["user_id"].reset_index(drop=True).astype("object")
    else:
        raw = pd.Series([None] * n)
    return pd.Series(
        [g if (g is not None and not pd.isna(g)) else f"_seed{i}" for i, g in enumerate(raw)]
    )
