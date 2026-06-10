"""
Load and score the trained promo-uplift logistic model at runtime.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import asyncpg

from services.promo_targeting_features import (
    FEATURES_CAT,
    FEATURES_NUM,
    features_row_for_user,
)

logger = logging.getLogger("uploadm8.promo_targeting_model")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "ml" / "promo_uplift_model.joblib"
_DEFAULT_REPORT_PATH = _REPO_ROOT / "data" / "ml" / "promo_targeting_baseline_report.json"

_cache_lock = threading.Lock()
_cached: Dict[str, Any] = {
    "pipeline": None,
    "report": None,
    "model_mtime": None,
    "report_mtime": None,
}


def _model_path() -> Path:
    raw = (os.environ.get("UM8_PROMO_MODEL_PATH") or "").strip()
    return Path(raw) if raw else _DEFAULT_MODEL_PATH


def _report_path() -> Path:
    raw = (os.environ.get("UM8_ML_ENGINE_REPORT_PATH") or "").strip()
    return Path(raw) if raw else _DEFAULT_REPORT_PATH


def _mtime(p: Path) -> Optional[float]:
    try:
        return p.stat().st_mtime if p.is_file() else None
    except OSError:
        return None


def _load_report() -> Dict[str, Any]:
    p = _report_path()
    mt = _mtime(p)
    if _cached["report"] is not None and _cached["report_mtime"] == mt:
        return _cached["report"]
    if not p.is_file():
        _cached["report"] = {}
        _cached["report_mtime"] = mt
        return {}
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("promo report read failed: %s", e)
        rep = {}
    _cached["report"] = rep if isinstance(rep, dict) else {}
    _cached["report_mtime"] = mt
    return _cached["report"]


def reload_model() -> None:
    with _cache_lock:
        _cached["pipeline"] = None
        _cached["report"] = None
        _cached["model_mtime"] = None
        _cached["report_mtime"] = None


def _load_pipeline():
    import joblib

    p = _model_path()
    mt = _mtime(p)
    if _cached["pipeline"] is not None and _cached["model_mtime"] == mt:
        return _cached["pipeline"]
    if not p.is_file():
        _cached["pipeline"] = None
        _cached["model_mtime"] = mt
        return None
    try:
        pipe = joblib.load(p)
    except Exception as e:
        logger.warning("promo model load failed: %s", e)
        pipe = None
    _cached["pipeline"] = pipe
    _cached["model_mtime"] = mt
    return pipe


def invalidate_cache() -> None:
    with _cache_lock:
        _cached["pipeline"] = None
        _cached["report"] = None
        _cached["model_mtime"] = None
        _cached["report_mtime"] = None


def model_ready() -> bool:
    with _cache_lock:
        rep = _load_report()
        if rep.get("status") not in (None, "ok") and not rep.get("roc_auc"):
            if rep.get("status") == "insufficient_label_variance":
                return False
        pipe = _load_pipeline()
        return pipe is not None


def recommended_threshold() -> float:
    rep = _load_report()
    for key in ("recommended_score_threshold", "default_score_threshold"):
        try:
            return float(rep.get(key))
        except (TypeError, ValueError):
            pass
    for row in rep.get("threshold_suggestions") or []:
        try:
            if float(row.get("precision") or 0) >= 0.15:
                return float(row.get("threshold") or 0.4)
        except (TypeError, ValueError):
            continue
    return 0.4


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _toy_propensity_score(features: Dict[str, Any]) -> float:
    u = float(features.get("uploads_30d") or features.get("uploads_window") or 0)
    ctr = float(features.get("nudge_ctr_pct") or features.get("avg_engagement_pct_30d") or 0)
    rev = 1.0 if float(features.get("revenue_7d") or 0) > 0 else 0.0
    z = -1.2 + 0.04 * u + 0.03 * ctr + 1.1 * rev
    return _sigmoid(z)


def score_features(features: Dict[str, Any]) -> Tuple[float, str]:
    """
    Returns (probability in [0,1], scorer_key).
    """
    with _cache_lock:
        pipe = _load_pipeline()
    if pipe is not None:
        try:
            import pandas as pd

            cols = FEATURES_NUM + FEATURES_CAT
            row = {k: features.get(k) for k in cols}
            X = pd.DataFrame([row])
            prob = float(pipe.predict_proba(X)[0, 1])
            return max(0.0, min(1.0, prob)), "logistic_regression_balanced_v1"
        except Exception as e:
            logger.warning("promo model predict failed: %s", e)
    return _toy_propensity_score(features), "propensity_v1_toy"


_VIEW_SNAPSHOT_SQL = (
    "SELECT * FROM v_promo_targeting_features WHERE touchpoint_id = $1 LIMIT 1"
)


async def score_user_propensity(
    conn: asyncpg.Connection,
    user_id: str,
    *,
    subscription_tier: str,
    range_key: str = "30d",
    channel: str = "email",
) -> Tuple[float, str]:
    # Preferred path: the curated view's snapshot row gives exact train/serve
    # parity (same SQL the training dataset is built from).
    try:
        rec = await conn.fetchrow(_VIEW_SNAPSHOT_SQL, f"user:{user_id}")
    except Exception as e:  # noqa: BLE001 - view may be absent; fall back below
        logger.warning("promo view snapshot fetch failed: %s", e)
        rec = None
    if rec is not None:
        return score_features(dict(rec))

    # Fallback path: assemble from live campaign features.
    from services.wallet_marketing import _user_campaign_features

    feats = await _user_campaign_features(conn, user_id, range_key)
    row = features_row_for_user(
        subscription_tier=subscription_tier,
        campaign_features=feats,
        channel=channel,
    )
    w = await conn.fetchrow(
        "SELECT put_balance, aic_balance FROM wallets WHERE user_id = $1::uuid",
        user_id,
    )
    if w:
        row["put_balance"] = int(w["put_balance"] or 0)
        row["aic_balance"] = int(w["aic_balance"] or 0)
    return score_features(row)


def ml_targeting_enabled() -> bool:
    """When on, campaign/touchpoint sends respect trained propensity thresholds."""
    from core.pipeline_env_defaults import env_bool

    return env_bool("MARKETING_ML_TARGETING_ENABLED", default=True)
