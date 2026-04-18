"""
M8 publish-hour prior model: HistGradientBoostingRegressor on PCI (strict) or PCI+uploads.

Default training uses **PCI ``published_at`` only** (UTC hour/DOW), optional allowlists
for ``source`` and ``content_kind``. Target is ``log1p(views)`` on catalog items.

Each training run inserts one row into ``m8_model_runs`` (metrics, SHAP summary,
binned calibration on the validation split) then replaces ``m8_publish_hour_priors``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("uploadm8.m8_publish_hour")

# Short-form platforms aligned with smart scheduling defaults.
M8_PRIOR_PLATFORMS: frozenset[str] = frozenset({"tiktok", "youtube", "instagram", "facebook"})

_MIN_TOTAL_ROWS = 800
_MIN_PLATFORM_ROWS = 80


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _parse_csv_lower(name: str) -> Optional[List[str]]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out or None


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    np.exp(x, out=x)
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    x /= s
    return x


def _calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Sort by predicted log1p(views); report mean actual vs mean pred per bin."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    n = len(y_t)
    if n < n_bins * 3:
        n_bins = max(3, min(n_bins, n // 3))
    if n_bins < 2 or n < 4:
        return {"calibration_bins": [], "note": "too_few_val_rows"}
    order = np.argsort(y_p)
    bins_out: List[Dict[str, Any]] = []
    for b in range(n_bins):
        lo = int(b * n / n_bins)
        hi = int((b + 1) * n / n_bins) if b < n_bins - 1 else n
        sl = order[lo:hi]
        if len(sl) == 0:
            continue
        bins_out.append(
            {
                "pred_mean": float(y_p[sl].mean()),
                "actual_mean": float(y_t[sl].mean()),
                "mean_residual": float((y_t[sl] - y_p[sl]).mean()),
                "n": int(len(sl)),
            }
        )
    return {"calibration_bins": bins_out}


def _shap_tree_summary(
    model: Any,
    X_sample: np.ndarray,
    feature_cols: List[str],
    *,
    max_samples: int = 2000,
) -> Dict[str, Any]:
    """Mean |SHAP| per feature on a training subsample (TreeExplainer)."""
    try:
        import shap
    except ImportError as e:
        return {"shap": "skipped", "reason": f"import_error:{e}"}

    try:
        n = min(max_samples, len(X_sample))
        if n < 50:
            return {"shap": "skipped", "reason": "too_few_rows"}
        xs = X_sample[:n]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(xs)
        sv = np.asarray(sv, dtype=np.float64)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)
        mas = np.abs(sv).mean(axis=0)
        pairs = sorted(
            ((feature_cols[i], float(mas[i])) for i in range(min(len(feature_cols), len(mas)))),
            key=lambda x: -x[1],
        )
        return {
            "shap_mean_abs": {k: v for k, v in pairs},
            "shap_top_features": [k for k, _ in pairs[:12]],
            "shap_n_samples": int(n),
        }
    except Exception as e:
        logger.warning("m8_train: SHAP failed: %s", e)
        return {"shap": "error", "detail": str(e)[:500]}


def _build_training_query(
    *,
    pci_only: bool,
    source_allow: Optional[List[str]],
    kind_allow: Optional[List[str]],
) -> Tuple[str, List[Any]]:
    """Returns (sql, [lookback_interval_str, ...])."""
    lookback_placeholder = "$1::text"
    params: List[Any] = []

    if pci_only:
        ts = "pci.published_at"
        where_time = f"""
            {ts} IS NOT NULL
            AND {ts} >= (timezone('utc', now()) - ({lookback_placeholder})::interval)
            AND {ts} < timezone('utc', now())
        """
    else:
        ts = "COALESCE(pci.published_at, u.completed_at, u.created_at)"
        where_time = f"""
            {ts} IS NOT NULL
            AND {ts} >= (timezone('utc', now()) - ({lookback_placeholder})::interval)
            AND {ts} < timezone('utc', now())
        """

    extra = ""
    pidx = 2
    if source_allow:
        extra += f" AND lower(trim(pci.source::text)) = ANY(${pidx}::text[])"
        params.append(source_allow)
        pidx += 1
    if kind_allow:
        extra += f" AND lower(trim(COALESCE(pci.content_kind::text, ''))) = ANY(${pidx}::text[])"
        params.append(kind_allow)
        pidx += 1

    sql = f"""
        SELECT
            lower(trim(pci.platform::text)) AS platform,
            (EXTRACT(DOW FROM timezone('UTC', {ts})))::int AS dow,
            (EXTRACT(HOUR FROM timezone('UTC', {ts})))::int AS hr,
            GREATEST(COALESCE(pci.views, 0), 0)::bigint AS views,
            GREATEST(COALESCE(pci.likes, 0), 0)::bigint AS likes,
            GREATEST(COALESCE(pci.duration_seconds, 0), 0)::int AS duration_seconds,
            COALESCE(lower(pci.source::text), '') AS source,
            COALESCE(lower(pci.content_kind::text), '') AS content_kind
        FROM platform_content_items pci
        LEFT JOIN uploads u ON u.id = pci.upload_id AND u.user_id = pci.user_id
        WHERE {where_time}
        {extra}
    """
    return sql, params


def _build_frame(rows: Sequence[Any]):
    import pandas as pd

    recs: List[dict] = []
    for r in rows:
        p = (r["platform"] or "").strip().lower()
        if p not in M8_PRIOR_PLATFORMS:
            continue
        hr = int(r["hr"])
        dow = int(r["dow"])
        if hr < 0 or hr > 23 or dow < 0 or dow > 6:
            continue
        v = int(r["views"] or 0)
        likes = int(r["likes"] or 0)
        dur = int(r["duration_seconds"] or 0)
        recs.append(
            {
                "platform": p,
                "hr": hr,
                "dow": dow,
                "log_dur": float(np.log1p(max(dur, 0))),
                "log_likes": float(np.log1p(max(likes, 0))),
                "y": float(np.log1p(max(v, 0))),
            }
        )
    return pd.DataFrame.from_records(recs)


def _fit_and_predict_hour_grid(df) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Returns (platform -> 24 prior weights, metrics including SHAP + calibration)."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    if df is None or len(df) < _MIN_TOTAL_ROWS:
        return {}, {"skipped": True, "reason": "insufficient_rows", "n_rows": 0 if df is None else len(df)}

    plat_d = pd.get_dummies(df["platform"], prefix="plat")
    X = pd.concat([df[["hr", "dow", "log_dur", "log_likes"]], plat_d], axis=1)
    y = df["y"].values
    feature_cols = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y, test_size=0.15, random_state=42, shuffle=True
    )
    model = HistGradientBoostingRegressor(
        max_depth=10,
        learning_rate=0.07,
        max_iter=280,
        min_samples_leaf=25,
        l2_regularization=1e-3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_hat = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_hat))

    cal = _calibration_bins(y_val, y_hat, n_bins=10)
    shap_info = _shap_tree_summary(model, X_train, feature_cols, max_samples=2000)

    priors: Dict[str, List[float]] = {}
    for plat in sorted(M8_PRIOR_PLATFORMS):
        sub = df[df["platform"] == plat]
        if len(sub) < _MIN_PLATFORM_ROWS:
            logger.info("m8_train: skip platform=%s (n=%s)", plat, len(sub))
            continue
        med = sub.median(numeric_only=True)
        dow_med = int(round(float(med["dow"])))
        dow_med = max(0, min(6, dow_med))
        log_dur = float(med["log_dur"])
        log_likes = float(med["log_likes"])

        grid_rows: List[dict] = []
        for h in range(24):
            row = {c: 0.0 for c in feature_cols}
            row["hr"] = h
            row["dow"] = dow_med
            row["log_dur"] = log_dur
            row["log_likes"] = log_likes
            for c in feature_cols:
                if c.startswith("plat_"):
                    row[c] = 1.0 if c == f"plat_{plat}" else 0.0
            grid_rows.append(row)
        X_grid = pd.DataFrame(grid_rows)[feature_cols]
        pred = model.predict(X_grid.values)
        w = _softmax_np(pred).tolist()
        priors[plat] = [float(x) for x in w]

    metrics: Dict[str, Any] = {
        "val_mae_log1p_views": mae,
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "platforms_written": sorted(priors.keys()),
        **cal,
        **shap_info,
    }
    return priors, metrics


async def train_m8_publish_hour_priors(pool: Any, *, lookback_days: int = 420) -> Dict[str, Any]:
    """
    Pull PCI training data, fit HGBR, insert ``m8_model_runs``, replace hour priors.

    Env:
      M8_TRAIN_PCI_ONLY (default true) — require ``pci.published_at``; if false, use
        COALESCE(pci.published_at, u.completed_at, u.created_at).
      M8_TRAIN_SOURCE_ALLOWLIST — e.g. ``uploadm8,linked`` (comma-separated, lower).
      M8_TRAIN_CONTENT_KIND_ALLOWLIST — e.g. ``reel,short`` (optional).
    """
    from core.scheduling import static_hour_prior_24

    pci_only = _env_bool("M8_TRAIN_PCI_ONLY", True)
    source_allow = _parse_csv_lower("M8_TRAIN_SOURCE_ALLOWLIST")
    kind_allow = _parse_csv_lower("M8_TRAIN_CONTENT_KIND_ALLOWLIST")
    lookback = f"{max(30, int(lookback_days))} days"

    sql, extra_params = _build_training_query(
        pci_only=pci_only,
        source_allow=source_allow,
        kind_allow=kind_allow,
    )
    fetch_params: List[Any] = [lookback] + extra_params

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *fetch_params)

    df = _build_frame(rows)
    train_config: Dict[str, Any] = {
        "lookback_days": int(lookback_days),
        "pci_only": pci_only,
        "source_allowlist": source_allow,
        "content_kind_allowlist": kind_allow,
        "sql_row_count": len(rows),
        "frame_row_count": int(len(df)),
    }

    priors: Dict[str, List[float]]
    model_version: str
    mae: Optional[float]
    run_metrics: Dict[str, Any]

    if len(df) < _MIN_TOTAL_ROWS:
        logger.warning(
            "m8_train: only %s rows (min %s) — writing static priors",
            len(df),
            _MIN_TOTAL_ROWS,
        )
        priors = {p: static_hour_prior_24(p) for p in sorted(M8_PRIOR_PLATFORMS)}
        run_metrics = {"skipped": True, "reason": "insufficient_rows", "n_rows": len(df), "fallback": "static"}
        model_version = "static-fallback"
        mae = None
    else:
        priors, run_metrics = _fit_and_predict_hour_grid(df)
        model_version = "hgb-v1"
        mae = run_metrics.get("val_mae_log1p_views")
        for p in sorted(M8_PRIOR_PLATFORMS):
            if p not in priors:
                priors[p] = static_hour_prior_24(p)
                run_metrics.setdefault("filled_static", []).append(p)

    run_id = uuid.uuid4()
    trained_at = datetime.now(timezone.utc)

    # Persist run (SHAP + calibration live under metrics JSONB).
    features_used: List[str] = []
    if len(df) >= _MIN_TOTAL_ROWS:
        import pandas as pd

        plat_d = pd.get_dummies(df["platform"], prefix="plat")
        X = pd.concat([df[["hr", "dow", "log_dur", "log_likes"]], plat_d], axis=1)
        features_used = list(X.columns)

    insert_run_sql = """
        INSERT INTO m8_model_runs (
            id, trained_at, model_version, train_row_count, val_mae_log1p_views,
            features_used, train_config, metrics
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb)
    """
    run_metrics_out = dict(run_metrics)
    run_metrics_out["training_run_id"] = str(run_id)
    run_metrics_out["trained_at"] = trained_at.isoformat()
    run_metrics_out["model_version"] = model_version

    insert_prior_sql = """
        INSERT INTO m8_publish_hour_priors (
            platform, hour_utc, prior_weight, trained_at, training_run_id,
            train_row_count, model_version, val_mae_log1p_views
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    records: List[tuple] = []
    for plat, weights in priors.items():
        for h in range(24):
            records.append(
                (
                    plat,
                    h,
                    float(weights[h]),
                    trained_at,
                    run_id,
                    int(len(df)),
                    model_version,
                    float(mae) if mae is not None else None,
                )
            )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM m8_publish_hour_priors")
            await conn.execute(
                insert_run_sql,
                run_id,
                trained_at,
                model_version,
                int(len(df)),
                mae,
                json.dumps(features_used),
                json.dumps(train_config, default=str),
                json.dumps(run_metrics_out, default=str),
            )
            await conn.executemany(insert_prior_sql, records)

    logger.info("m8_train: finished run_id=%s %s", run_id, run_metrics_out)
    return run_metrics_out


def training_lookback_days_from_env() -> int:
    raw = os.environ.get("M8_TRAIN_LOOKBACK_DAYS", "420")
    try:
        return max(30, int(raw))
    except ValueError:
        return 420
