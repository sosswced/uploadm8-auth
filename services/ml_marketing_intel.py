"""
Bridge trained ML model outputs into the marketing / marketing-ops AI loop.

``build_ai_truth_metrics`` historically derived ``ml_truth.top_strategies`` from
hand-written heuristics. This module surfaces the *actual* trained models so the AI
marketing strategist (and the marketing ops page) can capitalize on them:

  * **Promo uplift** (``scripts/build_promo_training_dataset.py`` →
    ``train_promo_uplift_baseline.py``): ROC-AUC / lift / recommended decision
    threshold → how confidently we can target the top conversion decile.
  * **Content success / hottest topic** (the per upload x platform engagement loop):
    the ranked hottest topics, hashtags, platform x topic, and packaging.

Reads from ``m8_model_runs`` (written by ``services/ml_engine``) DB-first, with a
fallback to the local ``content_success_report.json`` so the panel still works
before the first DB-recorded run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uploadm8.ml_marketing_intel")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTENT_REPORT_PATH = _REPO_ROOT / "data" / "ml" / "content_success_report.json"

PROMO_TASK = "promo_targeting_uplift_baseline"
CONTENT_TASK = "content_success_hotness"


def _as_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _f(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return round(float(v), 6)
    except (TypeError, ValueError):
        return None


def _best_threshold(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the highest-precision threshold that still predicts at least one positive."""
    rows = report.get("threshold_suggestions") or []
    best: Optional[Dict[str, Any]] = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        if int(r.get("predicted_positive_count") or 0) <= 0:
            continue
        prec = float(r.get("precision") or 0.0)
        if best is None or prec > float(best.get("precision") or 0.0):
            best = {
                "threshold": _f(r.get("threshold")),
                "precision": _f(prec),
                "predicted_positive_count": int(r.get("predicted_positive_count") or 0),
            }
    return best


async def _latest_run(conn, task: str) -> Optional[Dict[str, Any]]:
    row = await conn.fetchrow(
        """
        SELECT trained_at, model_version, train_row_count, metrics
          FROM m8_model_runs
         WHERE train_config->>'task' = $1
         ORDER BY trained_at DESC
         LIMIT 1
        """,
        task,
    )
    if not row:
        return None
    return {
        "trained_at": row["trained_at"].isoformat() if row.get("trained_at") else None,
        "model_version": row.get("model_version"),
        "train_rows": int(row.get("train_row_count") or 0),
        "metrics": _as_dict(row.get("metrics")),
    }


def _promo_summary(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not run:
        return None
    m = run["metrics"]
    return {
        "trained_at": run["trained_at"],
        "model_version": run["model_version"],
        "train_rows": run["train_rows"],
        "roc_auc": _f(m.get("roc_auc")),
        "average_precision": _f(m.get("average_precision")),
        "lift_at_10pct": _f(m.get("lift_at_10pct")),
        "lift_at_20pct": _f(m.get("lift_at_20pct")),
        "base_positive_rate_test": _f(m.get("base_positive_rate_test")),
        "recommended_threshold": _best_threshold(m),
        "status": m.get("status") or "ok",
    }


def _trim(rows: Any, n: int) -> List[Dict[str, Any]]:
    return [r for r in (rows or []) if isinstance(r, dict)][:n]


def _content_summary(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    trained_at = None
    model_version = None
    train_rows = 0
    if run:
        metrics = run["metrics"]
        trained_at = run["trained_at"]
        model_version = run["model_version"]
        train_rows = run["train_rows"]
    rankings = _as_dict(metrics.get("rankings"))
    if not rankings and _CONTENT_REPORT_PATH.is_file():
        # Fallback: read the local report (pre-DB-record, or engine ran without pool).
        try:
            report = json.loads(_CONTENT_REPORT_PATH.read_text(encoding="utf-8"))
            rankings = _as_dict(report.get("rankings"))
            metrics = metrics or report
            trained_at = trained_at or report.get("generated_at")
        except Exception as e:
            logger.debug("content report fallback failed: %s", e)
    if not rankings:
        return None
    return {
        "trained_at": trained_at,
        "model_version": model_version,
        "train_rows": train_rows,
        "rows": int(metrics.get("rows") or 0),
        "roc_auc": _f(metrics.get("roc_auc")),
        "lift_at_10pct": _f(metrics.get("lift_at_10pct")),
        "status": metrics.get("status") or "ok",
        "top_topics": _trim(rankings.get("top_topics"), 6),
        "top_hashtags": _trim(rankings.get("top_hashtags"), 8),
        "top_platform_topic": _trim(rankings.get("top_platform_topic"), 6),
        "top_packaging": _trim(rankings.get("top_packaging"), 5),
        "top_by_platform": _trim(rankings.get("top_by_platform"), 6),
    }


def _strategy_hints(
    promo: Optional[Dict[str, Any]], content: Optional[Dict[str, Any]]
) -> List[str]:
    hints: List[str] = []
    if promo and promo.get("status") == "ok":
        lift = promo.get("lift_at_10pct") or 0
        auc = promo.get("roc_auc") or 0
        if lift and lift >= 1.3 and auc and auc >= 0.6:
            hints.append(
                f"promo_model_ready_target_top_decile (lift@10%≈{lift:.2f}, auc≈{auc:.2f})"
            )
        elif promo.get("train_rows", 0) >= 8:
            hints.append("promo_model_warming_gather_more_conversion_labels")
    if content and content.get("top_topics"):
        t = content["top_topics"][0]
        hints.append(
            f"content_hottest_topic:{t.get('content_category')} "
            f"(eng≈{t.get('mean_engagement_pct')}%, hot_rate≈{t.get('hot_rate')})"
        )
        pt = (content.get("top_platform_topic") or [])
        if pt:
            top = pt[0]
            hints.append(
                f"content_best_platform_topic:{top.get('platform')}/{top.get('content_category')}"
            )
        pk = (content.get("top_packaging") or [])
        if pk:
            p = pk[0]
            hints.append(
                "content_best_packaging:"
                f"{p.get('caption_style')}/{p.get('caption_tone')}/{p.get('caption_voice')}"
            )
    return hints


async def fetch_ml_model_intelligence(conn) -> Dict[str, Any]:
    """Compact bundle of trained-model signals for the marketing AI + ops page."""
    try:
        promo_run = await _latest_run(conn, PROMO_TASK)
        content_run = await _latest_run(conn, CONTENT_TASK)
    except Exception as e:
        logger.warning("fetch_ml_model_intelligence query failed: %s", e)
        promo_run = content_run = None

    promo = _promo_summary(promo_run)
    content = _content_summary(content_run)
    hints = _strategy_hints(promo, content)
    return {
        "available": bool(promo or content),
        "promo_uplift": promo,
        "content_success": content,
        "hottest_content": (
            {
                "top_topics": content.get("top_topics"),
                "top_hashtags": content.get("top_hashtags"),
                "top_platform_topic": content.get("top_platform_topic"),
                "top_packaging": content.get("top_packaging"),
            }
            if content
            else None
        ),
        "strategy_hints": hints,
        "sources": {
            "promo": "m8_model_runs(train_config.task=promo_targeting_uplift_baseline)",
            "content": "m8_model_runs(train_config.task=content_success_hotness) + content_success_report.json",
        },
    }
