"""
Runtime content-hotness scoring (mirrors promo_targeting_model pattern).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uploadm8.content_success_model")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "ml" / "content_success_model.joblib"
_DEFAULT_REPORT_PATH = _REPO_ROOT / "data" / "ml" / "content_success_report.json"

_cache_lock = threading.Lock()
_cached: Dict[str, Any] = {"pipeline": None, "report": None, "model_mtime": None, "report_mtime": None}


def _model_path() -> Path:
    raw = (os.environ.get("UM8_CONTENT_MODEL_PATH") or "").strip()
    return Path(raw) if raw else _DEFAULT_MODEL_PATH


def _report_path() -> Path:
    raw = (os.environ.get("UM8_CONTENT_REPORT_PATH") or "").strip()
    return Path(raw) if raw else _DEFAULT_REPORT_PATH


def reload_model() -> None:
    with _cache_lock:
        _cached["pipeline"] = None
        _cached["report"] = None
        _cached["model_mtime"] = None
        _cached["report_mtime"] = None


def _load_report() -> Dict[str, Any]:
    p = _report_path()
    try:
        mt = p.stat().st_mtime if p.is_file() else None
    except OSError:
        mt = None
    if _cached["report"] is not None and _cached["report_mtime"] == mt:
        return _cached["report"]
    if not p.is_file():
        _cached["report"] = {}
        _cached["report_mtime"] = mt
        return {}
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("content report read failed: %s", e)
        rep = {}
    _cached["report"] = rep if isinstance(rep, dict) else {}
    _cached["report_mtime"] = mt
    return _cached["report"]


def _load_pipeline():
    import joblib

    p = _model_path()
    try:
        mt = p.stat().st_mtime if p.is_file() else None
    except OSError:
        mt = None
    if _cached["pipeline"] is not None and _cached["model_mtime"] == mt:
        return _cached["pipeline"]
    if not p.is_file():
        _cached["pipeline"] = None
        _cached["model_mtime"] = mt
        return None
    try:
        pipe = joblib.load(p)
    except Exception as e:
        logger.warning("content model load failed: %s", e)
        pipe = None
    _cached["pipeline"] = pipe
    _cached["model_mtime"] = mt
    return pipe


def model_ready() -> bool:
    return _load_pipeline() is not None


def _heuristic_hotness(features: Dict[str, Any]) -> float:
    views = float(features.get("avg_views_30d") or 0)
    eng = float(features.get("avg_engagement_pct_30d") or 0)
    z = -1.5 + 0.002 * views + 0.04 * eng
    return 1.0 / (1.0 + math.exp(-z))


def score_features(features: Dict[str, Any]) -> Tuple[float, str]:
    pipe = _load_pipeline()
    rep = _load_report()
    model_key = str(rep.get("model") or "content_success_v1")
    if pipe is not None:
        try:
            import pandas as pd

            cols = rep.get("feature_columns") or []
            if not isinstance(cols, (list, tuple)) or not cols:
                raise ValueError("content report missing feature_columns")
            row = {c: features.get(c) for c in cols}
            df = pd.DataFrame([row])
            proba = pipe.predict_proba(df)
            if proba is None or len(proba) < 1 or len(proba[0]) < 2:
                raise ValueError("predict_proba returned empty result")
            return float(proba[0][1]), model_key
        except Exception as e:
            logger.warning("content model predict failed: %s", e)
    return _heuristic_hotness(features), "content_hotness_heuristic"


def _band_for_score(score: float) -> str:
    return "high" if score >= 0.65 else ("medium" if score >= 0.4 else "low")


def score_upload_context(ctx) -> Tuple[float, str, str]:
    """Score one upload job context (0–1 hotness). Returns (score, model_key, band)."""
    feats: Dict[str, Any] = {
        "platform": (ctx.platforms or ["unknown"])[0] if getattr(ctx, "platforms", None) else "unknown",
        "content_category": getattr(ctx, "content_category", None) or "general",
        "caption_style": (getattr(ctx, "user_settings", None) or {}).get("captionStyle")
        or (getattr(ctx, "user_settings", None) or {}).get("caption_style"),
        "hashtag_count": len(getattr(ctx, "ai_hashtags", None) or []),
        "duration_seconds": float((getattr(ctx, "video_info", None) or {}).get("duration") or 0),
    }
    score, model_key = score_features(feats)
    return score, model_key, _band_for_score(score)


def score_presign_init(
    *,
    platforms: Optional[list],
    user_prefs: Optional[Dict[str, Any]],
    schedule_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight pre-upload hotness hint (no video metadata yet)."""
    prefs = user_prefs or {}
    from services.thumbnail_studio_strategy import read_thumbnail_studio_default_strategy, strategy_audience_niche

    nested = read_thumbnail_studio_default_strategy(prefs)
    feats: Dict[str, Any] = {
        "platform": (platforms or ["unknown"])[0],
        "content_category": strategy_audience_niche(prefs, "general"),
        "caption_style": prefs.get("captionStyle") or prefs.get("caption_style"),
        "hashtag_count": 0,
        "duration_seconds": 0.0,
    }
    score, model_key = score_features(feats)
    band = _band_for_score(score)
    hints: List[str] = []
    if band == "low":
        hints.append(
            "ML signals are thin for this packaging — try a stronger caption style, "
            "peak-hour smart schedule, or thumbnail persona before publishing."
        )
    elif band == "medium" and (schedule_mode or "") in ("scheduled", "smart"):
        hints.append(
            "Content-success priors are moderate — smart schedule and thumbnail tuning "
            "can lift reach on this upload."
        )
    return {
        "score": round(score, 4),
        "band": band,
        "model_key": model_key,
        "hints": hints,
    }
