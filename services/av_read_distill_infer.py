"""
Fail-closed P6 AV-read distill inference.

Skip TL only when ALL hold:
  1) AV_READ_DEPTH_SKIP_TL=1
  2) UM8_AV_READ_SKIP_TL_FLOORS_OK=1
  3) model loads (joblib bundle with deep + pre + text)
  4) confidence = 1 - P(needs_deep) >= 0.85
  5) vision not weak / route not force

Never replaces identity or grounding. Fusion must still run if VU empty after skip.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("uploadm8-worker")

_MODEL: Any = None
_MODEL_PATH = Path(
    os.environ.get("UM8_ML_ENGINE_AV_MODEL_PATH", "data/ml/av_read_distill_model.joblib")
)
_SKIP_CONF_FLOOR = float(os.environ.get("AV_READ_SKIP_TL_CONFIDENCE", "0.85") or "0.85")


def _env_on(name: str) -> bool:
    try:
        from services.av_read_runtime_flags import flag_enabled

        if name in (
            "AV_READ_DEPTH_SKIP_TL",
            "UM8_AV_READ_SKIP_TL_FLOORS_OK",
        ):
            return flag_enabled(name)
    except Exception:
        pass
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def skip_tl_gates_enabled() -> bool:
    """True when operator wants vision-first conditional TL path."""
    return _env_on("AV_READ_DEPTH_SKIP_TL") and _env_on("UM8_AV_READ_SKIP_TL_FLOORS_OK")


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not _MODEL_PATH.is_file():
        return None
    import joblib

    _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL


def _p_needs_deep(bundle: Any, ctx: Any) -> Optional[float]:
    """Return P(needs_deep=1) or None on failure."""
    if not isinstance(bundle, dict):
        return None
    deep = bundle.get("deep")
    pre = bundle.get("pre")
    text_pipe = bundle.get("text")
    if deep is None or pre is None or text_pipe is None:
        return None
    from services.av_read_features import CAT_FEATURES, NUM_FEATURES, feature_row_from_ctx

    row = feature_row_from_ctx(ctx)
    df = pd.DataFrame([row])
    for c in NUM_FEATURES:
        if c not in df.columns:
            df[c] = 0
    for c in CAT_FEATURES:
        if c not in df.columns:
            df[c] = "unknown"
    if "fusion_text" not in df.columns:
        df["fusion_text"] = ""
    X_struct = pre.transform(df)
    X_text = text_pipe.transform(df["fusion_text"])
    if hasattr(X_text, "toarray"):
        X_text = X_text.toarray()
    X = np.hstack([np.asarray(X_struct), np.asarray(X_text)])
    proba = deep.predict_proba(X)
    classes = list(getattr(deep, "classes_", []) or [])
    # Find index of positive class 1 / True
    idx = None
    for i, c in enumerate(classes):
        if c in (1, True, "1") or str(c) == "1":
            idx = i
            break
    if idx is None:
        # Binary with classes [0,1] usual; if single class, refuse skip
        if proba.shape[1] == 2:
            idx = 1
        else:
            return None
    return float(proba[0, idx])


def maybe_skip_twelvelabs(ctx: Any) -> Dict[str, Any]:
    """Return {skip_tl: bool, confidence: float, reason: str}. Default fail-closed."""
    if not _env_on("AV_READ_DEPTH_SKIP_TL"):
        return {"skip_tl": False, "confidence": 0.0, "reason": "kill_switch_off"}

    if not _env_on("UM8_AV_READ_SKIP_TL_FLOORS_OK"):
        return {"skip_tl": False, "confidence": 0.0, "reason": "floors_go_not_set"}

    try:
        arts = getattr(ctx, "output_artifacts", None) or {}
        route = arts.get("multimodal_depth_route_v1") if isinstance(arts, dict) else {}
        if isinstance(route, dict) and route.get("vision_weak"):
            return {"skip_tl": False, "confidence": 0.0, "reason": "vision_weak_fail_closed"}
        if isinstance(route, dict) and route.get("force_twelvelabs"):
            return {"skip_tl": False, "confidence": 0.0, "reason": "force_twelvelabs"}
        # Route may not be persisted yet (called mid-route_multimodal_depth).
        try:
            from core.vision_labels import vision_labels_are_weak

            vc = getattr(ctx, "vision_context", None) or {}
            if isinstance(vc, dict) and vc and vision_labels_are_weak(
                vc.get("label_names") or [],
                landmark_names=vc.get("landmark_names") or [],
                logo_names=vc.get("logo_names") or [],
                ocr_text=str(vc.get("ocr_text") or ""),
            ):
                return {"skip_tl": False, "confidence": 0.0, "reason": "vision_weak_fail_closed"}
        except Exception:
            pass

        bundle = _load_model()
        if bundle is None:
            return {"skip_tl": False, "confidence": 0.0, "reason": "model_missing"}

        p_deep = _p_needs_deep(bundle, ctx)
        if p_deep is None:
            return {"skip_tl": False, "confidence": 0.0, "reason": "model_infer_error"}

        conf = max(0.0, min(1.0, 1.0 - float(p_deep)))
        if conf < _SKIP_CONF_FLOOR:
            return {
                "skip_tl": False,
                "confidence": conf,
                "p_needs_deep": p_deep,
                "reason": "confidence_below_floor",
            }
        return {
            "skip_tl": True,
            "confidence": conf,
            "p_needs_deep": p_deep,
            "reason": "distill_calm",
        }
    except Exception as e:
        logger.debug("av_read distill infer fail-closed: %s", e)
        return {"skip_tl": False, "confidence": 0.0, "reason": f"error:{str(e)[:80]}"}


__all__ = ["maybe_skip_twelvelabs", "skip_tl_gates_enabled", "_p_needs_deep"]
