"""
Thumbnail quality checks — YouTube search-result preview (168×94) legibility proxy.

YouTube shows tiny thumbnails in search (~168×94 CSS reference size). We downscale the
final JPEG and score local contrast / edge energy so dashboards can flag weak designs.

Env:
  YOUTUBE_SEARCH_PREVIEW_QA_ENABLED  (default true)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger("uploadm8-worker.thumbnail_qa")

YOUTUBE_SEARCH_PREVIEW_QA = os.environ.get("YOUTUBE_SEARCH_PREVIEW_QA_ENABLED", "true").lower() == "true"

# Heuristic: below this "readability_score" (0–1), text may wash out at phone search size.
YOUTUBE_PREVIEW_READABILITY_FLOOR = float(os.environ.get("YOUTUBE_PREVIEW_READABILITY_FLOOR", "0.18") or 0.18)


def assess_youtube_search_preview_readability(image_path: Path) -> Dict[str, Any]:
    """
    Resize to 168×94, measure grayscale contrast (std) and edge density in three bands.
    Returns metrics; does not reject thumbnails by itself (caller decides).
    """
    out: Dict[str, Any] = {"ok": False, "readability_score": 0.0, "width": 168, "height": 94}
    try:
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.fit(img, (168, 94), method=Image.Resampling.LANCZOS)
        gray = np.asarray(ImageOps.grayscale(img), dtype=np.float32) / 255.0
        # Global contrast proxy
        contrast = float(np.std(gray))
        h, w = gray.shape
        third = max(1, h // 3)
        bands = (
            gray[0:third, :],
            gray[third : 2 * third, :],
            gray[2 * third :, :],
        )
        band_stds = [float(np.std(b)) for b in bands]
        edges = np.asarray(img.filter(ImageFilter.FIND_EDGES).convert("L"), dtype=np.float32) / 255.0
        edge_mean = float(np.mean(edges))
        # Combine: need both separation (contrast) and structure (edges) for text pops
        score = min(1.0, 0.55 * contrast + 0.25 * edge_mean + 0.20 * max(band_stds))
        out.update(
            {
                "ok": True,
                "readability_score": round(score, 4),
                "grayscale_std": round(contrast, 4),
                "edge_density": round(edge_mean, 4),
                "band_stds": [round(x, 4) for x in band_stds],
                "passes_floor": score >= YOUTUBE_PREVIEW_READABILITY_FLOOR,
                "floor": YOUTUBE_PREVIEW_READABILITY_FLOOR,
            }
        )
    except Exception as e:
        logger.debug("[thumbnail_qa] preview readability failed: %s", e)
        out["error"] = str(e)[:200]
    return out


def laplacian_variance(image_path: Path) -> float:
    """Variance of Laplacian on grayscale — higher = sharper, lower = more motion blur."""
    try:
        g = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        if g.shape[0] < 3 or g.shape[1] < 3:
            return 0.0
        lap = (
            -4.0 * g[1:-1, 1:-1]
            + g[:-2, 1:-1]
            + g[2:, 1:-1]
            + g[1:-1, :-2]
            + g[1:-1, 2:]
        )
        return float(np.var(lap))
    except Exception:
        return 0.0


def pick_tiktok_cover_offset_seconds(
    candidates: List[Tuple[Any, float]],
    path_to_offset: Dict[str, float],
    component_scores: Dict[str, Dict[str, float]],
    mode: str,
    best_path: Optional[Path] = None,
) -> Tuple[float, str]:
    """
    Choose a timestamp for TikTok cover frame selection.

    mode:
      motion_blur — prefer lower sharpness / lower Laplacian variance (dramatic motion feel)
      sharp — prefer highest sharpness (default product behavior)
      balanced — follow the selected best thumbnail frame when possible
    """
    mode = (mode or "motion_blur").strip().lower()
    if not candidates or not path_to_offset:
        return 1.5, "default"

    def offset_for(path) -> float:
        return float(path_to_offset.get(str(path), 1.5))

    if mode == "sharp":
        best = max(candidates, key=lambda x: float(component_scores.get(str(x[0]), {}).get("sharpness", 0.0)))
        return offset_for(best[0]), "sharp_max"

    if mode == "balanced":
        if best_path is not None and str(best_path) in path_to_offset:
            return offset_for(best_path), "balanced_best_frame"
        best = max(candidates, key=lambda x: float(x[1]))
        return offset_for(best[0]), "balanced_combined"

    # motion_blur: minimize laplacian variance (and prefer lower sharpness score)
    best_path = None
    best_key = None
    for path, _ in candidates:
        sp = float(component_scores.get(str(path), {}).get("sharpness", 0.5))
        lv = laplacian_variance(Path(path))
        # Lower lv + lower sharpness => more blur energy
        key = (lv + sp * 800.0)  # scale sharpness to comparable magnitude
        if best_key is None or key < best_key:
            best_key = key
            best_path = path
    if best_path is None:
        return 1.5, "default"
    return offset_for(best_path), "motion_blur_min_laplacian"
