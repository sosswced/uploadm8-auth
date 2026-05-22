"""Persist local thumbnail stage outputs for simulate / test runners."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

_CANONICAL_THUMB_PLATFORMS: tuple[str, ...] = ("youtube", "instagram", "facebook", "tiktok")


def persist_thumbnail_outputs(ctx, dest_root: Path) -> Dict[str, str]:
    """Copy winner, per-platform styled, and raw-best frames to durable JPEG paths."""
    dest_root.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    art = getattr(ctx, "output_artifacts", None) or {}

    plat_map: Dict[str, Any] = {}
    plat_raw = art.get("platform_thumbnail_map")
    if isinstance(plat_raw, str) and plat_raw.strip():
        try:
            plat_map = json.loads(plat_raw)
        except json.JSONDecodeError:
            plat_map = {}
    elif isinstance(plat_raw, dict):
        plat_map = dict(plat_raw)

    for plat, src in (plat_map or {}).items():
        sp = Path(str(src))
        if not sp.is_file():
            continue
        dst = dest_root / f"thumb_{plat}.jpg"
        shutil.copy2(sp, dst)
        saved[f"styled_{plat}"] = str(dst.resolve())

    winner_src = getattr(ctx, "thumbnail_path", None) or art.get("thumbnail")
    if winner_src:
        wp = Path(str(winner_src))
        if wp.is_file():
            dst = dest_root / "thumb_winner.jpg"
            shutil.copy2(wp, dst)
            saved["winner"] = str(dst.resolve())

    candidates = list(getattr(ctx, "thumbnail_paths", None) or [])
    if candidates:
        raw = Path(str(candidates[0]))
        winner_resolved = str(Path(str(winner_src)).resolve()) if winner_src else ""
        if raw.is_file() and str(raw.resolve()) != winner_resolved:
            dst = dest_root / "thumb_raw_best.jpg"
            shutil.copy2(raw, dst)
            saved["raw_best"] = str(dst.resolve())

    requested: List[str] = [
        str(p).strip().lower()
        for p in (getattr(ctx, "platforms", None) or [])
        if str(p).strip()
    ]
    rendered = sorted(plat_map.keys())
    missing = [p for p in _CANONICAL_THUMB_PLATFORMS if p in requested and p not in rendered]

    meta = {
        "render_method": art.get("thumbnail_render_method"),
        "selection_method": art.get("thumbnail_selection_method"),
        "category": art.get("thumbnail_category"),
        "platforms_requested": requested or list(_CANONICAL_THUMB_PLATFORMS),
        "platforms_rendered": rendered,
        "platforms_missing": missing,
        "headline": None,
        "dashcam_pov": art.get("thumbnail_dashcam_pov") == "1",
    }
    brief_raw = art.get("thumbnail_brief_json")
    if isinstance(brief_raw, str) and brief_raw.strip():
        try:
            brief = json.loads(brief_raw)
            if isinstance(brief, dict):
                meta["headline"] = brief.get("selected_headline")
        except json.JSONDecodeError:
            pass
    elif isinstance(brief_raw, dict):
        meta["headline"] = brief_raw.get("selected_headline")
    (dest_root / "thumb_meta.json").write_text(
        json.dumps({"outputs": saved, **meta}, indent=2),
        encoding="utf-8",
    )
    return saved


def print_thumbnail_outputs(saved: Dict[str, str], ctx) -> None:
    art = getattr(ctx, "output_artifacts", None) or {}
    print("Thumbnail outputs:")
    if not saved:
        print("  (none — stage skipped or produced no files)")
        return
    for key in sorted(saved.keys()):
        print(f"  {key}: {saved[key]}")
    render = art.get("thumbnail_render_method") or "-"
    select = art.get("thumbnail_selection_method") or "-"
    print(f"  render_method={render} selection={select}")
