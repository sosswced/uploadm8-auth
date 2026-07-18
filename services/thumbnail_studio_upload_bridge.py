"""Bridge Thumbnail Studio winners into the upload thumbnail pipeline.

When a user clicks “Use for my next upload”, we persist ``preview_r2_key`` and
``apply_mode`` on ``thumbnailStudioDefaultStrategy``. Upload jobs can then:

* ``cover_direct`` / ``pinned_cover`` — use the Studio JPEG for YouTube/Facebook
  covers and letterbox 9:16 for Instagram/TikTok so Meta ``cover_url`` exists at
  container create (Instagram cannot set cover after publish).
* ``support_image`` / ``fresh_generate`` — regenerate from the video frame but
  steer Pikzels with the Studio still + high ``image_weight``.
* ``strategy_only`` — strategy fields only; no Studio JPEG / YT support image.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uploadm8-api")

COVER_DIRECT_PLATFORMS = frozenset({"youtube", "facebook"})
VERTICAL_PLATFORMS = frozenset({"instagram", "tiktok"})


def strategy_preview_r2_key(strategy: Optional[Dict[str, Any]]) -> str:
    if not isinstance(strategy, dict):
        return ""
    for key in ("preview_r2_key", "previewR2Key", "variant_preview_r2_key"):
        raw = str(strategy.get(key) or "").strip()
        if raw:
            return raw
    return ""


def strategy_apply_mode(strategy: Optional[Dict[str, Any]], us: Optional[Dict[str, Any]] = None) -> str:
    """Resolve bridge mode from upload prefs overlay, then strategy, then legacy defaults."""
    if isinstance(us, dict):
        from services.thumbnail_apply_mode import to_bridge_apply_mode

        raw = us.get("thumbnail_apply_mode") or us.get("thumbnailApplyMode")
        if raw:
            return to_bridge_apply_mode(raw)
    if not isinstance(strategy, dict):
        return "support_image"
    mode = str(strategy.get("apply_mode") or strategy.get("applyMode") or "").strip().lower()
    if mode in ("pinned_cover", "pin", "cover"):
        mode = "cover_direct"
    if mode in ("fresh_generate", "fresh"):
        mode = "support_image"
    if mode in ("cover_direct", "support_image", "strategy_only"):
        return mode
    # Legacy strategies with an R2 key default to cover_direct so winners stick.
    if strategy_preview_r2_key(strategy):
        return "cover_direct"
    return "strategy_only"


async def download_studio_preview_to_path(r2_key: str, dest: Path) -> bool:
    """Download a Studio variant preview from R2 into ``dest``."""
    key = str(r2_key or "").strip()
    if not key or not dest:
        return False
    try:
        from stages import r2 as r2_stage

        await r2_stage.download_file(key, dest)
        return dest.exists() and dest.stat().st_size >= 512
    except Exception:
        logger.warning("studio winner R2 download failed key=%s", key[:120], exc_info=True)
        return False


def public_or_presigned_url_for_r2_key(r2_key: str, *, expires: int = 3600) -> str:
    key = str(r2_key or "").strip()
    if not key:
        return ""
    try:
        from stages import r2 as r2_stage

        url = r2_stage.get_public_url(key) or ""
        if str(url).startswith("http"):
            return str(url)
        signed = r2_stage.generate_presigned_url(key, expires=expires) or ""
        return str(signed) if str(signed).startswith("http") else ""
    except Exception:
        logger.debug("studio winner URL resolve failed", exc_info=True)
        return ""


def letterbox_to_vertical(src: Path, dest: Path, *, width: int = 1080, height: int = 1920) -> bool:
    """Fit a 16:9 (or other) Studio still into 9:16 with letterbox for IG/TikTok covers."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL unavailable — cannot letterbox studio winner for vertical cover")
        return False
    try:
        im = Image.open(src).convert("RGB")
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        scale = min(width / im.width, height / im.height)
        nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.paste(resized, ((width - nw) // 2, (height - nh) // 2))
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, format="JPEG", quality=92, optimize=True)
        return dest.exists() and dest.stat().st_size >= 512
    except Exception:
        logger.warning("studio winner letterbox failed", exc_info=True)
        return False


def similarity_score_paths(a: Path, b: Path) -> Optional[float]:
    """Cheap 0–1 similarity (average hash) for Studio winner vs upload result diagnostics."""
    try:
        from PIL import Image
        import imagehash  # type: ignore
    except ImportError:
        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            ia = Image.open(a).convert("RGB").resize((32, 32))
            ib = Image.open(b).convert("RGB").resize((32, 32))
            pa, pb = list(ia.getdata()), list(ib.getdata())
            if not pa or len(pa) != len(pb):
                return None
            dist = sum(abs(pa[i][0] - pb[i][0]) + abs(pa[i][1] - pb[i][1]) + abs(pa[i][2] - pb[i][2]) for i in range(len(pa)))
            max_d = 255 * 3 * len(pa)
            return round(1.0 - (dist / max_d), 4)
        except Exception:
            return None
    try:
        ha = imagehash.average_hash(Image.open(a))
        hb = imagehash.average_hash(Image.open(b))
        # 64-bit hash → normalize hamming distance
        dist = ha - hb
        return round(max(0.0, 1.0 - (float(dist) / 64.0)), 4)
    except Exception:
        return None


async def apply_studio_winner_to_upload_thumbs(
    *,
    strategy: Dict[str, Any],
    platforms: List[str],
    temp_dir: Path,
    upload_id: str,
    brief: Dict[str, Any],
    studio_opts: Optional[Dict[str, Any]],
    platform_map: Dict[str, str],
    report: Dict[str, Any],
    user_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], Dict[str, Any], List[str], Optional[Dict[str, Any]]]:
    """
    Apply Studio winner assets into the upload render plan.

    Returns ``(platform_map, brief, skip_studio_platforms, opts_overlay)``.
    Platforms in ``skip_studio_platforms`` already have covers (incl. IG/TikTok
    letterbox) and must not call Pikzels recreate — covers are ready for Meta
    container create.
    """
    skip_studio: List[str] = []
    opts_overlay: Optional[Dict[str, Any]] = None
    mode = strategy_apply_mode(strategy, user_settings)
    r2_key = strategy_preview_r2_key(strategy)
    report["studio_winner_apply_mode"] = mode
    report["studio_winner_preview_r2_key"] = r2_key[:200] if r2_key else None

    if mode == "strategy_only" or not r2_key:
        return platform_map, brief, skip_studio, opts_overlay

    dest = Path(temp_dir) / f"studio_winner_{upload_id}.jpg"
    ok = await download_studio_preview_to_path(r2_key, dest)
    if not ok:
        report["studio_winner_apply_error"] = "preview_r2_download_failed"
        return platform_map, brief, skip_studio, opts_overlay

    report["studio_winner_local_path"] = str(dest)

    from services.thumbnail_apply_mode import allow_youtube_support_image

    # Pinned cover: do not also attach as support_image (look is the pin itself).
    # Fresh/support_image: steer vertical regenerates with the Studio still.
    if mode == "support_image" and allow_youtube_support_image(user_settings or {}, apply_mode="fresh_generate"):
        support_url = public_or_presigned_url_for_r2_key(r2_key)
        if support_url:
            brief = dict(brief or {})
            brief["_uploadm8_pikzels_support_image_url"] = support_url
            report["studio_winner_support_url_set"] = True
        opts_overlay = dict(studio_opts or {})
        opts_overlay["image_weight"] = "high"
        report["studio_winner_image_weight"] = "high"
        report["studio_winner_support_only"] = True
        return platform_map, brief, skip_studio, opts_overlay

    plats = [str(p or "").strip().lower() for p in (platforms or []) if str(p or "").strip()]
    if mode == "cover_direct":
        for plat in plats:
            out = Path(temp_dir) / f"thumb_styled_{plat}_{upload_id}.jpg"
            try:
                if plat in COVER_DIRECT_PLATFORMS:
                    shutil.copyfile(dest, out)
                    platform_map[plat] = str(out)
                    skip_studio.append(plat)
                elif plat in VERTICAL_PLATFORMS:
                    # Instagram cover_url must exist at container create — letterbox now.
                    if letterbox_to_vertical(dest, out):
                        platform_map[plat] = str(out)
                        skip_studio.append(plat)
                        report.setdefault("studio_winner_letterbox_platforms", []).append(plat)
                    else:
                        report["studio_winner_letterbox_error"] = plat
            except OSError as e:
                report["studio_winner_apply_error"] = f"copy_failed:{e}"[:180]
        report["studio_winner_cover_direct_platforms"] = list(skip_studio)

    return platform_map, brief, skip_studio, opts_overlay
