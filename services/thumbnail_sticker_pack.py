"""
Build sticker specs from hydration, visual entities, and Video Intelligence tracks.

Stickers are **real crops from the source frame** (or text badges when no box exists).
Used by the local PIL compositor before optional Pikzels studio (YouTube ref layout).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.vision_entities import collect_visual_entities
from core.vision_labels import is_redundant_vision_label, resolve_ambient_profiles

_GENERIC_VI_OBJECTS = frozenset(
    {"vehicle", "car", "person", "people", "object", "animal", "structure", "land vehicle"}
)

# kind → default priority (lower = more important)
_KIND_PRIORITY: Dict[str, float] = {
    "landmark": 10,
    "restaurant": 12,
    "food": 14,
    "vehicle": 16,
    "signage": 18,
    "music": 20,
    "geo": 22,
    "trill": 24,
    "object": 26,
    "text_badge": 28,
}

_NICHE_KIND_ORDER: Dict[str, List[str]] = {
    "automotive": ["vehicle", "signage", "music", "geo", "trill", "landmark", "restaurant", "object"],
    "travel": ["landmark", "geo", "vehicle", "restaurant", "music", "signage", "object"],
    "food": ["restaurant", "food", "landmark", "music", "geo", "object"],
    "camping": ["landmark", "geo", "outdoors", "music", "object"],
    "fishing": ["object", "geo", "landmark", "music"],
    "general": ["landmark", "restaurant", "vehicle", "music", "geo", "signage", "trill", "object"],
}


@dataclass
class StickerSpec:
    slot_id: str
    kind: str
    label: str
    source: str
    priority: float
    box: Optional[Dict[str, float]] = None  # left, top, right, bottom in 0..1
    frame_offset_s: float = 0.0
    text_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "StickerSpec":
        return cls(
            slot_id=str(raw.get("slot_id") or ""),
            kind=str(raw.get("kind") or "object"),
            label=str(raw.get("label") or ""),
            source=str(raw.get("source") or ""),
            priority=float(raw.get("priority") or 99),
            box=raw.get("box") if isinstance(raw.get("box"), dict) else None,
            frame_offset_s=float(raw.get("frame_offset_s") or 0),
            text_only=bool(raw.get("text_only")),
        )


def sticker_pack_from_json(raw: Any) -> List[StickerSpec]:
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out: List[StickerSpec] = []
    for row in raw:
        if isinstance(row, dict):
            out.append(StickerSpec.from_dict(row))
    return out


def sticker_pack_to_json(stickers: Sequence[StickerSpec]) -> str:
    return json.dumps([s.to_dict() for s in stickers], default=str)


def _hp_evidence(ctx: Any) -> Dict[str, Any]:
    hp = getattr(ctx, "hydration_payload", None) or {}
    if isinstance(hp, dict):
        ev = hp.get("evidence")
        if isinstance(ev, dict):
            return ev
    return {}


def _category_for_ctx(ctx: Any) -> str:
    cat = str(getattr(ctx, "thumbnail_category", None) or "").strip().lower()
    if cat:
        return cat
    hp = getattr(ctx, "hydration_payload", None) or {}
    if isinstance(hp, dict) and hp.get("category"):
        return str(hp["category"]).strip().lower()
    return "general"


def _kind_rank(kind: str, niche: str) -> float:
    order = _NICHE_KIND_ORDER.get(niche) or _NICHE_KIND_ORDER["general"]
    try:
        idx = order.index(kind)
    except ValueError:
        idx = len(order) + 5
    return float(_KIND_PRIORITY.get(kind, 30) + idx)


def _best_vi_box(
    track: Dict[str, Any],
    offset_s: float,
    *,
    max_delta: float = 3.0,
) -> Optional[Dict[str, float]]:
    best: Optional[Dict[str, float]] = None
    best_delta = max_delta
    for fr in track.get("frames") or []:
        if not isinstance(fr, dict):
            continue
        box = fr.get("box")
        if not isinstance(box, dict):
            continue
        try:
            t = float(fr.get("t_s", 0))
        except (TypeError, ValueError):
            continue
        delta = abs(t - offset_s)
        if delta <= best_delta:
            best_delta = delta
            best = {
                "left": float(box.get("left", 0)),
                "top": float(box.get("top", 0)),
                "right": float(box.get("right", 1)),
                "bottom": float(box.get("bottom", 1)),
            }
    return best


def _vi_tracks(ctx: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for attr in ("video_intelligence", "video_intelligence_context"):
        vi = getattr(ctx, attr, None) or {}
        if isinstance(vi, dict):
            for tr in vi.get("object_tracks") or []:
                if isinstance(tr, dict):
                    out.append(tr)
    return out


def _ocr_sticker_candidates(ctx: Any, *, limit: int = 3) -> List[str]:
    vc = getattr(ctx, "vision_context", None) or {}
    ocr = str(vc.get("ocr_text") or "").strip() if isinstance(vc, dict) else ""
    if not ocr:
        return []
    parts: List[str] = []
    for block in re.split(r"[\n|]+", ocr):
        line = re.sub(r"\s+", " ", block.strip())
        if len(line) < 3 or len(line) > 48:
            continue
        if re.fullmatch(r"[\d\s./:-MPHmph]+", line):
            continue
        if line.lower() in parts:
            continue
        parts.append(line[:48])
        if len(parts) >= limit:
            break
    return parts


def build_sticker_pack(ctx: Any, frame_offset_s: float, *, max_stickers: int = 4) -> List[StickerSpec]:
    """
    Collect up to ``max_stickers`` sticker specs from ctx evidence.

    Image stickers prefer VI bounding boxes near ``frame_offset_s``; hydration-only
    signals become text badges (music, geo, trill).
    """
    niche = _category_for_ctx(ctx)
    ambient = resolve_ambient_profiles(
        category=niche,
        filename=str(getattr(ctx, "filename", "") or ""),
        vision_label_names=(getattr(ctx, "vision_context", None) or {}).get("label_names")
        if isinstance(getattr(ctx, "vision_context", None), dict)
        else None,
    )
    bundle = collect_visual_entities(
        vision_context=getattr(ctx, "vision_context", None),
        video_intelligence=getattr(ctx, "video_intelligence", None),
        video_intelligence_context=getattr(ctx, "video_intelligence_context", None),
        category=niche,
        filename=str(getattr(ctx, "filename", "") or ""),
    )
    ev = _hp_evidence(ctx)
    candidates: List[StickerSpec] = []
    seen_labels: set[str] = set()

    def _add(
        *,
        kind: str,
        label: str,
        source: str,
        box: Optional[Dict[str, float]] = None,
        text_only: bool = False,
    ) -> None:
        clean = re.sub(r"\s+", " ", str(label or "").strip())[:80]
        if len(clean) < 2:
            return
        key = clean.lower()
        if key in seen_labels:
            return
        if not text_only and is_redundant_vision_label(clean, ambient_profiles=ambient):
            return
        seen_labels.add(key)
        candidates.append(
            StickerSpec(
                slot_id=f"{kind}-{len(candidates)}",
                kind=kind,
                label=clean,
                source=source,
                priority=_kind_rank(kind, niche),
                box=box,
                frame_offset_s=float(frame_offset_s),
                text_only=text_only,
            )
        )

    # VI object tracks with boxes
    for tr in _vi_tracks(ctx):
        desc = str(tr.get("description") or "").strip()
        if not desc or desc.lower() in _GENERIC_VI_OBJECTS:
            continue
        if is_redundant_vision_label(desc, ambient_profiles=ambient):
            continue
        box = _best_vi_box(tr, frame_offset_s)
        kind = "signage" if re.search(r"sign|signal|billboard", desc, re.I) else "object"
        if bundle.vehicles and any(v.lower() in desc.lower() for v in bundle.vehicles):
            kind = "vehicle"
        _add(kind=kind, label=desc, source="vi_object", box=box, text_only=box is None)

    for name in bundle.landmarks[:2]:
        _add(kind="landmark", label=name, source="vision_landmark", text_only=True)
    for name in bundle.restaurants[:2]:
        _add(kind="restaurant", label=name, source="vision_restaurant", text_only=True)
    for name in (bundle.vehicles or bundle.brands)[:2]:
        _add(kind="vehicle", label=name, source="vision_vehicle", text_only=True)
    for name in bundle.signage[:2]:
        _add(kind="signage", label=name, source="vision_signage", text_only=True)

    mus = ev.get("music") if isinstance(ev.get("music"), dict) else {}
    ma = str(mus.get("artist") or "").strip()
    mt = str(mus.get("title") or "").strip()
    if ma or mt:
        _add(
            kind="music",
            label=" — ".join(p for p in (ma, mt) if p)[:64],
            source="hydration_music",
            text_only=True,
        )

    geo = ev.get("geo") if isinstance(ev.get("geo"), dict) else {}
    geo_label = str(geo.get("display") or "").strip()
    if not geo_label:
        city = str(geo.get("city") or "").strip()
        st = str(geo.get("state") or "").strip()
        if city and st:
            geo_label = f"{city}, {st}"
        elif city:
            geo_label = city
    if geo_label:
        _add(kind="geo", label=geo_label[:56], source="hydration_geo", text_only=True)

    tri = ev.get("trill") if isinstance(ev.get("trill"), dict) else {}
    tb = str(tri.get("bucket") or "").strip()
    tsc = tri.get("score")
    if tb or tsc is not None:
        try:
            sc = f"{float(tsc):.0f}" if tsc is not None else ""
        except (TypeError, ValueError):
            sc = ""
        trill_lbl = " ".join(p for p in (tb, f"score {sc}" if sc else "") if p).strip()
        if trill_lbl:
            _add(kind="trill", label=trill_lbl[:40], source="hydration_trill", text_only=True)

    osd = ev.get("osd") if isinstance(ev.get("osd"), dict) else {}
    try:
        mph = float(osd.get("max_speed_mph") or 0)
    except (TypeError, ValueError):
        mph = 0.0
    if mph >= 10 and niche in ("automotive", "travel", "general", "sports", "dashcam"):
        _add(kind="text_badge", label=f"{mph:.0f} MPH", source="hydration_osd", text_only=True)

    for line in _ocr_sticker_candidates(ctx):
        _add(kind="text_badge", label=line, source="vision_ocr", text_only=True)

    candidates.sort(key=lambda s: (s.priority, 0 if s.box else 1))
    return candidates[: max(1, int(max_stickers or 4))]
