"""Hero window + second-pass timestamp picks for non-dashcam clips.

Scores timed scene beats so thumbnail extraction can land on the money
moment (landmark / monument OCR) instead of the first generic outdoor still.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Never promote these into a hero window as the winner over a real noun.
_GENERIC_OUTDOOR = frozenset({
    "sky", "outdoor", "outdoors", "person", "people", "windshield",
    "architecture", "building", "nature", "landscape", "horizon",
})

_PROSE_NOUN_SCORE = 50
_BUILDING_SCORE = 20
_GENERIC_SCORE = 10


def score_hero_kind(kind: str, label: str = "") -> int:
    k = str(kind or "").strip().lower()
    lab = str(label or "").strip().lower()
    blob = f"{k} {lab}"
    if k in ("landmark",) or "landmark" in blob:
        return 100
    if k in ("monument", "plaza", "cathedral", "museum") or any(
        tok in lab for tok in ("monument", "memorial", "cathedral", "museum", "plaza", "plaça")
    ):
        return 90
    if k in ("logo", "brand"):
        return 80
    if k in ("music", "concert"):
        return 75
    if k in ("transcript", "speech"):
        return 70
    if k in ("on_screen_text", "ocr", "text"):
        return 60
    if lab in _GENERIC_OUTDOOR or k in _GENERIC_OUTDOOR:
        return _GENERIC_SCORE
    if k in ("building",) or lab == "building":
        return _BUILDING_SCORE
    if k in ("object", "noun", "scene") or lab:
        return _PROSE_NOUN_SCORE
    return _GENERIC_SCORE


def select_hero_window(
    candidates: Sequence[Dict[str, Any]],
    *,
    duration_s: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Pick the highest-scoring candidate; later timestamps win ties."""
    ranked: List[Dict[str, Any]] = []
    for raw in candidates or []:
        if not isinstance(raw, dict):
            continue
        try:
            t = float(raw.get("t_seconds") if raw.get("t_seconds") is not None else raw.get("t") or 0)
        except (TypeError, ValueError):
            continue
        if t < 0:
            continue
        kind = str(raw.get("kind") or raw.get("noun") or "scene")
        label = str(raw.get("label") or raw.get("noun") or kind)
        score = raw.get("score")
        try:
            score_i = int(score) if score is not None else score_hero_kind(kind, label)
        except (TypeError, ValueError):
            score_i = score_hero_kind(kind, label)
        ranked.append({
            "t_seconds": round(t, 3),
            "kind": kind,
            "label": label,
            "score": score_i,
        })
    if not ranked:
        return None
    ranked.sort(key=lambda c: (int(c["score"]), float(c["t_seconds"])))
    winner = dict(ranked[-1])
    if duration_s and winner["t_seconds"] > float(duration_s):
        winner["t_seconds"] = max(0.0, float(duration_s) - 0.5)
    return winner


def pick_second_pass_timestamps(
    beats: Sequence[Dict[str, Any]],
    *,
    max_n: int = 4,
    duration_s: float = 0.0,
    min_gap_s: float = 8.0,
) -> List[float]:
    """Up to ``max_n`` beat midpoints, longest / highest-score first, spread out."""
    scored: List[tuple[float, float, float]] = []  # (-score, -duration, t)
    for beat in beats or []:
        if not isinstance(beat, dict):
            continue
        try:
            start = float(beat.get("t") if beat.get("t") is not None else beat.get("t_seconds") or 0)
            dur = float(beat.get("duration_s") or 0)
        except (TypeError, ValueError):
            continue
        mid = start + (dur / 2.0 if dur > 0 else 0.0)
        if duration_s > 0:
            mid = min(max(0.15, mid), max(0.15, duration_s - 0.25))
        noun = str(beat.get("noun") or beat.get("label") or beat.get("kind") or "")
        score = score_hero_kind(str(beat.get("kind") or "noun"), noun)
        scored.append((-float(score), -dur, mid))
    scored.sort()
    picked: List[float] = []
    for _neg_score, _neg_dur, t in scored:
        if any(abs(t - p) < min_gap_s for p in picked):
            continue
        picked.append(round(t, 3))
        if len(picked) >= max(0, int(max_n)):
            break
    return picked


def pick_speech_second_pass_timestamps(
    ctx: Any,
    *,
    max_n: int = 4,
    duration_s: float = 0.0,
    min_gap_s: float = 8.0,
) -> List[float]:
    """≤4 timestamps from spoken event cues and VI on-screen text peaks."""
    beats: List[Dict[str, Any]] = []
    hint_slugs = (
        "stadium", "arena", "touchback", "kickoff", "field", "jersey",
        "museum", "harbor", "harbour", "plaza", "concert", "stage",
    )
    team_tokens: set[str] = set()
    try:
        from services.place_evidence import _TEAM_TOKENS

        team_tokens = {str(t).strip().lower() for t in _TEAM_TOKENS if str(t).strip()}
    except Exception:
        team_tokens = set()

    ac = getattr(ctx, "audio_context", None) or {}
    segs = ac.get("transcript_segments") if isinstance(ac, dict) else None
    if isinstance(segs, list):
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text") or seg.get("transcript") or "")
            low = text.lower()
            try:
                t = float(seg.get("start") if seg.get("start") is not None else seg.get("start_s") or 0)
            except (TypeError, ValueError):
                t = 0.0
            hit = any(h in low for h in hint_slugs) or any(
                tok in low for tok in team_tokens if len(tok) >= 4
            )
            if hit and text.strip():
                beats.append({
                    "t": t,
                    "duration_s": 2.0,
                    "kind": "transcript",
                    "noun": text.strip()[:80],
                })

    vi = (
        getattr(ctx, "video_intelligence_context", None)
        or getattr(ctx, "video_intelligence", None)
        or {}
    )
    if isinstance(vi, dict):
        for row in (vi.get("on_screen_text") or vi.get("text_detections") or [])[:24]:
            if isinstance(row, dict):
                text = str(row.get("text") or row.get("description") or "")
                try:
                    t = float(row.get("start_s") if row.get("start_s") is not None else row.get("t") or 0)
                except (TypeError, ValueError):
                    t = 0.0
            else:
                text = str(row)
                t = 0.0
            if text.strip():
                beats.append({
                    "t": t,
                    "duration_s": 1.5,
                    "kind": "ocr",
                    "noun": text.strip()[:80],
                })

    if not beats:
        return []
    return pick_second_pass_timestamps(
        beats, max_n=max_n, duration_s=duration_s, min_gap_s=min_gap_s
    )


def build_hero_window_v1(ctx: Any) -> Optional[Dict[str, Any]]:
    """Compose ``hero_window_v1`` from scene beats + Vision landmarks/OCR."""
    arts = getattr(ctx, "output_artifacts", None) or {}
    beats = []
    if isinstance(arts, dict):
        beats = list(arts.get("scene_beats_v1") or [])
    if not beats:
        try:
            from core.vision_labels import prose_scene_beats_from_vi

            beats = prose_scene_beats_from_vi(ctx)
        except Exception:
            beats = []

    duration_s = float(
        getattr(ctx, "duration_seconds", None)
        or getattr(ctx, "duration", None)
        or 0
    )
    vi_info = getattr(ctx, "video_info", None) or {}
    if duration_s <= 0 and isinstance(vi_info, dict):
        try:
            duration_s = float(vi_info.get("duration") or 0)
        except (TypeError, ValueError):
            duration_s = 0.0

    candidates: List[Dict[str, Any]] = []
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        try:
            start = float(beat.get("t") if beat.get("t") is not None else 0)
            dur = float(beat.get("duration_s") or 0)
        except (TypeError, ValueError):
            continue
        noun = str(beat.get("noun") or "")
        candidates.append({
            "t_seconds": start + dur / 2.0,
            "kind": str(beat.get("kind") or "noun"),
            "label": noun,
            "score": score_hero_kind(str(beat.get("kind") or "noun"), noun),
        })

    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        for name in vc.get("landmark_names") or []:
            label = str(name or "").strip()
            if not label:
                continue
            t_hint = 0.0
            stamped = vc.get("landmark_sample_times") or {}
            if isinstance(stamped, dict) and label in stamped:
                try:
                    t_hint = float(stamped[label])
                except (TypeError, ValueError):
                    t_hint = duration_s * 0.7 if duration_s else 0.0
            elif duration_s:
                t_hint = duration_s * 0.7
            candidates.append({
                "t_seconds": t_hint,
                "kind": "landmark",
                "label": label,
                "score": 100,
            })
        ocr = str(vc.get("ocr_text") or "")
        low = ocr.lower()
        if any(tok in low for tok in ("monument", "memorial", "cathedral", "museum", "plaza", "harbour", "harbor")):
            candidates.append({
                "t_seconds": duration_s * 0.65 if duration_s else 0.0,
                "kind": "monument",
                "label": ocr.split("\n")[0][:80],
                "score": 90,
            })

    return select_hero_window(candidates, duration_s=duration_s)


__all__ = [
    "build_hero_window_v1",
    "pick_second_pass_timestamps",
    "pick_speech_second_pass_timestamps",
    "score_hero_kind",
    "select_hero_window",
]
