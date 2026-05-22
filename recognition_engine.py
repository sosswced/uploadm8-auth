"""
Recognition Engine
==================

Aggregates and persists structured recognition data extracted by Google
Cloud Video Intelligence (object tracking, on-screen text, person detection,
logo recognition) into the ``video_recognition`` and
``upload_recognition_summary`` tables. The persisted data feeds:

* **Thumbnail studio** — picks the keyframe where the highest-confidence
  object track is centered and largest on screen
  (``select_thumbnail_keyframe_offset``).
* **Caption / hashtag pipeline** — surfaces brand callouts, named objects,
  full-clip OCR back to ``hydration_enforcer.EvidencePool`` so M8 outputs
  are forced to use them.
* **App ML / analytics** — daily KPI rollups can join on
  ``upload_recognition_summary`` to surface "what's actually in this user's
  catalog" without re-running Video Intelligence.
* **Admin KPIs** — top objects/logos/text across the platform answers the
  "what content lives here" question with one fast aggregate query.

This module is intentionally pure-data and side-effect free except for the
single async ``persist_recognition`` write. All other helpers can be reused
to derive in-memory rollups (e.g. for thumbnail keyframe selection) without
touching the DB.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("uploadm8-worker.recognition")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

# Generic descriptions we drop from "top object" rollups so the dashboard
# shows interesting things ("Tesla Model 3", "Bicycle") instead of useless
# blanket categories ("Vehicle").
_GENERIC_OBJECT_DESCRIPTIONS = {
    "vehicle", "person", "object", "animal", "structure", "land vehicle",
    "people", "human", "people walking", "outdoor", "indoor",
}


def _clean_description(raw: Any, *, max_len: int = 80) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    return s[:max_len]


def _top_descriptions(
    rows: Iterable[Dict[str, Any]],
    *,
    limit: int = 6,
    skip_generic: bool = True,
) -> List[str]:
    """Return up to ``limit`` distinct descriptions sorted by confidence."""
    seen: set = set()
    out: List[Tuple[str, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        desc = _clean_description(r.get("description"))
        if not desc:
            continue
        if skip_generic and desc.lower() in _GENERIC_OBJECT_DESCRIPTIONS:
            continue
        key = desc.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out.append((desc, conf))
    out.sort(key=lambda t: -t[1])
    return [d for d, _ in out[:limit]]


def _top_text(rows: Iterable[Dict[str, Any]], *, limit: int = 6) -> List[str]:
    """Return up to ``limit`` distinct OCR text snippets sorted by confidence."""
    seen: set = set()
    out: List[Tuple[str, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        text = _clean_description(r.get("text"), max_len=60)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out.append((text, conf))
    out.sort(key=lambda t: -t[1])
    return [d for d, _ in out[:limit]]


def _coverage_seconds(rows: Iterable[Dict[str, Any]]) -> float:
    """Sum on-screen seconds across all rows; saturates at the longest end."""
    total = 0.0
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            start = float(r.get("start_seconds") or r.get("start_s") or 0.0)
            end = float(r.get("end_seconds") or r.get("end_s") or 0.0)
        except (TypeError, ValueError):
            continue
        if end > start:
            total += end - start
    return round(total, 2)


def compute_hydration_score(summary: Dict[str, Any]) -> float:
    """A 0..1 confidence that the recognition payload can drive hydration.

    Used by the pattern_corpus filter (#8) to keep only past uploads whose
    captions were grounded in real evidence. Higher = better signal.
    """
    score = 0.0
    if int(summary.get("object_track_count") or 0) > 0:
        score += 0.30
    if int(summary.get("logo_count") or 0) > 0:
        score += 0.30
    if int(summary.get("text_detection_count") or 0) > 0:
        score += 0.20
    if summary.get("has_people"):
        score += 0.10
    if (summary.get("top_objects") or []):
        score += 0.05
    if (summary.get("top_logos") or []):
        score += 0.05
    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Build summary from a parsed Video Intelligence payload
# ---------------------------------------------------------------------------

def build_summary(vi_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the per-upload summary row from a parsed VI payload.

    Designed to be safe to call with ``None`` / empty / partial inputs;
    always returns a dict that satisfies the
    ``upload_recognition_summary`` schema.
    """
    if not isinstance(vi_payload, dict):
        vi_payload = {}
    objects = list(vi_payload.get("object_tracks") or [])
    persons = list(vi_payload.get("person_segments") or [])
    text = list(vi_payload.get("on_screen_text") or [])
    logos = list(vi_payload.get("logos") or [])

    summary = {
        "object_track_count": len(objects),
        "person_segment_count": len(persons),
        "text_detection_count": len(text),
        "logo_count": len(logos),
        "top_objects": _top_descriptions(objects, limit=6),
        "top_logos": _top_descriptions(logos, limit=6, skip_generic=False),
        "top_text": _top_text(text, limit=6),
        "has_people": len(persons) > 0,
        "coverage_seconds": _coverage_seconds(objects + persons + logos),
        "summary_text": _clean_description(vi_payload.get("summary_text"), max_len=2000),
    }
    summary["hydration_score"] = compute_hydration_score(summary)
    return summary


def merge_visual_catalog_into_summary(
    summary: Dict[str, Any],
    visual_recognition: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach Google Vision rollup buckets to the persisted summary row."""
    if not isinstance(summary, dict):
        summary = {}
    flat = {}
    if isinstance(visual_recognition, dict):
        flat = visual_recognition.get("flat") or {}
    if not flat:
        return summary
    out = dict(summary)
    out["recognition_flat"] = flat
    out["recognition_niche"] = str(visual_recognition.get("niche") or "general")[:64]
    # Top tokens per bucket for fast GIN-less reads in admin/widgets.
    tops: Dict[str, List[str]] = {}
    for key, names in flat.items():
        if isinstance(names, list) and names:
            tops[key] = [str(n)[:120] for n in names[:12]]
    out["recognition_buckets"] = tops
    return out


def select_thumbnail_keyframe_offset(
    vi_payload: Optional[Dict[str, Any]],
    *,
    duration_seconds: float = 0.0,
) -> Optional[float]:
    """Pick the most engaging frame timestamp for thumbnail selection.

    Algorithm:
      1. Highest-confidence object track that's NOT generic (Tesla Model 3
         beats "Vehicle"). Use the middle frame for centered composition.
      2. Logo callout if no good object track.
      3. Person segment midpoint when neither.
      4. Falls back to ``None`` so the legacy sharpness-based selector wins.

    The returned float is a wall-clock time in seconds suitable for ffmpeg
    ``-ss``. Capped at ``duration_seconds - 0.25`` when duration is known
    so we never overshoot the clip.
    """
    if not isinstance(vi_payload, dict):
        return None

    def _midframe_ts(track: Dict[str, Any]) -> Optional[float]:
        frames = track.get("frames") or []
        if frames:
            mid = frames[len(frames) // 2]
            ts = mid.get("t_s") if isinstance(mid, dict) else None
            if ts is not None:
                try:
                    return float(ts)
                except (TypeError, ValueError):
                    pass
        try:
            s = float(track.get("start_seconds") or track.get("start_s") or 0.0)
            e = float(track.get("end_seconds") or track.get("end_s") or 0.0)
        except (TypeError, ValueError):
            return None
        if e > s:
            return round((s + e) / 2, 2)
        return s if s > 0 else None

    # 1. Object tracks (skip generic descriptions)
    objects = vi_payload.get("object_tracks") or []
    candidates: List[Tuple[float, float]] = []
    for ot in objects:
        if not isinstance(ot, dict):
            continue
        desc = _clean_description(ot.get("description")).lower()
        if desc in _GENERIC_OBJECT_DESCRIPTIONS:
            continue
        try:
            conf = float(ot.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        ts = _midframe_ts(ot)
        if ts is not None:
            candidates.append((conf, ts))
    if candidates:
        candidates.sort(key=lambda t: -t[0])
        offset = candidates[0][1]
    else:
        # 2. Logo
        logos = vi_payload.get("logos") or []
        offset = None
        if logos:
            best = max(
                (l for l in logos if isinstance(l, dict)),
                key=lambda l: float(l.get("confidence") or 0.0),
                default=None,
            )
            if best is not None:
                ts = _midframe_ts(best)
                if ts is not None:
                    offset = ts
        if offset is None:
            # 3. Person segment
            persons = vi_payload.get("person_segments") or []
            if persons:
                best = max(
                    (p for p in persons if isinstance(p, dict)),
                    key=lambda p: float(p.get("confidence") or 0.0),
                    default=None,
                )
                if best is not None:
                    ts = _midframe_ts(best)
                    if ts is not None:
                        offset = ts
    if offset is None:
        return None
    if duration_seconds and duration_seconds > 0:
        offset = max(0.0, min(offset, duration_seconds - 0.25))
    return round(float(offset), 2)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def persist_recognition(
    db_pool,
    *,
    upload_id: str,
    user_id: str,
    vi_payload: Optional[Dict[str, Any]],
    visual_recognition: Optional[Dict[str, Any]] = None,
    category: str = "general",
) -> Dict[str, Any]:
    """Write per-detection rows + summary row for an upload.

    Returns the computed summary dict so callers can also stash it on
    ``ctx.recognition_summary`` for downstream stages (notify, KPI rollups,
    admin dashboards). Idempotent at the upload-id grain: re-running this
    function for the same upload replaces both the per-detection rows and
    the summary row in a single transaction.
    """
    summary = build_summary(vi_payload)
    summary = merge_visual_catalog_into_summary(summary, visual_recognition)
    if not db_pool:
        return summary
    if not upload_id or not user_id:
        return summary

    objects = list((vi_payload or {}).get("object_tracks") or [])
    persons = list((vi_payload or {}).get("person_segments") or [])
    text = list((vi_payload or {}).get("on_screen_text") or [])
    logos = list((vi_payload or {}).get("logos") or [])

    rows: List[Tuple[str, str, float, float, float, str, str, str]] = []

    def _add(kind: str, items: List[Dict[str, Any]], *, desc_key: str) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            desc = _clean_description(item.get(desc_key))
            if not desc and kind != "person":
                continue
            try:
                conf = float(item.get("confidence") or 0.0)
                start = float(item.get("start_seconds") or item.get("start_s") or 0.0)
                end = float(item.get("end_seconds") or item.get("end_s") or 0.0)
            except (TypeError, ValueError):
                continue
            frames_blob = json.dumps(item.get("frames") or [])
            attrs_blob = json.dumps({"attributes": item.get("attributes") or []}) if item.get("attributes") else "{}"
            raw_blob = json.dumps(item)
            rows.append((kind, desc, conf, start, end, frames_blob, attrs_blob, raw_blob))

    _add("object", objects, desc_key="description")
    _add("person", persons, desc_key="description")
    _add("text", text, desc_key="text")
    _add("logo", logos, desc_key="description")

    summary_blob = json.dumps(summary)
    raw_summary_blob = json.dumps(
        {
            "summary_text": summary.get("summary_text") or "",
            "recognition_flat": summary.get("recognition_flat") or {},
            "recognition_buckets": summary.get("recognition_buckets") or {},
            "recognition_niche": summary.get("recognition_niche") or "",
        }
    )

    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM video_recognition WHERE upload_id = $1",
                    upload_id,
                )
                if rows:
                    await conn.executemany(
                        """
                        INSERT INTO video_recognition (
                            upload_id, user_id, kind, description, confidence,
                            start_seconds, end_seconds, frames, attributes, raw
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10::jsonb)
                        """,
                        [
                            (
                                upload_id, user_id, kind, desc, conf,
                                start, end, frames, attrs, raw,
                            )
                            for (kind, desc, conf, start, end, frames, attrs, raw) in rows
                        ],
                    )
                await conn.execute(
                    """
                    INSERT INTO upload_recognition_summary (
                        upload_id, user_id,
                        object_track_count, person_segment_count,
                        text_detection_count, logo_count,
                        top_objects, top_logos, top_text,
                        has_people, coverage_seconds,
                        summary_text, hydration_score,
                        raw_summary, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, NOW()
                    )
                    ON CONFLICT (upload_id) DO UPDATE SET
                        object_track_count = EXCLUDED.object_track_count,
                        person_segment_count = EXCLUDED.person_segment_count,
                        text_detection_count = EXCLUDED.text_detection_count,
                        logo_count = EXCLUDED.logo_count,
                        top_objects = EXCLUDED.top_objects,
                        top_logos = EXCLUDED.top_logos,
                        top_text = EXCLUDED.top_text,
                        has_people = EXCLUDED.has_people,
                        coverage_seconds = EXCLUDED.coverage_seconds,
                        summary_text = EXCLUDED.summary_text,
                        hydration_score = EXCLUDED.hydration_score,
                        raw_summary = EXCLUDED.raw_summary,
                        updated_at = NOW()
                    """,
                    upload_id,
                    user_id,
                    summary["object_track_count"],
                    summary["person_segment_count"],
                    summary["text_detection_count"],
                    summary["logo_count"],
                    summary["top_objects"],
                    summary["top_logos"],
                    summary["top_text"],
                    summary["has_people"],
                    summary["coverage_seconds"],
                    summary["summary_text"],
                    summary["hydration_score"],
                    raw_summary_blob,
                )
        try:
            from services.visual_entity_memory import upsert_catalog_entities

            flat = summary.get("recognition_flat") or {}
            if isinstance(flat, dict) and flat:
                await upsert_catalog_entities(
                    db_pool,
                    user_id=user_id,
                    upload_id=upload_id,
                    catalog_flat=flat,
                    category=category,
                )
        except Exception as mem_e:
            logger.debug("[recognition] entity catalog upsert skipped: %s", mem_e)

        logger.info(
            "[recognition] persisted upload=%s objects=%d text=%d persons=%d logos=%d "
            "hydration_score=%.2f",
            upload_id,
            summary["object_track_count"],
            summary["text_detection_count"],
            summary["person_segment_count"],
            summary["logo_count"],
            summary["hydration_score"],
        )
    except Exception as e:
        err_str = str(e)
        if "relation" in err_str and "does not exist" in err_str:
            logger.warning(
                "[recognition] DB tables missing for upload=%s — run migrations to create "
                "video_recognition and upload_recognition_summary. Error: %s",
                upload_id,
                err_str[:200],
            )
        else:
            logger.warning("[recognition] persist failed for upload=%s: %s", upload_id, e)

    return summary
